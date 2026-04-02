"""Evaluate GSPO RL checkpoints (verl FSDP format) on configured benchmarks.

GSPO checkpoints produced by verl use FSDP-sharded weights stored under
``<weight_root>/global_step_<N>/actor/``.  Before inference, the shards
are merged into a standard HuggingFace model directory using
``verl.model_merger``.

Usage:
  python gspo/evaluation_gspo.py --config gspo/evaluation_gspo_config.yaml
  python gspo/evaluation_gspo.py --config gspo/evaluation_gspo_config.yaml --checkpoint_step 400
  python gspo/evaluation_gspo.py --config gspo/evaluation_gspo_config.yaml --checkpoint_path /path/to/global_step_400
"""

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DeepSeek_OCR_2 import apply_transformers_compat_shims, resolve_local_model_path
from calc_accuracy import SmilesEvaluator

SMILES_CANDIDATE_COLUMNS = ("SMILES", "smiles", "canonical_smiles")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    name: str
    val_csv: str
    data_mode: str
    realistic_image_root: str | None
    pre_rendered_image_dir: str | None
    instruction: str
    sample_num: int | None
    enabled: bool


@dataclass
class GspoEvalConfig:
    weight_root: str
    checkpoint_step: int | None
    checkpoint_name: str | None
    checkpoint_path: str | None
    output_dir: str
    merged_gspo_model_dir: str
    image_size: int
    base_size: int
    crop_mode: bool
    num_workers: int
    tanimoto: bool
    selected_benchmarks: list[str] | None
    benchmarks: list[BenchmarkConfig]
    backend: str = "transformers"
    # vLLM-specific options
    vllm_code_dir: str | None = None
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 8192
    vllm_tensor_parallel_size: int = 1
    # Multi-GPU data parallelism
    num_gpus: int = 1
    gpu_ids: list[int] | None = None
    # verl FSDP merge backend (fsdp or megatron)
    merge_backend: str = "fsdp"


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _find_smiles_column(df: pd.DataFrame) -> str:
    for column in SMILES_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No SMILES column found. Tried {SMILES_CANDIDATE_COLUMNS}, got {list(df.columns)}")


def load_eval_config(config_path: str) -> GspoEvalConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config root must be a mapping/object.")

    raw["weight_root"] = _resolve_path(raw.get("weight_root", "./weight_gspo_rl_verl"), config_file.parent)
    raw["output_dir"] = _resolve_path(
        raw.get("output_dir", "./gspo_evaluation_outputs"),
        config_file.parent,
    )
    raw["merged_gspo_model_dir"] = _resolve_path(
        raw.get("merged_gspo_model_dir", "./merged_gspo_models"),
        config_file.parent,
    )
    raw["checkpoint_path"] = raw.get("checkpoint_path", None)
    if raw["checkpoint_path"] not in (None, ""):
        raw["checkpoint_path"] = _resolve_path(raw["checkpoint_path"], config_file.parent)
    else:
        raw["checkpoint_path"] = None
    raw["checkpoint_step"] = raw.get("checkpoint_step", None)
    raw["checkpoint_name"] = raw.get("checkpoint_name", None)
    raw["image_size"] = int(raw.get("image_size", 768))
    raw["base_size"] = int(raw.get("base_size", 1024))
    raw["crop_mode"] = bool(raw.get("crop_mode", True))
    raw["num_workers"] = int(raw.get("num_workers", 16))
    raw["tanimoto"] = bool(raw.get("tanimoto", False))

    raw["backend"] = raw.get("backend", "transformers")
    raw["merge_backend"] = raw.get("merge_backend", "fsdp")

    vllm_code_dir = raw.get("vllm_code_dir", None)
    if vllm_code_dir not in (None, ""):
        raw["vllm_code_dir"] = _resolve_path(vllm_code_dir, config_file.parent)
    else:
        raw["vllm_code_dir"] = None
    raw["vllm_gpu_memory_utilization"] = float(raw.get("vllm_gpu_memory_utilization", 0.85))
    raw["vllm_max_model_len"] = int(raw.get("vllm_max_model_len", 8192))
    raw["vllm_tensor_parallel_size"] = int(raw.get("vllm_tensor_parallel_size", 1))
    raw["num_gpus"] = int(raw.get("num_gpus", 1))
    gpu_ids_raw = raw.get("gpu_ids", None)
    if gpu_ids_raw in (None, []):
        raw["gpu_ids"] = None
    else:
        raw["gpu_ids"] = [int(x) for x in gpu_ids_raw]

    selected = raw.get("selected_benchmarks", None)
    if selected in (None, []):
        raw["selected_benchmarks"] = None
    else:
        raw["selected_benchmarks"] = [str(x).strip() for x in selected if str(x).strip()]

    benchmarks_raw = raw.get("benchmarks", []) or []
    if not benchmarks_raw:
        raise ValueError("`benchmarks` cannot be empty in evaluation config.")

    parsed_benchmarks: list[BenchmarkConfig] = []
    for i, item in enumerate(benchmarks_raw):
        item = dict(item)
        item.setdefault("name", f"benchmark_{i}")
        item.setdefault("data_mode", "realistic")
        item.setdefault("instruction", "<image>\n Give me the SMILES of the molecule. ")
        item.setdefault("sample_num", None)
        item.setdefault("enabled", True)
        item.setdefault("realistic_image_root", None)
        item.setdefault("pre_rendered_image_dir", None)
        item["val_csv"] = _resolve_path(item["val_csv"], config_file.parent)
        if item.get("realistic_image_root"):
            item["realistic_image_root"] = _resolve_path(item["realistic_image_root"], config_file.parent)
        if item.get("pre_rendered_image_dir"):
            item["pre_rendered_image_dir"] = _resolve_path(item["pre_rendered_image_dir"], config_file.parent)
        parsed_benchmarks.append(BenchmarkConfig(**item))
    raw["benchmarks"] = parsed_benchmarks

    cfg = GspoEvalConfig(**raw)
    validate_eval_config(cfg)
    return cfg


def validate_eval_config(cfg: GspoEvalConfig) -> None:
    if not os.path.isdir(cfg.weight_root):
        raise ValueError(f"weight_root not found: {cfg.weight_root}")
    if cfg.num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    if cfg.image_size <= 0 or cfg.base_size <= 0:
        raise ValueError("image_size and base_size must be > 0")
    if cfg.num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if cfg.gpu_ids is not None:
        if len(cfg.gpu_ids) == 0:
            raise ValueError("gpu_ids cannot be empty")
        if any(gid < 0 for gid in cfg.gpu_ids):
            raise ValueError(f"gpu_ids must be non-negative, got {cfg.gpu_ids}")
        if len(set(cfg.gpu_ids)) != len(cfg.gpu_ids):
            raise ValueError(f"gpu_ids contains duplicates: {cfg.gpu_ids}")
    if cfg.backend not in ("transformers", "vllm"):
        raise ValueError(f"backend must be 'transformers' or 'vllm', got '{cfg.backend}'")
    if cfg.merge_backend not in ("fsdp", "megatron"):
        raise ValueError(f"merge_backend must be 'fsdp' or 'megatron', got '{cfg.merge_backend}'")
    if cfg.backend == "vllm":
        if not cfg.vllm_code_dir or not os.path.isdir(cfg.vllm_code_dir):
            raise ValueError(f"vllm_code_dir is required and must exist for vllm backend: {cfg.vllm_code_dir}")

    for i, bench in enumerate(cfg.benchmarks):
        if bench.data_mode not in ("realistic", "pre_rendered"):
            raise ValueError(f"benchmarks[{i}].data_mode must be realistic or pre_rendered")
        if not os.path.isfile(bench.val_csv):
            raise ValueError(f"benchmarks[{i}].val_csv not found: {bench.val_csv}")
        if bench.data_mode == "realistic":
            if not bench.realistic_image_root:
                raise ValueError(f"benchmarks[{i}].realistic_image_root is required for realistic mode")
            if not os.path.isdir(bench.realistic_image_root):
                raise ValueError(
                    f"benchmarks[{i}].realistic_image_root not found: {bench.realistic_image_root}"
                )
        if bench.data_mode == "pre_rendered":
            if not bench.pre_rendered_image_dir:
                raise ValueError(f"benchmarks[{i}].pre_rendered_image_dir is required for pre_rendered mode")
            if not os.path.isdir(bench.pre_rendered_image_dir):
                raise ValueError(
                    f"benchmarks[{i}].pre_rendered_image_dir not found: {bench.pre_rendered_image_dir}"
                )


# ---------------------------------------------------------------------------
# GSPO FSDP checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_gspo_checkpoint(weight_root: str) -> str | None:
    """Find the latest global_step_<N> checkpoint under weight_root."""
    path = Path(weight_root)
    if not path.is_dir():
        return None
    checkpoints = []
    for ckpt in path.glob("global_step_*"):
        if not ckpt.is_dir():
            continue
        try:
            step = int(ckpt.name.split("_")[-1])
        except (IndexError, ValueError):
            continue
        actor_dir = ckpt / "actor"
        if actor_dir.is_dir():
            checkpoints.append((step, ckpt))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def resolve_gspo_checkpoint_path(
    cfg: GspoEvalConfig,
    cli_checkpoint_path: str | None,
    cli_step: int | None,
) -> str:
    """Resolve the GSPO checkpoint directory (global_step_<N>)."""
    if cli_checkpoint_path:
        path = Path(cli_checkpoint_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"CLI checkpoint path not found: {path}")
        return str(path)
    if cli_step is not None:
        path = Path(cfg.weight_root) / f"global_step_{cli_step}"
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint for step {cli_step} not found: {path}")
        return str(path.resolve())
    if cfg.checkpoint_path:
        return cfg.checkpoint_path
    if cfg.checkpoint_step is not None:
        path = Path(cfg.weight_root) / f"global_step_{cfg.checkpoint_step}"
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint_step not found: {path}")
        return str(path.resolve())
    if cfg.checkpoint_name:
        path = Path(cfg.weight_root) / cfg.checkpoint_name
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint_name not found: {path}")
        return str(path.resolve())
    latest = _find_latest_gspo_checkpoint(cfg.weight_root)
    if latest is None:
        raise FileNotFoundError(f"No global_step_* found under {cfg.weight_root}")
    return latest


def _patch_automap_for_verl_merger(hf_config_dir: Path) -> dict | None:
    """Temporarily fix auto_map so verl model_merger recognises the class.

    verl's ``get_transformers_auto_model_class`` only handles
    AutoModelForCausalLM / AutoModelForTokenClassification /
    AutoModelForVision2Seq.  DeepSeek-OCR2 registers under ``AutoModel``,
    which causes ``NotImplementedError``.

    We rewrite ``"AutoModel"`` -> ``"AutoModelForCausalLM"`` in
    ``config.json`` before merging and return the original auto_map so
    the caller can restore it afterwards if needed.
    """
    config_json = hf_config_dir / "config.json"
    if not config_json.is_file():
        return None
    with open(config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    auto_map = cfg.get("auto_map")
    if not isinstance(auto_map, dict):
        return None
    if "AutoModel" in auto_map and "AutoModelForCausalLM" not in auto_map:
        original_auto_map = dict(auto_map)
        auto_map["AutoModelForCausalLM"] = auto_map.pop("AutoModel")
        cfg["auto_map"] = auto_map
        with open(config_json, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  [patch] Rewrote auto_map AutoModel -> AutoModelForCausalLM in {config_json}")
        return original_auto_map
    return None


def _restore_automap(hf_config_dir: Path, original_auto_map: dict) -> None:
    config_json = hf_config_dir / "config.json"
    if not config_json.is_file():
        return
    with open(config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["auto_map"] = original_auto_map
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def merge_gspo_fsdp_checkpoint(
    checkpoint_path: str,
    merged_model_dir: str,
    merge_backend: str = "fsdp",
) -> str:
    """Merge verl FSDP-sharded actor weights into a HuggingFace model dir.

    Uses ``python -m verl.model_merger merge`` under the hood.
    Returns the path to the merged HF model directory.
    """
    ckpt_name = Path(checkpoint_path).name
    actor_dir = Path(checkpoint_path) / "actor"
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"actor/ subdirectory not found in {checkpoint_path}")

    target_dir = Path(merged_model_dir) / ckpt_name
    marker_file = target_dir / ".merge_complete"

    if marker_file.is_file():
        print(f"Using cached merged GSPO model: {target_dir}")
        return str(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging FSDP shards from {actor_dir} -> {target_dir}")

    hf_config_dir = actor_dir / "huggingface"
    original_auto_map = _patch_automap_for_verl_merger(hf_config_dir)

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", merge_backend,
        "--trust-remote-code",
        "--local_dir", str(actor_dir),
        "--target_dir", str(target_dir),
    ]
    print(f"  cmd: {' '.join(cmd)}")
    try:
        ret = subprocess.run(cmd, check=False)
    finally:
        if original_auto_map is not None:
            _restore_automap(hf_config_dir, original_auto_map)

    if ret.returncode != 0:
        raise RuntimeError(
            f"verl.model_merger failed (exit={ret.returncode}). "
            f"Make sure verl is installed in the current environment."
        )

    _patch_merged_config_for_vllm(target_dir)

    marker_file.touch()
    print(f"Merged GSPO model saved to: {target_dir}")
    return str(target_dir)


def _patch_merged_config_for_vllm(target_dir: Path) -> None:
    """Adjust config.json so vLLM can load DeepSeek-OCR2 correctly."""
    config_json_path = target_dir / "config.json"
    if not config_json_path.is_file():
        return
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    cfg_data["model_type"] = "deepseek_vl_v2"
    cfg_data.pop("auto_map", None)
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False)

    for py_file in target_dir.glob("*.py"):
        py_file.unlink()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_image_path(row: pd.Series, bench: BenchmarkConfig) -> str:
    if bench.data_mode == "realistic":
        if "file_path" not in row or pd.isna(row["file_path"]):
            raise ValueError("Missing `file_path` column in benchmark CSV for realistic mode")
        rel_path = str(row["file_path"]).strip()
        path = rel_path if os.path.isabs(rel_path) else os.path.join(bench.realistic_image_root or "", rel_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    image_id = row.get("image_id", None)
    if image_id is None:
        image_id = row.get("Unnamed: 0", None)
    if image_id is None:
        raise ValueError("pre_rendered mode requires `image_id` (or `Unnamed: 0`) column")
    if isinstance(image_id, float) and math.isfinite(image_id) and image_id.is_integer():
        image_id = int(image_id)
    path = os.path.join(bench.pre_rendered_image_dir or "", f"{image_id}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def _apply_sample_num(df: pd.DataFrame, sample_num: int | None) -> pd.DataFrame:
    if sample_num is None:
        return df
    if sample_num <= 0:
        raise ValueError("sample_num must be > 0")
    if sample_num >= len(df):
        return df
    return df.sample(n=sample_num, random_state=3407).reset_index(drop=True)


def _save_benchmark_results(
    cfg: GspoEvalConfig,
    checkpoint_path: str,
    bench: BenchmarkConfig,
    image_ids: list,
    gold_smiles: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    benchmark_output_dir = Path(cfg.output_dir) / Path(checkpoint_path).name / bench.name
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = SmilesEvaluator(gold_smiles, num_workers=cfg.num_workers, tanimoto=cfg.tanimoto)
    scores = evaluator.evaluate(predictions)
    scores["accuracy"] = scores["canon_smiles"]
    scores["num_samples"] = len(gold_smiles)

    pred_df = pd.DataFrame(
        {"image_id": image_ids, "gold_smiles": gold_smiles, "pred_smiles": predictions}
    )
    pred_df.to_csv(benchmark_output_dir / "predictions.csv", index=False)
    with open(benchmark_output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    return scores


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(merged_model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek-OCR2 evaluation.")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    apply_transformers_compat_shims()
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        merged_model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = model.eval().cuda()
    return model, tokenizer


def run_benchmark_transformers(
    model,
    tokenizer,
    cfg: GspoEvalConfig,
    checkpoint_path: str,
    bench: BenchmarkConfig,
) -> dict[str, Any]:
    df = pd.read_csv(bench.val_csv)
    df = _apply_sample_num(df, bench.sample_num)
    smiles_col = _find_smiles_column(df)

    infer_output_dir = Path(cfg.output_dir) / Path(checkpoint_path).name / bench.name / "infer_artifacts"
    infer_output_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[str] = []
    image_ids: list = []
    for idx, row in df.iterrows():
        image_id = row.get("image_id", idx)
        image_ids.append(image_id)
        try:
            image_path = _resolve_image_path(row, bench)
            pred = model.infer(
                tokenizer,
                prompt=bench.instruction,
                image_file=image_path,
                output_path=str(infer_output_dir),
                base_size=cfg.base_size,
                image_size=cfg.image_size,
                crop_mode=cfg.crop_mode,
                eval_mode=True,
            )
            predictions.append(pred.strip() if isinstance(pred, str) else "")
        except Exception as exc:
            print(f"[{bench.name}] sample={idx} failed: {exc}")
            predictions.append("")

        if (idx + 1) % 20 == 0 or (idx + 1) == len(df):
            print(f"[{bench.name}] processed {idx + 1}/{len(df)}")

    gold_smiles = df[smiles_col].astype(str).tolist()
    return _save_benchmark_results(cfg, checkpoint_path, bench, image_ids, gold_smiles, predictions)


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def _setup_vllm_env(cfg: GspoEvalConfig, instruction: str, merged_model_path: str) -> None:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config_mod = types.ModuleType("config")
    config_mod.BASE_SIZE = cfg.base_size
    config_mod.IMAGE_SIZE = cfg.image_size
    config_mod.CROP_MODE = cfg.crop_mode
    config_mod.MIN_CROPS = 2
    config_mod.MAX_CROPS = 6
    config_mod.MAX_CONCURRENCY = 100
    config_mod.NUM_WORKERS = 64
    config_mod.PRINT_NUM_VIS_TOKENS = False
    config_mod.SKIP_REPEAT = True
    config_mod.MODEL_PATH = merged_model_path
    config_mod.INPUT_PATH = ""
    config_mod.OUTPUT_PATH = ""
    config_mod.PROMPT = instruction
    config_mod.TOKENIZER = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    sys.modules["config"] = config_mod

    vllm_dir = str(Path(cfg.vllm_code_dir).resolve())
    if vllm_dir not in sys.path:
        sys.path.insert(0, vllm_dir)


def load_vllm_model(merged_model_path: str, cfg: GspoEvalConfig):
    from vllm import LLM
    from vllm.model_executor.models.registry import ModelRegistry
    from deepseek_ocr2 import DeepseekOCR2ForCausalLM as VLLMDeepseekOCR2

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", VLLMDeepseekOCR2)

    llm = LLM(
        model=merged_model_path,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=cfg.vllm_max_model_len,
        swap_space=0,
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
    )
    return llm


def run_benchmark_vllm(
    llm,
    cfg: GspoEvalConfig,
    checkpoint_path: str,
    bench: BenchmarkConfig,
) -> dict[str, Any]:
    from PIL import Image
    from vllm import SamplingParams
    from process.image_process import DeepseekOCR2Processor

    df = pd.read_csv(bench.val_csv)
    df = _apply_sample_num(df, bench.sample_num)
    smiles_col = _find_smiles_column(df)

    processor = DeepseekOCR2Processor()
    batch_inputs: list[dict | None] = []
    image_ids: list = []

    for idx, row in df.iterrows():
        image_id = row.get("image_id", idx)
        image_ids.append(image_id)
        try:
            image_path = _resolve_image_path(row, bench)
            image = Image.open(image_path).convert("RGB")
            tokenized = processor.tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=cfg.crop_mode
            )
            batch_inputs.append({
                "prompt": bench.instruction,
                "multi_modal_data": {"image": tokenized},
            })
        except Exception as exc:
            print(f"[{bench.name}] sample={idx} preprocess failed: {exc}")
            batch_inputs.append(None)

    valid_inputs = [inp for inp in batch_inputs if inp is not None]
    print(f"[{bench.name}] Running vLLM inference on {len(valid_inputs)}/{len(df)} samples...")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        skip_special_tokens=False,
    )
    outputs = llm.generate(valid_inputs, sampling_params)

    stop_str = "\u003c\uff5cend\u2581of\u2581sentence\uff5c\u003e"
    predictions: list[str] = []
    output_idx = 0
    for inp in batch_inputs:
        if inp is not None:
            text = outputs[output_idx].outputs[0].text.strip()
            if text.endswith(stop_str):
                text = text[: -len(stop_str)].strip()
            predictions.append(text)
            output_idx += 1
        else:
            predictions.append("")

    gold_smiles = df[smiles_col].astype(str).tolist()
    return _save_benchmark_results(cfg, checkpoint_path, bench, image_ids, gold_smiles, predictions)


# ---------------------------------------------------------------------------
# Multi-GPU data-parallel workers
# ---------------------------------------------------------------------------

def _vllm_gpu_worker(
    physical_gpu_id: int,
    merged_model_path: str,
    cfg: GspoEvalConfig,
    checkpoint_path: str,
    benchmarks: list[BenchmarkConfig],
    result_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.set_device(0)

    cfg.vllm_tensor_parallel_size = 1

    instruction = benchmarks[0].instruction
    _setup_vllm_env(cfg, instruction, merged_model_path)
    llm = load_vllm_model(merged_model_path, cfg)

    gpu_scores: dict[str, Any] = {}
    for bench in benchmarks:
        print(f"\n[GPU {physical_gpu_id}] === Running benchmark: {bench.name} ===")
        scores = run_benchmark_vllm(llm, cfg, checkpoint_path, bench)
        gpu_scores[bench.name] = scores
        print(f"[GPU {physical_gpu_id}] {json.dumps({bench.name: scores}, indent=2)}")

    result_queue.put(gpu_scores)


def _transformers_gpu_worker(
    physical_gpu_id: int,
    merged_model_path: str,
    cfg: GspoEvalConfig,
    checkpoint_path: str,
    benchmarks: list[BenchmarkConfig],
    result_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.set_device(0)

    model, tokenizer = load_model_and_tokenizer(merged_model_path)

    gpu_scores: dict[str, Any] = {}
    for bench in benchmarks:
        print(f"\n[GPU {physical_gpu_id}] === Running benchmark: {bench.name} ===")
        scores = run_benchmark_transformers(model, tokenizer, cfg, checkpoint_path, bench)
        gpu_scores[bench.name] = scores
        print(f"[GPU {physical_gpu_id}] {json.dumps({bench.name: scores}, indent=2)}")

    result_queue.put(gpu_scores)


def _run_multi_gpu(
    cfg: GspoEvalConfig,
    checkpoint_path: str,
    merged_model_path: str,
    selected_benchmarks: list[BenchmarkConfig],
    config_path: str,
) -> dict[str, Any]:
    if cfg.gpu_ids:
        selected_gpu_ids = cfg.gpu_ids[: len(selected_benchmarks)]
    else:
        num_gpus = min(cfg.num_gpus, len(selected_benchmarks))
        selected_gpu_ids = list(range(num_gpus))
    num_slots = len(selected_gpu_ids)

    gpu_groups: list[list[BenchmarkConfig]] = [[] for _ in range(num_slots)]
    for i, bench in enumerate(selected_benchmarks):
        gpu_groups[i % num_slots].append(bench)

    if cfg.backend == "vllm":
        processes = []
        script_path = str(Path(__file__).resolve())
        work_dir = str(Path(__file__).resolve().parent)

        for slot_idx, physical_gpu_id in enumerate(selected_gpu_ids):
            if not gpu_groups[slot_idx]:
                continue
            bench_names = ",".join([b.name for b in gpu_groups[slot_idx]])
            cmd = [
                sys.executable,
                script_path,
                "--config",
                str(Path(config_path).resolve()),
                "--backend",
                "vllm",
                "--checkpoint_path",
                checkpoint_path,
                "--benchmarks",
                bench_names,
                "--num_gpus",
                "1",
                "--gpu_ids",
                "0",
            ]
            if cfg.output_dir:
                cmd.extend(["--output_dir", cfg.output_dir])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
            env["VLLM_USE_V1"] = "0"
            env["TOKENIZERS_PARALLELISM"] = "false"

            p = subprocess.Popen(cmd, cwd=work_dir, env=env)
            processes.append((physical_gpu_id, bench_names, p))
            print(f"[GPU {physical_gpu_id}] subprocess started for benchmarks: {bench_names}")

        for physical_gpu_id, bench_names, p in processes:
            code = p.wait()
            if code != 0:
                raise RuntimeError(
                    f"[GPU {physical_gpu_id}] subprocess failed (exit={code}) for benchmarks: {bench_names}"
                )

        all_scores: dict[str, Any] = {}
        ckpt_name = Path(checkpoint_path).name
        for bench in selected_benchmarks:
            metrics_path = Path(cfg.output_dir) / ckpt_name / bench.name / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"metrics.json not found after subprocess run: {metrics_path}")
            with open(metrics_path, "r", encoding="utf-8") as f:
                all_scores[bench.name] = json.load(f)
        return all_scores

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes_mp = []

    for slot_idx, physical_gpu_id in enumerate(selected_gpu_ids):
        if not gpu_groups[slot_idx]:
            continue
        if cfg.backend == "vllm":
            target = _vllm_gpu_worker
            args = (physical_gpu_id, merged_model_path, cfg, checkpoint_path,
                    gpu_groups[slot_idx], result_queue)
        else:
            target = _transformers_gpu_worker
            args = (physical_gpu_id, merged_model_path, cfg, checkpoint_path,
                    gpu_groups[slot_idx], result_queue)

        p = ctx.Process(target=target, args=args)
        p.start()
        processes_mp.append(p)
        print(f"[GPU {physical_gpu_id}] started with benchmarks: "
              f"{[b.name for b in gpu_groups[slot_idx]]}")

    all_scores: dict[str, Any] = {}
    for _ in processes_mp:
        gpu_scores = result_queue.get(timeout=7200)
        all_scores.update(gpu_scores)

    for p in processes_mp:
        p.join()

    return all_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate GSPO RL checkpoints (verl FSDP format) on configured benchmarks."
    )
    parser.add_argument(
        "--config", type=str,
        default=str(workspace / "evaluation_gspo_config.yaml"),
        help="Path to GSPO evaluation YAML config.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Optional explicit checkpoint directory path (e.g. /path/to/global_step_400).",
    )
    parser.add_argument(
        "--checkpoint_step", type=int, default=None,
        help="Optional checkpoint step, resolved as <weight_root>/global_step_<step>.",
    )
    parser.add_argument(
        "--benchmarks", type=str, default=None,
        help="Optional comma-separated benchmark names to run.",
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["transformers", "vllm"],
        help="Override backend from config.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs for data-parallel inference (overrides config).",
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated physical GPU ids for data parallelism, e.g. 1,3.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory from config.",
    )
    return parser.parse_args()


def select_benchmarks(cfg: GspoEvalConfig, cli_benchmarks: str | None) -> list[BenchmarkConfig]:
    if cli_benchmarks:
        selected = {x.strip() for x in cli_benchmarks.split(",") if x.strip()}
    elif cfg.selected_benchmarks:
        selected = set(cfg.selected_benchmarks)
    else:
        selected = set()

    output: list[BenchmarkConfig] = []
    for bench in cfg.benchmarks:
        if not bench.enabled:
            continue
        if selected and bench.name not in selected:
            continue
        output.append(bench)
    if not output:
        raise ValueError("No benchmark selected to run. Check enabled flags and selection options.")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    if args.backend:
        cfg.backend = args.backend
    if args.output_dir:
        cfg.output_dir = str(Path(args.output_dir).resolve())
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
    if args.gpu_ids:
        cfg.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    validate_eval_config(cfg)

    checkpoint_path = resolve_gspo_checkpoint_path(cfg, args.checkpoint_path, args.checkpoint_step)
    selected_benchmarks = select_benchmarks(cfg, args.benchmarks)

    print(f"Backend: {cfg.backend}")
    print(f"GSPO checkpoint: {checkpoint_path}")

    # Step 1: merge FSDP shards into HF model
    merged_model_path = merge_gspo_fsdp_checkpoint(
        checkpoint_path, cfg.merged_gspo_model_dir, cfg.merge_backend
    )
    print(f"Merged model path: {merged_model_path}")

    if cfg.gpu_ids:
        print(f"GPU ids: {cfg.gpu_ids}")
    else:
        print(f"GPUs: {cfg.num_gpus}")

    all_scores: dict[str, Any] = {}

    use_multi_gpu = (len(cfg.gpu_ids) > 1) if cfg.gpu_ids else (cfg.num_gpus > 1)
    if use_multi_gpu:
        all_scores = _run_multi_gpu(cfg, checkpoint_path, merged_model_path, selected_benchmarks, args.config)

    elif cfg.backend == "vllm":
        instruction = selected_benchmarks[0].instruction
        _setup_vllm_env(cfg, instruction, merged_model_path)
        llm = load_vllm_model(merged_model_path, cfg)

        for bench in selected_benchmarks:
            print(f"\n=== Running benchmark: {bench.name} ===")
            scores = run_benchmark_vllm(llm, cfg, checkpoint_path, bench)
            all_scores[bench.name] = scores
            print(json.dumps({bench.name: scores}, indent=2))

    else:
        model, tokenizer = load_model_and_tokenizer(merged_model_path)

        for bench in selected_benchmarks:
            print(f"\n=== Running benchmark: {bench.name} ===")
            scores = run_benchmark_transformers(model, tokenizer, cfg, checkpoint_path, bench)
            all_scores[bench.name] = scores
            print(json.dumps({bench.name: scores}, indent=2))

    summary_path = Path(cfg.output_dir) / Path(checkpoint_path).name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
