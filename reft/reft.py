"""Rejection Sampling Fine-Tuning (ReFT) for DeepSeek-OCR-2 SMILES OCR.

Pipeline per iteration:
  1. Generate N completions per training sample with vLLM
  2. Score each completion with SMILES reward function
  3. Keep best-of-N per sample (above min_reward threshold)
  4. Fine-tune on curated best completions via standard SFT

Usage:
  python reft/reft.py --config reft/reft_config.yaml
  python reft/reft.py --config reft/reft_config.yaml --phase generate
  python reft/reft.py --config reft/reft_config.yaml --phase train --iteration 0
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMILES_CANDIDATE_COLUMNS = ("SMILES", "smiles", "canonical_smiles")
DEFAULT_INSTRUCTION = "<image>\n Give me the SMILES of the molecule. "
_RENDERER_CACHE: Dict[Tuple[str, bool, bool], Any] = {}


# =========================================================================
# Reward scoring (reused from gspo_rl_verl.py)
# =========================================================================

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


def _canonicalize_smiles(
    smiles: str, ignore_chiral: bool = False, ignore_cistrans: bool = False
) -> Tuple[Optional[str], bool]:
    if not isinstance(smiles, str) or smiles == "":
        return None, False
    if ignore_cistrans:
        smiles = smiles.replace("/", "").replace("\\", "")
    try:
        from rdkit import Chem
        canon = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        return canon, True
    except Exception:
        return None, False


def _tanimoto_similarity(smi1: str, smi2: str) -> float:
    try:
        from rdkit import Chem, DataStructs
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        return float(DataStructs.FingerprintSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _repetition_penalty(smiles: str) -> float:
    if not smiles or len(smiles) < 20:
        return 0.0
    s = smiles
    length = len(s)
    max_run = 1
    cur_run = 1
    for i in range(1, length):
        if s[i] == s[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    char_repeat_ratio = max_run / length

    for pat_len in range(1, 5):
        if length < pat_len * 12:
            continue
        pat = s[:pat_len]
        repeats = 0
        for i in range(0, length - pat_len + 1, pat_len):
            if s[i : i + pat_len] == pat:
                repeats += 1
            else:
                break
        if repeats >= 12:
            return min(1.0, repeats * pat_len / length)

    if char_repeat_ratio > 0.3:
        return min(1.0, char_repeat_ratio)

    unique_chars = len(set(s))
    if length > 30 and unique_chars <= 3:
        return 0.5

    return 0.0


def _replace_empty(s: Optional[str]) -> str:
    return s if isinstance(s, str) and s != "" else "<empty>"


def compute_reward(
    gold_smiles: str,
    pred_smiles: str,
    reward_weights: Dict[str, float],
    chiral_no_annotation_reward: float = 0.0,
) -> float:
    canon_gold, valid_gold = _canonicalize_smiles(gold_smiles, ignore_cistrans=True)
    canon_pred, valid_pred = _canonicalize_smiles(pred_smiles, ignore_cistrans=True)
    graph_gold, valid_graph_gold = _canonicalize_smiles(
        gold_smiles, ignore_chiral=True, ignore_cistrans=True
    )
    graph_pred, valid_graph_pred = _canonicalize_smiles(
        pred_smiles, ignore_chiral=True, ignore_cistrans=True
    )
    canon_gold = _replace_empty(canon_gold)
    canon_pred = _replace_empty(canon_pred)
    graph_gold = _replace_empty(graph_gold)
    graph_pred = _replace_empty(graph_pred)

    has_chiral = "@" in (canon_gold or "")
    canon_match = float(valid_gold and valid_pred and (canon_gold == canon_pred))
    graph_sim = (
        _tanimoto_similarity(graph_gold, graph_pred)
        if (valid_graph_gold and valid_graph_pred)
        else 0.0
    )
    chiral_acc = (
        float(canon_match > 0.5) if has_chiral else chiral_no_annotation_reward
    )
    rep_penalty = _repetition_penalty(pred_smiles)

    w = reward_weights
    w_validity = float(w.get("validity", 0.0))
    w_tanimoto = float(w.get("tanimoto", 1.0))
    w_canon = float(w.get("canon_smiles", 8.0))
    w_graph = float(w.get("graph", 2.0))
    w_chiral = float(w.get("chiral", 1.0))
    w_rep = float(w.get("repetition_penalty", 2.0))

    weighted_sum = (
        w_validity * float(valid_pred)
        + w_tanimoto * _tanimoto_similarity(gold_smiles, pred_smiles)
        + w_canon * canon_match
        + w_graph * graph_sim
        + w_chiral * chiral_acc
        - w_rep * rep_penalty
    )
    w_total = w_validity + w_tanimoto + w_canon + w_graph + w_chiral
    if w_total <= 0:
        return 0.0
    return float(weighted_sum / w_total)


# =========================================================================
# Config loading
# =========================================================================

def _resolve_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _find_smiles_column(df: pd.DataFrame) -> str:
    for col in SMILES_CANDIDATE_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"No SMILES column found. Tried {SMILES_CANDIDATE_COLUMNS}, got {list(df.columns)}")


def _get_renderer(style: str, mol_augment: bool, include_condensed: bool):
    key = (style, bool(mol_augment), bool(include_condensed))
    renderer = _RENDERER_CACHE.get(key)
    if renderer is None:
        from dataset import MoleculeStyleConfig, MoleculeStyleRenderer
        style_cfg = MoleculeStyleConfig(
            render_style=style,
            mol_augment=bool(mol_augment),
            include_condensed=bool(include_condensed),
        )
        renderer = MoleculeStyleRenderer(style_cfg)
        _RENDERER_CACHE[key] = renderer
    return renderer


def _render_dynamic_sample(task: Tuple[str, str, str, bool, bool]) -> Tuple[bool, str, str, str]:
    smiles, image_path, style, mol_augment, include_condensed = task
    if os.path.isfile(image_path):
        return True, image_path, smiles, ""
    try:
        renderer = _get_renderer(style, mol_augment, include_condensed)
        image, _ = renderer.render(smiles)
        image.save(image_path)
        return True, image_path, smiles, ""
    except Exception as exc:
        return False, image_path, smiles, str(exc)


def load_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = config_file.parent

    cfg["pretrained_weight_path"] = _resolve_path(cfg["pretrained_weight_path"], base)
    cfg["output_dir"] = _resolve_path(cfg.get("output_dir", "./weight_reft"), base)
    cfg["merged_model_dir"] = _resolve_path(cfg.get("merged_model_dir", "./merged_reft_models"), base)

    vllm_cfg = cfg.get("vllm", {})
    if vllm_cfg.get("code_dir"):
        vllm_cfg["code_dir"] = _resolve_path(vllm_cfg["code_dir"], base)
    cfg["vllm"] = vllm_cfg

    for ts in cfg.get("train_sets", []):
        ts["train_csv"] = _resolve_path(ts["train_csv"], base)
        if ts.get("pre_rendered_image_dir"):
            ts["pre_rendered_image_dir"] = _resolve_path(ts["pre_rendered_image_dir"], base)
        if ts.get("realistic_image_root"):
            ts["realistic_image_root"] = _resolve_path(ts["realistic_image_root"], base)

    render_cfg = dict(cfg.get("render", {}) or {})
    render_cfg["num_workers"] = int(render_cfg.get("num_workers", 1))
    render_cfg["chunksize"] = int(render_cfg.get("chunksize", 64))
    cfg["render"] = render_cfg

    return cfg


# =========================================================================
# Phase 1: Image preparation
# =========================================================================

def _prepare_samples(cfg: dict, iteration_dir: Path) -> pd.DataFrame:
    """Load training CSVs, render dynamic images, resolve paths.

    Returns a DataFrame with columns: file_path, SMILES (gold), data_source.
    """
    seed = cfg.get("seed", 3407)
    render_dir = iteration_dir / "rendered_images"
    render_cfg = cfg.get("render", {})
    render_workers = max(1, int(render_cfg.get("num_workers", 1)))
    render_chunksize = max(1, int(render_cfg.get("chunksize", 64)))
    all_rows: List[Dict[str, str]] = []
    global_idx = 0

    for ts_idx, ts in enumerate(cfg.get("train_sets", [])):
        train_csv = ts["train_csv"]
        data_mode = ts.get("data_mode", "realistic")
        sample_num = ts.get("sample_num")

        df = pd.read_csv(train_csv)
        if sample_num and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)

        smiles_col = _find_smiles_column(df)

        if data_mode == "dynamic":
            style = ts.get("style", "molscribe_default")
            mol_augment = ts.get("mol_augment", True)
            include_condensed = ts.get("include_condensed", False)
            ts_render_dir = render_dir / f"set{ts_idx}_{style}"
            ts_render_dir.mkdir(parents=True, exist_ok=True)
            rendered = 0
            skipped = 0
            tasks: List[Tuple[str, str, str, bool, bool]] = []
            for row_idx, row in df.iterrows():
                smi = str(row[smiles_col]).strip()
                if not smi:
                    skipped += 1
                    continue
                img_path = ts_render_dir / f"{global_idx + row_idx}.png"
                tasks.append((smi, str(img_path), style, bool(mol_augment), bool(include_condensed)))

            print(
                f"[prepare] set{ts_idx} dynamic/{style}: "
                f"render_tasks={len(tasks)}, workers={render_workers}, chunksize={render_chunksize}"
            )

            if render_workers == 1:
                results_iter = map(_render_dynamic_sample, tasks)
                for i, (ok, image_path, smi, _err) in enumerate(results_iter, start=1):
                    if ok and os.path.isfile(image_path):
                        all_rows.append(
                            {
                                "file_path": image_path,
                                "SMILES": smi,
                                "data_source": f"set{ts_idx}_{style}",
                            }
                        )
                        rendered += 1
                    else:
                        skipped += 1
                    if i % 1000 == 0 or i == len(tasks):
                        print(f"[prepare] set{ts_idx} dynamic/{style}: {i}/{len(tasks)}")
            else:
                with ProcessPoolExecutor(max_workers=render_workers) as pool:
                    results_iter = pool.map(_render_dynamic_sample, tasks, chunksize=render_chunksize)
                    for i, (ok, image_path, smi, _err) in enumerate(results_iter, start=1):
                        if ok and os.path.isfile(image_path):
                            all_rows.append(
                                {
                                    "file_path": image_path,
                                    "SMILES": smi,
                                    "data_source": f"set{ts_idx}_{style}",
                                }
                            )
                            rendered += 1
                        else:
                            skipped += 1
                        if i % 1000 == 0 or i == len(tasks):
                            print(f"[prepare] set{ts_idx} dynamic/{style}: {i}/{len(tasks)}")

            global_idx += len(df)
            print(f"[prepare] set{ts_idx} dynamic/{style}: rendered={rendered}, skipped={skipped}")

        elif data_mode == "realistic":
            image_root = ts.get("realistic_image_root", "")
            if "file_path" not in df.columns:
                print(f"[prepare] set{ts_idx}: WARNING missing file_path column, skipping")
                continue
            added = 0
            for _, row in df.iterrows():
                rel_path = str(row["file_path"]).strip()
                abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(image_root, rel_path)
                if not os.path.isfile(abs_path):
                    continue
                all_rows.append({
                    "file_path": abs_path,
                    "SMILES": str(row[smiles_col]).strip(),
                    "data_source": f"set{ts_idx}_realistic",
                })
                added += 1
                global_idx += 1
            print(f"[prepare] set{ts_idx} realistic: added={added}")

        elif data_mode == "pre_rendered":
            image_dir = ts.get("pre_rendered_image_dir", "")
            id_col = None
            for c in ("image_id", "Unnamed: 0", "id", "idx"):
                if c in df.columns:
                    id_col = c
                    break
            added = 0
            for _, row in df.iterrows():
                img_id = row[id_col] if id_col else row.name
                if isinstance(img_id, float) and math.isfinite(img_id):
                    img_id = int(img_id)
                img_path = os.path.join(image_dir, f"{img_id}.png")
                if not os.path.isfile(img_path):
                    continue
                all_rows.append({
                    "file_path": img_path,
                    "SMILES": str(row[smiles_col]).strip(),
                    "data_source": f"set{ts_idx}_pre_rendered",
                })
                added += 1
                global_idx += 1
            print(f"[prepare] set{ts_idx} pre_rendered: added={added}")

    samples_df = pd.DataFrame(all_rows)
    print(f"[prepare] Total samples: {len(samples_df)}")
    return samples_df


# =========================================================================
# Phase 1: vLLM generation + scoring + filtering
# =========================================================================

def _setup_vllm_env(cfg: dict, model_path: str) -> None:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    instruction = cfg.get("instruction", DEFAULT_INSTRUCTION)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config_mod = types.ModuleType("config")
    config_mod.BASE_SIZE = cfg.get("base_size", 1024)
    config_mod.IMAGE_SIZE = cfg.get("image_size", 768)
    config_mod.CROP_MODE = cfg.get("crop_mode", True)
    config_mod.MIN_CROPS = 2
    config_mod.MAX_CROPS = 6
    config_mod.MAX_CONCURRENCY = 100
    config_mod.NUM_WORKERS = 64
    config_mod.PRINT_NUM_VIS_TOKENS = False
    config_mod.SKIP_REPEAT = True
    config_mod.MODEL_PATH = model_path
    config_mod.INPUT_PATH = ""
    config_mod.OUTPUT_PATH = ""
    config_mod.PROMPT = instruction
    config_mod.TOKENIZER = tokenizer
    config_mod._chemseek_injected = True
    sys.modules["config"] = config_mod

    vllm_dir = str(Path(cfg["vllm"]["code_dir"]).resolve())
    if vllm_dir not in sys.path:
        sys.path.insert(0, vllm_dir)


def _load_vllm_model(model_path: str, cfg: dict):
    from vllm import LLM
    from vllm.model_executor.models.registry import ModelRegistry
    from deepseek_ocr2 import DeepseekOCR2ForCausalLM as VLLMDeepseekOCR2

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", VLLMDeepseekOCR2)

    vllm_cfg = cfg.get("vllm", {})
    llm = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=vllm_cfg.get("max_model_len", 8192),
        swap_space=0,
        tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
    )
    return llm


def _generate_single_gpu(
    cfg: dict,
    samples_df: pd.DataFrame,
    model_path: str,
    output_csv: Path,
    gpu_tag: str = "",
) -> pd.DataFrame:
    """Run generation + scoring + filtering on a single GPU."""
    from PIL import Image
    from vllm import SamplingParams

    _setup_vllm_env(cfg, model_path)
    from process.image_process import DeepseekOCR2Processor

    llm = _load_vllm_model(model_path, cfg)
    processor = DeepseekOCR2Processor()

    gen_cfg = cfg.get("generation", {})
    n_completions = gen_cfg.get("n", 8)
    temperature = gen_cfg.get("temperature", 0.3)
    top_p = gen_cfg.get("top_p", 0.95)
    max_tokens = gen_cfg.get("max_tokens", 512)
    chunk_size = gen_cfg.get("chunk_size", 500)

    reward_cfg = cfg.get("reward", {})
    reward_weights = reward_cfg.get("weights", {})
    min_reward = float(reward_cfg.get("min_reward", 0.5))
    chiral_no_ann = float(reward_cfg.get("chiral_no_annotation_reward", 0.0))

    instruction = cfg.get("instruction", DEFAULT_INSTRUCTION)
    stop_str = "\u003c\uff5cend\u2581of\u2581sentence\uff5c\u003e"

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_completions,
        skip_special_tokens=False,
    )

    curated_rows: List[Dict[str, Any]] = []
    total_samples = len(samples_df)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    stats = {"total": 0, "kept": 0, "skipped_preprocess": 0, "below_threshold": 0}

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_samples)
        chunk_df = samples_df.iloc[start:end]

        batch_inputs = []
        batch_meta = []

        for _, row in chunk_df.iterrows():
            file_path = row["file_path"]
            gold_smi = row["SMILES"]
            try:
                image = Image.open(file_path).convert("RGB")
                tokenized = processor.tokenize_with_images(
                    images=[image], bos=True, eos=True, cropping=True
                )
                batch_inputs.append({
                    "prompt": instruction,
                    "multi_modal_data": {"image": tokenized},
                })
                batch_meta.append({
                    "file_path": file_path,
                    "gold_smiles": gold_smi,
                    "data_source": row.get("data_source", ""),
                })
            except Exception as exc:
                stats["skipped_preprocess"] += 1
                if stats["skipped_preprocess"] <= 5:
                    print(f"[{gpu_tag}] Preprocess failed: {exc}")

        if not batch_inputs:
            continue

        outputs = llm.generate(batch_inputs, sampling_params)

        for out_idx, output in enumerate(outputs):
            meta = batch_meta[out_idx]
            gold_smi = meta["gold_smiles"]
            stats["total"] += 1
            best_pred = ""
            best_reward = -1.0

            for completion in output.outputs:
                text = completion.text.strip()
                if text.endswith(stop_str):
                    text = text[: -len(stop_str)].strip()
                pred_smi = text.split("\n")[0].strip().replace(" ", "") if text else ""
                reward = compute_reward(gold_smi, pred_smi, reward_weights, chiral_no_ann)
                if reward > best_reward:
                    best_reward = reward
                    best_pred = pred_smi

            if best_reward >= min_reward and best_pred:
                curated_rows.append({
                    "file_path": meta["file_path"],
                    "SMILES": best_pred,
                    "gold_smiles": gold_smi,
                    "reward": round(best_reward, 6),
                    "data_source": meta["data_source"],
                })
                stats["kept"] += 1
            else:
                stats["below_threshold"] += 1

        print(
            f"[{gpu_tag}] Chunk {chunk_idx + 1}/{num_chunks} done. "
            f"Kept so far: {stats['kept']}/{stats['total']}"
        )

    curated_df = pd.DataFrame(curated_rows)
    curated_df.to_csv(output_csv, index=False)
    print(
        f"[{gpu_tag}] Done. kept={stats['kept']}/{stats['total']} "
        f"skip_preprocess={stats['skipped_preprocess']} below_threshold={stats['below_threshold']}"
    )
    return curated_df


def generate_and_filter(
    cfg: dict,
    samples_df: pd.DataFrame,
    model_path: str,
    iteration_dir: Path,
) -> pd.DataFrame:
    """Generate N completions per sample, score, and keep best above threshold.

    Supports multi-GPU data parallelism via subprocesses when vllm.gpu_ids
    contains more than one GPU id.
    """
    curated_csv = iteration_dir / "curated.csv"
    if curated_csv.is_file():
        print(f"[generate] Curated data already exists: {curated_csv}")
        return pd.read_csv(curated_csv)

    vllm_cfg = cfg.get("vllm", {})
    gpu_ids = vllm_cfg.get("gpu_ids", None)
    if gpu_ids is None:
        gpu_ids = [int(vllm_cfg.get("gpu_id", 0))]
    if not isinstance(gpu_ids, list) or len(gpu_ids) == 0:
        gpu_ids = [0]

    num_gpus = len(gpu_ids)

    if num_gpus <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        shard_csv = iteration_dir / "shard_0.csv"
        _generate_single_gpu(cfg, samples_df, model_path, shard_csv, gpu_tag=f"GPU{gpu_ids[0]}")
        curated_df = pd.read_csv(shard_csv)
    else:
        shards_dir = iteration_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        shard_size = (len(samples_df) + num_gpus - 1) // num_gpus
        shard_input_paths: List[Path] = []
        shard_output_paths: List[Path] = []
        for i in range(num_gpus):
            start = i * shard_size
            end = min(start + shard_size, len(samples_df))
            if start >= len(samples_df):
                break
            shard_df = samples_df.iloc[start:end]
            shard_in = shards_dir / f"input_{i}.csv"
            shard_df.to_csv(shard_in, index=False)
            shard_input_paths.append(shard_in)
            shard_output_paths.append(shards_dir / f"output_{i}.csv")

        print(
            f"[generate] Launching {len(shard_input_paths)} GPU workers: "
            f"gpu_ids={gpu_ids[:len(shard_input_paths)]}, "
            f"samples_per_gpu≈{shard_size}"
        )

        script_path = str(Path(__file__).resolve())
        config_path = str(Path(cfg["_config_path"]).resolve()) if "_config_path" in cfg else ""
        processes = []
        for i, (shard_in, shard_out) in enumerate(zip(shard_input_paths, shard_output_paths)):
            physical_gpu = gpu_ids[i]
            cmd = [
                sys.executable, script_path,
                "--_generate_shard",
                "--_shard_input", str(shard_in),
                "--_shard_output", str(shard_out),
                "--_model_path", model_path,
                "--config", config_path or str(SCRIPT_DIR / "reft_config.yaml"),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            env["VLLM_USE_V1"] = "0"
            env["TOKENIZERS_PARALLELISM"] = "false"
            p = subprocess.Popen(cmd, env=env, cwd=str(SCRIPT_DIR))
            processes.append((physical_gpu, p))
            print(f"[generate] GPU {physical_gpu}: PID {p.pid}, shard {i} ({len(pd.read_csv(shard_in))} samples)")

        for physical_gpu, p in processes:
            code = p.wait()
            if code != 0:
                raise RuntimeError(f"[generate] GPU {physical_gpu} subprocess failed (exit={code})")

        shard_dfs = []
        for shard_out in shard_output_paths:
            if shard_out.is_file():
                sdf = pd.read_csv(shard_out)
                if len(sdf) > 0:
                    shard_dfs.append(sdf)
        curated_df = pd.concat(shard_dfs, ignore_index=True) if shard_dfs else pd.DataFrame()

    curated_df.to_csv(curated_csv, index=False)

    total = len(samples_df)
    kept = len(curated_df)
    min_reward = float(cfg.get("reward", {}).get("min_reward", 0.5))
    print("=" * 60)
    print(f"[generate] Generation complete.")
    print(f"  Total samples:           {total}")
    print(f"  Kept (reward >= {min_reward}): {kept}")
    if kept > 0:
        print(f"  Average reward (kept):   {curated_df['reward'].mean():.4f}")
    print(f"  Curated CSV saved to:    {curated_csv}")
    print("=" * 60)

    return curated_df


# =========================================================================
# Phase 2: SFT on curated data
# =========================================================================

def _write_sft_config(
    cfg: dict,
    curated_csv: Path,
    iteration_output_dir: Path,
    iteration: int,
) -> Path:
    """Write a temporary progressive_sft_config.yaml for this iteration."""
    sft = cfg.get("sft", {})
    sft_config = {
        "pretrained_weight_path": cfg["_current_model_path"],
        "train_sets": [
            {
                "train_csv": str(curated_csv),
                "data_mode": "realistic",
                "realistic_image_root": "/",
                "instruction": cfg.get("instruction", DEFAULT_INSTRUCTION),
                "sample_num": None,
            }
        ],
        "enable_val_sets": False,
        "eval_every_steps": None,
        "eval_on_start": False,
        "eval_on_end": False,
        "val_sets": [],
        "seed": cfg.get("seed", 3407),
        "batch_size": sft.get("batch_size", 4),
        "grad_accum": sft.get("grad_accum", 8),
        "learning_rate": sft.get("learning_rate", 5e-6),
        "vision_learning_rate": sft.get("vision_learning_rate", None),
        "language_learning_rate": sft.get("language_learning_rate", None),
        "weight_decay": sft.get("weight_decay", 0.01),
        "warmup_steps": sft.get("warmup_steps", 100),
        "max_steps": sft.get("max_steps", 0),
        "epochs": sft.get("epochs", 0.0),
        "save_steps": sft.get("save_steps", 100),
        "optim": sft.get("optim", "adamw_torch_fused"),
        "allow_tf32": True,
        "enable_gradient_checkpointing": sft.get("enable_gradient_checkpointing", True),
        "dataloader_num_workers": sft.get("dataloader_num_workers", 8),
        "dataloader_prefetch_factor": 2,
        "dataloader_persistent_workers": False,
        "image_size": cfg.get("image_size", 768),
        "base_size": cfg.get("base_size", 1024),
        "crop_mode": cfg.get("crop_mode", True),
        "load_in_4bit": False,
        "attn_implementation": "flash_attention_2",
        "output_dir": str(iteration_output_dir / "checkpoints"),
        "ddp_find_unused_parameters": False,
        "ddp_backend": "nccl",
        "ddp_timeout_seconds": 1800,
        "use_accelerate": sft.get("use_accelerate", True),
        "accelerate_num_processes": sft.get("accelerate_num_processes", 8),
        "accelerate_gpu_ids": sft.get("accelerate_gpu_ids", None),
        "train_on_responses_only": sft.get("train_on_responses_only", True),
        "freeze_layers": cfg.get("freeze_layers", []),
        "freeze_modules": cfg.get("freeze_modules", []),
        "log_train_accuracy": True,
        "train_accuracy_log_steps": 20,
        "restart": {
            "enable": False,
            "dir": str(iteration_output_dir / "checkpoints"),
            "auto_resume_from_latest_checkpoint": False,
            "resume_from_checkpoint": None,
            "wandb_resume_same_run": False,
        },
        "wandb": cfg.get("wandb", False),
        "wandb_project": cfg.get("wandb_project", "ChemSeek-OCR"),
        "wandb_run_name": f"{cfg.get('wandb_run_name', 'reft')}-iter{iteration}",
        "wandb_api_key": cfg.get("wandb_api_key", None),
    }

    config_path = iteration_output_dir / f"sft_config_iter{iteration}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sft_config, f, default_flow_style=False, allow_unicode=True)
    print(f"[train] Wrote SFT config: {config_path}")
    return config_path


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    path = Path(ckpt_dir)
    if not path.is_dir():
        return None
    checkpoints = []
    for ckpt in path.glob("checkpoint-*"):
        if not ckpt.is_dir():
            continue
        try:
            step = int(ckpt.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, ckpt))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def run_sft(cfg: dict, curated_csv: Path, iteration_dir: Path, iteration: int) -> str:
    """Run SFT on curated data via progressive_sft.py subprocess."""

    sft_config_path = _write_sft_config(cfg, curated_csv, iteration_dir, iteration)
    sft_script = PROJECT_ROOT / "progressive_sft.py"
    if not sft_script.is_file():
        sft_script = PROJECT_ROOT / "full_sft.py"

    cmd = [sys.executable, str(sft_script), "--config", str(sft_config_path)]

    env = os.environ.copy()
    sft_cfg = cfg.get("sft", {})
    gpu_ids = sft_cfg.get("accelerate_gpu_ids")
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    wandb_key = cfg.get("wandb_api_key") or os.environ.get("WANDB_API_KEY")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key

    print("=" * 60)
    print(f"[train] Starting SFT iteration {iteration}")
    print(f"  Script:     {sft_script}")
    print(f"  Config:     {sft_config_path}")
    print(f"  Curated:    {curated_csv}")
    print(f"  Max steps:  {sft_cfg.get('max_steps', 500)}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"SFT training failed with exit code {result.returncode}")

    ckpt_dir = str(iteration_dir / "checkpoints")
    latest = _find_latest_checkpoint(ckpt_dir)
    if latest is None:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir} after training")

    print(f"[train] SFT iteration {iteration} complete. Latest checkpoint: {latest}")
    return latest


# =========================================================================
# Model preparation for vLLM
# =========================================================================

def prepare_model_for_vllm(cfg: dict, model_path: str) -> str:
    """Prepare a model checkpoint for vLLM inference.

    For full fine-tuned models, uses merge_and_save_model to create a
    vLLM-compatible copy.
    """
    from DeepSeek_OCR_2 import apply_transformers_compat_shims
    apply_transformers_compat_shims()

    from merge_lora_weight import merge_and_save_model

    merged_model_dir = cfg.get("merged_model_dir", str(SCRIPT_DIR / "merged_reft_models"))
    pretrained = cfg["pretrained_weight_path"]

    merged_path = merge_and_save_model(
        pretrained_weight_path=pretrained,
        checkpoint_path=model_path,
        merged_model_dir=merged_model_dir,
        full_or_lora="full",
    )
    print(f"[model] Prepared vLLM model: {merged_path}")
    return merged_path


# =========================================================================
# Main orchestration
# =========================================================================

def run_iteration(cfg: dict, iteration: int) -> str:
    """Run one ReFT iteration: generate -> filter -> train.

    Returns the path to the new checkpoint.
    """
    iteration_dir = Path(cfg["output_dir"]) / f"iteration_{iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    current_model = cfg["_current_model_path"]
    print(f"\n{'=' * 60}")
    print(f"ReFT Iteration {iteration}")
    print(f"  Model: {current_model}")
    print(f"  Output: {iteration_dir}")
    print(f"{'=' * 60}\n")

    # Phase 1a: Prepare images
    samples_csv = iteration_dir / "samples.csv"
    if samples_csv.is_file():
        print(f"[prepare] Reusing existing samples: {samples_csv}")
        samples_df = pd.read_csv(samples_csv)
    else:
        samples_df = _prepare_samples(cfg, iteration_dir)
        samples_df.to_csv(samples_csv, index=False)

    if len(samples_df) == 0:
        raise RuntimeError("No training samples after preparation")

    # Phase 1b: Generate + score + filter
    vllm_model_path = prepare_model_for_vllm(cfg, current_model)
    curated_df = generate_and_filter(cfg, samples_df, vllm_model_path, iteration_dir)

    if len(curated_df) == 0:
        print("[WARNING] No samples passed the reward threshold. Skipping SFT.")
        return current_model

    curated_csv = iteration_dir / "curated.csv"

    # Phase 2: SFT
    new_checkpoint = run_sft(cfg, curated_csv, iteration_dir, iteration)
    return new_checkpoint


def _run_generate_shard(args) -> None:
    """Subprocess entry-point: generate on a single shard using one GPU."""
    cfg = load_config(args.config)
    samples_df = pd.read_csv(args._shard_input)
    _generate_single_gpu(
        cfg, samples_df, args._model_path,
        Path(args._shard_output),
        gpu_tag=f"GPU{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ReFT for DeepSeek-OCR-2")
    parser.add_argument(
        "--config", type=str,
        default=str(SCRIPT_DIR / "reft_config.yaml"),
        help="Path to ReFT YAML config",
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        choices=["generate", "train"],
        help="Run only one phase (default: both)",
    )
    parser.add_argument(
        "--iteration", type=int, default=None,
        help="Run only a specific iteration (0-indexed)",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Override the model checkpoint path",
    )
    parser.add_argument("--_generate_shard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_shard_input", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_shard_output", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_model_path", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._generate_shard:
        _run_generate_shard(args)
        return

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    num_iterations = cfg.get("num_iterations", 3)

    current_model = args.model_path or cfg["pretrained_weight_path"]
    cfg["_current_model_path"] = current_model

    if args.iteration is not None:
        iterations = [args.iteration]
    else:
        iterations = list(range(num_iterations))

    for iteration in iterations:
        iteration_dir = Path(cfg["output_dir"]) / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        if args.phase == "generate":
            samples_csv = iteration_dir / "samples.csv"
            if samples_csv.is_file():
                samples_df = pd.read_csv(samples_csv)
            else:
                samples_df = _prepare_samples(cfg, iteration_dir)
                samples_df.to_csv(samples_csv, index=False)

            vllm_model_path = prepare_model_for_vllm(cfg, current_model)
            generate_and_filter(cfg, samples_df, vllm_model_path, iteration_dir)

        elif args.phase == "train":
            curated_csv = iteration_dir / "curated.csv"
            if not curated_csv.is_file():
                raise FileNotFoundError(
                    f"Curated data not found: {curated_csv}. Run --phase generate first."
                )
            new_checkpoint = run_sft(cfg, curated_csv, iteration_dir, iteration)
            current_model = new_checkpoint
            cfg["_current_model_path"] = current_model

        else:
            new_checkpoint = run_iteration(cfg, iteration)
            current_model = new_checkpoint
            cfg["_current_model_path"] = current_model

    print(f"\nReFT complete. Final model: {current_model}")


if __name__ == "__main__":
    main()
