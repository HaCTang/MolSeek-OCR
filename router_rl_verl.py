"""GRPO Routing Replay RL for DeepSeek-OCR-2 using verl framework.

This script provides:
  1. Data preparation: Convert CSV datasets to parquet format for verl
  2. Custom reward function: SMILES-based reward for molecular OCR
  3. Custom dataset class: Load image+SMILES data for verl
  4. Router parameter freezing: external_lib for MoE routing replay
  5. Training launch: Configure and start verl GRPO training

Usage:
  # Step 1: Prepare training data (run in chemseek-ocr conda env)
  conda activate chemseek-ocr
  python router_rl_verl.py prepare-data --config router_rl_verl_config.yaml

  # Step 2: Launch GRPO training (run in chemseek-ocr-verl conda env)
  conda activate chemseek-ocr-verl
  python router_rl_verl.py train --config router_rl_verl_config.yaml
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

_FIND_SMILES_COLUMNS = [
    "SMILES", "smiles", "Smiles", "canonical_smiles",
    "canon_smiles", "smi", "isosmiles", "input",
]

_ID_CANDIDATE_COLUMNS = [
    "image_id", "id", "index", "file_name", "image_file",
    "sample_id", "mol_id", "compound_id",
]


# =========================================================================
# Configuration
# =========================================================================

def _resolve_path(path_str: str, base_dir: Optional[Path] = None) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir or SCRIPT_DIR) / p
    return str(p.resolve())


def load_yaml_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = config_path.parent
    if "model" in cfg:
        cfg["model"]["path"] = _resolve_path(cfg["model"].get("path", ""), base)
    if "output_dir" in cfg:
        cfg["output_dir"] = _resolve_path(cfg["output_dir"], base)
    for ts in cfg.get("data", {}).get("train_sets", []):
        ts["train_csv"] = _resolve_path(ts["train_csv"], base)
        if ts.get("pre_rendered_image_dir"):
            ts["pre_rendered_image_dir"] = _resolve_path(ts["pre_rendered_image_dir"], base)
        if ts.get("realistic_image_root"):
            ts["realistic_image_root"] = _resolve_path(ts["realistic_image_root"], base)
    cfg.setdefault("data", {})
    cfg["data"]["output_dir"] = _resolve_path(
        cfg["data"].get("output_dir", "./verl_data"), base
    )
    return cfg


# =========================================================================
# Data Preparation (run in chemseek-ocr conda env)
# =========================================================================

def _find_smiles_col(columns: list) -> str:
    for c in _FIND_SMILES_COLUMNS:
        if c in columns:
            return c
    raise ValueError(f"Cannot find SMILES column in: {columns}")


def _sanitize_filename(raw: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(raw))
    return safe.strip("_") or "sample"


def _resolve_sample_id(row, fallback_idx: int) -> str:
    import pandas as pd
    for col in _ID_CANDIDATE_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, float) and val.is_integer():
                raw = str(int(val))
            else:
                raw = str(val).strip()
            if raw:
                return _sanitize_filename(raw)
    return str(fallback_idx)


def _resolve_with_alt_ext(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    stem, _ = os.path.splitext(path)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp"):
        candidate = f"{stem}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


def prepare_data(cfg: dict) -> None:
    """Convert CSV datasets to parquet format for verl GRPO training.

    Must run in the chemseek-ocr conda env (needs pandas, PIL, etc.).
    """
    import pandas as pd

    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("training", {}).get("seed", 3407)
    instruction = cfg.get("data", {}).get(
        "instruction", "<image>\n Give me the SMILES of the molecule. "
    )

    all_rows: List[Dict[str, str]] = []
    for ts in cfg["data"].get("train_sets", []):
        csv_path = ts["train_csv"]
        data_mode = ts.get("data_mode", "pre_rendered")
        sample_num = ts.get("sample_num")

        df = pd.read_csv(csv_path)
        if sample_num and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)

        smiles_col = _find_smiles_col(list(df.columns))
        found, skipped = 0, 0

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            image_path: Optional[str] = None

            if data_mode == "pre_rendered":
                img_dir = ts.get("pre_rendered_image_dir", "")
                sid = _resolve_sample_id(row, row_idx)
                raw = os.path.join(img_dir, f"{sid}.png")
                image_path = _resolve_with_alt_ext(raw)

            elif data_mode == "realistic":
                img_root = ts.get("realistic_image_root", "")
                if "file_path" in row.index:
                    rel = str(row["file_path"]).strip()
                    raw = rel if os.path.isabs(rel) else os.path.join(img_root, rel)
                    image_path = _resolve_with_alt_ext(raw)

            elif data_mode == "dynamic":
                print(
                    "Warning: dynamic mode requires chemseek-ocr env with rdkit renderer. "
                    "Please pre-render images first, then use pre_rendered mode."
                )
                continue

            if image_path is None:
                skipped += 1
                continue

            gold_smiles = str(row[smiles_col])
            all_rows.append({
                "image_path": image_path,
                "ground_truth": gold_smiles,
            })
            found += 1

        print(f"  [{data_mode}] {csv_path}: found={found}, skipped={skipped}")

    if not all_rows:
        print("Error: No valid samples found. Check image paths and data_mode.")
        return

    random.seed(seed)
    random.shuffle(all_rows)

    prompts = []
    image_paths = []
    data_sources = []
    ground_truths = []

    for r in all_rows:
        prompts.append(json.dumps(
            [{"role": "user", "content": instruction}]
        ))
        image_paths.append(r["image_path"])
        data_sources.append("chemseek_ocr")
        ground_truths.append(json.dumps({"ground_truth": r["ground_truth"]}))

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "prompt": prompts,
        "image_path": image_paths,
        "data_source": data_sources,
        "reward_model": ground_truths,
    })

    train_path = output_dir / "train.parquet"
    pq.write_table(table, str(train_path))

    # Also create a small validation split (last 2% or at least 100 samples)
    n_val = max(100, len(all_rows) // 50)
    val_rows = all_rows[-n_val:]
    val_table = pa.table({
        "prompt": [json.dumps([{"role": "user", "content": instruction}]) for _ in val_rows],
        "image_path": [r["image_path"] for r in val_rows],
        "data_source": ["chemseek_ocr"] * len(val_rows),
        "reward_model": [json.dumps({"ground_truth": r["ground_truth"]}) for r in val_rows],
    })
    val_path = output_dir / "val.parquet"
    pq.write_table(val_table, str(val_path))

    print(f"Saved {len(all_rows)} train samples to {train_path}")
    print(f"Saved {len(val_rows)} val samples to {val_path}")


# =========================================================================
# Custom Reward Function (loaded by verl via reward.custom_reward_function)
# =========================================================================

def _extract_smiles_candidate(text: str) -> str:
    if not text:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    line = raw.splitlines()[0].strip()
    if line.startswith("```"):
        line = line.strip("`").strip()
    line = line.replace(" ", "")
    return line


def _replace_empty(s: Optional[str]) -> str:
    return s if isinstance(s, str) and s != "" else "<empty>"


def _canonicalize_smiles(
    smiles: str, ignore_chiral: bool = False, ignore_cistrans: bool = False
) -> tuple:
    """Returns (canonical_smiles_or_None, is_valid)."""
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


def _compute_reward_components(
    gold_smiles: str, pred_smiles: str, chiral_no_annotation_reward: float = 1.0
) -> Dict[str, float]:
    canon_gold, _ = _canonicalize_smiles(gold_smiles, ignore_cistrans=True)
    canon_pred, valid_pred = _canonicalize_smiles(pred_smiles, ignore_cistrans=True)
    graph_gold, _ = _canonicalize_smiles(gold_smiles, ignore_chiral=True, ignore_cistrans=True)
    graph_pred, _ = _canonicalize_smiles(pred_smiles, ignore_chiral=True, ignore_cistrans=True)

    canon_gold = _replace_empty(canon_gold)
    canon_pred = _replace_empty(canon_pred)
    graph_gold = _replace_empty(graph_gold)
    graph_pred = _replace_empty(graph_pred)

    has_chiral = "@" in (canon_gold or "")
    chiral_acc = float(canon_gold == canon_pred) if has_chiral else chiral_no_annotation_reward

    return {
        "validity": float(valid_pred),
        "tanimoto": _tanimoto_similarity(gold_smiles, pred_smiles),
        "canon_smiles": float(canon_gold == canon_pred),
        "graph": float(graph_gold == graph_pred),
        "chiral": chiral_acc,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """SMILES-based reward function for molecular OCR.

    Signature matches verl's reward function interface:
      compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    """
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            gold_smiles = gt.get("ground_truth", ground_truth)
        except (json.JSONDecodeError, TypeError):
            gold_smiles = ground_truth
    elif isinstance(ground_truth, dict):
        gold_smiles = ground_truth.get("ground_truth", "")
    else:
        gold_smiles = str(ground_truth)

    pred_smiles = _extract_smiles_candidate(str(solution_str))

    w = kwargs.get("reward_weights", {})
    w_validity = float(w.get("validity", 2.0))
    w_tanimoto = float(w.get("tanimoto", 1.0))
    w_canon = float(w.get("canon_smiles", 2.0))
    w_graph = float(w.get("graph", 1.5))
    w_chiral = float(w.get("chiral", 1.5))
    chiral_no_ann = float(kwargs.get("chiral_no_annotation_reward", 1.0))

    components = _compute_reward_components(gold_smiles, pred_smiles, chiral_no_ann)

    total = (
        w_validity * components["validity"]
        + w_tanimoto * components["tanimoto"]
        + w_canon * components["canon_smiles"]
        + w_graph * components["graph"]
        + w_chiral * components["chiral"]
    )
    return float(total)


# =========================================================================
# Custom Dataset Class (loaded by verl via data.custom_cls)
# =========================================================================

import torch
from torch.utils.data import Dataset


class ChemSeekOCRDataset(Dataset):
    """Custom dataset for DeepSeek-OCR-2 GRPO training with verl.

    Reads parquet files produced by ``prepare_data`` and loads images from disk
    on the fly, keeping parquet files small.  Returns the dict format that
    verl's rollout & actor pipeline expects.
    """

    def __init__(
        self,
        data_files,
        tokenizer=None,
        config=None,
        processor=None,
        max_samples: int = -1,
    ):
        import datasets as hf_datasets

        if not isinstance(data_files, (list, tuple)):
            data_files = [data_files]

        dfs = []
        for f in data_files:
            if str(f).endswith(".parquet"):
                dfs.append(hf_datasets.load_dataset("parquet", data_files=str(f))["train"])
            elif str(f).endswith(".json"):
                dfs.append(hf_datasets.load_dataset("json", data_files=str(f))["train"])
            else:
                raise ValueError(f"Unsupported file format: {f}")

        self.data = hf_datasets.concatenate_datasets(dfs) if len(dfs) > 1 else dfs[0]

        if 0 < max_samples < len(self.data):
            import numpy as np
            indices = np.random.choice(len(self.data), size=max_samples, replace=False)
            self.data = self.data.select(indices.tolist())

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        print(f"[ChemSeekOCRDataset] loaded {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        row = self.data[idx]

        image_path = row.get("image_path", "")
        if image_path and os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (768, 768), (255, 255, 255))

        prompt_raw = row.get("prompt", '[{"role": "user", "content": "<image>\\n Give me the SMILES of the molecule. "}]')
        if isinstance(prompt_raw, str):
            prompt = json.loads(prompt_raw)
        else:
            prompt = list(prompt_raw)

        messages = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str) and "<image>" in content:
                parts = content.split("<image>")
                content_list = []
                for i, part in enumerate(parts):
                    if i > 0:
                        content_list.append({"type": "image", "image": image})
                    if part:
                        content_list.append({"type": "text", "text": part})
                messages.append({"role": role, "content": content_list})
            else:
                messages.append({"role": role, "content": content})

        reward_model = row.get("reward_model", "{}")
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except (json.JSONDecodeError, TypeError):
                reward_model = {"ground_truth": reward_model}

        return {
            "raw_prompt": messages,
            "data_source": row.get("data_source", "chemseek_ocr"),
            "reward_model": reward_model,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }

    def resume_dataset_state(self):
        pass


# =========================================================================
# Router Parameter Freezing – external_lib module
# =========================================================================

_ROUTER_FREEZE_EXT_CODE = '''\
"""Auto-generated external_lib for verl: freeze MoE gate parameters (routing replay)."""
import os
import torch
from transformers import AutoModel, AutoModelForCausalLM

EXPERT_PATTERNS = os.environ.get(
    "CHEMSEEK_EXPERT_PATTERNS",
    "mlp.experts.,mlp.shared_experts."
).split(",")


def _freeze_non_expert_params(model):
    trainable_count, total_count, frozen_gates = 0, 0, 0
    trainable_names = []
    for name, param in model.named_parameters():
        total_count += param.numel()
        lowered = name.lower()
        should_train = any(p.strip() in lowered for p in EXPERT_PATTERNS if p.strip())
        if ".mlp.gate" in lowered:
            should_train = False
            frozen_gates += 1
        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
            if len(trainable_names) < 16:
                trainable_names.append(name)
    ratio = 100.0 * trainable_count / max(total_count, 1)
    print(
        f"[Routing Replay] Trainable: {trainable_count:,}/{total_count:,} "
        f"({ratio:.2f}%), frozen MoE gates: {frozen_gates}"
    )
    if trainable_names:
        print("[Routing Replay] Trainable param preview:")
        for n in trainable_names:
            print(f"  - {n}")
    return model


_orig_auto_model_fp = AutoModel.from_pretrained.__func__
_orig_auto_causal_fp = AutoModelForCausalLM.from_pretrained.__func__


@classmethod
def _patched_auto_model_fp(cls, *args, **kwargs):
    model = _orig_auto_model_fp(cls, *args, **kwargs)
    return _freeze_non_expert_params(model)


@classmethod
def _patched_auto_causal_fp(cls, *args, **kwargs):
    model = _orig_auto_causal_fp(cls, *args, **kwargs)
    return _freeze_non_expert_params(model)


AutoModel.from_pretrained = _patched_auto_model_fp
AutoModelForCausalLM.from_pretrained = _patched_auto_causal_fp
print("[Routing Replay] Patched AutoModel & AutoModelForCausalLM for expert-only training.")
'''


def _write_ext_module(target_dir: Path) -> str:
    """Write the router-freeze external lib module and return its module name."""
    module_name = "_chemseek_router_freeze"
    ext_path = target_dir / f"{module_name}.py"
    ext_path.write_text(_ROUTER_FREEZE_EXT_CODE, encoding="utf-8")
    print(f"Wrote external_lib module: {ext_path}")
    return module_name


# =========================================================================
# Training Launch (run in chemseek-ocr-verl conda env)
# =========================================================================

def train(cfg: dict, config_path: str) -> None:
    """Build verl GRPO config overrides and launch training."""

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    grpo_cfg = cfg.get("grpo", {})
    gpu_cfg = cfg.get("gpu", {})
    rw_cfg = cfg.get("reward_weights", {})
    wandb_cfg = cfg.get("wandb", {})
    output_dir = cfg.get("output_dir", str(SCRIPT_DIR / "weight_router_rl_verl"))

    data_dir = Path(data_cfg["output_dir"])
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"

    if not train_parquet.is_file():
        print(f"Error: Training data not found at {train_parquet}")
        print("Run 'python router_rl_verl.py prepare-data --config ...' first (in chemseek-ocr env).")
        sys.exit(1)

    ext_module_name = _write_ext_module(SCRIPT_DIR)

    expert_patterns = cfg.get("expert_trainable_patterns", ["mlp.experts.", "mlp.shared_experts."])
    expert_patterns_str = ",".join(expert_patterns)

    script_path = str(SCRIPT_DIR / "router_rl_verl.py")

    n_gpus = gpu_cfg.get("n_gpus_per_node", 4)
    tp_size = gpu_cfg.get("tensor_model_parallel_size", 2)
    gpu_mem_util = gpu_cfg.get("gpu_memory_utilization", 0.5)

    train_batch = train_cfg.get("train_batch_size", n_gpus * 4)
    micro_bs = train_cfg.get("ppo_micro_batch_size_per_gpu", 2)
    mini_bs = train_cfg.get("ppo_mini_batch_size", train_batch)

    group_size = grpo_cfg.get("group_size", 6)
    kl_coef = grpo_cfg.get("kl_loss_coef", 0.01)
    kl_type = grpo_cfg.get("kl_loss_type", "low_var_kl")
    lr = train_cfg.get("learning_rate", 2e-6)
    epochs = train_cfg.get("total_epochs", 10)
    save_freq = cfg.get("save_freq", 50)
    test_freq = cfg.get("test_freq", 5)

    max_prompt_len = data_cfg.get("max_prompt_length", 4096)
    max_resp_len = data_cfg.get("max_response_length", 128)

    param_offload = str(gpu_cfg.get("param_offload", False))
    optim_offload = str(gpu_cfg.get("optimizer_offload", False))
    ref_offload = str(gpu_cfg.get("ref_param_offload", True))

    logger_list = '["console"]'
    if wandb_cfg.get("enabled", False):
        logger_list = '["console","wandb"]'

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # Algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        # Model
        f"actor_rollout_ref.model.path={model_cfg['path']}",
        "actor_rollout_ref.model.trust_remote_code=True",
        f"actor_rollout_ref.model.external_lib={ext_module_name}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # Actor
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.optim.weight_decay={train_cfg.get('weight_decay', 0.01)}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_bs}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={kl_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={kl_type}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.freeze_vision_tower=True",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optim_offload}",
        "+actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        f"actor_rollout_ref.rollout.n={group_size}",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.top_p=0.9",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_bs}",
        # Reference model
        f"actor_rollout_ref.ref.fsdp_config.param_offload={ref_offload}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_bs}",
        # Data
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.train_batch_size={train_batch}",
        f"data.max_prompt_length={max_prompt_len}",
        f"data.max_response_length={max_resp_len}",
        "data.filter_overlong_prompts=False",
        "data.truncation=left",
        "data.image_key=images",
        "data.trust_remote_code=True",
        f"data.custom_cls.path={script_path}",
        "data.custom_cls.name=ChemSeekOCRDataset",
        # Reward
        f"reward.custom_reward_function.path={script_path}",
        "reward.custom_reward_function.name=compute_score",
        f"reward.custom_reward_function.reward_kwargs.reward_weights.validity={rw_cfg.get('validity', 2.0)}",
        f"reward.custom_reward_function.reward_kwargs.reward_weights.tanimoto={rw_cfg.get('tanimoto', 1.0)}",
        f"reward.custom_reward_function.reward_kwargs.reward_weights.canon_smiles={rw_cfg.get('canon_smiles', 2.0)}",
        f"reward.custom_reward_function.reward_kwargs.reward_weights.graph={rw_cfg.get('graph', 1.5)}",
        f"reward.custom_reward_function.reward_kwargs.reward_weights.chiral={rw_cfg.get('chiral', 1.5)}",
        f"reward.custom_reward_function.reward_kwargs.chiral_no_annotation_reward={cfg.get('chiral_no_annotation_reward', 1.0)}",
        # Trainer
        "trainer.critic_warmup=0",
        f"trainer.logger={logger_list}",
        f"trainer.project_name={wandb_cfg.get('project', 'ChemSeek-OCR')}",
        f"trainer.experiment_name={wandb_cfg.get('run_name', 'router-grpo-verl')}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.total_epochs={epochs}",
        f"trainer.default_local_dir={output_dir}",
    ]

    env = os.environ.copy()
    env["CHEMSEEK_EXPERT_PATTERNS"] = expert_patterns_str
    pythonpath = str(SCRIPT_DIR)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    if train_cfg.get("allow_tf32", True):
        env.setdefault("NVIDIA_TF32_OVERRIDE", "1")

    if wandb_cfg.get("enabled", False):
        api_key = wandb_cfg.get("api_key") or os.environ.get("WANDB_API_KEY")
        if api_key:
            env["WANDB_API_KEY"] = api_key

    gpu_ids = gpu_cfg.get("gpu_ids")
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    print("=" * 72)
    print("Launching verl GRPO training")
    print("=" * 72)
    print(f"Model:       {model_cfg['path']}")
    print(f"Train data:  {train_parquet}")
    print(f"GPUs:        {n_gpus} (TP={tp_size})")
    print(f"Group size:  {group_size}")
    print(f"Batch size:  {train_batch}")
    print(f"LR:          {lr}")
    print(f"Epochs:      {epochs}")
    print(f"Output:      {output_dir}")
    print(f"Expert patterns: {expert_patterns}")
    print("=" * 72)
    print("Command:")
    print("  " + " \\\n    ".join(cmd[:3]) + " \\")
    print("    " + " \\\n    ".join(cmd[3:]))
    print("=" * 72)

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO Routing Replay RL for DeepSeek-OCR-2 (verl)"
    )
    subparsers = parser.add_subparsers(dest="command")

    prep = subparsers.add_parser(
        "prepare-data", help="Convert CSV datasets to parquet (run in chemseek-ocr env)"
    )
    prep.add_argument("--config", type=str, default=str(SCRIPT_DIR / "router_rl_verl_config.yaml"))

    tr = subparsers.add_parser(
        "train", help="Launch verl GRPO training (run in chemseek-ocr-verl env)"
    )
    tr.add_argument("--config", type=str, default=str(SCRIPT_DIR / "router_rl_verl_config.yaml"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command is None:
        print("Usage: python router_rl_verl.py {prepare-data,train} --config CONFIG")
        sys.exit(1)

    cfg = load_yaml_config(args.config)

    if args.command == "prepare-data":
        prepare_data(cfg)
    elif args.command == "train":
        train(cfg, args.config)


if __name__ == "__main__":
    main()
