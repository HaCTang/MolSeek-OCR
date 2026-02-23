"""GSPO RL for DeepSeek-OCR-2 using verl framework.

This script provides:
  1. Custom reward function: SMILES-based reward for molecular OCR
  2. Custom dataset class: Load image+SMILES data for verl
  3. Training launch: Configure and start verl GSPO training

GSPO (Group Sequence Policy Optimization) replaces KL regularization with
tight symmetric clipping (clip_ratio_low / clip_ratio_high), giving more
stable updates on MoE architectures.

Usage:
  # Step 1: Prepare training data (see prepare_verl_data.py)
  python prepare_verl_data.py --config gspo_rl_verl_config.yaml

  # Step 2: Launch GSPO training
  python gspo_rl_verl.py --config gspo_rl_verl_config.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


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
# Custom Reward Function (loaded by verl via reward.custom_reward_function)
# =========================================================================

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass


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
    gold_smiles: str, pred_smiles: str, chiral_no_annotation_reward: float = 0.0
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
    chiral_no_ann = float(kwargs.get("chiral_no_annotation_reward", 0.0))

    components = _compute_reward_components(gold_smiles, pred_smiles, chiral_no_ann)

    weighted_sum = (
        w_validity * components["validity"]
        + w_tanimoto * components["tanimoto"]
        + w_canon * components["canon_smiles"]
        + w_graph * components["graph"]
        + w_chiral * components["chiral"]
    )
    w_total = w_validity + w_tanimoto + w_canon + w_graph + w_chiral
    return float(weighted_sum / w_total) if w_total > 0 else 0.0


# =========================================================================
# Custom Dataset Class (loaded by verl via data.custom_cls)
# =========================================================================

import torch
from torch.utils.data import Dataset


class ChemSeekOCRDataset(Dataset):
    """Custom dataset for DeepSeek-OCR-2 GSPO training with verl.

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
# Training Launch (run in chemseek-ocr-verl conda env)
# =========================================================================

def train(cfg: dict, config_path: str) -> None:
    """Build verl GSPO config overrides and launch training."""

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    gspo_cfg = cfg.get("gspo", {})
    gpu_cfg = cfg.get("gpu", {})
    rw_cfg = cfg.get("reward_weights", {})
    wandb_cfg = cfg.get("wandb", {})
    output_dir = cfg.get("output_dir", str(SCRIPT_DIR / "weight_gspo_rl_verl"))

    data_dir = Path(data_cfg["output_dir"])
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"

    if not train_parquet.is_file():
        print(f"Error: Training data not found at {train_parquet}")
        print("Run 'python prepare_verl_data.py --config ...' first.")
        sys.exit(1)

    script_path = str(SCRIPT_DIR / "gspo_rl_verl.py")

    n_gpus = gpu_cfg.get("n_gpus_per_node", 4)
    tp_size = gpu_cfg.get("tensor_model_parallel_size", 2)
    gpu_mem_util = gpu_cfg.get("gpu_memory_utilization", 0.5)

    train_batch = train_cfg.get("train_batch_size", n_gpus * 4)
    micro_bs = train_cfg.get("ppo_micro_batch_size_per_gpu", 2)
    mini_bs = train_cfg.get("ppo_mini_batch_size", train_batch)

    group_size = gspo_cfg.get("group_size", 8)
    clip_ratio_low = gspo_cfg.get("clip_ratio_low", 3e-4)
    clip_ratio_high = gspo_cfg.get("clip_ratio_high", 4e-4)
    clip_ratio_c = gspo_cfg.get("clip_ratio_c", 10.0)
    use_kl_loss = gspo_cfg.get("use_kl_loss", False)
    kl_coef = gspo_cfg.get("kl_loss_coef", 0.0)
    loss_agg_mode = gspo_cfg.get("loss_agg_mode", "seq-mean-token-mean")
    use_dynamic_bsz = gspo_cfg.get("use_dynamic_bsz", True)

    lr = train_cfg.get("learning_rate", 1e-6)
    epochs = train_cfg.get("total_epochs", 10)
    save_freq = cfg.get("save_freq", 50)
    test_freq = cfg.get("test_freq", 5)

    max_prompt_len = data_cfg.get("max_prompt_length", 4096)
    max_resp_len = data_cfg.get("max_response_length", 128)

    actor_max_token_len_per_gpu = (max_prompt_len + max_resp_len) * 2

    param_offload = str(gpu_cfg.get("param_offload", False))
    optim_offload = str(gpu_cfg.get("optimizer_offload", False))
    ref_offload = str(gpu_cfg.get("ref_param_offload", True))

    logger_list = '["console"]'
    if wandb_cfg.get("enabled", False):
        logger_list = '["console","wandb"]'

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # Algorithm: GSPO uses GRPO advantage estimator with gspo loss mode
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        # Model
        f"actor_rollout_ref.model.path={model_cfg['path']}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # Actor – GSPO-specific parameters
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.optim.weight_decay={train_cfg.get('weight_decay', 0.1)}",
        f"actor_rollout_ref.actor.optim.clip_grad={train_cfg.get('clip_grad', 1.0)}",
        f"actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
        f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
        f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
        f"actor_rollout_ref.actor.clip_ratio_c={clip_ratio_c}",
        f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={kl_coef}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_bs}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.freeze_vision_tower=True",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optim_offload}",
        # Rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        f"actor_rollout_ref.rollout.n={group_size}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
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
        "data.return_raw_chat=True",
        "data.filter_overlong_prompts=True",
        "data.filter_overlong_prompts_workers=16",
        "data.truncation=error",
        "data.image_key=images",
        "data.trust_remote_code=True",
        f"data.custom_cls.path={script_path}",
        "data.custom_cls.name=ChemSeekOCRDataset",
        # Reward
        "reward.reward_manager.name=naive",
        f"++reward.custom_reward_function.path={script_path}",
        "++reward.custom_reward_function.name=compute_score",
        f"++reward.custom_reward_function.reward_kwargs.reward_weights.validity={rw_cfg.get('validity', 2.0)}",
        f"++reward.custom_reward_function.reward_kwargs.reward_weights.tanimoto={rw_cfg.get('tanimoto', 1.0)}",
        f"++reward.custom_reward_function.reward_kwargs.reward_weights.canon_smiles={rw_cfg.get('canon_smiles', 2.0)}",
        f"++reward.custom_reward_function.reward_kwargs.reward_weights.graph={rw_cfg.get('graph', 1.5)}",
        f"++reward.custom_reward_function.reward_kwargs.reward_weights.chiral={rw_cfg.get('chiral', 1.5)}",
        f"++reward.custom_reward_function.reward_kwargs.chiral_no_annotation_reward={cfg.get('chiral_no_annotation_reward', 1.0)}",
        # Trainer
        "trainer.critic_warmup=0",
        f"trainer.logger={logger_list}",
        f"trainer.project_name={wandb_cfg.get('project', 'ChemSeek-OCR')}",
        f"trainer.experiment_name={wandb_cfg.get('run_name', 'gspo-verl')}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.total_epochs={epochs}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.val_before_train=False",
    ]

    env = os.environ.copy()
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
    print("Launching verl GSPO training")
    print("=" * 72)
    print(f"Model:            {model_cfg['path']}")
    print(f"Train data:       {train_parquet}")
    print(f"GPUs:             {n_gpus} (TP={tp_size})")
    print(f"Group size:       {group_size}")
    print(f"Batch size:       {train_batch}")
    print(f"LR:               {lr}")
    print(f"Epochs:           {epochs}")
    print(f"GSPO clip_low:    {clip_ratio_low}")
    print(f"GSPO clip_high:   {clip_ratio_high}")
    print(f"GSPO clip_c:      {clip_ratio_c}")
    print(f"Loss agg mode:    {loss_agg_mode}")
    print(f"KL loss:          {use_kl_loss} (coef={kl_coef})")
    print(f"Output:           {output_dir}")
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSPO RL for DeepSeek-OCR-2 (verl)"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(SCRIPT_DIR / "gspo_rl_verl_config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
