import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from calc_accuracy import canonicalize_smiles, tanimoto_similarity
from dataset import DEFAULT_INSTRUCTION
from DeepSeek_OCR_2 import (
    _apply_subset_sampling,
    _build_conversation_dataset,
    _resolve_path,
    apply_deepseek_runtime_patches,
    apply_transformers_compat_shims,
    bf16_supported,
    is_main_process,
    load_modeling_utils_from_loaded_model,
    maybe_init_wandb,
    resolve_local_model_path,
    set_seed,
)


@dataclass
class TrainSetConfig:
    train_csv: str
    data_mode: str
    pre_rendered_image_dir: str | None
    realistic_image_root: str | None
    instruction: str
    style: str | None
    mol_augment: bool | None
    include_condensed: bool | None
    max_samples: int | None
    sample_num: int | None


@dataclass
class RestartConfig:
    enable: bool
    dir: str
    auto_resume_from_latest_checkpoint: bool
    resume_from_checkpoint: str | None
    wandb_resume_same_run: bool


@dataclass
class RewardWeightConfig:
    validity: float
    tanimoto: float
    canon_smiles: float
    graph: float
    chiral: float


@dataclass
class RouterRLConfig:
    pretrained_weight_path: str
    train_sets: List[TrainSetConfig]
    seed: int
    output_dir: str
    batch_size: int
    grad_accum: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    allow_tf32: bool
    load_in_4bit: bool
    attn_implementation: str
    image_size: int
    base_size: int
    crop_mode: bool
    dataloader_num_workers: int
    dataloader_persistent_workers: bool
    enable_gradient_checkpointing: bool
    train_on_responses_only: bool
    generation_num: int
    generation_max_new_tokens: int
    generation_temperature: float
    generation_top_p: float
    grpo_beta: float
    grpo_adv_epsilon: float
    use_reference_model: bool
    reward_weights: RewardWeightConfig
    chiral_no_annotation_reward: float
    router_trainable_patterns: List[str]
    use_accelerate: bool
    accelerate_num_processes: int
    accelerate_gpu_ids: str | None
    restart: RestartConfig
    save_steps: int
    log_steps: int
    wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_api_key: str | None


class RouterRLDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        messages = item["messages"]
        if len(messages) < 2:
            raise ValueError("Expected at least 2 messages (user + assistant) in base dataset sample.")
        user_message = dict(messages[0])
        gold_smiles = str(messages[-1]["content"]).strip()
        return {"user_message": user_message, "gold_smiles": gold_smiles}


class PromptOnlyDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return features


class DeepSeekPromptBuilder:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        text_encode: Any,
        basic_image_transform_cls: Any,
        dynamic_preprocess_fn: Any,
        image_size: int,
        base_size: int,
        crop_mode: bool,
        train_on_responses_only: bool,
    ):
        from DeepSeek_OCR_2 import DeepSeekOCR2DataCollator

        self.collator = DeepSeekOCR2DataCollator(
            tokenizer=tokenizer,
            model=model,
            text_encode=text_encode,
            basic_image_transform_cls=basic_image_transform_cls,
            dynamic_preprocess_fn=dynamic_preprocess_fn,
            image_size=image_size,
            base_size=base_size,
            crop_mode=crop_mode,
            train_on_responses_only=train_on_responses_only,
        )

    def build_prompt_inputs(self, user_message: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        processed = self.collator.process_single_sample([user_message])
        input_ids = processed["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        images_seq_mask = processed["images_seq_mask"].unsqueeze(0).to(device)
        images_spatial_crop = processed["images_spatial_crop"].to(device)
        images = [(processed["images_crop"].to(device), processed["images_ori"].to(device))]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

    def build_completion_inputs(
        self, user_message: Dict[str, Any], completion_text: str, device: torch.device
    ) -> Dict[str, Any]:
        processed = self.collator.process_single_sample(
            [user_message, {"role": "<|Assistant|>", "content": completion_text}]
        )
        input_ids = processed["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        images_seq_mask = processed["images_seq_mask"].unsqueeze(0).to(device)
        images_spatial_crop = processed["images_spatial_crop"].to(device)
        images = [(processed["images_crop"].to(device), processed["images_ori"].to(device))]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "prompt_token_count": int(processed["prompt_token_count"]),
        }


def _is_distributed_launched() -> bool:
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("RANK") is not None:
        return True
    if os.environ.get("ACCELERATE_PROCESS_INDEX") is not None:
        return True
    return False


def _maybe_launch_with_accelerate(cfg: RouterRLConfig, cli_args: argparse.Namespace) -> None:
    if not cfg.use_accelerate:
        return
    if _is_distributed_launched():
        return
    if os.environ.get("CHEMSEEK_ACCELERATE_LAUNCHED") == "1":
        return
    if cfg.accelerate_num_processes <= 1:
        return

    child_env = os.environ.copy()
    child_env["CHEMSEEK_ACCELERATE_LAUNCHED"] = "1"
    if cfg.accelerate_gpu_ids not in (None, ""):
        child_env["CUDA_VISIBLE_DEVICES"] = str(cfg.accelerate_gpu_ids)

    script_path = str(Path(__file__).resolve())
    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(cfg.accelerate_num_processes),
        script_path,
        "--config",
        str(Path(cli_args.config).resolve()),
    ]
    print(f"Launching with accelerate: {' '.join(command)}")
    subprocess.run(command, env=child_env, check=True)
    sys.exit(0)


def _extract_smiles_candidate(text: str) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    line = raw.splitlines()[0].strip()
    if line.startswith("```"):
        line = line.strip("`").strip()
    line = line.replace(" ", "")
    return line


def _replace_empty(smiles: str) -> str:
    return smiles if isinstance(smiles, str) and smiles != "" else "<empty>"


def _compute_reward_components(
    gold_smiles: str,
    pred_smiles: str,
    *,
    chiral_no_annotation_reward: float,
) -> Dict[str, float]:
    canon_gold, _ = canonicalize_smiles(gold_smiles, ignore_cistrans=True)
    canon_pred, valid_pred = canonicalize_smiles(pred_smiles, ignore_cistrans=True)
    graph_gold, _ = canonicalize_smiles(gold_smiles, ignore_chiral=True, ignore_cistrans=True)
    graph_pred, _ = canonicalize_smiles(pred_smiles, ignore_chiral=True, ignore_cistrans=True)

    canon_gold = _replace_empty(canon_gold)
    canon_pred = _replace_empty(canon_pred)
    graph_gold = _replace_empty(graph_gold)
    graph_pred = _replace_empty(graph_pred)

    has_chiral = "@" in canon_gold
    chiral_acc = float(canon_gold == canon_pred) if has_chiral else float(chiral_no_annotation_reward)

    components = {
        "validity": float(valid_pred),
        "tanimoto": float(tanimoto_similarity(gold_smiles, pred_smiles)),
        "canon_smiles": float(canon_gold == canon_pred),
        "graph": float(graph_gold == graph_pred),
        "chiral": chiral_acc,
    }
    return components


def _combine_reward(components: Dict[str, float], weights: RewardWeightConfig) -> float:
    total = (
        weights.validity * components["validity"]
        + weights.tanimoto * components["tanimoto"]
        + weights.canon_smiles * components["canon_smiles"]
        + weights.graph * components["graph"]
        + weights.chiral * components["chiral"]
    )
    return float(total)


def _set_router_trainable_parameters(model: Any, patterns: List[str]) -> None:
    normalized_patterns = [p.lower().strip() for p in patterns if str(p).strip()]
    if not normalized_patterns:
        normalized_patterns = ["mlp.gate.", "router"]

    trainable_count = 0
    total_count = 0
    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        total_count += param.numel()
        lowered = name.lower()
        should_train = any(pattern in lowered for pattern in normalized_patterns)
        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
            if len(trainable_names) < 24:
                trainable_names.append(name)

    ratio = 100.0 * float(trainable_count) / float(max(total_count, 1))
    print(f"Router trainable params: {trainable_count:,} / {total_count:,} ({ratio:.4f}%)")
    if trainable_names:
        preview = "\n".join([f"  - {n}" for n in trainable_names])
        print(f"Trainable parameter name preview:\n{preview}")

    if trainable_count == 0:
        # DeepSeek-OCR-2 MoE router modules are typically under `mlp.gate` (MoEGate),
        # not necessarily containing the literal token "router".
        fallback_patterns = ["mlp.gate."]
        print(
            "No parameters matched configured router patterns. "
            f"Trying fallback patterns: {fallback_patterns}"
        )
        trainable_count = 0
        total_count = 0
        trainable_names = []
        for name, param in model.named_parameters():
            total_count += param.numel()
            lowered = name.lower()
            should_train = any(pattern in lowered for pattern in fallback_patterns)
            param.requires_grad = should_train
            if should_train:
                trainable_count += param.numel()
                if len(trainable_names) < 24:
                    trainable_names.append(name)

        ratio = 100.0 * float(trainable_count) / float(max(total_count, 1))
        print(f"Fallback router trainable params: {trainable_count:,} / {total_count:,} ({ratio:.4f}%)")
        if trainable_names:
            preview = "\n".join([f"  - {n}" for n in trainable_names])
            print(f"Fallback trainable parameter name preview:\n{preview}")

    if trainable_count == 0:
        raise RuntimeError(
            "No trainable parameters matched router_trainable_patterns and fallback patterns. "
            f"Patterns: {normalized_patterns}, fallback: ['mlp.gate.']"
        )


def _iter_dataloader_forever(loader: DataLoader) -> Iterator[List[Dict[str, Any]]]:
    while True:
        for batch in loader:
            yield batch


def _build_optimizer(cfg: RouterRLConfig, model: Any) -> AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for optimizer.")
    use_fused = bool(torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    return AdamW(
        params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=use_fused if use_fused else False,
    )


def _build_scheduler(cfg: RouterRLConfig, optimizer: AdamW):
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if cfg.warmup_steps > 0 and current_step < cfg.warmup_steps:
            return float(current_step + 1) / float(max(cfg.warmup_steps, 1))
        remain = cfg.max_steps - current_step
        total_decay = max(cfg.max_steps - cfg.warmup_steps, 1)
        return max(float(remain) / float(total_decay), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_completion_logprob(model: Any, completion_inputs: Dict[str, Any]) -> torch.Tensor:
    outputs = model(
        input_ids=completion_inputs["input_ids"],
        attention_mask=completion_inputs["attention_mask"],
        images=completion_inputs["images"],
        images_seq_mask=completion_inputs["images_seq_mask"],
        images_spatial_crop=completion_inputs["images_spatial_crop"],
    )
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = completion_inputs["input_ids"][:, 1:]
    token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    prompt_token_count = int(completion_inputs["prompt_token_count"])
    start_idx = max(prompt_token_count - 1, 0)
    completion_mask = torch.arange(shift_labels.shape[1], device=shift_labels.device) >= start_idx
    completion_mask = completion_mask.unsqueeze(0)
    selected = token_log_probs[completion_mask]
    if selected.numel() == 0:
        return token_log_probs.mean()
    return selected.mean()


def _compute_kl_like_penalty(
    model: Any,
    ref_model: Any,
    completion_inputs: Dict[str, Any],
) -> torch.Tensor:
    current_logp = _compute_completion_logprob(model, completion_inputs)
    with torch.no_grad():
        ref_logp = _compute_completion_logprob(ref_model, completion_inputs)
    return (current_logp - ref_logp).pow(2)


def _generate_completion(
    model: Any,
    tokenizer: Any,
    prompt_inputs: Dict[str, Any],
    cfg: RouterRLConfig,
) -> str:
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            images=prompt_inputs["images"],
            images_seq_mask=prompt_inputs["images_seq_mask"],
            images_spatial_crop=prompt_inputs["images_spatial_crop"],
            do_sample=True,
            temperature=cfg.generation_temperature,
            top_p=cfg.generation_top_p,
            max_new_tokens=cfg.generation_max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    completion_ids = generated[0, prompt_len:]
    text = tokenizer.decode(
        completion_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return _extract_smiles_candidate(text)


def _build_train_dataset(cfg: RouterRLConfig) -> Dataset:
    base_sets: List[Dataset] = []
    for ts in cfg.train_sets:
        train_df = pd.read_csv(ts.train_csv)
        train_df = _apply_subset_sampling(
            train_df,
            sample_num=ts.sample_num,
            max_samples=ts.max_samples,
            data_mode=ts.data_mode,
            seed=cfg.seed,
        )
        base_dataset = _build_conversation_dataset(
            dataframe=train_df,
            data_mode=ts.data_mode,
            instruction=ts.instruction,
            style=ts.style,
            mol_augment=ts.mol_augment,
            include_condensed=ts.include_condensed,
            pre_rendered_image_dir=ts.pre_rendered_image_dir,
            realistic_image_root=ts.realistic_image_root,
            use_rendered_smiles_as_label=True,
        )
        base_sets.append(base_dataset)
    merged = base_sets[0] if len(base_sets) == 1 else torch.utils.data.ConcatDataset(base_sets)
    return RouterRLDataset(merged)


def _to_device(model: Any) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_config(cfg: RouterRLConfig) -> None:
    if cfg.attn_implementation not in ("eager", "flash_attention_2"):
        raise ValueError("attn_implementation must be 'eager' or 'flash_attention_2'.")
    if cfg.image_size != 768:
        raise ValueError("image_size must be 768 for DeepSeek-OCR-2.")
    if cfg.base_size not in (768, 1024):
        raise ValueError("base_size must be 768 or 1024.")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if cfg.grad_accum <= 0:
        raise ValueError("grad_accum must be > 0.")
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0.")
    if cfg.save_steps <= 0:
        raise ValueError("save_steps must be > 0.")
    if cfg.log_steps <= 0:
        raise ValueError("log_steps must be > 0.")
    if cfg.generation_num <= 1:
        raise ValueError("generation_num must be > 1 for GRPO group normalization.")
    if cfg.generation_max_new_tokens <= 0:
        raise ValueError("generation_max_new_tokens must be > 0.")
    if not cfg.train_sets:
        raise ValueError("train_sets must contain at least one dataset.")
    for i, ts in enumerate(cfg.train_sets):
        if ts.data_mode not in ("dynamic", "pre_rendered", "realistic"):
            raise ValueError(f"train_sets[{i}].data_mode must be dynamic/pre_rendered/realistic.")
        if not os.path.isfile(ts.train_csv):
            raise ValueError(f"train_sets[{i}].train_csv not found: {ts.train_csv}")
        if ts.data_mode == "pre_rendered":
            if not ts.pre_rendered_image_dir or not os.path.isdir(ts.pre_rendered_image_dir):
                raise ValueError(f"train_sets[{i}].pre_rendered_image_dir is invalid.")
        if ts.data_mode == "realistic":
            if not ts.realistic_image_root or not os.path.isdir(ts.realistic_image_root):
                raise ValueError(f"train_sets[{i}].realistic_image_root is invalid.")


def load_config(config_path: str) -> RouterRLConfig:
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

    with open(config_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config root must be a mapping/object.")
    if "pretrained_weight_path" not in raw and "model_root" in raw:
        raw["pretrained_weight_path"] = raw["model_root"]
    if "pretrained_weight_path" not in raw:
        raise ValueError("Missing required config key: pretrained_weight_path")

    raw["pretrained_weight_path"] = _resolve_path(raw["pretrained_weight_path"], config_file.parent)
    raw["output_dir"] = _resolve_path(raw.get("output_dir", "./weight_router_rl"), config_file.parent)
    raw["batch_size"] = int(raw.get("batch_size", 2))
    raw["grad_accum"] = int(raw.get("grad_accum", 4))
    raw["max_steps"] = int(raw.get("max_steps", 500))
    raw["warmup_steps"] = int(raw.get("warmup_steps", 50))
    raw["learning_rate"] = float(raw.get("learning_rate", 2e-6))
    raw["weight_decay"] = float(raw.get("weight_decay", 0.0))
    raw["max_grad_norm"] = float(raw.get("max_grad_norm", 1.0))
    raw["allow_tf32"] = bool(raw.get("allow_tf32", True))
    raw["load_in_4bit"] = bool(raw.get("load_in_4bit", False))
    raw["attn_implementation"] = raw.get("attn_implementation", "flash_attention_2")
    raw["image_size"] = int(raw.get("image_size", 768))
    raw["base_size"] = int(raw.get("base_size", 1024))
    raw["crop_mode"] = bool(raw.get("crop_mode", True))
    raw["dataloader_num_workers"] = int(raw.get("dataloader_num_workers", 4))
    raw["dataloader_persistent_workers"] = bool(raw.get("dataloader_persistent_workers", False))
    raw["enable_gradient_checkpointing"] = bool(raw.get("enable_gradient_checkpointing", True))
    raw["train_on_responses_only"] = bool(raw.get("train_on_responses_only", True))
    raw["generation_num"] = int(raw.get("generation_num", 4))
    raw["generation_max_new_tokens"] = int(raw.get("generation_max_new_tokens", 96))
    raw["generation_temperature"] = float(raw.get("generation_temperature", 0.8))
    raw["generation_top_p"] = float(raw.get("generation_top_p", 0.95))
    raw["grpo_beta"] = float(raw.get("grpo_beta", 0.02))
    raw["grpo_adv_epsilon"] = float(raw.get("grpo_adv_epsilon", 1e-6))
    raw["use_reference_model"] = bool(raw.get("use_reference_model", False))
    raw["chiral_no_annotation_reward"] = float(raw.get("chiral_no_annotation_reward", 1.0))
    raw["router_trainable_patterns"] = list(raw.get("router_trainable_patterns", ["router"]))
    raw["use_accelerate"] = bool(raw.get("use_accelerate", False))
    raw["accelerate_num_processes"] = int(raw.get("accelerate_num_processes", 1))
    raw["accelerate_gpu_ids"] = raw.get("accelerate_gpu_ids", None)
    raw["save_steps"] = int(raw.get("save_steps", 50))
    raw["log_steps"] = int(raw.get("log_steps", 5))
    raw["wandb"] = bool(raw.get("wandb", False))
    raw["wandb_project"] = str(raw.get("wandb_project", "ChemSeek-OCR"))
    raw["wandb_run_name"] = raw.get("wandb_run_name", None)
    if raw.get("wandb_api_key") in ("", None):
        raw["wandb_api_key"] = os.getenv("WANDB_API_KEY", None)

    reward_raw = dict(raw.get("reward_weights", {}) or {})
    reward_raw.setdefault("validity", 2.0)
    reward_raw.setdefault("tanimoto", 1.0)
    reward_raw.setdefault("canon_smiles", 2.0)
    reward_raw.setdefault("graph", 1.5)
    reward_raw.setdefault("chiral", 1.5)
    raw["reward_weights"] = RewardWeightConfig(**reward_raw)

    restart_raw = dict(raw.get("restart", {}) or {})
    restart_raw.setdefault("enable", False)
    restart_raw.setdefault("dir", raw["output_dir"])
    restart_raw.setdefault("auto_resume_from_latest_checkpoint", False)
    restart_raw.setdefault("resume_from_checkpoint", None)
    restart_raw.setdefault("wandb_resume_same_run", False)
    restart_raw["dir"] = _resolve_path(restart_raw["dir"], config_file.parent)
    raw["restart"] = RestartConfig(**restart_raw)

    if "train_sets" not in raw:
        raw["train_sets"] = [
            {
                "train_csv": raw.get("train_csv"),
                "data_mode": raw.get("data_mode", "dynamic"),
                "pre_rendered_image_dir": raw.get("pre_rendered_image_dir"),
                "realistic_image_root": raw.get("realistic_image_root"),
                "instruction": raw.get("instruction", DEFAULT_INSTRUCTION),
                "style": raw.get("style", "molscribe_default"),
                "mol_augment": raw.get("mol_augment", True),
                "include_condensed": raw.get("include_condensed", True),
                "max_samples": raw.get("max_samples", None),
                "sample_num": raw.get("sample_num", None),
            }
        ]
    parsed_train_sets: List[TrainSetConfig] = []
    for item in raw["train_sets"]:
        item = dict(item)
        item["train_csv"] = _resolve_path(item["train_csv"], config_file.parent)
        if item.get("pre_rendered_image_dir"):
            item["pre_rendered_image_dir"] = _resolve_path(item["pre_rendered_image_dir"], config_file.parent)
        if item.get("realistic_image_root"):
            item["realistic_image_root"] = _resolve_path(item["realistic_image_root"], config_file.parent)
        if item.get("instruction") in ("", None):
            item["instruction"] = DEFAULT_INSTRUCTION
        item.setdefault("style", None)
        item.setdefault("mol_augment", None)
        item.setdefault("include_condensed", None)
        item.setdefault("max_samples", None)
        item.setdefault("sample_num", None)
        item.setdefault("pre_rendered_image_dir", None)
        item.setdefault("realistic_image_root", None)
        parsed_train_sets.append(TrainSetConfig(**item))
    raw["train_sets"] = parsed_train_sets

    for legacy_key in [
        "train_csv",
        "data_mode",
        "pre_rendered_image_dir",
        "realistic_image_root",
        "instruction",
        "style",
        "mol_augment",
        "include_condensed",
        "max_samples",
        "sample_num",
    ]:
        raw.pop(legacy_key, None)

    cfg = RouterRLConfig(**raw)
    _validate_config(cfg)
    return cfg


def parse_cli_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="GRPO router RL for DeepSeek-OCR-2.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(workspace / "router_rl_config.yaml"),
        help="Path to router rl YAML config.",
    )
    args, _ = parser.parse_known_args()
    return args


def _save_checkpoint(model: Any, tokenizer: Any, output_dir: str, step: int) -> None:
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))


def _log_wandb(payload: Dict[str, float], step: int) -> None:
    if "wandb" not in sys.modules:
        return
    try:
        import wandb  # type: ignore[import-not-found]

        wandb.log(payload, step=step)
    except Exception:
        pass


def main() -> None:
    cli_args = parse_cli_args()
    cfg = load_config(cli_args.config)
    _maybe_launch_with_accelerate(cfg, cli_args)

    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_seed(cfg.seed)
    apply_transformers_compat_shims()
    os.makedirs(cfg.output_dir, exist_ok=True)
    maybe_init_wandb(cfg)

    model_root = Path(cfg.pretrained_weight_path).resolve()
    model_path = resolve_local_model_path(model_root)
    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
        quantization_config=quantization_config,
    )
    apply_deepseek_runtime_patches(model)
    model.config.use_cache = True
    if cfg.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    text_encode, basic_image_transform_cls, dynamic_preprocess_fn = load_modeling_utils_from_loaded_model(model)
    prompt_builder = DeepSeekPromptBuilder(
        tokenizer=tokenizer,
        model=model,
        text_encode=text_encode,
        basic_image_transform_cls=basic_image_transform_cls,
        dynamic_preprocess_fn=dynamic_preprocess_fn,
        image_size=cfg.image_size,
        base_size=cfg.base_size,
        crop_mode=cfg.crop_mode,
        train_on_responses_only=cfg.train_on_responses_only,
    )

    _set_router_trainable_parameters(model, cfg.router_trainable_patterns)
    device = _to_device(model)
    model.to(device)

    reference_model = None
    if cfg.use_reference_model and cfg.grpo_beta > 0:
        reference_model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
            quantization_config=quantization_config,
        )
        apply_deepseek_runtime_patches(reference_model)
        reference_model.config.use_cache = True
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False
        reference_model.to(device)
        print("Reference model enabled for KL-like GRPO regularization.")
    elif cfg.grpo_beta > 0 and not cfg.use_reference_model:
        print("Warning: grpo_beta > 0 but use_reference_model=false, KL-like regularization is disabled.")

    train_dataset = _build_train_dataset(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        persistent_workers=cfg.dataloader_persistent_workers and cfg.dataloader_num_workers > 0,
        collate_fn=PromptOnlyDataCollator(),
        drop_last=True,
    )
    stream = _iter_dataloader_forever(train_loader)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    print(
        f"Starting GRPO router RL: max_steps={cfg.max_steps}, batch_size={cfg.batch_size}, "
        f"grad_accum={cfg.grad_accum}, generations={cfg.generation_num}"
    )

    running_total_loss = 0.0
    running_reward = 0.0
    running_validity = 0.0
    running_tanimoto = 0.0
    running_canon = 0.0
    running_graph = 0.0
    running_chiral = 0.0

    model.train()
    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss_value = 0.0
        accum_reward_value = 0.0
        accum_component_sums = {
            "validity": 0.0,
            "tanimoto": 0.0,
            "canon_smiles": 0.0,
            "graph": 0.0,
            "chiral": 0.0,
        }
        accum_component_count = 0

        for _ in range(cfg.grad_accum):
            batch = next(stream)
            micro_losses: List[torch.Tensor] = []

            for sample in batch:
                user_message = sample["user_message"]
                gold_smiles = str(sample["gold_smiles"])

                prompt_inputs = prompt_builder.build_prompt_inputs(user_message, device)
                completions: List[str] = []
                rewards: List[float] = []
                completion_logprobs: List[torch.Tensor] = []
                component_records: List[Dict[str, float]] = []

                for _gen_idx in range(cfg.generation_num):
                    completion = _generate_completion(model, tokenizer, prompt_inputs, cfg)
                    completions.append(completion)

                    components = _compute_reward_components(
                        gold_smiles,
                        completion,
                        chiral_no_annotation_reward=cfg.chiral_no_annotation_reward,
                    )
                    reward_value = _combine_reward(components, cfg.reward_weights)
                    rewards.append(reward_value)
                    component_records.append(components)

                    completion_inputs = prompt_builder.build_completion_inputs(user_message, completion, device)
                    logp = _compute_completion_logprob(model, completion_inputs)
                    completion_logprobs.append(logp)

                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + cfg.grpo_adv_epsilon)

                sample_losses: List[torch.Tensor] = []
                for j in range(cfg.generation_num):
                    term = -advantages[j].detach() * completion_logprobs[j]
                    if reference_model is not None and cfg.grpo_beta > 0:
                        completion_inputs = prompt_builder.build_completion_inputs(
                            user_message, completions[j], device
                        )
                        kl_like = _compute_kl_like_penalty(model, reference_model, completion_inputs)
                        term = term + cfg.grpo_beta * kl_like
                    sample_losses.append(term)
                sample_loss = torch.stack(sample_losses).mean()
                micro_losses.append(sample_loss)

                accum_reward_value += float(sum(rewards) / max(len(rewards), 1))
                for c in component_records:
                    for key in accum_component_sums:
                        accum_component_sums[key] += float(c[key])
                    accum_component_count += 1

            if not micro_losses:
                continue
            micro_loss = torch.stack(micro_losses).mean()
            loss_for_backward = micro_loss / cfg.grad_accum
            loss_for_backward.backward()
            accum_loss_value += float(loss_for_backward.detach().item())

        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.max_grad_norm,
            )
        optimizer.step()
        scheduler.step()

        running_total_loss += accum_loss_value
        running_reward += accum_reward_value / max(cfg.grad_accum, 1)
        if accum_component_count > 0:
            running_validity += accum_component_sums["validity"] / accum_component_count
            running_tanimoto += accum_component_sums["tanimoto"] / accum_component_count
            running_canon += accum_component_sums["canon_smiles"] / accum_component_count
            running_graph += accum_component_sums["graph"] / accum_component_count
            running_chiral += accum_component_sums["chiral"] / accum_component_count

        if step % cfg.log_steps == 0:
            denom = float(cfg.log_steps)
            metrics = {
                "train/loss": running_total_loss / denom,
                "train/reward": running_reward / denom,
                "train/validity": running_validity / denom,
                "train/tanimoto": running_tanimoto / denom,
                "train/canon_smiles": running_canon / denom,
                "train/graph": running_graph / denom,
                "train/chiral": running_chiral / denom,
                "train/lr": float(scheduler.get_last_lr()[0]),
            }
            print(
                f"[step {step}] "
                f"loss={metrics['train/loss']:.6f} "
                f"reward={metrics['train/reward']:.4f} "
                f"valid={metrics['train/validity']:.4f} "
                f"tanimoto={metrics['train/tanimoto']:.4f} "
                f"canon={metrics['train/canon_smiles']:.4f} "
                f"graph={metrics['train/graph']:.4f} "
                f"chiral={metrics['train/chiral']:.4f}"
            )
            if cfg.wandb and is_main_process():
                _log_wandb(metrics, step=step)
            running_total_loss = 0.0
            running_reward = 0.0
            running_validity = 0.0
            running_tanimoto = 0.0
            running_canon = 0.0
            running_graph = 0.0
            running_chiral = 0.0

        if step % cfg.save_steps == 0 and is_main_process():
            _save_checkpoint(model, tokenizer, cfg.output_dir, step)

    if is_main_process():
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved router RL model to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
