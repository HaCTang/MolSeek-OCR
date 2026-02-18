import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from dataset import DEFAULT_INSTRUCTION
from DeepSeek_OCR_2 import (
    CleanEvalMetricsTrainer,
    DeepSeekOCR2DataCollator,
    PeriodicMultiValEvalCallback,
    _apply_subset_sampling,
    _build_conversation_dataset,
    _resolve_path,
    _resolve_resume_checkpoint,
    _torch_load_resume_compat,
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
class ValSetConfig:
    val_csv: str
    data_mode: str
    pre_rendered_image_dir: str | None
    realistic_image_root: str | None
    instruction: str
    style: str | None
    mol_augment: bool | None
    include_condensed: bool | None
    max_samples: int | None
    sample_num: int | None
    name: str | None


@dataclass
class RestartConfig:
    enable: bool
    dir: str
    auto_resume_from_latest_checkpoint: bool
    resume_from_checkpoint: str | None
    wandb_resume_same_run: bool


@dataclass
class FullSFTConfig:
    pretrained_weight_path: str
    train_sets: List[TrainSetConfig]
    enable_val_sets: bool
    val_sets: List[ValSetConfig]
    eval_every_steps: int | None
    eval_on_start: bool
    eval_on_end: bool
    seed: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    epochs: float
    save_steps: int | None
    optim: str
    allow_tf32: bool
    enable_gradient_checkpointing: bool
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    image_size: int
    base_size: int
    crop_mode: bool
    load_in_4bit: bool
    attn_implementation: str
    output_dir: str
    ddp_find_unused_parameters: bool
    ddp_backend: str | None
    ddp_timeout_seconds: int
    restart: RestartConfig
    use_accelerate: bool
    accelerate_num_processes: int
    accelerate_gpu_ids: str | None
    train_on_responses_only: bool
    wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_api_key: str | None


def _validate_config(cfg: FullSFTConfig) -> None:
    if cfg.base_size not in (768, 1024):
        raise ValueError(f"Unsupported base_size={cfg.base_size}. Use 768 or 1024.")
    if cfg.image_size != 768:
        raise ValueError(f"Unsupported image_size={cfg.image_size}. Use image_size=768.")
    if cfg.attn_implementation not in ("eager", "flash_attention_2"):
        raise ValueError("attn_implementation must be 'eager' or 'flash_attention_2'.")
    if cfg.optim not in ("adamw_torch", "adamw_torch_fused"):
        raise ValueError("optim must be 'adamw_torch' or 'adamw_torch_fused'")
    if cfg.save_steps is not None and cfg.save_steps <= 0:
        raise ValueError("save_steps must be > 0 when provided")
    if cfg.dataloader_prefetch_factor <= 0:
        raise ValueError("dataloader_prefetch_factor must be > 0")
    if cfg.ddp_backend not in (None, "nccl", "gloo", "mpi"):
        raise ValueError("ddp_backend must be one of: nccl, gloo, mpi, or null")
    if cfg.ddp_timeout_seconds <= 0:
        raise ValueError("ddp_timeout_seconds must be > 0")
    if cfg.accelerate_num_processes <= 0:
        raise ValueError("accelerate_num_processes must be > 0")
    if cfg.restart.enable and not cfg.restart.dir:
        raise ValueError("restart.dir is required when restart.enable=true")
    if cfg.restart.enable and cfg.restart.resume_from_checkpoint not in (None, ""):
        if not os.path.isdir(cfg.restart.resume_from_checkpoint):
            raise ValueError(f"restart.resume_from_checkpoint not found: {cfg.restart.resume_from_checkpoint}")
    if cfg.accelerate_gpu_ids is not None:
        gpu_ids = [x.strip() for x in str(cfg.accelerate_gpu_ids).split(",") if x.strip()]
        if not gpu_ids:
            raise ValueError("accelerate_gpu_ids must contain at least one GPU id, e.g. '0,1'")
        if cfg.use_accelerate and cfg.accelerate_num_processes > len(gpu_ids):
            raise ValueError(
                "accelerate_num_processes cannot exceed number of ids in accelerate_gpu_ids "
                f"({cfg.accelerate_num_processes} > {len(gpu_ids)})"
            )
    if not cfg.train_sets:
        raise ValueError("train_sets must contain at least one training dataset config.")
    for i, ts in enumerate(cfg.train_sets):
        if ts.data_mode not in ("dynamic", "pre_rendered", "realistic"):
            raise ValueError(f"train_sets[{i}].data_mode must be 'dynamic', 'pre_rendered' or 'realistic'")
        if not os.path.isfile(ts.train_csv):
            raise ValueError(f"train_sets[{i}].train_csv not found: {ts.train_csv}")
        if ts.data_mode == "dynamic":
            if not ts.style:
                raise ValueError(f"train_sets[{i}].style is required for dynamic mode")
            if ts.mol_augment is None:
                raise ValueError(f"train_sets[{i}].mol_augment is required for dynamic mode")
            if ts.include_condensed is None:
                raise ValueError(f"train_sets[{i}].include_condensed is required for dynamic mode")
        elif ts.data_mode == "pre_rendered":
            if not ts.pre_rendered_image_dir:
                raise ValueError(f"train_sets[{i}].pre_rendered_image_dir is required for pre_rendered mode")
            if not os.path.isdir(ts.pre_rendered_image_dir):
                raise ValueError(f"train_sets[{i}].pre_rendered_image_dir not found: {ts.pre_rendered_image_dir}")
        else:
            if not ts.realistic_image_root:
                raise ValueError(f"train_sets[{i}].realistic_image_root is required for realistic mode")
            if not os.path.isdir(ts.realistic_image_root):
                raise ValueError(f"train_sets[{i}].realistic_image_root not found: {ts.realistic_image_root}")
    if cfg.eval_every_steps is not None and cfg.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be > 0 when provided")
    if cfg.enable_val_sets:
        if not cfg.val_sets:
            raise ValueError("enable_val_sets=true but val_sets is empty")
        for i, vs in enumerate(cfg.val_sets):
            if vs.data_mode not in ("dynamic", "pre_rendered", "realistic"):
                raise ValueError(f"val_sets[{i}].data_mode must be 'dynamic', 'pre_rendered' or 'realistic'")
            if not os.path.isfile(vs.val_csv):
                raise ValueError(f"val_sets[{i}].val_csv not found: {vs.val_csv}")
            if vs.data_mode == "dynamic":
                if not vs.style:
                    raise ValueError(f"val_sets[{i}].style is required for dynamic mode")
                if vs.mol_augment is None:
                    raise ValueError(f"val_sets[{i}].mol_augment is required for dynamic mode")
                if vs.include_condensed is None:
                    raise ValueError(f"val_sets[{i}].include_condensed is required for dynamic mode")
            elif vs.data_mode == "pre_rendered":
                if not vs.pre_rendered_image_dir:
                    raise ValueError(f"val_sets[{i}].pre_rendered_image_dir is required for pre_rendered mode")
                if not os.path.isdir(vs.pre_rendered_image_dir):
                    raise ValueError(f"val_sets[{i}].pre_rendered_image_dir not found: {vs.pre_rendered_image_dir}")
            else:
                if not vs.realistic_image_root:
                    raise ValueError(f"val_sets[{i}].realistic_image_root is required for realistic mode")
                if not os.path.isdir(vs.realistic_image_root):
                    raise ValueError(f"val_sets[{i}].realistic_image_root not found: {vs.realistic_image_root}")
    if cfg.load_in_4bit:
        raise ValueError("Full fine-tuning does not support load_in_4bit=true. Set load_in_4bit=false.")


def load_config(config_path: str) -> FullSFTConfig:
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
    raw["output_dir"] = _resolve_path(raw["output_dir"], config_file.parent)
    raw["enable_val_sets"] = bool(raw.get("enable_val_sets", False))
    raw["eval_every_steps"] = raw.get("eval_every_steps", None)
    raw["eval_on_start"] = bool(raw.get("eval_on_start", True))
    raw["eval_on_end"] = bool(raw.get("eval_on_end", True))
    raw["optim"] = raw.get("optim", "adamw_torch_fused")
    raw["allow_tf32"] = bool(raw.get("allow_tf32", True))
    raw["enable_gradient_checkpointing"] = bool(raw.get("enable_gradient_checkpointing", False))
    raw["save_steps"] = raw.get("save_steps", None)
    raw["dataloader_prefetch_factor"] = int(raw.get("dataloader_prefetch_factor", 4))
    raw["dataloader_persistent_workers"] = bool(raw.get("dataloader_persistent_workers", True))
    raw["ddp_find_unused_parameters"] = bool(raw.get("ddp_find_unused_parameters", False))
    raw["ddp_backend"] = raw.get("ddp_backend", "nccl")
    raw["ddp_timeout_seconds"] = int(raw.get("ddp_timeout_seconds", 1800))
    raw["use_accelerate"] = bool(raw.get("use_accelerate", False))
    raw["accelerate_num_processes"] = int(raw.get("accelerate_num_processes", 1))
    raw["accelerate_gpu_ids"] = raw.get("accelerate_gpu_ids", None)
    restart_raw = dict(raw.get("restart", {}) or {})
    if "enable" not in restart_raw:
        restart_raw["enable"] = raw.get("restart", True)
    if "dir" not in restart_raw:
        restart_raw["dir"] = raw.get("output_dir")
    if "auto_resume_from_latest_checkpoint" not in restart_raw:
        restart_raw["auto_resume_from_latest_checkpoint"] = raw.get("auto_resume_from_latest_checkpoint", True)
    if "resume_from_checkpoint" not in restart_raw:
        restart_raw["resume_from_checkpoint"] = raw.get("resume_from_checkpoint", None)
    if "wandb_resume_same_run" not in restart_raw:
        restart_raw["wandb_resume_same_run"] = raw.get("wandb_resume_same_run", True)
    restart_raw["enable"] = bool(restart_raw.get("enable", True))
    restart_raw["dir"] = _resolve_path(restart_raw["dir"], config_file.parent)
    restart_raw["auto_resume_from_latest_checkpoint"] = bool(restart_raw.get("auto_resume_from_latest_checkpoint", True))
    restart_raw["resume_from_checkpoint"] = restart_raw.get("resume_from_checkpoint", None)
    if restart_raw["resume_from_checkpoint"] not in (None, ""):
        restart_raw["resume_from_checkpoint"] = _resolve_path(restart_raw["resume_from_checkpoint"], config_file.parent)
    restart_raw["wandb_resume_same_run"] = bool(restart_raw.get("wandb_resume_same_run", True))
    raw["restart"] = RestartConfig(**restart_raw)

    if "train_sets" not in raw:
        train_set = {
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
        raw["train_sets"] = [train_set]
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

    parsed_val_sets: List[ValSetConfig] = []
    for idx, item in enumerate(raw.get("val_sets", []) or []):
        item = dict(item)
        if "val_csv" not in item and "train_csv" in item:
            item["val_csv"] = item["train_csv"]
        item["val_csv"] = _resolve_path(item["val_csv"], config_file.parent)
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
        item.setdefault("name", f"val{idx}")
        parsed_val_sets.append(ValSetConfig(**item))
    raw["val_sets"] = parsed_val_sets
    if raw.get("wandb_api_key") in ("", None):
        raw["wandb_api_key"] = os.getenv("WANDB_API_KEY", None)

    for legacy_key in [
        "train_csv",
        "data_mode",
        "pre_rendered_image_dir",
        "realistic_image_root",
        "image_ext",
        "instruction",
        "style",
        "mol_augment",
        "include_condensed",
        "max_samples",
        "sample_num",
        "val_csv",
        "save_total_limit",
        "auto_resume_from_latest_checkpoint",
        "resume_from_checkpoint",
        "accelerate_max_restarts",
        "wandb_resume_same_run",
        "restart_strict",
    ]:
        raw.pop(legacy_key, None)
    cfg = FullSFTConfig(**raw)
    _validate_config(cfg)
    return cfg


def parse_cli_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Full fine-tune DeepSeek-OCR-2 with YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(workspace / "full_sft_config.yaml"),
        help="Path to full sft YAML config.",
    )
    args, _ = parser.parse_known_args()
    return args


def _is_distributed_launched() -> bool:
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("RANK") is not None:
        return True
    if os.environ.get("ACCELERATE_PROCESS_INDEX") is not None:
        return True
    return False


def _maybe_launch_with_accelerate(cfg: FullSFTConfig, cli_args: argparse.Namespace) -> None:
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


def _print_trainable_parameters(model) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")


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

    model_root = Path(cfg.pretrained_weight_path).resolve()
    model_path = resolve_local_model_path(model_root)
    os.makedirs(cfg.output_dir, exist_ok=True)
    checkpoint_dir = cfg.restart.dir if cfg.restart.enable else cfg.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    maybe_init_wandb(cfg)

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
    text_encode, basic_image_transform_cls, dynamic_preprocess_fn = load_modeling_utils_from_loaded_model(model)
    model.config.use_cache = not cfg.enable_gradient_checkpointing
    if cfg.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _print_trainable_parameters(model)

    train_datasets: List[Dataset] = []
    for ts in cfg.train_sets:
        import pandas as pd

        train_df = pd.read_csv(ts.train_csv)
        train_df = _apply_subset_sampling(
            train_df,
            sample_num=ts.sample_num,
            max_samples=ts.max_samples,
            data_mode=ts.data_mode,
            seed=cfg.seed,
        )
        dataset_item = _build_conversation_dataset(
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
        train_datasets.append(dataset_item)
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else torch.utils.data.ConcatDataset(train_datasets)

    val_named_datasets: List[tuple[str, Dataset]] = []
    if cfg.enable_val_sets and cfg.val_sets:
        import pandas as pd

        for i, vs in enumerate(cfg.val_sets):
            val_df = pd.read_csv(vs.val_csv)
            val_df = _apply_subset_sampling(
                val_df,
                sample_num=vs.sample_num,
                max_samples=vs.max_samples,
                data_mode=vs.data_mode,
                seed=cfg.seed,
            )
            val_dataset = _build_conversation_dataset(
                dataframe=val_df,
                data_mode=vs.data_mode,
                instruction=vs.instruction,
                style=vs.style,
                mol_augment=vs.mol_augment,
                include_condensed=vs.include_condensed,
                pre_rendered_image_dir=vs.pre_rendered_image_dir,
                realistic_image_root=vs.realistic_image_root,
                use_rendered_smiles_as_label=False,
            )
            val_name = (vs.name or f"val{i}").strip() or f"val{i}"
            val_named_datasets.append((val_name, val_dataset))

    data_collator = DeepSeekOCR2DataCollator(
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

    report_to = ["wandb"] if (cfg.wandb and is_main_process()) else ["none"]
    use_epoch_mode = cfg.epochs > 0
    max_steps = -1 if use_epoch_mode else cfg.max_steps
    num_train_epochs = cfg.epochs if use_epoch_mode else 1.0
    ddp_find_unused_parameters = cfg.ddp_find_unused_parameters
    if _is_distributed_launched() and not ddp_find_unused_parameters:
        ddp_find_unused_parameters = True
    dataloader_prefetch_factor = cfg.dataloader_prefetch_factor if cfg.dataloader_num_workers > 0 else None
    dataloader_persistent_workers = cfg.dataloader_persistent_workers and cfg.dataloader_num_workers > 0
    save_steps = cfg.save_steps if cfg.save_steps is not None else max(10, cfg.max_steps // 6 if cfg.max_steps > 0 else 100)
    resume_checkpoint = _resolve_resume_checkpoint(cfg)
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_steps=cfg.warmup_steps,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        seed=cfg.seed,
        fp16=not bf16_supported(),
        bf16=bf16_supported(),
        output_dir=checkpoint_dir,
        report_to=report_to,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_backend=cfg.ddp_backend,
        ddp_timeout=cfg.ddp_timeout_seconds,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="no",
    )
    trainer = CleanEvalMetricsTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        args=training_args,
    )
    if val_named_datasets and cfg.eval_every_steps:
        periodic_eval_callback = PeriodicMultiValEvalCallback(
            eval_sets=val_named_datasets,
            eval_every_steps=cfg.eval_every_steps,
        )
        periodic_eval_callback.trainer = trainer
        trainer.add_callback(periodic_eval_callback)
    if val_named_datasets and cfg.eval_on_start:
        for val_name, val_dataset in val_named_datasets:
            trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix=f"eval_{val_name}_step0")
    if resume_checkpoint:
        with _torch_load_resume_compat():
            trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    if val_named_datasets and cfg.eval_on_end:
        for val_name, val_dataset in val_named_datasets:
            trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix=f"eval_{val_name}_final")
    if trainer.is_world_process_zero():
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved full fine-tuned weights to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
