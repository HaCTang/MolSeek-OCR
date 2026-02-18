import contextlib
import importlib
import io
import inspect
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback

from dataset import ChemConversationDataset, DEFAULT_INSTRUCTION, make_style_config

def apply_transformers_compat_shims() -> None:
    """
    DeepSeek-OCR2 dynamic module expects `LlamaFlashAttention2` symbol.
    Some transformers versions (e.g. 4.57.x) removed/renamed it.
    Provide a safe fallback alias so loading can proceed.
    """
    try:
        import transformers.models.llama.modeling_llama as llama_modeling
    except Exception:
        return

    if not hasattr(llama_modeling, "LlamaFlashAttention2") and hasattr(llama_modeling, "LlamaAttention"):
        llama_modeling.LlamaFlashAttention2 = llama_modeling.LlamaAttention


def apply_deepseek_runtime_patches(model) -> None:
    """
    Patch DeepSeek-OCR2 forward to avoid in-place op on a leaf view:
    modeling_deepseekocr2.py does masked_scatter_ on inputs_embeds[idx].
    We ensure inputs_embeds is non-leaf by passing (embeds + 0.0).
    """
    model_cls = model.__class__
    if getattr(model_cls, "_chemseek_nonleaf_inputs_patch", False):
        return

    orig_forward = model_cls.forward
    allowed_kwargs = set(inspect.signature(orig_forward).parameters.keys())

    def patched_forward(self, *args, **kwargs):
        # Some Trainer integrations may pass helper kwargs unsupported by remote model forward.
        kwargs.pop("num_items_in_batch", None)

        if kwargs.get("inputs_embeds") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None:
                embeds = self.get_model().get_input_embeddings()(input_ids)
                kwargs["inputs_embeds"] = embeds + 0.0

        # Keep only kwargs that target forward actually accepts.
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

        # Silence noisy debug prints from remote model code (e.g. BASE/PATCHES shapes).
        with contextlib.redirect_stdout(io.StringIO()):
            return orig_forward(self, *args, **filtered_kwargs)

    model_cls.forward = patched_forward
    model_cls._chemseek_nonleaf_inputs_patch = True


def bf16_supported() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


SMILES_CANDIDATE_COLUMNS = ("SMILES", "smiles", "canonical_smiles")
ID_CANDIDATE_COLUMNS = ("Unnamed: 0", "id", "idx", "index", "image_id", "pubchem_cid")


def _find_smiles_column(df: pd.DataFrame) -> str:
    for column in SMILES_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No SMILES column found. Tried {SMILES_CANDIDATE_COLUMNS}, got {list(df.columns)}")


def _sanitize_filename(text: str) -> str:
    safe_chars: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "sample"


def _resolve_sample_id(row: pd.Series, fallback_idx: int) -> str:
    for col in ID_CANDIDATE_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, float) and val.is_integer():
                raw = str(int(val))
            else:
                raw = str(val).strip()
            if raw:
                return _sanitize_filename(raw)
    return str(fallback_idx)


def _resolve_with_alt_extensions(path: str) -> str | None:
    if os.path.isfile(path):
        return path
    stem, ext = os.path.splitext(path)
    ext_lower = ext.lower()
    alt_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]
    for candidate_ext in alt_exts:
        if candidate_ext == ext_lower:
            continue
        candidate_path = f"{stem}{candidate_ext}"
        if os.path.isfile(candidate_path):
            return candidate_path
        candidate_upper = f"{stem}{candidate_ext.upper()}"
        if os.path.isfile(candidate_upper):
            return candidate_upper
    return None


class PreRenderedChemConversationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        instruction: str = DEFAULT_INSTRUCTION,
        max_resample_attempts: int = 20,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_dir = image_dir
        self.instruction = instruction
        self.smiles_col = _find_smiles_column(self.df)
        self.max_resample_attempts = max(1, int(max_resample_attempts))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        last_error: str | None = None
        current_idx = idx
        for _ in range(self.max_resample_attempts):
            row = self.df.iloc[current_idx]
            sample_id = _resolve_sample_id(row, current_idx)
            raw_path = os.path.join(self.image_dir, f"{sample_id}.png")
            image_path = _resolve_with_alt_extensions(raw_path)
            if image_path is None:
                last_error = f"Pre-rendered image not found: {raw_path}"
                current_idx = random.randrange(len(self.df))
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                last_error = f"Failed to open pre-rendered image: {image_path} ({exc})"
                current_idx = random.randrange(len(self.df))
                continue
            target_smiles = str(row[self.smiles_col])
            return {
                "messages": [
                    {"role": "<|User|>", "content": self.instruction, "images": [image]},
                    {"role": "<|Assistant|>", "content": target_smiles},
                ],
                "meta": {"idx": current_idx, "sample_id": sample_id, "image_path": image_path},
            }
        raise RuntimeError(last_error or "Failed to fetch valid pre-rendered sample")


class RealisticChemConversationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: str,
        instruction: str = DEFAULT_INSTRUCTION,
        max_resample_attempts: int = 20,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = image_root
        self.instruction = instruction
        self.smiles_col = _find_smiles_column(self.df)
        self.max_resample_attempts = max(1, int(max_resample_attempts))
        if "file_path" not in self.df.columns:
            raise ValueError("realistic mode requires `file_path` column in train_csv")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        last_error: str | None = None
        current_idx = idx
        for _ in range(self.max_resample_attempts):
            row = self.df.iloc[current_idx]
            rel_path = str(row["file_path"]).strip()
            raw_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.image_root, rel_path)
            image_path = _resolve_with_alt_extensions(raw_path)
            if image_path is None:
                last_error = f"Realistic image not found: {raw_path}"
                current_idx = random.randrange(len(self.df))
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                last_error = f"Failed to open realistic image: {image_path} ({exc})"
                current_idx = random.randrange(len(self.df))
                continue
            target_smiles = str(row[self.smiles_col])
            return {
                "messages": [
                    {"role": "<|User|>", "content": self.instruction, "images": [image]},
                    {"role": "<|Assistant|>", "content": target_smiles},
                ],
                "meta": {"idx": current_idx, "file_path": rel_path, "image_path": image_path},
            }
        raise RuntimeError(last_error or "Failed to fetch valid realistic sample")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_local_model_path(model_root: Path) -> Path:
    if (model_root / "config.json").is_file():
        return model_root

    snapshot_dirs = sorted(model_root.glob("**/snapshots/*"))
    for path in reversed(snapshot_dirs):
        if (path / "config.json").is_file():
            return path

    raise FileNotFoundError(
        f"Cannot find local model snapshot under: {model_root}. "
        "Expected either config.json in root or huggingface snapshots."
    )


def load_modeling_utils_from_loaded_model(model) -> tuple[Any, Any, Any]:
    module_name = model.__class__.__module__
    module = importlib.import_module(module_name)
    required = ("text_encode", "BasicImageTransform", "dynamic_preprocess")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Module `{module_name}` missing required symbols: {missing}. "
            "Cannot build DeepSeekOCR2DataCollator."
        )
    return module.text_encode, module.BasicImageTransform, module.dynamic_preprocess


@dataclass
class DeepSeekOCR2DataCollator:
    tokenizer: Any
    model: Any
    text_encode: Any
    basic_image_transform_cls: Any
    dynamic_preprocess_fn: Any
    image_size: int = 768
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __post_init__(self):
        self.dtype = self.model.dtype
        self.image_transform = self.basic_image_transform_cls(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True,
        )
        self.patch_size = 16
        self.downsample_ratio = 4
        self.bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0

    def deserialize_image(self, image_data) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(io.BytesIO(image_data["bytes"]))
            return image.convert("RGB")
        raise ValueError(f"Unsupported image format: {type(image_data)}")

    def process_image(self, image: Image.Image):
        images_list, images_crop_list, images_spatial_crop = [], [], []

        if self.crop_mode:
            if image.size[0] <= 768 and image.size[1] <= 768:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = self.dynamic_preprocess_fn(
                    image, min_num=2, max_num=6, image_size=self.image_size, use_thumbnail=False
                )

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))

            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries_base) * num_queries_base
            tokenized_image += [self.image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num)) * (
                    num_queries * height_crop_num
                )
        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])
            if self.base_size <= 768:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image,
                    (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))
            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_data in message["images"]:
                    if img_data is not None:
                        images.append(self.deserialize_image(img_data))
        if not images:
            raise ValueError("No images found in sample")

        tokenized_str = [self.bos_id]
        images_seq_mask = [False]
        images_list, images_crop_list, images_spatial_crop = [], [], []
        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            text_splits = content.split("<image>")
            for i, text_sep in enumerate(text_splits):
                tokenized_sep = self.text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                if i < len(text_splits) - 1:
                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img = self.process_image(image)
                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)
                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))
                    image_idx += 1

        if not assistant_started:
            prompt_token_count = len(tokenized_str)

        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []
        for feature in features:
            try:
                batch_data.append(self.process_single_sample(feature["messages"]))
            except Exception:
                continue
        if not batch_data:
            raise ValueError("No valid samples in batch")

        input_ids_list = [item["input_ids"] for item in batch_data]
        images_seq_mask_list = [item["images_seq_mask"] for item in batch_data]
        prompt_token_counts = [item["prompt_token_count"] for item in batch_data]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[images_seq_mask] = -100
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        images_batch = [(item["images_crop"], item["images_ori"]) for item in batch_data]
        images_spatial_crop = torch.cat([item["images_spatial_crop"] for item in batch_data], dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }


class PeriodicMultiValEvalCallback(TrainerCallback):
    def __init__(self, eval_sets: List[tuple[str, Dataset]], eval_every_steps: int):
        self.eval_sets = eval_sets
        self.eval_every_steps = eval_every_steps
        self.trainer: Trainer | None = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        step = int(state.global_step)
        max_steps = int(state.max_steps) if state.max_steps is not None else -1
        # Evaluate only at intermediate checkpoints, excluding step 0 and final step.
        if step <= 0:
            return control
        if max_steps > 0 and step >= max_steps:
            return control
        if step % self.eval_every_steps != 0:
            return control

        for name, eval_dataset in self.eval_sets:
            self.trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=f"eval_{name}")
        return control


def _is_eval_timing_metric(metric_name: str) -> bool:
    if not metric_name.startswith("eval_"):
        return False
    timing_suffixes = (
        "_runtime",
        "_samples_per_second",
        "_steps_per_second",
        "_model_preparation_time",
    )
    return metric_name.endswith(timing_suffixes)


class CleanEvalMetricsTrainer(Trainer):
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:  # type: ignore[override]
        cleaned_logs = {k: v for k, v in logs.items() if not _is_eval_timing_metric(k)}
        if cleaned_logs:
            super().log(cleaned_logs, *args, **kwargs)

def _resolve_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _apply_subset_sampling(
    df: pd.DataFrame,
    *,
    sample_num: int | None,
    max_samples: int | None,
    data_mode: str,
    seed: int,
) -> pd.DataFrame:
    sampled = df
    if sample_num is not None:
        if sample_num <= 0:
            raise ValueError("sample_num must be > 0")
        if sample_num < len(sampled):
            sampled = sampled.sample(n=sample_num, random_state=seed).reset_index(drop=True)
    elif data_mode == "dynamic" and max_samples is not None:
        sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sampled = sampled.iloc[: max_samples].reset_index(drop=True)
    return sampled


def _build_conversation_dataset(
    dataframe: pd.DataFrame,
    *,
    data_mode: str,
    instruction: str,
    style: str | None,
    mol_augment: bool | None,
    include_condensed: bool | None,
    pre_rendered_image_dir: str | None,
    realistic_image_root: str | None,
    use_rendered_smiles_as_label: bool,
) -> Dataset:
    if data_mode == "dynamic":
        style_cfg = make_style_config(
            style=style,
            mol_augment=mol_augment,
            include_condensed=include_condensed,
        )
        return ChemConversationDataset(
            dataframe=dataframe,
            instruction=instruction,
            style_config=style_cfg,
            use_rendered_smiles_as_label=use_rendered_smiles_as_label,
        )
    if data_mode == "pre_rendered":
        return PreRenderedChemConversationDataset(
            dataframe=dataframe,
            image_dir=pre_rendered_image_dir,  # validated above
            instruction=instruction,
        )
    return RealisticChemConversationDataset(
        dataframe=dataframe,
        image_root=realistic_image_root,  # validated above
        instruction=instruction,
    )


def _find_latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
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


def _resolve_resume_checkpoint(cfg: Any) -> str | None:
    if not cfg.restart.enable:
        return None
    if cfg.restart.resume_from_checkpoint not in (None, ""):
        return cfg.restart.resume_from_checkpoint
    if cfg.restart.auto_resume_from_latest_checkpoint:
        return _find_latest_checkpoint(cfg.restart.dir)
    return None


@contextlib.contextmanager
def _torch_load_resume_compat():
    """
    PyTorch 2.6 changed torch.load default to weights_only=True.
    HF Trainer resume reads rng_state/checkpoint metadata that may contain
    non-tensor objects, so we force weights_only=False when not explicitly set.
    """
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]


def maybe_init_wandb(cfg: Any) -> None:
    if not cfg.wandb:
        return
    if not is_main_process():
        return

    if cfg.wandb_api_key:
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

    import wandb  # type: ignore[import-not-found]

    run_state_dir = cfg.restart.dir if cfg.restart.enable else cfg.output_dir
    run_id_file = Path(run_state_dir) / ".wandb_run_id"
    run_id: str | None = None
    if cfg.restart.enable and cfg.restart.wandb_resume_same_run and run_id_file.is_file():
        saved = run_id_file.read_text(encoding="utf-8").strip()
        if saved:
            run_id = saved

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        id=run_id,
        resume="allow" if (cfg.restart.enable and cfg.restart.wandb_resume_same_run) else None,
        config={k: v for k, v in vars(cfg).items()},
    )
    if cfg.restart.enable and cfg.restart.wandb_resume_same_run and run is not None:
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(str(run.id), encoding="utf-8")


def is_main_process() -> bool:
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    return rank in (None, "", "0") and local_rank in (None, "", "0")


