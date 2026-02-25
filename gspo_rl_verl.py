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
        self.image_patch_size = int(getattr(config, "image_patch_size", 14))
        self.max_prompt_length = int(getattr(config, "max_prompt_length", 4096))
        self.truncation = str(getattr(config, "truncation", "error"))
        self.return_raw_chat = bool(getattr(config, "return_raw_chat", True))
        self.apply_chat_template_kwargs = dict(getattr(config, "apply_chat_template_kwargs", {}))
        print(f"[ChemSeekOCRDataset] loaded {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def _build_deepseek_mm_payload(self, image):
        """Build DeepSeek-OCR2-vllm expected multimodal image payload."""
        if not hasattr(self, "_deepseek_mm_processor"):
            self._deepseek_mm_processor = None
        if self._deepseek_mm_processor is None:
            from process.image_process import DeepseekOCR2Processor
            self._deepseek_mm_processor = DeepseekOCR2Processor()
        return self._deepseek_mm_processor.tokenize_with_images(
            images=[image.convert("RGB")],
            bos=True,
            eos=True,
            cropping=True,
        )

    @staticmethod
    def _messages_to_plain_prompt(messages: list) -> str:
        """Fallback formatter when tokenizer has no chat_template."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    t = item.get("type")
                    if t == "image":
                        parts.append("<image>")
                    elif t == "text":
                        parts.append(str(item.get("text", "")))
            else:
                parts.append(str(content))
        prompt = "".join(parts).strip()
        return prompt or "<image>\n Give me the SMILES of the molecule. "

    def __getitem__(self, idx: int) -> dict:
        import re
        from PIL import Image
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask

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
                parts = [seg for seg in re.split(r"(<image>)", content) if seg]
                content_list = []
                for part in parts:
                    if part == "<image>":
                        content_list.append({"type": "image"})
                    elif part:
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

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            model_inputs = self.processor(
                text=[raw_prompt],
                images=[image.convert("RGB")],
                return_tensors="pt",
            )
        else:
            raw_prompt = self._messages_to_plain_prompt(messages)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        multi_modal_data = {"image": self._build_deepseek_mm_payload(image)}

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": multi_modal_data,
            "data_source": row.get("data_source", "chemseek_ocr"),
            "reward_model": reward_model,
            "extra_info": row.get("extra_info", {}) or {},
            "index": idx,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            **({"raw_prompt": messages} if self.return_raw_chat else {}),
        }

    def resume_dataset_state(self):
        pass


# =========================================================================
# Training Launch (run in chemseek-ocr-verl conda env)
# =========================================================================

_COMPAT_SENTINEL = "LlamaFlashAttention2"
_COMPAT_PATCH = "\n# Backward compat alias (transformers >= 4.46)\nLlamaFlashAttention2 = LlamaAttention\n"
_COMPAT_V2_SENTINEL = "class LlamaFlashAttention2Compat(LlamaAttention):"
_COMPAT_V2_PATCH = """
# Backward compat wrapper for DeepSeekV2-style attention calls.
class LlamaFlashAttention2Compat(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            if position_ids is None:
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

LlamaFlashAttention2 = LlamaFlashAttention2Compat
"""
_OCR2_TEXT_CONFIG_OLD = "self.text_config = config.text_config"
_OCR2_TEXT_CONFIG_NEW = (
    "self.text_config = getattr(config, \"text_config\", None)\n"
    "        if self.text_config is None and hasattr(config, \"get_text_config\"):\n"
    "            self.text_config = config.get_text_config()\n"
    "        if self.text_config is None:\n"
    "            raise AttributeError(\"DeepseekOCR2Config has neither text_config nor get_text_config\")"
)
_HF_OCR2_IMAGES_GUARD_OLD = (
    "if sam_model is not None and (input_ids.shape[1] != 1 or self.training) "
    "and torch.sum(images[0][1]).item() != 0:"
)
_HF_OCR2_IMAGES_GUARD_NEW = (
    "if sam_model is not None and images is not None and len(images) > 0 "
    "and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:"
)
_HF_V2_ATTN_UNPACK_OLD = """        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )"""
_HF_V2_ATTN_UNPACK_NEW = """        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if isinstance(attn_outputs, tuple):
            if len(attn_outputs) == 3:
                hidden_states, self_attn_weights, present_key_value = attn_outputs
            elif len(attn_outputs) == 2:
                hidden_states, self_attn_weights = attn_outputs
                present_key_value = None
            else:
                raise ValueError(f\"Unexpected attention outputs length: {len(attn_outputs)}\")
        else:
            raise TypeError(f\"Unexpected attention output type: {type(attn_outputs)}\")"""
_TARGET_VLLM_VERSION = "0.8.5"
_TARGET_TRANSFORMERS_VERSION = "4.57.0"


def _assert_target_runtime_versions() -> None:
    """Hard-check the runtime stack for the chemseek-ocr-verl environment."""
    try:
        import vllm
        import transformers
    except Exception as exc:
        print(f"[env-check] ERROR: failed to import runtime deps: {exc}")
        sys.exit(1)

    vllm_ver = str(getattr(vllm, "__version__", "unknown"))
    tf_ver = str(getattr(transformers, "__version__", "unknown"))
    if vllm_ver != _TARGET_VLLM_VERSION or tf_ver != _TARGET_TRANSFORMERS_VERSION:
        print("[env-check] ERROR: version mismatch for chemseek-ocr-verl")
        print(f"  expected: vllm=={_TARGET_VLLM_VERSION}, transformers=={_TARGET_TRANSFORMERS_VERSION}")
        print(f"  current:  vllm=={vllm_ver}, transformers=={tf_ver}")
        print("Please align the environment, then rerun.")
        sys.exit(1)


def _ensure_transformers_compat() -> None:
    """Patch transformers so ``from ... import LlamaFlashAttention2`` works.

    Newer transformers removed this class; DeepSeek-OCR2 modeling code still
    imports it.  We append a one-line alias to the installed source file so
    that *every* process (including Ray workers) picks it up automatically.
    """
    try:
        import transformers.models.llama.modeling_llama as llama_mod
        src_path = llama_mod.__file__
        if src_path is None:
            return
        with open(src_path, "r") as f:
            content = f.read()
        patches = []
        if _COMPAT_SENTINEL not in content:
            patches.append(_COMPAT_PATCH)
        if _COMPAT_V2_SENTINEL not in content:
            patches.append("\n" + _COMPAT_V2_PATCH.strip() + "\n")
        if not patches:
            return
        with open(src_path, "a") as f:
            f.write("".join(patches))
        print(f"[compat] Patched {src_path}: upgraded llama attention compatibility")
    except Exception as exc:
        print(f"[compat] WARNING: could not patch transformers: {exc}")


def _ensure_deepseek_ocr2_vllm_compat(vllm_code_dir: Optional[str]) -> None:
    """Patch DeepSeek-OCR2-vllm for transformers>=4.57 config API changes."""
    if not vllm_code_dir:
        return

    src_path = Path(vllm_code_dir) / "deepseek_ocr2.py"
    if not src_path.is_file():
        return

    try:
        content = src_path.read_text(encoding="utf-8")
        if _OCR2_TEXT_CONFIG_NEW in content:
            return
        if _OCR2_TEXT_CONFIG_OLD not in content:
            print(
                f"[compat] WARNING: expected snippet not found in {src_path}; "
                "skip DeepSeek-OCR2-vllm text_config patch"
            )
            return
        content = content.replace(_OCR2_TEXT_CONFIG_OLD, _OCR2_TEXT_CONFIG_NEW, 1)
        src_path.write_text(content, encoding="utf-8")
        print(f"[compat] Patched {src_path}: text_config -> get_text_config fallback")
    except Exception as exc:
        print(f"[compat] WARNING: could not patch {src_path}: {exc}")


def _ensure_local_checkpoint_compat(model_path: str) -> None:
    """Patch local HF checkpoint modeling code for transformers>=4.57."""
    model_dir = Path(model_path)
    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
        / model_dir.name.replace("-", "_hyphen_")
    )
    ocr2_candidates = [model_dir / "modeling_deepseekocr2.py", cache_dir / "modeling_deepseekocr2.py"]
    v2_candidates = [model_dir / "modeling_deepseekv2.py", cache_dir / "modeling_deepseekv2.py"]

    for src_path in ocr2_candidates:
        if not src_path.is_file():
            continue
        try:
            content = src_path.read_text(encoding="utf-8")
            if _HF_OCR2_IMAGES_GUARD_NEW in content:
                continue
            if _HF_OCR2_IMAGES_GUARD_OLD not in content:
                print(
                    f"[compat] WARNING: expected image guard snippet not found in {src_path}; "
                    "skip ocr2 image guard patch"
                )
                continue
            content = content.replace(_HF_OCR2_IMAGES_GUARD_OLD, _HF_OCR2_IMAGES_GUARD_NEW, 1)
            src_path.write_text(content, encoding="utf-8")
            print(f"[compat] Patched {src_path}: added images=None guard")
        except Exception as exc:
            print(f"[compat] WARNING: could not patch {src_path}: {exc}")

    for src_path in v2_candidates:
        if not src_path.is_file():
            continue
        try:
            content = src_path.read_text(encoding="utf-8")
            if _HF_V2_ATTN_UNPACK_NEW in content:
                continue
            if _HF_V2_ATTN_UNPACK_OLD not in content:
                print(
                    f"[compat] WARNING: expected attention unpack snippet not found in {src_path}; "
                    "skip deepseekv2 patch"
                )
                continue
            content = content.replace(_HF_V2_ATTN_UNPACK_OLD, _HF_V2_ATTN_UNPACK_NEW, 1)
            src_path.write_text(content, encoding="utf-8")
            print(f"[compat] Patched {src_path}: attention unpack 2/3 return compatible")
        except Exception as exc:
            print(f"[compat] WARNING: could not patch {src_path}: {exc}")


def train(cfg: dict, config_path: str) -> None:
    """Build verl GSPO config overrides and launch training."""

    _assert_target_runtime_versions()
    _ensure_transformers_compat()

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    gspo_cfg = cfg.get("gspo", {})
    gpu_cfg = cfg.get("gpu", {})
    rw_cfg = cfg.get("reward_weights", {})
    wandb_cfg = cfg.get("wandb", {})
    output_dir = cfg.get("output_dir", str(SCRIPT_DIR / "weight_gspo_rl_verl"))
    _ensure_local_checkpoint_compat(str(model_cfg.get("path", "")))

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
    vllm_architecture = str(model_cfg.get("vllm_architecture", "DeepseekOCR2ForCausalLM"))
    vllm_code_dir = model_cfg.get("vllm_code_dir")
    if vllm_code_dir:
        vllm_code_dir = _resolve_path(vllm_code_dir, Path(config_path).resolve().parent)
    else:
        fallback_vllm_dir = (
            SCRIPT_DIR.parent
            / "DeepSeek-OCR-2"
            / "DeepSeek-OCR2-master"
            / "DeepSeek-OCR2-vllm"
        )
        if fallback_vllm_dir.is_dir():
            vllm_code_dir = str(fallback_vllm_dir.resolve())
    _ensure_deepseek_ocr2_vllm_compat(vllm_code_dir)

    train_batch = train_cfg.get("train_batch_size", n_gpus * 4)
    micro_bs = train_cfg.get("ppo_micro_batch_size_per_gpu", 1)
    mini_bs = train_cfg.get("ppo_mini_batch_size", train_batch)

    group_size = gspo_cfg.get("group_size", 8)

    # Memory-safe defaults for 48GB GPUs with DeepSeek-OCR2 full-parameter RL.
    if train_cfg.get("memory_safe_mode", True):
        if micro_bs > 1:
            print(f"[memory-safe] ppo_micro_batch_size_per_gpu {micro_bs} -> 1")
            micro_bs = 1
        if train_batch > 8:
            print(f"[memory-safe] train_batch_size {train_batch} -> 8")
            train_batch = 8
        if mini_bs > train_batch:
            print(f"[memory-safe] ppo_mini_batch_size {mini_bs} -> {train_batch}")
            mini_bs = train_batch

    # verl normalizes: effective = mini_bs * group_size // n_gpus
    # it must be divisible by micro_bs
    from math import gcd
    unit = n_gpus * micro_bs
    step_mini = unit // gcd(group_size, unit)
    if mini_bs % step_mini != 0:
        orig = mini_bs
        mini_bs = max(step_mini, (mini_bs // step_mini) * step_mini)
        print(f"[auto-fix] ppo_mini_batch_size {orig} -> {mini_bs} "
              f"(must be a multiple of {step_mini} for {n_gpus} GPUs, "
              f"group_size={group_size}, micro_bs={micro_bs})")
    step_tb = n_gpus // gcd(group_size, n_gpus)
    if train_batch % step_tb != 0:
        orig = train_batch
        train_batch = max(step_tb, (train_batch // step_tb) * step_tb)
        print(f"[auto-fix] train_batch_size {orig} -> {train_batch} "
              f"(must be a multiple of {step_tb} for {n_gpus} GPUs, "
              f"group_size={group_size})")
    if mini_bs > train_batch:
        mini_bs = train_batch
        print(f"[auto-fix] ppo_mini_batch_size clamped to train_batch_size={train_batch}")
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
    if train_cfg.get("memory_safe_mode", True) and max_prompt_len > 2048:
        print(f"[memory-safe] max_prompt_length {max_prompt_len} -> 2048")
        max_prompt_len = 2048

    token_multiplier = float(train_cfg.get("actor_token_len_multiplier", 1.0))
    actor_max_token_len_per_gpu = int((max_prompt_len + max_resp_len) * token_multiplier)

    param_offload = str(gpu_cfg.get("param_offload", False))
    optim_offload = str(gpu_cfg.get("optimizer_offload", False))
    ref_offload = str(gpu_cfg.get("ref_param_offload", True))

    logger_list = '["console"]'
    if wandb_cfg.get("enabled", False):
        logger_list = '["console","wandb"]'

    cmd = [
        sys.executable, str(SCRIPT_DIR / "verl_main_ppo_compat.py"),
        # Algorithm: GSPO uses GRPO advantage estimator with gspo loss mode
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        # Model
        f"actor_rollout_ref.model.path={model_cfg['path']}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.external_lib=vllm_deepseekocr2_patch",
        # Actor – GSPO-specific parameters
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.optim.weight_decay={train_cfg.get('weight_decay', 0.1)}",
        f"actor_rollout_ref.actor.optim.clip_grad={train_cfg.get('clip_grad', 1.0)}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
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
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.architectures=[\"{vllm_architecture}\"]",
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
        # Reward (verl>=0.6.1): use reward_model + custom_reward_function
        "reward_model.reward_manager=naive",
        f"custom_reward_function.path={script_path}",
        "custom_reward_function.name=compute_score",
        f"+custom_reward_function.reward_kwargs.reward_weights.validity={rw_cfg.get('validity', 2.0)}",
        f"+custom_reward_function.reward_kwargs.reward_weights.tanimoto={rw_cfg.get('tanimoto', 1.0)}",
        f"+custom_reward_function.reward_kwargs.reward_weights.canon_smiles={rw_cfg.get('canon_smiles', 2.0)}",
        f"+custom_reward_function.reward_kwargs.reward_weights.graph={rw_cfg.get('graph', 1.5)}",
        f"+custom_reward_function.reward_kwargs.reward_weights.chiral={rw_cfg.get('chiral', 1.5)}",
        f"+custom_reward_function.reward_kwargs.chiral_no_annotation_reward={cfg.get('chiral_no_annotation_reward', 1.0)}",
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
    pythonpath_parts = [str(SCRIPT_DIR)]
    if vllm_code_dir:
        pythonpath_parts.append(str(vllm_code_dir))
    old_pythonpath = env.get("PYTHONPATH")
    if old_pythonpath:
        pythonpath_parts.append(old_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    if train_cfg.get("allow_tf32", True):
        env.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    # vLLM's CuMemAllocator is incompatible with expandable_segments.
    if "PYTORCH_CUDA_ALLOC_CONF" in env and "expandable_segments:True" in env["PYTORCH_CUDA_ALLOC_CONF"]:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("VLLM_USE_V1", "0")
    if vllm_code_dir:
        env["CHEMSEEK_VLLM_CODE_DIR"] = vllm_code_dir
    env["CHEMSEEK_MODEL_PATH"] = str(model_cfg["path"])
    env["CHEMSEEK_PROMPT"] = str(data_cfg.get("instruction", "<image>\n Give me the SMILES of the molecule. "))

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
    print(f"vLLM arch:        {vllm_architecture}")
    print(f"vLLM code dir:    {vllm_code_dir}")
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
