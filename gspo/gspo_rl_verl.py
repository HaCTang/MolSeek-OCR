"""GSPO RL for DeepSeek-OCR-2 using verl framework.

This script provides:
  1. Custom reward function: SMILES-based reward for molecular OCR
  2. Custom dataset class: Load image+SMILES data for verl
  3. Training launch: Configure and start verl GSPO training

GSPO (Group Sequence Policy Optimization) replaces KL regularization with
tight symmetric clipping (clip_ratio_low / clip_ratio_high), giving more
stable updates on MoE architectures.

Usage:
  # Step 1: Prepare training data
  python gspo/prepare_verl_data.py --config gspo/gspo_rl_verl_config.yaml

  # Step 2: Launch GSPO training
  python gspo/gspo_rl_verl.py --config gspo/gspo_rl_verl_config.yaml
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_with_live_reward_tqdm(cmd: list[str], env: dict) -> int:
    """Run trainer command and stream per-step reward with tqdm."""
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    step_reward_re = re.compile(
        r"step:(?P<step>\d+).*?critic/rewards/mean:(?P<reward>[-+0-9.eE]+)"
    )
    total_step_re = re.compile(r"Training Progress:\s*\d+%\|.*?\|\s*(\d+)/(\d+)")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    pbar = None
    last_step = 0

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

        total_match = total_step_re.search(line)
        if total_match and tqdm is not None and pbar is None:
            total_steps = int(total_match.group(2))
            pbar = tqdm(
                total=total_steps,
                desc="GSPO Reward",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )

        reward_match = step_reward_re.search(line)
        if reward_match:
            step = int(reward_match.group("step"))
            reward = float(reward_match.group("reward"))
            if pbar is not None:
                if step > last_step:
                    pbar.update(step - last_step)
                pbar.set_postfix_str(f"reward={reward:.6f}")
            else:
                print(f"[reward] step={step} reward={reward:.6f}")
            last_step = max(last_step, step)

    retcode = proc.wait()
    if pbar is not None:
        pbar.close()
    return retcode


# =========================================================================
# Configuration
# =========================================================================

def _resolve_path(path_str: str, base_dir: Optional[Path] = None) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir or SCRIPT_DIR) / p
    return str(p.resolve())


def _stringify_hydra_value(value: Any) -> str:
    """Convert a YAML value to Hydra CLI literal."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _collect_hydra_cli_overrides(cfg: dict) -> Dict[str, str]:
    """Collect overrides from YAML using original Hydra dotted names.

    Supports:
      1) top-level dotted keys: {"actor_rollout_ref.rollout.n": 5}
      2) grouped mapping: {"hydra_cli_overrides": {...}}
    """
    out: Dict[str, str] = {}

    grouped = cfg.get("hydra_cli_overrides", {})
    if isinstance(grouped, dict):
        for k, v in grouped.items():
            if isinstance(k, str) and "." in k:
                out[k] = _stringify_hydra_value(v)

    for k, v in cfg.items():
        if isinstance(k, str) and "." in k:
            out[k] = _stringify_hydra_value(v)
    return out


def _apply_hydra_cli_overrides(cmd: list[str], overrides: Dict[str, str]) -> list[str]:
    """Apply Hydra dotted-key overrides onto an existing CLI list."""
    if not overrides:
        return cmd

    key_to_idx: Dict[str, int] = {}
    for i, item in enumerate(cmd):
        if "=" in item:
            key, _ = item.split("=", 1)
            key_to_idx[key] = i

    for key, value in overrides.items():
        entry = f"{key}={value}"
        if key in key_to_idx:
            cmd[key_to_idx[key]] = entry
        else:
            cmd.append(entry)
    return cmd


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
    cfg.setdefault("validation", {})
    if cfg["validation"].get("val_parquet"):
        cfg["validation"]["val_parquet"] = _resolve_path(cfg["validation"]["val_parquet"], base)
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
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line.replace(" ", "")
    return ""


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


def _repetition_penalty(smiles: str) -> float:
    """Return a penalty in [0, 1] for degenerate repetitive SMILES.

    0 = no penalty (clean), 1 = maximally repetitive.
    Detects patterns like "CCCCCC...", "C.C.C.C...", "ClClCl..." etc.
    """
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
            if s[i:i + pat_len] == pat:
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


def _compute_reward_components(
    gold_smiles: str, pred_smiles: str, chiral_no_annotation_reward: float = 0.0
) -> Dict[str, float]:
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

    return {
        "validity": float(valid_pred),
        "tanimoto": _tanimoto_similarity(gold_smiles, pred_smiles),
        "canon_smiles": canon_match,
        "graph": graph_sim,
        "chiral": chiral_acc,
        "repetition_penalty": rep_penalty,
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
    w_validity = float(w.get("validity", 0.5))
    w_tanimoto = float(w.get("tanimoto", 2.0))
    w_canon = float(w.get("canon_smiles", 3.0))
    w_graph = float(w.get("graph", 2.0))
    w_chiral = float(w.get("chiral", 1.0))
    w_rep_penalty = float(w.get("repetition_penalty", 2.0))
    chiral_no_ann = float(kwargs.get("chiral_no_annotation_reward", 0.0))

    components = _compute_reward_components(gold_smiles, pred_smiles, chiral_no_ann)

    weighted_sum = (
        w_validity * components["validity"]
        + w_tanimoto * components["tanimoto"]
        + w_canon * components["canon_smiles"]
        + w_graph * components["graph"]
        + w_chiral * components["chiral"]
        - w_rep_penalty * components["repetition_penalty"]
    )
    w_total = w_validity + w_tanimoto + w_canon + w_graph + w_chiral
    score = float(weighted_sum / w_total) if w_total > 0 else 0.0
    score = max(0.0, score)

    return score


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
        # DeepSeek-OCR2-vLLM registers the modality name as "image".
        self.image_key = str(getattr(config, "image_key", "image") or "image")
        self.apply_chat_template_kwargs = dict(getattr(config, "apply_chat_template_kwargs", {}))
        print(f"[ChemSeekOCRDataset] loaded {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _ensure_config_module():
        """Inject a synthetic 'config' module so that DeepseekOCR2Processor
        uses the correct PROMPT and TOKENIZER (same trick as evaluation.py).
        Without this, the default config.py has a document-OCR prompt and wrong
        tokenizer, so the model never receives correct image context."""
        import sys, types
        if "config" in sys.modules and getattr(sys.modules["config"], "_chemseek_injected", False):
            return
        model_path = os.environ.get("CHEMSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2")
        prompt = os.environ.get("CHEMSEEK_PROMPT", "<image>\n Give me the SMILES of the molecule. ")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config_mod = types.ModuleType("config")
        config_mod.BASE_SIZE = 1024
        config_mod.IMAGE_SIZE = 768
        config_mod.CROP_MODE = True
        config_mod.MIN_CROPS = 2
        config_mod.MAX_CROPS = 6
        config_mod.MAX_CONCURRENCY = 100
        config_mod.NUM_WORKERS = 64
        config_mod.PRINT_NUM_VIS_TOKENS = False
        config_mod.SKIP_REPEAT = True
        config_mod.MODEL_PATH = model_path
        config_mod.INPUT_PATH = ""
        config_mod.OUTPUT_PATH = ""
        config_mod.PROMPT = prompt
        config_mod.TOKENIZER = tokenizer
        config_mod._chemseek_injected = True
        sys.modules["config"] = config_mod
        print(f"[ChemSeekOCRDataset] Injected config module: PROMPT={repr(prompt[:60])}")

    def _build_deepseek_mm_payload(self, image):
        """Build DeepSeek-OCR2-vllm expected multimodal image payload."""
        self._ensure_config_module()
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

        # Keep RL rollout prompt format aligned with SFT/eval:
        # a single "<image>" token followed by instruction text.
        raw_prompt = self._messages_to_plain_prompt(messages)
        if "<image>" not in raw_prompt:
            raw_prompt = "<image>\n Give me the SMILES of the molecule. "

        model_inputs = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        mm_payload = self._build_deepseek_mm_payload(image)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        multi_modal_data = {self.image_key: mm_payload}
        if self.image_key != "image":
            # The DeepSeek-OCR2-vLLM plugin expects "image" as modality key.
            multi_modal_data["image"] = mm_payload

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=True)
        image_token_id = getattr(self.tokenizer, "vocab", {}).get("<image>")
        if image_token_id is not None and image_token_id not in raw_prompt_ids:
            fallback_prompt = "<image>\n Give me the SMILES of the molecule. "
            raw_prompt = fallback_prompt
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=True)
            model_inputs = self.tokenizer(
                raw_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
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
_FREEZE_EXT_MODULE_NAME = "_chemseek_gspo_freeze"
_FREEZE_EXT_CODE = '''\
"""Auto-generated external_lib for verl: DeepSeek OCR2 patch + layer freezing."""
import os
import sys
import types

# ---------------------------------------------------------------------------
# Inject a synthetic "config" module BEFORE any DeepSeek-OCR2-vllm imports.
# Without this, the default config.py uses a document-OCR prompt and a
# wrong tokenizer, causing the model to never receive correct image context.
# This mirrors evaluation.py._setup_vllm_env().
# ---------------------------------------------------------------------------
def _setup_config_module():
    if "config" in sys.modules and getattr(sys.modules["config"], "_chemseek_injected", False):
        return
    model_path = os.environ.get("CHEMSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2")
    prompt = os.environ.get("CHEMSEEK_PROMPT", "<image>\\n Give me the SMILES of the molecule. ")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config_mod = types.ModuleType("config")
    config_mod.BASE_SIZE = 1024
    config_mod.IMAGE_SIZE = 768
    config_mod.CROP_MODE = True
    config_mod.MIN_CROPS = 2
    config_mod.MAX_CROPS = 6
    config_mod.MAX_CONCURRENCY = 100
    config_mod.NUM_WORKERS = 64
    config_mod.PRINT_NUM_VIS_TOKENS = False
    config_mod.SKIP_REPEAT = True
    config_mod.MODEL_PATH = model_path
    config_mod.INPUT_PATH = ""
    config_mod.OUTPUT_PATH = ""
    config_mod.PROMPT = prompt
    config_mod.TOKENIZER = tokenizer
    config_mod._chemseek_injected = True
    sys.modules["config"] = config_mod
    print(
        f"[GSPO config] Injected config module: "
        f"MODEL_PATH={model_path}, PROMPT={repr(prompt[:60])}"
    )

_setup_config_module()

from transformers import AutoModel, AutoModelForCausalLM

try:
    import vllm_deepseekocr2_patch  # noqa: F401
except Exception as exc:
    print(f"[GSPO freeze] WARN: failed to import vllm_deepseekocr2_patch: {exc}")


def _parse_freeze_layers():
    raw = os.environ.get("CHEMSEEK_FREEZE_LAYERS", "")
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except Exception:
            continue
        if value >= 0:
            out.append(value)
    return sorted(set(out))


FREEZE_LAYERS = _parse_freeze_layers()


def _parse_freeze_modules():
    raw = os.environ.get("CHEMSEEK_FREEZE_MODULES", "")
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


FREEZE_MODULES = _parse_freeze_modules()


def _apply_layer_freeze(model):
    if not FREEZE_LAYERS and not FREEZE_MODULES:
        print("[GSPO freeze] freeze_layers/freeze_modules are empty; skip freezing.")
        return model

    prefixes = []
    for idx in FREEZE_LAYERS:
        prefixes.extend(
            [
                f"model.layers.{idx}.",
                f"layers.{idx}.",
                f"model.model.layers.{idx}.",
            ]
        )

    frozen_count = 0
    frozen_elems = 0
    module_hit_counter = {m: 0 for m in FREEZE_MODULES}
    for name, param in model.named_parameters():
        by_layer = any(name.startswith(prefix) for prefix in prefixes)
        by_module = False
        for module_pattern in FREEZE_MODULES:
            if module_pattern in name:
                module_hit_counter[module_pattern] += 1
                by_module = True
        if by_layer or by_module:
            if param.requires_grad:
                frozen_count += 1
                frozen_elems += int(param.numel())
            param.requires_grad = False

    print(
        f"[GSPO freeze] Applied freeze_layers={FREEZE_LAYERS}, "
        f"freeze_modules={FREEZE_MODULES}; "
        f"frozen params={frozen_count}, frozen elements={frozen_elems:,}"
    )
    if FREEZE_MODULES:
        print("[GSPO freeze] freeze_modules hit counts:")
        for module_pattern, hit_count in module_hit_counter.items():
            print(f"  - {module_pattern}: {hit_count}")
    return model


_orig_auto_model_fp = AutoModel.from_pretrained.__func__
_orig_auto_causal_fp = AutoModelForCausalLM.from_pretrained.__func__


@classmethod
def _patched_auto_model_fp(cls, *args, **kwargs):
    model = _orig_auto_model_fp(cls, *args, **kwargs)
    return _apply_layer_freeze(model)


@classmethod
def _patched_auto_causal_fp(cls, *args, **kwargs):
    model = _orig_auto_causal_fp(cls, *args, **kwargs)
    return _apply_layer_freeze(model)


AutoModel.from_pretrained = _patched_auto_model_fp
AutoModelForCausalLM.from_pretrained = _patched_auto_causal_fp
print(
    "[GSPO freeze] Patched AutoModel "
    f"for freeze_layers={FREEZE_LAYERS}, freeze_modules={FREEZE_MODULES}"
)
'''


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


def _write_freeze_ext_module(target_dir: Path) -> str:
    ext_path = target_dir / f"{_FREEZE_EXT_MODULE_NAME}.py"
    ext_path.write_text(_FREEZE_EXT_CODE, encoding="utf-8")
    print(f"Wrote external_lib module: {ext_path}")
    return _FREEZE_EXT_MODULE_NAME


def train(cfg: dict, config_path: str) -> None:
    """Launch verl GSPO with YAML-provided Hydra overrides."""

    _assert_target_runtime_versions()
    _ensure_transformers_compat()

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    wandb_cfg = cfg.get("wandb", {})
    output_dir = cfg.get("output_dir", str(SCRIPT_DIR / "weight_gspo_rl_verl"))
    freeze_layers = cfg.get("freeze_layers", cfg.get("training", {}).get("freeze_layers", [])) or []
    freeze_modules = cfg.get("freeze_modules", cfg.get("training", {}).get("freeze_modules", [])) or []
    if not isinstance(freeze_layers, list):
        raise ValueError("training.freeze_layers must be a list, e.g. [0,1,2]")
    if not isinstance(freeze_modules, list):
        raise ValueError("training.freeze_modules must be a list, e.g. ['model.embed_tokens']")
    freeze_layers = [int(x) for x in freeze_layers]
    if any(x < 0 for x in freeze_layers):
        raise ValueError("training.freeze_layers must contain non-negative integers")
    freeze_modules = [str(x).strip() for x in freeze_modules if str(x).strip()]
    _ensure_local_checkpoint_compat(str(model_cfg.get("path", "")))
    ext_module_name = _write_freeze_ext_module(SCRIPT_DIR)

    data_dir = Path(data_cfg["output_dir"])
    train_parquet = data_dir / "train.parquet"
    val_parquet_cfg = cfg.get("val_parquet", cfg.get("validation", {}).get("val_parquet"))
    val_parquet = Path(val_parquet_cfg) if val_parquet_cfg else (data_dir / "val.parquet")

    if not train_parquet.is_file():
        print(f"Error: Training data not found at {train_parquet}")
        print("Run 'python gspo/prepare_verl_data.py --config gspo/gspo_rl_verl_config.yaml' first.")
        sys.exit(1)

    script_path = str(SCRIPT_DIR / "gspo_rl_verl.py")

    vllm_architecture = str(model_cfg.get("vllm_architecture", "DeepseekOCR2ForCausalLM"))
    use_fused_kernels = bool(model_cfg.get("use_fused_kernels", False))
    if use_fused_kernels and "deepseekocr2" in vllm_architecture.lower():
        print(
            "[compat] DeepseekOCR2 is incompatible with verl fused-kernel forward path "
            "(unexpected cache_position kwarg). Auto-disable use_fused_kernels."
        )
        use_fused_kernels = False
    vllm_code_dir = model_cfg.get("vllm_code_dir")
    if vllm_code_dir:
        vllm_code_dir = _resolve_path(vllm_code_dir, Path(config_path).resolve().parent)
    else:
        fallback_vllm_dir = (
            PROJECT_ROOT.parent
            / "DeepSeek-OCR-2"
            / "DeepSeek-OCR2-master"
            / "DeepSeek-OCR2-vllm"
        )
        if fallback_vllm_dir.is_dir():
            vllm_code_dir = str(fallback_vllm_dir.resolve())
    _ensure_deepseek_ocr2_vllm_compat(vllm_code_dir)

    logger_list = '["console"]'
    if wandb_cfg.get("enabled", False):
        logger_list = '["console","wandb"]'

    val_file_for_cmd = val_parquet if val_parquet.is_file() else train_parquet

    cmd = [
        sys.executable, str(SCRIPT_DIR / "verl_main_ppo_compat.py"),
        # Core paths/integration only; hyperparameters come from hydra_cli_overrides
        f"actor_rollout_ref.model.path={model_cfg['path']}",
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.architectures=[\"{vllm_architecture}\"]",
        f"actor_rollout_ref.model.use_fused_kernels={use_fused_kernels}",
        f"actor_rollout_ref.model.external_lib={ext_module_name}",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_file_for_cmd}",
        f"data.custom_cls.path={script_path}",
        f"custom_reward_function.path={script_path}",
        f"trainer.logger={logger_list}",
        f"trainer.default_local_dir={output_dir}",
    ]
    hydra_overrides = _collect_hydra_cli_overrides(cfg)
    cmd = _apply_hydra_cli_overrides(cmd, hydra_overrides)

    env = os.environ.copy()
    pythonpath_parts = [str(SCRIPT_DIR), str(PROJECT_ROOT)]
    if vllm_code_dir:
        pythonpath_parts.append(str(vllm_code_dir))
    old_pythonpath = env.get("PYTHONPATH")
    if old_pythonpath:
        pythonpath_parts.append(old_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    if bool(cfg.get("allow_tf32", cfg.get("training", {}).get("allow_tf32", True))):
        env.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    # vLLM's CuMemAllocator is incompatible with expandable_segments.
    if "PYTORCH_CUDA_ALLOC_CONF" in env and "expandable_segments:True" in env["PYTORCH_CUDA_ALLOC_CONF"]:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("VLLM_USE_V1", "0")
    if vllm_code_dir:
        env["CHEMSEEK_VLLM_CODE_DIR"] = vllm_code_dir
    env["CHEMSEEK_MODEL_PATH"] = str(model_cfg["path"])
    env["CHEMSEEK_PROMPT"] = str(data_cfg.get("instruction", "<image>\n Give me the SMILES of the molecule. "))
    env["CHEMSEEK_FREEZE_LAYERS"] = ",".join(str(x) for x in freeze_layers)
    env["CHEMSEEK_FREEZE_MODULES"] = ",".join(freeze_modules)

    if wandb_cfg.get("enabled", False):
        api_key = wandb_cfg.get("api_key") or os.environ.get("WANDB_API_KEY")
        if api_key:
            env["WANDB_API_KEY"] = api_key

    gpu_ids = cfg.get("gpu_ids", cfg.get("gpu", {}).get("gpu_ids"))
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    def _ov(key: str, default: str = "<auto>") -> str:
        return hydra_overrides.get(key, default)

    print("=" * 72)
    print("Launching verl GSPO training")
    print("=" * 72)
    print(f"Model:            {model_cfg['path']}")
    print(f"Train data:       {train_parquet}")
    print(
        "GPUs/TP:          "
        f"{_ov('trainer.n_gpus_per_node')} / {_ov('actor_rollout_ref.rollout.tensor_model_parallel_size')}"
    )
    print(f"vLLM arch:        {vllm_architecture}")
    print(f"vLLM code dir:    {vllm_code_dir}")
    print(f"Group size:       {_ov('actor_rollout_ref.rollout.n')}")
    print(f"Batch size:       {_ov('data.train_batch_size')}")
    print(f"LR:               {_ov('actor_rollout_ref.actor.optim.lr')}")
    print(f"Freeze layers:    {freeze_layers if freeze_layers else '[] (disabled)'}")
    print(f"Freeze modules:   {freeze_modules if freeze_modules else '[] (disabled)'}")
    print(f"Epochs:           {_ov('trainer.total_epochs')}")
    print(
        "Validation:       "
        f"test_freq={_ov('trainer.test_freq')} val_before_train={_ov('trainer.val_before_train')}"
    )
    print(f"Val data:         {val_file_for_cmd}")
    print(f"GSPO clip_low:    {_ov('actor_rollout_ref.actor.clip_ratio_low')}")
    print(f"GSPO clip_high:   {_ov('actor_rollout_ref.actor.clip_ratio_high')}")
    print(f"GSPO clip_c:      {_ov('actor_rollout_ref.actor.clip_ratio_c')}")
    print(f"Loss agg mode:    {_ov('actor_rollout_ref.actor.loss_agg_mode')}")
    print(
        "KL loss:          "
        f"{_ov('actor_rollout_ref.actor.use_kl_loss')} (coef={_ov('actor_rollout_ref.actor.kl_loss_coef')})"
    )
    print(f"Output:           {output_dir}")
    print("=" * 72)
    print("Command:")
    print("  " + " \\\n    ".join(cmd[:3]) + " \\")
    print("    " + " \\\n    ".join(cmd[3:]))
    print("=" * 72)

    retcode = _run_with_live_reward_tqdm(cmd, env)
    sys.exit(retcode)


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
