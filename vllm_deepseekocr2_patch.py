"""Register DeepseekOCR2 custom vLLM model before rollout starts.

This module is imported by verl via `actor_rollout_ref.model.external_lib`.
It mirrors evaluation.py behavior by registering DeepseekOCR2ForCausalLM into
vLLM's ModelRegistry and exposing the DeepSeek-OCR2-vllm code directory.
"""

import os
import sys
import types
from pathlib import Path


def _resolve_vllm_code_dir() -> str | None:
    env_path = os.environ.get("CHEMSEEK_VLLM_CODE_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_dir():
            return str(p)

    # Fallback for current project layout:
    #   ChemLLM/ChemSeek-OCR  (this file)
    #   ChemLLM/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm
    candidate = (
        Path(__file__).resolve().parent.parent
        / "DeepSeek-OCR-2"
        / "DeepSeek-OCR2-master"
        / "DeepSeek-OCR2-vllm"
    )
    if candidate.is_dir():
        return str(candidate)
    return None


def _install_runtime_config() -> None:
    model_path = os.environ.get("CHEMSEEK_MODEL_PATH", "").strip()
    if not model_path:
        return

    config_mod = types.ModuleType("config")
    config_mod.BASE_SIZE = int(os.environ.get("CHEMSEEK_BASE_SIZE", "1024"))
    config_mod.IMAGE_SIZE = int(os.environ.get("CHEMSEEK_IMAGE_SIZE", "768"))
    config_mod.CROP_MODE = os.environ.get("CHEMSEEK_CROP_MODE", "true").lower() == "true"
    config_mod.MIN_CROPS = int(os.environ.get("CHEMSEEK_MIN_CROPS", "2"))
    config_mod.MAX_CROPS = int(os.environ.get("CHEMSEEK_MAX_CROPS", "6"))
    config_mod.MAX_CONCURRENCY = int(os.environ.get("CHEMSEEK_MAX_CONCURRENCY", "100"))
    config_mod.NUM_WORKERS = int(os.environ.get("CHEMSEEK_NUM_WORKERS", "64"))
    config_mod.PRINT_NUM_VIS_TOKENS = False
    config_mod.SKIP_REPEAT = True
    config_mod.MODEL_PATH = model_path
    config_mod.INPUT_PATH = ""
    config_mod.OUTPUT_PATH = ""
    config_mod.PROMPT = os.environ.get("CHEMSEEK_PROMPT", "<image>\n Give me the SMILES of the molecule. ")
    # Do not create tokenizer object at module import time; it may contain
    # protobuf descriptors that break Ray/pickle in actor creation.
    # Tokenizer will be lazily created in vllm_deepseekocr2_compat.
    config_mod.TOKENIZER = None
    sys.modules["config"] = config_mod


def _register() -> None:
    vllm_code_dir = _resolve_vllm_code_dir()
    if vllm_code_dir and vllm_code_dir not in sys.path:
        sys.path.insert(0, vllm_code_dir)

    _install_runtime_config()

    # Use lazy registration string to avoid importing CUDA-heavy modules here.
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekOCR2ForCausalLM",
        "vllm_deepseekocr2_compat:DeepseekOCR2ForCausalLM",
    )
    print("[vllm patch] registered DeepseekOCR2ForCausalLM")


try:
    _register()
except Exception as exc:
    print(f"[vllm patch] WARNING: failed to register DeepseekOCR2 model: {exc}")

