"""Compatibility bridge for DeepSeek-OCR2 custom vLLM model.

This module keeps one import target that can work for both:
  - old vLLM (e.g. 0.8.5)
  - newer vLLM (e.g. 0.12) where some symbols moved/changed.
"""

from __future__ import annotations

import os

import vllm.model_executor as _model_executor


def _ensure_sampling_metadata() -> None:
    if hasattr(_model_executor, "SamplingMetadata"):
        return
    try:
        # vLLM >= 0.12
        from vllm.v1.sample.metadata import SamplingMetadata as sampling_metadata
    except Exception:
        from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata as sampling_metadata
    _model_executor.SamplingMetadata = sampling_metadata


def _ensure_set_default_torch_dtype() -> None:
    # Old DeepSeek-OCR2-vllm imports this helper from vLLM; in some newer
    # versions it moved/was removed.
    from contextlib import contextmanager

    import torch
    import vllm.model_executor.model_loader.utils as loader_utils

    if hasattr(loader_utils, "set_default_torch_dtype"):
        return

    @contextmanager
    def set_default_torch_dtype(dtype):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(old_dtype)

    loader_utils.set_default_torch_dtype = set_default_torch_dtype


_ensure_sampling_metadata()
_ensure_set_default_torch_dtype()


def _patch_processor_tokenizer_lazy_load() -> None:
    from process.image_process import DeepseekOCR2Processor
    from transformers import AutoTokenizer

    original_init = DeepseekOCR2Processor.__init__

    def patched_init(
        self,
        tokenizer=None,
        *args,
        **kwargs,
    ):
        if tokenizer is None:
            model_path = os.environ.get("CHEMSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return original_init(self, tokenizer=tokenizer, *args, **kwargs)

    DeepseekOCR2Processor.__init__ = patched_init


_patch_processor_tokenizer_lazy_load()


from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # noqa: E402,F401

