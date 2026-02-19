import argparse
import os
import subprocess
import sys
from pathlib import Path

import full_sft as full_core
from transformers import AutoConfig
from transformers.modeling_utils import load_sharded_checkpoint


def parse_cli_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Progressive full SFT from merged LoRA model.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(workspace / "progressive_sft_config.yaml"),
        help="Path to progressive full sft YAML config.",
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


def _maybe_launch_with_accelerate(cfg, cli_args: argparse.Namespace) -> None:
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


def _patch_model_loading_for_vllm_merged(cfg) -> None:
    """
    Some merged checkpoints are prepared for vLLM and have config.model_type=deepseek_vl_v2.
    HF AutoModel cannot resolve that type in this training environment.
    Fallback: load model weights from merged path, but force config from base DeepSeek-OCR-2.
    """
    original_from_pretrained = full_core.AutoModel.from_pretrained

    def patched_from_pretrained(model_name_or_path, *args, **kwargs):
        try:
            return original_from_pretrained(model_name_or_path, *args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if ("model type `deepseek_vl_v2`" not in msg) and ("modeling_deepseekocr2.py" not in msg):
                raise

            script_dir = Path(__file__).resolve().parent
            base_candidates = [
                script_dir / "../DeepSeek-OCR-2",
                script_dir / "../DeepSeek-OCR-2/hf_cache/models--deepseek-ai--DeepSeek-OCR-2",
            ]
            base_model_root = None
            for candidate in base_candidates:
                resolved = candidate.resolve()
                if resolved.exists():
                    base_model_root = resolved
                    break
            if base_model_root is None:
                raise RuntimeError(
                    "Detected vLLM-merged checkpoint (model_type=deepseek_vl_v2), "
                    "but cannot find base DeepSeek-OCR-2 path for architecture config. "
                    "Please make sure ../DeepSeek-OCR-2 exists."
                ) from exc

            base_model_path = full_core.resolve_local_model_path(base_model_root)
            base_config = AutoConfig.from_pretrained(str(base_model_path), trust_remote_code=True)
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["config"] = base_config
            base_source = str(base_model_path)
            print(
                "Detected vLLM-merged checkpoint config; "
                f"falling back to base config from: {base_model_path}"
            )
            model = original_from_pretrained(base_source, *args, **fallback_kwargs)

            merged_path = str(Path(model_name_or_path).resolve())
            load_sharded_checkpoint(model, merged_path, strict=False)
            print(f"Loaded merged checkpoint weights from: {merged_path}")
            return model

    full_core.AutoModel.from_pretrained = patched_from_pretrained


def main() -> None:
    cli_args = parse_cli_args()
    cfg = full_core.load_config(cli_args.config)
    _maybe_launch_with_accelerate(cfg, cli_args)
    _patch_model_loading_for_vllm_merged(cfg)

    # Reuse full_sft implementation while forcing progressive CLI/config.
    full_core.parse_cli_args = lambda: cli_args
    full_core._maybe_launch_with_accelerate = lambda *_args, **_kwargs: None
    full_core.main()


if __name__ == "__main__":
    main()
