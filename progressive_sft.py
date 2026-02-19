import argparse
import os
import subprocess
import sys
from pathlib import Path

import full_sft as full_core


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


def main() -> None:
    cli_args = parse_cli_args()
    cfg = full_core.load_config(cli_args.config)
    _maybe_launch_with_accelerate(cfg, cli_args)

    # Reuse full_sft implementation while forcing progressive CLI/config.
    full_core.parse_cli_args = lambda: cli_args
    full_core._maybe_launch_with_accelerate = lambda *_args, **_kwargs: None
    full_core.main()


if __name__ == "__main__":
    main()
