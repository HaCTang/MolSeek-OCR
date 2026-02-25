"""Compatibility launcher for `verl.trainer.main_ppo`.

This shim patches symbols expected by the current local verl codebase but
missing in some transformers versions (e.g. AutoModelForVision2Seq), then
delegates to verl's main entrypoint.
"""

from __future__ import annotations

import transformers


def _patch_transformers_symbols() -> None:
    if (
        not hasattr(transformers, "AutoModelForVision2Seq")
        and hasattr(transformers, "AutoModelForImageTextToText")
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText


def main() -> None:
    _patch_transformers_symbols()
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()

