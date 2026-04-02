"""Process-wide compatibility patches loaded automatically by Python.

Python imports `sitecustomize` at startup when it is on `PYTHONPATH`.
We use it so Ray worker subprocesses inherit the same transformers shims.
"""

from __future__ import annotations

try:
    import transformers

    # verl expects this symbol, but some transformers versions expose only
    # AutoModelForImageTextToText.
    if (
        not hasattr(transformers, "AutoModelForVision2Seq")
        and hasattr(transformers, "AutoModelForImageTextToText")
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
except Exception:
    # Keep startup robust; if patching fails we let normal imports continue.
    pass

