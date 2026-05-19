from __future__ import annotations

from typing import Any

STEP_EXECUTION_METHODS = ("prepare_encode", "denoise_step", "step_scheduler", "post_decode")


def supports_step_execution(pipeline: Any) -> bool:
    return all(callable(getattr(pipeline, name, None)) for name in STEP_EXECUTION_METHODS)
