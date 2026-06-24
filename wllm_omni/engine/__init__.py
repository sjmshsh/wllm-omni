from __future__ import annotations

__all__ = ["DiffusionEngine", "ModelRunner"]


def __getattr__(name: str):
    if name == "DiffusionEngine":
        from wllm_omni.engine.diffusion_engine import DiffusionEngine

        return DiffusionEngine
    if name == "ModelRunner":
        from wllm_omni.engine.model_runner import ModelRunner

        return ModelRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
