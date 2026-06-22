from __future__ import annotations

__all__ = ["ModelRunner", "OmniEngine"]


def __getattr__(name: str):
    if name == "ModelRunner":
        from wllm_omni.engine.model_runner import ModelRunner

        return ModelRunner
    if name == "OmniEngine":
        from wllm_omni.engine.omni_engine import OmniEngine

        return OmniEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
