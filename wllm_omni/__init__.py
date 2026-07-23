from __future__ import annotations

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_MODEL",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_PROMPT",
    "EngineConfig",
    "MiniOmniRuntime",
    "MiniOmniTrace",
    "OmniLLM",
    "PipelineConfig",
    "PipelineRegistry",
    "OmniOutput",
    "OmniRequest",
    "OmniSamplingParams",
    "PRESETS",
]


def __getattr__(name: str):
    if name in {"DEFAULT_IMAGE", "DEFAULT_MODEL", "DEFAULT_NEGATIVE_PROMPT", "DEFAULT_PROMPT", "EngineConfig"}:
        from wllm_omni import config

        return getattr(config, name)
    if name in {"MiniOmniRuntime", "MiniOmniTrace"}:
        from wllm_omni.engine import mini_omni_runtime

        return getattr(mini_omni_runtime, name)
    if name == "OmniLLM":
        from wllm_omni.llm import OmniLLM

        return OmniLLM
    if name in {"PipelineConfig", "PipelineRegistry"}:
        from wllm_omni.engine import pipeline

        return getattr(pipeline, name)
    if name == "OmniOutput":
        from wllm_omni.outputs import OmniOutput

        return OmniOutput
    if name == "OmniRequest":
        from wllm_omni.request import OmniRequest

        return OmniRequest
    if name in {"OmniSamplingParams", "PRESETS"}:
        from wllm_omni import sampling_params

        return getattr(sampling_params, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
