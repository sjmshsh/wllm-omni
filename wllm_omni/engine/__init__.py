from __future__ import annotations

__all__ = ["DiffusionEngine", "ModelRunner", "PipelineConfig", "PipelineRegistry", "StageGraph", "StageScheduler"]


def __getattr__(name: str):
    if name == "DiffusionEngine":
        from wllm_omni.engine.diffusion_engine import DiffusionEngine

        return DiffusionEngine
    if name == "ModelRunner":
        from wllm_omni.engine.model_runner import ModelRunner

        return ModelRunner
    if name == "PipelineConfig":
        from wllm_omni.engine.pipeline import PipelineConfig

        return PipelineConfig
    if name == "PipelineRegistry":
        from wllm_omni.engine.pipeline import PipelineRegistry

        return PipelineRegistry
    if name == "StageGraph":
        from wllm_omni.engine.stage_graph import StageGraph

        return StageGraph
    if name == "StageScheduler":
        from wllm_omni.engine.stage_scheduler import StageScheduler

        return StageScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
