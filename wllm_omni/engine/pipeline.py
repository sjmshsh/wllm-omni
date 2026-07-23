from __future__ import annotations

from dataclasses import dataclass

from wllm_omni.model_types import ModelParadigm


AR_TEXT_PIPELINE = "ar_text"
WAN_I2V_PIPELINE = "wan_i2v"
QWEN_TO_WAN_I2V_PIPELINE = "qwen_to_wan_i2v"
DEFAULT_MINI_OMNI_PIPELINE = QWEN_TO_WAN_I2V_PIPELINE


@dataclass(frozen=True, slots=True)
class PipelineStageConfig:
    node_id: str
    paradigm: ModelParadigm


@dataclass(frozen=True, slots=True)
class PipelineEdgeConfig:
    source: str
    target: str
    connector: str


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    name: str
    stages: tuple[PipelineStageConfig, ...]
    edges: tuple[PipelineEdgeConfig, ...] = ()

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.node_id for stage in self.stages)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("PipelineConfig.name cannot be empty.")
        if not self.stages:
            raise ValueError(f"Pipeline {self.name!r} requires at least one stage.")
        stage_ids = self.stage_ids
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"Pipeline {self.name!r} contains duplicate stage ids: {stage_ids!r}.")
        known_stage_ids = set(stage_ids)
        for edge in self.edges:
            if edge.source not in known_stage_ids:
                raise ValueError(
                    f"Pipeline {self.name!r} edge source {edge.source!r} is not a registered stage."
                )
            if edge.target not in known_stage_ids:
                raise ValueError(
                    f"Pipeline {self.name!r} edge target {edge.target!r} is not a registered stage."
                )


class PipelineRegistry:
    """Registry of model-level stage pipelines.

    This mirrors the vLLM-Omni style split where a model/pipeline chooses a
    fixed stage topology, while user prompts and media are runtime inputs to
    that topology.
    """

    def __init__(self, pipelines: list[PipelineConfig] | None = None):
        self._pipelines: dict[str, PipelineConfig] = {}
        for pipeline in pipelines or []:
            self.register(pipeline)

    def register(self, pipeline: PipelineConfig) -> None:
        pipeline.validate()
        if pipeline.name in self._pipelines:
            raise ValueError(f"Duplicate pipeline name={pipeline.name!r}.")
        self._pipelines[pipeline.name] = pipeline

    def get(self, name: str) -> PipelineConfig:
        try:
            return self._pipelines[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._pipelines))
            raise KeyError(f"Unknown pipeline {name!r}. Available pipelines: {available}.") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._pipelines))


def default_pipeline_registry() -> PipelineRegistry:
    return PipelineRegistry(
        [
            PipelineConfig(
                name=AR_TEXT_PIPELINE,
                stages=(
                    PipelineStageConfig(
                        node_id="ar.prompt_bridge",
                        paradigm=ModelParadigm.AUTOREGRESSIVE,
                    ),
                ),
            ),
            PipelineConfig(
                name=WAN_I2V_PIPELINE,
                stages=(
                    PipelineStageConfig(
                        node_id="diffusion.wan22_i2v",
                        paradigm=ModelParadigm.DIFFUSION,
                    ),
                ),
            ),
            PipelineConfig(
                name=QWEN_TO_WAN_I2V_PIPELINE,
                stages=(
                    PipelineStageConfig(
                        node_id="ar.prompt_bridge",
                        paradigm=ModelParadigm.AUTOREGRESSIVE,
                    ),
                    PipelineStageConfig(
                        node_id="diffusion.wan22_i2v",
                        paradigm=ModelParadigm.DIFFUSION,
                    ),
                ),
                edges=(
                    PipelineEdgeConfig(
                        source="ar.prompt_bridge",
                        target="diffusion.wan22_i2v",
                        connector="ar_to_diffusion",
                    ),
                ),
            ),
        ]
    )
