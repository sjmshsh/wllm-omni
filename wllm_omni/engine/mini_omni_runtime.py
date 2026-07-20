from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from wllm_omni.config import EngineConfig
from wllm_omni.engine.connectors import ARToDiffusionConnector, CallableARToDiffusionConnector, StageConnector
from wllm_omni.engine.pipeline import (
    DEFAULT_MINI_OMNI_PIPELINE,
    PipelineConfig,
    PipelineEdgeConfig,
    PipelineRegistry,
    PipelineStageConfig,
    default_pipeline_registry,
)
from wllm_omni.engine.stage import ARStage, DiffusionStage, Stage, StageOutput
from wllm_omni.engine.stage_graph import StageGraph
from wllm_omni.engine.stage_scheduler import StageExecutionRecord, StageScheduler, StageSchedulerResult
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class OmniStageRecord:
    name: str
    paradigm: ModelParadigm
    request_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MiniOmniTrace:
    request_id: str
    pipeline: str
    stages: list[OmniStageRecord] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)


class MiniOmniRuntime:
    """A tiny vLLM-Omni-style stage-graph runtime for AR -> diffusion.

    The runtime owns the top-level stage graph. Each stage owns its own
    engine/scheduler/runner/executor stack.
    """

    def __init__(
        self,
        config: EngineConfig,
        connector: StageConnector | Callable[[OmniRequest, ARTextOutput], OmniRequest] | None = None,
        ar_pipeline: ARPipeline | None = None,
        registry: PipelineRegistry | None = None,
    ):
        self.config = config
        self.registry = registry or default_pipeline_registry()
        self.pipeline = self.registry.get(config.pipeline or DEFAULT_MINI_OMNI_PIPELINE)
        self.connector = self._normalize_connector(connector)
        self.graph = self._build_graph(self.pipeline, ar_pipeline)
        self.stage_scheduler = StageScheduler(self.graph)
        self.last_trace: MiniOmniTrace | None = None

    def generate_ar(self, request: OmniRequest) -> ARTextOutput:
        leaves = self.graph.leaves()
        if len(leaves) != 1 or leaves[0].stage.paradigm != ModelParadigm.AUTOREGRESSIVE:
            raise RuntimeError(
                f"Pipeline {self.pipeline.name!r} is not an AR-only pipeline. "
                "Use pipeline='ar_text' for AR-only generation."
            )
        result = self.stage_scheduler.run(request)
        self.last_trace = self._make_trace(result)
        if len(result.final_outputs) != 1:
            raise RuntimeError(
                f"AR generation expects one final output from pipeline {self.pipeline.name!r}, "
                f"got {len(result.final_outputs)}."
            )
        try:
            return self._ar_output(result.final_outputs[0])
        except TypeError as exc:
            raise RuntimeError(
                f"Pipeline {self.pipeline.name!r} does not produce AR text output. "
                "Use pipeline='ar_text' for AR-only generation."
            ) from exc

    def generate(self, request: OmniRequest) -> list[OmniOutput]:
        result = self.stage_scheduler.run(request)
        self.last_trace = self._make_trace(result)
        return [self._diffusion_output(output) for output in result.final_outputs]

    def _build_graph(self, pipeline: PipelineConfig, ar_pipeline: ARPipeline | None) -> StageGraph:
        graph = StageGraph()
        for stage_config in pipeline.stages:
            graph.add_node(stage_config.node_id, self._make_stage(stage_config, ar_pipeline))
        for edge_config in pipeline.edges:
            graph.add_edge(edge_config.source, edge_config.target, self._make_connector(edge_config))
        return graph

    def _make_stage(self, stage_config: PipelineStageConfig, ar_pipeline: ARPipeline | None) -> Stage:
        if stage_config.paradigm == ModelParadigm.AUTOREGRESSIVE:
            return ARStage(self.config, pipeline=ar_pipeline)
        if stage_config.paradigm == ModelParadigm.DIFFUSION:
            return DiffusionStage(self.config)
        raise ValueError(
            f"Pipeline {self.pipeline.name!r} stage {stage_config.node_id!r} "
            f"uses unsupported paradigm={stage_config.paradigm!r}."
        )

    def _make_connector(self, edge_config: PipelineEdgeConfig) -> StageConnector:
        if edge_config.connector == "ar_to_diffusion":
            return self.connector
        raise ValueError(
            f"Pipeline {self.pipeline.name!r} edge {edge_config.source!r}->{edge_config.target!r} "
            f"uses unsupported connector={edge_config.connector!r}."
        )

    @staticmethod
    def _normalize_connector(
        connector: StageConnector | Callable[[OmniRequest, ARTextOutput], OmniRequest] | None,
    ) -> StageConnector:
        if connector is None:
            return ARToDiffusionConnector()
        if isinstance(connector, StageConnector):
            return connector
        return CallableARToDiffusionConnector(connector)

    def _make_trace(self, result: StageSchedulerResult) -> MiniOmniTrace:
        return MiniOmniTrace(
            request_id=result.root_request_id,
            pipeline=self.pipeline.name,
            stages=[self._make_stage_record_from_record(record) for record in result.records],
            graph_nodes=[record.node_id for record in result.records],
        )

    @staticmethod
    def _make_stage_record_from_record(record: StageExecutionRecord) -> OmniStageRecord:
        metadata = dict(record.metadata)
        paradigm = ModelParadigm(metadata.pop("paradigm"))
        return OmniStageRecord(
            name=record.stage_name,
            paradigm=paradigm,
            request_id=record.request_id,
            metadata=metadata,
        )

    @staticmethod
    def _make_stage_record_from_stage(
        stage_name: str,
        paradigm: ModelParadigm,
        output: StageOutput,
        elapsed_s: float,
    ) -> OmniStageRecord:
        metadata = dict(output.metadata)
        metadata["elapsed_s"] = elapsed_s
        return OmniStageRecord(name=stage_name, paradigm=paradigm, request_id=output.request_id, metadata=metadata)

    @staticmethod
    def _ar_output(output: StageOutput) -> ARTextOutput:
        if not isinstance(output.data, ARTextOutput):
            raise TypeError(f"Expected ARTextOutput, got {type(output.data).__name__}.")
        return output.data

    @staticmethod
    def _diffusion_output(output: StageOutput) -> OmniOutput:
        if not isinstance(output.data, OmniOutput):
            raise TypeError(f"Expected OmniOutput, got {type(output.data).__name__}.")
        return output.data
