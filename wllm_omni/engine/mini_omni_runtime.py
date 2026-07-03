from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from wllm_omni.config import EngineConfig
from wllm_omni.engine.connectors import ARToDiffusionConnector, CallableARToDiffusionConnector, StageConnector
from wllm_omni.engine.stage import ARStage, DiffusionStage, StageOutput
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
    ):
        self.config = config
        self.ar_stage = ARStage(config, pipeline=ar_pipeline)
        self.diffusion_stage = DiffusionStage(config)
        self.connector = self._normalize_connector(connector)
        self.graph = self._build_default_graph()
        self.stage_scheduler = StageScheduler(self.graph)
        self.last_trace: MiniOmniTrace | None = None

    def generate_ar(self, request: OmniRequest) -> ARTextOutput:
        stage_output, elapsed_s = StageScheduler._run_stage(self.ar_stage, request)
        self.last_trace = MiniOmniTrace(
            request_id=request.request_id,
            stages=[
                self._make_stage_record_from_stage(
                    self.ar_stage.name,
                    self.ar_stage.paradigm,
                    stage_output,
                    elapsed_s,
                )
            ],
            graph_nodes=[self.ar_stage.name],
        )
        return self._ar_output(stage_output)

    def generate(self, request: OmniRequest) -> list[OmniOutput]:
        result = self.stage_scheduler.run(request)
        self.last_trace = self._make_trace(result)
        return [self._diffusion_output(output) for output in result.final_outputs]

    def _build_default_graph(self) -> StageGraph:
        graph = StageGraph()
        graph.add_node("ar.prompt_bridge", self.ar_stage)
        graph.add_node("diffusion.wan22_i2v", self.diffusion_stage)
        graph.add_edge("ar.prompt_bridge", "diffusion.wan22_i2v", self.connector)
        return graph

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
