from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from wllm_omni.config import EngineConfig
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import clone_sampling_params
from wllm_omni.engine.stage import ARStage, DiffusionStage, StageOutput


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


class MiniOmniRuntime:
    """A tiny vLLM-Omni-style runtime for AR -> diffusion composition.

    The runtime lives above concrete model engines. It coordinates stage order,
    while each stage owns its engine/scheduler/runner/executor details.
    """

    def __init__(
        self,
        config: EngineConfig,
        connector: Callable[[OmniRequest, ARTextOutput], OmniRequest] | None = None,
        ar_pipeline: ARPipeline | None = None,
    ):
        self.config = config
        self.ar_stage = ARStage(config, pipeline=ar_pipeline)
        self.diffusion_stage = DiffusionStage(config)
        self.connector = connector or self._default_connector
        self.last_trace: MiniOmniTrace | None = None

    def generate_ar(self, request: OmniRequest) -> ARTextOutput:
        stage_output, elapsed_s = self._run_stage(self.ar_stage, request)
        self.last_trace = MiniOmniTrace(
            request_id=request.request_id,
            stages=[self._make_stage_record(self.ar_stage, stage_output, elapsed_s)],
        )
        return self._ar_output(stage_output)

    def generate(self, request: OmniRequest) -> list[OmniOutput]:
        trace = MiniOmniTrace(request_id=request.request_id)

        ar_stage_output, ar_elapsed_s = self._run_stage(self.ar_stage, request)
        trace.stages.append(self._make_stage_record(self.ar_stage, ar_stage_output, ar_elapsed_s))
        ar_output = self._ar_output(ar_stage_output)

        diffusion_request = self.connector(request, ar_output)
        diffusion_stage_output, diffusion_elapsed_s = self._run_stage(self.diffusion_stage, diffusion_request)
        diffusion_stage_record = self._make_stage_record(
            self.diffusion_stage,
            diffusion_stage_output,
            diffusion_elapsed_s,
        )
        diffusion_stage_record.metadata["source_request_id"] = request.request_id
        trace.stages.append(diffusion_stage_record)

        self.last_trace = trace
        return [self._diffusion_output(diffusion_stage_output)]

    @staticmethod
    def _run_stage(stage, request: OmniRequest) -> tuple[StageOutput, float]:
        prepare_metadata = stage.prepare()
        start = perf_counter()
        output = stage.run(request)
        elapsed_s = perf_counter() - start
        if prepare_metadata:
            output.metadata.update(prepare_metadata)
        return output, elapsed_s

    @staticmethod
    def _make_stage_record(stage, output: StageOutput, elapsed_s: float) -> OmniStageRecord:
        metadata = dict(output.metadata)
        metadata["elapsed_s"] = elapsed_s
        return OmniStageRecord(
            name=stage.name,
            paradigm=stage.paradigm,
            request_id=output.request_id,
            metadata=metadata,
        )

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

    @staticmethod
    def _default_connector(request: OmniRequest, ar_output: ARTextOutput) -> OmniRequest:
        sampling = clone_sampling_params(request.sampling_params)
        return OmniRequest(
            prompt=ar_output.text,
            image=request.image,
            sampling_params=sampling,
            model_paradigm=ModelParadigm.DIFFUSION,
            request_id=request.request_id,
        )
