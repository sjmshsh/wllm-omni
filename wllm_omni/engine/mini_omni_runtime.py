from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from wllm_omni.config import EngineConfig
from wllm_omni.engine.omni_engine import OmniEngine
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_executor import ARExecutor
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput, TransformersARPipeline
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import clone_sampling_params


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

    The runtime lives above concrete model executors. It keeps AR and diffusion
    independent, then connects them through an explicit bridge function.
    """

    def __init__(
        self,
        config: EngineConfig,
        connector: Callable[[OmniRequest, ARTextOutput], OmniRequest] | None = None,
        ar_pipeline: ARPipeline | None = None,
    ):
        self.config = config
        self.ar_executor = ARExecutor(ar_pipeline or self._make_ar_pipeline(config))
        self.diffusion_engine: OmniEngine | None = None
        self.connector = connector or self._default_connector
        self.last_trace: MiniOmniTrace | None = None

    def generate_ar(self, request: OmniRequest) -> ARTextOutput:
        ar_output, elapsed_s = self._run_ar_stage(request)
        self.last_trace = MiniOmniTrace(
            request_id=request.request_id,
            stages=[self._make_ar_stage_record(request, ar_output, elapsed_s)],
        )
        return ar_output

    def generate(self, request: OmniRequest) -> list[OmniOutput]:
        trace = MiniOmniTrace(request_id=request.request_id)
        ar_output, elapsed_s = self._run_ar_stage(request)
        trace.stages.append(self._make_ar_stage_record(request, ar_output, elapsed_s))

        diffusion_request = self.connector(request, ar_output)
        trace.stages.append(
            OmniStageRecord(
                name="diffusion.wan22_i2v",
                paradigm=ModelParadigm.DIFFUSION,
                request_id=diffusion_request.request_id,
                metadata={
                    "source_request_id": request.request_id,
                    "bridge": "ar_text_to_diffusion_prompt",
                },
            )
        )

        outputs = self._diffusion_engine().generate(diffusion_request)
        self.last_trace = trace
        return outputs

    def _run_ar_stage(self, request: OmniRequest) -> tuple[ARTextOutput, float]:
        start = perf_counter()
        ar_output = self.ar_executor.generate_text(request)
        return ar_output, perf_counter() - start

    @staticmethod
    def _make_ar_stage_record(request: OmniRequest, ar_output: ARTextOutput, elapsed_s: float) -> OmniStageRecord:
        return OmniStageRecord(
            name="ar.prompt_bridge",
            paradigm=ModelParadigm.AUTOREGRESSIVE,
            request_id=request.request_id,
            metadata={
                "elapsed_s": elapsed_s,
                "mode": ar_output.metadata.get("mode"),
                "model": ar_output.metadata.get("model"),
                "input_tokens": ar_output.metadata.get("input_tokens"),
                "output_tokens": ar_output.metadata.get("token_count", len(ar_output.tokens)),
            },
        )

    def _diffusion_engine(self) -> OmniEngine:
        if self.diffusion_engine is None:
            self.diffusion_engine = OmniEngine(self.config)
        return self.diffusion_engine

    @staticmethod
    def _make_ar_pipeline(config: EngineConfig) -> ARPipeline | None:
        if config.ar_model is None:
            return None
        return TransformersARPipeline(
            config.ar_model,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            max_new_tokens=config.ar_max_new_tokens,
        )

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
