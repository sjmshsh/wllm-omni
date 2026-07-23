from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from wllm_omni.config import EngineConfig
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.engine.diffusion_engine import DiffusionEngine
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class StageOutput:
    request_id: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    name: str
    paradigm: ModelParadigm

    def prepare(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def run(self, request: OmniRequest) -> StageOutput:
        pass


class ARStage(Stage):
    name = "ar.prompt_bridge"
    paradigm = ModelParadigm.AUTOREGRESSIVE

    def __init__(self, config: EngineConfig, pipeline: ARPipeline | None = None):
        self.engine = AREngine(config, pipeline=pipeline)

    def run(self, request: OmniRequest) -> StageOutput:
        ar_output = self.engine.generate(request)
        return StageOutput(
            request_id=request.request_id,
            data=ar_output,
            metadata={
                "mode": ar_output.metadata.get("mode"),
                "model": ar_output.metadata.get("model"),
                "input_tokens": ar_output.metadata.get("input_tokens"),
                "prefill_tokens": ar_output.metadata.get("prefill_tokens"),
                "output_tokens": ar_output.metadata.get("output_tokens", ar_output.metadata.get("token_count", len(ar_output.tokens))),
                "generated_tokens": ar_output.metadata.get("generated_tokens", ar_output.metadata.get("token_count", len(ar_output.tokens))),
                "prefill_elapsed_s": ar_output.metadata.get("prefill_elapsed_s"),
                "decode_elapsed_s": ar_output.metadata.get("decode_elapsed_s"),
                "ttft_s": ar_output.metadata.get("ttft_s"),
                "decode_model_steps": ar_output.metadata.get("decode_model_steps"),
                "decode_model_calls": ar_output.metadata.get("decode_model_calls", ar_output.metadata.get("decode_model_steps")),
                "decode_scheduler_steps": ar_output.metadata.get("decode_scheduler_steps"),
                "scheduler_steps": ar_output.metadata.get("scheduler_steps"),
                "prefill_steps": ar_output.metadata.get("prefill_steps"),
                "decode_step_mean_ms": ar_output.metadata.get("decode_step_mean_ms"),
                "decode_step_max_ms": ar_output.metadata.get("decode_step_max_ms"),
                "stopped_by_eos": ar_output.metadata.get("stopped_by_eos"),
                "stop_reason": ar_output.metadata.get("stop_reason"),
                "kv_cache_enabled": ar_output.metadata.get("kv_cache_enabled"),
                "kv_cache_type": ar_output.metadata.get("kv_cache_type"),
            },
        )


class DiffusionStage(Stage):
    name = "diffusion.wan22_i2v"
    paradigm = ModelParadigm.DIFFUSION

    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine: DiffusionEngine | None = None

    def prepare(self) -> dict[str, Any]:
        if self.engine is not None:
            return {"load_elapsed_s": 0.0, "load_was_cold": False}
        start = perf_counter()
        self.engine = DiffusionEngine(self.config)
        return {"load_elapsed_s": perf_counter() - start, "load_was_cold": True}

    def run(self, request: OmniRequest) -> StageOutput:
        outputs = self._engine().generate(request)
        if not outputs:
            raise RuntimeError("Diffusion stage finished without output.")
        return StageOutput(
            request_id=request.request_id,
            data=outputs[0],
        )

    def _engine(self) -> DiffusionEngine:
        if self.engine is None:
            self.engine = DiffusionEngine(self.config)
        return self.engine
