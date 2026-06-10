from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wllm_omni.models import ModelExecutor
from wllm_omni.request import OmniRequest
from wllm_omni.worker.utils import (
    ForwardBatch,
    ModelForwardOutput,
    ModelParadigm,
    RequestState,
    RunnerOutput,
)


@dataclass(slots=True)
class ARState:
    request: OmniRequest
    input_ids: Any = None
    positions: Any = None
    kv_cache_handle: Any = None
    sampling_state: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultimodalState:
    request: OmniRequest
    text_inputs: Any = None
    media_inputs: Any = None
    media_features: Any = None
    feature_cache_handle: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorldModelState:
    request: OmniRequest
    observations: Any = None
    action_history: Any = None
    latent_state: Any = None
    rollout_state: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class _UnsupportedExecutor(ModelExecutor):
    paradigm: ModelParadigm
    state_cls: type

    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        return RequestState(
            req_id=request.request_id,
            sched_req_id=sched_req_id,
            paradigm=self.paradigm,
            payload=self.state_cls(request=request),
        )

    def batch_key(self, state: RequestState) -> tuple:
        return (self.paradigm.value, state.sched_req_id)

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id for state in states],
            mode="unsupported",
            payload=[state.payload for state in states],
        )

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        error = f"{self.__class__.__name__} is a registration skeleton and has no model implementation yet."
        return ModelForwardOutput(
            outputs=[RunnerOutput(req_id=req_id, finished=True, error=error) for req_id in batch.req_ids]
        )

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            state.error = item.error
            state.finished = item.finished

    def collect_outputs(
        self,
        states: list[RequestState],
        output: ModelForwardOutput,
    ) -> list[RunnerOutput]:
        return output.outputs

    def release(self, state: RequestState) -> None:
        state.payload = None


class ARExecutor(_UnsupportedExecutor):
    paradigm = ModelParadigm.AUTOREGRESSIVE
    state_cls = ARState


class MultimodalExecutor(_UnsupportedExecutor):
    paradigm = ModelParadigm.MULTIMODAL
    state_cls = MultimodalState


class WorldModelExecutor(_UnsupportedExecutor):
    paradigm = ModelParadigm.WORLD_MODEL
    state_cls = WorldModelState
