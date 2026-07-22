from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput, IdentityARPipeline
from wllm_omni.worker.utils import (
    ExecutionPhase,
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
)

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class ARState:
    request: OmniRequest
    output: ARTextOutput | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ARExecutor(ModelExecutor):
    """AR executor used by the mini-Omni runtime.

    The executor still runs at request level, but the pipeline underneath now
    exposes prefill/decode/KV-cache boundaries. Streaming is intentionally not
    advertised until tokens are emitted across the scheduler boundary.
    """

    paradigm = ModelParadigm.AUTOREGRESSIVE
    capabilities = frozenset({ExecutorCapability.STEPWISE, ExecutorCapability.KV_CACHE})

    def __init__(self, pipeline: ARPipeline | None = None):
        self.pipeline = pipeline or IdentityARPipeline()

    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        return RequestState(
            req_id=request.request_id,
            sched_req_id=sched_req_id,
            paradigm=self.paradigm,
            payload=ARState(request=request),
        )

    def batch_key(self, state: RequestState) -> tuple:
        return (self.paradigm.value,)

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id for state in states],
            phase=ExecutionPhase.STEP,
            payload=[self._state_payload(state) for state in states],
        )

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"ARExecutor cannot run batch for paradigm={batch.paradigm}.")
        states = self._batch_payload(batch)
        outputs: list[RunnerOutput] = []
        ar_outputs: list[ARTextOutput] = []
        for req_id, state in zip(batch.req_ids, states, strict=True):
            ar_output = self.pipeline.generate(state.request)
            state.output = ar_output
            ar_outputs.append(ar_output)
            outputs.append(RunnerOutput(req_id=req_id, step_index=1, finished=True))
        return ModelForwardOutput(outputs=outputs, payload=ar_outputs)

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        ar_outputs = output.payload if isinstance(output.payload, list) else []
        output_by_req_id = {item.req_id: item for item in output.outputs}
        ar_by_request_id = {item.request_id: item for item in ar_outputs if isinstance(item, ARTextOutput)}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            payload = self._state_payload(state)
            payload.output = ar_by_request_id.get(state.req_id)
            state.step_index = item.step_index or 1
            state.error = item.error
            state.finished = item.finished

    def collect_outputs(
        self,
        states: list[RequestState],
        output: ModelForwardOutput,
    ) -> list[RunnerOutput]:
        results: list[RunnerOutput] = []
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            payload = self._state_payload(state)
            results.append(
                RunnerOutput(
                    req_id=state.sched_req_id,
                    step_index=item.step_index,
                    finished=item.finished,
                    result=payload.output,
                    error=item.error,
                )
            )
        return results

    def release(self, state: RequestState) -> None:
        state.payload = None

    @staticmethod
    def _state_payload(state: RequestState) -> ARState:
        if not isinstance(state.payload, ARState):
            raise TypeError(f"Expected ARState payload, got {type(state.payload).__name__}.")
        return state.payload

    @staticmethod
    def _batch_payload(batch: ForwardBatch) -> list[ARState]:
        if not isinstance(batch.payload, list):
            raise TypeError(f"Expected ARState list payload, got {type(batch.payload).__name__}.")
        for item in batch.payload:
            if not isinstance(item, ARState):
                raise TypeError(f"Expected ARState payload item, got {type(item).__name__}.")
        return batch.payload
