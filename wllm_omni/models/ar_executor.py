from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor
from wllm_omni.models.ar_pipeline import (
    ARDecodeOutput,
    ARPipeline,
    ARPrefillOutput,
    ARStepOutput,
    ARTextOutput,
    IdentityARPipeline,
)
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
    prefill: ARPrefillOutput | None = None
    decode: ARDecodeOutput | None = None
    output: ARTextOutput | None = None
    last_step_output: ARStepOutput | None = None
    scheduler_steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_prefill(self) -> bool:
        return self.prefill is not None

    @property
    def decode_finished(self) -> bool:
        return self.decode is not None and self.decode.finished


class ARExecutor(ModelExecutor):
    """Scheduler-visible AR executor for the mini-Omni runtime.

    The executor keeps AR KV/decode state in RequestState and advances one
    prefill/decode unit per scheduler iteration. This is still single-process
    and greedy, but it exposes the same boundary later needed for streaming,
    KV cache management, and decode batching.
    """

    paradigm = ModelParadigm.AUTOREGRESSIVE
    capabilities = frozenset({
        ExecutorCapability.STEPWISE,
        ExecutorCapability.KV_CACHE,
        ExecutorCapability.STREAMING,
    })

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
        return (self.paradigm.value, self._phase_for_state(self._state_payload(state)).value)

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        if not states:
            raise ValueError("ARExecutor cannot build an empty forward batch.")
        payloads = [self._state_payload(state) for state in states]
        phase = self._phase_for_state(payloads[0])
        for payload in payloads:
            item_phase = self._phase_for_state(payload)
            if item_phase != phase:
                raise ValueError(f"Mixed AR phases in one batch: {phase} and {item_phase}.")
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id for state in states],
            phase=phase,
            payload=payloads,
        )

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"ARExecutor cannot run batch for paradigm={batch.paradigm}.")
        states = self._batch_payload(batch)
        outputs: list[RunnerOutput] = []
        for req_id, state in zip(batch.req_ids, states, strict=True):
            if batch.phase == ExecutionPhase.PREPARE:
                outputs.append(self._run_prefill(req_id, state))
            elif batch.phase == ExecutionPhase.STEP:
                outputs.append(self._run_decode_step(req_id, state))
            elif batch.phase == ExecutionPhase.FINALIZE:
                outputs.append(self._run_finalize(req_id, state))
            else:
                outputs.append(RunnerOutput(req_id=req_id, finished=True, error=f"Unsupported AR phase: {batch.phase}"))
        return ModelForwardOutput(outputs=outputs, payload=states)

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            payload = self._state_payload(state)
            state.initialized = payload.has_prefill
            state.step_index = item.step_index or payload.scheduler_steps
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
            if payload.last_step_output is not None:
                results.append(
                    RunnerOutput(
                        req_id=state.sched_req_id,
                        step_index=item.step_index,
                        finished=False,
                        result=payload.last_step_output,
                        error=item.error,
                    )
                )
            results.append(
                RunnerOutput(
                    req_id=state.sched_req_id,
                    step_index=item.step_index,
                    finished=item.finished,
                    result=payload.output if item.finished else None,
                    error=item.error,
                )
            )
        return results

    def release(self, state: RequestState) -> None:
        state.payload = None

    def _run_prefill(self, req_id: str, state: ARState) -> RunnerOutput:
        state.scheduler_steps += 1
        state.prefill_steps += 1
        state.prefill = self.pipeline.prefill(state.request)
        state.decode = self.pipeline.init_decode()
        state.last_step_output = None
        return RunnerOutput(req_id=req_id, step_index=state.scheduler_steps, finished=False)

    def _run_decode_step(self, req_id: str, state: ARState) -> RunnerOutput:
        state.scheduler_steps += 1
        state.decode_steps += 1
        if state.prefill is None:
            return RunnerOutput(req_id=req_id, step_index=state.scheduler_steps, finished=True, error="AR decode before prefill.")
        if state.decode is None:
            state.decode = self.pipeline.init_decode()
        before_len = len(state.decode.token_ids)
        self.pipeline.decode_step(state.request, state.prefill, state.decode)
        state.last_step_output = self._make_step_output(state, before_len)
        if state.decode.finished:
            state.output = self._finalize_output(state)
            return RunnerOutput(req_id=req_id, step_index=state.scheduler_steps, finished=True)
        return RunnerOutput(req_id=req_id, step_index=state.scheduler_steps, finished=False)

    def _run_finalize(self, req_id: str, state: ARState) -> RunnerOutput:
        state.scheduler_steps += 1
        state.output = self._finalize_output(state)
        return RunnerOutput(req_id=req_id, step_index=state.scheduler_steps, finished=True)

    def _finalize_output(self, state: ARState) -> ARTextOutput:
        if state.prefill is None or state.decode is None:
            raise RuntimeError("Cannot finalize AR output before prefill/decode.")
        output = self.pipeline.finalize(state.request, state.prefill, state.decode)
        output.metadata.update({
            "scheduler_steps": state.scheduler_steps,
            "prefill_steps": state.prefill_steps,
            "generated_tokens": len(output.token_ids),
            "output_tokens": len(output.token_ids),
            "decode_model_calls": output.metadata.get("decode_model_calls", output.metadata.get("decode_model_steps")),
            "decode_scheduler_steps": state.decode_steps,
        })
        return output

    def _make_step_output(self, state: ARState, before_len: int) -> ARStepOutput | None:
        if state.decode is None:
            return None
        token_id: int | None = None
        token = ""
        text_delta = ""
        if len(state.decode.token_ids) > before_len:
            token_id = state.decode.token_ids[-1]
            token = state.decode.tokens[-1] if state.decode.tokens else ""
            if hasattr(self.pipeline, "tokenizer"):
                previous_text = self.pipeline.tokenizer.decode(
                    state.decode.token_ids[:before_len],
                    skip_special_tokens=True,
                )
            else:
                previous_text = ""
            current_text = state.decode.text
            text_delta = current_text[len(previous_text):] if current_text.startswith(previous_text) else token
        return ARStepOutput(
            request_id=state.request.request_id,
            token_id=token_id,
            token=token,
            text_delta=text_delta,
            step_index=state.scheduler_steps,
            finished=state.decode.finished,
            stop_reason=state.decode.stop_reason,
            metadata={
                "decode_scheduler_steps": state.decode_steps,
                "generated_tokens": len(state.decode.token_ids),
                "decode_model_calls": len(state.decode.step_elapsed_s),
            },
        )

    @staticmethod
    def _phase_for_state(state: ARState) -> ExecutionPhase:
        if state.output is not None:
            return ExecutionPhase.FINALIZE
        if state.prefill is None:
            return ExecutionPhase.PREPARE
        if state.decode is not None and state.decode.finished:
            return ExecutionPhase.FINALIZE
        return ExecutionPhase.STEP

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
