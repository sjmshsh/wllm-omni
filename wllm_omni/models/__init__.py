from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm
from wllm_omni.worker.utils import (
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
)

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest

STEP_EXECUTION_METHODS = ("prepare_encode", "denoise_step", "step_scheduler", "post_decode")


def supports_step_execution(pipeline: Any) -> bool:
    return all(callable(getattr(pipeline, name, None)) for name in STEP_EXECUTION_METHODS)


class ModelExecutor(ABC):
    """Executor contract used by ModelRunner.

    The runner owns request lifecycle and batching orchestration. Concrete
    executors own model-family details such as KV cache, diffusion latents,
    multimodal feature caches, or world-model rollout state.
    """

    paradigm: ModelParadigm
    capabilities: frozenset[ExecutorCapability] = frozenset()

    @abstractmethod
    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        pass

    @abstractmethod
    def batch_key(self, state: RequestState) -> tuple:
        pass

    @abstractmethod
    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        pass

    @abstractmethod
    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        pass

    @abstractmethod
    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        pass

    @abstractmethod
    def collect_outputs(
        self,
        states: list[RequestState],
        output: ModelForwardOutput,
    ) -> list[RunnerOutput]:
        pass

    def release(self, state: RequestState) -> None:
        pass


class ExecutorRegistry:
    """Paradigm-indexed executor registry used by the generic runner."""

    def __init__(self, executors: list[ModelExecutor]):
        if not executors:
            raise ValueError("ExecutorRegistry requires at least one executor.")

        self._executors: dict[ModelParadigm, ModelExecutor] = {}
        for executor in executors:
            if executor.paradigm in self._executors:
                raise ValueError(f"Duplicate executor registered for paradigm={executor.paradigm}.")
            self._executors[executor.paradigm] = executor
        self.default_executor = executors[0]

    @property
    def executors(self) -> dict[ModelParadigm, ModelExecutor]:
        return dict(self._executors)

    def resolve_request(self, request: OmniRequest) -> ModelExecutor:
        paradigm = getattr(request, "model_paradigm", None)
        if paradigm is None:
            return self.default_executor
        return self.resolve_paradigm(paradigm)

    def resolve_state(self, state: RequestState) -> ModelExecutor:
        return self.resolve_paradigm(state.paradigm)

    def resolve_paradigm(self, paradigm: ModelParadigm | str) -> ModelExecutor:
        if isinstance(paradigm, str):
            try:
                paradigm = ModelParadigm(paradigm)
            except ValueError as exc:
                known = ", ".join(item.value for item in self._executors)
                raise ValueError(f"Unknown model paradigm={paradigm!r}; registered paradigms: {known}.") from exc
        executor = self._executors.get(paradigm)
        if executor is None:
            known = ", ".join(item.value for item in self._executors)
            raise ValueError(f"No executor registered for paradigm={paradigm.value}; registered paradigms: {known}.")
        return executor
