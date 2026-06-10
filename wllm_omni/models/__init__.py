from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wllm_omni.request import OmniRequest
from wllm_omni.worker.utils import (
    ForwardBatch,
    ModelForwardOutput,
    ModelParadigm,
    RequestState,
    RunnerOutput,
)

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
