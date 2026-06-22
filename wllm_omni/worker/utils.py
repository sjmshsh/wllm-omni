from __future__ import annotations

from dataclasses import dataclass, field
import enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm

if TYPE_CHECKING:
    import torch
    from PIL import Image

    from wllm_omni.outputs import OmniOutput
    from wllm_omni.sampling_params import OmniSamplingParams


class ExecutionPhase(str, enum.Enum):
    PREPARE = "prepare"
    STEP = "step"
    FINALIZE = "finalize"


class ExecutorCapability(str, enum.Enum):
    STEPWISE = "stepwise"
    CACHEABLE_PREPARE = "cacheable_prepare"
    MULTIMODAL_INPUT = "multimodal_input"
    STREAMING = "streaming"
    KV_CACHE = "kv_cache"


@dataclass(slots=True)
class RequestState:
    req_id: str
    sched_req_id: str
    paradigm: ModelParadigm
    payload: Any
    step_index: int = 0
    initialized: bool = False
    finished: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForwardBatch:
    paradigm: ModelParadigm
    req_ids: list[str]
    phase: ExecutionPhase
    payload: Any = None

    @property
    def mode(self) -> str:
        """Compatibility view for older debug code that printed batch.mode."""
        return self.phase.value


@dataclass(slots=True)
class RunnerState:
    """Diffusion step state used by the current Wan executor."""

    req_id: str
    sampling: OmniSamplingParams
    prompt: str
    image: str | Path | Image.Image
    negative_prompt: str
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    step_index: int = 0
    scheduler: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        if self.timesteps is None:
            return 0
        return int(self.timesteps.shape[0])

    @property
    def current_timestep(self) -> torch.Tensor | None:
        if self.timesteps is None or self.step_index >= self.total_steps:
            return None
        return self.timesteps[self.step_index]

    @property
    def denoise_completed(self) -> bool:
        return self.total_steps > 0 and self.step_index >= self.total_steps


@dataclass(slots=True)
class RunnerOutput:
    req_id: str
    step_index: int | None = None
    finished: bool = False
    result: OmniOutput | None = None
    error: str | None = None


@dataclass(slots=True)
class ModelForwardOutput:
    outputs: list[RunnerOutput] = field(default_factory=list)
    payload: Any = None


@dataclass(slots=True)
class RunnerBatchOutput:
    outputs: list[RunnerOutput] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.outputs) == 0

    def to_single(self) -> RunnerOutput:
        if len(self.outputs) != 1:
            raise ValueError(f"Expected exactly one runner output, got {len(self.outputs)}.")
        return self.outputs[0]
