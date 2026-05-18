from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from wllm_omni.outputs import OmniOutput
from wllm_omni.sampling_params import OmniSamplingParams


@dataclass(slots=True)
class RunnerState:
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
