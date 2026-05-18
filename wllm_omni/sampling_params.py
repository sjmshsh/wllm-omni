from dataclasses import dataclass

from wllm_omni.config import DEFAULT_NEGATIVE_PROMPT


@dataclass(slots=True)
class OmniSamplingParams:
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    flow_shift: float
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 42
    fps: int = 16


PRESETS = {
    "quality": OmniSamplingParams(800, 576, 17, 12, 5.0, 3.0, fps=16),
}


def clone_sampling_params(params: OmniSamplingParams) -> OmniSamplingParams:
    return OmniSamplingParams(
        height=params.height,
        width=params.width,
        num_frames=params.num_frames,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        flow_shift=params.flow_shift,
        negative_prompt=params.negative_prompt,
        seed=params.seed,
        fps=params.fps,
    )
