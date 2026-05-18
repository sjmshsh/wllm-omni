from pathlib import Path

from PIL import Image

from nanovllm_omni.config import DEFAULT_IMAGE, DEFAULT_MODEL, DEFAULT_PROMPT, EngineConfig
from nanovllm_omni.engine.omni_engine import OmniEngine
from nanovllm_omni.outputs import OmniOutput
from nanovllm_omni.request import OmniRequest
from nanovllm_omni.sampling_params import PRESETS, OmniSamplingParams, clone_sampling_params
from nanovllm_omni.utils import save_video


class OmniLLM:

    def __init__(self, model: str = DEFAULT_MODEL, **kwargs):
        self.config = EngineConfig(model=model, **kwargs)
        self.engine = OmniEngine(self.config)

    @staticmethod
    def preset(name: str) -> OmniSamplingParams:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}")
        return clone_sampling_params(PRESETS[name])

    def generate(
        self,
        image: str | Path | Image.Image = DEFAULT_IMAGE,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str | None = None,
        sampling_params: OmniSamplingParams | None = None,
    ) -> OmniOutput:
        sampling_params = clone_sampling_params(PRESETS["quality"]) if sampling_params is None else sampling_params
        if negative_prompt is not None:
            sampling_params.negative_prompt = negative_prompt
        request = OmniRequest(prompt=prompt, image=image, sampling_params=sampling_params)
        outputs = self.engine.generate(request)
        if not outputs:
            raise RuntimeError("Generation finished without output. Check runner or scheduler logs for the failing step.")
        return outputs[0]

    def save(self, output: OmniOutput, output_path: str | Path):
        save_video(output.frames, output_path, output.fps)
