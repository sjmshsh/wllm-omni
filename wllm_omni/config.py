from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(REPO_ROOT / "models" / "Wan2.2-TI2V-5B-Diffusers")
DEFAULT_IMAGE = str(REPO_ROOT / "assets" / "i2v_input.JPG")
DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, "
    "and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring "
    "the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the "
    "refreshing atmosphere of the seaside."
)
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards, ghosting, double image, "
    "duplicate subject, motion trails, afterimage, frame blending"
)


@dataclass(slots=True)
class EngineConfig:
    model: str = DEFAULT_MODEL
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Wan VAE stays in fp32 by default for decode stability.
    vae_dtype: torch.dtype = torch.float32
    local_files_only: bool = True
    use_cpu_offload: bool = True
    max_num_seqs: int = 2
    prompt_cache_size: int = 8
    image_cache_size: int = 4
    condition_cache_size: int = 4
    enable_profiling: bool = False
    probe_condition_cache: bool = False
