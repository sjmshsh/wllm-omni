import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def resize_with_aspect(image: Image.Image, height: int, width: int) -> Image.Image:
    src_ratio = image.height / image.width
    dst_ratio = height / width
    if abs(src_ratio - dst_ratio) < 1e-3:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    target_area = height * width
    fit_height = math.floor(math.sqrt(target_area * src_ratio) / 32) * 32
    fit_width = math.floor(math.sqrt(target_area / src_ratio) / 32) * 32
    return image.resize((fit_width, fit_height), Image.Resampling.LANCZOS)


def save_video(frames: list[Image.Image], output_path: str | Path, fps: int):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(frames):
            frame.save(f"{temp_dir}/frame_{i:04d}.png")
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                f"{temp_dir}/frame_%04d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
        )
