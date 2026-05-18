from dataclasses import dataclass, field

from PIL import Image


@dataclass(slots=True)
class OmniOutput:
    request_id: str
    frames: list[Image.Image]
    width: int
    height: int
    fps: int
    stage_durations: dict[str, float] = field(default_factory=dict)
    scheduler: str = "euler"
