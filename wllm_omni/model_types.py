from __future__ import annotations

import enum


class ModelParadigm(str, enum.Enum):
    DIFFUSION = "diffusion"
    AUTOREGRESSIVE = "ar"
    MULTIMODAL = "multimodal"
    WORLD_MODEL = "world_model"
