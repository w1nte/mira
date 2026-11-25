"""Stage version strings (computed without loading any model)."""

from __future__ import annotations

from ...config import Settings


def faces_stage_version(settings: Settings) -> str:
    cfg = settings.faces
    return f"{cfg.model}-det{cfg.det_size}-px{cfg.max_image_px}-v1"


def vlm_stage_version(settings: Settings, model: str | None = None) -> str:
    from ...vlm.prompt import prompt_version
    from ...vlm.schema import Vocabulary

    if model is None:
        model = "fake" if settings.vlm.backend == "fake" else settings.vlm.model
    return f"{model}|{prompt_version(Vocabulary.load(settings.vlm.vocabulary_file))}"
