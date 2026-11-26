"""Stage version strings (computed without loading any model)."""

from __future__ import annotations

from ...config import Settings


def faces_stage_version(settings: Settings) -> str:
    cfg = settings.faces
    return f"{cfg.model}-det{cfg.det_size}-px{cfg.max_image_px}-v1"


def ages_stage_version(settings: Settings) -> str:
    from ...faces.agenet import MODEL_VERSION

    mode = settings.faces.age_model
    model = settings.faces.model if mode == "insightface" else f"{mode}:{MODEL_VERSION}"
    return f"{model}-v1"


def vlm_stage_version(settings: Settings, model: str | None = None) -> str:
    from ...vlm.prompt import prompt_version
    from ...vlm.schema import Vocabulary

    if model is None:
        model = "fake" if settings.vlm.backend == "fake" else settings.vlm.model
    return f"{model}|{prompt_version(Vocabulary.load(settings.vlm.vocabulary_file))}"


def embed_stage_version(settings: Settings, fake: bool | None = None) -> str:
    from ...search.document import DOC_VERSION

    if fake is None:
        fake = settings.vlm.backend == "fake"
    model = "fake" if fake else settings.search.embed_model
    return f"{model}|doc{DOC_VERSION}"
