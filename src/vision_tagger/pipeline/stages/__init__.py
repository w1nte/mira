"""Pipeline stages and the default stage factory."""

from __future__ import annotations

from collections.abc import Callable

from ...config import Settings
from ...db.session import Database
from .base import Stage
from .versions import (
    ages_stage_version,
    animals_stage_version,
    embed_stage_version,
    faces_stage_version,
    nsfw_stage_version,
    visual_stage_version,
    vlm_stage_version,
)

STAGE_NAMES = ("nsfw", "visual", "faces", "ages", "animals", "vlm", "embed")


def default_stage_factory(db: Database, settings: Settings) -> Callable[[str], Stage]:
    """Create stages lazily (heavy imports only when a stage actually runs)."""

    def factory(name: str) -> Stage:
        if name == "nsfw":
            from .nsfw import NsfwStage

            return NsfwStage(db, settings)
        if name == "visual":
            from .visual import VisualStage

            return VisualStage(db, settings)
        if name == "faces":
            from .faces import FacesStage

            return FacesStage(db, settings)
        if name == "ages":
            from .ages import AgesStage

            return AgesStage(db, settings)
        if name == "animals":
            from .animals import AnimalsStage

            return AnimalsStage(db, settings)
        if name == "vlm":
            from .vlm import VlmStage

            return VlmStage(db, settings)
        if name == "embed":
            from .embed import EmbedStage

            return EmbedStage(db, settings)
        raise KeyError(f"unknown stage {name!r}")

    return factory


def stage_versions(settings: Settings) -> dict[str, str]:
    """Current version string per stage, without loading any model."""
    return {
        "nsfw": nsfw_stage_version(settings),
        "visual": visual_stage_version(settings),
        "faces": faces_stage_version(settings),
        "ages": ages_stage_version(settings),
        "animals": animals_stage_version(settings),
        "vlm": vlm_stage_version(settings),
        "embed": embed_stage_version(settings),
    }


__all__ = ["STAGE_NAMES", "Stage", "default_stage_factory", "stage_versions"]
