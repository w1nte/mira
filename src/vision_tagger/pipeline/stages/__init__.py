"""Pipeline stages and the default stage factory."""

from __future__ import annotations

from collections.abc import Callable

from ...config import Settings
from ...db.session import Database
from .base import Stage
from .versions import embed_stage_version, faces_stage_version, vlm_stage_version

STAGE_NAMES = ("faces", "vlm", "embed")


def default_stage_factory(db: Database, settings: Settings) -> Callable[[str], Stage]:
    """Create stages lazily (heavy imports only when a stage actually runs)."""

    def factory(name: str) -> Stage:
        if name == "faces":
            from .faces import FacesStage

            return FacesStage(db, settings)
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
        "faces": faces_stage_version(settings),
        "vlm": vlm_stage_version(settings),
        "embed": embed_stage_version(settings),
    }


__all__ = ["STAGE_NAMES", "Stage", "default_stage_factory", "stage_versions"]
