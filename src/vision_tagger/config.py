"""Application settings.

Settings are read (highest priority first) from:

1. keyword arguments passed to :class:`Settings`
2. environment variables prefixed with ``VT_`` (nested with ``__``, e.g. ``VT_VLM__MODEL``)
3. a TOML file (``config.toml`` in the working directory, or the path in ``VT_CONFIG``)
4. the defaults below
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
]


class PathsConfig(BaseModel):
    image_roots: list[Path] = Field(default_factory=list)
    data_dir: Path = Path("data")
    references_dir: Path = Path("references")


class IndexConfig(BaseModel):
    extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    workers: int = 4
    commit_every: int = 200
    follow_symlinks: bool = False
    skip_hidden: bool = True
    # A file is a duplicate if its pHash equals an indexed one AND the dHash differs in at
    # most this many bits (re-encodes / resized copies differ by 0-4 bits).
    dhash_max_distance: int = 5
    # Folder names that are never scanned (Windows system folders on drive roots).
    skip_dirs: list[str] = Field(
        default_factory=lambda: ["$RECYCLE.BIN", "System Volume Information", "$WinREAgent"]
    )


class PipelineConfig(BaseModel):
    batch_size: int = 16
    max_attempts: int = 3
    stages: list[str] = Field(default_factory=lambda: ["faces", "vlm"])
    heartbeat_seconds: float = 5.0


class FacesConfig(BaseModel):
    model: str = "buffalo_l"
    model_root: Path | None = None  # defaults to ~/.insightface
    providers: list[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    det_size: int = 640
    max_image_px: int = 2048
    min_det_score: float = 0.6
    min_face_px: int = 40
    crop_size: int = 160
    # clustering (cosine distance = 1 - cosine similarity)
    cluster_eps: float = 0.5
    cluster_min_samples: int = 3
    split_eps: float = 0.35
    assign_threshold: float = 0.5
    # reference database
    reference_threshold: float = 0.45
    auto_accept_threshold: float = 0.6
    references_per_named_cluster: int = 10


class VlmConfig(BaseModel):
    backend: Literal["ollama", "fake"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5vl:7b"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    max_image_px: int = 1024
    timeout: float = 600.0
    concurrency: int = 1
    keep_alive: str = "10m"
    retries: int = 1
    max_list_items: int = 15
    vocabulary_file: Path | None = None
    synonyms_file: Path | None = None


class DerivedConfig(BaseModel):
    work_path_keywords: list[str] = Field(
        default_factory=lambda: [
            "screenshot",
            "screen shot",
            "bildschirmfoto",
            "scan",
            "work",
            "arbeit",
            "office",
            "invoice",
            "rechnung",
            "receipt",
            "whiteboard",
        ]
    )
    work_yes_threshold: float = 0.6
    work_maybe_threshold: float = 0.35


class ExportConfig(BaseModel):
    mode: Literal["copy", "hardlink", "symlink"] = "copy"


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    page_size: int = 120
    thumb_size: int = 320
    # Extra Host names accepted by the UI (e.g. the machine's LAN name when host="0.0.0.0").
    # Requests for any other host are rejected to block DNS-rebinding attacks.
    allowed_hosts: list[str] = Field(default_factory=list)


_config_file: Path | None = None


def _resolve_config_file() -> Path | None:
    if _config_file is not None:
        return _config_file
    env = os.environ.get("VT_CONFIG")
    if env:
        return Path(env)
    default = Path("config.toml")
    return default if default.exists() else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    faces: FacesConfig = Field(default_factory=FacesConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    derived: DerivedConfig = Field(default_factory=DerivedConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        toml_file = _resolve_config_file()
        if toml_file is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_file))
        return tuple(sources)

    # ---- derived paths -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.paths.data_dir.expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vision_tagger.sqlite3"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache_dir / "thumbs"

    @property
    def face_crops_dir(self) -> Path:
        return self.cache_dir / "faces"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.thumbs_dir, self.face_crops_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings(config_file: Path | None = None, **overrides: object) -> Settings:
    """Load settings, optionally from an explicit TOML file."""
    global _config_file
    if config_file is not None:
        if not config_file.exists():
            raise FileNotFoundError(f"config file not found: {config_file}")
        _config_file = config_file
    return Settings(**overrides)  # type: ignore[arg-type]
