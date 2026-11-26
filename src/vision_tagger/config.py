"""Application settings.

Settings are read (highest priority first) from:

1. keyword arguments passed to :class:`Settings`
2. environment variables prefixed with ``VT_`` (nested with ``__``, e.g. ``VT_VLM__MODEL``)
3. a TOML file (``config.toml`` in the working directory, or the path in ``VT_CONFIG``),
   brought up to date by :mod:`vision_tagger.config_migrate` when it is older than the app
4. the defaults below
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .config_migrate import KEY as CONFIG_KEY
from .config_migrate import ConfigMigration, load_config

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
    # Write gallery thumbnails while indexing (from the image decoded for hashing anyway) and
    # create missing ones for files indexed earlier, so the gallery never waits for them.
    thumbnails: bool = True


#: every stage, in pipeline order (``pipeline.stages`` defaults to all of them). "visual" comes
#: first: it is the fastest stage and makes the search find photos by what they show soonest.
DEFAULT_STAGES = ("visual", "faces", "ages", "animals", "vlm", "embed")


class PipelineConfig(BaseModel):
    batch_size: int = 16
    max_attempts: int = 3
    stages: list[str] = Field(default_factory=lambda: list(DEFAULT_STAGES))
    heartbeat_seconds: float = 5.0
    # closing the app pauses a running pipeline; the next start continues paused pipelines
    resume_on_start: bool = True

    @field_validator("stages")
    @classmethod
    def _ages_with_faces(cls, stages: list[str]) -> list[str]:
        # "ages" re-estimates face ages of faces found before; it belongs to the faces stage
        if "faces" in stages and "ages" not in stages:
            stages = list(stages)
            stages.insert(stages.index("faces") + 1, "ages")
        return stages


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
    # face ages: "blend" = AgeNet (children, teens) combined with InsightFace (adults, old age)
    age_model: Literal["blend", "agenet", "insightface"] = "blend"
    age_model_dir: Path | None = None  # defaults to <model_root>/models/agegender_faceapi
    # clustering (cosine distance = 1 - cosine similarity)
    cluster_eps: float = 0.5
    cluster_min_samples: int = 3
    split_eps: float = 0.35
    assign_threshold: float = 0.5
    # reference database
    reference_threshold: float = 0.45
    auto_accept_threshold: float = 0.6
    references_per_named_cluster: int = 10


class AnimalsConfig(BaseModel):
    """Animal detection and pets (see ``animals``). Runs on ``faces.providers``.

    The thresholds were chosen on real photos (9 dogs and cats with 5-6 photos each, plus 60
    photos of 10 breeds); see the README section "Animals and pets".
    """

    model_dir: Path | None = None  # defaults to <faces.model_root>/models/animals
    max_image_px: int = 1600  # photos are downscaled before detection and embedding
    # detections weaker or smaller than this are dropped (short side, original pixels)
    min_score: float = 0.5
    min_box_px: int = 32
    nms_iou: float = 0.45
    # weaker or smaller animals are kept, but not grouped automatically
    group_min_score: float = 0.6
    group_min_px: int = 64
    # grouping: cosine distance (1 - similarity) between photos of the same animal
    cluster_eps: float = 0.3
    cluster_min_samples: int = 2
    # similarity to a pet's average to join it (named and edited pets keep their photos)
    assign_threshold: float = 0.7
    # species the detector mixes up on pets (a chow chow as a bear, a poodle as a sheep) are
    # grouped together; the pet's species is what most of its photos say
    species_pools: list[list[str]] = Field(
        default_factory=lambda: [["cat", "dog", "bear", "sheep"]]
    )
    crop_size: int = 160  # pictures of the animals in the web UI
    crop_margin: float = 0.1


class VisualConfig(BaseModel):
    """Visual search (see ``visual``): the ``visual`` stage embeds every photo, the search
    compares the request with it. SigLIP 2 (Apache-2.0), downloaded once (970 MB)."""

    # "siglip2-base-fast": three times as fast on the CPU, finds a little less
    model: Literal["siglip2-base", "siglip2-base-fast"] = "siglip2-base"
    model_dir: Path | None = None  # defaults to <faces.model_root>/models/visual
    providers: list[str] | None = None  # defaults to faces.providers (CUDA, else CPU)
    batch_size: int = 16  # photos per model call


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
    # A pipeline run that needs a model Ollama does not have (vlm.model, search.embed_model)
    # lets Ollama download it first. Off: the run stops and asks for `ollama pull`. The web UI
    # and `vision-tagger models pull` download models on request either way.
    auto_pull: bool = True


class SearchWeights(BaseModel):
    # what the photo shows (visual stage), its description (search index), the properties and
    # literal words; a photo is scored by the parts it has (no description yet: the others)
    visual: float = 0.45
    semantic: float = 0.25
    properties: float = 0.20
    keywords: float = 0.10


class SearchConfig(BaseModel):
    # LLM used to interpret queries / re-rank results; defaults to vlm.model (text-only use)
    model: str | None = None
    embed_model: str = "nomic-embed-text"
    # nomic-embed-text expects task prefixes; set both to "" for models that don't (bge-m3)
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    embed_batch: int = 16
    result_limit: int = 200
    rerank_top: int = 40
    rerank_batch: int = 20
    weights: SearchWeights = Field(default_factory=SearchWeights)


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
    # XMP sidecar names: "append" = photo.jpg.xmp (digiKam, darktable, Immich, PhotoPrism),
    # "replace" = photo.xmp (Lightroom style; two photos named alike share one then)
    sidecar_naming: Literal["append", "replace"] = "append"


class GeoConfig(BaseModel):
    """GPS places and the map (see ``geo.places``)."""

    # find places at the end of every index run (also: `vision-tagger places`, map page)
    auto: bool = True
    # photos closer than this (km) end up in the same place; larger groups are split
    cluster_km: float = 0.75
    max_place_km: float = 5.0
    # groups with the same automatic name this close (km, centre to centre) are one place: a
    # town and its outskirts; same-named towns farther apart get unique names instead
    merge_km: float = 25.0
    # offline names come from GeoNames; the data (3 MB) is downloaded once into
    # <data_dir>/models/geo. A local GeoNames dump (cities500.zip, cities1000.txt, ...) can be
    # used instead.
    download_gazetteer: bool = True
    cities_file: Path | None = None
    # Optional OpenStreetMap names (e.g. "Prenzlauer Berg, Berlin") from Nominatim. Off by
    # default: it sends the (rounded) position of every place to that server.
    nominatim: bool = False
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    language: str = "en"  # language of the OpenStreetMap names
    # Map background. Tile requests tell the tile server which area you look at; point this
    # at your own tile server, or set it to "" for a map without background.
    tile_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    tile_attribution: str = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )
    tile_max_zoom: int = 19


class RemoteConfig(BaseModel):
    """Remote mode: the app runs on a server; photos are uploaded through the web UI."""

    upload_dir: Path | None = None  # where uploads are stored (required in remote mode)
    auto_process: bool = True  # index uploads and run the pipeline automatically
    chunk_mb: int = 8  # upload chunk size
    max_file_gb: float = 50.0  # largest single upload (a ZIP can be big)
    max_zip_gb: float = 200.0  # largest total size extracted from one ZIP


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    page_size: int = 120
    thumb_size: int = 320
    # Extra Host names accepted by the UI (e.g. the machine's LAN name when host="0.0.0.0").
    # Requests for any other host are rejected to block DNS-rebinding attacks.
    allowed_hosts: list[str] = Field(default_factory=list)
    # Optional login password (strongly recommended in remote mode). Set it with the
    # environment variable VT_WEB__PASSWORD rather than in a file.
    password: str = ""


class UpdatesConfig(BaseModel):
    """Looking for new releases on GitHub (see ``updates``). Nothing is downloaded by itself."""

    # "auto": only the installed desktop app looks (once a day); "on": also a source checkout,
    # the server or Docker; "off": never
    check: Literal["auto", "on", "off"] = "auto"
    prereleases: bool = False  # also tell about release candidates and betas
    # internal: where the GitHub API is (tests and trial runs point it at a fake server)
    api_url: str = "https://api.github.com"


_config_file: Path | None = None
#: what loading the config file last did (shown by `vision-tagger migrate`)
last_config_migration: ConfigMigration | None = None


def _load_toml(path: Path) -> dict[str, Any]:
    global last_config_migration
    data, result = load_config(path)
    if result.changed:
        last_config_migration = result
    data.pop(CONFIG_KEY, None)
    return data


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

    # "local": photos are in folders on this computer (pick them on the Import page).
    # "remote": the app runs on a server; photos are uploaded and stored in remote.upload_dir.
    mode: Literal["local", "remote"] = "local"
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    faces: FacesConfig = Field(default_factory=FacesConfig)
    animals: AnimalsConfig = Field(default_factory=AnimalsConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    derived: DerivedConfig = Field(default_factory=DerivedConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    updates: UpdatesConfig = Field(default_factory=UpdatesConfig)

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
            # older config files are brought up to date first (config_migrate)
            sources.append(InitSettingsSource(settings_cls, init_kwargs=_load_toml(toml_file)))
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

    @property
    def animal_crops_dir(self) -> Path:
        return self.cache_dir / "animals"

    @property
    def remote_mode(self) -> bool:
        return self.mode == "remote"

    @property
    def upload_dir(self) -> Path:
        """Where uploaded photos are stored (remote mode)."""
        if self.remote.upload_dir is None:
            raise ValueError(
                "Remote mode needs a folder for the uploaded photos: set remote.upload_dir "
                "(or VT_REMOTE__UPLOAD_DIR, or `serve --remote --upload-dir PATH`)."
            )
        return self.remote.upload_dir.expanduser().resolve()

    @property
    def uploads_tmp_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def exports_dir(self) -> Path:
        """Export target in remote mode (remote users must not write anywhere on the server)."""
        return self.data_dir / "exports"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.thumbs_dir, self.face_crops_dir, self.animal_crops_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.remote_mode:
            for d in (self.upload_dir, self.uploads_tmp_dir, self.exports_dir):
                d.mkdir(parents=True, exist_ok=True)


def load_settings(config_file: Path | None = None, **overrides: object) -> Settings:
    """Load settings, optionally from an explicit TOML file."""
    global _config_file
    if config_file is not None:
        if not config_file.exists():
            raise FileNotFoundError(f"config file not found: {config_file}")
        _config_file = config_file
    return Settings(**overrides)  # type: ignore[arg-type]
