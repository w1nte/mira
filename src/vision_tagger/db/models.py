"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --------------------------------------------------------------------------- status constants
class Status:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"

    ALL = (PENDING, RUNNING, DONE, ERROR, SKIPPED)


class Assignment:
    AUTO = "auto"
    MANUAL = "manual"
    REFERENCE = "reference"


# --------------------------------------------------------------------------- index
class Image(Base):
    __tablename__ = "images"
    # area (bounding box) filters and the map's heatmap read only this index
    __table_args__ = (Index("ix_images_lat_lon", "lat", "lon"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    path_key: Mapped[str] = mapped_column(Text, unique=True, index=True)
    root: Mapped[str] = mapped_column(Text)
    rel_path: Mapped[str] = mapped_column(Text)
    folder: Mapped[str] = mapped_column(Text, index=True, default="")
    filename: Mapped[str] = mapped_column(Text)
    ext: Mapped[str] = mapped_column(String(16), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    mtime: Mapped[float] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    date_source: Mapped[str] = mapped_column(String(32), default="mtime")
    # Duplicate = same pHash AND nearly the same dHash (see indexer.find_duplicate); the pHash
    # alone collides for distinct but similar images (blank pages, same-template forms).
    phash: Mapped[str] = mapped_column(String(32), index=True)
    dhash: Mapped[str] = mapped_column(String(32), index=True, default="", server_default="")
    # "metadata" is reserved on declarative classes, hence the attribute name ``meta``.
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    camera: Mapped[str | None] = mapped_column(Text, index=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    missing: Mapped[bool] = mapped_column(Boolean, default=False)
    # GPS position in decimal degrees (copied from meta["gps"]) for the map and area filters;
    # place_id is the named place the photo belongs to (see geo.places).
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    place_id: Mapped[int | None] = mapped_column(
        ForeignKey("places.id", ondelete="SET NULL"), index=True
    )

    stages: Mapped[list[StageStatus]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True
    )
    faces: Mapped[list[Face]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Face.id",
    )
    animals: Mapped[list[Animal]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Animal.id",
    )
    properties: Mapped[list[ImageProperty]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True
    )
    duplicates: Mapped[list[DuplicateFile]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True
    )


class DuplicateFile(Base):
    __tablename__ = "duplicate_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    path: Mapped[str] = mapped_column(Text)
    path_key: Mapped[str] = mapped_column(Text, unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    mtime: Mapped[float] = mapped_column(Float, default=0.0)
    found_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    image: Mapped[Image] = relationship(back_populates="duplicates")


# --------------------------------------------------------------------------- job state
class StageStatus(Base):
    __tablename__ = "stage_status"
    __table_args__ = (
        # Serves the runner's claim (one stage and status, by image id) and covers the
        # dashboard's status and outdated-version counts.
        Index("ix_stage_status_stage_status_image", "stage", "status", "image_id", "stage_version"),
    )

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    stage: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default=Status.PENDING)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    stage_version: Mapped[str | None] = mapped_column(String(64))
    run_id: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    image: Mapped[Image] = relationship(back_populates="stages")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    stages: Mapped[str] = mapped_column(Text)
    batch_size: Mapped[int] = mapped_column(Integer)
    limit: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text)


class Control(Base):
    """Single-row table used to steer the pipeline from the CLI and the web UI."""

    __tablename__ = "control"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    desired_state: Mapped[str] = mapped_column(String(16), default="stopped")
    active_stage: Mapped[str | None] = mapped_column(String(32))
    active_run_id: Mapped[int | None] = mapped_column(Integer)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime)
    message: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)


class Collection(Base):
    """A saved view of the library: a name for a gallery filter (``ImageFilter`` query)."""

    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    query: Mapped[str] = mapped_column(Text, default="", server_default="")
    position: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- places
class PlaceSource:
    GEONAMES = "geonames"  # nearest town of the offline GeoNames gazetteer
    OSM = "osm"  # OpenStreetMap reverse geocoding (Nominatim, opt-in)
    USER = "user"  # named by the user
    COORDS = "coords"  # no gazetteer available: the coordinates as name


class Place(Base):
    """GPS-tagged photos grouped by distance, with a name (see ``geo.places``).

    ``user_named`` places keep their name, centre and photos when places are found again, like
    locked face clusters.
    """

    __tablename__ = "places"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    # the automatic name before it was made unique ("Neustadt, Germany" for a place shown as
    # "Neustadt (Coburg), Germany"); places with the same automatic name nearby are joined
    auto_name: Mapped[str | None] = mapped_column(Text)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    radius_m: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    country: Mapped[str | None] = mapped_column(Text)
    admin: Mapped[str | None] = mapped_column(Text)
    locality: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        String(16), default=PlaceSource.GEONAMES, server_default=PlaceSource.GEONAMES
    )
    user_named: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class GeocodeCache(Base):
    """Answers of the online reverse geocoder, so every location is asked for only once."""

    __tablename__ = "geocode_cache"

    key: Mapped[str] = mapped_column(String(96), primary_key=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- faces
class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    notes: Mapped[str | None] = mapped_column(Text)
    birth_year: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    clusters: Mapped[list[Cluster]] = relationship(
        back_populates="person", foreign_keys="Cluster.person_id"
    )
    references: Mapped[list[PersonReference]] = relationship(
        back_populates="person", cascade="all, delete-orphan", passive_deletes=True
    )


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), index=True
    )
    suggested_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL")
    )
    suggestion_score: Mapped[float | None] = mapped_column(Float)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    cover_face_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    person: Mapped[Person | None] = relationship(
        back_populates="clusters", foreign_keys=[person_id]
    )
    suggested_person: Mapped[Person | None] = relationship(foreign_keys=[suggested_person_id])
    faces: Mapped[list[Face]] = relationship(back_populates="cluster")


class Face(Base):
    __tablename__ = "faces"
    # Covering indexes: face rows are large (embedding), so the frequent "faces of an image /
    # of a cluster that are not ignored" lookups and counts should never read them.
    __table_args__ = (
        Index("ix_faces_image_ignored_cluster", "image_id", "ignored", "cluster_id"),
        Index("ix_faces_cluster_ignored_image_age", "cluster_id", "ignored", "image_id", "age"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    det_score: Mapped[float] = mapped_column(Float)
    landmarks: Mapped[list[Any] | None] = mapped_column(JSON)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String(1))
    face_px: Mapped[int] = mapped_column(Integer, default=0)
    cluster_id: Mapped[int | None] = mapped_column(ForeignKey("clusters.id", ondelete="SET NULL"))
    assignment: Mapped[str] = mapped_column(String(16), default=Assignment.AUTO)
    ignored: Mapped[bool] = mapped_column(Boolean, default=False)
    clusterable: Mapped[bool] = mapped_column(Boolean, default=True)

    image: Mapped[Image] = relationship(back_populates="faces")
    cluster: Mapped[Cluster | None] = relationship(back_populates="faces")


class PersonReference(Base):
    __tablename__ = "person_references"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    source_path: Mapped[str | None] = mapped_column(Text)
    face_id: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    person: Mapped[Person] = relationship(back_populates="references")


# --------------------------------------------------------------------------- animals
class Pet(Base):
    """A group of photos of one animal (see ``animals.grouping``); a name makes it a pet.

    Unnamed, unlocked pets are made by automatic grouping and may be dissolved when the animals
    are grouped again; ``locked`` pets (named, merged or edited by the user) are kept, like
    locked face clusters.
    """

    __tablename__ = "pets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    species: Mapped[str] = mapped_column(String(32))
    locked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    animal_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cover_animal_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    animals: Mapped[list[Animal]] = relationship(back_populates="pet")


class Animal(Base):
    """An animal found in a photo: box, species, detector score and DINOv2 embedding."""

    __tablename__ = "animals"
    # covering indexes like the faces ones: animal rows are large (embedding)
    __table_args__ = (
        Index("ix_animals_image_ignored_pet", "image_id", "ignored", "pet_id"),
        Index("ix_animals_pet_ignored_image", "pet_id", "ignored", "image_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    species: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    box_px: Mapped[int] = mapped_column(Integer, default=0)
    pet_id: Mapped[int | None] = mapped_column(ForeignKey("pets.id", ondelete="SET NULL"))
    assignment: Mapped[str] = mapped_column(String(16), default=Assignment.AUTO)
    ignored: Mapped[bool] = mapped_column(Boolean, default=False)
    groupable: Mapped[bool] = mapped_column(Boolean, default=True)

    image: Mapped[Image] = relationship(back_populates="animals")
    pet: Mapped[Pet | None] = relationship(back_populates="animals")


# --------------------------------------------------------------------------- properties
class ImageProperty(Base):
    __tablename__ = "image_properties"
    __table_args__ = (
        # Covers "image has key=value" checks (gallery filters) with a single seek, the value
        # counts (property pages, facets) and the "best examples" of a value.
        Index("ix_image_properties_key_value_image", "key", "value", "image_id", "confidence"),
        Index("ix_image_properties_image_key", "image_id", "key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16), default="vlm")

    image: Mapped[Image] = relationship(back_populates="properties")


class PropertyAlias(Base):
    """Maps ``(key, from_value)`` to a canonical value; ``to_value=None`` deletes the value."""

    __tablename__ = "property_aliases"
    __table_args__ = (UniqueConstraint("key", "from_value"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(64))
    from_value: Mapped[str] = mapped_column(Text)
    to_value: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class VlmResult(Base):
    __tablename__ = "vlm_results"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(String(64))
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    caption: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ImageEmbedding(Base):
    """Text embedding of an image's search document (see ``search.document``)."""

    __tablename__ = "image_embeddings"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(Text)
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float16, L2-normalised
    doc_hash: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class VisualEmbedding(Base):
    """Embedding of the photo itself in a shared image-text space (see ``visual``).

    Search compares it with the embedding of the query text, and "Similar photos" with the
    ones of other photos. ``model`` names the vector space (only vectors of the configured
    model are searched); ``stage_version`` the exact model files and preprocessing, so a change
    re-embeds the photos.
    """

    __tablename__ = "visual_embeddings"
    # the search index loads and checks one model's rows without reading the vectors
    __table_args__ = (Index("ix_visual_embeddings_model_created", "model", "created_at"),)

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64))
    stage_version: Mapped[str] = mapped_column(String(64))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float16, L2-normalised
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
