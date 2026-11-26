"""Stage 2: extract stable, clusterable properties with a local vision-language model."""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

from PIL.Image import Image as PILImage
from sqlalchemy import delete, select

from ...db.models import Image, ImageProperty, PropertyAlias, VlmResult, utcnow
from ...derived import derive
from ...imaging import load_image
from ...vlm.backend import VlmBackend, VlmConnectionError, make_backend
from ...vlm.normalize import Normalizer
from ...vlm.prompt import USER_PROMPT, prompt_version, system_prompt
from ...vlm.schema import CONFIDENCE_SUFFIX, Vocabulary, default_synonyms_path
from ..control import mark_pending
from .base import Stage, StageAbort
from .ollama import ensure_model

log = logging.getLogger(__name__)

VLM_SOURCES = ("vlm", "inferred", "derived")


def load_aliases(db) -> dict[tuple[str, str], str | None]:
    with db.session() as s:
        return {(a.key, a.from_value): a.to_value for a in s.scalars(select(PropertyAlias))}


def build_properties(
    vocab: Vocabulary, data: dict[str, Any], normalizer: Normalizer, max_items: int
) -> list[tuple[str, str, float | None, str]]:
    """Flatten a coerced VLM answer into ``(key, value, confidence, source)`` rows."""
    rows: list[tuple[str, str, float | None, str]] = []
    for f in vocab.fields.values():
        if not f.store:
            continue
        value = data.get(f.name)
        source = "inferred" if f.inferred else "vlm"
        conf = data.get(f.name + CONFIDENCE_SUFFIX) if f.inferred else None
        base = f.base_type
        if base == "integer":
            continue  # numbers are bucketed by derived properties
        if base == "boolean":
            if value is None:
                continue
            values = ["yes" if value else "no"]
        elif base == "list":
            values = normalizer.canonical_list(f.name, value or [], min(f.max_items, max_items))
        elif base == "enum_list":
            values = list(value or [])
        elif base == "short_text":
            values = [value] if value else []
        else:
            values = [value] if value else []
        for v in values:
            if base in ("enum", "enum_list", "boolean", "short_text"):
                cv = normalizer.canonical(f.name, str(v), free_text=base == "short_text")
            else:
                cv = v
            if cv:
                rows.append((f.name, cv, conf, source))
    return rows


class VlmStage(Stage):
    name: ClassVar[str] = "vlm"

    def __init__(self, db, settings, backend: VlmBackend | None = None) -> None:
        super().__init__(db, settings)
        cfg = settings.vlm
        self.vocab = Vocabulary.load(cfg.vocabulary_file)
        self.synonyms_path = cfg.synonyms_file or default_synonyms_path()
        self.backend = backend or make_backend(cfg, self.vocab)
        self.schema = self.vocab.json_schema()
        self.system = system_prompt(self.vocab)
        self.prompt_version = prompt_version(self.vocab)
        self.concurrency = max(1, cfg.concurrency)
        self.prefetch = 1  # decode the next image while the server answers
        self._normalizer: Normalizer | None = None
        self._normalizer_at = 0.0

    @property
    def version(self) -> str:
        return f"{self.backend.model}|{self.prompt_version}"

    def info(self) -> dict[str, object]:
        return self.backend.health()

    def setup(self) -> None:
        ensure_model(self, self.backend, self.backend.model)  # downloads it (vlm.auto_pull)
        health = self.backend.health()
        if not health.get("ok", False):
            raise RuntimeError(f"VLM backend not ready: {health.get('error', health)}")
        self.normalizer(refresh=True)

    def teardown(self) -> None:
        self.backend.unload()

    def normalizer(self, refresh: bool = False) -> Normalizer:
        # Aliases can change in the UI while a run is active -> refresh every 30 s.
        if refresh or self._normalizer is None or time.monotonic() - self._normalizer_at > 30:
            self._normalizer = Normalizer.from_file(self.synonyms_path, load_aliases(self.db))
            self._normalizer_at = time.monotonic()
        return self._normalizer

    def prepare(self, image_id: int) -> tuple[Image, PILImage]:
        """Load the image row and decode the image (runs ahead in a worker thread)."""
        with self.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise LookupError(f"image {image_id} not found")
            s.expunge(img)
        return img, load_image(img.path, max_px=self.settings.vlm.max_image_px)

    def analyze(
        self, image_id: int, prepared: tuple[Image, PILImage] | None = None
    ) -> tuple[Image, dict[str, Any], float]:
        img, pil = prepared if prepared is not None else self.prepare(image_id)
        last_exc: Exception | None = None
        started = time.monotonic()
        for attempt in range(self.settings.vlm.retries + 1):
            try:
                raw = self.backend.analyze(pil, self.system, USER_PROMPT, self.schema)
                return img, self.vocab.coerce(raw), time.monotonic() - started
            except VlmConnectionError:
                raise
            except (ValueError, RuntimeError) as exc:
                last_exc = exc
                log.info(
                    "VLM answer for image %s unusable (attempt %d): %s", image_id, attempt + 1, exc
                )
        assert last_exc is not None
        raise last_exc

    def process(self, image_id: int) -> None:
        self.process_prepared(image_id, None)

    def process_prepared(self, image_id: int, prepared: tuple[Image, PILImage] | None) -> None:
        try:
            if prepared is None:
                img, data, duration = self.analyze(image_id)
            else:
                img, data, duration = self.analyze(image_id, prepared)
        except VlmConnectionError as exc:
            # Server down? Stop the run instead of burning every image's retry budget.
            health = self.backend.health()
            if not health.get("ok", False):
                raise StageAbort(
                    f"VLM backend unreachable ({health.get('error', exc)}); run stopped, "
                    "images stay pending"
                ) from exc
            raise
        normalizer = self.normalizer()
        rows = build_properties(self.vocab, data, normalizer, self.settings.vlm.max_list_items)
        for d in derive(img, data, self.settings.derived):
            value = normalizer.resolve_alias(d.key, d.value)
            if value:
                rows.append((d.key, value, d.confidence, "derived"))
        with self.db.write() as s:
            s.execute(
                delete(ImageProperty).where(
                    ImageProperty.image_id == image_id, ImageProperty.source.in_(VLM_SOURCES)
                )
            )
            seen: set[tuple[str, str]] = set()
            for key, value, conf, source in rows:
                if (key, value) in seen:
                    continue
                seen.add((key, value))
                s.add(
                    ImageProperty(
                        image_id=image_id, key=key, value=value, confidence=conf, source=source
                    )
                )
            result = s.get(VlmResult, image_id) or VlmResult(image_id=image_id)
            result.model = self.backend.model
            result.prompt_version = self.prompt_version
            result.raw_json = data
            result.caption = data.get("caption") or None
            result.duration_s = round(duration, 2)
            result.created_at = utcnow()
            s.merge(result)
            # the search document changed -> rebuild this image's embedding
            mark_pending(s, "embed", [image_id])
