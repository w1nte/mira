"""Reference database: known people used to name face clusters.

Layout::

    references/
      Anna Example/
        portrait1.jpg
        portrait2.png
      Ben Example/
        ...

The largest face in every reference image is embedded and stored in ``person_references``.
Clusters are then compared with each person's mean reference embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import FacesConfig, Settings
from ..db.models import Cluster, Person, PersonReference
from ..db.session import Database
from ..imaging import load_image
from .analyzer import FaceAnalyzer
from .clustering import get_or_create_person
from .vectors import centroid, from_blob, stack, to_blob

log = logging.getLogger(__name__)


@dataclass
class ReferenceImportStats:
    persons: int = 0
    added: int = 0
    skipped: int = 0
    no_face: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def import_references(
    db: Database, settings: Settings, analyzer: FaceAnalyzer | None = None
) -> ReferenceImportStats:
    stats = ReferenceImportStats()
    root = Path(settings.paths.references_dir).expanduser()
    if not root.is_dir():
        return stats
    exts = {e.lower() for e in settings.index.extensions}
    analyzer = analyzer or FaceAnalyzer(settings.faces)
    with db.session() as s:
        known = {p for (p,) in s.execute(select(PersonReference.source_path)) if p}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in exts)
        if not files:
            continue
        stats.persons += 1
        for f in files:
            key = str(f.resolve())
            if key in known:
                stats.skipped += 1
                continue
            try:
                face = analyzer.largest_face(load_image(f, max_px=settings.faces.max_image_px))
            except Exception as exc:
                stats.errors.append(f"{f}: {exc}")
                continue
            if face is None:
                stats.no_face.append(str(f))
                continue
            with db.write() as s:
                person = get_or_create_person(s, person_dir.name)
                s.add(
                    PersonReference(
                        person_id=person.id, source_path=key, embedding=to_blob(face.embedding)
                    )
                )
            stats.added += 1
    return stats


def person_reference_matrix(s: Session) -> tuple[list[int], np.ndarray]:
    """Return person ids and their normalised mean reference embeddings."""
    by_person: dict[int, list[bytes]] = {}
    for pid, emb in s.execute(select(PersonReference.person_id, PersonReference.embedding)):
        by_person.setdefault(pid, []).append(emb)
    ids = sorted(by_person)
    if not ids:
        return [], np.zeros((0, 512), dtype=np.float32)
    return ids, np.vstack([centroid(stack(by_person[p])) for p in ids])


def match_clusters(s: Session, cfg: FacesConfig) -> dict[str, int]:
    """Suggest (or auto-assign) persons for clusters based on the reference DB."""
    result = {"suggested": 0, "assigned": 0}
    pids, pmat = person_reference_matrix(s)
    clusters = list(s.scalars(select(Cluster).where(Cluster.centroid.is_not(None))))
    for c in clusters:
        if c.locked and c.person_id is not None:
            continue  # confirmed by the user
        if not pids:
            c.suggested_person_id = None
            c.suggestion_score = None
            continue
        sims = pmat @ from_blob(c.centroid)
        best = int(np.argmax(sims))
        score = float(sims[best])
        pid = pids[best]
        if not c.locked and score >= cfg.auto_accept_threshold:
            c.person_id = pid
            c.suggested_person_id = None
            c.suggestion_score = score
            result["assigned"] += 1
        elif score >= cfg.reference_threshold:
            if not c.locked and c.person_id is not None and c.person_id != pid:
                c.person_id = None  # an earlier automatic assignment no longer holds
            c.suggested_person_id = pid
            c.suggestion_score = score
            result["suggested"] += 1
        else:
            if not c.locked:
                c.person_id = None
            c.suggested_person_id = None
            c.suggestion_score = None
    s.flush()
    return result


def person_names(s: Session) -> dict[int, str]:
    return {p.id: p.name for p in s.scalars(select(Person))}
