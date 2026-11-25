"""Face clustering and manual cluster operations.

Rules that keep user work intact:

* clusters with ``locked=True`` (named, merged, split or edited by the user) are never
  dissolved by automatic clustering,
* faces with ``assignment="manual"`` are never moved automatically,
* ignored faces never take part in clustering.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from ..config import FacesConfig
from ..db.models import Assignment, Cluster, Face, Person, PersonReference
from .vectors import centroid, from_blob, normalize, stack, to_blob

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _synced(fn: Callable[P, R]) -> Callable[P, R]:
    """Flush and expire the session after bulk UPDATE/DELETE statements.

    Bulk statements bypass the identity map, so ORM objects loaded earlier in the same
    session could otherwise show stale values.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        result = fn(*args, **kwargs)
        s = args[0]
        assert isinstance(s, Session)
        s.flush()
        s.expire_all()
        return result

    return wrapper


@dataclass
class ClusterStats:
    assigned_to_existing: int = 0
    new_clusters: int = 0
    clustered_faces: int = 0
    noise: int = 0
    removed_clusters: int = 0

    def __str__(self) -> str:
        return (
            f"{self.assigned_to_existing} faces joined existing clusters, "
            f"{self.new_clusters} new clusters ({self.clustered_faces} faces), "
            f"{self.noise} unclustered"
        )


def _dbscan(m: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(m) == 0:
        return np.zeros(0, dtype=int)
    if len(m) < max(1, min_samples):
        return np.full(len(m), -1, dtype=int)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute", n_jobs=-1)
    return model.fit_predict(m)


def _cluster_centroids(s: Session, cluster_ids: Iterable[int] | None = None):
    stmt = select(Cluster.id, Cluster.centroid).where(Cluster.centroid.is_not(None))
    if cluster_ids is not None:
        stmt = stmt.where(Cluster.id.in_(list(cluster_ids)))
    rows = s.execute(stmt).all()
    ids = [r[0] for r in rows]
    mat = stack([r[1] for r in rows]) if rows else np.zeros((0, 512), dtype=np.float32)
    return ids, mat


def update_cluster_stats(s: Session, cluster_ids: Iterable[int] | None = None) -> int:
    """Recompute centroid, face count and cover face; delete clusters that became empty."""
    stmt = select(Cluster)
    if cluster_ids is not None:
        ids = list(set(cluster_ids))
        if not ids:
            return 0
        stmt = stmt.where(Cluster.id.in_(ids))
    removed = 0
    for cluster in s.scalars(stmt):
        rows = s.execute(
            select(Face.id, Face.embedding, Face.det_score, Face.face_px).where(
                Face.cluster_id == cluster.id, Face.ignored.is_(False)
            )
        ).all()
        if not rows:
            s.delete(cluster)
            removed += 1
            continue
        cluster.face_count = len(rows)
        cluster.centroid = to_blob(centroid(stack([r[1] for r in rows])))
        best = max(rows, key=lambda r: (r[2] or 0) * min(r[3] or 1, 300))
        cluster.cover_face_id = best[0]
    s.flush()
    return removed


@_synced
def recluster(s: Session, cfg: FacesConfig, *, full: bool = False) -> ClusterStats:
    """Cluster all free faces.

    ``full=True`` dissolves every unlocked cluster first (their faces become free again);
    otherwise only faces without a cluster are considered (incremental mode).
    """
    stats = ClusterStats()
    if full:
        unlocked = list(s.scalars(select(Cluster.id).where(Cluster.locked.is_(False))))
        if unlocked:
            s.execute(
                update(Face)
                .where(Face.cluster_id.in_(unlocked), Face.assignment != Assignment.MANUAL)
                .values(cluster_id=None, assignment=Assignment.AUTO)
            )
            # manual faces inside unlocked clusters should not exist, but keep them safe
            still_used = set(
                s.scalars(select(Face.cluster_id).where(Face.cluster_id.in_(unlocked)).distinct())
            )
            to_delete = [c for c in unlocked if c not in still_used]
            if to_delete:
                s.execute(delete(Cluster).where(Cluster.id.in_(to_delete)))
            stats.removed_clusters = len(to_delete)
        s.flush()

    free = s.execute(
        select(Face.id, Face.embedding).where(
            Face.cluster_id.is_(None),
            Face.ignored.is_(False),
            Face.clusterable.is_(True),
            Face.assignment != Assignment.MANUAL,
        )
    ).all()
    if not free:
        return stats
    face_ids = np.array([r[0] for r in free])
    m = normalize(stack([r[1] for r in free]))

    # 1) attach to existing clusters (locked ones in full mode, all in incremental mode)
    existing_stmt = select(Cluster.id)
    if full:
        existing_stmt = existing_stmt.where(Cluster.locked.is_(True))
    cids, cmat = _cluster_centroids(s, s.scalars(existing_stmt).all())
    touched: set[int] = set()
    remaining = np.ones(len(face_ids), dtype=bool)
    if len(cids):
        sims = m @ cmat.T
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(face_ids)), best]
        hits = best_sim >= cfg.assign_threshold
        for idx in np.flatnonzero(hits):
            cid = cids[int(best[idx])]
            s.execute(
                update(Face)
                .where(Face.id == int(face_ids[idx]))
                .values(cluster_id=cid, assignment=Assignment.AUTO)
            )
            touched.add(cid)
        remaining &= ~hits
        stats.assigned_to_existing = int(hits.sum())

    # 2) DBSCAN for the rest
    rest_ids = face_ids[remaining]
    labels = _dbscan(m[remaining], cfg.cluster_eps, cfg.cluster_min_samples)
    for label in sorted(set(labels.tolist()) - {-1}):
        members = rest_ids[labels == label]
        cluster = Cluster(locked=False)
        s.add(cluster)
        s.flush()
        s.execute(
            update(Face)
            .where(Face.id.in_([int(i) for i in members]))
            .values(cluster_id=cluster.id, assignment=Assignment.AUTO)
        )
        touched.add(cluster.id)
        stats.new_clusters += 1
        stats.clustered_faces += len(members)
    stats.noise = int((labels == -1).sum())
    update_cluster_stats(s, touched)
    return stats


# --------------------------------------------------------------------------- manual operations
def _lock(s: Session, cluster_ids: Iterable[int]) -> None:
    ids = [c for c in cluster_ids if c is not None]
    if ids:
        s.execute(update(Cluster).where(Cluster.id.in_(ids)).values(locked=True))


def get_or_create_person(s: Session, name: str) -> Person:
    name = " ".join(name.split())
    if not name:
        raise ValueError("person name must not be empty")
    person = s.scalar(select(Person).where(func.lower(Person.name) == name.lower()))
    if person is None:
        person = Person(name=name)
        s.add(person)
        s.flush()
    return person


def add_cluster_references(s: Session, cluster_id: int, person_id: int, k: int) -> int:
    """Use the ``k`` best faces of a confirmed cluster as references for the person."""
    if k <= 0:
        return 0
    rows = s.execute(
        select(Face.id, Face.embedding, Face.det_score, Face.face_px)
        .where(Face.cluster_id == cluster_id, Face.ignored.is_(False))
        .order_by((Face.det_score * Face.face_px).desc())
        .limit(k)
    ).all()
    face_ids = [r[0] for r in rows]
    if face_ids:
        s.execute(
            delete(PersonReference).where(
                PersonReference.person_id == person_id, PersonReference.face_id.in_(face_ids)
            )
        )
    for fid, emb, _score, _px in rows:
        s.add(PersonReference(person_id=person_id, face_id=fid, embedding=emb))
    return len(rows)


def name_cluster(
    s: Session, cluster_id: int, name: str | None, *, references: int = 0
) -> Person | None:
    """Assign a person name to a cluster (``None``/empty removes the name)."""
    cluster = s.get(Cluster, cluster_id)
    if cluster is None:
        raise LookupError(f"cluster {cluster_id} not found")
    cluster.locked = True
    cluster.suggested_person_id = None
    cluster.suggestion_score = None
    if not name or not name.strip():
        cluster.person_id = None
        return None
    person = get_or_create_person(s, name)
    cluster.person_id = person.id
    add_cluster_references(s, cluster.id, person.id, references)
    s.flush()
    return person


def accept_suggestion(s: Session, cluster_id: int, *, references: int = 0) -> Person | None:
    cluster = s.get(Cluster, cluster_id)
    if cluster is None or cluster.suggested_person_id is None:
        return None
    person = s.get(Person, cluster.suggested_person_id)
    assert person is not None
    return name_cluster(s, cluster_id, person.name, references=references)


@_synced
def merge_clusters(s: Session, target_id: int, source_ids: Iterable[int]) -> Cluster:
    target = s.get(Cluster, target_id)
    if target is None:
        raise LookupError(f"cluster {target_id} not found")
    sources = [c for c in source_ids if c != target_id]
    for cid in sources:
        src = s.get(Cluster, cid)
        if src is None:
            continue
        if target.person_id is None and src.person_id is not None:
            target.person_id = src.person_id
        s.execute(update(Face).where(Face.cluster_id == cid).values(cluster_id=target_id))
        s.delete(src)
    target.locked = True
    s.flush()
    update_cluster_stats(s, [target_id])
    return target


@_synced
def move_faces(s: Session, face_ids: Iterable[int], target_cluster_id: int | None) -> None:
    """Move faces to a cluster (``None`` = new cluster). Marks the faces as manual."""
    ids = list(face_ids)
    if not ids:
        return
    old = set(s.scalars(select(Face.cluster_id).where(Face.id.in_(ids)).distinct()))
    if target_cluster_id is None:
        cluster = Cluster(locked=True)
        s.add(cluster)
        s.flush()
        target_cluster_id = cluster.id
    s.execute(
        update(Face)
        .where(Face.id.in_(ids))
        .values(cluster_id=target_cluster_id, assignment=Assignment.MANUAL, ignored=False)
    )
    _lock(s, [*old, target_cluster_id])
    s.flush()
    update_cluster_stats(s, [c for c in [*old, target_cluster_id] if c is not None])


@_synced
def split_faces(s: Session, face_ids: Iterable[int]) -> int | None:
    """Move the selected faces into a new (locked) cluster. Returns the new cluster id."""
    ids = list(face_ids)
    if not ids:
        return None
    cluster = Cluster(locked=True)
    s.add(cluster)
    s.flush()
    new_id = cluster.id
    move_faces(s, ids, new_id)
    return new_id if s.get(Cluster, new_id) is not None else None


@_synced
def move_faces_to_person(s: Session, face_ids: Iterable[int], name: str) -> int:
    """Move faces into the person's largest cluster (creating one if needed)."""
    person = get_or_create_person(s, name)
    cluster = s.scalar(
        select(Cluster)
        .where(Cluster.person_id == person.id)
        .order_by(Cluster.face_count.desc())
        .limit(1)
    )
    if cluster is None:
        cluster = Cluster(locked=True, person_id=person.id)
        s.add(cluster)
        s.flush()
    move_faces(s, face_ids, cluster.id)
    return cluster.id


@_synced
def ignore_faces(s: Session, face_ids: Iterable[int], ignored: bool = True) -> None:
    ids = list(face_ids)
    if not ids:
        return
    old = set(s.scalars(select(Face.cluster_id).where(Face.id.in_(ids)).distinct()))
    values: dict[str, object] = {"ignored": ignored, "assignment": Assignment.MANUAL}
    if ignored:
        values["cluster_id"] = None
    else:
        values["assignment"] = Assignment.AUTO
    s.execute(update(Face).where(Face.id.in_(ids)).values(**values))
    _lock(s, old)
    s.flush()
    update_cluster_stats(s, [c for c in old if c is not None])


@_synced
def auto_split(s: Session, cluster_id: int, eps: float, min_samples: int = 2) -> list[int]:
    """Re-cluster one cluster with a tighter ``eps``.

    The largest sub-group stays in the original cluster, other sub-groups become new locked
    clusters. Faces that do not belong to any sub-group stay where they are.
    """
    rows = s.execute(
        select(Face.id, Face.embedding).where(
            Face.cluster_id == cluster_id, Face.ignored.is_(False)
        )
    ).all()
    if len(rows) < 2 * min_samples:
        return []
    ids = np.array([r[0] for r in rows])
    labels = _dbscan(normalize(stack([r[1] for r in rows])), eps, min_samples)
    groups = [lab for lab in set(labels.tolist()) if lab != -1]
    if len(groups) <= 1:
        return []
    groups.sort(key=lambda lab: -(labels == lab).sum())
    new_ids: list[int] = []
    for lab in groups[1:]:
        cluster = Cluster(locked=True)
        s.add(cluster)
        s.flush()
        s.execute(
            update(Face)
            .where(Face.id.in_([int(i) for i in ids[labels == lab]]))
            .values(cluster_id=cluster.id, assignment=Assignment.MANUAL)
        )
        new_ids.append(cluster.id)
    _lock(s, [cluster_id])
    s.flush()
    update_cluster_stats(s, [cluster_id, *new_ids])
    return new_ids


def similar_clusters(s: Session, cluster_id: int, limit: int = 8) -> list[tuple[int, float]]:
    """Clusters whose centroid is closest to the given cluster (merge candidates)."""
    cluster = s.get(Cluster, cluster_id)
    if cluster is None or cluster.centroid is None:
        return []
    ids, mat = _cluster_centroids(s)
    if not ids:
        return []
    sims = mat @ from_blob(cluster.centroid)
    order = np.argsort(-sims)
    out = [(ids[i], float(sims[i])) for i in order if ids[i] != cluster_id]
    return out[:limit]


# --------------------------------------------------------------------------- persons
def rename_person(s: Session, person_id: int, name: str) -> Person:
    """Rename a person; renaming to an existing name merges the two persons."""
    person = s.get(Person, person_id)
    if person is None:
        raise LookupError(f"person {person_id} not found")
    name = " ".join(name.split())
    if not name:
        raise ValueError("person name must not be empty")
    other = s.scalar(
        select(Person).where(func.lower(Person.name) == name.lower(), Person.id != person_id)
    )
    if other is not None:
        merge_persons(s, other.id, [person_id])
        return other
    person.name = name
    s.flush()
    return person


@_synced
def merge_persons(s: Session, target_id: int, source_ids: Iterable[int]) -> Person:
    target = s.get(Person, target_id)
    if target is None:
        raise LookupError(f"person {target_id} not found")
    sources = [p for p in source_ids if p != target_id]
    if sources:
        s.execute(update(Cluster).where(Cluster.person_id.in_(sources)).values(person_id=target_id))
        s.execute(
            update(Cluster)
            .where(Cluster.suggested_person_id.in_(sources))
            .values(suggested_person_id=target_id)
        )
        s.execute(
            update(PersonReference)
            .where(PersonReference.person_id.in_(sources))
            .values(person_id=target_id)
        )
        s.execute(delete(Person).where(Person.id.in_(sources)))
    return target


@_synced
def delete_person(s: Session, person_id: int) -> None:
    """Delete a person (clusters stay, but become unnamed)."""
    s.execute(update(Cluster).where(Cluster.person_id == person_id).values(person_id=None))
    s.execute(
        update(Cluster)
        .where(Cluster.suggested_person_id == person_id)
        .values(suggested_person_id=None, suggestion_score=None)
    )
    s.execute(delete(PersonReference).where(PersonReference.person_id == person_id))
    s.execute(delete(Person).where(Person.id == person_id))
