"""Face clustering and manual cluster operations.

Rules that keep user work intact:

* clusters with ``locked=True`` (named, merged, split or edited by the user) are never
  dissolved by automatic clustering,
* faces with ``assignment="manual"`` are never moved automatically,
* ignored faces never take part in clustering,
* re-detecting faces of an image (``replace_faces``) carries cluster, assignment and ignore
  state over to the new detections,
* references (``person_references``) built from faces follow those faces.
"""

from __future__ import annotations

import functools
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ..config import FacesConfig
from ..db.models import Assignment, Cluster, Face, Person, PersonReference
from ..db.util import chunked
from .vectors import EMBEDDING_DIM, centroid, from_blob, normalize, stack, to_blob

if TYPE_CHECKING:
    from ..db.session import Database

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

SIMILARITY_CHUNK = 8192


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


@dataclass
class ClusterPlan:
    """Result of the (lock-free) computation phase of :func:`recluster`."""

    to_existing: dict[int, list[int]] = field(default_factory=dict)  # cluster id -> face ids
    new_groups: list[list[int]] = field(default_factory=list)  # face ids per new cluster
    noise: int = 0


# --------------------------------------------------------------------------- helpers
def _dbscan(m: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(m) == 0:
        return np.zeros(0, dtype=int)
    if len(m) < max(1, min_samples):
        return np.full(len(m), -1, dtype=int)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute", n_jobs=-1)
    return model.fit_predict(m)


def best_matches(
    m: np.ndarray, cmat: np.ndarray, chunk: int = SIMILARITY_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """Index and cosine similarity of the best column of ``cmat`` for every row of ``m``.

    Works in row chunks so memory stays bounded (no full ``len(m) x len(cmat)`` matrix).
    """
    best = np.zeros(len(m), dtype=np.int64)
    best_sim = np.full(len(m), -np.inf, dtype=np.float32)
    if len(m) == 0 or len(cmat) == 0:
        return best, best_sim
    for start in range(0, len(m), chunk):
        sims = m[start : start + chunk] @ cmat.T
        idx = sims.argmax(axis=1)
        best[start : start + chunk] = idx
        best_sim[start : start + chunk] = sims[np.arange(len(idx)), idx]
    return best, best_sim


def _cluster_centroids(s: Session, cluster_ids: Iterable[int] | None = None):
    rows: list = []
    base = select(Cluster.id, Cluster.centroid).where(Cluster.centroid.is_not(None))
    if cluster_ids is None:
        rows = list(s.execute(base).all())
    else:
        for part in chunked(list(cluster_ids)):
            rows.extend(s.execute(base.where(Cluster.id.in_(part))).all())
    ids = [r[0] for r in rows]
    mat = stack([r[1] for r in rows]) if rows else np.zeros((0, EMBEDDING_DIM), np.float32)
    return ids, mat


def _by_id(rows: Iterable[Any]) -> list[Any]:
    """Rows sorted by face id (their first column).

    Centroids, cover faces and DBSCAN labels depend on the row order, which would otherwise
    follow whatever index SQLite picks. Sorted here, not in SQL, so SQLite does not sort the
    embedding blobs or fall back to a full table scan.
    """
    return sorted(rows, key=lambda r: r[0])


def _update_faces(s: Session, face_ids: Sequence[int], **values: object) -> None:
    for part in chunked([int(i) for i in face_ids]):
        s.execute(update(Face).where(Face.id.in_(part)).values(**values))


def _face_clusters(s: Session, face_ids: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for part in chunked(list(face_ids)):
        out.update(
            c
            for c in s.scalars(select(Face.cluster_id).where(Face.id.in_(part)).distinct())
            if c is not None
        )
    return out


def _drop_face_references(s: Session, face_ids: Sequence[int]) -> None:
    """References built from these faces no longer describe their (former) person."""
    for part in chunked(list(face_ids)):
        s.execute(delete(PersonReference).where(PersonReference.face_id.in_(part)))


def update_cluster_stats(s: Session, cluster_ids: Iterable[int] | None = None) -> int:
    """Recompute centroid, face count and cover face; delete clusters that became empty."""
    if cluster_ids is None:
        clusters = list(s.scalars(select(Cluster)))
    else:
        ids = sorted({c for c in cluster_ids if c is not None})
        clusters = []
        for part in chunked(ids):
            clusters.extend(s.scalars(select(Cluster).where(Cluster.id.in_(part))))
    removed = 0
    for cluster in clusters:
        rows = _by_id(
            s.execute(
                select(Face.id, Face.embedding, Face.det_score, Face.face_px).where(
                    Face.cluster_id == cluster.id, Face.ignored.is_(False)
                )
            )
        )
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


# --------------------------------------------------------------------------- clustering
def _free_faces_stmt():
    return select(Face.id, Face.embedding).where(
        Face.cluster_id.is_(None),
        Face.ignored.is_(False),
        Face.clusterable.is_(True),
        Face.assignment != Assignment.MANUAL,
    )


def _dissolve_unlocked(s: Session) -> int:
    unlocked = list(s.scalars(select(Cluster.id).where(Cluster.locked.is_(False))))
    if not unlocked:
        return 0
    for part in chunked(unlocked):
        s.execute(
            update(Face)
            .where(Face.cluster_id.in_(part), Face.assignment != Assignment.MANUAL)
            .values(cluster_id=None, assignment=Assignment.AUTO)
        )
    # manual faces inside unlocked clusters should not exist, but keep such clusters
    still_used: set[int] = set()
    for part in chunked(unlocked):
        still_used.update(
            s.scalars(select(Face.cluster_id).where(Face.cluster_id.in_(part)).distinct())
        )
    to_delete = [c for c in unlocked if c not in still_used]
    for part in chunked(to_delete):
        s.execute(delete(Cluster).where(Cluster.id.in_(part)))
    s.flush()
    return len(to_delete)


def _load_inputs(s: Session, full: bool):
    free = _by_id(s.execute(_free_faces_stmt()))
    face_ids = np.array([r[0] for r in free], dtype=np.int64)
    # one float32 copy of the embeddings (~400 MB for 200k faces), normalised in place
    m = (
        normalize(stack([r[1] for r in free]), inplace=True)
        if free
        else np.zeros((0, EMBEDDING_DIM))
    )
    existing_stmt = select(Cluster.id)
    if full:
        existing_stmt = existing_stmt.where(Cluster.locked.is_(True))
    cids, cmat = _cluster_centroids(s, list(s.scalars(existing_stmt)))
    return face_ids, m, cids, cmat


def plan_clusters(
    face_ids: np.ndarray, m: np.ndarray, cids: list[int], cmat: np.ndarray, cfg: FacesConfig
) -> ClusterPlan:
    """Pure computation: assign faces to existing clusters, DBSCAN the rest."""
    plan = ClusterPlan()
    remaining = np.ones(len(face_ids), dtype=bool)
    if len(cids) and len(face_ids):
        best, best_sim = best_matches(m, cmat)
        hits = best_sim >= cfg.assign_threshold
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in np.flatnonzero(hits):
            groups[cids[int(best[idx])]].append(int(face_ids[idx]))
        plan.to_existing = dict(groups)
        remaining &= ~hits
    rest_ids = face_ids[remaining]
    labels = _dbscan(m[remaining], cfg.cluster_eps, cfg.cluster_min_samples)
    for label in sorted(set(labels.tolist()) - {-1}):
        plan.new_groups.append([int(i) for i in rest_ids[labels == label]])
    plan.noise = int((labels == -1).sum())
    return plan


def apply_plan(s: Session, plan: ClusterPlan, stats: ClusterStats) -> None:
    """Write a plan; only faces that are *still* free are touched (user edits win)."""
    free_now: set[int] = set()
    all_ids = [f for ids in plan.to_existing.values() for f in ids] + [
        f for g in plan.new_groups for f in g
    ]
    for part in chunked(all_ids):
        free_now.update(
            s.scalars(
                select(Face.id).where(
                    Face.id.in_(part),
                    Face.cluster_id.is_(None),
                    Face.ignored.is_(False),
                    Face.assignment != Assignment.MANUAL,
                )
            )
        )
    existing = set()
    for part in chunked(list(plan.to_existing)):
        existing.update(s.scalars(select(Cluster.id).where(Cluster.id.in_(part))))
    touched: set[int] = set()
    for cid, ids in plan.to_existing.items():
        ids = [i for i in ids if i in free_now]
        if not ids or cid not in existing:
            continue
        _update_faces(s, ids, cluster_id=cid, assignment=Assignment.AUTO)
        touched.add(cid)
        stats.assigned_to_existing += len(ids)
    for group in plan.new_groups:
        ids = [i for i in group if i in free_now]
        if not ids:
            continue
        cluster = Cluster(locked=False)
        s.add(cluster)
        s.flush()
        _update_faces(s, ids, cluster_id=cluster.id, assignment=Assignment.AUTO)
        touched.add(cluster.id)
        stats.new_clusters += 1
        stats.clustered_faces += len(ids)
    stats.noise = plan.noise
    update_cluster_stats(s, touched)


@_synced
def recluster(s: Session, cfg: FacesConfig, *, full: bool = False) -> ClusterStats:
    """Cluster all free faces inside one session (simple API, used by tests and small jobs).

    ``full=True`` dissolves every unlocked cluster first (their faces become free again);
    otherwise only faces without a cluster are considered (incremental mode).
    For large libraries prefer :func:`recluster_db`, which does not hold the write lock while
    DBSCAN runs.
    """
    stats = ClusterStats()
    if full:
        stats.removed_clusters = _dissolve_unlocked(s)
    face_ids, m, cids, cmat = _load_inputs(s, full)
    if len(face_ids):
        apply_plan(s, plan_clusters(face_ids, m, cids, cmat, cfg), stats)
    return stats


def recluster_db(db: Database, cfg: FacesConfig, *, full: bool = False) -> ClusterStats:
    """Like :func:`recluster`, but with short write transactions.

    1. dissolve unlocked clusters (short write), 2. read the free faces, 3. run DBSCAN without
    any transaction, 4. apply the result (short write, guarded against concurrent edits).
    """
    stats = ClusterStats()
    if full:
        with db.write() as s:
            stats.removed_clusters = _dissolve_unlocked(s)
    with db.session() as s:
        face_ids, m, cids, cmat = _load_inputs(s, full)
    if not len(face_ids):
        return stats
    plan = plan_clusters(face_ids, m, cids, cmat, cfg)
    with db.write() as s:
        apply_plan(s, plan, stats)
    return stats


# --------------------------------------------------------------------------- face replacement
def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@_synced
def replace_faces(
    s: Session, image_id: int, new_faces: list[Face], iou_threshold: float = 0.5
) -> list[int]:
    """Replace the faces of an image with fresh detections, keeping the user's work.

    Old and new faces are matched by bounding-box IoU; matched new faces inherit cluster,
    assignment, ignore state and a manual age, and references move to the new face ids. New
    faces are inserted before the old ones are deleted, so no cluster becomes empty in between.
    Returns the ids of the removed old faces (for crop cleanup).
    """
    old = list(s.scalars(select(Face).where(Face.image_id == image_id).order_by(Face.id)))
    pairs = sorted(
        (
            (_iou((o.x1, o.y1, o.x2, o.y2), (n.x1, n.y1, n.x2, n.y2)), oi, ni)
            for oi, o in enumerate(old)
            for ni, n in enumerate(new_faces)
        ),
        reverse=True,
    )
    matched: dict[int, int] = {}  # new index -> old index
    used_old: set[int] = set()
    for score, oi, ni in pairs:
        if score < iou_threshold:
            break
        if oi in used_old or ni in matched:
            continue
        used_old.add(oi)
        matched[ni] = oi
        o, n = old[oi], new_faces[ni]
        n.cluster_id, n.assignment, n.ignored = o.cluster_id, o.assignment, o.ignored
        n.manual_age = o.manual_age
    for n in new_faces:
        n.image_id = image_id
        s.add(n)
    s.flush()
    for ni, oi in matched.items():
        s.execute(
            update(PersonReference)
            .where(PersonReference.face_id == old[oi].id)
            .values(face_id=new_faces[ni].id, embedding=new_faces[ni].embedding)
        )
    unmatched = [o.id for i, o in enumerate(old) if i not in used_old]
    _drop_face_references(s, unmatched)
    affected = {o.cluster_id for o in old} | {n.cluster_id for n in new_faces}
    old_ids = [o.id for o in old]
    for part in chunked(old_ids):
        s.execute(delete(Face).where(Face.id.in_(part)))
    s.flush()
    update_cluster_stats(s, [c for c in affected if c is not None])
    return old_ids


@_synced
def delete_image_faces(s: Session, image_id: int, crops_dir: Path | None = None) -> list[int]:
    """Delete all faces of an image (content changed), keeping cluster stats consistent."""
    rows = s.execute(select(Face.id, Face.cluster_id).where(Face.image_id == image_id)).all()
    ids = [r[0] for r in rows]
    if not ids:
        return []
    _drop_face_references(s, ids)
    for part in chunked(ids):
        s.execute(delete(Face).where(Face.id.in_(part)))
    s.flush()
    update_cluster_stats(s, [r[1] for r in rows if r[1] is not None])
    if crops_dir is not None:
        from ..imaging import remove_face_crops

        remove_face_crops(crops_dir, ids)
    return ids


# --------------------------------------------------------------------------- manual operations
def _lock(s: Session, cluster_ids: Iterable[int | None]) -> None:
    ids = [c for c in cluster_ids if c is not None]
    for part in chunked(ids):
        s.execute(update(Cluster).where(Cluster.id.in_(part)).values(locked=True))


def find_person(s: Session, name: str, exclude_id: int | None = None) -> Person | None:
    """Case-insensitive lookup that also works for non-ASCII names.

    SQLite's ``lower()`` only folds ASCII, so the comparison happens in Python.
    """
    key = " ".join(name.split()).casefold()
    for person in s.scalars(select(Person)):
        if person.id != exclude_id and person.name.casefold() == key:
            return person
    return None


def get_or_create_person(s: Session, name: str) -> Person:
    name = " ".join(name.split())
    if not name:
        raise ValueError("person name must not be empty")
    person = find_person(s, name)
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
        .order_by((Face.det_score * Face.face_px).desc(), Face.id)
        .limit(k)
    ).all()
    _drop_face_references(s, [r[0] for r in rows])
    for fid, emb, _score, _px in rows:
        s.add(PersonReference(person_id=person_id, face_id=fid, embedding=emb))
    return len(rows)


def _cluster_face_ids(s: Session, cluster_id: int) -> list[int]:
    return list(s.scalars(select(Face.id).where(Face.cluster_id == cluster_id)))


@_synced
def name_cluster(
    s: Session, cluster_id: int, name: str | None, *, references: int = 0
) -> Person | None:
    """Assign a person name to a cluster (``None``/empty removes the name).

    References built from this cluster's faces always follow the new name (or are removed
    when the name is removed), so a rename never leaves wrong references behind.
    """
    cluster = s.get(Cluster, cluster_id)
    if cluster is None:
        raise LookupError(f"cluster {cluster_id} not found")
    cluster.locked = True
    cluster.suggested_person_id = None
    cluster.suggestion_score = None
    face_ids = _cluster_face_ids(s, cluster_id)
    if not name or not name.strip():
        cluster.person_id = None
        _drop_face_references(s, face_ids)
        return None
    person = get_or_create_person(s, name)
    if cluster.person_id != person.id:
        for part in chunked(face_ids):
            s.execute(
                update(PersonReference)
                .where(PersonReference.face_id.in_(part))
                .values(person_id=person.id)
            )
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
        elif src.person_id is not None and src.person_id != target.person_id:
            # faces of the merged cluster now belong to the target's person
            face_ids = _cluster_face_ids(s, cid)
            for part in chunked(face_ids):
                s.execute(
                    update(PersonReference)
                    .where(PersonReference.face_id.in_(part))
                    .values(person_id=target.person_id)
                )
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
    old = _face_clusters(s, ids)
    if target_cluster_id is None:
        cluster = Cluster(locked=True)
        s.add(cluster)
        s.flush()
        target_cluster_id = cluster.id
    _drop_face_references(s, ids)
    _update_faces(s, ids, cluster_id=target_cluster_id, assignment=Assignment.MANUAL, ignored=False)
    _lock(s, [*old, target_cluster_id])
    s.flush()
    update_cluster_stats(s, [*old, target_cluster_id])


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
    old = _face_clusters(s, ids)
    values: dict[str, object] = {"ignored": ignored, "assignment": Assignment.MANUAL}
    if ignored:
        values["cluster_id"] = None
        _drop_face_references(s, ids)
    else:
        values["assignment"] = Assignment.AUTO
    _update_faces(s, ids, **values)
    _lock(s, old)
    s.flush()
    update_cluster_stats(s, old)


@_synced
def unassign_faces(s: Session, face_ids: Iterable[int]) -> None:
    """Take faces out of their cluster ("that's not this person!").

    They become unclustered and *manual*, so automatic clustering will not put them back;
    they show up under "Unclustered faces" for the user to assign.
    """
    ids = list(face_ids)
    if not ids:
        return
    old = _face_clusters(s, ids)
    _drop_face_references(s, ids)
    _update_faces(s, ids, cluster_id=None, assignment=Assignment.MANUAL)
    _lock(s, old)
    s.flush()
    update_cluster_stats(s, old)


@_synced
def auto_split(s: Session, cluster_id: int, eps: float, min_samples: int = 2) -> list[int]:
    """Re-cluster one cluster with a tighter ``eps``.

    The largest sub-group stays in the original cluster, other sub-groups become new locked
    clusters. Faces that do not belong to any sub-group stay where they are.
    """
    rows = _by_id(
        s.execute(
            select(Face.id, Face.embedding).where(
                Face.cluster_id == cluster_id, Face.ignored.is_(False)
            )
        )
    )
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
        moved = [int(i) for i in ids[labels == lab]]
        _drop_face_references(s, moved)
        _update_faces(s, moved, cluster_id=cluster.id, assignment=Assignment.MANUAL)
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
    other = find_person(s, name, exclude_id=person_id)
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
