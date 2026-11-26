""" "Who's that?" - a face quiz that doubles as a labelling and quality-control tool.

* **Name the strangers**: shows the largest unnamed cluster with suggested names (reference
  match + most similar known persons); answering names the cluster.
* **Quiz me**: shows a random face of a named person with four choices. Afterwards the player
  sees how sure the face recognition was ("you vs. the AI"), can pull a wrongly clustered face
  out ("that's not X!") or move it to the person they picked when the AI agrees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import Cluster, Face, Person
from .vectors import from_blob, normalize

STREAK_TITLES = [
    (25, "👑 Face whisperer"),
    (15, "🦉 Family historian"),
    (10, "🔥 On fire"),
    (5, "⭐ Sharp eyes"),
    (3, "👍 Warming up"),
]


def streak_title(streak: int) -> str | None:
    for threshold, title in STREAK_TITLES:
        if streak >= threshold:
            return title
    return None


@dataclass
class Option:
    person_id: int | None
    name: str
    score: float | None = None  # similarity, for display


@dataclass
class NameQuestion:
    cluster_id: int
    face_ids: list[int]
    image_id: int | None
    face_count: int
    options: list[Option]
    remaining: int


@dataclass
class QuizQuestion:
    face_id: int
    image_id: int
    cluster_id: int
    options: list[Option]


@dataclass
class Verdict:
    correct: bool
    answer: Option
    chosen: Option
    ai_answer: float  # similarity of the face to the correct person
    ai_chosen: float | None  # similarity to the person the player picked
    ai_agrees_with_player: bool = False
    title: str | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- person centroids
def person_centroids(s: Session) -> tuple[list[int], list[str], np.ndarray]:
    """One normalised centroid per named person (face-count weighted over their clusters)."""
    acc: dict[int, np.ndarray] = {}
    names: dict[int, str] = {}
    for pid, name, centroid, count in s.execute(
        select(Person.id, Person.name, Cluster.centroid, Cluster.face_count)
        .join(Cluster, Cluster.person_id == Person.id)
        .where(Cluster.centroid.is_not(None))
    ):
        vec = from_blob(centroid) * max(1, count)
        acc[pid] = acc.get(pid, 0) + vec
        names[pid] = name
    pids = sorted(acc)
    if not pids:
        return [], [], np.zeros((0, 512), dtype=np.float32)
    return pids, [names[p] for p in pids], normalize(np.vstack([acc[p] for p in pids]))


def _similar_persons(
    s: Session, vec: np.ndarray, limit: int, exclude: set[int | None] | None = None
) -> list[Option]:
    exclude = exclude or set()
    pids, names, mat = person_centroids(s)
    if not pids:
        return []
    sims = mat @ vec
    order = np.argsort(-sims)
    out = [Option(pids[i], names[i], float(sims[i])) for i in order if pids[i] not in exclude]
    return out[:limit]


# --------------------------------------------------------------------------- name the strangers
def next_name_question(s: Session, skip: set[int] | None = None) -> NameQuestion | None:
    skip = skip or set()
    stmt = (
        select(Cluster)
        .where(Cluster.person_id.is_(None), Cluster.face_count > 0)
        .order_by(Cluster.face_count.desc(), Cluster.id)
    )
    remaining = (
        s.scalar(select(func.count()).select_from(Cluster).where(Cluster.person_id.is_(None))) or 0
    )
    for cluster in s.scalars(stmt):
        if cluster.id in skip:
            continue
        faces = list(
            s.scalars(
                select(Face)
                .where(Face.cluster_id == cluster.id, Face.ignored.is_(False))
                .order_by((Face.det_score * Face.face_px).desc(), Face.id)
                .limit(6)
            )
        )
        if not faces:
            continue
        options: list[Option] = []
        if cluster.suggested_person_id is not None:
            p = s.get(Person, cluster.suggested_person_id)
            if p is not None:
                options.append(Option(p.id, p.name, cluster.suggestion_score))
        if cluster.centroid is not None:
            options += _similar_persons(
                s, from_blob(cluster.centroid), 4 - len(options), {o.person_id for o in options}
            )
        return NameQuestion(
            cluster_id=cluster.id,
            face_ids=[f.id for f in faces],
            image_id=faces[0].image_id,
            face_count=cluster.face_count,
            options=options[:4],
            remaining=remaining,
        )
    return None


# --------------------------------------------------------------------------- quiz me
def can_quiz(s: Session) -> bool:
    return (
        s.scalar(
            select(func.count(func.distinct(Cluster.person_id))).where(
                Cluster.person_id.is_not(None), Cluster.face_count > 0
            )
        )
        or 0
    ) >= 2


def next_quiz_question(
    s: Session, rng: random.Random | None = None, exclude_faces: set[int] | None = None
) -> QuizQuestion | None:
    rng = rng or random.Random()
    if not can_quiz(s):
        return None
    clusters = list(
        s.execute(
            select(Cluster.id, Cluster.person_id, Cluster.face_count).where(
                Cluster.person_id.is_not(None), Cluster.face_count > 0
            )
        )
    )
    # pick a person first (so big clusters don't dominate), then one of their faces
    by_person: dict[int, list[int]] = {}
    for cid, pid, _n in clusters:
        by_person.setdefault(pid, []).append(cid)
    pid = rng.choice(sorted(by_person))
    face_rows = list(
        s.execute(
            select(Face.id, Face.image_id, Face.cluster_id, Face.embedding).where(
                Face.cluster_id.in_(by_person[pid]), Face.ignored.is_(False)
            )
        )
    )
    candidates = [r for r in face_rows if r[0] not in (exclude_faces or set())] or face_rows
    face_id, image_id, cluster_id, emb = rng.choice(candidates)
    answer = s.get(Person, pid)
    assert answer is not None
    # distractors: two look-alikes (hard) and the rest random
    similar = _similar_persons(s, from_blob(emb), 3, {pid})
    others = [
        p for p in s.scalars(select(Person).where(Person.id.in_(list(by_person)))) if p.id != pid
    ]
    distractors = list(similar[:2])
    pool = [p for p in others if p.id not in {d.person_id for d in distractors}]
    rng.shuffle(pool)
    distractors += [Option(p.id, p.name) for p in pool[: 3 - len(distractors)]]
    options = [Option(answer.id, answer.name), *distractors[:3]]
    rng.shuffle(options)
    return QuizQuestion(face_id=face_id, image_id=image_id, cluster_id=cluster_id, options=options)


def judge(s: Session, face_id: int, chosen_person_id: int, streak: int = 0) -> Verdict:
    face = s.get(Face, face_id)
    if face is None or face.cluster_id is None:
        raise LookupError("face is no longer in a named cluster")
    cluster = s.get(Cluster, face.cluster_id)
    if cluster is None or cluster.person_id is None:
        raise LookupError("face is no longer in a named cluster")
    answer = s.get(Person, cluster.person_id)
    chosen = s.get(Person, chosen_person_id)
    assert answer is not None
    vec = from_blob(face.embedding)
    pids, _names, mat = person_centroids(s)
    sims = dict(zip(pids, (mat @ vec).tolist(), strict=True)) if pids else {}
    correct = chosen_person_id == answer.id
    verdict = Verdict(
        correct=correct,
        answer=Option(answer.id, answer.name, sims.get(answer.id)),
        chosen=Option(chosen_person_id, chosen.name if chosen else "?", sims.get(chosen_person_id)),
        ai_answer=float(sims.get(answer.id, 0.0)),
        ai_chosen=sims.get(chosen_person_id),
    )
    if not correct and verdict.ai_chosen is not None:
        verdict.ai_agrees_with_player = verdict.ai_chosen > verdict.ai_answer
    verdict.title = streak_title(streak + 1) if correct else None
    return verdict


def ai_comment(verdict: Verdict) -> str:
    """A playful one-liner comparing the player with the face recognition."""
    sure = verdict.ai_answer
    if verdict.correct:
        if sure >= 0.6:
            return f"The AI agrees - it was {sure:.0%} sure too."
        if sure >= 0.4:
            return f"Nice! The AI was only {sure:.0%} sure - you beat it."
        return f"Impressive - the AI barely recognised this one ({sure:.0%})."
    if verdict.ai_agrees_with_player:
        return (
            f"Hmm… the AI also thinks this looks more like {verdict.chosen.name} "
            f"({verdict.ai_chosen:.0%} vs {sure:.0%}). Maybe the cluster is wrong?"
        )
    return f"The face is filed under {verdict.answer.name} (AI: {sure:.0%} sure)."
