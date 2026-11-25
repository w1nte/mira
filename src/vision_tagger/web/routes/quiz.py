""" "Who's that?" face quiz (name unnamed clusters, test yourself, fix clustering mistakes)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response
from sqlalchemy import select

from ...db.models import Cluster, Person
from ...faces import clustering as C
from ...faces.ages import check_image_date
from ...faces.quiz import (
    ai_comment,
    can_quiz,
    judge,
    next_name_question,
    next_quiz_question,
    streak_title,
)
from ..common import ctx, ints, render

router = APIRouter()


@dataclass
class QuizState:
    """Game state travels in hidden form fields (the server stays stateless)."""

    mode: str = "quiz"
    score: int = 0
    rounds: int = 0
    streak: int = 0
    best: int = 0
    named: int = 0
    skip: list[int] = field(default_factory=list)
    seen: list[int] = field(default_factory=list)

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> QuizState:
        def num(key: str) -> int:
            try:
                return max(0, int(data.get(key) or 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            mode="name" if data.get("mode") == "name" else "quiz",
            score=num("score"),
            rounds=num("rounds"),
            streak=num("streak"),
            best=num("best"),
            named=num("named"),
            skip=ints(str(data.get("skip") or "").split(","))[-200:],
            seen=ints(str(data.get("seen") or "").split(","))[-30:],
        )

    def fields(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "score": str(self.score),
            "rounds": str(self.rounds),
            "streak": str(self.streak),
            "best": str(self.best),
            "named": str(self.named),
            "skip": ",".join(map(str, self.skip)),
            "seen": ",".join(map(str, self.seen)),
        }


async def _state(request: Request) -> tuple[QuizState, Mapping[str, Any]]:
    if request.method == "POST":
        form = await request.form()
        return QuizState.parse(form), form
    return QuizState.parse(request.query_params), request.query_params


def _card(request: Request, state: QuizState, **extra: Any) -> Response:
    c = ctx(request)
    with c.db.session() as s:
        quiz_ok = can_quiz(s)
        question = None
        if state.mode == "name":
            question = next_name_question(s, set(state.skip))
        elif quiz_ok and "verdict" not in extra:
            question = next_quiz_question(s, random.Random(), set(state.seen))
        persons = list(s.scalars(select(Person.name)))
    return render(
        request,
        "partials/quiz_card.html",
        state=state,
        question=question,
        quiz_ok=quiz_ok,
        persons=sorted(persons),
        title=streak_title(state.streak),
        **extra,
    )


@router.get("/quiz")
async def quiz_page(request: Request) -> Response:
    state, _ = await _state(request)
    card = _card(request, state)
    return render(request, "quiz.html", state=state, card=card.body.decode())


@router.get("/quiz/card")
async def quiz_card(request: Request) -> Response:
    state, _ = await _state(request)
    return _card(request, state)


@router.post("/quiz/name")
async def quiz_name(request: Request) -> Response:
    state, form = await _state(request)
    c = ctx(request)
    cluster_id = ints([str(form.get("cluster_id"))])
    name = str(form.get("name") or "").strip()
    flash = None
    if cluster_id and form.get("action") == "skip":
        state.skip.append(cluster_id[0])
    elif cluster_id and name:
        with c.db.write() as s:
            if s.get(Cluster, cluster_id[0]) is not None:
                C.name_cluster(
                    s, cluster_id[0], name, references=c.settings.faces.references_per_named_cluster
                )
                state.named += 1
                flash = f"✓ Named “{name}”"
    return _card(request, state, flash=flash)


@router.post("/quiz/answer")
async def quiz_answer(request: Request) -> Response:
    state, form = await _state(request)
    c = ctx(request)
    face_ids = ints([str(form.get("face_id"))])
    choice = ints([str(form.get("choice"))])
    if not face_ids or not choice:
        return _card(request, state)
    try:
        with c.db.session() as s:
            verdict = judge(s, face_ids[0], choice[0], streak=state.streak)
    except LookupError:
        return _card(request, state, flash="That face changed meanwhile – here is a new one.")
    state.rounds += 1
    state.seen.append(face_ids[0])
    if verdict.correct:
        state.score += 1
        state.streak += 1
        state.best = max(state.best, state.streak)
    else:
        state.streak = 0
    image_id = int(str(form.get("image_id") or 0) or 0)
    return _card(
        request,
        state,
        verdict=verdict,
        comment=ai_comment(verdict),
        face_id=face_ids[0],
        image_id=image_id,
        age_line=_age_line(c, image_id, face_ids[0], verdict.answer.name),
    )


def _age_line(c, image_id: int, face_id: int, name: str) -> str | None:
    """'Anna was 7 here (2015).' - uses birth year + photo date, or the face estimate."""
    with c.db.session() as s:
        check = check_image_date(s, image_id) if image_id else None
    info = check.faces.get(face_id) if check else None
    if info is None:
        return None
    if info.calendar_age is not None and info.calendar_age >= 0 and not check.suspicious:
        return f"🎂 {name} was {info.calendar_age} here ({check.recorded_year})."
    if info.face_age is not None:
        return f"🎂 {name} looks about {info.face_age} here."
    return None


@router.post("/quiz/fix")
async def quiz_fix(request: Request) -> Response:
    """'That's not X!' (unassign) or 'move it to the person I picked'."""
    state, form = await _state(request)
    c = ctx(request)
    face_ids = ints([str(form.get("face_id"))])
    flash = None
    if face_ids:
        with c.db.write() as s:
            if form.get("action") == "move" and str(form.get("person") or "").strip():
                C.move_faces_to_person(s, face_ids, str(form.get("person")))
                flash = f"✓ Moved the face to {form.get('person')}. Thanks for the fix!"
            else:
                C.unassign_faces(s, face_ids)
                flash = "✓ Removed from that person – find it under Faces → Unclustered."
    return _card(request, state, flash=flash)
