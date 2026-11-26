from __future__ import annotations

import random

from sqlalchemy import select

from vision_tagger.db.models import Assignment, Cluster, Face, PersonReference
from vision_tagger.faces import clustering as C
from vision_tagger.faces.quiz import (
    ai_comment,
    can_quiz,
    judge,
    next_name_question,
    next_quiz_question,
    streak_title,
)

from .helpers import make_face_dataset


def setup_people(s, settings, named=("Anna", "Ben")):
    people, _ = make_face_dataset(s, persons=3, per_person=4)
    C.recluster(s, settings.faces, full=True)
    clusters = [next(iter({s.get(Face, f).cluster_id for f in people[i]})) for i in range(3)]
    for cid, name in zip(clusters, named, strict=False):
        C.name_cluster(s, cid, name)
    return people, clusters


def test_name_question_offers_similar_people_and_names_cluster(db, settings):
    with db.write() as s:
        _people, clusters = setup_people(s, settings)
        q = next_name_question(s)
        assert q.cluster_id == clusters[2]  # the only unnamed cluster
        assert len(q.face_ids) == 4 and q.remaining == 1
        assert {o.name for o in q.options} == {"Anna", "Ben"}
        assert next_name_question(s, skip={clusters[2]}) is None
        C.name_cluster(s, q.cluster_id, "Cleo")
        assert next_name_question(s) is None


def test_quiz_question_and_judging(db, settings):
    with db.write() as s:
        assert not can_quiz(s)
        _people, _clusters = setup_people(s, settings, named=("Anna", "Ben", "Cleo"))
        assert can_quiz(s)
        rng = random.Random(1)
        q = next_quiz_question(s, rng)
        names = {o.name for o in q.options}
        assert len(q.options) == 3 and names == {"Anna", "Ben", "Cleo"}
        answer = s.get(Cluster, q.cluster_id).person
        right = judge(s, q.face_id, answer.id, streak=2)
        assert right.correct and right.ai_answer > 0.5
        assert right.title == streak_title(3) == "Warming up"
        assert "agrees" in ai_comment(right)
        other = next(o for o in q.options if o.person_id != answer.id)
        wrong = judge(s, q.face_id, other.person_id)
        assert not wrong.correct and not wrong.ai_agrees_with_player
        assert answer.name in ai_comment(wrong)


def test_ai_agrees_when_face_is_in_the_wrong_cluster(db, settings):
    with db.write() as s:
        people, clusters = setup_people(s, settings, named=("Anna", "Ben", "Cleo"))
        intruder = people[1][0]  # one of Ben's faces, misfiled under Anna
        C.move_faces(s, [intruder], clusters[0])
        ben = s.get(Cluster, clusters[1]).person
        verdict = judge(s, intruder, ben.id)
        assert not verdict.correct and verdict.ai_agrees_with_player
        assert "cluster is wrong" in ai_comment(verdict)


def test_unassign_keeps_face_out_of_clustering(db, settings):
    with db.write() as s:
        people, clusters = setup_people(s, settings)
        C.name_cluster(s, clusters[0], "Anna", references=4)
        face = people[0][0]
        C.unassign_faces(s, [face])
        f = s.get(Face, face)
        assert f.cluster_id is None and f.assignment == Assignment.MANUAL
        assert s.scalar(select(PersonReference).where(PersonReference.face_id == face)) is None
        C.recluster(s, settings.faces, full=True)
        assert s.get(Face, face).cluster_id is None
