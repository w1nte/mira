from __future__ import annotations

import numpy as np
from sqlalchemy import func, select

from vision_tagger.db.models import Assignment, Cluster, Face, Person, PersonReference
from vision_tagger.faces import clustering as C
from vision_tagger.faces.references import match_clusters
from vision_tagger.faces.vectors import to_blob

from .helpers import add_face, add_image, make_face_dataset, noisy


def cluster_of(s, face_ids):
    return {s.get(Face, f).cluster_id for f in face_ids}


def test_recluster_finds_people(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=3, per_person=6)
        stats = C.recluster(s, settings.faces, full=True)
        assert stats.new_clusters == 3
        assert stats.noise == 0
        clusters = [cluster_of(s, ids) for ids in people.values()]
        assert all(len(c) == 1 for c in clusters)
        assert len(set.union(*clusters)) == 3
        for c in s.scalars(select(Cluster)):
            assert c.face_count == 6 and c.cover_face_id is not None and c.centroid


def test_low_quality_faces_are_not_clustered(db, settings):
    with db.write() as s:
        _people, centers = make_face_dataset(s, persons=1, per_person=4)
        img = add_image(s, 100)
        small = add_face(s, img.id, centers[0], face_px=10, clusterable=False)
        C.recluster(s, settings.faces, full=True)
        assert s.get(Face, small.id).cluster_id is None


def test_locked_and_manual_survive_full_recluster(db, settings):
    with db.write() as s:
        people, _centers = make_face_dataset(s, persons=2, per_person=5)
        C.recluster(s, settings.faces, full=True)
        cid_a = next(iter(cluster_of(s, people[0])))
        C.name_cluster(s, cid_a, "Anna")
        # user moves one face of person 1 into Anna's cluster on purpose
        moved = people[1][0]
        C.move_faces(s, [moved], cid_a)

    with db.write() as s:
        C.recluster(s, settings.faces, full=True)
        assert s.get(Cluster, cid_a) is not None
        assert s.get(Cluster, cid_a).person.name == "Anna"
        assert s.get(Face, moved).cluster_id == cid_a
        assert s.get(Face, moved).assignment == Assignment.MANUAL
        assert cluster_of(s, people[0]) == {cid_a}


def test_incremental_assigns_new_faces_to_existing_cluster(db, settings):
    rng = np.random.default_rng(42)
    with db.write() as s:
        people, centers = make_face_dataset(s, persons=2, per_person=5)
        C.recluster(s, settings.faces, full=True)
        cid = next(iter(cluster_of(s, people[0])))
        img = add_image(s, 200)
        new_face = add_face(s, img.id, noisy(centers[0], rng))
        stats = C.recluster(s, settings.faces, full=False)
        assert stats.assigned_to_existing == 1
        assert s.get(Face, new_face.id).cluster_id == cid
        assert s.get(Cluster, cid).face_count == 6


def test_merge_split_ignore(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=3, per_person=4)
        C.recluster(s, settings.faces, full=True)
        a, b, c = (next(iter(cluster_of(s, people[i]))) for i in range(3))
        C.name_cluster(s, b, "Ben")
        merged = C.merge_clusters(s, a, [b])
        assert merged.locked and merged.face_count == 8
        assert merged.person.name == "Ben"
        assert s.get(Cluster, b) is None

        new_id = C.split_faces(s, people[1][:2])
        assert s.get(Cluster, new_id).face_count == 2
        assert s.get(Cluster, a).face_count == 6

        C.ignore_faces(s, [people[2][0]])
        f = s.get(Face, people[2][0])
        assert f.ignored and f.cluster_id is None
        assert s.get(Cluster, c).face_count == 3 and s.get(Cluster, c).locked


def test_auto_split_separates_mixed_cluster(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=2, per_person=5)
        C.recluster(s, settings.faces, full=True)
        a = next(iter(cluster_of(s, people[0])))
        b = next(iter(cluster_of(s, people[1])))
        C.merge_clusters(s, a, [b])
        assert s.get(Cluster, a).face_count == 10
        new = C.auto_split(s, a, eps=settings.faces.split_eps)
        assert len(new) == 1
        assert s.get(Cluster, a).face_count == 5
        assert s.get(Cluster, new[0]).face_count == 5


def test_move_faces_to_person_and_name_creates_references(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=2, per_person=4)
        C.recluster(s, settings.faces, full=True)
        a = next(iter(cluster_of(s, people[0])))
        C.name_cluster(s, a, "Anna", references=3)
        assert s.scalar(select(func.count()).select_from(PersonReference)) == 3
        cid = C.move_faces_to_person(s, people[1][:1], "anna")  # case-insensitive
        assert cid == a
        assert s.scalar(select(func.count()).select_from(Person)) == 1


def test_reference_matching_suggests_and_assigns(db, settings):
    rng = np.random.default_rng(7)
    with db.write() as s:
        people, centers = make_face_dataset(s, persons=2, per_person=5)
        C.recluster(s, settings.faces, full=True)
        anna = C.get_or_create_person(s, "Anna")
        ben = C.get_or_create_person(s, "Ben")
        for _ in range(3):
            s.add(PersonReference(person_id=anna.id, embedding=to_blob(noisy(centers[0], rng))))
        # weak reference for Ben -> only a suggestion
        s.add(PersonReference(person_id=ben.id, embedding=to_blob(noisy(centers[1], rng, r=1.1))))
        s.flush()
        settings.faces.auto_accept_threshold = 0.75
        settings.faces.reference_threshold = 0.4
        res = match_clusters(s, settings.faces)
        a = s.get(Cluster, next(iter(cluster_of(s, people[0]))))
        b = s.get(Cluster, next(iter(cluster_of(s, people[1]))))
        assert a.person_id == anna.id and not a.locked
        assert b.person_id is None and b.suggested_person_id == ben.id
        assert res == {"assigned": 1, "suggested": 1}
        C.accept_suggestion(s, b.id)
        assert b.person_id == ben.id and b.locked


def test_similar_clusters(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=3, per_person=4)
        C.recluster(s, settings.faces, full=True)
        a = next(iter(cluster_of(s, people[0])))
        sims = C.similar_clusters(s, a)
        assert len(sims) == 2 and all(-1 <= v <= 1 for _, v in sims)


def test_non_ascii_names_are_case_insensitive(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=2, per_person=4)
        C.recluster(s, settings.faces, full=True)
        a = next(iter(cluster_of(s, people[0])))
        b = next(iter(cluster_of(s, people[1])))
        C.name_cluster(s, a, "Özlem")
        C.name_cluster(s, b, "özlem")  # same person, different case (non-ASCII)
        assert s.scalar(select(func.count()).select_from(Person)) == 1
        other = C.get_or_create_person(s, "Ангелина")
        merged = C.rename_person(s, other.id, "ÖZLEM")
        assert merged.name == "Özlem"
        assert s.scalar(select(func.count()).select_from(Person)) == 1


def test_references_follow_faces(db, settings):
    with db.write() as s:
        people, _ = make_face_dataset(s, persons=2, per_person=4)
        C.recluster(s, settings.faces, full=True)
        a = next(iter(cluster_of(s, people[0])))
        anna = C.name_cluster(s, a, "Anna", references=3)
        C.name_cluster(s, a, "Ben")  # rename: Anna's face references move to Ben
        ben = C.find_person(s, "Ben")
        refs = list(s.scalars(select(PersonReference)))
        assert refs and all(r.person_id == ben.id for r in refs)
        assert anna.id != ben.id
        C.ignore_faces(s, [refs[0].face_id])  # ignored faces are no references
        assert s.scalar(select(func.count()).select_from(PersonReference)) == 2
        C.name_cluster(s, a, None)  # unnamed cluster -> its references are gone
        assert s.scalar(select(func.count()).select_from(PersonReference)) == 0


def test_recluster_db_short_transactions_and_concurrent_write(db, settings, monkeypatch):
    """DBSCAN runs without the write lock, and faces edited meanwhile are left alone."""
    import threading

    from vision_tagger.faces import clustering as mod

    with db.write() as s:
        people, _ = make_face_dataset(s, persons=2, per_person=5)
        grabbed = people[1][0]
    started, release = threading.Event(), threading.Event()
    real = mod._dbscan

    def slow_dbscan(m, eps, min_samples):
        started.set()
        assert release.wait(10)
        return real(m, eps, min_samples)

    monkeypatch.setattr(mod, "_dbscan", slow_dbscan)
    result = {}
    t = threading.Thread(
        target=lambda: result.update(stats=C.recluster_db(db, settings.faces, full=True))
    )
    t.start()
    assert started.wait(10)
    # a concurrent write (e.g. user moves a face) must not block on the clustering
    with db.write() as s:
        C.ignore_faces(s, [grabbed])
    release.set()
    t.join(10)
    assert result["stats"].new_clusters == 2
    with db.session() as s:
        f = s.get(Face, grabbed)
        assert f.ignored and f.cluster_id is None  # user edit won
