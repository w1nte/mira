from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from starlette.datastructures import QueryParams

from vision_tagger.db.models import Cluster, ImageProperty, Person, PropertyAlias
from vision_tagger.export import export_images
from vision_tagger.faces import clustering as C
from vision_tagger.indexer import Indexer
from vision_tagger.properties import delete_value, merge_values
from vision_tagger.query import (
    ImageFilter,
    all_image_ids,
    count_images,
    date_histogram,
    page_images,
    person_counts,
    property_counts,
)

from .conftest import make_image
from .helpers import add_face, add_image, person_centers


@pytest.fixture
def library(db):
    """5 images: Anna in 1,2,3 ; Ben in 2,3,4 ; properties per image."""
    centers = person_centers(2)
    with db.write() as s:
        imgs = [
            add_image(s, i, recorded_at=datetime(2020 + (i % 2), 1 + i, 5), folder=f"trip/{i % 2}")
            for i in range(5)
        ]
        anna = Cluster(locked=True, person=Person(name="Anna"))
        ben = Cluster(locked=True, person=Person(name="Ben"))
        unnamed = Cluster(locked=False)
        s.add_all([anna, ben, unnamed])
        s.flush()
        for i in (0, 1, 2):
            add_face(s, imgs[i].id, centers[0], cluster_id=anna.id)
        for i in (1, 2, 3):
            add_face(s, imgs[i].id, centers[1], cluster_id=ben.id)
        add_face(s, imgs[4].id, centers[1], cluster_id=unnamed.id)
        props = {
            0: [("location_type", "beach"), ("objects", "umbrella")],
            1: [("location_type", "beach"), ("work_image", "no")],
            2: [("location_type", "office"), ("work_image", "yes")],
            3: [("location_type", "seaside"), ("objects", "boat")],
            4: [("location_type", "office"), ("work_image", "yes")],
        }
        for i, kv in props.items():
            for k, v in kv:
                s.add(ImageProperty(image_id=imgs[i].id, key=k, value=v))
        ids = [i.id for i in imgs]
        persons = {"anna": anna.person_id, "ben": ben.person_id, "unnamed": unnamed.id}
    return db, ids, persons


def ids_for(db, f):
    with db.session() as s:
        return sorted(all_image_ids(s, f))


def test_person_and_property_combinations(library):
    db, ids, p = library
    i = ids
    assert ids_for(db, ImageFilter(persons=[p["anna"], p["ben"]])) == [i[1], i[2]]
    assert ids_for(db, ImageFilter(persons=[p["anna"], p["ben"]], persons_mode="any")) == i[:4]
    assert ids_for(db, ImageFilter(clusters=[p["unnamed"]])) == [i[4]]
    assert ids_for(db, ImageFilter(props=[("location_type", "beach")])) == [i[0], i[1]]
    assert ids_for(
        db,
        ImageFilter(
            props=[("location_type", "beach"), ("location_type", "seaside")], props_mode="any"
        ),
    ) == [i[0], i[1], i[3]]
    # Anna, but not at work
    assert ids_for(db, ImageFilter(persons=[p["anna"]], exclude_props=[("work_image", "yes")])) == [
        i[0],
        i[1],
    ]
    # persons AND property
    assert ids_for(db, ImageFilter(persons=[p["ben"]], props=[("location_type", "office")])) == [
        i[2]
    ]
    assert ids_for(db, ImageFilter(exclude_persons=[p["anna"]])) == [i[3], i[4]]


def test_date_folder_faces_filters(library):
    db, ids, _ = library
    assert ids_for(db, ImageFilter(date_from="2021", date_to="2021")) == [ids[1], ids[3]]
    assert ids_for(db, ImageFilter(date_from="2020-03", date_to="2020-03")) == [ids[2]]
    assert ids_for(db, ImageFilter(folder="trip/1")) == [ids[1], ids[3]]
    assert ids_for(db, ImageFilter(has_faces=True)) == ids
    assert ids_for(db, ImageFilter(stage="faces", status="pending")) == ids
    with db.session() as s:
        hist = dict(date_histogram(s, ImageFilter(), "year"))
        assert hist == {"2020": 3, "2021": 2}


def test_filter_query_string_roundtrip():
    qs = "p=1&p=2&pm=any&t=location_type%3Abeach&xt=work_image%3Ayes&from=2020&faces=yes&sort=name"
    f = ImageFilter.from_params(QueryParams(qs))
    assert f.persons == [1, 2] and f.persons_mode == "any"
    assert f.props == [("location_type", "beach")]
    assert f.exclude_props == [("work_image", "yes")]
    assert f.has_faces is True and f.sort == "name"
    assert ImageFilter.from_params(QueryParams(f.query_string())) == f
    assert (
        ImageFilter.from_params(QueryParams(f.without_prop("location_type", "beach"))).props == []
    )
    assert ImageFilter().is_empty


def test_facet_counts_and_paging(library):
    db, _ids, p = library
    with db.session() as s:
        counts = property_counts(s)
        assert counts["location_type"][0] == ("beach", 2) or counts["location_type"][0] == (
            "office",
            2,
        )
        filtered = property_counts(s, ImageFilter(persons=[p["anna"]]))
        assert dict(filtered["location_type"]) == {"beach": 2, "office": 1}
        people = {person.name: n for person, n in person_counts(s)}
        assert people == {"Anna": 3, "Ben": 3}
        assert count_images(s, ImageFilter(props=[("work_image", "yes")])) == 2
        page = page_images(s, ImageFilter(sort="date_asc"), offset=1, limit=2)
        assert len(page) == 2


def test_merge_and_delete_property_values(library):
    db, ids, _ = library
    with db.write() as s:
        changed = merge_values(s, "location_type", ["seaside"], "beach")
        assert changed == 1
    assert ids_for(db, ImageFilter(props=[("location_type", "beach")])) == [ids[0], ids[1], ids[3]]
    with db.write() as s:
        # merging beach into coast re-points the existing alias seaside->beach to coast
        merge_values(s, "location_type", ["beach"], "coast")
        aliases = {(a.from_value, a.to_value) for a in s.scalars(select(PropertyAlias))}
        assert aliases == {("seaside", "coast"), ("beach", "coast")}
        assert delete_value(s, "objects", "boat") == 1
    with db.session() as s:
        assert "boat" not in dict(property_counts(s)["objects"])


def test_person_merge_and_rename(library):
    db, ids, p = library
    with db.write() as s:
        C.rename_person(s, p["ben"], "anna")  # same name -> merge
        assert s.scalar(select(Person).where(Person.name == "Ben")) is None
    assert ids_for(db, ImageFilter(persons=[p["anna"]])) == ids[:4]
    with db.write() as s:
        C.delete_person(s, p["anna"])
        assert s.scalar(select(Person)) is None


def test_export_modes(db, settings, photos, tmp_path):
    make_image(photos / "a" / "x.jpg", seed=1, date=datetime(2020, 5, 1))
    make_image(photos / "b" / "x.jpg", seed=2, date=datetime(2021, 6, 1))
    Indexer(db, settings).run()

    out = tmp_path / "export_flat"
    stats = export_images(db, ImageFilter(), out, mode="copy")
    assert stats.exported == 2
    names = sorted(p.name for p in out.iterdir())
    assert "export_manifest.csv" in names and "x.jpg" in names
    assert len([n for n in names if n.startswith("x")]) == 2  # name collision resolved

    out = tmp_path / "export_date"
    export_images(db, ImageFilter(), out, mode="hardlink", layout="date")
    assert (out / "2020" / "05" / "x.jpg").exists()
    assert (out / "2021" / "06" / "x.jpg").exists()

    out = tmp_path / "export_folder"
    stats = export_images(db, ImageFilter(date_from="2021"), out, mode="symlink", layout="folder")
    assert stats.exported == 1
    exported = out / "b" / "x.jpg"
    assert exported.exists()
    assert Path(photos / "b" / "x.jpg").exists()  # originals untouched
