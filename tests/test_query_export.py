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


def test_alternatives_are_or_inside_and_across_conditions(library):
    """(A or B) and C; exclusions still apply; persons keep their AND / OR switch."""
    db, i, p = library
    beach_or_seaside = [("location_type", "beach"), ("location_type", "seaside")]
    assert ids_for(db, ImageFilter(prop_groups=[beach_or_seaside])) == [i[0], i[1], i[3]]
    # (beach or seaside) and Ben
    assert ids_for(db, ImageFilter(prop_groups=[beach_or_seaside], persons=[p["ben"]])) == [
        i[1],
        i[3],
    ]
    # alternatives of different keys: umbrella or work photo
    umbrella_or_work = [("objects", "umbrella"), ("work_image", "yes")]
    assert ids_for(db, ImageFilter(prop_groups=[umbrella_or_work])) == [i[0], i[2], i[4]]
    # two OR conditions are AND-ed: (beach or seaside) and (umbrella or boat)
    boats = [("objects", "umbrella"), ("objects", "boat")]
    assert ids_for(db, ImageFilter(prop_groups=[beach_or_seaside, boats])) == [i[0], i[3]]
    # and with a plain property condition
    assert ids_for(
        db, ImageFilter(props=[("work_image", "no")], prop_groups=[beach_or_seaside])
    ) == [i[1]]
    # exclusions stay strict: (beach or office) without Anna, without work photos
    beach_or_office = [("location_type", "beach"), ("location_type", "office")]
    assert ids_for(db, ImageFilter(prop_groups=[beach_or_office], exclude_persons=[p["anna"]])) == [
        i[4]
    ]
    assert ids_for(
        db, ImageFilter(prop_groups=[beach_or_office], exclude_props=[("work_image", "yes")])
    ) == [i[0], i[1]]
    # persons: "Anna or Ben" and "Anna and Ben" with any OR condition
    both = ImageFilter(persons=[p["anna"], p["ben"]], prop_groups=[beach_or_seaside])
    assert ids_for(db, both) == [i[1]]
    assert ids_for(db, both.copy(persons_mode="any")) == [i[0], i[1], i[3]]


def test_date_ranges_are_alternatives(library):
    """Dates: 2020-01-05, 2021-02-05, 2020-03-05, 2021-04-05, 2020-05-05."""
    db, i, _ = library
    jan_or_april = ImageFilter(dates=[("2020-01", "2020-01"), ("2021-04", None)])
    assert ids_for(db, jan_or_april) == [i[0], i[3]]
    assert ids_for(db, jan_or_april.copy(date_from="2021")) == [i[3]]  # and the range
    assert ids_for(db, ImageFilter(dates=[(None, "2020-01"), ("2020-05", "2020-05")])) == [
        i[0],
        i[4],
    ]
    # a single range is the usual from / to (date inputs, timeline)
    one = ImageFilter(dates=[("2021", "2021")])
    assert (one.date_from, one.date_to, one.dates) == ("2021", "2021", [])


def test_alternatives_url_round_trip():
    qs = (
        "t=location_type%3Abeach%7Clocation_type%3Amountain&t=season%3Asummer"
        "&d=2019&d=2021-06..2021-08&place=3&place=4&p=1&p=2&pm=any"
    )
    f = ImageFilter.from_params(QueryParams(qs))
    assert f.prop_groups == [[("location_type", "beach"), ("location_type", "mountain")]]
    assert f.props == [("season", "summer")]
    assert f.dates == [("2019", "2019"), ("2021-06", "2021-08")]
    assert f.places == [3, 4] and f.persons_mode == "any"
    assert ImageFilter.from_params(QueryParams(f.query_string())) == f
    assert f.has_prop("location_type", "mountain") and not f.has_prop("location_type", "city")
    # values may hold commas, colons and parentheses (place names)
    odd = ImageFilter.from_params(QueryParams("t=place:Neustadt (Coburg), Germany|place:A:B"))
    assert odd.prop_groups == [[("place", "Neustadt (Coburg), Germany"), ("place", "A:B")]]
    assert ImageFilter.from_params(QueryParams("d=..2005&d=2022..&d=bogus")).dates == [
        (None, "2005"),
        ("2022", None),
    ]
    # building alternatives: joins the condition on the same key, removing one splits it
    single = ImageFilter(props=[("season", "summer"), ("objects", "boat")])
    alt = ImageFilter.from_params(QueryParams(single.with_prop_alt("season", "winter")))
    assert alt.props == [("objects", "boat")]
    assert alt.prop_groups == [[("season", "summer"), ("season", "winter")]]
    more = ImageFilter.from_params(QueryParams(alt.with_prop_alt("season", "spring")))
    assert more.prop_groups == [[("season", "summer"), ("season", "winter"), ("season", "spring")]]
    back = ImageFilter.from_params(QueryParams(alt.without_prop("season", "winter")))
    assert back.props == [("objects", "boat"), ("season", "summer")] and not back.prop_groups
    assert alt.without_key("season").prop_groups == [] and alt.without_key("season").props == [
        ("objects", "boat")
    ]
    # without a condition on the key, an alternative is simply required
    assert "t=weather%3Asnowy" in single.with_prop_alt("weather", "snowy")
    anna_or = ImageFilter.from_params(QueryParams(ImageFilter(persons=[3]).with_person_alt(7)))
    assert anna_or.persons == [3, 7] and anna_or.persons_mode == "any"


def test_old_urls_keep_their_meaning(library):
    """``tm=any`` (the former "any property" switch) and plain repeated ``t`` are unchanged."""
    db, i, _ = library
    old_any = ImageFilter.from_params(
        QueryParams("t=location_type%3Abeach&t=location_type%3Aseaside&tm=any")
    )
    assert ids_for(db, old_any) == [i[0], i[1], i[3]]
    assert old_any.query_string() == "t=location_type%3Abeach%7Clocation_type%3Aseaside"
    old_all = ImageFilter.from_params(QueryParams("t=location_type:beach&t=objects:umbrella"))
    assert ids_for(db, old_all) == [i[0]]
    assert old_all.props == [("location_type", "beach"), ("objects", "umbrella")]
    assert ImageFilter.from_params(QueryParams("place=12")).places == [12]
    # a value with a "|" that does not start a new key:value stays one value
    assert ImageFilter.from_params(QueryParams("t=tags:a|b")).props == [("tags", "a|b")]


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


def test_folder_filter_is_literal(db):
    with db.write() as s:
        add_image(s, 1, folder="100%/done")
        add_image(s, 2, folder="100x/done")
        add_image(s, 3, folder="a_b")
        add_image(s, 4, folder="axb/c")
    assert len(ids_for(db, ImageFilter(folder="100%"))) == 1
    assert len(ids_for(db, ImageFilter(folder="a_b"))) == 1
    assert len(ids_for(db, ImageFilter(text="a_b"))) == 0  # file names don't contain it
    assert len(ids_for(db, ImageFilter(folder="100%\\done"))) == 1  # backslashes accepted


def test_large_id_lists_do_not_hit_sqlite_parameter_limit(db, tmp_path):
    """Exports and face moves over more than 32,766 rows must work."""
    import numpy as np
    from sqlalchemy import insert

    from vision_tagger.db.models import Face, Image
    from vision_tagger.faces.vectors import to_blob

    n = 33_000
    with db.write() as s:
        s.execute(
            insert(Image),
            [
                {
                    "path": f"/nowhere/{i}.jpg",
                    "path_key": f"/nowhere/{i}.jpg",
                    "root": "/nowhere",
                    "rel_path": f"{i}.jpg",
                    "folder": "",
                    "filename": f"{i}.jpg",
                    "ext": ".jpg",
                    "size_bytes": 1,
                    "mtime": 0.0,
                    "phash": f"{i:016x}",
                    "dhash": "",
                    "date_source": "mtime",
                    "metadata": {},
                }
                for i in range(n)
            ],
        )
        emb = to_blob(np.ones(4, dtype=np.float32))
        s.execute(
            insert(Face),
            [
                {
                    "image_id": i + 1,
                    "x1": 0,
                    "y1": 0,
                    "x2": 1,
                    "y2": 1,
                    "det_score": 0.9,
                    "embedding": emb,
                    "face_px": 50,
                }
                for i in range(n)
            ],
        )
    stats = export_images(db, ImageFilter(), tmp_path / "big")
    assert stats.skipped == n  # files don't exist, but no "too many SQL variables"
    with db.write() as s:
        face_ids = list(s.scalars(select(Face.id)))
        cid = C.split_faces(s, face_ids)
        assert s.get(Cluster, cid).face_count == n


def test_neighbours_match_gallery_order(db):
    from vision_tagger.query import neighbours

    with db.write() as s:
        for i in range(8):
            img = add_image(
                s,
                i,
                recorded_at=None if i in (2, 5) else datetime(2020, 1, 1 + (i % 3)),
                folder="x",
            )
            img.size_bytes = 100 * (i % 3)
            img.filename = f"f{i % 4}.jpg"
    for sort in ("date_desc", "date_asc", "name", "path", "size", "faces"):
        f = ImageFilter(sort=sort)
        with db.session() as s:
            order = all_image_ids(s, f)
            for pos, image_id in enumerate(order):
                prev_id, next_id = neighbours(s, f, image_id)
                assert prev_id == (order[pos - 1] if pos else None), (sort, pos)
                assert next_id == (order[pos + 1] if pos + 1 < len(order) else None), (sort, pos)


def test_folder_children(db):
    from vision_tagger.query import folder_children

    with db.write() as s:
        add_image(s, 1, folder="2020/Italy")
        add_image(s, 2, folder="2020/Italy/Rome")
        add_image(s, 3, folder="2020/Spain")
        add_image(s, 4, folder="2021")
        add_image(s, 5, folder="")
    with db.session() as s:
        top = {name: n for _full, name, n in folder_children(s, ImageFilter())}
        assert top == {"2020": 3, "2021": 1, "": 1}
        sub = folder_children(s, ImageFilter(folder="2020"))
        assert [(full, n) for full, _name, n in sub] == [("2020/Italy", 2), ("2020/Spain", 1)]
