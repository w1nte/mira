from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import select

from vision_tagger.db.models import Cluster, Face, Person
from vision_tagger.faces.align import ARCFACE_112, align_face, frontalness, template
from vision_tagger.faces.clustering import update_cluster_stats
from vision_tagger.faces.generations import (
    Frame,
    aligned_path,
    aligned_portrait,
    compare_at_same_age,
    growup_animation,
    growup_frames,
    render_animation,
    resemblance,
)
from vision_tagger.faces.vectors import normalize
from vision_tagger.web.app import create_app

from .helpers import add_face, add_image, noisy, person_centers

BBOX = (100.0, 60.0, 200.0, 180.0)


def landmarks_in(bbox=BBOX) -> list[list[float]]:
    """ArcFace template points placed inside ``bbox`` (a plausible, frontal face)."""
    x1, y1, x2, _ = bbox
    scale = (x2 - x1) / 112.0
    return [[x1 + x * scale, y1 + y * scale] for x, y in ARCFACE_112.tolist()]


def portrait_file(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (320, 240), tuple(int(v) for v in rng.integers(40, 200, 3)))
    draw = ImageDraw.Draw(im)
    draw.ellipse(BBOX, fill=(220, 180, 150))
    for x, y in landmarks_in():
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return path


# --------------------------------------------------------------------------- alignment
def _centroid(arr: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    mask = np.all(np.abs(arr.astype(int) - np.array(color)) < 60, axis=2)
    ys, xs = np.nonzero(mask)
    assert len(xs), f"colour {color} not found"
    return np.array([xs.mean(), ys.mean()])


def test_align_face_puts_eyes_and_mouth_on_the_template():
    # a tilted, scaled face: template points rotated by 20 degrees
    angle = np.deg2rad(20)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = (ARCFACE_112 - 56) @ rot.T * 1.6 + np.array([210, 150])
    im = Image.new("RGB", (420, 320), (90, 90, 90))
    draw = ImageDraw.Draw(im)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for (x, y), color in zip(src, colors, strict=True):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)

    aligned = align_face(im, src.tolist(), (0, 0, 1, 1), size=256, tone=False)

    assert aligned.image.size == (256, 256)
    out = np.asarray(aligned.image)
    target = template(256)
    for k in (0, 1, 3):  # both eyes and a mouth corner land where every other face has them
        assert np.linalg.norm(_centroid(out, colors[k]) - target[k]) < 3.0
    assert aligned.frontal > 0.9


def test_align_face_without_landmarks_centres_the_box():
    im = Image.new("RGB", (300, 300), (0, 0, 0))
    ImageDraw.Draw(im).rectangle([40, 40, 100, 100], fill=(255, 255, 255))
    aligned = align_face(im, None, (40, 40, 100, 100), size=128, tone=False)
    centre = _centroid(np.asarray(aligned.image), (255, 255, 255))
    assert np.linalg.norm(centre - np.array([64, 64])) < 2.0


def test_frontalness_drops_for_turned_heads():
    frontal = ARCFACE_112.copy()
    turned = ARCFACE_112.copy()
    turned[2, 0] += 14  # nose far towards one eye
    assert frontalness(frontal) > 0.95
    assert frontalness(turned) < 0.5


# --------------------------------------------------------------------------- data
@pytest.fixture
def family(db, photos):
    """Mia (born 2010), her brother Leo (born 1980, looks alike) and Zoe (unrelated)."""
    rng = np.random.default_rng(7)
    mia_c, leo_own, zoe_c = person_centers(3, seed=3)
    leo_c = normalize(0.75 * mia_c + 0.66 * leo_own)
    ids = {}
    n = 0
    with db.write() as s:

        def person(name, birth, center, photos_by_age, extra=()):
            nonlocal n
            cluster = Cluster(locked=True, person=Person(name=name, birth_year=birth))
            s.add(cluster)
            s.flush()
            for age in photos_by_age:
                n += 1
                img = add_image(
                    s,
                    n,
                    path=str(portrait_file(photos / f"{name}{age}.jpg", n)),
                    width=320,
                    height=240,
                    recorded_at=datetime(birth + age, 6, 1),
                )
                add_face(
                    s,
                    img.id,
                    noisy(center, rng, 0.35),
                    cluster_id=cluster.id,
                    age=age,
                    x1=BBOX[0],
                    y1=BBOX[1],
                    x2=BBOX[2],
                    y2=BBOX[3],
                    landmarks=landmarks_in(),
                )
            for kw in extra:
                n += 1
                when = kw.pop("recorded_at")
                img = add_image(
                    s,
                    n,
                    path=str(portrait_file(photos / f"{name}x{n}.jpg", n)),
                    width=320,
                    height=240,
                    recorded_at=when,
                )
                kw.setdefault("landmarks", landmarks_in())
                add_face(
                    s,
                    img.id,
                    noisy(center, rng, 0.35),
                    cluster_id=cluster.id,
                    x1=BBOX[0],
                    y1=BBOX[1],
                    x2=BBOX[2],
                    y2=BBOX[3],
                    **kw,
                )
            update_cluster_stats(s, [cluster.id])
            return cluster.person_id

        ids["mia"] = person(
            "Mia",
            2010,
            mia_c,
            [1, 2, 3, 5, 7, 9],
            extra=[
                # camera clock reset: says 2000, the face looks 13 -> shown as "~13 years"
                {"recorded_at": datetime(2000, 1, 1), "age": 13},
                # a sharper, more frontal face at age 5 replaces the first one
                {"recorded_at": datetime(2015, 8, 1), "age": 5, "det_score": 0.99},
                # a tiny or blurry face never becomes a frame
                {"recorded_at": datetime(2014, 1, 1), "age": 4, "clusterable": False},
            ],
        )
        ids["leo"] = person("Leo", 1980, leo_c, [1, 3, 5, 9, 30])
        ids["zoe"] = person("Zoe", 1995, zoe_c, [2, 5, 9, 25])
    return ids


def test_growup_frames_are_ordered_by_age_not_date(db, family):
    with db.session() as s:
        frames = growup_frames(s, family["mia"])
        best5 = s.scalar(select(Face.id).where(Face.det_score == 0.99))
    assert [f.age for f in frames] == [1, 2, 3, 5, 7, 9, 13]
    assert frames[-1].age_estimated and frames[-1].label == "~13 years"
    assert frames[0].label == "1 year · 2011"
    assert next(f for f in frames if f.age == 5).face_id == best5
    assert 4 not in [f.age for f in frames]  # non-clusterable face skipped


def test_growup_frames_are_thinned_but_keep_the_span(db, family):
    with db.session() as s:
        frames = growup_frames(s, family["mia"], max_frames=3)
    assert [f.age for f in frames] == [1, 5, 13]


def test_aligned_portrait_is_cached(db, family, settings):
    cache = settings.cache_dir / "aligned"
    with db.session() as s:
        face_id = growup_frames(s, family["mia"])[0].face_id
        path = aligned_portrait(s, cache, face_id, 256)
        assert path == aligned_path(cache, face_id, 256)
        with Image.open(path) as im:
            assert im.size == (256, 256)
        mtime = path.stat().st_mtime_ns
        assert aligned_portrait(s, cache, face_id, 256).stat().st_mtime_ns == mtime
        assert aligned_portrait(s, cache, 999_999, 256) is None


def test_render_animation_frames_and_timing():
    portraits = [Image.new("RGB", (64, 64), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    data = render_animation(portraits, [("Mia", f"{a} years") for a in (1, 5, 9)], fade_steps=4)
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "WEBP" and im.is_animated
        assert im.n_frames == 3 + 2 * 4  # stills plus cross-fades
        im.seek(0)
        im.load()
        first = im.info["duration"]
        im.seek(1)
        im.load()
        assert first > im.info["duration"]  # the first age is held longer than a fade step


def test_growup_animation_renders_once(db, family, settings, monkeypatch):
    cache = settings.cache_dir / "aligned"
    with db.session() as s:
        data, frames = growup_animation(s, cache, family["mia"], size=96)
        assert len(frames) == 7
        with Image.open(io.BytesIO(data)) as im:
            assert im.size == (96, 96) and im.n_frames > 7
        from vision_tagger.faces import generations as G

        monkeypatch.setattr(G, "render_animation", lambda *a, **k: pytest.fail("not cached"))
        again, _ = growup_animation(s, cache, family["mia"], size=96)
        assert again == data


def test_compare_at_same_age_pairs_matching_ages(db, family):
    with db.session() as s:
        pairs = compare_at_same_age(s, family["mia"], family["leo"])
        strangers = compare_at_same_age(s, family["mia"], family["zoe"])
    # exact age matches win: Mia at 2 would take Leo's age-3 photo away from Mia at 3
    assert [p.age for p in pairs] == [1, 3, 5, 9]
    assert len({p.b.face_id for p in pairs}) == len(pairs)  # each face used once
    assert all(abs(p.a.age - p.b.age) <= 1 for p in pairs)
    assert np.mean([p.similarity for p in pairs]) > np.mean([p.similarity for p in strangers])


def test_resemblance_ranks_relatives_first(db, family):
    with db.session() as s:
        ranking = resemblance(s, family["mia"])
    assert [r.name for r in ranking] == ["Leo", "Zoe"]
    assert ranking[0].same_age_pairs >= 4
    assert ranking[0].similarity > ranking[1].similarity


def test_frame_labels():
    when = datetime(2015, 5, 1)
    assert Frame(1, 1, 7, False, when, 1.0).label == "7 years · 2015"
    assert Frame(1, 1, 7, True, when, 1.0).label == "~7 years"
    assert Frame(1, 1, None, True, when, 1.0).label == "2015"


# --------------------------------------------------------------------------- web + CLI
@pytest.fixture
def web(settings, family):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def test_generations_pages(web, family):
    assert 'href="/generations"' in ok(web.get("/")).text
    page = ok(web.get("/generations")).text
    assert "Mia" in page and "Leo" in page and "/media/aligned/" in page

    page = ok(web.get(f"/generations/{family['mia']}")).text
    assert 'id="growup-frames"' in page and "~13 years" in page
    assert "Looks most like" in page and "Leo" in page
    assert "onsubmit" not in page  # no inline JavaScript

    page = ok(web.get(f"/generations/{family['mia']}/compare/{family['leo']}")).text
    assert "at the same age" in page and "most alike" in page

    r = web.get(
        f"/generations/{family['mia']}/compare?other={family['zoe']}", follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/generations/{family['mia']}/compare/{family['zoe']}"

    assert "Watch Mia grow up" in ok(web.get(f"/persons/{family['mia']}")).text


def test_generations_media_and_download(web, family):
    with web.app.state.ctx.db.session() as s:
        face_id = s.scalar(select(Face.id).order_by(Face.id))
    r = ok(web.get(f"/media/aligned/{face_id}?size=256"))
    assert r.headers["content-type"] == "image/webp"
    assert ok(web.get(f"/media/aligned/{face_id}?size=7")).content[:4] == b"RIFF"
    assert web.get("/media/aligned/999999").status_code == 404

    r = ok(web.get(f"/generations/{family['mia']}/animation.webp?download=1"))
    assert r.headers["content-type"] == "image/webp"
    assert 'filename="Mia growing up.webp"' in r.headers["content-disposition"]
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.is_animated
    assert web.get("/generations/999/animation.webp").status_code == 404
    assert web.get("/generations/999", follow_redirects=False).status_code == 303


def test_generations_empty_states(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        assert "Nothing to show yet" in ok(c.get("/generations")).text
        with app.state.ctx.db.write() as s:
            s.add(Person(name="Solo"))
            s.flush()
            pid = s.scalar(select(Person.id))
        assert "two or more different ages" in ok(c.get(f"/generations/{pid}")).text
        assert c.get(f"/generations/{pid}/animation.webp").status_code == 404


def test_cli_growup(family, settings, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from vision_tagger import cli

    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "_db", lambda: _db(settings))
    out = tmp_path / "mia.webp"
    result = CliRunner().invoke(cli.app, ["growup", "mia", str(out)])
    assert result.exit_code == 0, result.output
    assert "7 ages" in result.output and out.read_bytes()[:4] == b"RIFF"
    result = CliRunner().invoke(cli.app, ["growup", "nobody", str(out)])
    assert result.exit_code == 1


def _db(settings):
    from vision_tagger.db.session import Database

    return Database(settings.db_path)
