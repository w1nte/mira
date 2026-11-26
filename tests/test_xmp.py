from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_tagger import xmp as X
from vision_tagger.db.models import Cluster, ImageProperty, Person, VlmResult
from vision_tagger.export import export_images, write_sidecars_in_place
from vision_tagger.query import ImageFilter

from .helpers import add_face, add_image, person_centers

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/",
    "vt": X.NS_VT,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def library(db, photos):
    """Two photos: one with a person, properties and a caption; one dated only by file time."""
    files = []
    for i, fmt in enumerate(("JPEG", "PNG")):
        path = photos / f"photo{i}.{'jpg' if fmt == 'JPEG' else 'png'}"
        arr = np.random.default_rng(i).integers(0, 255, (300, 400, 3), dtype=np.uint8)
        Image.fromarray(arr).save(path, fmt)
        files.append(path)
    with db.write() as s:
        a = add_image(s, 1, path=str(files[0]), width=400, height=300)
        a.recorded_at, a.date_source = datetime(2019, 7, 14, 15, 2, 11), "exif"
        b = add_image(s, 2, path=str(files[1]), width=400, height=300)
        b.date_source = "mtime"
        for img, path in ((a, files[0]), (b, files[1])):
            img.filename, img.rel_path, img.ext = path.name, path.name, path.suffix
        cluster = Cluster(locked=True, person=Person(name='Anna "Nan" & <Co>'))
        s.add(cluster)
        s.flush()
        add_face(
            s, a.id, person_centers(1)[0], cluster_id=cluster.id, x1=100, y1=60, x2=180, y2=160
        )
        for key, value, conf in [
            ("objects", "sand_castle", None),
            ("location_type", "beach", None),
            ("event_type", "none", None),
            ("aesthetic", "high", None),
            ("work_image", "no", 0.9),
        ]:
            s.add(ImageProperty(image_id=a.id, key=key, value=value, confidence=conf, source="vlm"))
        s.add(
            VlmResult(
                image_id=a.id, model="m", prompt_version="p", raw_json={}, caption="Kids & a castle"
            )
        )
    return files


def parse(packet: bytes) -> ET.Element:
    root = ET.fromstring(packet.decode("utf-8").split("?>", 1)[1].rsplit("<?xpacket", 1)[0])
    return root.find("rdf:RDF/rdf:Description", NS)


def bag(desc: ET.Element, tag: str) -> list[str]:
    return [li.text for li in desc.findall(f"{tag}/rdf:Bag/rdf:li", NS)]


def test_xmp_carries_keywords_people_caption_regions_and_date(db, library):
    with db.session() as s:
        from vision_tagger.db.models import Image as ImageRow

        metas = X.collect(s, list(s.query(ImageRow).order_by(ImageRow.id)))
    desc = parse(X.to_xmp(metas[1]))
    name = 'Anna "Nan" & <Co>'
    assert bag(desc, "dc:subject") == ["sand castle", "beach", name]  # "none" is no keyword
    hierarchy = bag(desc, "lr:hierarchicalSubject")
    assert f"People|{name}" in hierarchy and "vision-tagger|Aesthetic|high" in hierarchy
    assert bag(desc, "Iptc4xmpExt:PersonInImage") == [name]
    assert desc.find("dc:description/rdf:Alt/rdf:li", NS).text == "Kids & a castle"
    assert desc.get(f"{{{NS['exif']}}}DateTimeOriginal") == "2019-07-14T15:02:11"
    region = desc.find("mwg-rs:Regions/mwg-rs:RegionList/rdf:Bag/rdf:li/rdf:Description", NS)
    assert region.get(f"{{{NS['mwg-rs']}}}Name") == name
    area = region.find("mwg-rs:Area", NS)
    assert float(area.get(f"{{{NS['stArea']}}}x")) == pytest.approx(140 / 400, abs=1e-4)
    assert float(area.get(f"{{{NS['stArea']}}}h")) == pytest.approx(100 / 300, abs=1e-4)
    assert "work_image=no (0.90)" in bag(desc, "vt:Properties")
    plain = parse(X.to_xmp(metas[2]))
    assert plain.get(f"{{{NS['exif']}}}DateTimeOriginal") is None  # file times are guesses


def test_embedding_keeps_the_image_and_is_readable(tmp_path):
    packet = X.to_xmp(X.PhotoMeta(1, "x", 10, 10, None, "", None, "ab", caption="hello"))
    for fmt, suffix in (("JPEG", ".jpg"), ("PNG", ".png")):
        path = tmp_path / f"a{suffix}"
        exif = Image.Exif()
        exif[0x0110] = "Camera"
        Image.fromarray(np.full((40, 60, 3), 90, np.uint8)).save(path, fmt, exif=exif)
        with Image.open(path) as im:
            before = np.asarray(im.convert("RGB")).copy()
        assert X.embed_into_copy(path, packet)
        with Image.open(path) as im:
            assert np.array_equal(np.asarray(im.convert("RGB")), before)  # pixels untouched
            raw = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")
            assert raw and b"hello" in (raw if isinstance(raw, bytes) else raw.encode())
            assert im.getexif().get(0x0110) == "Camera"  # EXIF kept
        assert not X.embed_into_copy(path, packet)  # existing XMP is never replaced
    webp = tmp_path / "a.webp"
    Image.new("RGB", (8, 8)).save(webp, "WEBP")
    assert not X.embed_into_copy(webp, packet)  # no simple XMP slot: sidecar only
    mislabeled = tmp_path / "png-named.jpg"
    Image.new("RGB", (8, 8)).save(mislabeled, "PNG")
    assert X.embed_into_copy(mislabeled, packet)  # format by content, not by name


def test_export_with_metadata(db, library, tmp_path, settings):
    originals = {p: sha(p) for p in library}
    out = tmp_path / "out"
    stats = export_images(db, ImageFilter(), out, metadata=("xmp", "embed", "index"))
    assert stats.exported == 2 and stats.sidecars == 2 and stats.embedded == 2
    assert all(sha(p) == h for p, h in originals.items())  # originals untouched
    assert b"sand castle" in (out / "photo0.jpg.xmp").read_bytes()
    assert b"sand castle" in (out / "photo0.jpg").read_bytes()  # embedded into the copy
    rows = list(csv.DictReader((out / "vision-tagger.csv").open(encoding="utf-8-sig")))
    assert [r["file"] for r in rows] == ["photo0.jpg", "photo1.png"]
    assert rows[0]["persons"] == 'Anna "Nan" & <Co>' and "beach" in rows[0]["keywords"]
    doc = json.loads((out / "vision-tagger.json").read_text(encoding="utf-8"))
    assert doc["photos"][0]["faces"][0]["name"] == 'Anna "Nan" & <Co>'
    export_images(db, ImageFilter(), out, metadata=("index",))  # exporting again: merged
    doc = json.loads((out / "vision-tagger.json").read_text(encoding="utf-8"))
    assert len(doc["photos"]) == 4  # the second run made copies with new names
    assert len({p["file"] for p in doc["photos"]}) == 4


def test_links_are_never_modified(db, library, tmp_path):
    original = sha(library[0])
    out = tmp_path / "links"
    stats = export_images(db, ImageFilter(), out, mode="hardlink", metadata=("xmp", "embed"))
    assert stats.embedded == 0 and stats.sidecars == 2
    assert sha(library[0]) == original and sha(out / "photo0.jpg") == original


def test_sidecars_next_to_originals_respect_other_apps(db, library):
    foreign = X.sidecar_path(library[1])
    foreign.write_text("<x:xmpmeta>darktable history</x:xmpmeta>")
    before = {p: sha(p) for p in library}
    stats = write_sidecars_in_place(db, ImageFilter())
    assert stats.sidecars == 1 and stats.kept == 1
    assert "darktable" in foreign.read_text()
    assert X.is_ours(X.sidecar_path(library[0]))
    assert {p: sha(p) for p in library} == before
    assert write_sidecars_in_place(db, ImageFilter()).sidecars == 1  # ours get updated
    assert X.sidecar_path(library[0], "replace").name == "photo0.xmp"


def test_web_export_and_sidecars(settings, library, db):
    from fastapi.testclient import TestClient

    from vision_tagger.web.app import create_app

    app = create_app(settings)
    with TestClient(app) as c:
        page = c.get("/images").text
        assert 'name="metadata" value="xmp" checked' in page and 'action="/images/sidecars"' in page
        out = Path(settings.data_dir).parent / "webout"
        r = c.post(
            "/images/export",
            data={"target": str(out), "query": "", "metadata": ["xmp", "index"]},
            follow_redirects=False,
        )
        assert r.status_code == 303
        _wait(app)
        assert (out / "photo0.jpg.xmp").exists() and (out / "vision-tagger.csv").exists()
        assert b"sand castle" not in (out / "photo0.jpg").read_bytes()  # "embed" not ticked
        assert (
            c.post("/images/sidecars", data={"query": ""}, follow_redirects=False).status_code
            == 303
        )
        _wait(app)
        assert X.sidecar_path(library[0]).exists()
        r = c.post(
            "/images/sidecars", data={"query": ""}, headers={"Origin": "http://evil.example"}
        )
        assert r.status_code == 403


def _wait(app, timeout=20):
    import time

    deadline = time.time() + timeout
    while app.state.ctx.jobs.current is not None and app.state.ctx.jobs.current.status == "running":
        assert time.time() < deadline
        time.sleep(0.05)
