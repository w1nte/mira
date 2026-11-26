"""Animals and pets: detector maths, grouping that keeps the user's work, properties, filters,
the stage, the pages and the search (fake models: no download, no GPU)."""

from __future__ import annotations

import hashlib
import math

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import func, select

from vision_tagger import xmp
from vision_tagger.animals import COCO_ANIMALS
from vision_tagger.animals import grouping as G
from vision_tagger.animals.detector import PAD_VALUE, grids, letterbox, nms, postprocess
from vision_tagger.animals.embedder import PAD_RGB, crop_box, pool, square_crop
from vision_tagger.animals.models import ModelFile, ModelUnavailable, ensure_model
from vision_tagger.animals.sync import desired, sync_all, sync_images
from vision_tagger.config import PipelineConfig, Settings
from vision_tagger.db.models import (
    Animal,
    Assignment,
    Cluster,
    Image,
    ImageProperty,
    Person,
    Pet,
    PropertyAlias,
    StageStatus,
    Status,
)
from vision_tagger.imaging import face_crop_path
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages import STAGE_NAMES, stage_versions
from vision_tagger.pipeline.stages.animals import AnimalsStage
from vision_tagger.query import ImageFilter, count_images, pet_counts
from vision_tagger.search.document import build_document
from vision_tagger.search.engine import SearchEngine
from vision_tagger.search.interpret import KeywordInterpreter, system_prompt
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app

from .conftest import make_image
from .helpers import (
    FakeAnimalAnalyzer,
    add_animal,
    add_face,
    add_image,
    animal_centers,
    near,
    person_centers,
)

DE = {"Accept-Language": "de"}


def meta_props(s, image_id: int) -> list[ImageProperty]:
    return list(s.scalars(select(ImageProperty).where(ImageProperty.image_id == image_id)))


def props_of(s, image_id: int) -> set[tuple[str, str]]:
    return {
        (p.key, p.value)
        for p in s.scalars(
            select(ImageProperty).where(
                ImageProperty.image_id == image_id, ImageProperty.key.in_(["pet", "animal"])
            )
        )
    }


# --------------------------------------------------------------------------- YOLOX maths
def test_letterbox_scales_pads_top_left_and_uses_bgr():
    red = PILImage.new("RGB", (1280, 720), (255, 0, 0))
    tensor, ratio = letterbox(red)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert ratio == pytest.approx(0.5)
    assert (tensor[0, 0, :360, :640] == 0).all()  # B
    assert (tensor[0, 2, :360, :640] == 255).all()  # R: channels are BGR
    assert (tensor[0, :, 360:, :] == PAD_VALUE).all()  # padding below, not centred
    tall, ratio = letterbox(PILImage.new("RGB", (300, 600), (0, 0, 255)))
    assert ratio == pytest.approx(640 / 600)
    assert (tall[0, 0, :, :320] == 255).all() and (tall[0, :, :, 320:] == PAD_VALUE).all()


def _cell(stride: int, gx: int, gy: int) -> int:
    offset = {8: 0, 16: 80 * 80, 32: 80 * 80 + 40 * 40}[stride]
    return offset + gy * (640 // stride) + gx


def _pred(raw, stride, gx, gy, ox, oy, w, h, obj, classes):
    i = _cell(stride, gx, gy)
    raw[i, :4] = [ox, oy, math.log(w / stride), math.log(h / stride)]
    raw[i, 4] = obj
    for cls, p in classes.items():
        raw[i, 5 + cls] = p


def test_yolox_postprocess_on_a_synthetic_output():
    cells, strides = grids()
    assert len(cells) == 8400 and strides[0, 0] == 8 and strides[-1, 0] == 32
    raw = np.zeros((8400, 85), dtype=np.float32)
    # a dog centred at (100, 200), 80 x 120 input pixels: score 0.9 * 0.9
    _pred(raw, 8, 12, 25, 0.5, 0.0, 80, 120, 0.9, {16: 0.9})
    # the same animal from the next cell, taken for a cat: merged into the dog by NMS
    _pred(raw, 8, 13, 25, -0.5, 0.0, 80, 120, 0.8, {15: 0.7})
    # a person over the dog: never reported, never suppresses the dog
    _pred(raw, 16, 6, 12, 0.25, 0.5, 90, 130, 0.95, {0: 0.95})
    # a teddy bear (best class 77) is not a bear
    _pred(raw, 32, 5, 5, 0.5, 0.5, 60, 60, 0.9, {77: 0.8, 21: 0.3})
    # a weak bird: below the threshold
    _pred(raw, 8, 60, 60, 0.5, 0.5, 30, 30, 0.6, {14: 0.5})
    # a second dog near the right edge: clipped to the image
    _pred(raw, 16, 30, 10, 0.5, 0.5, 64, 64, 0.8, {16: 0.875})
    found = postprocess(raw, 0.5, (1000, 700), score_threshold=0.5, nms_iou=0.45)
    assert [d.species for d in found] == ["dog", "dog"]
    assert found[0].score == pytest.approx(0.81)
    assert found[0].bbox == pytest.approx((120, 280, 280, 520))  # input pixels / ratio 0.5
    assert found[1].bbox == pytest.approx((912, 272, 1000, 400))
    assert found[1].score == pytest.approx(0.7)
    # restricted to some classes; nothing over the threshold gives nothing
    assert postprocess(raw, 0.5, (1000, 700), score_threshold=0.5, classes={14: "bird"}) == []
    only_cats = postprocess(raw, 0.5, (1000, 700), score_threshold=0.5, classes={15: "cat"})
    assert [(d.species, round(d.score, 2)) for d in only_cats] == [("cat", 0.56)]
    assert postprocess(raw, 0.5, (1000, 700), score_threshold=0.95) == []
    assert set(COCO_ANIMALS.values()) >= {"cat", "dog", "bird", "horse", "giraffe"}


def test_nms_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45) == [0, 2]


# --------------------------------------------------------------------------- DINOv2 crops
def test_crop_box_margin_and_clamping():
    assert crop_box((400, 300), (100, 100, 300, 200), 0.1) == (80, 90, 320, 210)
    assert crop_box((400, 300), (0, 0, 50, 50), 0.1) == (0, 0, 55, 55)
    assert crop_box((400, 300), (390, 290, 400, 300), 0.5) == (385, 285, 400, 300)


def test_square_crop_pads_instead_of_stretching():
    im = PILImage.new("RGB", (400, 300), (10, 200, 30))
    crop = square_crop(im, (100, 100, 300, 200), margin=0.1, size=224)
    assert crop.size == (224, 224)
    arr = np.asarray(crop).astype(int)
    assert tuple(arr[2, 112]) == PAD_RGB  # wide animal: padded above and below
    assert tuple(arr[112, 2]) == (10, 200, 30)  # but the full width is the animal
    assert np.abs(arr[112, 112] - (10, 200, 30)).max() <= 1


def test_pool_uses_the_class_token():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(2, 257, 384)).astype(np.float32)
    vec = pool(hidden)
    assert vec.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vec, axis=1), 1)
    assert np.allclose(vec[0], hidden[0, 0] / np.linalg.norm(hidden[0, 0]))
    assert pool(hidden, "cls+mean").shape == (2, 768)


# --------------------------------------------------------------------------- downloads
def _spec(data: bytes) -> ModelFile:
    return ModelFile(
        name="tiny.onnx",
        url="https://example.invalid/tiny.onnx",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        license="Apache-2.0",
        title="Tiny",
    )


def test_model_download_is_verified_and_atomic(tmp_path):
    data = b"onnx" * 1000
    spec = _spec(data)
    calls = []

    def ok(request):
        calls.append(request.url)
        return httpx.Response(200, content=data)

    path = ensure_model(tmp_path, spec, client=httpx.Client(transport=httpx.MockTransport(ok)))
    assert path.read_bytes() == data and len(calls) == 1
    ensure_model(tmp_path, spec, client=httpx.Client(transport=httpx.MockTransport(ok)))
    assert len(calls) == 1  # present and correct: not downloaded again

    path.write_bytes(b"x" * len(data))  # same size, other content: replaced
    ensure_model(tmp_path, spec, client=httpx.Client(transport=httpx.MockTransport(ok)))
    assert path.read_bytes() == data and len(calls) == 2


def test_model_download_errors_are_clear(tmp_path):
    spec = _spec(b"model")

    def tampered(request):
        return httpx.Response(200, content=b"evil!")

    with pytest.raises(ModelUnavailable, match="not the expected file"):
        ensure_model(tmp_path, spec, client=httpx.Client(transport=httpx.MockTransport(tampered)))

    def offline(request):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(ModelUnavailable, match="cannot download the Tiny model") as err:
        ensure_model(tmp_path, spec, client=httpx.Client(transport=httpx.MockTransport(offline)))
    assert spec.url in str(err.value) and spec.sha256 in str(err.value)
    assert list(tmp_path.iterdir()) == []  # nothing half-written is left behind


# --------------------------------------------------------------------------- grouping
@pytest.fixture
def pets_db(db):
    """Three photos each of two dogs and a cat, plus a photo of a bird (all ungrouped)."""
    rng = np.random.default_rng(3)
    centers = animal_centers(4)
    ids: dict[str, list[int]] = {"rex": [], "lucky": [], "tom": [], "tweety": []}
    with db.write() as s:
        n = 0
        for key, species, c, count in (
            ("rex", "dog", 0, 3),
            ("lucky", "dog", 1, 3),
            ("tom", "cat", 2, 3),
            ("tweety", "bird", 3, 1),
        ):
            for _ in range(count):
                img = add_image(s, n)
                img.stages.append(StageStatus(stage="embed", status=Status.DONE))
                ids[key].append(add_animal(s, img.id, near(centers[c], rng), species=species).id)
                n += 1
    sync_all(db)  # the species, as the stage writes them when it finds the animals
    return db, ids, centers


def groups(s) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for a in s.scalars(select(Animal).where(Animal.pet_id.is_not(None))):
        out.setdefault(a.pet_id, set()).add(a.id)
    return out


def test_regroup_finds_each_animal(pets_db, settings):
    db, ids, _ = pets_db
    with db.write() as s:
        stats = G.regroup(s, settings.animals, full=True)
    assert stats.new_pets == 3 and stats.noise == 1
    with db.session() as s:
        found = sorted(groups(s).values(), key=min)
        assert found == [set(ids["rex"]), set(ids["lucky"]), set(ids["tom"])]
        species = {p.species for p in s.scalars(select(Pet))}
        assert species == {"dog", "cat"}
        assert s.get(Animal, ids["tweety"][0]).pet_id is None
        # the photos got their species
        assert props_of(s, s.get(Animal, ids["tom"][0]).image_id) == {("animal", "cat")}


def test_species_pools_group_a_dog_taken_for_a_bear(pets_db, settings):
    db, ids, _centers = pets_db
    with db.write() as s:
        s.get(Animal, ids["rex"][0]).species = "bear"  # the detector's mistake
    with db.write() as s:
        G.regroup(s, settings.animals, full=True)
    with db.session() as s:
        rex = s.get(Animal, ids["rex"][0])
        assert rex.pet_id is not None and rex.pet.species == "dog"  # most photos say dog
        assert {a.id for a in rex.pet.animals} == set(ids["rex"])
        assert props_of(s, rex.image_id) == {("animal", "dog")}
    settings.animals.species_pools = []  # strictly per species: the bear photo stays alone
    with db.write() as s:
        G.regroup(s, settings.animals, full=True)
    with db.session() as s:
        assert s.get(Animal, ids["rex"][0]).pet_id != s.get(Animal, ids["rex"][1]).pet_id


def test_names_moves_and_removals_survive_regrouping_and_redetection(pets_db, settings):
    db, ids, _ = pets_db
    cfg = settings.animals
    with db.write() as s:
        G.regroup(s, cfg, full=True)
        rex = s.get(Animal, ids["rex"][0]).pet_id
        lucky = s.get(Animal, ids["lucky"][0]).pet_id
        tom = s.get(Animal, ids["tom"][0]).pet_id
        G.name_pet(s, rex, "Bello")
        G.move_animals(s, [ids["lucky"][0]], rex)  # "this one is Bello too"
        G.ignore_animals(s, [ids["rex"][2]])  # not an animal
        G.unassign_animals(s, [ids["lucky"][1]])  # not Lucky
    with db.write() as s:
        G.regroup(s, cfg, full=True)
        G.regroup(s, cfg, full=True)
    with db.session() as s:
        bello = s.get(Pet, rex)
        assert bello.name == "Bello" and bello.locked
        assert {a.id for a in bello.animals if not a.ignored} == {
            ids["rex"][0],
            ids["rex"][1],
            ids["lucky"][0],
        }
        removed = s.get(Animal, ids["rex"][2])
        assert removed.ignored and removed.pet_id is None
        taken_out = s.get(Animal, ids["lucky"][1])
        assert taken_out.pet_id is None and taken_out.assignment == Assignment.MANUAL
        assert s.get(Pet, lucky) is not None  # edited (locked): kept with its last photo
        assert s.get(Animal, ids["tom"][0]).pet_id is not None
        assert s.get(Pet, tom) is None or s.get(Pet, tom).animal_count == 3
        image = s.get(Animal, ids["lucky"][0]).image_id
        assert props_of(s, image) == {("animal", "dog"), ("pet", "Bello")}

    # the photos are analysed again (e.g. a newer model): pet, manual and removed carry over
    with db.write() as s:
        for animal_id in (ids["lucky"][0], ids["rex"][2]):
            old = s.get(Animal, animal_id)
            new = Animal(
                x1=old.x1 + 2,
                y1=old.y1,
                x2=old.x2,
                y2=old.y2 + 3,
                species=old.species,
                score=0.95,
                embedding=old.embedding,
                box_px=old.box_px,
            )
            removed_ids = G.replace_animals(s, old.image_id, [new])
            assert removed_ids == [animal_id]
    with db.session() as s:
        bello = s.get(Pet, rex)
        moved = [a for a in bello.animals if a.assignment == Assignment.MANUAL]
        assert len(moved) == 1 and moved[0].id not in ids["lucky"]
        assert s.scalar(select(func.count()).select_from(Animal).where(Animal.ignored)) == 1
        assert bello.animal_count == 3


def test_new_photos_join_named_pets_and_named_pets_are_kept(pets_db, settings):
    db, ids, centers = pets_db
    cfg = settings.animals
    rng = np.random.default_rng(11)
    with db.write() as s:
        G.regroup(s, cfg, full=True)
        rex = s.get(Animal, ids["rex"][0]).pet_id
        G.name_pet(s, rex, "Bello")
        G.ignore_animals(s, ids["rex"])  # every photo removed: Bello stays, without photos
    with db.session() as s:
        assert s.get(Pet, rex).animal_count == 0 and s.get(Pet, rex).name == "Bello"
    with db.write() as s:
        img_id = add_image(s, 50).id
        close = add_animal(s, img_id, near(centers[0], rng, r=0.3)).id
        far = add_animal(s, img_id, near(animal_centers(5)[4], rng), x1=400, x2=600).id
        G.regroup(s, cfg)  # incremental: Bello's average still attracts its photos
    with db.session() as s:
        assert s.get(Animal, close).pet_id == rex
        assert s.get(Animal, far).pet_id is None
        assert props_of(s, img_id) == {("animal", "dog"), ("pet", "Bello")}
    with db.write() as s:  # an unnamed group that loses its photos disappears
        tom = s.get(Animal, ids["tom"][0]).pet_id
        G.ignore_animals(s, ids["tom"])
        assert s.get(Pet, tom) is None


def test_naming_with_an_existing_name_merges_and_species_can_be_corrected(pets_db, settings):
    db, ids, _ = pets_db
    with db.write() as s:
        G.regroup(s, settings.animals, full=True)
        rex = s.get(Animal, ids["rex"][0]).pet_id
        lucky = s.get(Animal, ids["lucky"][0]).pet_id
        G.name_pet(s, rex, "  Bello ")
        merged = G.name_pet(s, lucky, "bello")  # same name, other spelling: one pet
        assert merged.id == rex and s.get(Pet, lucky) is None
        assert merged.animal_count == 6 and merged.name == "Bello"
        G.set_pet_species(s, rex, "bear")
        with pytest.raises(ValueError):
            G.set_pet_species(s, rex, "unicorn")
    with db.session() as s:
        assert props_of(s, s.get(Animal, ids["lucky"][2]).image_id) == {
            ("animal", "bear"),
            ("pet", "Bello"),
        }
        assert G.similar_pets(s, rex)  # the cat group
        assert [p.name for p, _n in pet_counts(s)] == ["Bello"]


def test_move_to_a_name_creates_the_pet(pets_db, settings):
    db, ids, _ = pets_db
    with db.write() as s:
        pet_id = G.move_animals_to_name(s, [ids["tweety"][0]], "Hansi")
        assert s.get(Pet, pet_id).name == "Hansi" and s.get(Pet, pet_id).species == "bird"
        assert G.move_animals_to_name(s, [ids["tom"][0]], "hansi") == pet_id
        with pytest.raises(ValueError):
            G.move_animals_to_name(s, [ids["tom"][1]], "   ")


# --------------------------------------------------------------------------- properties
def test_properties_follow_names_aliases_and_only_species_requeue_the_search(pets_db, settings):
    db, ids, _ = pets_db
    with db.write() as s:
        G.regroup(s, settings.animals, full=True)
        rex = s.get(Animal, ids["rex"][0]).pet_id
        image = s.get(Animal, ids["rex"][0]).image_id
        for st in s.scalars(select(StageStatus).where(StageStatus.stage == "embed")):
            st.status = Status.DONE
        G.name_pet(s, rex, "Bello")
    with db.session() as s:
        assert props_of(s, image) == {("animal", "dog"), ("pet", "Bello")}
        source = s.scalars(select(ImageProperty.source).where(ImageProperty.key == "pet")).first()
        assert source == "animal"
        meta = xmp.collect(s, [s.get(Image, image)])[image]  # other photo apps get them too
        assert meta.keywords[:2] == ["Bello", "dog"]
        assert "vision-tagger|Pet|Bello" in meta.hierarchical
        assert "Animal: dog." in build_document(s.get(Image, image), meta_props(s, image), None)
        assert "Bello" not in build_document(s.get(Image, image), meta_props(s, image), None)
        # pet names are not part of the search document: no re-embedding for a name
        assert s.get(StageStatus, (image, "embed")).status == Status.DONE
    with db.write() as s:
        s.add(PropertyAlias(key="pet", from_value="Bello", to_value="Bello the dog"))
    with db.write() as s:
        sync_images(s, [image])
        G.set_pet_species(s, rex, "sheep")
    with db.session() as s:
        assert props_of(s, image) == {("animal", "sheep"), ("pet", "Bello the dog")}
        assert s.get(StageStatus, (image, "embed")).status == Status.PENDING
    # a photo whose animal is removed loses the properties; sync_all finds stale rows
    with db.write() as s:
        G.ignore_animals(s, [ids["rex"][0]])
        s.add(ImageProperty(image_id=image, key="pet", value="Stale", source="animal"))
    assert sync_all(db) == 1
    with db.session() as s:
        assert props_of(s, image) == set()
        assert desired(s, [image]) == {}


# --------------------------------------------------------------------------- filters
def test_pet_filter_and_or_with_persons(pets_db):
    db, ids, _ = pets_db
    f = ImageFilter.from_params({"pet": ["3", "x"], "xpet": "4", "p": "1", "pm": "any"})
    assert f.pets == [3] and f.exclude_pets == [4] and f.who_count == 2
    assert ImageFilter.from_params(dict(f.to_params())).to_params() == f.to_params()
    assert ("pm", "any") in f.to_params() and "pet=3" in f.with_pet(5)
    assert f.without_pet(3) == "p=1&xpet=4"
    with db.write() as s:
        rex = G.move_animals_to_name(s, ids["rex"], "Bello")
        anna = Cluster(locked=True, person=Person(name="Anna"))
        s.add(anna)
        s.flush()
        rex_images = [s.get(Animal, a).image_id for a in ids["rex"]]
        add_face(s, rex_images[0], person_centers(1)[0], cluster_id=anna.id)
        add_face(s, s.get(Animal, ids["tom"][0]).image_id, person_centers(1)[0], cluster_id=anna.id)
        person = anna.person_id
    with db.session() as s:
        assert count_images(s, ImageFilter(pets=[rex])) == 3
        assert count_images(s, ImageFilter(pets=[rex], persons=[person])) == 1  # AND
        assert count_images(s, ImageFilter(pets=[rex], persons=[person], persons_mode="any")) == 4
        assert count_images(s, ImageFilter(exclude_pets=[rex])) == 7
        assert [n for _p, n in pet_counts(s, ImageFilter(persons=[person]))] == [1]


# --------------------------------------------------------------------------- the stage
def test_stage_runs_in_the_runner_and_keeps_names(db, settings, photos):
    settings.animals.max_image_px = 200  # boxes must be scaled back to the original
    for i in range(4):
        make_image(photos / f"{i}.jpg", seed=i, size=(800, 600))
    Indexer(db, settings, stages=["animals"]).run()
    analyzer = FakeAnimalAnalyzer()
    stage = AnimalsStage(db, settings, analyzer=analyzer)
    summary = PipelineRunner(db, settings, lambda _n: stage).run(["animals"])
    assert summary.status == "completed", summary.messages
    assert "full grouping" in summary.messages[0] and analyzer.calls == 4
    with db.session() as s:
        animals = s.scalars(select(Animal)).all()
        assert len(animals) == 4
        a = animals[0]
        assert a.x1 == pytest.approx(40, abs=2) and a.x2 == pytest.approx(400, abs=2)
        assert a.box_px == pytest.approx(300, abs=3) and a.groupable
        assert all(face_crop_path(settings.animal_crops_dir, x.id).exists() for x in animals)
        assert s.scalar(select(func.count()).select_from(Pet)) == 1
        pet_id = animals[0].pet_id
        statuses = {st.status for st in s.scalars(select(StageStatus))}
        assert statuses == {Status.DONE}
        assert s.get(StageStatus, (1, "animals")).stage_version == stage.version
    with db.write() as s:
        G.name_pet(s, pet_id, "Bello")
    stage.process(1)  # analysed again: same pet, same name, no duplicates
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(Animal)) == 4
        again = s.scalar(select(Animal).where(Animal.image_id == 1))
        assert again.pet_id == pet_id
        assert props_of(s, 1) == {("animal", "dog"), ("pet", "Bello")}

    # small or uncertain animals are kept but not grouped; tiny ones are dropped
    weak = FakeAnimalAnalyzer(
        lambda image: [
            ("cat", (0.1, 0.1, 0.2, 0.2), animal_centers(2)[1]),  # 80 px: kept, not grouped
            ("cat", (0.5, 0.5, 0.52, 0.52), animal_centers(2)[1]),  # 16 px: dropped
        ]
    )
    AnimalsStage(db, settings, analyzer=weak).process(2)
    with db.session() as s:
        cats = s.scalars(select(Animal).where(Animal.image_id == 2)).all()
        assert [(c.species, c.groupable) for c in cats] == [("cat", False)]


def test_stage_registry_and_config():
    assert STAGE_NAMES.index("animals") == STAGE_NAMES.index("ages") + 1
    assert Settings().pipeline.stages == ["visual", "faces", "ages", "animals", "vlm", "embed"]
    assert PipelineConfig(stages=["animals"]).stages == ["animals"]
    version = stage_versions(Settings())["animals"]
    assert version.startswith("yolox_s:c5c2d13e+dinov2s:f22797ea")


# --------------------------------------------------------------------------- search
@pytest.fixture
def pet_library(db, settings):
    """Photo 1: Anna with Bello at the beach; 2: Anna; 3: Minka at the beach; 4: Bello in
    winter."""
    centers = animal_centers(2)
    with db.write() as s:
        anna = Cluster(locked=True, person=Person(name="Anna"))
        s.add(anna)
        s.flush()
        images = [add_image(s, n) for n in range(1, 5)]
        image_ids = [img.id for img in images]
        for img, key, value in (
            (images[0], "location_type", "beach"),
            (images[2], "location_type", "beach"),
            (images[3], "season", "winter"),
        ):
            s.add(ImageProperty(image_id=img.id, key=key, value=value, source="vlm"))
        for img in images[:2]:
            add_face(s, img.id, person_centers(1)[0], cluster_id=anna.id)
        bello = [add_animal(s, images[i].id, centers[0]).id for i in (0, 3)]
        minka = add_animal(s, images[2].id, centers[1], species="cat").id
        G.move_animals_to_name(s, bello, "Bello")
        G.move_animals_to_name(s, [minka], "Minka")
    return db, image_ids


def search(db, settings, query, chat=None):
    engine = SearchEngine(
        db, settings, backend=FakeBackend(Vocabulary.load(), chat_responses=chat or [])
    )
    return engine, engine.search(query)


def test_keyword_search_with_pet_names(pet_library, settings):
    db, ids = pet_library
    _engine, result = search(db, settings, "Bello am Strand")
    it = result.interpretation
    assert [n for _i, n in it.pets] == ["Bello"] and not it.persons
    assert [h.image_id for h in result.hits] == [ids[0], ids[3]]  # strict: only Bello
    assert result.hits[0].properties > 0  # the beach photo first
    _engine, result = search(db, settings, "photos of Minka in winter")
    assert [h.image_id for h in result.hits] == [ids[2]]
    _engine, result = search(db, settings, "Anna with Bello")
    assert [h.image_id for h in result.hits] == [ids[0]]
    _engine, result = search(db, settings, "Anna ohne Bello")
    assert [n for _i, n in result.interpretation.exclude_pets] == ["Bello"]
    assert [h.image_id for h in result.hits] == [ids[1]]


def test_llm_search_knows_the_pets(pet_library, settings):
    db, ids = pet_library
    answer = {  # the LLM put the dog among the persons: it still counts as the pet
        "persons": ["Anna", "bello"],
        "persons_mode": "all",
        "exclude_persons": [],
        "must": [],
        "should": [{"key": "location_type", "value": "beach", "weight": 1}],
        "exclude": [],
        "date_from": "",
        "date_to": "",
        "semantic_query": "a woman with her dog at the beach",
        "keywords": [],
    }
    engine, result = search(db, settings, "Anna mit Bello am Strand", chat=[answer])
    it = result.interpretation
    assert it.mode == "llm"
    assert [n for _i, n in it.persons] == ["Anna"] and [n for _i, n in it.pets] == ["Bello"]
    assert [h.image_id for h in result.hits] == [ids[0]]
    prompt = system_prompt(engine.context())
    assert "Pets: Bello (dog), Minka (cat)" in prompt
    assert "- animal: bird, cat, dog" in prompt  # the detector's species are known values
    it = KeywordInterpreter().interpret("Minka oder Bello", engine.context())
    assert sorted(n for _i, n in it.pets) == ["Bello", "Minka"]


# --------------------------------------------------------------------------- web UI
@pytest.fixture
def client(settings, photos):
    dog, cat = animal_centers(2)

    def two_animals(image):  # wide photos show the dog, narrow ones the cat
        if image.width >= 320:
            return [("dog", (0.05, 0.45, 0.5, 0.95), dog)]
        return [("cat", (0.4, 0.1, 0.9, 0.6), cat)]

    for i in range(6):
        make_image(photos / f"{i}.jpg", seed=i, size=(320, 240) if i % 2 else (300, 240))
    app = create_app(settings)
    with TestClient(app) as c:
        ctx = app.state.ctx
        Indexer(ctx.db, settings, stages=["animals"]).run()
        stage = AnimalsStage(ctx.db, settings, analyzer=FakeAnimalAnalyzer(two_animals))
        assert PipelineRunner(ctx.db, settings, lambda _n: stage).run(["animals"]).status == (
            "completed"
        )
        c.ctx = ctx
        yield c


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


def test_animal_pages_render_in_english_and_german(client):
    with client.ctx.db.session() as s:
        pets = {p.species: p.id for p in s.scalars(select(Pet))}
        animal = s.scalar(select(Animal).where(Animal.pet_id == pets["dog"]))
        animal_id, image_id = animal.id, animal.image_id
    urls = [
        "/animals",
        "/animals?show=unnamed&species=dog&sort=newest&q=x",
        f"/pets/{pets['dog']}",
        "/animals/ungrouped",
        "/animals/ignored",
        f"/images/{image_id}",
        f"/images?pet={pets['dog']}",
        "/images",
        "/properties",
        "/properties/animal",
        "/pipeline",
        "/partials/status",
        "/",
    ]
    for url in urls:
        ok(client.get(url))
        ok(client.get(url, headers=DE))
    page = ok(client.get("/animals")).text
    assert "Dog #" in page and "Cat #" in page and "/media/animal/" in page
    page = ok(client.get("/animals", headers=DE)).text
    assert "Tiere" in page and "Hund #" in page and "Katze" in page
    page = ok(client.get(f"/images/{image_id}", headers=DE)).text
    assert "animalbox" in page and "Tierart" in page and "Hund" in page
    assert ok(client.get(f"/media/animal/{animal_id}")).headers["content-type"] == "image/webp"
    assert client.get("/media/animal/999999").status_code == 404
    page = ok(client.get("/pipeline", headers=DE)).text
    assert "Stufe 1c · YOLOX + DINOv2" in page or "YOLOX + DINOv2" in page
    assert ok(client.get("/api/pets")).json()[0]["species"] in ("dog", "cat")


def test_naming_merging_and_moving_from_the_ui(client):
    with client.ctx.db.session() as s:
        pets = {p.species: p.id for p in s.scalars(select(Pet))}
        cats = list(s.scalars(select(Animal.id).where(Animal.pet_id == pets["cat"])))
    dog = pets["dog"]
    card = client.post(
        f"/pets/{dog}/name", data={"name": "Bello", "card": "1"}, headers={"HX-Request": "true"}
    )
    assert card.status_code == 200 and 'value="Bello"' in card.text and "pet-card" in card.text
    page = ok(client.get(f"/images?pet={dog}")).text
    assert "Bello" in page and "filter-summary" in page  # the chip shows the name
    page = ok(client.get(f"/images?pet={dog}", headers=DE)).text
    assert "Haustiere" in page or "Bello" in page
    home = ok(client.get("/")).text
    assert f"pet={dog}" in home  # a collection for the named pet

    r = client.post("/animals/action", data={"action": "unassign", "animal_ids": [str(cats[0])]})
    assert r.status_code in (200, 303)
    client.post("/animals/action", data={"action": "pet", "animal_ids": [str(cats[1])], "name": ""})
    client.post(
        "/animals/action", data={"action": "pet", "animal_ids": [str(cats[1])], "name": "Bello"}
    )
    client.post("/animals/action", data={"action": "ignore", "animal_ids": [str(cats[2])]})
    client.post("/animals/action", data={"action": "unignore", "animal_ids": [str(cats[2])]})
    client.post(
        "/animals/action",
        data={"action": "new", "animal_ids": [str(cats[0])], "name": "Minka"},
    )
    client.post(f"/pets/{dog}/species", data={"species": "sheep"})
    with client.ctx.db.session() as s:
        assert s.get(Animal, cats[1]).pet_id == dog
        minka = s.scalar(select(Pet).where(Pet.name == "Minka"))
        assert minka is not None and s.get(Animal, cats[0]).pet_id == minka.id
        assert s.get(Pet, dog).species == "sheep"
        assert not s.get(Animal, cats[2]).ignored
    client.post("/pets/merge", data={"pet_ids": [str(dog), str(minka.id)]})
    with client.ctx.db.session() as s:
        assert s.get(Pet, minka.id) is None  # the named, larger group is kept
        assert s.get(Animal, cats[0]).pet_id == dog
    r = client.post("/pets/merge", data={"pet_ids": [str(dog)]})
    assert r.status_code in (200, 303)
    assert client.post("/pets/99999/name", data={"name": "x"}).status_code == 404
    r = client.post("/actions/regroup-animals?full=1")
    assert r.status_code in (200, 303)


def test_animal_actions_need_post_from_this_ui(client):
    r = client.post(
        "/animals/action",
        data={"action": "ignore", "animal_ids": ["1"]},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403
    assert client.get("/animals/action").status_code == 405
