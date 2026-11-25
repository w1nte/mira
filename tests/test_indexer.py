from __future__ import annotations

import shutil
from datetime import datetime

from sqlalchemy import func, select

from vision_tagger.db.models import DuplicateFile, Image, ImageProperty, StageStatus, Status
from vision_tagger.imaging import date_from_filename
from vision_tagger.indexer import Indexer, path_key

from .conftest import make_image, set_mtime


def test_date_from_filename_patterns() -> None:
    assert date_from_filename("IMG_20230102_030405.jpg") == datetime(2023, 1, 2, 3, 4, 5)
    assert date_from_filename("PXL_20221231_235959123.jpg") == datetime(2022, 12, 31, 23, 59, 59)
    assert date_from_filename("WhatsApp Image 2021-07-08 at 09.10.11.jpeg") == datetime(
        2021, 7, 8, 9, 10, 11
    )
    assert date_from_filename("Screenshot_2020-02-03.png") == datetime(2020, 2, 3)
    assert date_from_filename("holiday.jpg") is None
    assert date_from_filename("a1b2c3d4e5f6.jpg") is None


def test_index_creates_rows_and_dedups(db, settings, photos) -> None:
    make_image(
        photos / "a.jpg",
        seed=1,
        date=datetime(2020, 5, 6, 7, 8, 9),
        camera=("Canon", "Canon EOS R5"),
    )
    make_image(photos / "sub" / "b.png", seed=2)
    make_image(photos / "sub" / "deeper" / "IMG_20190101_101010.jpg", seed=3)
    shutil.copy(photos / "a.jpg", photos / "sub" / "a_copy.jpg")
    (photos / "notes.txt").write_text("not an image")
    (photos / "broken.jpg").write_bytes(b"not really a jpeg")

    stats = Indexer(db, settings).run()

    assert stats.found == 5
    assert stats.new == 3
    assert stats.duplicates == 1
    assert stats.errors == 1
    with db.session() as s:
        images = {i.filename: i for i in s.scalars(select(Image))}
        assert set(images) == {"a.jpg", "b.png", "IMG_20190101_101010.jpg"}
        a = images["a.jpg"]
        assert a.recorded_at == datetime(2020, 5, 6, 7, 8, 9)
        assert a.date_source == "exif"
        assert a.camera == "Canon EOS R5"
        assert a.meta["make"] == "Canon"
        assert len(a.phash) == 16
        assert images["IMG_20190101_101010.jpg"].date_source == "filename"
        assert images["b.png"].date_source == "mtime"
        assert images["b.png"].folder == "sub"
        dup = s.scalars(select(DuplicateFile)).one()
        assert dup.image_id == a.id
        assert dup.path.endswith("a_copy.jpg")
        # every image has a pending row per stage
        rows = s.scalars(select(StageStatus)).all()
        assert len(rows) == 3 * len(settings.pipeline.stages)
        assert all(r.status == Status.PENDING for r in rows)


def test_rescan_is_incremental(db, settings, photos) -> None:
    make_image(photos / "a.jpg", seed=1)
    make_image(photos / "b.jpg", seed=2)
    Indexer(db, settings).run()

    stats = Indexer(db, settings).run()
    assert stats.unchanged == 2
    assert stats.candidates == 0

    make_image(photos / "c.jpg", seed=3)
    stats = Indexer(db, settings).run()
    assert stats.new == 1 and stats.unchanged == 2


def test_moved_file_is_repointed(db, settings, photos) -> None:
    make_image(photos / "old" / "a.jpg", seed=1)
    Indexer(db, settings).run()
    (photos / "new").mkdir()
    shutil.move(photos / "old" / "a.jpg", photos / "new" / "a.jpg")

    stats = Indexer(db, settings).run()
    assert stats.moved == 1
    with db.session() as s:
        img = s.scalars(select(Image)).one()
        assert img.path_key == path_key(photos / "new" / "a.jpg")
        assert img.folder == "new"
        assert not img.missing
        assert s.scalar(select(func.count()).select_from(DuplicateFile)) == 0


def test_missing_files_are_flagged(db, settings, photos) -> None:
    make_image(photos / "a.jpg", seed=1)
    make_image(photos / "b.jpg", seed=2)
    Indexer(db, settings).run()
    (photos / "b.jpg").unlink()
    stats = Indexer(db, settings).run()
    assert stats.missing == 1
    with db.session() as s:
        assert s.scalar(select(Image.missing).where(Image.filename == "b.jpg")) is True


def test_changed_content_requeues(db, settings, photos) -> None:
    p = make_image(photos / "a.jpg", seed=1)
    Indexer(db, settings).run()
    with db.write() as s:
        for row in s.scalars(select(StageStatus)):
            row.status = Status.DONE
    make_image(p, seed=99)
    set_mtime(p, datetime(2024, 1, 1))
    stats = Indexer(db, settings).run()
    assert stats.updated == 1
    with db.session() as s:
        assert {r.status for r in s.scalars(select(StageStatus))} == {Status.PENDING}
        keys = {p.key for p in s.scalars(select(ImageProperty))}
        assert {"year", "orientation", "capture_device"} <= keys  # exif props survive


def test_same_phash_but_different_content_is_not_a_duplicate(db, settings, photos, monkeypatch):
    """pHash collisions (blank pages, same-template forms) must not drop distinct images."""
    import imagehash as ih

    real_phash = ih.phash
    monkeypatch.setattr(ih, "phash", lambda im: real_phash(Image_new_blank()))
    make_image(photos / "form1.png", seed=1)
    make_image(photos / "form2.png", seed=2)
    shutil.copy(photos / "form1.png", photos / "form1_copy.png")
    stats = Indexer(db, settings).run()
    assert stats.new == 2  # same pHash, different dHash -> two images
    assert stats.duplicates == 1  # exact copy is still a duplicate
    with db.session() as s:
        hashes = {i.phash for i in s.scalars(select(Image))}
        assert len(hashes) == 1


def Image_new_blank():
    from PIL import Image as PILImage

    return PILImage.new("RGB", (64, 64), "white")


def test_duplicate_is_promoted_when_original_disappears(db, settings, photos):
    make_image(photos / "orig" / "a.jpg", seed=1)
    Indexer(db, settings).run()
    (photos / "copies").mkdir()
    shutil.copy(photos / "orig" / "a.jpg", photos / "copies" / "a.jpg")
    Indexer(db, settings).run()
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(DuplicateFile)) == 1
    (photos / "orig" / "a.jpg").unlink()
    stats = Indexer(db, settings).run()
    assert stats.missing == 0
    with db.session() as s:
        img = s.scalars(select(Image)).one()
        assert img.path_key == path_key(photos / "copies" / "a.jpg")
        assert img.folder == "copies" and not img.missing
        assert s.scalar(select(func.count()).select_from(DuplicateFile)) == 0


def test_vanished_duplicate_records_are_removed(db, settings, photos):
    make_image(photos / "a.jpg", seed=1)
    shutil.copy(photos / "a.jpg", photos / "b.jpg")
    Indexer(db, settings).run()
    (photos / "b.jpg").unlink()
    Indexer(db, settings).run()
    with db.session() as s:
        assert s.scalar(select(func.count()).select_from(DuplicateFile)) == 0


def test_folders_are_stored_with_forward_slashes(db, settings, photos):
    make_image(photos / "2021" / "Italy" / "a.jpg", seed=1)
    Indexer(db, settings).run()
    with db.session() as s:
        img = s.scalars(select(Image)).one()
        assert img.folder == "2021/Italy" and img.rel_path == "2021/Italy/a.jpg"


def test_scanner_skips_system_dirs_and_survives_link_loops(tmp_path):
    from pathlib import Path

    import pytest

    from vision_tagger.indexer import iter_image_files

    root = tmp_path / "r"
    make_image(root / "a.jpg", seed=1)
    make_image(root / "$RECYCLE.BIN" / "deleted.jpg", seed=2)
    make_image(root / "sub" / "b.jpg", seed=3)
    try:
        (root / "sub" / "loop").symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    found = sorted(
        Path(p).name
        for p, _ in iter_image_files(
            root, [".jpg"], follow_symlinks=True, skip_dirs=["$RECYCLE.BIN"]
        )
    )
    assert found == ["a.jpg", "b.jpg"]
