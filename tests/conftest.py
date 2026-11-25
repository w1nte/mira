from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageDraw

from vision_tagger.config import Settings
from vision_tagger.db.session import Database


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        paths={"data_dir": tmp_path / "data", "image_roots": [tmp_path / "photos"]},
        vlm={"backend": "fake"},
        pipeline={"batch_size": 4},
        index={"workers": 2, "commit_every": 3},
        web={"allowed_hosts": ["testserver"]},
    )
    s.ensure_dirs()
    (tmp_path / "photos").mkdir(exist_ok=True)
    return s


@pytest.fixture
def db(settings: Settings) -> Database:
    database = Database(settings.db_path)
    database.create_all()
    yield database
    database.dispose()


def make_image(
    path: Path,
    *,
    seed: int = 0,
    size: tuple[int, int] = (320, 240),
    date: datetime | None = None,
    camera: tuple[str, str] | None = None,
    fmt: str | None = None,
) -> Path:
    """Create a structurally distinct image (different seeds -> different pHash)."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, size=(6, 8, 3), dtype=np.uint8)
    im = Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    for _ in range(4):
        x, y = int(rng.integers(0, size[0])), int(rng.integers(0, size[1]))
        draw.ellipse(
            [x - 30, y - 30, x + 30, y + 30], fill=tuple(int(v) for v in rng.integers(0, 255, 3))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if camera:
        exif[271], exif[272] = camera
    if date:
        exif.get_ifd(ExifTags.IFD.Exif)[36867] = date.strftime("%Y:%m:%d %H:%M:%S")
    kwargs = {}
    if camera or date:
        kwargs["exif"] = exif
    im.save(path, fmt, **kwargs)
    return path


@pytest.fixture
def photos(settings: Settings) -> Path:
    return Path(settings.paths.image_roots[0])


def set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))
