"""Export a filter result into a folder (copy, hardlink or symlink). Originals stay untouched."""

from __future__ import annotations

import csv
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy import select

from .db.models import Image
from .db.session import Database
from .query import ImageFilter, all_image_ids

log = logging.getLogger(__name__)

Mode = Literal["copy", "hardlink", "symlink"]
Layout = Literal["flat", "date", "folder"]


@dataclass
class ExportStats:
    target: str
    exported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    fallbacks: int = 0


def _target_name(img: Image, layout: Layout) -> Path:
    if layout == "date" and img.recorded_at:
        return Path(f"{img.recorded_at:%Y}") / f"{img.recorded_at:%m}" / img.filename
    if layout == "folder":
        return Path(img.rel_path)
    return Path(img.filename)


def _unique(path: Path, image_id: int) -> Path:
    if not path.exists() and not path.is_symlink():
        return path
    candidate = path.with_name(f"{path.stem}_{image_id}{path.suffix}")
    n = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = path.with_name(f"{path.stem}_{image_id}_{n}{path.suffix}")
        n += 1
    return candidate


def _place(src: Path, dst: Path, mode: Mode) -> bool:
    """Create ``dst``; returns ``True`` if a fallback to copying was needed."""
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return False
        except OSError:
            shutil.copy2(src, dst)  # different drive / filesystem without hardlinks
            return True
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return False
        except OSError:
            # Windows needs Developer Mode or admin rights for symlinks.
            shutil.copy2(src, dst)
            return True
    shutil.copy2(src, dst)
    return False


def export_images(
    db: Database,
    f: ImageFilter,
    target: str | Path,
    *,
    mode: Mode = "copy",
    layout: Layout = "flat",
    image_ids: list[int] | None = None,
) -> ExportStats:
    target = Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    stats = ExportStats(target=str(target))
    with db.session() as s:
        ids = image_ids if image_ids is not None else all_image_ids(s, f)
        images = {i.id: i for i in s.scalars(select(Image).where(Image.id.in_(ids)))}
        s.expunge_all()
    manifest = target / "export_manifest.csv"
    with manifest.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(["image_id", "source", "target", "mode"])
        for image_id in ids:
            img = images.get(image_id)
            if img is None:
                continue
            src = Path(img.path)
            if not src.exists():
                stats.skipped += 1
                stats.errors.append(f"missing: {src}")
                continue
            dst = _unique(target / _target_name(img, layout), img.id)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if _place(src, dst, mode):
                    stats.fallbacks += 1
            except OSError as exc:
                stats.errors.append(f"{src}: {exc}")
                continue
            writer.writerow([img.id, str(src), str(dst), mode])
            stats.exported += 1
    return stats
