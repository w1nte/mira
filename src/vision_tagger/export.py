"""Export a filter result into a folder (copy, hardlink or symlink). Originals stay untouched.

With ``metadata`` the export also hands vision-tagger's findings to other applications (see
:mod:`vision_tagger.xmp`): ``"xmp"`` writes XMP sidecar files, ``"embed"`` puts the XMP into
exported JPEG/PNG copies, ``"index"`` writes ``vision-tagger.csv`` / ``.json``.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy import select

from . import xmp as X
from .db.models import Image
from .db.session import Database
from .db.util import chunked
from .query import ImageFilter, all_image_ids

log = logging.getLogger(__name__)

Mode = Literal["copy", "hardlink", "symlink"]
Layout = Literal["flat", "date", "folder"]
METADATA = ("xmp", "embed", "index")


@dataclass
class ExportStats:
    target: str
    exported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    fallbacks: int = 0
    sidecars: int = 0
    embedded: int = 0
    kept: int = 0  # sidecars of other applications that were left alone


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


def _iter_chunks(db: Database, ids: list[int]) -> Iterator[list[Image]]:
    """Images in chunks (SQLite limits the number of bound parameters), keeping order."""
    for part in chunked(ids):
        with db.session() as s:
            by_id = {i.id: i for i in s.scalars(select(Image).where(Image.id.in_(part)))}
            s.expunge_all()
        yield [by_id[i] for i in part if i in by_id]


def _iter_images(db: Database, ids: list[int]) -> Iterator[Image]:
    for chunk in _iter_chunks(db, ids):
        yield from chunk


def _metadata(db: Database, images: list[Image]) -> dict[int, X.PhotoMeta]:
    with db.session() as s:
        return X.collect(s, images)


def export_images(
    db: Database,
    f: ImageFilter,
    target: str | Path,
    *,
    mode: Mode = "copy",
    layout: Layout = "flat",
    image_ids: list[int] | None = None,
    metadata: Collection[str] = (),
    sidecar_naming: str = "append",
) -> ExportStats:
    target = Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    stats = ExportStats(target=str(target))
    with db.session() as s:
        ids = image_ids if image_ids is not None else all_image_ids(s, f)
    records: list[dict] = []
    manifest = target / "export_manifest.csv"
    with manifest.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(["image_id", "source", "target", "mode"])
        for chunk in _iter_chunks(db, ids):
            metas = _metadata(db, chunk) if metadata else {}
            for img in chunk:
                src = Path(img.path)
                if not src.exists():
                    stats.skipped += 1
                    stats.errors.append(f"missing: {src}")
                    continue
                dst = _unique(target / _target_name(img, layout), img.id)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fallback = _place(src, dst, mode)
                except OSError as exc:
                    stats.errors.append(f"{src}: {exc}")
                    continue
                stats.fallbacks += fallback
                writer.writerow([img.id, str(src), str(dst), mode])
                stats.exported += 1
                meta = metas.get(img.id)
                if meta is None:
                    continue
                packet = X.to_xmp(meta)
                try:
                    # a hard or symbolic link *is* the original: only real copies are changed
                    if "embed" in metadata and (mode == "copy" or fallback):
                        stats.embedded += X.embed_into_copy(dst, packet)
                    if "xmp" in metadata:
                        X.write_sidecar(dst, packet, sidecar_naming, force=True)
                        stats.sidecars += 1
                except OSError as exc:
                    stats.errors.append(f"{dst}: metadata: {exc}")
                if "index" in metadata:
                    records.append(meta.record(dst.relative_to(target).as_posix()))
    if records:
        X.write_index_files(target, records)
    return stats


def write_sidecars_in_place(
    db: Database,
    f: ImageFilter,
    *,
    image_ids: list[int] | None = None,
    naming: str = "append",
) -> ExportStats:
    """XMP sidecars next to the original photos (the photos themselves are not touched).

    For libraries managed in place by digiKam, darktable, Immich or PhotoPrism. Sidecars that
    another application wrote are left alone (``kept``); ours are updated.
    """
    stats = ExportStats(target="(next to the originals)")
    with db.session() as s:
        ids = image_ids if image_ids is not None else all_image_ids(s, f)
    for chunk in _iter_chunks(db, ids):
        metas = _metadata(db, chunk)
        for img in chunk:
            src = Path(img.path)
            if not src.exists():
                stats.skipped += 1
                continue
            try:
                if X.write_sidecar(src, X.to_xmp(metas[img.id]), naming):
                    stats.sidecars += 1
                else:
                    stats.kept += 1
            except OSError as exc:
                stats.errors.append(f"{src}: {exc}")
    return stats
