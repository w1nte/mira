"""Hand what vision-tagger found over to other photo applications.

Everything is written as **XMP**, the metadata standard that photo applications share:

* ``dc:subject``: keywords (descriptive properties such as objects, place, event, and the
  names of the people),
* ``lr:hierarchicalSubject``: every property as a hierarchical keyword
  (``vision-tagger|Location type|beach``, ``People|Anna``), as Lightroom, digiKam and
  darktable show them,
* ``dc:description``: the caption of the vision model,
* ``Iptc4xmpExt:PersonInImage`` and MWG face regions (``mwg-rs:Regions``) with the names of
  the recognised people (face areas for the image as displayed),
* ``exif:DateTimeOriginal`` / ``xmp:CreateDate`` / ``photoshop:DateCreated``: the capture date,
  including dates you corrected (file-time dates are left out, they are guesses),
* ``vt:``: vision-tagger's own namespace with every property and its confidence.

XMP goes into sidecar files (``photo.jpg.xmp``: digiKam, darktable, Immich, PhotoPrism,
Lightroom for raw files, ExifTool) or into exported JPEG/PNG copies (Lightroom, Apple Photos,
Windows Explorer, Bridge). Originals are never modified, and an XMP packet that is already in a
file is never replaced (the sidecar carries the data then).

:func:`write_index_files` adds ``vision-tagger.csv`` and ``vision-tagger.json`` to an export
folder: one record per photo, for spreadsheets and scripts.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import struct
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import Cluster, Face, Image, ImageProperty, Person, VlmResult
from .db.util import chunked

NS_VT = "https://github.com/w1nte/vision-tagger/ns/1.0/"
XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
XMP_EXT_HEADER = b"http://ns.adobe.com/xmp/extension/\x00"
MARKER = "vision-tagger"  # our sidecars contain it; others are never overwritten

# properties that make good plain keywords; everything else is in the hierarchy only
KEYWORD_KEYS = (
    "pet",
    "animal",
    "objects",
    "animals",
    "vehicles",
    "food",
    "tags",
    "landmark",
    "location_type",
    "setting",
    "event_type",
    "main_activity",
    "season",
    "time_of_day",
    "weather",
    "composition",
    "image_type",
    "work_category",
)
NOT_A_KEYWORD = {"", "none", "unknown", "other", "no", "not_work", "n/a"}
TRUSTED_DATES = {"manual", "exif", "exif_digitized", "exif_datetime", "filename"}


@dataclass
class FaceRegion:
    name: str
    x: float  # centre, 0..1 of the displayed image
    y: float
    w: float
    h: float


@dataclass
class PhotoMeta:
    image_id: int
    source: str
    width: int | None
    height: int | None
    recorded_at: datetime | None
    date_source: str
    camera: str | None
    phash: str
    caption: str | None = None
    persons: list[str] = field(default_factory=list)
    regions: list[FaceRegion] = field(default_factory=list)
    properties: dict[str, list[tuple[str, float | None]]] = field(default_factory=dict)

    @property
    def keywords(self) -> list[str]:
        out: list[str] = []
        for key in KEYWORD_KEYS:
            for value, _conf in self.properties.get(key, []):
                word = value.replace("_", " ")
                if value.lower() not in NOT_A_KEYWORD and word not in out:
                    out.append(word)
        return out + [p for p in self.persons if p not in out]

    @property
    def hierarchical(self) -> list[str]:
        out = [f"People|{p}" for p in self.persons]
        for key in sorted(self.properties):
            label = key.replace("_", " ").capitalize()
            for value, _conf in self.properties[key]:
                out.append(f"vision-tagger|{label}|{value.replace('_', ' ')}".replace("\n", " "))
        return out

    def record(self, file: str | None = None) -> dict:
        """Plain dict for the JSON index."""
        return {
            "file": file,
            "source": self.source,
            "image_id": self.image_id,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "date_source": self.date_source,
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "caption": self.caption,
            "persons": self.persons,
            "keywords": self.keywords,
            "faces": [asdict(r) for r in self.regions],
            "properties": {k: [v for v, _ in vals] for k, vals in sorted(self.properties.items())},
            "phash": self.phash,
        }


# --------------------------------------------------------------------------- collect
def collect(s: Session, images: list[Image]) -> dict[int, PhotoMeta]:
    """Everything worth exporting for these images (few queries per chunk of ids)."""
    metas = {
        img.id: PhotoMeta(
            image_id=img.id,
            source=img.path,
            width=img.width,
            height=img.height,
            recorded_at=img.recorded_at,
            date_source=img.date_source or "",
            camera=img.camera,
            phash=img.phash,
        )
        for img in images
    }
    for part in chunked(list(metas)):
        for image_id, caption in s.execute(
            select(VlmResult.image_id, VlmResult.caption).where(VlmResult.image_id.in_(part))
        ):
            metas[image_id].caption = caption or None
        for image_id, key, value, conf in s.execute(
            select(
                ImageProperty.image_id,
                ImageProperty.key,
                ImageProperty.value,
                ImageProperty.confidence,
            )
            .where(ImageProperty.image_id.in_(part))
            .order_by(ImageProperty.image_id, ImageProperty.key, ImageProperty.id)
        ):
            metas[image_id].properties.setdefault(key, []).append((value, conf))
        rows = s.execute(
            select(Face.image_id, Face.x1, Face.y1, Face.x2, Face.y2, Person.name)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .where(Face.image_id.in_(part), Face.ignored.is_(False))
            .order_by(Face.image_id, Face.x1)
        )
        for image_id, x1, y1, x2, y2, name in rows:
            m = metas[image_id]
            if name not in m.persons:
                m.persons.append(name)
            if m.width and m.height:
                m.regions.append(
                    FaceRegion(
                        name=name,
                        x=round((x1 + x2) / 2 / m.width, 5),
                        y=round((y1 + y2) / 2 / m.height, 5),
                        w=round((x2 - x1) / m.width, 5),
                        h=round((y2 - y1) / m.height, 5),
                    )
                )
    return metas


# --------------------------------------------------------------------------- XMP
def _bag(tag: str, items: list[str]) -> str:
    if not items:
        return ""
    lis = "".join(f"<rdf:li>{escape(i)}</rdf:li>" for i in items)
    return f"   <{tag}><rdf:Bag>{lis}</rdf:Bag></{tag}>\n"


def to_xmp(meta: PhotoMeta) -> bytes:
    attrs = [
        'xmlns:dc="http://purl.org/dc/elements/1.1/"',
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
        'xmlns:exif="http://ns.adobe.com/exif/1.0/"',
        'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"',
        'xmlns:lr="http://ns.adobe.com/lightroom/1.0/"',
        'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"',
        'xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"',
        'xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#"',
        'xmlns:stDim="http://ns.adobe.com/xap/1.0/sType/Dimensions#"',
        f'xmlns:vt="{NS_VT}"',
        f"xmp:MetadataDate={quoteattr(datetime.now().astimezone().isoformat(timespec='seconds'))}",
        f"vt:ImageId={quoteattr(str(meta.image_id))}",
        f"vt:PHash={quoteattr(meta.phash)}",
    ]
    if meta.recorded_at and meta.date_source in TRUSTED_DATES:
        when = quoteattr(meta.recorded_at.isoformat(timespec="seconds"))
        attrs += [f"exif:DateTimeOriginal={when}", f"xmp:CreateDate={when}"]
        attrs += [f"photoshop:DateCreated={when}", f"vt:DateSource={quoteattr(meta.date_source)}"]
    body = ""
    if meta.caption:
        body += (
            '   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">'
            f"{escape(meta.caption)}</rdf:li></rdf:Alt></dc:description>\n"
        )
    body += _bag("dc:subject", meta.keywords)
    body += _bag("lr:hierarchicalSubject", meta.hierarchical)
    body += _bag("Iptc4xmpExt:PersonInImage", meta.persons)
    if meta.regions and meta.width and meta.height:
        regions = "".join(
            "<rdf:li><rdf:Description "
            f'mwg-rs:Name={quoteattr(r.name)} mwg-rs:Type="Face">'
            f'<mwg-rs:Area stArea:x="{r.x}" stArea:y="{r.y}" stArea:w="{r.w}" '
            f'stArea:h="{r.h}" stArea:unit="normalized"/>'
            "</rdf:Description></rdf:li>"
            for r in meta.regions
        )
        body += (
            '   <mwg-rs:Regions rdf:parseType="Resource">'
            f'<mwg-rs:AppliedToDimensions stDim:w="{meta.width}" stDim:h="{meta.height}" '
            'stDim:unit="pixel"/>'
            f"<mwg-rs:RegionList><rdf:Bag>{regions}</rdf:Bag></mwg-rs:RegionList>"
            "</mwg-rs:Regions>\n"
        )
    props = [
        f"{key}={value}" + (f" ({conf:.2f})" if conf is not None else "")
        for key, values in sorted(meta.properties.items())
        for value, conf in values
    ]
    body += _bag("vt:Properties", props)
    nl = "\n  "
    xml = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{MARKER}">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'  <rdf:Description rdf:about=""{nl}  {(nl + "  ").join(attrs)}>\n'
        f"{body}"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return xml.encode("utf-8")


# --------------------------------------------------------------------------- sidecars
def sidecar_path(photo: Path, naming: str = "append") -> Path:
    """``photo.jpg.xmp`` (digiKam, darktable, Immich) or ``photo.xmp`` (Lightroom style)."""
    if naming == "replace":
        return photo.with_suffix(".xmp")
    return photo.with_name(photo.name + ".xmp")


def is_ours(sidecar: Path) -> bool:
    try:
        with sidecar.open("rb") as fh:
            return MARKER.encode() in fh.read(4096)
    except OSError:
        return False


def write_sidecar(photo: Path, xmp: bytes, naming: str = "append", *, force: bool = False) -> bool:
    """Write the sidecar; never replaces one that another application wrote (unless ``force``)."""
    target = sidecar_path(photo, naming)
    if target.exists() and not force and not is_ours(target):
        return False
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(xmp)
    tmp.replace(target)
    return True


# --------------------------------------------------------------------------- embedding
def _jpeg_segments(data: bytes):
    """(marker, start, end) of the header segments up to the image data."""
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise ValueError("not a JPEG segment")
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker == 0xDA:  # start of scan: image data follows
            return
        length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        yield marker, pos, pos + 2 + length
        pos += 2 + length


def has_xmp(data: bytes) -> bool:
    if data[:2] == b"\xff\xd8":
        return any(
            m == 0xE1 and data[s + 4 : s + 4 + len(XMP_HEADER)] == XMP_HEADER
            for m, s, _e in _jpeg_segments(data)
        )
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return b"iTXtXML:com.adobe.xmp" in data[: data.find(b"IDAT") if b"IDAT" in data else None]
    return False


def embed_jpeg(data: bytes, xmp: bytes) -> bytes:
    """Insert an XMP APP1 segment after the JFIF/EXIF headers (image data untouched)."""
    payload = XMP_HEADER + xmp
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("XMP packet too large for a JPEG segment")
    insert_at = 2
    for marker, _start, end in _jpeg_segments(data):
        if marker in (0xE0, 0xE1):  # JFIF, EXIF: keep them first
            insert_at = end
        else:
            break
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return data[:insert_at] + segment + data[insert_at:]


def embed_png(data: bytes, xmp: bytes) -> bytes:
    """Insert an ``iTXt`` chunk with the XMP packet right after the header chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    ihdr_end = 8 + 12 + struct.unpack(">I", data[8:12])[0]
    chunk_data = b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp
    chunk = (
        struct.pack(">I", len(chunk_data))
        + b"iTXt"
        + chunk_data
        + struct.pack(">I", zlib.crc32(b"iTXt" + chunk_data) & 0xFFFFFFFF)
    )
    return data[:ihdr_end] + chunk + data[ihdr_end:]


def embed_into_copy(path: Path, xmp: bytes) -> bool:
    """Embed the packet into an exported *copy* (JPEG, PNG; told apart by content, not name).

    Returns False if the format has no simple XMP slot, or if the file already carries XMP
    (that packet is kept untouched; the sidecar has vision-tagger's data then).
    """
    with path.open("rb") as fh:
        head = fh.read(8)
    if head[:2] == b"\xff\xd8":
        embed = embed_jpeg
    elif head == b"\x89PNG\r\n\x1a\n":
        embed = embed_png
    else:
        return False
    data = path.read_bytes()
    if has_xmp(data):
        return False
    try:
        new = embed(data, xmp)
    except (ValueError, struct.error):
        return False
    st = path.stat()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(new)
    tmp.replace(path)
    with contextlib.suppress(OSError):
        os.utime(path, (st.st_atime, st.st_mtime))  # keep the file date of the original
    return True


# --------------------------------------------------------------------------- index files
INDEX_JSON = "vision-tagger.json"
INDEX_CSV = "vision-tagger.csv"
CSV_COLUMNS = (
    "file",
    "recorded_at",
    "date_source",
    "persons",
    "keywords",
    "caption",
    "camera",
    "width",
    "height",
    "source",
    "image_id",
)


def write_index_files(target: Path, records: list[dict]) -> None:
    """Merge records (keyed by their ``file``) into the index files of an export folder."""
    path = target / INDEX_JSON
    existing: dict[str, dict] = {}
    if path.exists():
        with contextlib.suppress(OSError, ValueError):
            for rec in json.loads(path.read_text(encoding="utf-8")).get("photos", []):
                existing[rec.get("file") or rec.get("source")] = rec
    for rec in records:
        existing[rec.get("file") or rec.get("source")] = rec
    photos = sorted(existing.values(), key=lambda r: str(r.get("file")))
    doc = {"generator": MARKER, "updated": datetime.now().isoformat(timespec="seconds")}
    doc["photos"] = photos
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    csv_path = target / INDEX_CSV
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel reads UTF-8
        writer = csv.DictWriter(fh, CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in photos:
            row = dict(rec)
            row["persons"] = "; ".join(rec.get("persons") or [])
            row["keywords"] = "; ".join(rec.get("keywords") or [])
            writer.writerow(row)
    tmp.replace(csv_path)
