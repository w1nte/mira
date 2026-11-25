"""Image loading, metadata extraction and thumbnail / face-crop caches."""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

try:  # HEIC/HEIF support (iPhone photos)
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:  # pragma: no cover - optional
    HEIF_SUPPORTED = False

Image.MAX_IMAGE_PIXELS = 300_000_000  # allow large panoramas, still guard against bombs

__all__ = [
    "HEIF_SUPPORTED",
    "ImageInfo",
    "UnidentifiedImageError",
    "date_from_filename",
    "ensure_face_crop",
    "ensure_thumbnail",
    "face_crop_path",
    "load_image",
    "read_image_info",
    "resize_max",
    "thumbnail_path",
    "to_rgb",
]

_TAG_DATETIME = 306
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_ORIENTATION = 274


# --------------------------------------------------------------------------- basic helpers
def to_rgb(im: Image.Image) -> Image.Image:
    """Convert any PIL mode to RGB, compositing transparency onto white."""
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def resize_max(im: Image.Image, max_px: int) -> Image.Image:
    """Downscale so that the longest edge is at most ``max_px`` (never upscales)."""
    w, h = im.size
    longest = max(w, h)
    if max_px <= 0 or longest <= max_px:
        return im
    scale = max_px / longest
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def load_image(path: str | Path, max_px: int | None = None) -> Image.Image:
    """Open an image, apply EXIF orientation and return an RGB image.

    For JPEGs ``max_px`` enables a fast reduced-size decode via ``draft()``.
    """
    with Image.open(path) as im:
        if max_px and im.format == "JPEG":
            im.draft("RGB", (max_px, max_px))
        if getattr(im, "is_animated", False):
            im.seek(0)
        im.load()
        out = ImageOps.exif_transpose(im) or im
        out = to_rgb(out)
    if max_px:
        out = resize_max(out, max_px)
    return out


# --------------------------------------------------------------------------- metadata
@dataclass
class ImageInfo:
    width: int
    height: int
    recorded_at: datetime | None
    date_source: str
    camera: str | None
    meta: dict[str, Any] = field(default_factory=dict)
    image: Image.Image | None = None  # small, oriented RGB image (for hashing)


def _parse_exif_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    s = str(value).strip().replace("\x00", "")
    if not s or s.startswith("0000"):
        return None
    for fmt, length in (
        ("%Y:%m:%d %H:%M:%S", 19),
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y:%m:%d %H:%M", 16),
        ("%Y:%m:%d", 10),
    ):
        try:
            dt = datetime.strptime(s[:length], fmt)
        except ValueError:
            continue
        if 1900 <= dt.year <= 2100:
            return dt
    return None


_FILENAME_DATE = re.compile(
    r"(?<!\d)(?P<y>(?:19|20)\d{2})[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<d>0[1-9]|[12]\d|3[01])"
    r"(?:[-_ T.]?(?:at[-_ ])?(?P<H>[01]\d|2[0-3])[-_.:h]?(?P<M>[0-5]\d)[-_.:m]?(?P<S>[0-5]\d)?"
    r"(?:[.,]?\d{3})?)?"
    r"(?!\d)"
)


def date_from_filename(name: str) -> datetime | None:
    """Extract a date from typical camera / messenger file names.

    Examples: ``IMG_20230101_120000.jpg``, ``PXL_20230101_120000123.jpg``,
    ``WhatsApp Image 2023-01-01 at 12.00.00.jpeg``, ``Screenshot 2023-01-01 120000.png``.
    """
    stem = Path(name).stem
    for match in _FILENAME_DATE.finditer(stem):
        g = match.groupdict()
        try:
            return datetime(
                int(g["y"]),
                int(g["m"]),
                int(g["d"]),
                int(g["H"] or 0),
                int(g["M"] or 0),
                int(g["S"] or 0),
            )
        except ValueError:
            continue
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8").strip("\x00 ")
        except UnicodeDecodeError:
            return None
        return text if text.isprintable() else None
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    if isinstance(value, int | str | bool) or value is None:
        return value.strip("\x00 ") if isinstance(value, str) else value
    try:
        f = float(value)  # IFDRational and friends
    except (TypeError, ValueError, ZeroDivisionError):
        return str(value)
    return round(f, 6) if f == f else None  # drop NaN


def _gps_to_decimal(values: Any, ref: Any) -> float | None:
    try:
        d, m, s = (float(v) for v in values)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    dec = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if str(ref).upper() in ("S", "W"):
        dec = -dec
    return round(dec, 7)


_EXIF_FIELDS = {
    "Make": "make",
    "Model": "model",
    "Software": "software",
    "LensModel": "lens",
    "LensMake": "lens_make",
    "ISOSpeedRatings": "iso",
    "PhotographicSensitivity": "iso",
    "ExposureTime": "exposure_time",
    "FNumber": "f_number",
    "FocalLength": "focal_length",
    "FocalLengthIn35mmFilm": "focal_length_35mm",
    "Flash": "flash",
    "WhiteBalance": "white_balance",
    "ExposureProgram": "exposure_program",
    "OffsetTimeOriginal": "offset_time",
    "ImageDescription": "description",
    "Artist": "artist",
    "BodySerialNumber": "body_serial",
}


def _extract_exif(im: Image.Image) -> tuple[dict[str, Any], dict[str, datetime | None], int]:
    meta: dict[str, Any] = {}
    dates: dict[str, datetime | None] = {}
    orientation = 1
    try:
        exif = im.getexif()
    except Exception:
        return meta, dates, orientation
    if not exif:
        return meta, dates, orientation
    merged: dict[int, Any] = dict(exif.items())
    with contextlib.suppress(Exception):
        merged.update(exif.get_ifd(ExifTags.IFD.Exif).items())
    for tag_id, value in merged.items():
        name = ExifTags.TAGS.get(tag_id)
        if name in _EXIF_FIELDS:
            v = _jsonable(value)
            if v not in (None, "", []):
                meta[_EXIF_FIELDS[name]] = v
    orientation = int(merged.get(_TAG_ORIENTATION, 1) or 1)
    dates["exif"] = _parse_exif_datetime(merged.get(_TAG_DATETIME_ORIGINAL))
    dates["exif_digitized"] = _parse_exif_datetime(merged.get(_TAG_DATETIME_DIGITIZED))
    dates["exif_datetime"] = _parse_exif_datetime(merged.get(_TAG_DATETIME))
    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps = {}
    if gps:
        lat = _gps_to_decimal(gps.get(2), gps.get(1)) if 2 in gps else None
        lon = _gps_to_decimal(gps.get(4), gps.get(3)) if 4 in gps else None
        if lat is not None and lon is not None and not (lat == 0 and lon == 0):
            meta["gps"] = {"lat": lat, "lon": lon}
            if 6 in gps:
                alt = _jsonable(gps.get(6))
                if isinstance(alt, float | int):
                    meta["gps"]["alt"] = alt
    meta["has_exif"] = True
    if orientation != 1:
        meta["orientation"] = orientation
    return meta, dates, orientation


def _camera_name(meta: dict[str, Any]) -> str | None:
    make = str(meta.get("make") or "").strip()
    model = str(meta.get("model") or "").strip()
    if not make and not model:
        return None
    if make and model.lower().startswith(make.split()[0].lower()):
        return model
    return f"{make} {model}".strip()


def read_image_info(path: str | Path, mtime: float, hash_px: int = 256) -> ImageInfo:
    """Read metadata + a small oriented RGB version of the image for hashing."""
    path = Path(path)
    with Image.open(path) as im:
        fmt = im.format
        mode = im.mode
        width, height = im.size
        meta, dates, orientation = _extract_exif(im)
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        n_frames = getattr(im, "n_frames", 1)
        if fmt == "JPEG":
            im.draft("RGB", (hash_px, hash_px))
        im.load()
        small = ImageOps.exif_transpose(im) or im
        small = to_rgb(small)
        small = resize_max(small, max(hash_px * 2, 64))

    meta.update({"format": fmt, "mode": mode})
    if n_frames and n_frames > 1:
        meta["frames"] = n_frames

    recorded_at: datetime | None = None
    date_source = "mtime"
    for source in ("exif", "exif_digitized", "exif_datetime"):
        if dates.get(source):
            recorded_at, date_source = dates[source], source
            break
    if recorded_at is None:
        fn_date = date_from_filename(path.name)
        if fn_date is not None:
            recorded_at, date_source = fn_date, "filename"
    if recorded_at is None:
        recorded_at = datetime.fromtimestamp(mtime)
        date_source = "mtime"

    return ImageInfo(
        width=width,
        height=height,
        recorded_at=recorded_at,
        date_source=date_source,
        camera=_camera_name(meta),
        meta=meta,
        image=small,
    )


# --------------------------------------------------------------------------- caches
def _sharded(base: Path, item_id: int, suffix: str) -> Path:
    return base / f"{item_id // 1000:04d}" / f"{item_id}{suffix}"


def thumbnail_path(thumbs_dir: Path, image_id: int, size: int) -> Path:
    return _sharded(thumbs_dir, image_id, f"_{size}.webp")


def ensure_thumbnail(thumbs_dir: Path, image_id: int, source: str | Path, size: int) -> Path:
    target = thumbnail_path(thumbs_dir, image_id, size)
    if target.exists():
        try:
            if target.stat().st_mtime >= Path(source).stat().st_mtime:
                return target
        except OSError:
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    im = load_image(source, max_px=size * 2)
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    tmp = target.with_suffix(".tmp")
    im.save(tmp, "WEBP", quality=80, method=4)
    tmp.replace(target)
    return target


def face_crop_path(crops_dir: Path, face_id: int) -> Path:
    return _sharded(crops_dir, face_id, ".webp")


def ensure_face_crop(
    crops_dir: Path,
    face_id: int,
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    size: int = 160,
    margin: float = 0.3,
) -> Path:
    """Save a square crop around ``bbox`` (in ``image`` coordinates)."""
    target = face_crop_path(crops_dir, face_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    box = (
        max(0, round(cx - half)),
        max(0, round(cy - half)),
        min(image.width, round(cx + half)),
        min(image.height, round(cy + half)),
    )
    crop = image.crop(box)
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    crop.save(target, "WEBP", quality=85)
    return target
