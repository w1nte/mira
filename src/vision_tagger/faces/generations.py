"""Generations: watch a person grow up, and compare family members at the same age.

* :func:`growup_frames` picks one face per age (the sharpest, most frontal one) and orders
  them by **age**, which combines the birth year with the photo date and falls back to the
  face's estimated age when a date is doubtful (see :mod:`faces.ages`).
* :func:`render_animation` cross-fades the eye-aligned portraits into an animated WebP.
* :func:`compare_at_same_age` pairs two persons' faces at matching ages and measures how
  similar the faces are; :func:`resemblance` ranks a person's look-alikes in the family.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Face, Person
from ..db.models import Image as ImageRow
from ..db.util import chunked
from ..imaging import load_image
from .ages import FaceAge, analyse_person
from .align import align_face, frontalness
from .vectors import from_blob, normalize

log = logging.getLogger(__name__)

RENDER_VERSION = "2"
MAX_FRAMES = 40
ZOOM = 0.38  # face width relative to the portrait (room for hair and chin)


@dataclass
class Frame:
    face_id: int
    image_id: int
    age: int | None  # years
    age_estimated: bool  # True = from the face, not from birth year + date
    recorded_at: datetime | None
    quality: float
    person: str = ""

    @property
    def label(self) -> str:
        if self.age is None:
            return f"{self.recorded_at:%Y}" if self.recorded_at else ""
        age = (
            f"{'~' if self.age_estimated else ''}{self.age} {'year' if self.age == 1 else 'years'}"
        )
        year = f" · {self.recorded_at:%Y}" if self.recorded_at and not self.age_estimated else ""
        return f"{age}{year}"


def _size_score(det_score: float, face_px: int) -> float:
    return float(det_score or 0) * min(1.0, (face_px or 0) / 160.0)


def _quality(det_score: float, face_px: int, landmarks) -> float:
    frontal = frontalness(np.asarray(landmarks, dtype=np.float32)) if landmarks else 0.5
    return _size_score(det_score, face_px) * (0.25 + 0.75 * frontal)


_ROUND = 8  # faces per age whose landmarks are decoded per round (see _best_frames)


@dataclass
class _Candidate:
    bound: float  # the quality if the face were perfectly frontal (an upper bound)
    face_id: int
    info: FaceAge
    age: int
    estimated: bool
    det_score: float
    face_px: int


def _best_frames(s: Session, by_age: dict[int, list[_Candidate]], person: str) -> list[Frame]:
    """Per age the face with the highest quality (ties: the lowest face id), youngest first.

    Quality needs the landmarks (JSON) and the frontalness of each face, but never exceeds the
    candidate's ``bound``. So faces are evaluated from the highest bound down, a few per age
    and round (one query for all ages), until no remaining face can reach the best quality:
    the result is exact, but only a fraction of the faces is decoded.
    """
    for candidates in by_age.values():
        candidates.sort(key=lambda c: (-c.bound, c.face_id))
    best: dict[int, Frame] = {}
    pos = dict.fromkeys(by_age, 0)
    while True:
        batch: list[_Candidate] = []
        for age, candidates in by_age.items():
            current = best.get(age)
            take = [
                c
                for c in candidates[pos[age] : pos[age] + _ROUND]
                if current is None or c.bound >= current.quality
            ]
            pos[age] += _ROUND
            batch += take
        if not batch:
            break
        landmarks: dict[int, object] = {}
        for part in chunked(sorted(c.face_id for c in batch)):
            landmarks.update(
                s.execute(select(Face.id, Face.landmarks).where(Face.id.in_(part))).all()
            )
        for c in batch:
            quality = _quality(c.det_score, c.face_px, landmarks.get(c.face_id)) * (
                0.7 if c.estimated else 1.0
            )
            current = best.get(c.age)
            if (
                current is None
                or quality > current.quality
                or (quality == current.quality and c.face_id < current.face_id)
            ):
                best[c.age] = Frame(
                    face_id=c.face_id,
                    image_id=c.info.image_id,
                    age=c.age,
                    age_estimated=c.estimated,
                    recorded_at=c.info.recorded_at,
                    quality=quality,
                    person=person,
                )
    return [best[a] for a in sorted(best)]


def growup_frames(s: Session, person_id: int, max_frames: int = MAX_FRAMES) -> list[Frame]:
    """One good face per age (the best :func:`_quality`), youngest first."""
    person = s.get(Person, person_id)
    if person is None:
        return []
    ages = analyse_person(s, person)
    by_face = {f.face_id: f for f in ages.faces}
    if not by_face:
        return []
    by_age: dict[int, list[_Candidate]] = {}
    for part in chunked(sorted(by_face)):
        for face_id, det, px, clusterable in s.execute(
            select(Face.id, Face.det_score, Face.face_px, Face.clusterable).where(Face.id.in_(part))
        ):
            info = by_face[face_id]
            if not clusterable:
                continue  # tiny or blurry detections make ugly frames
            if info.calendar_age is not None and not info.suspicious:
                age, estimated = info.calendar_age, False
            elif info.face_age is not None:
                age, estimated = info.face_age, True
            else:
                continue
            bound = _size_score(det, px) * (0.7 if estimated else 1.0)
            age = max(0, age)
            by_age.setdefault(age, []).append(
                _Candidate(bound, face_id, info, age, estimated, det, px)
            )
    frames = _best_frames(s, by_age, person.name)
    if len(frames) > max_frames:  # keep the span, thin out evenly
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]
    return frames


# --------------------------------------------------------------------------- aligned portraits
def aligned_path(cache_dir: Path, face_id: int, size: int) -> Path:
    return cache_dir / f"{face_id // 1000:04d}" / f"{face_id}_{size}_v{RENDER_VERSION}.webp"


def _decode_px(face: Face, width: int | None, height: int | None, size: int) -> int | None:
    """Decode size that keeps the face sharp in the portrait without loading huge originals.

    Small faces in large photos are decoded at full resolution, big faces need far less.
    """
    longest = max(width or 0, height or 0)
    side = max(face.x2 - face.x1, face.y2 - face.y1)
    if not longest or side <= 0:
        return 2000
    needed = longest * (size * ZOOM * 1.25) / side  # face box -> its width in the portrait
    return None if needed >= longest else max(size, round(needed))


def aligned_portrait(s: Session, cache_dir: Path, face_id: int, size: int = 512) -> Path | None:
    """Eye-aligned portrait of a face (cached on disk)."""
    target = aligned_path(cache_dir, face_id, size)
    if target.exists():
        return target
    row = s.execute(
        select(Face, ImageRow.path, ImageRow.width, ImageRow.height)
        .join(ImageRow, ImageRow.id == Face.image_id)
        .where(Face.id == face_id)
    ).first()
    if row is None:
        return None
    face, path, full_width, full_height = row
    try:
        im = load_image(path, max_px=_decode_px(face, full_width, full_height, size))
    except OSError as exc:
        log.warning("cannot load %s: %s", path, exc)
        return None
    scale = im.width / full_width if full_width else 1.0
    landmarks = [[x * scale, y * scale] for x, y in face.landmarks] if face.landmarks else None
    bbox = (face.x1 * scale, face.y1 * scale, face.x2 * scale, face.y2 * scale)
    aligned = align_face(im, landmarks, bbox, size=size, zoom=ZOOM)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    aligned.image.save(tmp, "WEBP", quality=88)
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------- animation
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _caption(im: Image.Image, title: str, subtitle: str) -> Image.Image:
    out = im.convert("RGBA")
    band = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)
    w, h = out.size
    draw.rectangle([0, int(h * 0.84), w, h], fill=(0, 0, 0, 140))
    big, small = _font(max(14, h // 16)), _font(max(11, h // 26))
    draw.text((w * 0.05, h * 0.855), title, font=big, fill=(255, 255, 255, 255))
    draw.text((w * 0.05, h * 0.93), subtitle, font=small, fill=(225, 225, 225, 235))
    return Image.alpha_composite(out, band).convert("RGB")


def animation_key(frames: Iterable[Frame], size: int) -> str:
    blob = ",".join(f"{f.face_id}:{f.age}:{f.label}" for f in frames) + f"|{size}|{RENDER_VERSION}"
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def render_animation(
    portraits: list[Image.Image],
    captions: list[tuple[str, str]],
    *,
    hold_ms: int = 650,
    fade_steps: int = 10,
    fade_ms: int = 55,
) -> bytes:
    """Cross-fade aligned portraits into a looping animated WebP."""
    if not portraits:
        raise ValueError("no frames")
    stills = [_caption(p, *c) for p, c in zip(portraits, captions, strict=True)]
    frames: list[Image.Image] = []
    durations: list[int] = []
    for i, still in enumerate(stills):
        frames.append(still)
        durations.append(hold_ms * (3 if i in (0, len(stills) - 1) else 1))
        if i + 1 < len(stills):
            a, b = portraits[i].convert("RGB"), portraits[i + 1].convert("RGB")
            for step in range(1, fade_steps + 1):
                alpha = step / (fade_steps + 1)
                # caption switches half way through the fade
                cap = captions[i] if alpha < 0.5 else captions[i + 1]
                frames.append(_caption(Image.blend(a, b, alpha), *cap))
                durations.append(fade_ms)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        quality=80,
        method=4,
    )
    return buf.getvalue()


def growup_animation(
    s: Session, cache_dir: Path, person_id: int, size: int = 384
) -> tuple[bytes, list[Frame]] | None:
    """Animated WebP of a person growing up (cached by the selected faces)."""
    frames = growup_frames(s, person_id)
    if len(frames) < 2:
        return None
    key = animation_key(frames, size)
    target = cache_dir / "animations" / f"person{person_id}_{key}.webp"
    if target.exists():
        return target.read_bytes(), frames
    portraits, captions = [], []
    for f in frames:
        path = aligned_portrait(s, cache_dir, f.face_id, 512)
        if path is None:
            continue
        with Image.open(path) as im:
            portraits.append(im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
        captions.append((f.person, f.label))
    if len(portraits) < 2:
        return None
    data = render_animation(portraits, captions)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data, frames


# --------------------------------------------------------------------------- comparisons
@dataclass
class AgePair:
    age: int
    a: Frame
    b: Frame
    similarity: float  # cosine similarity of the two face embeddings


def _embeddings(
    s: Session, face_ids: Iterable[int], known: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    """Normalised embeddings by face id; ids already in ``known`` are not loaded again."""
    out = known if known is not None else {}
    missing = sorted({f for f in face_ids if f not in out})
    for part in chunked(missing):
        for fid, emb in s.execute(select(Face.id, Face.embedding).where(Face.id.in_(part))):
            out[fid] = normalize(from_blob(emb))
    return out


#: ``person_id -> growup frames``; lets callers pass cached frames (see ``web.context``)
FramesOf = Callable[[int], list[Frame]]


def compare_at_same_age(
    s: Session,
    person_a: int,
    person_b: int,
    tolerance: int = 1,
    frames_of: FramesOf | None = None,
) -> list[AgePair]:
    """Faces of two persons at (about) the same age, with their similarity."""
    frames = frames_of or (lambda pid: growup_frames(s, pid))
    return pair_frames(s, frames(person_a), frames(person_b), tolerance)


def pair_frames(
    s: Session,
    fa: list[Frame],
    fb: list[Frame],
    tolerance: int = 1,
    embeddings: dict[int, np.ndarray] | None = None,
) -> list[AgePair]:
    """Match frames of two persons by age; exact matches win, every face is used once.

    ``embeddings``: a cache shared between calls (filled as needed).
    """
    candidates = [
        (abs(a.age - b.age), -(a.quality + b.quality), a, b)
        for a in fa
        for b in fb
        if a.age is not None and b.age is not None and abs(a.age - b.age) <= tolerance
    ]
    if not candidates:
        return []
    emb = _embeddings(s, [f.face_id for f in fa + fb], embeddings)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[AgePair] = []
    for _diff, _q, a, b in sorted(candidates, key=lambda c: (c[0], c[1], c[2].age)):
        if a.face_id in used_a or b.face_id in used_b:
            continue
        if a.face_id not in emb or b.face_id not in emb:
            continue  # deleted meanwhile (frames may come from a cache)
        used_a.add(a.face_id)
        used_b.add(b.face_id)
        pairs.append(AgePair(a.age, a, b, float(emb[a.face_id] @ emb[b.face_id])))
    pairs.sort(key=lambda p: p.age)
    return pairs


@dataclass
class Resemblance:
    person_id: int
    name: str
    similarity: float
    same_age_pairs: int


def resemblance(s: Session, person_id: int, frames_of: FramesOf | None = None) -> list[Resemblance]:
    """Who in the family looks most like this person?

    Uses faces at matching ages when both persons have them (fair for children vs. parents),
    otherwise the overall face centroids. Needs the frames of every person: pass
    ``frames_of`` to reuse cached ones.
    """
    frames = frames_of or (lambda pid: growup_frames(s, pid))
    me = s.get(Person, person_id)
    if me is None:
        return []
    from .quiz import person_centroids

    pids, names, mat = person_centroids(s)
    if person_id not in pids:
        return []
    mine = mat[pids.index(person_id)]
    my_frames = frames(person_id)
    embeddings: dict[int, np.ndarray] = {}  # my faces are loaded once, not once per person
    out: list[Resemblance] = []
    for pid, name, vec in zip(pids, names, mat, strict=True):
        if pid == person_id:
            continue
        pairs = pair_frames(s, my_frames, frames(pid), embeddings=embeddings)
        sim = float(np.mean([p.similarity for p in pairs])) if pairs else float(mine @ vec)
        out.append(Resemblance(pid, name, sim, len(pairs)))
    out.sort(key=lambda r: -r.similarity)
    return out
