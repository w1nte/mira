"""Ages of persons in photos, birth-year estimation and detection of doubtful photo dates.

Two independent signals are combined:

* the **calendar age**: photo date minus the person's birth year,
* the **face age**: InsightFace's per-face age estimate (noisy, but independent of dates).

Photo dates are not always right: camera clocks reset to 2000/1970, scans and copies only
carry the file time, messenger apps strip EXIF. So the date source is weighted (EXIF counts
most, the file modification time least), and photos where both signals disagree strongly are
flagged with a suggested year instead of being trusted blindly.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Cluster, Face, Image, Person

# how much a photo date can be trusted, by where it came from
DATE_RELIABILITY = {
    "manual": 1.0,
    "exif": 1.0,
    "exif_digitized": 0.9,
    "exif_datetime": 0.7,
    "filename": 0.6,
    "mtime": 0.15,
}
MIN_SAMPLES = 3


def reliability(date_source: str | None) -> float:
    return DATE_RELIABILITY.get(date_source or "", 0.3)


def age_tolerance(face_age: float) -> float:
    """Allowed difference between calendar and face age (face age is less precise for adults)."""
    return 4.0 + 0.25 * max(0.0, face_age)


@dataclass
class FaceAge:
    face_id: int
    image_id: int
    recorded_at: datetime | None
    date_source: str
    face_age: int | None
    calendar_age: int | None = None
    suspicious: bool = False
    suggested_year: int | None = None

    @property
    def weight(self) -> float:
        return reliability(self.date_source)

    @property
    def best_age(self) -> int | None:
        """Calendar age when the date fits, otherwise the face estimate."""
        if self.calendar_age is not None and not self.suspicious:
            return self.calendar_age
        return self.face_age


@dataclass
class BirthYearEstimate:
    year: int
    low: int
    high: int
    samples: int

    @property
    def confidence(self) -> str:
        spread = self.high - self.low
        if self.samples >= 8 and spread <= 6:
            return "high"
        if self.samples >= MIN_SAMPLES and spread <= 12:
            return "medium"
        return "low"


@dataclass
class PersonAges:
    person: Person
    birth_year: int | None  # the one used for calculations
    birth_source: str | None  # "set" | "estimated" | None
    estimate: BirthYearEstimate | None
    faces: list[FaceAge] = field(default_factory=list)
    current_age: int | None = None

    @property
    def suspicious(self) -> list[FaceAge]:
        return [f for f in self.faces if f.suspicious]


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    pairs = sorted(zip(values, weights, strict=True))
    total = sum(w for _, w in pairs)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= q * total:
            return value
    return pairs[-1][0]


def estimate_birth_year(faces: list[FaceAge]) -> BirthYearEstimate | None:
    """Robust birth-year estimate from ``photo year - face age``.

    Photos with trustworthy dates dominate (weighted median); implausible samples are removed
    with a MAD filter before the final estimate.
    """
    samples = [
        (f.recorded_at.year - f.face_age, f.weight)
        for f in faces
        if f.recorded_at is not None and f.face_age is not None and f.weight >= 0.5
    ]
    if len(samples) < MIN_SAMPLES:  # not enough good dates: use everything, weighted
        samples = [
            (f.recorded_at.year - f.face_age, f.weight)
            for f in faces
            if f.recorded_at is not None and f.face_age is not None
        ]
    if len(samples) < MIN_SAMPLES:
        return None
    values = [float(v) for v, _ in samples]
    weights = [w for _, w in samples]
    median = _weighted_quantile(values, weights, 0.5)
    mad = max(2.0, statistics.median(abs(v - median) for v in values))
    inliers = [(v, w) for v, w in zip(values, weights, strict=True) if abs(v - median) <= 2.5 * mad]
    if len(inliers) < MIN_SAMPLES:
        inliers = list(zip(values, weights, strict=True))
    iv = [v for v, _ in inliers]
    iw = [w for _, w in inliers]
    year = round(_weighted_quantile(iv, iw, 0.5))
    low = round(min(_weighted_quantile(iv, iw, 0.25), year - 2))
    high = round(max(_weighted_quantile(iv, iw, 0.75), year + 2))
    return BirthYearEstimate(year=year, low=low, high=high, samples=len(inliers))


def _check(face: FaceAge, birth_year: int) -> None:
    if face.recorded_at is None:
        return
    face.calendar_age = face.recorded_at.year - birth_year
    if face.calendar_age < 0:  # "taken" before the person was born
        face.suspicious = True
    elif face.face_age is not None:
        face.suspicious = abs(face.calendar_age - face.face_age) > age_tolerance(face.face_age)
    if face.suspicious and face.face_age is not None:
        face.suggested_year = birth_year + face.face_age


def person_faces(s: Session, person_id: int) -> list[FaceAge]:
    rows = s.execute(
        select(Face.id, Face.image_id, Face.age, Image.recorded_at, Image.date_source)
        .join(Cluster, Cluster.id == Face.cluster_id)
        .join(Image, Image.id == Face.image_id)
        .where(Cluster.person_id == person_id, Face.ignored.is_(False), Image.missing.is_(False))
    )
    return [
        FaceAge(face_id=f, image_id=i, recorded_at=r, date_source=src or "", face_age=a)
        for f, i, a, r, src in rows
    ]


def analyse_person(s: Session, person: Person, today: date | None = None) -> PersonAges:
    today = today or date.today()
    faces = person_faces(s, person.id)
    estimate = estimate_birth_year(faces)
    if person.birth_year:
        birth, source = person.birth_year, "set"
    elif estimate is not None and estimate.confidence != "low":
        birth, source = estimate.year, "estimated"
    else:
        birth, source = None, None
    if birth is not None:
        for f in faces:
            _check(f, birth)
    faces.sort(key=lambda f: (f.recorded_at or datetime.min, f.face_id))
    return PersonAges(
        person=person,
        birth_year=birth,
        birth_source=source,
        estimate=estimate,
        faces=faces,
        current_age=(today.year - birth) if birth else None,
    )


# --------------------------------------------------------------------------- per image
@dataclass
class FaceInImage:
    face_id: int
    person: str
    face_age: int | None
    calendar_age: int | None
    birth_source: str | None


@dataclass
class DateCheck:
    recorded_year: int | None
    date_source: str
    suspicious: bool
    suggested_year: int | None
    low: int | None
    high: int | None
    faces: dict[int, FaceInImage]


def _birth_years(s: Session, person_ids: set[int]) -> dict[int, tuple[int, str]]:
    out: dict[int, tuple[int, str]] = {}
    for pid in person_ids:
        person = s.get(Person, pid)
        if person is None:
            continue
        if person.birth_year:
            out[pid] = (person.birth_year, "set")
            continue
        estimate = estimate_birth_year(person_faces(s, pid))
        if estimate is not None and estimate.confidence != "low":
            out[pid] = (estimate.year, "estimated")
    return out


def check_image_date(s: Session, image_id: int) -> DateCheck | None:
    """Ages of the known persons in an image and whether the photo date fits their faces."""
    img = s.get(Image, image_id)
    if img is None:
        return None
    rows = list(
        s.execute(
            select(Face.id, Face.age, Cluster.person_id, Person.name)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .where(Face.image_id == image_id, Face.ignored.is_(False))
        )
    )
    if not rows:
        return None
    births = _birth_years(s, {r[2] for r in rows})
    year = img.recorded_at.year if img.recorded_at else None
    faces: dict[int, FaceInImage] = {}
    implied: list[tuple[int, float]] = []  # (year implied by the face, tolerance)
    for face_id, face_age, pid, name in rows:
        birth = births.get(pid)
        calendar = (year - birth[0]) if birth and year is not None else None
        faces[face_id] = FaceInImage(face_id, name, face_age, calendar, birth[1] if birth else None)
        if birth and face_age is not None:
            implied.append((birth[0] + face_age, age_tolerance(face_age)))
    if not implied or year is None:
        return DateCheck(year, img.date_source, False, None, None, None, faces)
    center = round(statistics.median(y for y, _ in implied))
    tolerance = statistics.median(t for _, t in implied)
    if len(implied) > 1:  # several faces agree -> more confident
        tolerance *= 0.8
    before_birth = any(f.calendar_age is not None and f.calendar_age < 0 for f in faces.values())
    suspicious = before_birth or abs(year - center) > tolerance
    return DateCheck(
        recorded_year=year,
        date_source=img.date_source,
        suspicious=suspicious,
        suggested_year=center if suspicious else None,
        low=round(center - tolerance),
        high=round(center + tolerance),
        faces=faces,
    )


def age_label(face: FaceInImage, date_suspicious: bool = False) -> str | None:
    """Short label for a face box: the calendar age, or ``~face age`` if the date is doubtful."""
    if face.calendar_age is not None and face.calendar_age >= 0 and not date_suspicious:
        return str(face.calendar_age)
    if face.face_age is not None:
        return f"~{face.face_age}"
    return None


def correct_image_date(s: Session, image_id: int, year: int, month: int | None = None) -> None:
    """Set a photo's date by hand (kept on re-index); keeps month/day where sensible."""
    from ..db.models import ImageProperty
    from ..pipeline.control import mark_pending

    img = s.get(Image, image_id)
    if img is None:
        raise LookupError(f"image {image_id} not found")
    if not 1800 <= year <= date.today().year + 1:
        raise ValueError(f"implausible year {year}")
    old = img.recorded_at
    m = month or (old.month if old else 1)
    d = old.day if old and month is None else 1
    try:
        new = datetime(year, m, d, *(old.timetuple()[3:6] if old else (12, 0, 0)))
    except ValueError:  # 29 February
        new = datetime(year, m, 28, 12, 0, 0)
    img.recorded_at = new
    img.date_source = "manual"
    for prop in s.scalars(
        select(ImageProperty).where(ImageProperty.image_id == image_id, ImageProperty.key == "year")
    ):
        prop.value = str(year)
    mark_pending(s, "embed", [image_id])  # the search document contains the date
