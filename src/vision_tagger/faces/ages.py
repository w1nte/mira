"""Ages of persons in photos, birth-year estimation and detection of doubtful photo dates.

Two independent signals are combined:

* the **calendar age**: photo date minus the person's birth year (exact when both are right),
* the **face age**: estimated from the face (see :mod:`faces.agenet`; noisy, but independent
  of dates).

How the pieces fit together:

* Face ages are uncertain, and *how* uncertain depends on the age: a toddler's age is visible to
  within a year or two, an adult's only to within 5-10 years. :func:`age_sigma` models that, and
  everything below weighs evidence by it.
* Every birth year you set also teaches vision-tagger how far off the face model is on *your*
  photos: :class:`AgeCalibration` learns a correction per age range from faces of persons with a
  known birth year and a trustworthy date, and applies it to all other faces.
* Photo dates are not always right: camera clocks reset to 2000/1970, scans and copies only carry
  the file time, messenger apps strip EXIF. Date sources are weighted (EXIF and manual dates
  count most, the file time least), and a date is only called doubtful when the faces disagree
  far beyond the model's error. The calendar age stays the age shown everywhere else.
* When the faces and the date disagree, the user decides which is right: fix the date
  (:func:`correct_image_date`) or set the person's age in that photo (:func:`set_face_age`). A
  manual age replaces the face estimate everywhere and counts as nearly exact.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from sqlalchemy import case, func, select
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
# a date is doubtful when the faces imply a year this many standard deviations away
DOUBT_Z = {"exif": 3.5, "exif_digitized": 3.5, "exif_datetime": 3.0, "filename": 3.0}
DOUBT_Z_DEFAULT = 2.5  # file times and unknown sources
# error of an age set by hand: exact, but a birth year and a photo year only give the age to
# within a year
MANUAL_SIGMA = 0.6
MAX_AGE = 120


def reliability(date_source: str | None) -> float:
    return DATE_RELIABILITY.get(date_source or "", 0.3)


def age_sigma(age: float) -> float:
    """Typical error (one standard deviation, in years) of a face age estimate at ``age``."""
    return 1.5 + 0.12 * max(0.0, age)


def age_tolerance(face_age: float) -> float:
    """Allowed difference between calendar and face age before a date is called doubtful."""
    return DOUBT_Z_DEFAULT * age_sigma(face_age)


def face_quality(face_px: float | None, det_score: float | None, clusterable: bool = True) -> float:
    """0..1 weight of a face for age estimation: tiny, blurry or uncertain faces count less."""
    size = min(1.0, max(0.15, (face_px or 0) / 90.0))
    score = min(1.0, max(0.2, ((det_score or 0) - 0.4) / 0.45))
    return size * score * (1.0 if clusterable else 0.5)


# --------------------------------------------------------------------------- calibration
@dataclass
class AgeCalibration:
    """Correction of face ages per age range, learned from persons with a known birth year.

    ``offsets[i]`` is added to face ages around ``centers[i]`` (linear in between). Each offset is
    the median error in that range, shrunk towards zero by how uncertain that median is (few or
    contradicting examples -> little correction, many consistent ones -> nearly the full one).
    """

    centers: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    samples: int = 0

    EDGES = (0.0, 3.0, 7.0, 13.0, 20.0, 30.0, 45.0, 60.0, 120.0)
    MIN_TOTAL = 20

    @property
    def active(self) -> bool:
        return bool(self.centers) and any(abs(o) >= 0.5 for o in self.offsets)

    def apply(self, age: float | None) -> float | None:
        if age is None or not self.centers:
            return age
        return max(0.0, age + float(np.interp(age, self.centers, self.offsets)))

    @classmethod
    def fit(cls, pairs: list[tuple[float, float, float]]) -> AgeCalibration:
        """``pairs``: (face age, true age, weight). Implausible pairs are left out."""
        clean = [(p, t, w) for p, t, w in pairs if t >= 0 and abs(t - p) <= 3 * age_sigma(t) + 8]
        if len(clean) < cls.MIN_TOTAL:
            return cls(samples=len(clean))
        centers, offsets = [], []
        for lo, hi in zip(cls.EDGES, cls.EDGES[1:], strict=False):
            errors = [t - p for p, t, _w in clean if lo <= p < hi]
            centers.append((lo + min(hi, 90.0)) / 2.0)
            if len(errors) >= 3:
                median = statistics.median(errors)
                spread = max(1.0, statistics.pstdev(errors))
                se = 1.253 * spread / math.sqrt(len(errors))  # standard error of a median
                offsets.append(median * median**2 / (median**2 + se**2 + 1e-9))
            else:
                offsets.append(0.0)
        # keep the corrected scale monotonic (an older face never gets a younger age)
        for i in range(1, len(centers)):
            lower = centers[i - 1] + offsets[i - 1]
            if centers[i] + offsets[i] < lower:
                offsets[i] = lower - centers[i]
        return cls(tuple(centers), tuple(round(o, 2) for o in offsets), len(clean))


_CALIBRATION_CACHE: dict[tuple, AgeCalibration] = {}


def get_calibration(s: Session) -> AgeCalibration:
    """Calibration for this library (cached until faces, dates or birth years change)."""
    stats = tuple(
        s.execute(
            select(
                func.count(Face.id),
                func.max(Face.id),
                func.coalesce(func.sum(Face.age), 0),
                func.coalesce(func.sum(Person.birth_year), 0),
                func.count(case((Image.date_source == "manual", 1))),
            )
            .select_from(Face)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .join(Image, Image.id == Face.image_id)
            .where(Person.birth_year.is_not(None))
        ).one()
    )
    key = (str(s.get_bind().url), *stats)
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached
    rows = s.execute(
        select(Face.age, Face.face_px, Face.det_score, Image.recorded_at, Person.birth_year)
        .join(Cluster, Cluster.id == Face.cluster_id)
        .join(Person, Person.id == Cluster.person_id)
        .join(Image, Image.id == Face.image_id)
        .where(
            Person.birth_year.is_not(None),
            Face.age.is_not(None),
            Face.ignored.is_(False),
            Face.clusterable.is_(True),
            Image.recorded_at.is_not(None),
            Image.date_source.in_(["manual", "exif", "exif_digitized"]),
        )
    )
    pairs = [
        (float(age), float(when.year - birth), face_quality(px, det))
        for age, px, det, when, birth in rows
    ]
    calibration = AgeCalibration.fit(pairs)
    if len(_CALIBRATION_CACHE) > 16:
        _CALIBRATION_CACHE.clear()
    _CALIBRATION_CACHE[key] = calibration
    return calibration


# --------------------------------------------------------------------------- per person
@dataclass
class FaceAge:
    face_id: int
    image_id: int
    recorded_at: datetime | None
    date_source: str
    face_age: int | None  # calibrated face estimate
    calendar_age: int | None = None
    suspicious: bool = False
    suggested_year: int | None = None
    quality: float = 1.0
    manual: bool = False  # ``face_age`` is the age the user set

    @property
    def weight(self) -> float:
        return reliability(self.date_source)

    @property
    def sigma(self) -> float:
        """Typical error of ``face_age`` in years."""
        return MANUAL_SIGMA if self.manual else age_sigma(self.face_age or 0)

    @property
    def best_age(self) -> int | None:
        """The age set by the user, else the calendar age when the date fits, else the face
        estimate."""
        if self.manual:
            return self.face_age
        if self.calendar_age is not None and not self.suspicious:
            return self.calendar_age
        return self.face_age


@dataclass
class BirthYearEstimate:
    year: int
    low: int
    high: int
    samples: int
    se: float = 99.0  # standard error of ``year``

    @property
    def confidence(self) -> str:
        if self.samples >= 6 and self.se <= 1.5:
            return "high"
        if self.samples >= MIN_SAMPLES and self.se <= 3.5:
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

    Each photo counts by how much its date can be trusted, how good the face is and how precise
    a face age is at that age (children count much more than adults). Photos with a reliable
    date are used alone when there are enough of them. Outliers (wrong dates, misassigned faces)
    are removed before the weighted median is taken.
    """

    def samples_for(min_reliability: float) -> list[tuple[float, float]]:
        return [
            (
                f.recorded_at.year - f.face_age,
                f.weight * f.quality / f.sigma**2,
            )
            for f in faces
            if f.recorded_at is not None
            and f.face_age is not None
            and f.weight >= min_reliability
            and f.quality > 0
        ]

    samples = samples_for(0.6)
    if len(samples) < MIN_SAMPLES:  # not enough good dates: use everything, weighted
        samples = samples_for(0.0)
    if len(samples) < MIN_SAMPLES:
        return None
    values = [float(v) for v, _ in samples]
    weights = [w for _, w in samples]
    median = _weighted_quantile(values, weights, 0.5)
    mad = max(1.5, _weighted_quantile([abs(v - median) for v in values], weights, 0.5))
    inliers = [(v, w) for v, w in zip(values, weights, strict=True) if abs(v - median) <= 3 * mad]
    if len(inliers) < MIN_SAMPLES:
        inliers = list(zip(values, weights, strict=True))
    iv = [v for v, _ in inliers]
    iw = [w for _, w in inliers]
    year = round(_weighted_quantile(iv, iw, 0.5))
    # standard error: inverse-variance weights, inflated when the samples disagree more than
    # the model's error explains
    total = sum(iw)
    mean = sum(v * w for v, w in inliers) / total
    chi2 = sum(w * (v - mean) ** 2 for v, w in inliers) / max(1, len(inliers) - 1)
    se = math.sqrt(max(1.0, chi2) / total) if total > 0 else 99.0
    reliable = sum(1 for f in faces if f.weight >= 0.6)
    if reliable < MIN_SAMPLES:
        se = max(se, 2.0)  # dates themselves are doubtful
    half = max(1, round(1.96 * se))
    return BirthYearEstimate(
        year=year, low=year - half, high=year + half, samples=len(inliers), se=round(se, 2)
    )


def _check(face: FaceAge, birth_year: int, birth_se: float = 0.0, certain: bool = True) -> None:
    """Calendar age for ``face`` and whether its photo date contradicts the face."""
    if face.recorded_at is None:
        return
    face.calendar_age = face.recorded_at.year - birth_year
    if face.date_source == "manual":
        return  # set by the user
    if face.calendar_age < 0:  # "taken" before the person was born
        face.suspicious = certain or face.calendar_age < -round(2 * birth_se) - 1
    elif face.face_age is not None:
        mid = (face.calendar_age + face.face_age) / 2.0
        own = MANUAL_SIGMA if face.manual else age_sigma(mid)
        sigma = math.sqrt(own**2 + birth_se**2)
        z = DOUBT_Z.get(face.date_source, DOUBT_Z_DEFAULT) + (0.0 if certain else 0.5)
        face.suspicious = abs(face.calendar_age - face.face_age) > z * sigma
    if face.suspicious and face.face_age is not None:
        face.suggested_year = birth_year + face.face_age


def person_faces(
    s: Session,
    person_id: int | None,
    calibration: AgeCalibration | None = None,
    *,
    cluster_id: int | None = None,
) -> list[FaceAge]:
    """The faces of a person (or of one cluster: ``person_id=None, cluster_id=...``)."""
    calibration = calibration if calibration is not None else get_calibration(s)
    owner = Cluster.person_id == person_id if cluster_id is None else Face.cluster_id == cluster_id
    rows = s.execute(
        select(
            Face.id,
            Face.image_id,
            Face.age,
            Face.manual_age,
            Face.face_px,
            Face.det_score,
            Face.clusterable,
            Image.recorded_at,
            Image.date_source,
        )
        .join(Cluster, Cluster.id == Face.cluster_id)
        .join(Image, Image.id == Face.image_id)
        .where(owner, Face.ignored.is_(False), Image.missing.is_(False))
    )
    out = []
    for fid, iid, age, manual, px, det, clusterable, when, src in rows:
        if manual is not None:
            out.append(FaceAge(fid, iid, when, src or "", manual, quality=1.0, manual=True))
            continue
        corrected = calibration.apply(float(age)) if age is not None else None
        out.append(
            FaceAge(
                face_id=fid,
                image_id=iid,
                recorded_at=when,
                date_source=src or "",
                face_age=round(corrected) if corrected is not None else None,
                quality=face_quality(px, det, bool(clusterable)),
            )
        )
    return out


def analyse_person(s: Session, person: Person, today: date | None = None) -> PersonAges:
    today = today or date.today()
    faces = person_faces(s, person.id)
    estimate = estimate_birth_year(faces)
    if person.birth_year:
        birth, source, se, certain = person.birth_year, "set", 0.0, True
    elif estimate is not None and estimate.confidence != "low":
        birth, source, se, certain = estimate.year, "estimated", estimate.se, False
    else:
        birth, source, se, certain = None, None, 0.0, False
    if birth is not None:
        for f in faces:
            _check(f, birth, se, certain)
    faces.sort(key=lambda f: (f.recorded_at or datetime.min, f.face_id))
    return PersonAges(
        person=person,
        birth_year=birth,
        birth_source=source,
        estimate=estimate,
        faces=faces,
        current_age=(today.year - birth) if birth else None,
    )


def image_ages(
    s: Session, person_id: int | None = None, cluster_id: int | None = None
) -> dict[int, tuple[int, bool]]:
    """Age of one person (or the person of an unnamed cluster) in each of their photos:
    ``{image id: (age, estimated)}`` for the gallery's sort by age.

    The age is :attr:`FaceAge.best_age`: the age set by hand, else the calendar age when the
    date fits the face, else the face estimate (``estimated``). So photos with a wrong date
    still land where the person's age puts them. Several faces of the person in one photo: the
    youngest.
    """
    if person_id is not None:
        person = s.get(Person, person_id)
        faces = analyse_person(s, person).faces if person is not None else []
    elif cluster_id is not None:
        faces = person_faces(s, None, cluster_id=cluster_id)
    else:
        return {}
    out: dict[int, tuple[int, bool]] = {}
    for f in faces:
        age = f.best_age
        if age is None:
            continue
        estimated = not f.manual and (f.calendar_age is None or f.suspicious)
        current = out.get(f.image_id)
        if current is None or max(0, age) < current[0]:
            out[f.image_id] = (max(0, age), estimated)
    return out


# --------------------------------------------------------------------------- per image
@dataclass
class FaceInImage:
    face_id: int
    person: str
    face_age: int | None
    calendar_age: int | None
    birth_source: str | None
    manual: bool = False  # ``face_age`` is the age the user set
    person_id: int | None = None


@dataclass
class DateCheck:
    recorded_year: int | None
    date_source: str
    suspicious: bool
    suggested_year: int | None
    low: int | None
    high: int | None
    faces: dict[int, FaceInImage]


def _birth_years(
    s: Session, person_ids: set[int], calibration: AgeCalibration
) -> dict[int, tuple[int, str, float]]:
    """(birth year, "set" | "estimated", standard error) per person where known well enough."""
    out: dict[int, tuple[int, str, float]] = {}
    for pid in person_ids:
        person = s.get(Person, pid)
        if person is None:
            continue
        if person.birth_year:
            out[pid] = (person.birth_year, "set", 0.0)
            continue
        estimate = estimate_birth_year(person_faces(s, pid, calibration))
        if estimate is not None and estimate.confidence != "low":
            out[pid] = (estimate.year, "estimated", estimate.se)
    return out


def check_image_date(s: Session, image_id: int) -> DateCheck | None:
    """Ages of the known persons in an image and whether the photo date fits their faces."""
    img = s.get(Image, image_id)
    if img is None:
        return None
    rows = list(
        s.execute(
            select(Face.id, Face.age, Face.manual_age, Cluster.person_id, Person.name)
            .join(Cluster, Cluster.id == Face.cluster_id)
            .join(Person, Person.id == Cluster.person_id)
            .where(Face.image_id == image_id, Face.ignored.is_(False))
        )
    )
    if not rows:
        return None
    calibration = get_calibration(s)
    births = _birth_years(s, {r[3] for r in rows}, calibration)
    year = img.recorded_at.year if img.recorded_at else None
    faces: dict[int, FaceInImage] = {}
    implied: list[tuple[float, float]] = []  # (year implied by the face, its standard deviation)
    certain = True
    for face_id, raw_age, manual, pid, name in rows:
        if manual is not None:
            face_age, own_sigma = manual, MANUAL_SIGMA
        else:
            corrected = calibration.apply(float(raw_age)) if raw_age is not None else None
            face_age = round(corrected) if corrected is not None else None
            own_sigma = age_sigma(face_age or 0)
        birth = births.get(pid)
        calendar = (year - birth[0]) if birth and year is not None else None
        faces[face_id] = FaceInImage(
            face_id,
            name,
            face_age,
            calendar,
            birth[1] if birth else None,
            manual=manual is not None,
            person_id=pid,
        )
        if birth and face_age is not None:
            implied.append((birth[0] + face_age, math.hypot(own_sigma, birth[2])))
            certain = certain and birth[1] == "set"
    if not implied or year is None:
        return DateCheck(year, img.date_source, False, None, None, None, faces)
    # several faces: inverse-variance weighted mean of the years they imply
    weights = [1.0 / sd**2 for _, sd in implied]
    center = sum(y * w for (y, _), w in zip(implied, weights, strict=True)) / sum(weights)
    sigma = 1.0 / math.sqrt(sum(weights))
    before_birth = any(
        f.calendar_age is not None and f.calendar_age < 0 and f.birth_source == "set"
        for f in faces.values()
    )
    z = DOUBT_Z.get(img.date_source, DOUBT_Z_DEFAULT) + (0.0 if certain else 0.5)
    suspicious = img.date_source != "manual" and (before_birth or abs(year - center) > z * sigma)
    return DateCheck(
        recorded_year=year,
        date_source=img.date_source,
        suspicious=suspicious,
        suggested_year=round(center) if suspicious else None,
        low=math.floor(center - 2 * sigma),
        high=math.ceil(center + 2 * sigma),
        faces=faces,
    )


def age_label(face: FaceInImage, date_suspicious: bool = False) -> str | None:
    """Short label for a face box: the age set by the user, the calendar age, or ``~face age``
    if the date is doubtful."""
    if face.manual and face.face_age is not None:
        return str(face.face_age)
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


def set_face_age(s: Session, face_id: int, age: int | None) -> Face:
    """Set the person's age in a photo by hand (``None`` goes back to the face estimate).

    For a photo whose date is right but whose face looks older or younger than the person was:
    the manual age replaces the estimate in date checks, birth-year estimates and the gallery's
    sort by age, and is kept when the faces are detected again.
    """
    face = s.get(Face, face_id)
    if face is None:
        raise LookupError(f"face {face_id} not found")
    if age is not None and not 0 <= age <= MAX_AGE:
        raise ValueError(f"implausible age {age}")
    face.manual_age = age
    return face
