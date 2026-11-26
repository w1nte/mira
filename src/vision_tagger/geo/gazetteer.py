"""Offline place names: the town a position belongs to, from GeoNames data.

Data
    GeoNames "cities with a population of at least 1000" (https://www.geonames.org/,
    CC BY 4.0). download.geonames.org publishes a new dump every day, so a download from there
    cannot be checked against a fixed checksum. The default download is an immutable copy of
    that data from the npm registry instead: the ``all-the-cities`` package, version 3.1.0
    (MIT licensed; its ``cities.pbf`` holds the GeoNames name, country, admin code, feature
    code, population and position of 135,000 towns). Its size and SHA-256 are checked.
    ``geo.cities_file`` can point at an official GeoNames dump instead (``cities500.zip``,
    ``cities1000.txt``, ...: the tab-separated "geoname" table).

Either source is converted once into a compact ``.npz`` file in ``<data_dir>/models/geo``;
lookups use a haversine ball tree over the positions.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
import threading
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..db.models import PlaceSource
from . import EARTH_KM, PlaceName, geo_dir

if TYPE_CHECKING:
    import httpx

    from ..config import Settings

log = logging.getLogger(__name__)

SOURCE_URL = "https://registry.npmjs.org/all-the-cities/-/all-the-cities-3.1.0.tgz"
SOURCE_SHA256 = "9f482772ce7754c1d44bc66d7d52154bcb891c1efe457212a5fa07fb19e0d8be"
SOURCE_SIZE = 3_226_187
SOURCE_MEMBER = "package/cities.pbf"
FORMAT = 1  # layout of the converted .npz file
DEFAULT_FILE = f"cities-all-the-cities-3.1.0-v{FORMAT}.npz"
LOCAL_FILE = f"cities-local-v{FORMAT}.npz"

#: not a town of its own: sections of a city, historical, abandoned or destroyed places
SKIP_FEATURES = frozenset({"PPLX", "PPLH", "PPLQ", "PPLW", "PPLCH"})
#: farther than this from every town, a place is named by its coordinates
MAX_TOWN_KM = 40.0
_MAX_MEMBER = 64 * 1024 * 1024

#: (name, country code, admin1 code, feature code, population, lat, lon)
Row = tuple[str, str, str, str, int, float, float]


@dataclass(frozen=True)
class Town:
    name: str
    country_code: str
    admin_code: str
    population: int
    lat: float
    lon: float
    distance_km: float


@cache
def country_names() -> dict[str, str]:
    """ISO 3166 alpha-2 code -> English name (GeoNames countryInfo, CC BY 4.0)."""
    text = (
        resources.files("vision_tagger")
        .joinpath("resources/geo/countries.json")
        .read_text(encoding="utf-8")
    )
    return {k: v for k, v in json.loads(text).items() if not k.startswith("@")}


class _Strings:
    """Many short strings in one UTF-8 buffer (the .npz holds no pickled object arrays)."""

    def __init__(self, blob: np.ndarray, offsets: np.ndarray) -> None:
        self.blob = blob
        self.offsets = offsets

    @staticmethod
    def pack(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encoded = [v.encode("utf-8") for v in values]
        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(e) for e in encoded])
        return np.frombuffer(b"".join(encoded), dtype=np.uint8), offsets

    def __getitem__(self, i: int) -> str:
        return bytes(self.blob[self.offsets[i] : self.offsets[i + 1]]).decode("utf-8")


class Gazetteer:
    """Towns with positions and populations; finds the town a position belongs to."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        population: np.ndarray,
        country: np.ndarray,
        names: _Strings,
        admins: _Strings,
        meta: dict[str, Any] | None = None,
    ) -> None:
        from sklearn.neighbors import BallTree

        self.lat = lat
        self.lon = lon
        self.population = population
        self.country = country
        self.names = names
        self.admins = admins
        self.meta = meta or {}
        self._tree = (
            BallTree(np.radians(np.column_stack([lat, lon]).astype(np.float64)), metric="haversine")
            if len(lat)
            else None
        )

    def __len__(self) -> int:
        return len(self.lat)

    # ------------------------------------------------------------------ building / storage
    @classmethod
    def from_rows(cls, rows: Iterable[Row], meta: dict[str, Any] | None = None) -> Gazetteer:
        names: list[str] = []
        admins: list[str] = []
        countries: list[str] = []
        pops: list[int] = []
        lats: list[float] = []
        lons: list[float] = []
        for name, cc, admin, feature, pop, lat, lon in rows:
            if (
                feature in SKIP_FEATURES
                or not name
                or not (-90 <= lat <= 90 and -180 <= lon <= 180)
            ):
                continue
            names.append(name)
            admins.append(admin or "")
            countries.append((cc or "")[:2].upper())
            pops.append(max(0, min(int(pop or 0), 2**32 - 1)))
            lats.append(lat)
            lons.append(lon)
        return cls(
            np.asarray(lats, dtype=np.float32),
            np.asarray(lons, dtype=np.float32),
            np.asarray(pops, dtype=np.uint32),
            np.asarray(countries, dtype="<U2"),
            _Strings(*_Strings.pack(names)),
            _Strings(*_Strings.pack(admins)),
            meta,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                lat=self.lat,
                lon=self.lon,
                population=self.population,
                country=self.country,
                names_blob=self.names.blob,
                names_offsets=self.names.offsets,
                admins_blob=self.admins.blob,
                admins_offsets=self.admins.offsets,
                meta=np.array(json.dumps({**self.meta, "format": FORMAT})),
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Gazetteer:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path.name}: unknown format {meta.get('format')}")
            return cls(
                data["lat"],
                data["lon"],
                data["population"],
                data["country"],
                _Strings(data["names_blob"], data["names_offsets"]),
                _Strings(data["admins_blob"], data["admins_offsets"]),
                meta,
            )

    # ------------------------------------------------------------------ lookups
    def nearest(self, lat: float, lon: float, k: int = 16) -> list[Town]:
        if self._tree is None:
            return []
        dist, idx = self._tree.query(np.radians([[lat, lon]]), k=min(k, len(self)))
        return [
            Town(
                name=self.names[int(i)],
                country_code=str(self.country[i]),
                admin_code=self.admins[int(i)],
                population=int(self.population[i]),
                lat=float(self.lat[i]),
                lon=float(self.lon[i]),
                distance_km=float(d) * EARTH_KM,
            )
            for d, i in zip(dist[0], idx[0], strict=True)
        ]

    def town(self, lat: float, lon: float, max_km: float = MAX_TOWN_KM) -> Town | None:
        """The town a photo taken here belongs to: near, and big towns reach farther.

        Score = distance / (population / 1000) ** 0.25: a city of a million people 5.6 km away
        ties with a village of 1000 people 1 km away. So a photo in a city's suburb gets the
        city's name, a photo in a village next to it the village's name.
        """
        best: tuple[float, Town] | None = None
        for t in self.nearest(lat, lon):
            if t.distance_km > max_km:
                continue
            score = t.distance_km / (max(t.population, 1000) / 1000.0) ** 0.25
            if best is None or score < best[0]:
                best = (score, t)
        return best[1] if best else None

    def describe(self, lat: float, lon: float) -> PlaceName | None:
        """``"Berlin, Germany"``; US towns as ``"Austin, TX"``; city states just ``"Monaco"``."""
        t = self.town(lat, lon)
        if t is None:
            return None
        country = country_names().get(t.country_code) or t.country_code or None
        admin = t.admin_code if t.country_code == "US" and t.admin_code.isalpha() else None
        if admin:
            name = f"{t.name}, {admin}"
        elif country and country != t.name:
            name = f"{t.name}, {country}"
        else:
            name = t.name
        return PlaceName(name, t.name, admin, country, PlaceSource.GEONAMES)


# --------------------------------------------------------------------------- parsers
def parse_pbf(buf: bytes) -> Iterator[Row]:
    """Towns from ``all-the-cities``' ``cities.pbf``: length-prefixed protobuf messages.

    Fields (see the package's index.js): 1 id, 2 name, 3 country, 4 alternative name,
    5 / 6 municipality, 7 feature code, 8 admin code, 9 population, 10 / 11 longitude /
    latitude (zigzag varints, delta-coded across messages, in 1e-5 degrees).
    """
    data = memoryview(buf)
    n = len(data)
    pos = 0
    lat_acc = lon_acc = 0

    def varint() -> int:
        nonlocal pos
        result = shift = 0
        while True:
            if pos >= n or shift > 63:
                raise ValueError("truncated or invalid cities.pbf")
            b = data[pos]
            pos += 1
            result |= (b & 0x7F) << shift
            if b < 0x80:
                return result
            shift += 7

    while pos < n:
        end = varint() + pos
        if end > n:
            raise ValueError("truncated cities.pbf")
        fields: dict[int, Any] = {}
        while pos < end:
            key = varint()
            tag, wire = key >> 3, key & 7
            if wire == 0:
                value = varint()
                if tag in (1, 10, 11):
                    value = (value >> 1) ^ -(value & 1)
                if tag == 10:
                    lon_acc += value
                elif tag == 11:
                    lat_acc += value
                else:
                    fields[tag] = value
            elif wire == 2:
                length = varint()
                if pos + length > end:
                    raise ValueError("truncated cities.pbf")
                fields[tag] = bytes(data[pos : pos + length]).decode("utf-8", "replace")
                pos += length
            else:
                raise ValueError(f"unexpected protobuf wire type {wire}")
        yield (
            str(fields.get(2, "")),
            str(fields.get(3, "")),
            str(fields.get(8, "")),
            str(fields.get(7, "")),
            int(fields.get(9, 0)),
            lat_acc / 1e5,
            lon_acc / 1e5,
        )


def parse_geonames(lines: Iterable[str]) -> Iterator[Row]:
    """Populated places (feature class P) from a GeoNames "geoname" table (tab separated)."""
    for line in lines:
        cols = line.rstrip("\r\n").split("\t")
        if len(cols) < 15 or cols[6] != "P":
            continue
        try:
            lat, lon = float(cols[4]), float(cols[5])
            pop = int(cols[14] or 0)
        except ValueError:
            continue
        yield (cols[1], cols[8], cols[10], cols[7], pop, lat, lon)


def _geonames_lines(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [
                m
                for m in zf.infolist()
                if m.filename.lower().endswith(".txt") and "readme" not in m.filename.lower()
            ]
            if not members:
                raise ValueError(f"{path}: no GeoNames table inside")
            with zf.open(members[0]) as raw:
                yield from io.TextIOWrapper(raw, encoding="utf-8")
        return
    with path.open(encoding="utf-8") as fh:
        yield from fh


def _tar_member(archive: bytes, name: str) -> bytes:
    """One file of a .tgz, read in memory (nothing is extracted to disk)."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        member = tf.getmember(name)
        if not member.isfile() or member.size > _MAX_MEMBER:
            raise ValueError(f"unexpected archive member {name}")
        fh = tf.extractfile(member)
        if fh is None:
            raise ValueError(f"cannot read {name}")
        return fh.read()


# --------------------------------------------------------------------------- download / cache
def download_default(target: Path, client: httpx.Client | None = None) -> Gazetteer:
    """Download the pinned GeoNames copy, verify it and store the converted ``target``."""
    import httpx

    log.info(
        "downloading place names (GeoNames data, %d KB) from %s", SOURCE_SIZE // 1024, SOURCE_URL
    )
    buf = bytearray()
    own = client is None
    http = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        with http.stream("GET", SOURCE_URL) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                buf += chunk
                if len(buf) > SOURCE_SIZE:
                    raise ValueError("place-name download is larger than expected")
    finally:
        if own:
            http.close()
    if len(buf) != SOURCE_SIZE:
        raise ValueError(f"place-name download has {len(buf)} bytes, expected {SOURCE_SIZE}")
    digest = hashlib.sha256(buf).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"checksum mismatch for {SOURCE_URL}: {digest}")
    gaz = Gazetteer.from_rows(
        parse_pbf(_tar_member(bytes(buf), SOURCE_MEMBER)),
        {"source": SOURCE_URL, "sha256": SOURCE_SHA256},
    )
    gaz.save(target)
    log.info("saved %d towns to %s", len(gaz), target)
    return gaz


def _local(source: Path, target: Path) -> Gazetteer:
    """A user-provided GeoNames dump, converted once (again when the file changes)."""
    st = source.stat()
    signature = f"{source.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if target.exists():
        try:
            gaz = _load_cached(target)
            if gaz.meta.get("signature") == signature:
                return gaz
        except (OSError, ValueError, KeyError) as exc:
            log.info("rebuilding %s: %s", target.name, exc)
    gaz = Gazetteer.from_rows(
        parse_geonames(_geonames_lines(source)), {"source": str(source), "signature": signature}
    )
    gaz.save(target)
    log.info("converted %d towns from %s", len(gaz), source)
    return _load_cached(target)


_lock = threading.Lock()
_download_lock = threading.Lock()
_loaded: dict[Path, tuple[float, Gazetteer]] = {}


def _load_cached(path: Path) -> Gazetteer:
    mtime = path.stat().st_mtime
    with _lock:
        hit = _loaded.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    gaz = Gazetteer.load(path)
    with _lock:
        _loaded[path] = (mtime, gaz)
    return gaz


def load_gazetteer(settings: Settings, client: httpx.Client | None = None) -> Gazetteer | None:
    """The towns for offline names: downloaded once, then loaded from disk (and memory).

    ``None`` (with a log message) if they are not available: places are then named by their
    coordinates until a later run can get them.
    """
    cfg = settings.geo
    folder = geo_dir(settings)
    try:
        if cfg.cities_file:
            return _local(Path(cfg.cities_file).expanduser(), folder / LOCAL_FILE)
        target = folder / DEFAULT_FILE
        if not target.exists():
            if not cfg.download_gazetteer:
                return None
            with _download_lock:
                if not target.exists():
                    download_default(target, client)
        return _load_cached(target)
    except Exception as exc:  # a missing name source must never break indexing
        log.warning("place names are not available: %s", exc)
        return None
