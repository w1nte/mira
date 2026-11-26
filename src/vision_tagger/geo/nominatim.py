"""Optional OpenStreetMap place names via Nominatim reverse geocoding.

Off by default (``geo.nominatim``): every lookup sends a position to the server. When on, the
public server's usage policy (https://operations.osmfoundation.org/policies/nominatim/) is
followed:

* at most one request per second, one at a time (``RateLimiter``, shared by the process);
* an identifying User-Agent (the application, no personal data);
* every answer is cached in the database (``geocode_cache``), keyed by the position rounded
  to three decimals (about 100 m), which is also the precision sent to the server;
* after a refusal or a network error no further requests are made in the same run.

OpenStreetMap data is © OpenStreetMap contributors, available under the ODbL.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from importlib import metadata
from typing import TYPE_CHECKING, Any

from ..db.models import GeocodeCache, PlaceSource
from . import PlaceName

if TYPE_CHECKING:
    import httpx

    from ..db.session import Database

log = logging.getLogger(__name__)

PUBLIC_URL = "https://nominatim.openstreetmap.org"
ZOOM = 14  # suburb / neighbourhood level
PRECISION = 3  # decimals of the position sent and cached (~100 m)


def user_agent() -> str:
    try:
        version = metadata.version("vision-tagger")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        version = "dev"
    return f"vision-tagger/{version} (self-hosted photo library)"


class GeocoderUnavailable(RuntimeError):
    """The server refused, failed or could not be reached; stop asking for this run."""


class RateLimiter:
    """At most one call per ``interval`` seconds, one at a time."""

    def __init__(
        self,
        interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.last: float | None = None

    def run(self, fn: Callable[[], Any]) -> Any:
        with self.lock:
            if self.last is not None:
                wait = self.last + self.interval - self.clock()
                if wait > 0:
                    self.sleep(wait)
            try:
                return fn()
            finally:
                self.last = self.clock()


#: one limit for every client of this process
PUBLIC_LIMIT = RateLimiter(1.0)


class Nominatim:
    def __init__(
        self,
        base_url: str = PUBLIC_URL,
        *,
        language: str = "en",
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.language = language or "en"
        self.limiter = limiter or PUBLIC_LIMIT
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"User-Agent": user_agent(), "Accept-Language": self.language},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self.requests = 0

    def close(self) -> None:
        self.client.close()

    def cache_key(self, lat: float, lon: float) -> str:
        return f"nominatim:{ZOOM}:{self.language}:{lat:.{PRECISION}f},{lon:.{PRECISION}f}"

    def reverse(self, lat: float, lon: float) -> dict[str, Any] | None:
        """The raw answer for a position, ``None`` where there is no address (open sea)."""
        import httpx

        params = {
            "format": "jsonv2",
            "lat": f"{lat:.{PRECISION}f}",
            "lon": f"{lon:.{PRECISION}f}",
            "zoom": ZOOM,
            "addressdetails": 1,
            "accept-language": self.language,
        }

        def request() -> httpx.Response:
            self.requests += 1
            return self.client.get("/reverse", params=params)

        try:
            resp = self.limiter.run(request)
        except httpx.HTTPError as exc:
            raise GeocoderUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code != 200:
            raise GeocoderUnavailable(f"HTTP {resp.status_code} from {self.base_url}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise GeocoderUnavailable("invalid answer") from exc
        if not isinstance(data, dict) or "error" in data:
            return None
        return data

    def lookup(self, db: Database, lat: float, lon: float, *, refresh: bool = False):
        """:meth:`reverse` through the database cache (``refresh`` asks again)."""
        key = self.cache_key(lat, lon)
        if not refresh:
            with db.session() as s:
                row = s.get(GeocodeCache, key)
                if row is not None:
                    return row.result
        result = self.reverse(round(lat, PRECISION), round(lon, PRECISION))
        with db.write() as s:
            s.merge(GeocodeCache(key=key, result=result))
        return result


def osm_name(result: dict[str, Any] | None) -> PlaceName | None:
    """``"Prenzlauer Berg, Berlin"`` (district, town) or ``"Füssen, Germany"`` (town, country)."""
    if not result:
        return None
    a = result.get("address") or {}

    def first(*keys: str) -> str | None:
        for k in keys:
            v = a.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    town = first("city", "town", "village", "municipality", "hamlet")
    district = first("suburb", "city_district", "borough", "quarter", "neighbourhood")
    admin = first("state", "region", "province", "county")
    country = first("country")
    if district and town and district != town:
        name = f"{district}, {town}"
    elif town:
        name = f"{town}, {country}" if country and country != town else town
    elif district:
        name = f"{district}, {admin or country}" if (admin or country) else district
    else:
        named = result.get("name")
        if isinstance(named, str) and named.strip():
            name = named.strip()
        else:
            parts = [p.strip() for p in str(result.get("display_name") or "").split(",")]
            parts = [p for p in parts if p]
            if not parts:
                return None
            name = ", ".join(parts[:2])
    return PlaceName(name, town or district, admin, country, PlaceSource.OSM)
