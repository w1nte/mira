"""GPS places: offline names from GeoNames, optional OpenStreetMap names, the map data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings

EARTH_KM = 6371.0088


@dataclass(frozen=True)
class PlaceName:
    """A name for a position, e.g. ``"Berlin, Germany"`` (locality Berlin, country Germany)."""

    name: str
    locality: str | None
    admin: str | None
    country: str | None
    source: str  # PlaceSource value


def geo_dir(settings: Settings) -> Path:
    """Downloaded and converted place-name data: ``<data_dir>/models/geo``."""
    return settings.data_dir / "models" / "geo"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def coords_name(lat: float, lon: float) -> str:
    """``52.520° N, 13.405° E`` - the name of a place nobody could name."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.3f}° {ns}, {abs(lon):.3f}° {ew}"
