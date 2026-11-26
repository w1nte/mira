"""Data for the map page: heatmap points and places of the filtered photos (read only)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import Image, Place
from ..query import ImageFilter, conditions

#: heatmap cells; above this many the positions are rounded more coarsely
MAX_POINTS = 15_000


def heat_points(s: Session, f: ImageFilter, max_points: int = MAX_POINTS) -> dict[str, Any]:
    """``[lat, lon, photos]`` per cell of about 100 m (coarser for very spread libraries).

    100,000 photos taken at a few hundred places become a few thousand short triples.
    """
    lat_r, lon_r = func.round(Image.lat, 3), func.round(Image.lon, 3)
    rows = s.execute(
        select(lat_r, lon_r, func.count())
        .where(*conditions(f), Image.lat.is_not(None), Image.lon.is_not(None))
        .group_by(lat_r, lon_r)
    ).all()
    decimals = 3
    cells: dict[tuple[float, float], int] = {(float(a), float(b)): int(n) for a, b, n in rows}
    while len(cells) > max_points and decimals > 0:
        decimals -= 1
        coarse: dict[tuple[float, float], int] = defaultdict(int)
        for (a, b), n in cells.items():
            coarse[(round(a, decimals), round(b, decimals))] += n
        cells = coarse
    points = [[a, b, n] for (a, b), n in sorted(cells.items())]
    bounds = None
    if points:
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    return {
        "points": points,
        "photos": sum(cells.values()),
        "max": max(cells.values(), default=0),
        "decimals": decimals,
        "bounds": bounds,
    }


def place_summaries(s: Session, f: ImageFilter) -> list[dict[str, Any]]:
    """Places of the filtered photos, most photos first, with the filtered photo counts."""
    n = func.count(Image.id)
    rows = s.execute(
        select(
            Place.id,
            Place.name,
            Place.lat,
            Place.lon,
            Place.radius_m,
            Place.user_named,
            n,
        )
        .join(Image, Image.place_id == Place.id)
        .where(*conditions(f))
        .group_by(Place.id)
        .order_by(n.desc(), Place.id)
    ).all()
    return [
        {
            "id": pid,
            "name": name,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "radius_m": round(radius or 0.0),
            "user_named": bool(user_named),
            "count": count,
        }
        for pid, name, lat, lon, radius, user_named, count in rows
    ]


def place_detail(s: Session, place_id: int, f: ImageFilter, limit: int = 8) -> dict | None:
    """One place with the number of filtered photos in it and the newest of them."""
    place = s.get(Place, place_id)
    if place is None:
        return None
    conds = conditions(f.copy(places=[place_id]))
    count = s.scalar(select(func.count()).select_from(Image).where(*conds)) or 0
    images = list(
        s.scalars(
            select(Image.id)
            .where(*conds)
            .order_by(Image.recorded_at.desc(), Image.id.desc())
            .limit(limit)
        )
    )
    return {
        "id": place.id,
        "name": place.name,
        "locality": place.locality,
        "admin": place.admin,
        "country": place.country,
        "source": place.source,
        "user_named": place.user_named,
        "lat": round(place.lat, 5),
        "lon": round(place.lon, 5),
        "radius_m": round(place.radius_m or 0.0),
        "count": count,
        "images": images,
    }


def gps_counts(s: Session, f: ImageFilter) -> dict[str, int]:
    """Filtered photos with a position, and photos with a position but no place yet."""
    with_gps = (
        s.scalar(
            select(func.count()).select_from(Image).where(*conditions(f), Image.lat.is_not(None))
        )
        or 0
    )
    unplaced = (
        s.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.lat.is_not(None), Image.place_id.is_(None))
        )
        or 0
    )
    return {"gps": with_gps, "unplaced": unplaced}
