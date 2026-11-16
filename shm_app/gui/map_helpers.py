"""Shared geographic geometry and Matplotlib basemap rendering helpers."""

from __future__ import annotations

from functools import lru_cache
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from matplotlib.collections import PolyCollection
import numpy as np

from shm_app.seismology.traveltime import sphere_add, sphere_diff

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_COUNTRIES_ASSET = _ASSETS_DIR / "world_countries.json.gz"


@lru_cache(maxsize=1)
def load_world_countries() -> List[Dict[str, Any]]:
    """Load bundled simplified Natural Earth country polygons."""
    if not _COUNTRIES_ASSET.is_file():
        return []
    try:
        with gzip.open(_COUNTRIES_ASSET, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


@lru_cache(maxsize=1)
def get_country_polygons() -> List[np.ndarray]:
    """Return cached numpy polygon arrays for bundled world countries."""
    polys: List[np.ndarray] = []
    for country in load_world_countries():
        for ring in country.get("r", []):
            arr = np.asarray(ring, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                polys.append(arr)
    return polys


def split_antimeridian_polyline(lons: Sequence[float], lats: Sequence[float]) -> List[np.ndarray]:
    """Split a lon/lat polyline wherever consecutive points jump across +-180 deg."""
    if len(lons) < 2:
        return []
    segments: List[np.ndarray] = []
    cur_x: List[float] = [float(lons[0])]
    cur_y: List[float] = [float(lats[0])]
    for i in range(1, len(lons)):
        x0, x1 = float(lons[i - 1]), float(lons[i])
        y1 = float(lats[i])
        if abs(x1 - x0) > 180.0:
            if len(cur_x) >= 2:
                segments.append(np.column_stack([cur_x, cur_y]))
            cur_x = [x1]
            cur_y = [y1]
        else:
            cur_x.append(x1)
            cur_y.append(y1)
    if len(cur_x) >= 2:
        segments.append(np.column_stack([cur_x, cur_y]))
    return segments


def great_circle_segments(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    n_pts: int = 64,
) -> List[np.ndarray]:
    """Compute great-circle arc segments between two points without antimeridian wrapping artifacts."""
    dist_deg, az_12, _ = sphere_diff(lat1, lon1, lat2, lon2)
    if dist_deg < 1e-4:
        return []
    if abs(dist_deg - 180.0) < 0.05:
        n_pts = max(n_pts, 90)
    lons: List[float] = []
    lats: List[float] = []
    for f in np.linspace(0.0, 1.0, max(8, n_pts)):
        plat, plon = sphere_add(lat1, lon1, float(f) * dist_deg, az_12)
        lons.append(plon)
        lats.append(plat)
    return split_antimeridian_polyline(lons, lats)


def distance_ring_segments(lat0: float, lon0: float, dist_deg: float, n_pts: int = 181) -> List[np.ndarray]:
    """Compute concentric angular distance circle segments around (lat0, lon0)."""
    lons: List[float] = []
    lats: List[float] = []
    for az in np.linspace(0.0, 360.0, max(36, n_pts)):
        plat, plon = sphere_add(lat0, lon0, dist_deg, float(az))
        lons.append(plon)
        lats.append(plat)
    return split_antimeridian_polyline(lons, lats)


def draw_country_basemap(
    ax: Any,
    polys: Optional[Sequence[np.ndarray]] = None,
    *,
    edge_linewidth: float = 0.65,
    grid_color: str = "#1b2b3f",
    axis_color: str = "#293d57",
    axis_lw: float = 0.8,
    axis_alpha: float = 0.7,
    xticks: Optional[np.ndarray] = None,
    yticks: Optional[np.ndarray] = None,
) -> None:
    """Draw shaded world country polygons, optional ticks, gridlines, and zero axes on ax."""
    country_polys = get_country_polygons() if polys is None else polys
    if country_polys:
        ax.add_collection(
            PolyCollection(
                country_polys,
                facecolors="#182536",
                edgecolors="#344a66",
                linewidths=edge_linewidth,
                zorder=1,
            )
        )
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, color=grid_color, ls=":", lw=0.6, zorder=0)
    ax.axhline(0.0, color=axis_color, lw=axis_lw, alpha=axis_alpha, zorder=0)
    ax.axvline(0.0, color=axis_color, lw=axis_lw, alpha=axis_alpha, zorder=0)
