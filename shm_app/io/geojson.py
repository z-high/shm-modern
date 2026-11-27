"""GeoJSON RFC 7946 exporter and parser for seismic events, confidence ellipses, and stations."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from shm_app.core.models import AnalysisParamSet, Trace
from shm_app.seismology.traveltime import sphere_add

KM_PER_DEG = 111.1949266


def _json_safe(obj: Any) -> Any:
    """Recursively convert values into strict RFC-8259 JSON-safe primitives (no NaN/Inf)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if type(obj).__module__ == "numpy":
        if hasattr(obj, "tolist"):
            return _json_safe(obj.tolist())
        try:
            val = float(obj)
            return val if math.isfinite(val) else None
        except (TypeError, ValueError):
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def _resolve_ellipse(params: AnalysisParamSet) -> Optional[Tuple[float, float, float]]:
    """Return (semi_major_km, semi_minor_km, strike_deg) if available, finite, and positive."""
    err_ell = getattr(params, "error_ellipse", None)
    if err_ell is not None and len(err_ell) == 3:
        a_km, b_km, strike_deg = float(err_ell[0]), float(err_ell[1]), float(err_ell[2])
        if all(math.isfinite(v) for v in (a_km, b_km, strike_deg)):
            if b_km > a_km:
                a_km, b_km = b_km, a_km
            if a_km > 0.0 and b_km > 0.0:
                return a_km, b_km, strike_deg % 180.0
    d_lat = float(getattr(params, "d_lat", 0.0) or 0.0)
    d_lon = float(getattr(params, "d_lon", 0.0) or 0.0)
    if math.isfinite(d_lat) and math.isfinite(d_lon) and d_lat > 0.0 and d_lon > 0.0:
        if d_lat >= d_lon:
            return d_lat, d_lon, 0.0
        return d_lon, d_lat, 90.0
    return None


def _build_confidence_ellipse_ring(
    center_lat: float,
    center_lon: float,
    semi_major_km: float,
    semi_minor_km: float,
    strike_deg: float,
    num_points: int = 72,
) -> List[List[float]]:
    """Project horizontal confidence ellipse onto WGS84 [lon, lat] counterclockwise closed ring (RFC 7946)."""
    theta = math.radians(strike_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    ring: List[List[float]] = []
    n_pts = max(8, int(num_points))
    for k in range(n_pts):
        psi = 2.0 * math.pi * k / float(n_pts)
        u = semi_major_km * math.cos(psi)
        v = semi_minor_km * math.sin(psi)
        # Orthonormal right-handed basis in (East=x, North=y):
        # e_major = (sin_t, cos_t), e_minor_ccw = (-cos_t, sin_t) -> det = +1 (CCW)
        dx_km = u * sin_t - v * cos_t
        dy_km = u * cos_t + v * sin_t
        dist_deg = math.hypot(dx_km, dy_km) / KM_PER_DEG
        azim_deg = math.degrees(math.atan2(dx_km, dy_km)) % 360.0
        pt_lat, pt_lon = sphere_add(center_lat, center_lon, dist_deg, azim_deg)
        ring.append([round(float(pt_lon), 6), round(float(pt_lat), 6)])

    ring.append(list(ring[0]))
    return ring


def export_geojson(
    traces: Sequence[Trace],
    params: AnalysisParamSet,
    include_stations: bool = True,
    include_ellipse: bool = True,
    focal_mechanism: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export event epicenter, 95% confidence ellipse Polygon, and stations as RFC 7946 FeatureCollection."""
    features: List[Dict[str, Any]] = []

    ellipse_tuple = _resolve_ellipse(params)

    # 1. Event Epicenter Point Feature
    epi_props: Dict[str, Any] = {
        "feature_type": "epicenter",
        "event_id": int(getattr(params, "event_id", 0) or 0),
        "origin": params.origin or "",
        "latitude": float(params.lat),
        "longitude": float(params.lon),
        "depth_km": float(params.depth),
        "d_depth_km": float(getattr(params, "d_depth", 0.0) or 0.0),
        "event_type": params.event_type or "t",
        "region_name": params.region_name or "",
        "source": params.source or "SHM",
        "mag_ml": float(getattr(params, "mag_ml", 0.0) or 0.0),
        "mag_mb": float(getattr(params, "mag_mb", 0.0) or 0.0),
        "mag_ms": float(getattr(params, "mag_ms", 0.0) or 0.0),
        "mag_mw": float(getattr(params, "mag_mw", 0.0) or 0.0),
        "mag_mbb": float(getattr(params, "mag_mbb", 0.0) or 0.0),
        "rmse": float(getattr(params, "rmse", 0.0) or 0.0),
        "wrmse": float(getattr(params, "wrmse", 0.0) or 0.0),
        "chi2_reduced": float(getattr(params, "chi2_reduced", 0.0) or 0.0),
        "condition_number": float(getattr(params, "condition_number", 1.0) or 1.0),
    }
    if ellipse_tuple is not None:
        epi_props["error_ellipse"] = {
            "semi_major_km": ellipse_tuple[0],
            "semi_minor_km": ellipse_tuple[1],
            "strike_deg": ellipse_tuple[2],
            "confidence_level": 0.95,
        }
    fm = focal_mechanism if focal_mechanism is not None else getattr(params, "focal_mechanism", None)
    if isinstance(fm, dict) and fm:
        epi_props["focal_mechanism"] = dict(fm)

    features.append(
        {
            "type": "Feature",
            "id": f"event-{epi_props['event_id'] or 1}",
            "geometry": {
                "type": "Point",
                "coordinates": [float(params.lon), float(params.lat)],
            },
            "properties": epi_props,
        }
    )

    # 2. 95% Confidence Ellipse Polygon Feature
    if (
        include_ellipse
        and ellipse_tuple is not None
        and math.isfinite(float(params.lat))
        and math.isfinite(float(params.lon))
    ):
        a_km, b_km, strike_deg = ellipse_tuple
        ring = _build_confidence_ellipse_ring(params.lat, params.lon, a_km, b_km, strike_deg)
        features.append(
            {
                "type": "Feature",
                "id": f"confidence-ellipse-{epi_props['event_id'] or 1}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
                "properties": {
                    "feature_type": "confidence_ellipse",
                    "confidence_level": 0.95,
                    "semi_major_km": float(a_km),
                    "semi_minor_km": float(b_km),
                    "strike_deg": float(strike_deg),
                    "center_lat": float(params.lat),
                    "center_lon": float(params.lon),
                },
            }
        )

    # 3. Station Point Features with residuals & distances
    if include_stations and traces:
        grouped: Dict[str, Dict[str, Any]] = {}
        param_resids = getattr(params, "residuals", None) or {}

        for tr in traces:
            if abs(float(tr.lat)) <= 1e-9 and abs(float(tr.lon)) <= 1e-9:
                continue
            sta = (tr.station or "").strip().upper()
            if not sta:
                continue
            entry = grouped.get(sta)
            if entry is None:
                entry = {
                    "station": sta,
                    "network": (tr.network or "").strip().upper(),
                    "lat": float(tr.lat),
                    "lon": float(tr.lon),
                    "distance_deg": float(tr.distance),
                    "azimuth_deg": float(tr.azimuth),
                    "channels": [],
                    "picks": [],
                    "residual_s": None,
                }
                grouped[sta] = entry

            ch3 = tr.channel3
            if ch3 not in entry["channels"]:
                entry["channels"].append(ch3)
            if tr.distance > 0.0 and entry["distance_deg"] == 0.0:
                entry["distance_deg"] = float(tr.distance)
            if tr.azimuth != 0.0 and entry["azimuth_deg"] == 0.0:
                entry["azimuth_deg"] = float(tr.azimuth)

            for pk in tr.picks or []:
                entry["picks"].append(
                    {
                        "phase": pk.name,
                        "onset": pk.onset,
                        "residual_s": float(pk.resid),
                        "weight": int(pk.weight),
                        "quality": int(pk.quality),
                        "polarity": pk.sign_char,
                    }
                )
                if entry["residual_s"] is None or (entry["residual_s"] == 0.0 and pk.resid != 0.0):
                    entry["residual_s"] = float(pk.resid)

            if entry["residual_s"] is None and sta in param_resids:
                entry["residual_s"] = float(param_resids[sta])

        for sta in sorted(grouped.keys()):
            st = grouped[sta]
            resid_val = float(st["residual_s"]) if st["residual_s"] is not None else 0.0
            features.append(
                {
                    "type": "Feature",
                    "id": f"station-{sta}",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(st["lon"]), float(st["lat"])],
                    },
                    "properties": {
                        "feature_type": "station",
                        "station": sta,
                        "network": st["network"],
                        "channels": st["channels"],
                        "distance_deg": float(st["distance_deg"]),
                        "azimuth_deg": float(st["azimuth_deg"]),
                        "residual_s": resid_val,
                        "picks": st["picks"],
                    },
                }
            )

    return _json_safe(
        {
            "type": "FeatureCollection",
            "features": features,
        }
    )


def write_geojson(
    filepath: str,
    traces: Sequence[Trace],
    params: AnalysisParamSet,
    include_stations: bool = True,
    include_ellipse: bool = True,
    focal_mechanism: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> None:
    """Write RFC 7946 GeoJSON FeatureCollection to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = _json_safe(
        export_geojson(
            traces=traces,
            params=params,
            include_stations=include_stations,
            include_ellipse=include_ellipse,
            focal_mechanism=focal_mechanism,
        )
    )
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, allow_nan=False)


def parse_geojson(data_or_path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and validate an RFC 7946 GeoJSON FeatureCollection from dict, JSON string, or file path."""
    if isinstance(data_or_path, dict):
        fc = data_or_path
    elif isinstance(data_or_path, str):
        s = data_or_path.lstrip()
        if s.startswith("{"):
            fc = json.loads(data_or_path)
        else:
            if not os.path.isfile(data_or_path):
                raise FileNotFoundError(f"GeoJSON file not found: {data_or_path}")
            with open(data_or_path, "r", encoding="utf-8") as f:
                fc = json.load(f)
    else:
        raise TypeError("GeoJSON input must be a dict, JSON string, or file path")

    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON root must be a FeatureCollection")
    features = fc.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection must contain a 'features' list")

    epicenter: Optional[Dict[str, Any]] = None
    confidence_ellipse: Optional[Dict[str, Any]] = None
    stations: List[Dict[str, Any]] = []

    for feat in features:
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            raise ValueError(f"Invalid GeoJSON Feature entry: {feat!r}")
        props = feat.get("properties") or {}
        ftype = props.get("feature_type", "")
        if ftype == "epicenter":
            epicenter = feat
        elif ftype == "confidence_ellipse":
            confidence_ellipse = feat
        elif ftype == "station":
            stations.append(feat)

    return {
        "feature_collection": fc,
        "features": features,
        "epicenter": epicenter,
        "confidence_ellipse": confidence_ellipse,
        "stations": stations,
    }
