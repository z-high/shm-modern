"""Earthquake location solvers: slowness-azimuth, S-P differential, depth phase, and Geiger/LocSAT."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import (
    ConfidenceEllipse,
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import QUALITY_TO_WEIGHT, AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.seismology.calibration import (
    SlownessCalibrationTable,
    _resolve_calib_table,
)
from shm_app.seismology.diagnostics import (
    compute_condition_number,
    compute_confidence_ellipse_2d,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_mad,
    compute_normality_stats,
    compute_reduced_chi2,
    compute_rmse,
    compute_robust_sigma,
    compute_wrmse,
)
from shm_app.seismology.traveltime import TravelTimeManager, get_ttt_manager, sphere_add, sphere_diff

DEG_TO_KM = 111.1949266

def locate_by_slowness_azimuth(
    station_code: str,
    phase_name: str,
    onset_epoch: float,
    slowness_s_deg: float,
    baz_deg: float,
    depth_km: float = 33.0,
    config: Optional[SHMConfig] = None,
    ttt: Optional[TravelTimeManager] = None,
    calib_table: Optional[SlownessCalibrationTable] = None,
    apply_calibration: Optional[bool] = None,
) -> AnalysisParamSet:
    """Invert observed array slowness & backazimuth for epicentral distance, (lat, lon), origin time,
    and Flinn-Engdahl region matching Locate -> By Slowness in cblib.c / earthloc.c.
    """
    cfg = config or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)
    stat = cfg.get_station(station_code)

    raw_slo = float(slowness_s_deg)
    raw_baz = float(baz_deg) % 360.0

    active_table = _resolve_calib_table(
        calib_table,
        apply_calibration,
        stations=[station_code],
        config=cfg,
    )

    eff_slo = raw_slo
    eff_baz = raw_baz
    res_meta: Optional[Dict[str, float]] = None
    comment_str = ""

    if active_table is not None:
        raw_dist = tm.distance_from_slowness(phase_name, raw_slo, depth_km)
        if raw_dist is None:
            raw_dist = max(1.0, (11.0 - raw_slo) * 10.0)
        raw_lat, raw_lon = sphere_add(stat.lat, stat.lon, raw_dist, raw_baz)

        cal_slo, cal_baz, d_slo, d_az = active_table.calibrate(raw_slo, raw_baz)
        eff_slo = cal_slo
        eff_baz = cal_baz
        comment_str = (
            f"Calibrated slo {raw_slo:.2f}->{cal_slo:.2f} s/deg, "
            f"baz {raw_baz:.1f}->{cal_baz:.1f} deg"
        )
    else:
        raw_dist = 0.0
        raw_lat, raw_lon = 0.0, 0.0

    dist_deg = tm.distance_from_slowness(phase_name, eff_slo, depth_km)
    if dist_deg is None:
        dist_deg = max(1.0, (11.0 - eff_slo) * 10.0)

    tt_sec = tm.travel_time(phase_name, dist_deg, depth_km) or (dist_deg * eff_slo)
    orig_epoch = onset_epoch - tt_sec
    lat, lon = sphere_add(stat.lat, stat.lon, dist_deg, eff_baz)
    igreg, grname, isreg, srname = cfg.fer.lookup(lat, lon)

    if active_table is not None:
        res_meta = {
            "calib_applied": 1.0,
            "raw_slowness": raw_slo,
            "raw_azimuth": raw_baz,
            "raw_distance": float(raw_dist),
            "raw_lat": float(raw_lat),
            "raw_lon": float(raw_lon),
            "cal_slowness": float(eff_slo),
            "cal_azimuth": float(eff_baz),
            "cal_distance": float(dist_deg),
            "cal_lat": float(lat),
            "cal_lon": float(lon),
        }

    return AnalysisParamSet(
        distance=dist_deg,
        azimuth=eff_baz,
        slowness=eff_slo,
        depth=depth_km,
        origin=format_sh_time(orig_epoch),
        lat=lat,
        lon=lon,
        phase=phase_name,
        region_id=igreg,
        region_name=grname,
        sregion_id=isreg,
        sregion_name=srname,
        ref_lat=stat.lat,
        ref_lon=stat.lon,
        source=station_code,
        comment=comment_str,
        residuals=res_meta,
    )


def locate_by_sp_diff(
    traces: Sequence[Trace],
    depth_km: float,
    ttt: Optional[TravelTimeManager] = None,
    config: Optional[SHMConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Compute epicentral distance and origin time from first visible trace with S and P picks."""
    cfg = config or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)
    for tr in traces:
        p_pk = tr.get_pick("P") or tr.get_pick("Pg") or tr.get_pick("Pn")
        s_pk = tr.get_pick("S") or tr.get_pick("Sg") or tr.get_pick("Sn")
        if p_pk and s_pk:
            dt_sp = abs(s_pk.onset_epoch - p_pk.onset_epoch)
            dist = tm.distance_from_diff_time(p_pk.name, s_pk.name, dt_sp, depth_km)
            if dist is not None:
                tt_p = tm.travel_time(p_pk.name, dist, depth_km) or 0.0
                return {
                    "station": tr.station,
                    "dt_sp": dt_sp,
                    "distance": dist,
                    "origin": format_sh_time(p_pk.onset_epoch - tt_p),
                }
    return None


def fit_depth_phase(
    traces: Sequence[Trace],
    distance_deg: float,
    ttt: Optional[TravelTimeManager] = None,
    config: Optional[SHMConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Invert P - pP/sP differential travel times across all available traces for weighted focal depth."""
    cfg = config or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)

    fits: List[Dict[str, Any]] = []

    for tr in traces:
        p_pk = tr.get_pick("P") or tr.get_pick("Pg") or tr.get_pick("Pn")
        if not p_pk or not p_pk.onset:
            continue

        # Determine distance for this station
        dist_i = tr.distance if tr.distance > 0 else distance_deg

        # Test both pP and sP if present on this trace
        for ph_name in ("pP", "sP"):
            dp_pk = tr.get_pick(ph_name)
            if dp_pk and dp_pk.onset:
                dt_dp = abs(dp_pk.onset_epoch - p_pk.onset_epoch)
                z_fit, z_err = tm.depth_from_depth_phase(p_pk.name, dp_pk.name, dt_dp, dist_i)
                if z_fit is not None:
                    sigma = max(float(z_err or 2.0), 0.5)
                    qual = max(0, min(10, int(getattr(dp_pk, "quality", 4))))
                    shm_w = float(QUALITY_TO_WEIGHT[qual])
                    weight = max(0.1, shm_w) / (sigma ** 2)
                    fits.append({
                        "station": tr.station,
                        "phase": ph_name,
                        "main_phase": p_pk.name,
                        "dt_dp": float(dt_dp),
                        "distance": float(dist_i),
                        "depth": float(z_fit),
                        "d_depth": float(sigma),
                        "weight": float(weight),
                    })

    if not fits:
        return None

    # Weighted mean focal depth
    total_w = sum(f["weight"] for f in fits)
    mean_z = sum(f["weight"] * f["depth"] for f in fits) / total_w

    for f in fits:
        f["residual"] = round(float(f["depth"] - mean_z), 3)

    n = len(fits)
    if n == 1:
        sigma_z = fits[0]["d_depth"]
        var_z = sigma_z ** 2
    else:
        # Weighted sample variance and standard error of weighted mean
        formal_var = 1.0 / total_w
        sample_var = sum(f["weight"] * ((f["depth"] - mean_z) ** 2) for f in fits) / total_w
        sem = math.sqrt(sample_var / (n - 1)) if n > 1 else fits[0]["d_depth"]
        sigma_z = max(math.sqrt(formal_var), sem)
        var_z = sigma_z ** 2

    stations_list = list(dict.fromkeys(f["station"] for f in fits))
    phases_list = list(dict.fromkeys(f["phase"] for f in fits))

    return {
        "depth": round(float(mean_z), 2),
        "std_depth": round(float(sigma_z), 2),
        "d_depth": round(float(sigma_z), 2),
        "count": n,
        "details": fits,
        "station": ", ".join(stations_list),
        "phase": "/".join(phases_list),
        "dt_dp": float(np.mean([f["dt_dp"] for f in fits])),
        "distance": float(np.mean([f["distance"] for f in fits])),
        "variance": round(float(var_z), 4),
        "stations": stations_list,
        "phases": phases_list,
        "individual_fits": fits,
    }


def locate_network_geiger(
    picks: List[PhasePick],
    depth_km: float = 33.0,
    free_depth: bool = False,
    initial_lat: Optional[float] = None,
    initial_lon: Optional[float] = None,
    initial_baz: Optional[float] = None,
    max_iter: int = 20,
    config: Optional[SHMConfig] = None,
    ttt: Optional[TravelTimeManager] = None,
    return_residuals: bool = False,
    excluded_stations: Optional[Sequence[str]] = None,
    weight_overrides: Optional[Dict[str, float]] = None,
    return_diagnostics: bool = False,
    traces: Optional[Sequence[Trace]] = None,
    robust: bool = False,
) -> Union[
    AnalysisParamSet,
    Tuple[AnalysisParamSet, Dict[str, float]],
    Tuple[AnalysisParamSet, Dict[str, float], FitDiagnosticResult],
]:
    """Locate an earthquake from multi-station arrival picks (and optional S-P or backazimuth constraints)
    using damped Gauss-Newton / Geiger iterative inversion matching earthloc.c & LocSAT.

    Supports station exclusions, custom weight overrides, covariance and error ellipse computation.
    """
    cfg = config or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)

    valid_picks = [p for p in picks if p.onset and p.station]
    if not valid_picks:
        raise ValueError("No valid phase picks provided for network location")

    # Build station coordinate fallback map from traces or pick metadata
    trace_coords: Dict[str, Tuple[float, float]] = {}
    if traces:
        for tr in traces:
            st_u = (tr.station or "").strip().upper()
            if not st_u or st_u in trace_coords:
                continue
            lat_v: Optional[float] = None
            lon_v: Optional[float] = None
            if "station_lat" in tr.extra_headers and "station_lon" in tr.extra_headers:
                try:
                    lat_v = float(tr.extra_headers["station_lat"])
                    lon_v = float(tr.extra_headers["station_lon"])
                except (TypeError, ValueError):
                    lat_v = lon_v = None
            elif tr.lat != 0.0 or tr.lon != 0.0:
                lat_v, lon_v = float(tr.lat), float(tr.lon)
            if (
                lat_v is not None
                and lon_v is not None
                and math.isfinite(lat_v)
                and math.isfinite(lon_v)
                and (abs(lat_v) > 1e-9 or abs(lon_v) > 1e-9)
            ):
                trace_coords[st_u] = (lat_v, lon_v)

    def _resolve_station_coords(station_code: str, pick: Optional[PhasePick] = None) -> Optional[Tuple[float, float]]:
        st_u = (station_code or "").strip().upper()
        st_inf = cfg.get_station(st_u)
        if abs(st_inf.lat) > 1e-9 or abs(st_inf.lon) > 1e-9:
            return float(st_inf.lat), float(st_inf.lon)
        if st_u in trace_coords:
            return trace_coords[st_u]
        if pick is not None:
            for lat_attr, lon_attr in (("station_lat", "station_lon"), ("lat", "lon")):
                la = getattr(pick, lat_attr, None)
                lo = getattr(pick, lon_attr, None)
                if la is not None and lo is not None:
                    try:
                        la_f, lo_f = float(la), float(lo)
                        if math.isfinite(la_f) and math.isfinite(lo_f) and (abs(la_f) > 1e-9 or abs(lo_f) > 1e-9):
                            return la_f, lo_f
                    except (TypeError, ValueError):
                        pass
            extra = getattr(pick, "extra_headers", None)
            if isinstance(extra, dict) and "station_lat" in extra and "station_lon" in extra:
                try:
                    la_f = float(extra["station_lat"])
                    lo_f = float(extra["station_lon"])
                    if math.isfinite(la_f) and math.isfinite(lo_f) and (abs(la_f) > 1e-9 or abs(lo_f) > 1e-9):
                        return la_f, lo_f
                except (TypeError, ValueError):
                    pass
        if st_u in cfg.stations:
            return float(st_inf.lat), float(st_inf.lon)
        return None

    excluded_set = set(str(s).upper() for s in (excluded_stations or []))
    weight_map = {str(k).upper(): float(v) for k, v in (weight_overrides or {}).items()}

    def _override_weight(p: PhasePick) -> Optional[float]:
        st = p.station.upper()
        ident = f"{st}.{p.comp.upper() if p.comp else 'Z'}.{p.name.upper()}"
        if ident in weight_map:
            return float(weight_map[ident])
        key_sp = f"{st}:{p.name.upper()}"
        if key_sp in weight_map:
            return float(weight_map[key_sp])
        if st in weight_map:
            return float(weight_map[st])
        return None

    def _is_excluded(p: PhasePick) -> bool:
        st = p.station.upper()
        if st in excluded_set:
            return True
        if f"{st}:{p.name.upper()}" in excluded_set or f"{st}:{p.name}" in excluded_set:
            return True
        ident = f"{st}.{p.comp.upper() if p.comp else 'Z'}.{p.name.upper()}"
        if ident in excluded_set:
            return True
        ov = _override_weight(p)
        if ov is not None:
            return ov <= 0.0
        return float(p.weight) <= 0.0

    def _get_pick_weight(p: PhasePick) -> float:
        ov = _override_weight(p)
        if ov is not None:
            return max(0.0, ov)
        if float(p.weight) <= 0.0:
            return 0.0
        return float(p.weight) / 4.0

    active_picks = [p for p in valid_picks if not _is_excluded(p)]
    active_ids = {id(p) for p in active_picks}

    # Group picks by station to check if we have S-P differential times at any station
    by_stat: Dict[str, Dict[str, PhasePick]] = {}
    for p in active_picks:
        by_stat.setdefault(p.station.upper(), {})[p.name] = p

    ref_pick = active_picks[0] if active_picks else valid_picks[0]
    first_coords = _resolve_station_coords(ref_pick.station, ref_pick)
    if first_coords is None:
        first_stat = cfg.get_station(ref_pick.station)
        first_lat, first_lon = float(first_stat.lat), float(first_stat.lon)
    else:
        first_lat, first_lon = first_coords

    # Initial guess
    z = max(0.0, float(depth_km))
    used_sp_or_slo = False
    lat = initial_lat if initial_lat is not None else first_lat + 2.0
    lon = initial_lon if initial_lon is not None else first_lon + 2.0

    # Improve initial guess if we have S-P or slowness + backazimuth
    if initial_lat is None or initial_lon is None:
        for st_code, ph_map in by_stat.items():
            st_coords = _resolve_station_coords(st_code)
            if st_coords is None:
                continue
            st_lat, st_lon = st_coords
            p_pk = ph_map.get("P") or ph_map.get("Pg") or ph_map.get("Pn")
            s_pk = ph_map.get("S") or ph_map.get("Sg") or ph_map.get("Sn")
            if p_pk and s_pk:
                dt_sp = abs(s_pk.onset_epoch - p_pk.onset_epoch)
                d_est = tm.distance_from_diff_time(p_pk.name, s_pk.name, dt_sp, z)
                baz = p_pk.beam_azim or initial_baz or 45.0
                if d_est is not None:
                    lat, lon = sphere_add(st_lat, st_lon, d_est, baz)
                    used_sp_or_slo = True
                    break
            elif p_pk and p_pk.slowness > 0.0:
                d_est = tm.distance_from_slowness(p_pk.name, p_pk.slowness, z)
                baz = p_pk.beam_azim or initial_baz or 45.0
                if d_est is not None:
                    lat, lon = sphere_add(st_lat, st_lon, d_est, baz)
                    used_sp_or_slo = True
                    break

    def _optimal_t0_and_sse(lat_cand: float, lon_cand: float, z_cand: float) -> Tuple[float, float]:
        diffs: List[float] = []
        ws: List[float] = []
        for pk in active_picks:
            st_c = _resolve_station_coords(pk.station, pk)
            if st_c is None:
                continue
            d_c, _, _ = sphere_diff(st_c[0], st_c[1], lat_cand, lon_cand)
            tt_c = tm.travel_time(pk.name, max(0.05, d_c), z_cand)
            if tt_c is None:
                continue
            w_c = _get_pick_weight(pk)
            if w_c <= 0.0:
                continue
            diffs.append(pk.onset_epoch - tt_c)
            ws.append(w_c)
        if not diffs:
            d0, _, _ = sphere_diff(first_lat, first_lon, lat_cand, lon_cand)
            tt0 = tm.travel_time(ref_pick.name, max(0.1, d0), z_cand) or (d0 * 10.0)
            return ref_pick.onset_epoch - tt0, float("inf")
        d_arr = np.asarray(diffs, dtype=np.float64)
        w_arr = np.asarray(ws, dtype=np.float64)
        t0_est = float(np.sum(w_arr * d_arr) / np.sum(w_arr))
        sse_est = float(np.sum(w_arr * ((d_arr - t0_est) ** 2)))
        return t0_est, sse_est

    pw_candidate: Optional[Tuple[float, float]] = None
    if not used_sp_or_slo and (initial_lat is None and initial_lon is None) and active_picks:
        candidates: List[Tuple[float, float]] = [(first_lat + 2.0, first_lon + 2.0)]
        earliest_pk = min(active_picks, key=lambda pk: pk.onset_epoch)
        e_coords = _resolve_station_coords(earliest_pk.station, earliest_pk)
        if e_coords is not None:
            candidates.append((max(-89.5, min(89.5, e_coords[0] + 0.1)), ((e_coords[1] + 0.1 + 180.0) % 360.0) - 180.0))
        st_coords_all = [c for c in (_resolve_station_coords(pk.station, pk) for pk in active_picks) if c is not None]
        if st_coords_all:
            c_lat = sum(c[0] for c in st_coords_all) / len(st_coords_all)
            s_lon = sum(math.sin(math.radians(c[1])) for c in st_coords_all)
            c_lon_v = sum(math.cos(math.radians(c[1])) for c in st_coords_all)
            c_lon = math.degrees(math.atan2(s_lon, c_lon_v)) if (s_lon != 0.0 or c_lon_v != 0.0) else 0.0
            candidates.append((max(-89.5, min(89.5, c_lat + 0.05)), ((c_lon + 0.05 + 180.0) % 360.0) - 180.0))

            # Optional plane-wave teleseismic start candidate (evaluated post-convergence
            # against the local start so local/regional events are never displaced).
            p_rows: List[Tuple[float, float, float, float]] = []
            seen_p_sta: set[str] = set()
            cos_c = max(0.05, math.cos(math.radians(c_lat)))
            for pk in active_picks:
                if pk.name.upper() not in ("P", "PN", "PG", "PKP"):
                    continue
                st_u = pk.station.upper()
                if st_u in seen_p_sta:
                    continue
                sc = _resolve_station_coords(pk.station, pk)
                if sc is None:
                    continue
                w_pk = _get_pick_weight(pk)
                if w_pk <= 0.0:
                    continue
                seen_p_sta.add(st_u)
                dlon = ((sc[1] - c_lon + 180.0) % 360.0) - 180.0
                x_e = dlon * 111.195 * cos_c
                y_n = (sc[0] - c_lat) * 111.195
                p_rows.append((x_e, y_n, float(pk.onset_epoch), w_pk))

            if len(p_rows) >= 4:
                t_ref = min(r[2] for r in p_rows)
                A_pw = np.array([[1.0, r[0], r[1]] for r in p_rows], dtype=np.float64)
                b_pw = np.array([r[2] - t_ref for r in p_rows], dtype=np.float64)
                w_arr_pw = np.array([r[3] for r in p_rows], dtype=np.float64)
                try:
                    def _fit_pw(A_sub: np.ndarray, b_sub: np.ndarray, w_sub: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
                        ws = np.sqrt(w_sub)
                        Aw_s = A_sub * ws[:, None]
                        bw_s = b_sub * ws
                        if np.linalg.matrix_rank(Aw_s) < 3:
                            return None
                        c_s, _, _, _ = np.linalg.lstsq(Aw_s, bw_s, rcond=None)
                        pred_s = A_sub @ c_s
                        bm = float(np.sum(w_sub * b_sub) / np.sum(w_sub))
                        sst = float(np.sum(w_sub * ((b_sub - bm) ** 2)))
                        ssr = float(np.sum(w_sub * ((b_sub - pred_s) ** 2)))
                        r2_s = (1.0 - ssr / sst) if sst > 1e-6 else 0.0
                        return c_s, r2_s

                    fit1 = _fit_pw(A_pw, b_pw, w_arr_pw)
                    if fit1 is not None:
                        coef, r2_pw = fit1
                        # Outlier-resistant refit when >= 5 stations are available (handles high-leverage edge outliers)
                        if len(p_rows) >= 5 and r2_pw < 0.95:
                            active_mask = np.ones(len(p_rows), dtype=bool)
                            max_drops = min(2, len(p_rows) - 4)
                            for _ in range(max_drops):
                                if r2_pw >= 0.98:
                                    break
                                res_all = np.abs(b_pw - (A_pw @ coef))
                                res_all[~active_mask] = -1.0
                                worst_i = int(np.argmax(res_all))
                                cand_mask = active_mask.copy()
                                cand_mask[worst_i] = False
                                if int(np.sum(cand_mask)) < 4:
                                    break
                                fit_drop = _fit_pw(A_pw[cand_mask], b_pw[cand_mask], w_arr_pw[cand_mask])
                                if fit_drop is not None and fit_drop[1] > r2_pw + 0.05:
                                    coef, r2_pw = fit_drop
                                    active_mask = cand_mask
                                else:
                                    break

                        s_e = -float(coef[1])
                        s_n = -float(coef[2])
                        s_deg = math.hypot(s_e, s_n) * 111.195
                        if r2_pw >= 0.85 and math.isfinite(s_deg) and 4.4 <= s_deg <= 12.5:
                            baz_pw = math.degrees(math.atan2(s_e, s_n)) % 360.0
                            d_pw = tm.distance_from_slowness("P", s_deg, z)
                            if d_pw is not None and math.isfinite(d_pw) and d_pw >= 15.0:
                                pw_candidate = sphere_add(c_lat, c_lon, d_pw, baz_pw)
                except np.linalg.LinAlgError:
                    pass

        best_cand = (float("inf"), lat, lon, ref_pick.onset_epoch)
        for c_la, c_lo in candidates:
            t0_c, sse_c = _optimal_t0_and_sse(c_la, c_lo, z)
            if sse_c < best_cand[0]:
                best_cand = (sse_c, c_la, c_lo, t0_c)
        if math.isfinite(best_cand[0]):
            lat, lon, t0 = best_cand[1], best_cand[2], best_cand[3]
        else:
            t0, _ = _optimal_t0_and_sse(lat, lon, z)
    else:
        t0, _ = _optimal_t0_and_sse(lat, lon, z)

    def _evaluate_system(
        t0_val: float,
        lat_val: float,
        lon_val: float,
        z_val: float,
        compute_jac: bool = True,
        irls_scale: Optional[Dict[int, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows_j: List[List[float]] = []
        res_vec: List[float] = []
        w_vec: List[float] = []
        cos_l = max(0.05, math.cos(math.radians(lat_val)))
        eps_lat = 0.05
        eps_lon = eps_lat / max(0.1, cos_l)

        for pk in active_picks:
            st_c = _resolve_station_coords(pk.station, pk)
            if st_c is None:
                continue
            st_la, st_lo = st_c
            dist_deg, _az_st_ev, _baz = sphere_diff(st_la, st_lo, lat_val, lon_val)
            tt = tm.travel_time(pk.name, max(0.05, dist_deg), z_val)
            if tt is None:
                continue

            res = pk.onset_epoch - (t0_val + tt)
            w_p = _get_pick_weight(pk)
            if irls_scale is not None:
                w_p *= float(irls_scale.get(id(pk), 1.0))
            if w_p <= 0.0:
                continue

            if compute_jac:
                d_lat_p, _, _ = sphere_diff(st_la, st_lo, min(89.95, lat_val + eps_lat), lon_val)
                tt_lat = tm.travel_time(pk.name, max(0.05, d_lat_p), z_val) or tt
                d_lon_p, _, _ = sphere_diff(st_la, st_lo, lat_val, lon_val + eps_lon)
                tt_lon = tm.travel_time(pk.name, max(0.05, d_lon_p), z_val) or tt

                dt_dlat = (tt_lat - tt) / eps_lat
                dt_dlon = (tt_lon - tt) / eps_lon
                if free_depth:
                    tt_z = tm.travel_time(pk.name, max(0.05, dist_deg), z_val + 2.0) or tt
                    dt_dz = (tt_z - tt) / 2.0
                    rows_j.append([1.0, dt_dlat, dt_dlon, dt_dz])
                else:
                    rows_j.append([1.0, dt_dlat, dt_dlon])
            res_vec.append(res)
            w_vec.append(w_p)

        j_mat = np.array(rows_j, dtype=np.float64) if rows_j else np.zeros((0, 4 if free_depth else 3), dtype=np.float64)
        r_arr = np.array(res_vec, dtype=np.float64)
        w_arr = np.array(w_vec, dtype=np.float64)
        return j_mat, r_arr, w_arr

    converged = False
    actual_iters = 0
    error_msg = ""
    final_J = np.zeros((0, 4 if free_depth else 3), dtype=np.float64)
    final_res = np.zeros(0, dtype=np.float64)
    final_weights = np.zeros(0, dtype=np.float64)

    if len(active_picks) < 3:
        error_msg = f"Insufficient active picks ({len(active_picks)} < 3) for location"
        logger.warning(error_msg)
    else:
        def _run_geiger(
            t0_s: float,
            lat_s: float,
            lon_s: float,
            z_s: float,
            irls_scale: Optional[Dict[int, float]] = None,
        ) -> Tuple[float, float, float, float, int, bool, Optional[str], np.ndarray, np.ndarray, np.ndarray]:
            lam = 1e-2
            iters_s = 0
            conv_s = False
            err_s: Optional[str] = None
            for it in range(max_iter):
                J, r, w_arr = _evaluate_system(t0_s, lat_s, lon_s, z_s, compute_jac=True, irls_scale=irls_scale)
                if r.size < 3:
                    if iters_s == 0:
                        err_s = f"Insufficient valid travel-time picks ({r.size} < 3) for location"
                    break

                iters_s = it + 1
                sse_curr = float(np.sum(w_arr * (r ** 2)))
                cos_lat_it = max(0.05, math.cos(math.radians(lat_s)))
                damp_diag = np.array(
                    [1.0, 1.0, cos_lat_it ** 2, 1.0] if free_depth else [1.0, 1.0, cos_lat_it ** 2],
                    dtype=np.float64,
                )

                W = np.diag(w_arr)
                JT_W = J.T @ W
                JT_W_J = JT_W @ J
                g = JT_W @ r

                accepted = False
                best_step = np.zeros(J.shape[1], dtype=np.float64)
                for lam_try in (lam, lam * 4.0, lam * 16.0, lam * 64.0, lam * 256.0):
                    H = JT_W_J + np.diag(damp_diag * lam_try)
                    try:
                        delta_m = np.linalg.solve(H, g)
                    except np.linalg.LinAlgError:
                        continue

                    step_deg = math.hypot(float(delta_m[1]), float(delta_m[2]) * cos_lat_it)
                    if step_deg > 5.0:
                        delta_m = delta_m * (5.0 / step_deg)
                    if free_depth and delta_m.size >= 4 and abs(float(delta_m[3])) > 50.0:
                        delta_m = delta_m * (50.0 / abs(float(delta_m[3])))

                    for alpha in (1.0, 0.5, 0.25):
                        dm_a = delta_m * alpha
                        t0_try = t0_s + float(dm_a[0])
                        lat_try = max(-89.9, min(89.9, lat_s + float(dm_a[1])))
                        lon_try = ((lon_s + float(dm_a[2]) + 180.0) % 360.0) - 180.0
                        z_try = max(0.0, min(700.0, z_s + float(dm_a[3]))) if (free_depth and dm_a.size >= 4) else z_s

                        _, r_try, w_try = _evaluate_system(
                            t0_try, lat_try, lon_try, z_try, compute_jac=False, irls_scale=irls_scale
                        )
                        if r_try.size >= 3:
                            sse_try = float(np.sum(w_try * (r_try ** 2)))
                            if sse_try <= sse_curr + 1e-12:
                                t0_s, lat_s, lon_s, z_s = t0_try, lat_try, lon_try, z_try
                                best_step = dm_a
                                lam = max(1e-5, lam_try * 0.5)
                                accepted = True
                                break
                    if accepted:
                        break

                if not accepted:
                    # No trial step reduced SSE (at local minimum or flat plateau)
                    conv_s = True
                    break

                if math.hypot(float(best_step[1]), float(best_step[2]) * cos_lat_it) < 1e-4 and abs(float(best_step[0])) < 1e-3:
                    conv_s = True
                    break

            fJ, fr, fw = _evaluate_system(t0_s, lat_s, lon_s, z_s, compute_jac=True, irls_scale=irls_scale)
            return t0_s, lat_s, lon_s, z_s, iters_s, conv_s, err_s, fJ, fr, fw

        def _compute_huber_scale(t0_c: float, lat_c: float, lon_c: float, z_c: float) -> Tuple[Dict[int, float], bool]:
            cur_res_by_id: Dict[int, float] = {}
            for pk in active_picks:
                st_c = _resolve_station_coords(pk.station, pk)
                if st_c is None:
                    continue
                d_i, _, _ = sphere_diff(st_c[0], st_c[1], lat_c, lon_c)
                tt_i = tm.travel_time(pk.name, max(0.05, d_i), z_c)
                if tt_i is not None and _get_pick_weight(pk) > 0.0:
                    cur_res_by_id[id(pk)] = float(pk.onset_epoch - (t0_c + tt_i))
            if len(cur_res_by_id) < 5:
                return {}, False
            r_vals = np.array(list(cur_res_by_id.values()), dtype=np.float64)
            med_r = float(np.median(r_vals))
            mad_r = float(np.median(np.abs(r_vals - med_r)))
            scale_r = max(0.35, 1.4826 * mad_r)
            huber_k = 1.5 * scale_r
            gross_k = max(3.5, 4.5 * scale_r)
            any_dw = False
            s_map: Dict[int, float] = {}
            for pid, rv in cur_res_by_id.items():
                ar = abs(rv - med_r)
                if ar > gross_k:
                    s_map[pid] = 0.01
                    any_dw = True
                elif ar > huber_k:
                    s_map[pid] = huber_k / ar
                    any_dw = True
                else:
                    s_map[pid] = 1.0
            return s_map, any_dw

        def _robust_loss(res_arr: np.ndarray, w_base: np.ndarray) -> float:
            if res_arr.size < 3:
                return float("inf")
            if not robust:
                return float(np.sum(w_base * (res_arr ** 2)))
            # Huber/capped loss (delta=2.0s) so one gross outlier doesn't mask a cleaner basin
            delta = 2.0
            ar = np.abs(res_arr)
            hub = np.where(ar <= delta, 0.5 * (ar ** 2), delta * (ar - 0.5 * delta))
            return float(np.sum(w_base * hub))

        def _refine_robust(
            sol_in: Tuple[float, float, float, float, int, bool, Optional[str], np.ndarray, np.ndarray, np.ndarray]
        ) -> Tuple[float, float, float, float, int, bool, Optional[str], np.ndarray, np.ndarray, np.ndarray]:
            if not robust or sol_in[8].size < 5:
                return sol_in
            sol = sol_in
            for _ in range(3):
                s_map, any_dw = _compute_huber_scale(sol[0], sol[1], sol[2], sol[3])
                if not any_dw:
                    break
                sol = _run_geiger(sol[0], sol[1], sol[2], sol[3], irls_scale=s_map)
            return sol

        z_init = z
        loc_sol = _run_geiger(t0, lat, lon, z_init)
        if robust:
            loc_sol = _refine_robust(loc_sol)

        best_sol = loc_sol
        if pw_candidate is not None:
            pw_lat, pw_lon = pw_candidate
            t0_pw, _ = _optimal_t0_and_sse(pw_lat, pw_lon, z_init)
            # In robust mode, pre-downweight gross outliers at the MAD-trimmed plane-wave start
            # so a single bad station cannot pull Geiger out of the teleseismic basin on iteration 1.
            pw_init_scale: Optional[Dict[int, float]] = None
            if robust:
                s0, dw0 = _compute_huber_scale(t0_pw, pw_lat, pw_lon, z_init)
                if dw0:
                    pw_init_scale = s0
            pw_sol = _run_geiger(t0_pw, pw_lat, pw_lon, z_init, irls_scale=pw_init_scale)
            if robust:
                pw_sol = _refine_robust(pw_sol)
            # Compare unweighted/base-weight loss so downweighting doesn't artificially bias the comparison
            _, r_loc_raw, w_loc_raw = _evaluate_system(best_sol[0], best_sol[1], best_sol[2], best_sol[3], compute_jac=False)
            _, r_pw_raw, w_pw_raw = _evaluate_system(pw_sol[0], pw_sol[1], pw_sol[2], pw_sol[3], compute_jac=False)
            if _robust_loss(r_pw_raw, w_pw_raw) + 1e-6 < _robust_loss(r_loc_raw, w_loc_raw):
                best_sol = pw_sol

        t0, lat, lon, z, actual_iters, converged, error_msg, final_J, final_res, final_weights = best_sol

    # Calculate residuals for ALL picks (active + excluded)
    station_residuals: Dict[str, float] = {}
    residual_entries: List[ResidualEntry] = []
    active_resids: List[float] = []

    for p in active_picks:
        st_c = _resolve_station_coords(p.station, p)
        if st_c is None:
            continue
        dist_deg, _, _ = sphere_diff(st_c[0], st_c[1], lat, lon)
        tt = tm.travel_time(p.name, max(0.05, dist_deg), z)
        if tt is not None:
            active_resids.append(float(p.onset_epoch - (t0 + tt)))

    sigma_hat = compute_robust_sigma(active_resids) if active_resids else 0.5
    outlier_thresh = max(0.3, 3.0 * sigma_hat)

    for p in valid_picks:
        st_c = _resolve_station_coords(p.station, p)
        if st_c is None:
            continue
        dist_deg, _az, baz = sphere_diff(st_c[0], st_c[1], lat, lon)
        tt = tm.travel_time(p.name, max(0.05, dist_deg), z)
        if tt is not None:
            calc_t = t0 + tt
            res = float(p.onset_epoch - calc_t)
            p.resid = res
            station_residuals[f"{p.station}:{p.name}"] = res

            is_active = id(p) in active_ids
            w_val = _get_pick_weight(p) if is_active else 0.0
            is_outlier = abs(res) > outlier_thresh

            entry = ResidualEntry(
                identifier=f"{p.station}.{p.comp or 'Z'}.{p.name}",
                observed=float(p.onset_epoch),
                calculated=float(calc_t),
                residual=res,
                normalized_residual=float(res / max(0.05, sigma_hat)),
                weight=w_val if is_active else 0.0,
                uncertainty=max(0.05, sigma_hat),
                distance_deg=round(dist_deg, 3),
                azimuth_deg=round(baz, 1),
                station=p.station,
                phase=p.name,
                channel=p.comp or "Z",
                is_outlier=is_outlier,
            )
            residual_entries.append(entry)

    # Statistical diagnostics
    n_params = 4 if free_depth else 3
    dof = max(1, int(final_res.size) - n_params)
    rms = compute_rmse(active_resids)
    wrms = compute_wrmse(final_res, final_weights) if final_res.size > 0 else rms
    mad_val = compute_mad(active_resids)
    chi2_red = compute_reduced_chi2(active_resids, dof=dof)
    kappa = compute_condition_number(final_J) if final_J.size > 0 else 1.0

    s_sq = (float(np.sum(final_weights * (final_res ** 2))) / dof) if final_res.size > 0 else (rms ** 2)
    cov_m = compute_covariance_matrix(final_J, final_weights, s_sq=s_sq) if final_J.size > 0 else np.eye(n_params) * (rms ** 2)
    corr_m = compute_correlation_matrix(cov_m)

    cos_lat = max(1e-3, math.cos(math.radians(lat)))
    if cov_m.shape[0] >= 3 and final_J.size > 0:
        scale_2d = np.diag([1.0, cos_lat])
        cov_epi = scale_2d @ cov_m[1:3, 1:3] @ scale_2d
        semi_major, semi_minor, strike_deg = compute_confidence_ellipse_2d(
            cov_epi, dof=dof, deg_to_km=DEG_TO_KM
        )
        d_lat = float(np.sqrt(max(0.0, cov_m[1, 1])) * DEG_TO_KM)
        d_lon = float(np.sqrt(max(0.0, cov_m[2, 2])) * DEG_TO_KM * cos_lat)
        lon_se_deg = float(np.sqrt(max(0.0, cov_m[2, 2])))
        d_t0 = float(np.sqrt(max(0.0, cov_m[0, 0])))
        d_z = float(np.sqrt(max(0.0, cov_m[3, 3]))) if free_depth and cov_m.shape[0] >= 4 else 0.0
    else:
        semi_major, semi_minor, strike_deg = (rms * 5.0, rms * 5.0, 0.0)
        d_lat, d_lon, d_t0, d_z = (rms * 5.0, rms * 5.0, rms, 0.0)
        lon_se_deg = d_lon / (DEG_TO_KM * cos_lat)

    conf_ellipse = ConfidenceEllipse(
        semi_major_km=round(semi_major, 3),
        semi_minor_km=round(semi_minor, 3),
        strike_deg=round(strike_deg, 1),
        center_x=round(lon, 4),
        center_y=round(lat, 4),
    )

    outlier_count = sum(1 for e in residual_entries if e.is_outlier)
    norm_stats = compute_normality_stats(active_resids)

    lat_se_deg = d_lat / DEG_TO_KM
    param_dict = {
        "origin_time": ParameterUncertainty(
            name="origin_time",
            value=t0,
            standard_error=d_t0,
            confidence_interval_95=(t0 - 1.96 * d_t0, t0 + 1.96 * d_t0),
            unit="s",
        ),
        "latitude": ParameterUncertainty(
            name="latitude",
            value=lat,
            standard_error=lat_se_deg,
            confidence_interval_95=(lat - 1.96 * lat_se_deg, lat + 1.96 * lat_se_deg),
            unit="deg",
        ),
        "longitude": ParameterUncertainty(
            name="longitude",
            value=lon,
            standard_error=lon_se_deg,
            confidence_interval_95=(lon - 1.96 * lon_se_deg, lon + 1.96 * lon_se_deg),
            unit="deg",
        ),
    }
    if free_depth:
        param_dict["depth"] = ParameterUncertainty(
            name="depth",
            value=z,
            standard_error=d_z,
            confidence_interval_95=(max(0.0, z - 1.96 * d_z), z + 1.96 * d_z),
            unit="km",
        )

    diag_title = (
        f"Multi-Station Geiger Inversion — {error_msg}"
        if error_msg
        else "Multi-Station Geiger Inversion"
    )
    diag_ctx: Dict[str, Any] = {
        "picks": picks,
        "depth_km": depth_km,
        "free_depth": free_depth,
        "t0": t0,
        "lat": lat,
        "lon": lon,
        "depth": z,
        "status": "insufficient_picks" if error_msg else ("converged" if converged else "max_iter"),
    }
    if error_msg:
        diag_ctx["error"] = error_msg

    diag_result = FitDiagnosticResult(
        fit_type=FitType.HYPOCENTER_GEIGER,
        title=diag_title,
        converged=converged,
        iterations=actual_iters,
        dof=dof,
        rmse=round(rms, 4),
        wrmse=round(wrms, 4),
        mad=round(mad_val, 4),
        chi2_reduced=round(chi2_red, 4),
        condition_number=round(kappa, 2),
        parameters=param_dict,
        covariance_matrix=cov_m,
        correlation_matrix=corr_m,
        confidence_ellipse=conf_ellipse,
        residuals=residual_entries,
        outlier_count=outlier_count,
        normality_p_value=norm_stats.get("p_value", 1.0),
        skewness=norm_stats.get("skewness", 0.0),
        kurtosis=norm_stats.get("kurtosis", 0.0),
        context_data=diag_ctx,
    )

    ref_dist, ref_baz, _az_ev_to_ref = sphere_diff(first_lat, first_lon, lat, lon)
    ref_slo = tm.slowness(ref_pick.name, ref_dist, z) or 0.0
    igreg, grname, isreg, srname = cfg.fer.lookup(lat, lon)
    comment_str = (
        f"Error: {error_msg} ({len(active_picks)}/{len(valid_picks)} picks)"
        if error_msg
        else f"Geiger fit RMS={rms:.2f} s ({len(active_picks)}/{len(valid_picks)} picks)"
    )

    param = AnalysisParamSet(
        distance=ref_dist,
        azimuth=ref_baz,
        slowness=ref_slo,
        depth=z,
        d_depth=round(d_z, 2) if free_depth else 0.0,
        depth_type=3 if free_depth else 1,
        origin=format_sh_time(t0),
        lat=lat,
        d_lat=round(d_lat, 3),
        lon=lon,
        d_lon=round(d_lon, 3),
        phase=ref_pick.name,
        region_id=igreg,
        region_name=grname,
        sregion_id=isreg,
        sregion_name=srname,
        ref_lat=first_lat,
        ref_lon=first_lon,
        source="LOCSAT/GEIGER",
        comment=comment_str,
        residuals=station_residuals,
        error_ellipse=(semi_major, semi_minor, strike_deg),
        condition_number=round(kappa, 2),
        chi2_reduced=round(chi2_red, 4),
        rmse=round(rms, 4),
        wrmse=round(wrms, 4),
    )

    if return_diagnostics:
        return param, station_residuals, diag_result
    if return_residuals:
        return param, station_residuals
    return param

