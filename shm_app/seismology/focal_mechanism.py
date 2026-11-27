"""Pure Python/NumPy Double-Couple Focal Mechanism Solver & Stereonet Projection (M-25).

Implements:
- Aki & Richards (1980) double-couple P-wave radiation pattern R_theor(strike, dip, rake; ih, az).
- Coarse 3D grid search + local fine refinement over (strike, dip, rake) minimizing weighted
  first-motion polarity misfit with nodal-margin tie breaking.
- Auxiliary nodal plane (strike2, dip2, rake2) and P/T/B principal axes (trend, plunge).
- Equal-area (Schmidt) lower-hemisphere projection and beachball contour/nodal-curve helpers.
- First-motion polarity & takeoff-angle extraction from Workspace Trace/PhasePick objects.

NO PyQt6, scipy, or obspy imports — pure NumPy + Python stdlib only.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import PhasePick, Trace
from shm_app.seismology.traveltime import TravelTimeManager, get_ttt_manager, sphere_diff

logger = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0


def _source_vp_kms(depth_km: float) -> float:
    """Approximate IASP91-like P-wave velocity (km/s) at source depth."""
    z = max(0.0, float(depth_km))
    if z < 20.0:
        return 5.8
    if z < 35.0:
        return 6.5
    if z < 210.0:
        return 8.04
    if z < 410.0:
        return 8.50
    if z < 660.0:
        return 9.80
    return 10.75


def fault_normal_and_slip(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit fault-normal n and slip d vectors in (North, East, Down) coordinates
    following Aki & Richards (1980), Box 4.4 / Eq. 4.88.
    """
    phi = math.radians(float(strike_deg))
    delta = math.radians(float(dip_deg))
    lam = math.radians(float(rake_deg))

    sp, cp = math.sin(phi), math.cos(phi)
    sd, cd = math.sin(delta), math.cos(delta)
    sl, cl = math.sin(lam), math.cos(lam)

    n = np.array([-sd * sp, sd * cp, -cd], dtype=np.float64)
    d = np.array(
        [
            cl * cp + cd * sl * sp,
            cl * sp - cd * sl * cp,
            -sd * sl,
        ],
        dtype=np.float64,
    )
    return n, d


def _fault_normal_and_slip_batch(
    strikes_deg: np.ndarray,
    dips_deg: np.ndarray,
    rakes_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized fault normal N (G, 3) and slip D (G, 3) for G focal mechanisms."""
    phi = np.radians(strikes_deg)
    delta = np.radians(dips_deg)
    lam = np.radians(rakes_deg)

    sp, cp = np.sin(phi), np.cos(phi)
    sd, cd = np.sin(delta), np.cos(delta)
    sl, cl = np.sin(lam), np.cos(lam)

    n_mat = np.column_stack([-sd * sp, sd * cp, -cd])
    d_mat = np.column_stack(
        [
            cl * cp + cd * sl * sp,
            cl * sp - cd * sl * cp,
            -sd * sl,
        ]
    )
    return n_mat, d_mat


def p_wave_radiation_pattern(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
    takeoff_deg: Union[float, np.ndarray],
    azimuth_deg: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute Aki & Richards (1980) P-wave radiation pattern R_theor for a double-couple source.

    Parameters
    ----------
    strike_deg : float
        Fault strike in [0, 360) deg clockwise from North.
    dip_deg : float
        Fault dip in [0, 90] deg from horizontal.
    rake_deg : float
        Slip rake in [-180, 180] deg.
    takeoff_deg : float or np.ndarray
        Ray takeoff angle i_h in degrees measured from downward vertical (0 = straight down, 90 = horizontal).
    azimuth_deg : float or np.ndarray
        Station azimuth from source in degrees clockwise from North.

    Returns
    -------
    float or np.ndarray
        Dimensionless P-wave radiation amplitude in [-1, 1]: R = 2 (gamma . n)(gamma . d).
    """
    n_vec, d_vec = fault_normal_and_slip(strike_deg, dip_deg, rake_deg)
    ih = np.radians(np.asarray(takeoff_deg, dtype=np.float64))
    az = np.radians(np.asarray(azimuth_deg, dtype=np.float64))

    sih = np.sin(ih)
    gx = sih * np.cos(az)
    gy = sih * np.sin(az)
    gz = np.cos(ih)

    gn = gx * n_vec[0] + gy * n_vec[1] + gz * n_vec[2]
    gd = gx * d_vec[0] + gy * d_vec[1] + gz * d_vec[2]
    rad = 2.0 * gn * gd

    if np.ndim(takeoff_deg) == 0 and np.ndim(azimuth_deg) == 0:
        return float(rad)
    return rad


def _norm_rake(rake_deg: float) -> float:
    """Wrap rake into [-180, 180] degrees."""
    r = ((float(rake_deg) + 180.0) % 360.0) - 180.0
    if r == -180.0 and float(rake_deg) > 0.0:
        return 180.0
    return r


def _plane_from_normal_and_slip(n_vec: np.ndarray, d_vec: np.ndarray) -> Tuple[float, float, float]:
    """Recover (strike, dip, rake) with dip in [0, 90] from normal and slip vectors in NED."""
    n = np.asarray(n_vec, dtype=np.float64)
    d = np.asarray(d_vec, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    d_norm = np.linalg.norm(d)
    if n_norm > 0:
        n = n / n_norm
    if d_norm > 0:
        d = d / d_norm

    # Enforce upward-pointing normal (n_D <= 0) so dip is in [0, 90]
    if n[2] > 0.0:
        n = -n
        d = -d

    dip_rad = math.acos(max(-1.0, min(1.0, -float(n[2]))))
    dip_deg = math.degrees(dip_rad)

    if math.sin(dip_rad) < 1e-8:
        # Horizontal plane: choose strike along slip vector
        strike_deg = math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360.0
        rake_deg = 0.0
    else:
        strike_rad = math.atan2(-float(n[0]), float(n[1]))
        strike_deg = math.degrees(strike_rad) % 360.0
        s_vec = np.array([math.cos(strike_rad), math.sin(strike_rad), 0.0], dtype=np.float64)
        u_vec = np.cross(n, s_vec)  # up-dip vector in plane
        cos_r = float(np.dot(d, s_vec))
        sin_r = float(np.dot(d, u_vec))
        rake_deg = _norm_rake(math.degrees(math.atan2(sin_r, cos_r)))

    return float(strike_deg % 360.0), float(dip_deg), float(rake_deg)


def auxiliary_nodal_plane(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
) -> Tuple[float, float, float]:
    """Compute the conjugate auxiliary nodal plane (strike2, dip2, rake2)."""
    n1, d1 = fault_normal_and_slip(strike_deg, dip_deg, rake_deg)
    return _plane_from_normal_and_slip(d1, n1)


def _vec_to_trend_plunge(vec: np.ndarray) -> Tuple[float, float]:
    """Convert a 3D NED unit vector to lower-hemisphere (trend_deg, plunge_deg)."""
    v = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-15:
        return 0.0, 0.0
    v = v / norm
    if v[2] < 0.0:
        v = -v
    plunge = math.degrees(math.asin(max(0.0, min(1.0, float(v[2])))))
    horiz = math.hypot(float(v[0]), float(v[1]))
    if horiz < 1e-10:
        trend = 0.0
    else:
        trend = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 360.0
    return float(trend), float(plunge)


def principal_axes(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
) -> Dict[str, Tuple[float, float]]:
    """Compute lower-hemisphere (trend_deg, plunge_deg) for P, T, and B axes."""
    n, d = fault_normal_and_slip(strike_deg, dip_deg, rake_deg)
    t_vec = (n + d) / math.sqrt(2.0)
    p_vec = (n - d) / math.sqrt(2.0)
    b_vec = np.cross(n, d)
    return {
        "P": _vec_to_trend_plunge(p_vec),
        "T": _vec_to_trend_plunge(t_vec),
        "B": _vec_to_trend_plunge(b_vec),
    }


def project_schmidt(trend_deg: float, plunge_deg: float) -> Tuple[float, float]:
    """Equal-area (Schmidt) lower-hemisphere projection onto unit circle (x=East, y=North).

    Parameters
    ----------
    trend_deg : float
        Azimuth clockwise from North in degrees.
    plunge_deg : float
        Plunge downward from horizontal in degrees [0, 90]. Negative values are projected
        via their lower-hemisphere antipode so x^2 + y^2 <= 1 always holds.
    """
    tr = float(trend_deg)
    pl = float(plunge_deg)
    if pl < 0.0:
        pl = -pl
        tr = tr + 180.0
    pl = max(0.0, min(90.0, pl))
    ih_half = math.radians(0.5 * (90.0 - pl))
    r = math.sqrt(2.0) * math.sin(ih_half)
    r = min(1.0, max(0.0, r))
    az = math.radians(tr % 360.0)
    x = r * math.sin(az)
    y = r * math.cos(az)
    return float(x), float(y)


def _nodal_plane_curve_xy(
    strike_deg: float,
    dip_deg: float,
    n_pts: int = 181,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample lower-hemisphere Schmidt projection curve of a nodal plane (strike, dip)."""
    phi = math.radians(float(strike_deg))
    delta = math.radians(float(dip_deg))

    s_vec = np.array([math.cos(phi), math.sin(phi), 0.0], dtype=np.float64)
    u_down = np.array(
        [-math.cos(delta) * math.sin(phi), math.cos(delta) * math.cos(phi), math.sin(delta)],
        dtype=np.float64,
    )

    alphas = np.linspace(0.0, math.pi, max(11, int(n_pts)))
    xs = np.empty(alphas.size, dtype=np.float64)
    ys = np.empty(alphas.size, dtype=np.float64)
    for idx, a in enumerate(alphas):
        vec = math.cos(a) * s_vec + math.sin(a) * u_down
        tr, pl = _vec_to_trend_plunge(vec)
        xs[idx], ys[idx] = project_schmidt(tr, pl)
    return xs, ys


def compute_beachball_contours(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
    n_grid: int = 121,
) -> Dict[str, Any]:
    """Compute Schmidt lower-hemisphere beachball sign grid and nodal plane curves for Matplotlib.

    Returns
    -------
    dict with keys:
        x_grid, y_grid : 1D coordinate arrays in [-1, 1]
        rad_grid : (n_grid, n_grid) P-wave radiation amplitude (NaN outside unit disk)
        sign_grid : (n_grid, n_grid) +1 (compressional) / -1 (dilatational) (NaN outside unit disk)
        mask : boolean (n_grid, n_grid) inside unit disk (x^2 + y^2 <= 1)
        plane1_x, plane1_y : 1D nodal curve arrays for primary plane
        plane2_x, plane2_y : 1D nodal curve arrays for auxiliary plane
        plane1 : (strike, dip, rake)
        plane2 : (strike2, dip2, rake2)
        axes : {'P': (trend, plunge), 'T': (trend, plunge), 'B': (trend, plunge)}
        p_xy, t_xy, b_xy : projected (x, y) coordinates of P, T, B axes
    """
    n_pts = max(21, int(n_grid))
    axis_vals = np.linspace(-1.0, 1.0, n_pts, dtype=np.float64)
    xx, yy = np.meshgrid(axis_vals, axis_vals)
    r2 = xx ** 2 + yy ** 2
    inside = r2 <= 1.0 + 1e-12
    r2_safe = np.where(inside, np.minimum(1.0, r2), 0.0)

    # Exact inverse Schmidt projection to lower-hemisphere unit ray gamma (N, E, D)
    factor = np.sqrt(np.maximum(0.0, 2.0 - r2_safe))
    gn_arr = yy * factor  # North
    ge_arr = xx * factor  # East
    gd_arr = 1.0 - r2_safe  # Down

    n_vec, d_vec = fault_normal_and_slip(strike_deg, dip_deg, rake_deg)
    dot_n = gn_arr * n_vec[0] + ge_arr * n_vec[1] + gd_arr * n_vec[2]
    dot_d = gn_arr * d_vec[0] + ge_arr * d_vec[1] + gd_arr * d_vec[2]
    rad_raw = 2.0 * dot_n * dot_d

    rad_grid = np.where(inside, rad_raw, np.nan)
    sign_raw = np.where(rad_raw >= 0.0, 1.0, -1.0)
    sign_grid = np.where(inside, sign_raw, np.nan)

    s2, d2, r2_ang = auxiliary_nodal_plane(strike_deg, dip_deg, rake_deg)
    p1x, p1y = _nodal_plane_curve_xy(strike_deg, dip_deg)
    p2x, p2y = _nodal_plane_curve_xy(s2, d2)

    axes = principal_axes(strike_deg, dip_deg, rake_deg)
    p_xy = project_schmidt(*axes["P"])
    t_xy = project_schmidt(*axes["T"])
    b_xy = project_schmidt(*axes["B"])

    return {
        "x_grid": axis_vals,
        "y_grid": axis_vals,
        "rad_grid": rad_grid,
        "sign_grid": sign_grid,
        "mask": inside,
        "plane1_x": p1x,
        "plane1_y": p1y,
        "plane2_x": p2x,
        "plane2_y": p2y,
        "plane1": (float(strike_deg % 360.0), float(dip_deg), _norm_rake(rake_deg)),
        "plane2": (s2, d2, r2_ang),
        "axes": axes,
        "p_xy": p_xy,
        "t_xy": t_xy,
        "b_xy": b_xy,
    }


def estimate_takeoff_angle(
    dist_deg: float,
    depth_km: float = 10.0,
    phase: str = "P",
    ttt: Optional[TravelTimeManager] = None,
) -> float:
    """Estimate P-wave takeoff angle i_h in degrees (0 = straight down, 90 = horizontal, >90 upgoing)."""
    dist = max(0.0, float(dist_deg))
    z = max(0.0, float(depth_km))
    p_up = (phase or "P").strip().upper()
    if dist < 1e-6:
        return 180.0 if (z > 0.0 and p_up in ("PG", "SG")) else 0.0

    # Direct crustal Pg/Sg rays from a subsurface focus travel upward to surface stations
    if p_up in ("PG", "SG") and dist < 3.0:
        if z > 0.0:
            dist_km = dist * 111.1949266
            return float(90.0 + math.degrees(math.atan2(z, max(1e-3, dist_km))))
        return 90.0

    tm = ttt if ttt is not None else get_ttt_manager()
    slo = tm.slowness(phase, dist, z)
    if slo is None and phase != "P":
        slo = tm.slowness("P", dist, z)

    vp_src = _source_vp_kms(z)
    if p_up.startswith("S"):
        vp_src = vp_src / 1.7320508
    r_src = max(100.0, EARTH_RADIUS_KM - z)

    if slo is not None and slo > 0.0:
        # sin(i_h) = p_{s/deg} * (180/pi) * v_p(z) / (R_E - z)
        sin_ih = (float(slo) * (180.0 / math.pi) * vp_src) / r_src
        if sin_ih <= 1.0:
            ih_deg = math.degrees(math.asin(max(0.0, sin_ih)))
            return float(ih_deg)
        # Super-critical / local direct crustal ray: upgoing ray if short distance
        if dist < 1.5 and z > 0.0:
            dist_km = dist * 111.1949266
            return float(90.0 + math.degrees(math.atan2(z, max(1e-3, dist_km))))
        return 90.0

    # Geometric fallback if travel-time table is outside bounds
    dist_km = dist * 111.1949266
    if dist < 1.5 and z > 0.0:
        return float(90.0 + math.degrees(math.atan2(z, max(1e-3, dist_km))))
    return float(max(5.0, min(85.0, 90.0 - dist * 0.8)))


def _select_polarity_pick(tr: Trace) -> Optional[PhasePick]:
    """Return the first P-phase (or non-zero polarity) PhasePick on trace."""
    preferred_names = ("P", "Pg", "Pn", "Pb", "Pdiff", "PKP", "PKPdf")
    for pname in preferred_names:
        for pk in tr.picks:
            if pk.name == pname and int(pk.sign) != 0:
                return pk
    for pk in tr.picks:
        if int(pk.sign) != 0 and pk.name.upper().startswith("P"):
            return pk
    for pk in tr.picks:
        if int(pk.sign) != 0:
            return pk
    return None


def extract_polarities_from_traces(
    traces: Sequence[Trace],
    event_lat: float,
    event_lon: float,
    depth_km: float = 10.0,
    config: Optional[SHMConfig] = None,
    ttt: Optional[TravelTimeManager] = None,
) -> List[Dict[str, Any]]:
    """Extract first-motion polarities and ray takeoff/azimuth angles from Traces.

    Uses PhasePick.sign (+1 compressional, -1 dilatational), station coordinates from
    SHMConfig (or trace extra_headers 'station_lat'/'station_lon'), and travel-time slowness
    for takeoff angle estimation.
    """
    cfg = config if config is not None else get_config()
    tm = ttt if ttt is not None else get_ttt_manager()

    observations: List[Dict[str, Any]] = []
    for tr in traces:
        pk = _select_polarity_pick(tr)
        if pk is None or int(pk.sign) == 0:
            continue

        st_inf = cfg.get_station(tr.station)
        st_lat = float(st_inf.lat)
        st_lon = float(st_inf.lon)
        if st_lat == 0.0 and st_lon == 0.0:
            if "station_lat" in tr.extra_headers and "station_lon" in tr.extra_headers:
                st_lat = float(tr.extra_headers["station_lat"])
                st_lon = float(tr.extra_headers["station_lon"])
            elif tr.lat != 0.0 or tr.lon != 0.0:
                st_lat = float(tr.lat)
                st_lon = float(tr.lon)

        if not (math.isfinite(st_lat) and math.isfinite(st_lon)) or (abs(st_lat) < 1e-9 and abs(st_lon) < 1e-9):
            sta_lbl = tr.station or pk.station or "<unknown>"
            msg = f"Skipping station {sta_lbl!r}: unresolved (0, 0) or non-finite coordinates"
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            continue

        dist_deg, az_ev_to_sta, _ = sphere_diff(float(event_lat), float(event_lon), st_lat, st_lon)
        takeoff_deg = estimate_takeoff_angle(dist_deg, depth_km=depth_km, phase=pk.name, ttt=tm)
        pol = 1 if int(pk.sign) > 0 else -1
        w = float(pk.weight) if pk.weight > 0 else 1.0

        observations.append(
            {
                "station": tr.station or pk.station,
                "phase": pk.name,
                "azimuth": float(az_ev_to_sta),
                "takeoff": float(takeoff_deg),
                "polarity": pol,
                "weight": w,
                "distance_deg": float(dist_deg),
            }
        )
    return observations


def _parse_polarity_observations(
    polarities: Union[Sequence[Tuple[Any, ...]], Sequence[Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Normalize polarity observations into (az_deg, takeoff_deg, pol, weights, labels)."""
    az_list: List[float] = []
    ih_list: List[float] = []
    pol_list: List[float] = []
    w_list: List[float] = []
    labels: List[str] = []

    for idx, item in enumerate(polarities):
        if isinstance(item, dict):
            az_val = item.get("azimuth", item.get("azimuth_deg", item.get("az")))
            ih_val = item.get("takeoff", item.get("takeoff_angle_deg", item.get("takeoff_deg", item.get("ih"))))
            raw_pol = item.get("polarity", item.get("pol", item.get("sign", 0)))
            w_val = float(item.get("weight", 1.0))
            lbl = str(item.get("station", f"STA{idx + 1}"))
        elif isinstance(item, (tuple, list)):
            if len(item) < 3:
                raise ValueError(f"Polarity tuple must have at least 3 elements (az, takeoff, pol), got {item}")
            az_val, ih_val, raw_pol = item[0], item[1], item[2]
            w_val = float(item[3]) if len(item) >= 4 else 1.0
            lbl = str(item[4]) if len(item) >= 5 else f"STA{idx + 1}"
        else:
            raise TypeError(f"Unsupported polarity observation type: {type(item)}")

        if az_val is None or ih_val is None:
            raise ValueError(f"Missing azimuth or takeoff angle in observation {item}")

        if isinstance(raw_pol, str):
            pol_int = PhasePick.sign_from_char(raw_pol)
        else:
            pol_int = 1 if float(raw_pol) > 0 else (-1 if float(raw_pol) < 0 else 0)

        if pol_int == 0 or w_val <= 0.0:
            continue

        az_list.append(float(az_val) % 360.0)
        ih_list.append(float(ih_val))
        pol_list.append(float(pol_int))
        w_list.append(w_val)
        labels.append(lbl)

    if not pol_list:
        raise ValueError("At least one valid non-zero polarity observation is required")

    return (
        np.asarray(az_list, dtype=np.float64),
        np.asarray(ih_list, dtype=np.float64),
        np.asarray(pol_list, dtype=np.float64),
        np.asarray(w_list, dtype=np.float64),
        labels,
    )


def _evaluate_grid_batch(
    strikes: np.ndarray,
    dips: np.ndarray,
    rakes: np.ndarray,
    gamma: np.ndarray,
    pol: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate weighted polarity misfit and nodal margin over (G,) candidate mechanisms."""
    n_mat, d_mat = _fault_normal_and_slip_batch(strikes, dips, rakes)
    gn = n_mat @ gamma.T  # (G, N)
    gd = d_mat @ gamma.T  # (G, N)
    rad = 2.0 * gn * gd  # (G, N)

    sgn_rad = np.where(np.abs(rad) < 1e-9, 0.0, np.sign(rad))
    err = np.where(sgn_rad == 0.0, 0.5, np.maximum(0.0, -pol[None, :] * sgn_rad))

    sum_w = float(np.sum(weights))
    weighted_misfit = err @ weights  # (G,)
    # Margin away from nodal planes: positive when predictions agree with polarities
    signed_margin = (rad @ (weights * pol)) / sum_w
    abs_margin = np.abs(signed_margin)
    # Tie-breaker favors positive signed margin first, then largest |sum y_i R_i|
    score = weighted_misfit - 1e-6 * signed_margin - 1e-9 * abs_margin
    return weighted_misfit, signed_margin, score


def solve_focal_mechanism(
    polarities: Union[Sequence[Tuple[Any, ...]], Sequence[Dict[str, Any]]],
    grid_step_deg: float = 10.0,
    refine: bool = True,
) -> Dict[str, Any]:
    """Invert P-wave first-motion polarities for double-couple focal mechanism (strike, dip, rake).

    Parameters
    ----------
    polarities : sequence of (azimuth_deg, takeoff_angle_deg, polarity) tuples or dicts
        First-motion polarity observations (+1 compressional, -1 dilatational).
    grid_step_deg : float
        Coarse 3D grid search step in degrees (default 10.0).
    refine : bool
        If True, perform multi-stage local 3D refinement around top coarse candidates.

    Returns
    -------
    Dict[str, Any]
        Contains strike, dip, rake, strike2, dip2, rake2, auxiliary_plane,
        axes (P, T, B trend/plunge), misfit_ratio, weighted_misfit, n_misfit,
        n_obs, margin, converged, station_matches.
    """
    az_deg, ih_deg, pol, weights, labels = _parse_polarity_observations(polarities)
    n_obs = int(pol.size)
    sum_w = float(np.sum(weights))

    az_rad = np.radians(az_deg)
    ih_rad = np.radians(ih_deg)
    sih = np.sin(ih_rad)
    gamma = np.column_stack([sih * np.cos(az_rad), sih * np.sin(az_rad), np.cos(ih_rad)])

    step = max(2.0, min(45.0, float(grid_step_deg)))
    s_vals = np.arange(0.0, 360.0, step, dtype=np.float64)
    n_dip = max(2, int(round(90.0 / step)) + 1)
    d_vals = np.linspace(0.0, 90.0, n_dip, dtype=np.float64)
    r_vals = np.arange(-180.0, 180.0, step, dtype=np.float64)

    sg, dg, rg = np.meshgrid(s_vals, d_vals, r_vals, indexing="ij")
    s_flat = sg.ravel()
    d_flat = dg.ravel()
    r_flat = rg.ravel()

    w_misfit, margin_arr, score = _evaluate_grid_batch(s_flat, d_flat, r_flat, gamma, pol, weights)
    best_idx = int(np.argmin(score))
    best_s = float(s_flat[best_idx])
    best_d = float(d_flat[best_idx])
    best_r = float(r_flat[best_idx])
    best_score = float(score[best_idx])

    if refine:
        # Select up to 8 diverse top coarse candidates to refine locally
        order = np.argsort(score)
        seeds: List[Tuple[float, float, float]] = []
        min_w_misfit = float(w_misfit[order[0]])
        for idx in order:
            if len(seeds) >= 8:
                break
            if float(w_misfit[idx]) > min_w_misfit + 0.5 * float(np.min(weights)) and len(seeds) >= 3:
                break
            cand = (float(s_flat[idx]), float(d_flat[idx]), float(r_flat[idx]))
            # Avoid near-duplicate seeds
            too_close = any(
                min(abs(cand[0] - sc[0]), 360.0 - abs(cand[0] - sc[0])) < step * 0.9
                and abs(cand[1] - sc[1]) < step * 0.9
                and min(abs(cand[2] - sc[2]), 360.0 - abs(cand[2] - sc[2])) < step * 0.9
                for sc in seeds
            )
            if not too_close:
                seeds.append(cand)

        stage_steps = [max(0.5, step / 4.0), max(0.2, step / 12.0), 0.1]
        for s0, d0, r0 in seeds:
            cs, cd, cr = s0, d0, r0
            for st_local in stage_steps:
                half = 3.0 * st_local
                ds_arr = np.linspace(cs - half, cs + half, 7, dtype=np.float64) % 360.0
                dd_arr = np.clip(np.linspace(cd - half, cd + half, 7, dtype=np.float64), 0.0, 90.0)
                dr_arr = ((np.linspace(cr - half, cr + half, 7, dtype=np.float64) + 180.0) % 360.0) - 180.0

                lsg, ldg, lrg = np.meshgrid(ds_arr, dd_arr, dr_arr, indexing="ij")
                lsf, ldf, lrf = lsg.ravel(), ldg.ravel(), lrg.ravel()
                _, _, l_score = _evaluate_grid_batch(lsf, ldf, lrf, gamma, pol, weights)
                li = int(np.argmin(l_score))
                cs, cd, cr = float(lsf[li]), float(ldf[li]), float(lrf[li])
                if float(l_score[li]) < best_score:
                    best_score = float(l_score[li])
                    best_s, best_d, best_r = cs, cd, cr

    best_s = float(best_s % 360.0)
    best_d = float(max(0.0, min(90.0, best_d)))
    best_r = _norm_rake(best_r)

    rad_pred = np.asarray(
        p_wave_radiation_pattern(best_s, best_d, best_r, ih_deg, az_deg),
        dtype=np.float64,
    )
    sgn_pred = np.where(np.abs(rad_pred) < 1e-9, 0, np.where(rad_pred > 0.0, 1, -1))
    err_arr = np.where(
        sgn_pred == 0,
        0.5,
        np.maximum(0.0, -pol * sgn_pred),
    )
    weighted_misfit = float(np.sum(weights * err_arr))
    misfit_ratio = float(weighted_misfit / sum_w) if sum_w > 0.0 else 0.0
    n_misfit = int(np.sum(sgn_pred != pol.astype(int)))
    abs_margin = float(abs(np.sum(weights * pol * rad_pred)) / sum_w) if sum_w > 0.0 else 0.0

    s2, d2, r2 = auxiliary_nodal_plane(best_s, best_d, best_r)
    axes = principal_axes(best_s, best_d, best_r)

    station_matches: List[Dict[str, Any]] = []
    for idx in range(n_obs):
        obs_p = int(pol[idx])
        pred_r = float(rad_pred[idx])
        pred_p = int(sgn_pred[idx])
        station_matches.append(
            {
                "station": labels[idx],
                "azimuth": float(az_deg[idx]),
                "takeoff": float(ih_deg[idx]),
                "observed": obs_p,
                "predicted_rad": pred_r,
                "predicted_sign": pred_p,
                "weight": float(weights[idx]),
                "matched": bool(pred_p == obs_p),
            }
        )

    return {
        "strike": best_s,
        "dip": best_d,
        "rake": best_r,
        "strike2": s2,
        "dip2": d2,
        "rake2": r2,
        "auxiliary_plane": (s2, d2, r2),
        "axes": axes,
        "misfit_ratio": misfit_ratio,
        "weighted_misfit": weighted_misfit,
        "n_misfit": n_misfit,
        "n_obs": n_obs,
        "margin": abs_margin,
        "converged": bool(n_obs > 0),
        "station_matches": station_matches,
    }
