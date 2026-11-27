"""Amplitude/period measurement and earthquake magnitude formulas (ML, mb, MS, mB)."""

from __future__ import annotations

import math
from typing import Dict
import numpy as np

from shm_app.core.models import Trace
from shm_app.processing.filters import restitute_ground_displacement_nm

# Richter/Schneider ML distance attenuation table siv_ml_sigma[0..20] for 0..10 deg in 0.5 deg steps (seismics.c)
ML_SIGMA_TABLE = [
    1.40, 1.40, 1.50, 2.20, 2.60, 2.75, 2.90, 3.00, 3.10, 3.20,
    3.30, 3.35, 3.40, 3.55, 3.55, 3.70, 3.70, 3.85, 3.85, 4.00, 4.00,
]

# Gutenberg-Richter mb Q(Delta, h) calibration table mb_sigma from seismics.c
# Rows: distances 0..105 deg in 5-deg steps (22 rows); Columns: depths 0..700 km in 50-km steps (15 cols)
MB_SIGMA_DIST = np.arange(0.0, 110.0, 5.0, dtype=np.float64)
MB_SIGMA_DEPTH = np.arange(0.0, 750.0, 50.0, dtype=np.float64)
MB_SIGMA_GRID = np.array(
    [
        [9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9],
        [4.0, 3.9, 3.7, 3.7, 3.8, 4.0, 4.2, 4.3, 4.2, 4.2, 4.3, 4.4, 4.6, 4.8, 5.0],
        [4.4, 4.2, 4.1, 4.0, 4.1, 4.2, 4.2, 4.3, 4.3, 4.4, 4.5, 4.6, 4.8, 4.9, 5.0],
        [7.0, 6.0, 5.3, 4.9, 4.3, 4.3, 4.3, 4.4, 4.5, 4.6, 4.7, 4.9, 5.0, 5.2, 5.2],
        [6.0, 6.0, 5.9, 5.8, 5.5, 5.3, 5.2, 5.2, 5.3, 5.4, 5.4, 5.5, 5.6, 5.7, 5.7],
        [6.5, 6.3, 6.2, 6.0, 5.9, 5.8, 5.7, 5.7, 5.6, 5.6, 5.6, 5.6, 5.7, 5.7, 5.8],
        [6.6, 6.5, 6.3, 6.2, 6.0, 5.9, 5.9, 5.8, 5.8, 5.7, 5.7, 5.7, 5.7, 5.7, 5.8],
        [6.7, 6.6, 6.5, 6.4, 6.3, 6.1, 6.0, 6.0, 5.9, 5.8, 5.7, 5.7, 5.7, 5.7, 5.7],
        [6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.7, 5.6, 5.6, 5.5, 5.5, 5.4],
        [6.8, 6.6, 6.5, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.7, 5.6, 5.6, 5.7, 5.6],
        [6.7, 6.6, 6.5, 6.3, 6.2, 6.1, 6.0, 5.9, 5.9, 5.8, 5.7, 5.7, 5.6, 5.5, 5.5],
        [6.8, 6.7, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.9, 5.8, 5.8, 5.7, 5.7, 5.6],
        [7.1, 6.9, 6.8, 6.6, 6.4, 6.2, 6.1, 6.0, 6.0, 5.9, 5.9, 5.8, 5.8, 5.7, 5.7],
        [7.0, 6.9, 6.7, 6.5, 6.3, 6.2, 6.1, 6.1, 6.0, 6.0, 5.9, 5.8, 5.8, 5.7, 5.7],
        [6.9, 6.9, 6.8, 6.6, 6.5, 6.3, 6.2, 6.2, 6.1, 6.0, 5.9, 5.9, 5.9, 5.9, 5.9],
        [7.0, 6.9, 6.8, 6.7, 6.5, 6.4, 6.3, 6.2, 6.2, 6.1, 6.0, 6.0, 5.9, 5.9, 5.9],
        [6.8, 6.7, 6.7, 6.5, 6.4, 6.3, 6.3, 6.2, 6.2, 6.1, 6.1, 6.0, 5.9, 5.9, 5.9],
        [7.0, 6.9, 6.8, 6.6, 6.4, 6.3, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2],
        [7.3, 7.1, 6.9, 6.8, 6.6, 6.5, 6.5, 6.4, 6.4, 6.4, 6.5, 6.5, 6.5, 6.4, 6.4],
        [7.7, 7.5, 7.4, 7.2, 7.1, 7.0, 6.9, 6.9, 6.8, 6.8, 6.8, 6.8, 6.8, 6.9, 6.9],
        [7.9, 7.8, 7.5, 7.4, 7.3, 7.2, 7.2, 7.1, 7.1, 7.1, 7.2, 7.2, 7.2, 7.3, 7.4],
        [8.4, 8.2, 7.9, 7.7, 7.5, 7.5, 7.5, 7.4, 7.4, 7.4, 7.5, 7.5, 7.5, 7.7, 7.8],
    ],
    dtype=np.float64,
)


def measure_amplitude_and_period(
    tr: Trace,
    t_lo: float,
    t_hi: float,
    mode: str = "PEAK_TO_PEAK",
) -> Dict[str, float]:
    """Measure peak-to-peak (half of max-min) or zero-to-peak amplitude and dominant period
    in window [t_lo, t_hi] (seconds relative to trace start) matching cblib.c.
    """
    dt = tr.delta if tr.delta > 0 else 0.05
    i0 = max(0, int(round(min(t_lo, t_hi) / dt)))
    i1 = min(tr.length, int(round(max(t_lo, t_hi) / dt)) + 1)
    if i1 <= i0 + 1:
        i0, i1 = 0, tr.length
    seg = tr.data[i0:i1]
    if seg.size < 2:
        return {"ampl_raw": 0.0, "ampl_nm": 0.0, "period": 1.0, "ampl_time": t_lo}

    idx_min = int(np.argmin(seg))
    idx_max = int(np.argmax(seg))
    vmin = float(seg[idx_min])
    vmax = float(seg[idx_max])

    if mode.upper().startswith("ZERO"):
        if abs(vmax) >= abs(vmin):
            raw_ampl = abs(vmax)
            idx_ext = idx_max
        else:
            raw_ampl = abs(vmin)
            idx_ext = idx_min
    else:
        # Peak-to-peak / 2
        raw_ampl = 0.5 * abs(vmax - vmin)
        idx_ext = idx_max if abs(vmax) >= abs(vmin) else idx_min

    half_per = abs(idx_max - idx_min) * dt
    period = 2.0 * half_per if half_per >= dt else max(0.2, (i1 - i0) * dt * 0.5)
    ampl_nm = restitute_ground_displacement_nm(raw_ampl, period, tr.filter_name)

    return {
        "ampl_raw": raw_ampl,
        "ampl_nm": ampl_nm,
        "ampl_veloc": raw_ampl,
        "period": period,
        "ampl_time": (i0 + idx_ext) * dt,
    }


def compute_magnitude_ml(ampl_nm: float, dist_deg: float, use_iaspei: bool = False) -> float:
    """Compute local magnitude ML from displacement amplitude (nm) and epicentral distance (deg)
    matching si_CompMagnitudes in seismics.c.
    """
    if ampl_nm <= 0.0 or dist_deg <= 0.0:
        return 0.0
    if use_iaspei:
        # IASPEI 2011 standard Wood-Anderson formula (dist in km, A in nm)
        r_km = dist_deg * DEG_TO_KM
        return math.log10(ampl_nm) + 1.11 * math.log10(r_km) + 0.00189 * r_km - 2.09

    # Classical SHM Richter/Schneider table interpolation (A in micrometers = ampl_nm / 1000)
    idx_f = max(0.0, min(20.0, dist_deg * 2.0))
    i0 = int(math.floor(idx_f))
    i1 = min(20, i0 + 1)
    frac = idx_f - i0
    sigma = ML_SIGMA_TABLE[i0] + frac * (ML_SIGMA_TABLE[i1] - ML_SIGMA_TABLE[i0])
    return math.log10(ampl_nm / 1000.0) + sigma


def compute_magnitude_mb(ampl_nm: float, period_sec: float, dist_deg: float, depth_km: float = 33.0) -> float:
    """Compute body-wave magnitude mb = log10(A_um / T) + Q(Delta, h) using Gutenberg-Richter mb_sigma
    table from seismics.c.
    """
    if ampl_nm <= 0.0 or period_sec <= 0.0 or dist_deg <= 0.0:
        return 0.0
    d_clamped = max(5.0, min(105.0, dist_deg))
    z_clamped = max(0.0, min(700.0, depth_km))

    id0 = min(20, max(1, int(d_clamped // 5.0)))
    id1 = min(21, id0 + 1)
    fd = (d_clamped - id0 * 5.0) / 5.0

    iz0 = min(13, max(0, int(z_clamped // 50.0)))
    iz1 = min(14, iz0 + 1)
    fz = (z_clamped - iz0 * 50.0) / 50.0

    q00 = MB_SIGMA_GRID[id0, iz0]
    q10 = MB_SIGMA_GRID[id1, iz0]
    q01 = MB_SIGMA_GRID[id0, iz1]
    q11 = MB_SIGMA_GRID[id1, iz1]
    q_val = (1.0 - fd) * (1.0 - fz) * q00 + fd * (1.0 - fz) * q10 + (1.0 - fd) * fz * q01 + fd * fz * q11

    a_um = ampl_nm / 1000.0
    return math.log10(a_um / period_sec) + float(q_val)


def compute_magnitude_ms(ampl_nm: float, period_sec: float, dist_deg: float) -> float:
    """Compute surface-wave magnitude MS using the Prague formula:
    MS = log10(A_um / T) + 1.66 * log10(Delta_deg) + 3.3
    matching seismics.c.
    """
    if ampl_nm <= 0.0 or period_sec <= 0.0 or dist_deg <= 0.0:
        return 0.0
    a_um = ampl_nm / 1000.0
    return math.log10(a_um / period_sec) + 1.66 * math.log10(dist_deg) + 3.3


def compute_magnitude_mbb(vel_ampl_nm_s: float, dist_deg: float, depth_km: float = 33.0) -> float:
    """Compute broadband body-wave magnitude mB = log10(V_max_um_s / (2*pi)) + Q(Delta, h)."""
    if vel_ampl_nm_s <= 0.0 or dist_deg <= 0.0:
        return 0.0
    equiv_disp_nm = vel_ampl_nm_s / (2.0 * math.pi)
    return compute_magnitude_mb(equiv_disp_nm, 1.0, dist_deg, depth_km)


def compute_trace_window_magnitudes(
    tr: Trace,
    t_lo: float,
    t_hi: float,
    dist_deg: float,
    depth_km: float = 33.0,
) -> Dict[str, float]:
    """Measure amplitude/period in [t_lo, t_hi] and compute ML, mb, MS, and mB magnitudes."""
    ap = measure_amplitude_and_period(tr, t_lo, t_hi)
    ml = compute_magnitude_ml(ap["ampl_nm"], min(10.0, dist_deg))
    mb = compute_magnitude_mb(ap["ampl_nm"], ap["period"], dist_deg, depth_km)
    ms = compute_magnitude_ms(ap["ampl_nm"], max(15.0, ap["period"]), dist_deg)
    mbb = compute_magnitude_mbb(ap["ampl_veloc"], dist_deg, depth_km)
    return {
        **ap,
        "ml": ml,
        "mb": mb,
        "ms": ms,
        "mbb": mbb,
    }

