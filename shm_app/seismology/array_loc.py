"""Array processing (Plane-wave fit, Beamforming, Alignment, Vespagram, Wideband FK analysis),
earthquake location (slowness inversion, S-P differential, multi-station Geiger/network solver),
cross-correlation & Baer-Kradolfer onset picking, and magnitude calculations (ML, mb, MS, mB)
matching shmath.c, fkwid.c, shcorr.c, earthloc.c, seismics.c, and cblib.c.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import (
    ConfidenceEllipse,
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.processing.filters import restitute_ground_displacement_nm
from shm_app.processing.trace_ops import resample_traces_to_common_grid
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
    detect_outliers_3sigma,
)
from shm_app.seismology.traveltime import TravelTimeManager, get_ttt_manager, sphere_add, sphere_diff

DEG_TO_KM = 111.1949266

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


def plane_wave_locate(
    arrival_times: List[float],
    x_km: List[float],
    y_km: List[float],
    elev_m: Optional[List[float]] = None,
    vp_kms: Optional[List[float]] = None,
    dt: float = 0.05,
) -> Dict[str, Any]:
    """Fit plane-wave backazimuth (deg) and slowness (s/deg) with uncertainties and station residuals
    matching mt_locate in shmath.c.
    """
    n = len(arrival_times)
    if n < 3:
        raise ValueError("Plane-wave location requires at least 3 station arrival picks")

    t_in = np.asarray(arrival_times, dtype=np.float64).copy()
    x = np.asarray(x_km, dtype=np.float64).copy()
    y = np.asarray(y_km, dtype=np.float64).copy()

    # Shift to geometric center (xmean, ymean, tmean) matching mt_locate in shmath.c
    xm, ym, tm = float(np.mean(x)), float(np.mean(y)), float(np.mean(t_in))
    xc = x - xm
    yc = y - ym
    tc = t_in - tm

    # Iterative elevation correction if elevations are provided
    sx, sy = 0.0, 0.0
    n_iter = 2 if (elev_m is not None and any(e != 0.0 for e in elev_m)) else 1
    for _ in range(n_iter):
        tc_work = tc.copy()
        if elev_m is not None and (sx != 0.0 or sy != 0.0):
            s_sec_km = math.hypot(sx, sy)
            el = np.asarray(elev_m, dtype=np.float64)
            el_c = (el - np.mean(el)) / 1000.0  # km
            vp = np.asarray(vp_kms if vp_kms else [5.8] * n, dtype=np.float64)
            vp = np.where(vp > 0.1, vp, 5.8)
            cos_i = np.sqrt(np.maximum(0.0, 1.0 - (vp * s_sec_km) ** 2))
            dt_elev = (el_c / vp) * cos_i
            tc_work = tc - (dt_elev - np.mean(dt_elev))

        axx = float(np.dot(xc, xc))
        axy = float(np.dot(xc, yc))
        ayy = float(np.dot(yc, yc))
        tx = float(np.dot(tc_work, xc))
        ty = float(np.dot(tc_work, yc))
        tq = float(np.dot(tc_work, tc_work))

        det = axx * ayy - axy * axy
        if abs(det) < 1e-10:
            raise ValueError("Collinear array stations (zero determinant in mt_locate)")

        sx = (tx * ayy - ty * axy) / det
        sy = (ty * axx - tx * axy) / det

    # Backazimuth points FROM array TOWARD source (opposite propagation vector (sx, sy))
    if sy == 0.0:
        azimuth = 90.0 if sx < 0.0 else 270.0
    else:
        azimuth = math.degrees(math.atan(sx / sy))
        if sy > 0.0:
            azimuth += 180.0
        elif sx > 0.0:
            azimuth += 360.0

    norm_s = math.hypot(sx, sy)
    slowness = DEG_TO_KM * norm_s
    if norm_s < 1e-10:
        slowerr, azimerr = 0.0, 0.0
    else:
        inv_norm = 1.0 / norm_s
        abs_det = abs(det)
        if n == 3:
            r = math.atan2(sx, sy)
            saz, caz = math.sin(r), math.cos(r)
            rf = inv_norm * dt / abs_det
            scr1 = saz * axx + caz * axy
            scr2 = saz * axy + caz * ayy
            azimerr = math.degrees(rf * math.hypot(scr1, scr2))
            scr1 = caz * axx - saz * axy
            scr2 = caz * axy - saz * ayy
            rf *= inv_norm * math.hypot(scr1, scr2)
            slowerr = rf * DEG_TO_KM / (inv_norm * inv_norm)
        else:
            r = math.sqrt(abs(tq - tx * sx - ty * sy) / float(n - 3))
            sx2, sy2, sxy = sx * sx, sy * sy, sx * sy
            slowerr = r * math.sqrt(max(0.0, (ayy * sx2 - 2.0 * axy * sxy + axx * sy2) / abs_det)) * inv_norm * DEG_TO_KM
            azimerr = math.degrees(
                r * math.sqrt(max(0.0, (ayy * sy2 + 2.0 * axy * sxy + axx * sx2) / abs_det)) * (inv_norm ** 2)
            )

    # Compute individual station residuals: t_obs - t_pred
    t_pred = xc * sx + yc * sy
    residuals = tc_work - t_pred

    return {
        "azimuth": float(azimuth % 360.0),
        "d_azimuth": float(azimerr),
        "slowness": float(slowness),
        "d_slowness": float(slowerr),
        "residuals": [float(r) for r in residuals],
        "sx": float(sx),
        "sy": float(sy),
    }


def beam_delays_sec(
    x_km: List[float],
    y_km: List[float],
    azimuth_deg: float,
    slowness_s_deg: float,
    elev_m: Optional[List[float]] = None,
    vp_kms: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute plane-wave arrival time delays (in seconds) relative to origin (0,0) matching mt_beamshift.

    Note: A wave coming from backazimuth `azimuth_deg` arrives EARLIER at stations in the direction
    of `azimuth_deg`, so its propagation vector is (-sin(az), -cos(az)).
    Wait: in SHM's `mt_beamshift`:
      `azimuth_math = (90 - azimuth) * pi/180` -> `cos(azimuth_math) = sin(azimuth)`, `sin(azimuth_math) = cos(azimuth)`
      `delay = (x * sin(azimuth) + y * cos(azimuth)) * (slowness / 111.195)`
    And when forming a beam, sample `t - delay` is summed so that early arrivals (positive `x*sin(az)+y*cos(az)`)
    are shifted forward by `+delay` (or sampled at `t - delay`) to align with the reference point!
    """
    az_r = math.radians(azimuth_deg)
    s_km = slowness_s_deg / DEG_TO_KM
    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    delays = (x * math.sin(az_r) + y * math.cos(az_r)) * s_km

    if elev_m is not None:
        el_km = np.asarray(elev_m, dtype=np.float64) / 1000.0
        vp = np.asarray(vp_kms if vp_kms else [5.8] * x.size, dtype=np.float64)
        vp = np.where(vp > 0.1, vp, 5.8)
        cos_i = np.sqrt(np.maximum(0.0, 1.0 - (vp * abs(s_km)) ** 2))
        # Higher elevation stations receive upgoing wavefront later by + (el_km / vp) * cos_i,
        # so relative to early-arrival positive convention we subtract the elevation delay
        delays = delays - (el_km / vp) * cos_i

    return delays


def compute_beam(
    traces: List[Trace],
    azimuth_deg: float,
    slowness_s_deg: float,
    ref_station: str = "CENTRE",
    nth_root: float = 1.0,
    config: Optional[SHMConfig] = None,
) -> Tuple[Trace, List[float]]:
    """Form an array delay-and-sum (or N-th root) BEAM trace matching sh_beam in shmenu3.c.

    Returns (beam_trace, delays_applied_sec).
    """
    if not traces:
        raise ValueError("Cannot compute beam from empty trace list")
    cfg = config or get_config()
    stations = [t.station for t in traces]
    rel_coords = cfg.relative_xy_km(stations, ref_station=ref_station)
    x_km = [c[0] for c in rel_coords]
    y_km = [c[1] for c in rel_coords]
    elev_m = [c[2] for c in rel_coords]
    vp_kms = [c[3] for c in rel_coords]

    delays = beam_delays_sec(x_km, y_km, azimuth_deg, slowness_s_deg, elev_m=elev_m, vp_kms=vp_kms)

    ref = traces[0]
    t0, _dt, grid_t, aligned = resample_traces_to_common_grid(
        traces,
        offsets_sec=delays,
        include_display_offset=False,
        shift_grid_bounds=False,
    )

    p = max(1.0, float(nth_root))
    stack = np.zeros(grid_t.size, dtype=np.float64)
    for y in aligned:
        # Wave arrives at station at t_ref - d_sec, so traces were shifted by +d_sec onto grid_t
        if p > 1.0:
            stack += np.sign(y) * (np.abs(y) ** (1.0 / p))
        else:
            stack += y

    stack /= len(traces)
    if p > 1.0:
        stack = np.sign(stack) * (np.abs(stack) ** p)

    beam_tr = ref.copy(copy_picks=False)
    beam_tr.data = stack
    beam_tr.start_epoch = t0
    beam_tr.display_offset = 0.0
    beam_tr.station = "BEAM"
    beam_tr.azimuth = azimuth_deg
    beam_tr.slowness = slowness_s_deg
    beam_tr.comment = f"BEAM az={azimuth_deg:.1f} slo={slowness_s_deg:.2f} ref={ref_station}"
    return beam_tr, [float(d) for d in delays]


def align_traces_on_phase(
    traces: List[Trace],
    phase_name: str = "P",
    create_sum_trace: bool = True,
) -> Optional[Trace]:
    """Align traces on their picked phase arrival time (or theoretical arrival) and optionally return ALIGN stack."""
    pick_epochs: List[Tuple[Trace, float]] = []
    for tr in traces:
        pk = tr.get_pick(phase_name)
        if pk is not None and pk.onset:
            pick_epochs.append((tr, pk.onset_epoch))
        elif phase_name in tr.theo_picks:
            pick_epochs.append((tr, tr.theo_picks[phase_name]))

    if not pick_epochs:
        return None

    ref_epoch = pick_epochs[0][1]
    for tr, p_ep in pick_epochs:
        tr.display_offset = ref_epoch - p_ep

    if not create_sum_trace:
        return None

    from shm_app.processing.trace_ops import sum_traces

    aligned_tr = sum_traces([tr for tr, _ in pick_epochs])
    if aligned_tr is not None:
        aligned_tr.station = "ALIGN"
        aligned_tr.comment = f"ALIGN on {phase_name}"
    return aligned_tr


def compute_vespagram(
    traces: List[Trace],
    azimuth_deg: float,
    slow_lo: float = 2.0,
    slow_hi: float = 12.0,
    slow_step: float = 0.5,
    power: float = 2.0,
    ref_station: str = "CENTRE",
    config: Optional[SHMConfig] = None,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> Tuple[List[Trace], np.ndarray, np.ndarray, np.ndarray]:
    """Compute N-th root Vespagram slowness gather matching sh_vespa in shmenu5.c.

    Returns (vespa_traces, times_sec, slownesses_s_deg, vespa_matrix_2d).
    """
    slownesses = np.arange(slow_lo, slow_hi + 0.5 * slow_step, max(0.05, slow_step), dtype=np.float64)
    vespa_traces: List[Trace] = []
    rows: List[np.ndarray] = []
    total_steps = len(slownesses)

    for idx, slo in enumerate(slownesses):
        if progress_callback is not None:
            if not progress_callback(idx, total_steps):
                return vespa_traces, np.array([]), slownesses[:idx], np.zeros((1, 1))

        b_tr, _ = compute_beam(
            traces,
            azimuth_deg=azimuth_deg,
            slowness_s_deg=float(slo),
            ref_station=ref_station,
            nth_root=power,
            config=config,
        )
        b_tr.station = f"V{slo:4.1f}"
        b_tr.slowness = float(slo)
        vespa_traces.append(b_tr)
        rows.append(b_tr.data)

    if progress_callback is not None:
        progress_callback(total_steps, total_steps)

    mat = np.vstack(rows) if rows else np.zeros((1, 1))
    times = np.arange(mat.shape[1], dtype=np.float64) * (traces[0].delta if traces else 0.05)
    return vespa_traces, times, slownesses, mat


def compute_fk_analysis(
    traces: List[Trace],
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    f_lo: float = 0.5,
    f_hi: float = 3.0,
    max_slowness: float = 12.0,
    grid_points: int = 41,
    config: Optional[SHMConfig] = None,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> Dict[str, Any]:
    """Wideband coherent Frequency-Wavenumber (FK) analysis with 9-point 2D quadratic sub-grid
    peak interpolation matching fkwid.c.
    """
    if len(traces) < 3:
        raise ValueError("FK analysis requires at least 3 array traces")

    cfg = config or get_config()
    dt = traces[0].delta if traces[0].delta > 0 else 0.05
    stations = [t.station for t in traces]
    rel_coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
    x_deg = np.array([c[0] / DEG_TO_KM for c in rel_coords], dtype=np.float64)
    y_deg = np.array([c[1] / DEG_TO_KM for c in rel_coords], dtype=np.float64)

    # Extract window from each trace
    segs: List[np.ndarray] = []
    min_len = min(t.length for t in traces)
    if t_lo is not None and t_hi is not None:
        i0 = max(0, int(round(min(t_lo, t_hi) / dt)))
        i1 = min(min_len, int(round(max(t_lo, t_hi) / dt)) + 1)
    else:
        i0, i1 = 0, min_len
    if i1 <= i0 + 4:
        i0, i1 = 0, min_len

    for tr in traces:
        s = tr.data[i0:i1] - np.mean(tr.data[i0:i1])
        segs.append(s)

    mat_data = np.vstack(segs)  # (M, L)
    m_stat, l_samps = mat_data.shape
    nfft = 1
    while nfft < l_samps:
        nfft <<= 1

    specs = np.fft.rfft(mat_data, n=nfft, axis=1)  # (M, nfft//2 + 1)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    f_mask = (freqs >= min(f_lo, f_hi)) & (freqs <= max(f_lo, f_hi)) & (freqs > 0)
    if not np.any(f_mask):
        f_mask = freqs > 0

    F_sub = specs[:, f_mask]  # (M, K)
    omega_sub = 2.0 * math.pi * freqs[f_mask]  # (K,)
    total_energy = float(np.sum(np.abs(F_sub) ** 2))
    if total_energy <= 1e-30:
        total_energy = 1.0

    n_grid = max(11, int(grid_points) | 1)  # odd grid size
    s_axis = np.linspace(-max_slowness, max_slowness, n_grid)
    # Slowness vector (sx, sy) pointing toward source (backazimuth direction):
    # delay_m = x_deg[m]*sx + y_deg[m]*sy -> steering phase exp(-i * omega * delay_m)
    # Wait: if wave comes FROM (sx, sy), arrival time at (x_m, y_m) is t_0 - (x_m*sx + y_m*sy),
    # so its Fourier phase is exp(+i * omega * (x_m*sx + y_m*sy)), and multiplying by exp(-i * omega * (x_m*sx + y_m*sy)) aligns phases!
    sx_grid, sy_grid = np.meshgrid(s_axis, s_axis)  # (G, G)
    # Computetau(g_y, g_x, m) = x_deg[m]*sx + y_deg[m]*sy
    tau = sx_grid[:, :, None] * x_deg[None, None, :] + sy_grid[:, :, None] * y_deg[None, None, :]  # (G, G, M)
    # Sum over stations for each frequency:
    total_steps = len(omega_sub)
    power_grid = np.zeros((n_grid, n_grid), dtype=np.float64)
    for k_idx, w_k in enumerate(omega_sub):
        if progress_callback is not None:
            if not progress_callback(k_idx, total_steps):
                return {}
        steer = np.exp(-1j * w_k * tau)  # (G, G, M)
        beam_f = np.sum(steer * F_sub[None, None, :, k_idx], axis=2)  # (G, G)
        power_grid += np.abs(beam_f) ** 2

    if progress_callback is not None:
        progress_callback(total_steps, total_steps)

    semblance_grid = power_grid / (m_stat * total_energy)

    # Find peak on grid
    flat_idx = int(np.argmax(semblance_grid))
    iy, ix = divmod(flat_idx, n_grid)
    ds = float(s_axis[1] - s_axis[0])
    sx_peak = float(s_axis[ix])
    sy_peak = float(s_axis[iy])
    peak_power = float(semblance_grid[iy, ix])

    # 9-point 2D orthogonal quadratic sub-grid peak interpolation matching fkwid.c
    if 1 <= ix < n_grid - 1 and 1 <= iy < n_grid - 1:
        patch = semblance_grid[iy - 1 : iy + 2, ix - 1 : ix + 2]
        # d_x = (sum_{r} (patch[r, 2] - patch[r, 0])) / (2 * sum_{r} (patch[r, 2] - 2*patch[r, 1] + patch[r, 0]))
        gx = float(np.sum(patch[:, 2] - patch[:, 0]))
        hxx = float(np.sum(patch[:, 2] - 2.0 * patch[:, 1] + patch[:, 0]))
        gy = float(np.sum(patch[2, :] - patch[0, :]))
        hyy = float(np.sum(patch[2, :] - 2.0 * patch[1, :] + patch[0, :]))
        if hxx < -1e-12:
            dx_sub = max(-0.5, min(0.5, -0.5 * gx / hxx))
            sx_peak += dx_sub * ds
        if hyy < -1e-12:
            dy_sub = max(-0.5, min(0.5, -0.5 * gy / hyy))
            sy_peak += dy_sub * ds

    slowness = math.hypot(sx_peak, sy_peak)
    baz_deg = math.degrees(math.atan2(sx_peak, sy_peak)) % 360.0
    app_vel = (DEG_TO_KM / slowness) if slowness > 1e-6 else 999.9
    power_db = 10.0 * math.log10(max(1e-6, peak_power))

    return {
        "azimuth": baz_deg,
        "slowness": slowness,
        "apparent_velocity_kms": app_vel,
        "semblance": peak_power,
        "power_db": power_db,
        "sx": sx_peak,
        "sy": sy_peak,
        "s_axis": s_axis,
        "grid": semblance_grid,
    }


def locate_by_slowness_azimuth(
    station_code: str,
    phase_name: str,
    onset_epoch: float,
    slowness_s_deg: float,
    baz_deg: float,
    depth_km: float = 33.0,
    config: Optional[SHMConfig] = None,
    ttt: Optional[TravelTimeManager] = None,
) -> AnalysisParamSet:
    """Invert observed array slowness & backazimuth for epicentral distance, (lat, lon), origin time,
    and Flinn-Engdahl region matching Locate -> By Slowness in cblib.c / earthloc.c.
    """
    cfg = config or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)
    stat = cfg.get_station(station_code)

    dist_deg = tm.distance_from_slowness(phase_name, slowness_s_deg, depth_km)
    if dist_deg is None:
        dist_deg = max(1.0, (11.0 - slowness_s_deg) * 10.0)

    tt_sec = tm.travel_time(phase_name, dist_deg, depth_km) or (dist_deg * slowness_s_deg)
    orig_epoch = onset_epoch - tt_sec
    lat, lon = sphere_add(stat.lat, stat.lon, dist_deg, baz_deg)
    igreg, grname, isreg, srname = cfg.fer.lookup(lat, lon)

    return AnalysisParamSet(
        distance=dist_deg,
        azimuth=baz_deg,
        slowness=slowness_s_deg,
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
                    qual = getattr(dp_pk, "quality", 4)
                    weight = max(0.1, float(qual)) / (sigma ** 2)
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

    excluded_set = set(str(s).upper() for s in (excluded_stations or []))
    weight_map = {str(k).upper(): float(v) for k, v in (weight_overrides or {}).items()}

    def _is_excluded(p: PhasePick) -> bool:
        st = p.station.upper()
        if st in excluded_set:
            return True
        if f"{st}:{p.name.upper()}" in excluded_set or f"{st}:{p.name}" in excluded_set:
            return True
        ident = f"{st}.{p.comp.upper() if p.comp else 'Z'}.{p.name.upper()}"
        if ident in excluded_set:
            return True
        if ident in weight_map and weight_map[ident] <= 0.0:
            return True
        if f"{st}:{p.name.upper()}" in weight_map and weight_map[f"{st}:{p.name.upper()}"] <= 0.0:
            return True
        if st in weight_map and weight_map[st] <= 0.0:
            return True
        return False

    def _get_pick_weight(p: PhasePick) -> float:
        st = p.station.upper()
        ident = f"{st}.{p.comp.upper() if p.comp else 'Z'}.{p.name.upper()}"
        if ident in weight_map:
            return max(0.01, float(weight_map[ident]))
        if f"{st}:{p.name.upper()}" in weight_map:
            return max(0.01, float(weight_map[f"{st}:{p.name.upper()}"]))
        if st in weight_map:
            return max(0.01, float(weight_map[st]))
        return max(0.1, float(p.weight) / 4.0) if p.weight <= 4 else 0.1

    active_picks = [p for p in valid_picks if not _is_excluded(p)]
    if len(active_picks) < 3:
        active_picks = list(valid_picks)

    # Group picks by station to check if we have S-P differential times at any station
    by_stat: Dict[str, Dict[str, PhasePick]] = {}
    for p in active_picks:
        by_stat.setdefault(p.station.upper(), {})[p.name] = p

    first_stat = cfg.get_station(active_picks[0].station)

    # Initial guess
    lat = initial_lat if initial_lat is not None else first_stat.lat + 2.0
    lon = initial_lon if initial_lon is not None else first_stat.lon + 2.0
    z = max(0.0, float(depth_km))

    # Improve initial guess if we have S-P or slowness + backazimuth
    for st_code, ph_map in by_stat.items():
        st_inf = cfg.get_station(st_code)
        p_pk = ph_map.get("P") or ph_map.get("Pg") or ph_map.get("Pn")
        s_pk = ph_map.get("S") or ph_map.get("Sg") or ph_map.get("Sn")
        if p_pk and s_pk:
            dt_sp = abs(s_pk.onset_epoch - p_pk.onset_epoch)
            d_est = tm.distance_from_diff_time(p_pk.name, s_pk.name, dt_sp, z)
            baz = p_pk.beam_azim or initial_baz or 45.0
            if d_est is not None:
                lat, lon = sphere_add(st_inf.lat, st_inf.lon, d_est, baz)
                break
        elif p_pk and p_pk.slowness > 0.0:
            d_est = tm.distance_from_slowness(p_pk.name, p_pk.slowness, z)
            baz = p_pk.beam_azim or initial_baz or 45.0
            if d_est is not None:
                lat, lon = sphere_add(st_inf.lat, st_inf.lon, d_est, baz)
                break

    # Estimate initial origin time t0 from earliest pick
    d0, _, _ = sphere_diff(first_stat.lat, first_stat.lon, lat, lon)
    tt0 = tm.travel_time(active_picks[0].name, max(0.1, d0), z) or (d0 * 10.0)
    t0 = active_picks[0].onset_epoch - tt0

    converged = False
    actual_iters = 0
    final_J = np.zeros((0, 4 if free_depth else 3), dtype=np.float64)
    final_res = np.zeros(0, dtype=np.float64)
    final_weights = np.zeros(0, dtype=np.float64)

    # Iterative damped least squares over (t0, lat, lon[, z])
    for it in range(max_iter):
        actual_iters = it + 1
        rows_j: List[List[float]] = []
        res_vec: List[float] = []
        weights: List[float] = []

        for p in active_picks:
            st = cfg.get_station(p.station)
            dist_deg, _az_st_ev, _baz = sphere_diff(st.lat, st.lon, lat, lon)
            tt = tm.travel_time(p.name, max(0.05, dist_deg), z)
            if tt is None:
                continue

            res = p.onset_epoch - (t0 + tt)
            w = _get_pick_weight(p)

            # Partial derivatives via finite differences (dlat, dlon = 0.05 deg, dz = 2.0 km)
            eps_deg = 0.05
            d_lat_p, _, _ = sphere_diff(st.lat, st.lon, lat + eps_deg, lon)
            tt_lat = tm.travel_time(p.name, max(0.05, d_lat_p), z) or tt
            d_lon_p, _, _ = sphere_diff(st.lat, st.lon, lat, lon + eps_deg)
            tt_lon = tm.travel_time(p.name, max(0.05, d_lon_p), z) or tt

            dt_dlat = (tt_lat - tt) / eps_deg
            dt_dlon = (tt_lon - tt) / eps_deg

            if free_depth:
                tt_z = tm.travel_time(p.name, max(0.05, dist_deg), z + 2.0) or tt
                dt_dz = (tt_z - tt) / 2.0
                rows_j.append([1.0, dt_dlat, dt_dlon, dt_dz])
            else:
                rows_j.append([1.0, dt_dlat, dt_dlon])
            res_vec.append(res)
            weights.append(w)

        if len(rows_j) < 3:
            break

        J = np.array(rows_j, dtype=np.float64)
        r = np.array(res_vec, dtype=np.float64)
        W = np.diag(weights)
        JT_W = J.T @ W
        H = JT_W @ J + np.eye(J.shape[1]) * 1e-2  # Levenberg-Marquardt damping
        g = JT_W @ r
        try:
            delta_m = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        t0 += float(delta_m[0])
        lat = max(-89.9, min(89.9, lat + float(delta_m[1])))
        lon = ((lon + float(delta_m[2]) + 180.0) % 360.0) - 180.0
        if free_depth and delta_m.size >= 4:
            z = max(0.0, min(700.0, z + float(delta_m[3])))

        final_J = J
        final_res = r
        final_weights = np.array(weights, dtype=np.float64)

        if np.linalg.norm(delta_m[1:3]) < 1e-4 and abs(delta_m[0]) < 1e-3:
            converged = True
            break

    # Calculate residuals for ALL picks (active + excluded)
    station_residuals: Dict[str, float] = {}
    residual_entries: List[ResidualEntry] = []
    active_resids: List[float] = []

    for p in active_picks:
        st = cfg.get_station(p.station)
        dist_deg, _, _ = sphere_diff(st.lat, st.lon, lat, lon)
        tt = tm.travel_time(p.name, max(0.05, dist_deg), z)
        if tt is not None:
            active_resids.append(float(p.onset_epoch - (t0 + tt)))

    sigma_hat = compute_robust_sigma(active_resids) if active_resids else 0.5
    outlier_thresh = max(0.3, 3.0 * sigma_hat)

    for p in valid_picks:
        st = cfg.get_station(p.station)
        dist_deg, _az, baz = sphere_diff(st.lat, st.lon, lat, lon)
        tt = tm.travel_time(p.name, max(0.05, dist_deg), z)
        if tt is not None:
            calc_t = t0 + tt
            res = float(p.onset_epoch - calc_t)
            p.resid = res
            station_residuals[f"{p.station}:{p.name}"] = res

            is_active = p in active_picks
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
    dof = max(1, len(active_picks) - n_params)
    rms = compute_rmse(active_resids)
    wrms = compute_wrmse(final_res, final_weights) if final_res.size > 0 else rms
    mad_val = compute_mad(active_resids)
    chi2_red = compute_reduced_chi2(active_resids, dof=dof)
    kappa = compute_condition_number(final_J) if final_J.size > 0 else 1.0

    s_sq = (float(np.sum(final_weights * (final_res ** 2))) / dof) if final_res.size > 0 else (rms ** 2)
    cov_m = compute_covariance_matrix(final_J, final_weights, s_sq=s_sq) if final_J.size > 0 else np.eye(n_params) * (rms ** 2)
    corr_m = compute_correlation_matrix(cov_m)

    if cov_m.shape[0] >= 3:
        cov_epi = cov_m[1:3, 1:3]
        semi_major, semi_minor, strike_deg = compute_confidence_ellipse_2d(
            cov_epi, dof=dof, deg_to_km=DEG_TO_KM
        )
        d_lat = float(np.sqrt(max(0.0, cov_m[1, 1])) * DEG_TO_KM)
        d_lon = float(np.sqrt(max(0.0, cov_m[2, 2])) * DEG_TO_KM)
        d_t0 = float(np.sqrt(max(0.0, cov_m[0, 0])))
        d_z = float(np.sqrt(max(0.0, cov_m[3, 3]))) if free_depth and cov_m.shape[0] >= 4 else 0.0
    else:
        semi_major, semi_minor, strike_deg = (rms * 5.0, rms * 5.0, 0.0)
        d_lat, d_lon, d_t0, d_z = (rms * 5.0, rms * 5.0, rms, 0.0)

    conf_ellipse = ConfidenceEllipse(
        semi_major_km=round(semi_major, 3),
        semi_minor_km=round(semi_minor, 3),
        strike_deg=round(strike_deg, 1),
        center_x=round(lon, 4),
        center_y=round(lat, 4),
    )

    outlier_count = sum(1 for e in residual_entries if e.is_outlier)
    norm_stats = compute_normality_stats(active_resids)

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
            standard_error=d_lat / DEG_TO_KM,
            confidence_interval_95=(lat - 1.96 * (d_lat / DEG_TO_KM), lat + 1.96 * (d_lat / DEG_TO_KM)),
            unit="deg",
        ),
        "longitude": ParameterUncertainty(
            name="longitude",
            value=lon,
            standard_error=d_lon / DEG_TO_KM,
            confidence_interval_95=(lon - 1.96 * (d_lon / DEG_TO_KM), lon + 1.96 * (d_lon / DEG_TO_KM)),
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

    diag_result = FitDiagnosticResult(
        fit_type=FitType.HYPOCENTER_GEIGER,
        title="Multi-Station Geiger Inversion",
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
        context_data={
            "picks": picks,
            "depth_km": depth_km,
            "free_depth": free_depth,
            "t0": t0,
            "lat": lat,
            "lon": lon,
            "depth": z,
        },
    )

    ref_dist, _az, ref_baz = sphere_diff(first_stat.lat, first_stat.lon, lat, lon)
    ref_slo = tm.slowness(valid_picks[0].name, ref_dist, z) or 0.0
    igreg, grname, isreg, srname = cfg.fer.lookup(lat, lon)

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
        phase=valid_picks[0].name,
        region_id=igreg,
        region_name=grname,
        sregion_id=isreg,
        sregion_name=srname,
        ref_lat=first_stat.lat,
        ref_lon=first_stat.lon,
        source="LOCSAT/GEIGER",
        comment=f"Geiger fit RMS={rms:.2f} s ({len(active_picks)}/{len(valid_picks)} picks)",
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



def correlate_traces(
    ref_tr: Trace,
    target_tr: Trace,
    t_ref_lo: float,
    t_ref_hi: float,
    max_lag_sec: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute zero-mean normalized cross-correlation between a reference window [t_ref_lo, t_ref_hi]
    on ref_tr and target_tr matching mt_corr in shcorr.c.

    Returns (lags_sec, corr_coeffs, best_lag_sec, max_corr_coeff).
    """
    dt = ref_tr.delta if ref_tr.delta > 0 else 0.05
    i0 = max(0, int(round(min(t_ref_lo, t_ref_hi) / dt)))
    i1 = min(ref_tr.length, int(round(max(t_ref_lo, t_ref_hi) / dt)) + 1)
    if i1 <= i0 + 2:
        i0, i1 = 0, min(ref_tr.length, 100)

    x = ref_tr.data[i0:i1] - np.mean(ref_tr.data[i0:i1])
    norm_x = float(np.linalg.norm(x))
    if norm_x <= 1e-30:
        return np.array([0.0]), np.array([0.0]), 0.0, 0.0

    win_len = x.size
    max_lag_samps = max(1, int(round(max_lag_sec / dt)))
    lags_samps = np.arange(-max_lag_samps, max_lag_samps + 1)
    lags_sec = lags_samps * dt
    corrs = np.zeros(lags_samps.size, dtype=np.float64)

    # Center sample on target_tr corresponding to i0 on ref_tr (accounting for start_epoch difference)
    epoch_i0 = ref_tr.start_epoch + ref_tr.display_offset + i0 * dt
    j0_center = int(round((epoch_i0 - (target_tr.start_epoch + target_tr.display_offset)) / dt))

    for idx, lag in enumerate(lags_samps):
        j_start = j0_center + int(lag)
        j_end = j_start + win_len
        if j_start < 0 or j_end > target_tr.length:
            continue
        y = target_tr.data[j_start:j_end] - np.mean(target_tr.data[j_start:j_end])
        norm_y = float(np.linalg.norm(y))
        if norm_y > 1e-30:
            corrs[idx] = float(np.dot(x, y)) / (norm_x * norm_y)

    best_idx = int(np.argmax(corrs))
    best_lag = float(lags_sec[best_idx])
    best_val = float(corrs[best_idx])

    # Parabolic sub-sample refinement around best_idx
    if 0 < best_idx < corrs.size - 1:
        y_m, y_0, y_p = corrs[best_idx - 1], corrs[best_idx], corrs[best_idx + 1]
        denom = y_m - 2.0 * y_0 + y_p
        if denom < -1e-12:
            delta_s = 0.5 * (y_m - y_p) / denom
            best_lag += delta_s * dt

    return lags_sec, corrs, best_lag, best_val


def corrpick_array(
    traces: List[Trace],
    ref_trace_idx: int = 0,
    phase_name: str = "P",
    win_pre_sec: float = 1.0,
    win_post_sec: float = 3.0,
    max_lag_sec: float = 2.5,
    min_corr: float = 0.5,
) -> List[PhasePick]:
    """Automatically pick phase_name across all array traces by cross-correlating against ref_trace_idx."""
    if not traces or not (0 <= ref_trace_idx < len(traces)):
        return []
    ref_tr = traces[ref_trace_idx]
    ref_pk = ref_tr.get_pick(phase_name)
    if ref_pk is None or not ref_pk.onset:
        return []

    ref_rel_t = ref_pk.onset_epoch - ref_tr.start_epoch
    t_lo = max(0.0, ref_rel_t - win_pre_sec)
    t_hi = min(ref_tr.duration, ref_rel_t + win_post_sec)

    created: List[PhasePick] = []
    for idx, tr in enumerate(traces):
        if idx == ref_trace_idx:
            created.append(ref_pk)
            continue
        _lags, _corrs, best_lag, max_c = correlate_traces(ref_tr, tr, t_lo, t_hi, max_lag_sec=max_lag_sec)
        if max_c >= min_corr:
            new_pk = PhasePick(
                name=phase_name,
                onset=format_sh_time(ref_pk.onset_epoch + best_lag),
                spec=ref_pk.spec,
                sign=ref_pk.sign,
                source="a",
                filter_name=tr.filter_name,
                station=tr.station,
                comp=tr.comp,
                chan1=tr.chan1,
                chan2=tr.chan2,
            )
            new_pk.set_quality(min(10, max(2, int(round(max_c * 8)))))
            tr.add_or_replace_pick(new_pk)
            created.append(new_pk)
    return created


def pick_onset_baer_kradolfer(
    tr: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    thresh_sigma: float = 3.5,
) -> Optional[PhasePick]:
    """Automatic phase onset picker using the Baer & Kradolfer (1987) characteristic function
    matching pk_pick in shpick.c / seismics.c.
    """
    dt = tr.delta if tr.delta > 0 else 0.05
    if t_lo is not None and t_hi is not None:
        i0 = max(0, int(round(min(t_lo, t_hi) / dt)))
        i1 = min(tr.length, int(round(max(t_lo, t_hi) / dt)) + 1)
    else:
        i0, i1 = 0, tr.length

    seg = tr.data[i0:i1]
    if seg.size < 10:
        return None

    y = seg - np.mean(seg)
    dy = np.diff(y, prepend=y[0])
    y2_sum = float(np.sum(y * y))
    dy2_sum = float(np.sum(dy * dy))
    r_ratio = dy2_sum / y2_sum if y2_sum > 1e-30 else 1.0
    cf = (y * y) + (dy * dy) / max(1e-12, r_ratio)

    # Running mean & standard deviation of characteristic function
    n_noise = max(5, min(50, cf.size // 4))
    mu0 = float(np.mean(cf[:n_noise]))
    sig0 = max(1e-12, float(np.std(cf[:n_noise])))

    onset_idx = int(np.argmax(cf))
    for k in range(n_noise, cf.size):
        if (cf[k] - mu0) / sig0 >= thresh_sigma:
            onset_idx = k
            break

    # Determine first-motion polarity sign (+1 compression 'c', -1 dilatation 'd')
    look = y[onset_idx : min(y.size, onset_idx + 6)]
    ext_idx = int(np.argmax(np.abs(look))) if look.size > 0 else 0
    sign_val = 1 if (look.size > 0 and look[ext_idx] >= 0) else -1

    snr = math.sqrt(float(np.max(cf[onset_idx:])) / max(1e-12, mu0))
    onset_ep = tr.start_epoch + (i0 + onset_idx) * dt
    pk = PhasePick(
        name="P",
        onset=format_sh_time(onset_ep),
        spec="i" if snr > 5.0 else "e",
        sign=sign_val,
        source="a",
        snr=snr,
        filter_name=tr.filter_name,
        station=tr.station,
        comp=tr.comp,
        chan1=tr.chan1,
        chan2=tr.chan2,
    )
    pk.set_quality(min(8, max(2, int(round(min(8.0, snr))))))
    return pk


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
