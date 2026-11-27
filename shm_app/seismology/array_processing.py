"""Array processing: plane-wave fit, beamforming, alignment, vespagram, wideband FK, and ARF."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np

logger = logging.getLogger(__name__)

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import Trace
from shm_app.processing.trace_ops import resample_traces_to_common_grid
from shm_app.seismology.calibration import (
    SlownessCalibrationTable,
    _resolve_calib_table,
)
from shm_app.seismology.diagnostics import (
    compute_mad,
    compute_reduced_chi2,
    compute_rmse,
    compute_robust_sigma,
)

DEG_TO_KM = 111.1949266

def plane_wave_locate(
    arrival_times: List[float],
    x_km: List[float],
    y_km: List[float],
    elev_m: Optional[List[float]] = None,
    vp_kms: Optional[List[float]] = None,
    dt: float = 0.05,
    calib_table: Optional[SlownessCalibrationTable] = None,
    apply_calibration: Optional[bool] = None,
    stations: Optional[Sequence[str]] = None,
    config: Optional[SHMConfig] = None,
) -> Dict[str, Any]:
    """Fit plane-wave backazimuth (deg) and slowness (s/deg) with uncertainties and station residuals
    matching mt_locate in shmath.c.
    """
    n = len(arrival_times)
    if n < 3:
        raise ValueError("Plane-wave location requires at least 3 station arrival picks")
    if len(x_km) != n or len(y_km) != n:
        raise ValueError(
            f"Length mismatch in plane_wave_locate: arrival_times={n}, x_km={len(x_km)}, y_km={len(y_km)}"
        )
    if elev_m is not None and len(elev_m) != n:
        raise ValueError(f"Length mismatch in plane_wave_locate: elev_m={len(elev_m)} != {n}")
    if vp_kms is not None and len(vp_kms) != n:
        raise ValueError(f"Length mismatch in plane_wave_locate: vp_kms={len(vp_kms)} != {n}")

    t_in = np.asarray(arrival_times, dtype=np.float64).copy()
    x = np.asarray(x_km, dtype=np.float64).copy()
    y = np.asarray(y_km, dtype=np.float64).copy()
    if t_in.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("arrival_times, x_km, and y_km must be 1D arrays")
    if not (np.all(np.isfinite(t_in)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("arrival_times, x_km, and y_km must contain only finite values")
    if elev_m is not None:
        el_check = np.asarray(elev_m, dtype=np.float64)
        if el_check.ndim != 1 or not np.all(np.isfinite(el_check)):
            raise ValueError("elev_m must be a finite 1D sequence")
    if vp_kms is not None:
        vp_check = np.asarray(vp_kms, dtype=np.float64)
        if vp_check.ndim != 1 or not np.all(np.isfinite(vp_check)):
            raise ValueError("vp_kms must be a finite 1D sequence")
    dt_eff = float(dt) if (dt is not None and math.isfinite(float(dt)) and float(dt) > 0.0) else 0.05

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
            rf = inv_norm * dt_eff / abs_det
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

    raw_az = float(azimuth % 360.0)
    raw_slo = float(slowness)
    out: Dict[str, Any] = {
        "azimuth": raw_az,
        "d_azimuth": float(azimerr),
        "slowness": raw_slo,
        "d_slowness": float(slowerr),
        "residuals": [float(r) for r in residuals],
        "sx": float(sx),
        "sy": float(sy),
        "calib_applied": False,
    }

    active_table = _resolve_calib_table(calib_table, apply_calibration, stations=stations, config=config)
    if active_table is not None:
        cal_slo, cal_az, d_slo, d_az = active_table.calibrate(raw_slo, raw_az)
        cal_az_r = math.radians(cal_az)
        cal_s_km = cal_slo / DEG_TO_KM
        cal_sx = -cal_s_km * math.sin(cal_az_r)
        cal_sy = -cal_s_km * math.cos(cal_az_r)
        out.update(
            {
                "azimuth": cal_az,
                "slowness": cal_slo,
                "sx": float(cal_sx),
                "sy": float(cal_sy),
                "calib_applied": True,
                "raw_azimuth": raw_az,
                "raw_slowness": raw_slo,
                "raw_sx": float(sx),
                "raw_sy": float(sy),
                "cal_azimuth": cal_az,
                "cal_slowness": cal_slo,
                "cal_sx": float(cal_sx),
                "cal_sy": float(cal_sy),
                "calib_d_slowness": float(d_slo),
                "calib_d_azimuth": float(d_az),
            }
        )
    return out


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
    if not traces:
        raise ValueError("Cannot compute vespagram from empty trace list")

    cfg = config or get_config()
    stations = [t.station for t in traces]
    rel_coords = cfg.relative_xy_km(stations, ref_station=ref_station)
    x_km = [c[0] for c in rel_coords]
    y_km = [c[1] for c in rel_coords]
    elev_m = [c[2] for c in rel_coords]
    vp_kms = [c[3] for c in rel_coords]

    ref = traces[0]
    dt = ref.delta if ref.delta > 0.0 else 0.05
    t0 = min(tr.start_epoch for tr in traces)
    t1 = max(tr.start_epoch + tr.duration for tr in traces)
    n_samps = max(1, int(round((t1 - t0) / dt)) + 1)
    grid_t = t0 + np.arange(n_samps, dtype=np.float64) * dt
    tr_t0_list = [tr.start_epoch + np.arange(tr.length, dtype=np.float64) * (tr.delta if tr.delta > 0.0 else dt) for tr in traces]

    p = max(1.0, float(power))
    n_traces = len(traces)

    slownesses = np.arange(slow_lo, slow_hi + 0.5 * slow_step, max(0.05, slow_step), dtype=np.float64)
    vespa_traces: List[Trace] = []
    rows: List[np.ndarray] = []
    total_steps = len(slownesses)

    for idx, slo in enumerate(slownesses):
        if progress_callback is not None:
            if not progress_callback(idx, total_steps):
                return vespa_traces, np.array([]), slownesses[:idx], np.zeros((1, 1))

        delays = beam_delays_sec(x_km, y_km, azimuth_deg, float(slo), elev_m=elev_m, vp_kms=vp_kms)
        stack = np.zeros(grid_t.size, dtype=np.float64)
        for tr, tr_t0, d_sec in zip(traces, tr_t0_list, delays):
            y = np.interp(grid_t, tr_t0 + float(d_sec), tr.data, left=0.0, right=0.0)
            if p > 1.0:
                stack += np.sign(y) * (np.abs(y) ** (1.0 / p))
            else:
                stack += y
        stack /= n_traces
        if p > 1.0:
            stack = np.sign(stack) * (np.abs(stack) ** p)

        b_tr = ref.copy(copy_picks=False)
        b_tr.data = stack
        b_tr.start_epoch = t0
        b_tr.display_offset = 0.0
        b_tr.station = f"V{slo:4.1f}"
        b_tr.azimuth = azimuth_deg
        b_tr.slowness = float(slo)
        b_tr.comment = f"VESPA az={azimuth_deg:.1f} slo={slo:.2f} ref={ref_station}"
        vespa_traces.append(b_tr)
        rows.append(b_tr.data)

    if progress_callback is not None:
        progress_callback(total_steps, total_steps)

    mat = np.vstack(rows) if rows else np.zeros((1, 1))
    times = np.arange(mat.shape[1], dtype=np.float64) * dt
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
    calib_table: Optional[SlownessCalibrationTable] = None,
    apply_calibration: Optional[bool] = None,
) -> Dict[str, Any]:
    """Wideband coherent Frequency-Wavenumber (FK) analysis with 9-point 2D quadratic sub-grid
    peak interpolation matching fkwid.c.
    """
    if len(traces) < 3:
        raise ValueError("FK analysis requires at least 3 array traces")

    cfg = config or get_config()
    stations = [t.station for t in traces]
    rel_coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
    x_deg = np.array([c[0] / DEG_TO_KM for c in rel_coords], dtype=np.float64)
    y_deg = np.array([c[1] / DEG_TO_KM for c in rel_coords], dtype=np.float64)

    # Align all traces onto a common time grid (accounting for staggered start_epoch and delta)
    norm_traces: List[Trace] = []
    default_dt = next((t.delta for t in traces if t.delta > 0.0), 0.05)
    for t in traces:
        if t.delta > 0.0:
            norm_traces.append(t)
        else:
            tc = t.copy(copy_picks=False)
            tc.delta = default_dt
            norm_traces.append(tc)

    t0, dt, grid_t, aligned = resample_traces_to_common_grid(
        norm_traces,
        include_display_offset=False,
        shift_grid_bounds=False,
    )
    if dt <= 0.0:
        dt = default_dt

    t_rel_full = grid_t - t0
    if t_lo is not None and t_hi is not None:
        lo_v = min(float(t_lo), float(t_hi))
        hi_v = max(float(t_lo), float(t_hi))
        # Support either relative seconds (from t0) or absolute POSIX epoch seconds
        t_cmp = grid_t if (lo_v > 1e7 and t0 > 1e7) else t_rel_full
        mask = (t_cmp >= lo_v - 1e-9) & (t_cmp <= hi_v + 1e-9)
        if int(np.count_nonzero(mask)) > 4:
            aligned_segs = [a[mask] for a in aligned]
        else:
            aligned_segs = aligned
    else:
        aligned_segs = aligned

    segs: List[np.ndarray] = [s - np.mean(s) for s in aligned_segs]

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

    out: Dict[str, Any] = {
        "azimuth": baz_deg,
        "slowness": slowness,
        "apparent_velocity_kms": app_vel,
        "semblance": peak_power,
        "power_db": power_db,
        "sx": sx_peak,
        "sy": sy_peak,
        "s_axis": s_axis,
        "grid": semblance_grid,
        "calib_applied": False,
    }

    active_table = _resolve_calib_table(calib_table, apply_calibration, stations=stations, config=cfg)
    if active_table is not None:
        cal_slo, cal_az, d_slo, d_az = active_table.calibrate(slowness, baz_deg)
        cal_az_r = math.radians(cal_az)
        cal_sx = cal_slo * math.sin(cal_az_r)
        cal_sy = cal_slo * math.cos(cal_az_r)
        cal_app_vel = (DEG_TO_KM / cal_slo) if cal_slo > 1e-6 else 999.9
        out.update(
            {
                "azimuth": cal_az,
                "slowness": cal_slo,
                "apparent_velocity_kms": cal_app_vel,
                "sx": float(cal_sx),
                "sy": float(cal_sy),
                "calib_applied": True,
                "raw_azimuth": baz_deg,
                "raw_slowness": slowness,
                "raw_sx": sx_peak,
                "raw_sy": sy_peak,
                "cal_azimuth": cal_az,
                "cal_slowness": cal_slo,
                "cal_sx": float(cal_sx),
                "cal_sy": float(cal_sy),
                "calib_d_slowness": float(d_slo),
                "calib_d_azimuth": float(d_az),
            }
        )
    return out


# ---------------------------------------------------------------------------
# M-15: Beamforming Channel Residuals, Coherence & 2D Array Response Function
# ---------------------------------------------------------------------------

POOR_ELEMENT_CC_THRESHOLD = 0.70


def compute_beam_diagnostics(
    traces: Sequence[Trace],
    azimuth_deg: float,
    slowness_s_deg: float,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    ref_station: str = "CENTRE",
    config: Optional[SHMConfig] = None,
) -> Dict[str, Any]:
    """Compute aligned channel waveforms, beam ``b(t)``, channel residual waveforms
    ``r_m(t) = u_m(t + tau_m) - b(t)``, cross-correlation coefficients ``CC_m in [-1, 1]``,
    RMS residual envelope, and poor-element flags (``CC_m < 0.70``).
    """
    if not traces:
        raise ValueError("Cannot compute beam diagnostics from empty trace list")

    cfg = config or get_config()
    trace_list = list(traces)
    stations = [tr.station for tr in trace_list]
    rel_coords = cfg.relative_xy_km(stations, ref_station=ref_station)
    x_km = [c[0] for c in rel_coords]
    y_km = [c[1] for c in rel_coords]
    elev_m = [c[2] for c in rel_coords]
    vp_kms = [c[3] for c in rel_coords]

    delays = beam_delays_sec(x_km, y_km, azimuth_deg, slowness_s_deg, elev_m=elev_m, vp_kms=vp_kms)
    t0, _dt, grid_t, aligned = resample_traces_to_common_grid(
        trace_list,
        offsets_sec=delays,
        include_display_offset=False,
        shift_grid_bounds=False,
    )

    t_rel_full = grid_t - t0
    if t_lo is not None and t_hi is not None:
        lo_v = min(float(t_lo), float(t_hi))
        hi_v = max(float(t_lo), float(t_hi))
        # Support either relative seconds or absolute POSIX epoch seconds
        t_cmp = grid_t if (lo_v > 1e7 and t0 > 1e7) else t_rel_full
        mask = (t_cmp >= lo_v - 1e-9) & (t_cmp <= hi_v + 1e-9)
        if int(np.count_nonzero(mask)) >= 3:
            t_rel = t_rel_full[mask]
            aligned_segs = [a[mask] for a in aligned]
        else:
            t_rel = t_rel_full
            aligned_segs = aligned
    else:
        t_rel = t_rel_full
        aligned_segs = aligned

    mat = np.vstack(aligned_segs)  # (M, L)
    beam = np.mean(mat, axis=0)
    res_mat = mat - beam[None, :]
    mean_residual = np.sqrt(np.mean(res_mat ** 2, axis=0))

    b_zm = beam - np.mean(beam)
    norm_b = float(np.linalg.norm(b_zm))

    cc_list: List[float] = []
    channel_rms: List[float] = []
    channel_names: List[str] = []
    channel_ids: List[str] = []
    poor_indices: List[int] = []
    poor_channels: List[str] = []

    for idx, tr in enumerate(trace_list):
        u_zm = mat[idx] - np.mean(mat[idx])
        norm_u = float(np.linalg.norm(u_zm))
        if norm_u > 1e-30 and norm_b > 1e-30:
            cc_val = float(np.clip(np.dot(u_zm, b_zm) / (norm_u * norm_b), -1.0, 1.0))
        else:
            cc_val = 0.0
        rms_val = float(np.sqrt(np.mean(res_mat[idx] ** 2)))

        cc_list.append(cc_val)
        channel_rms.append(rms_val)
        channel_names.append(tr.station)
        channel_ids.append(f"{tr.station}.{tr.comp or 'Z'}")
        if cc_val < POOR_ELEMENT_CC_THRESHOLD:
            poor_indices.append(idx)
            poor_channels.append(tr.station)

    return {
        "beam_time": t_rel,
        "beam_waveform": beam,
        "aligned_waveforms": mat,
        "channel_residuals": res_mat,
        "mean_residual": mean_residual,
        "channel_rms": channel_rms,
        "channel_cc": cc_list,
        "channel_names": channel_names,
        "channel_ids": channel_ids,
        "poor_channels": poor_channels,
        "poor_indices": poor_indices,
        "mean_cc": float(np.mean(cc_list)) if cc_list else 0.0,
        "delays_sec": [float(d) for d in delays],
        "azimuth_deg": float(azimuth_deg % 360.0),
        "slowness_s_deg": float(slowness_s_deg),
    }


def compute_array_response_function(
    x_km: Sequence[float],
    y_km: Sequence[float],
    slowness_max: float = 20.0,
    n_grid: int = 101,
    freq_hz: Union[float, Sequence[float]] = 1.0,
) -> Dict[str, Any]:
    """Compute the 2D Array Response Function (ARF) on a symmetric ``(sx, sy)`` slowness grid:

        |A(k)|^2 = (1 / M^2) * | sum_m exp(i * k . x_m) |^2

    with ``k . x_m = 2 * pi * f * (sx * x_deg_m + sy * y_deg_m)``.
    """
    x_arr = np.asarray(x_km, dtype=np.float64)
    y_arr = np.asarray(y_km, dtype=np.float64)
    if x_arr.size == 0 or x_arr.size != y_arr.size:
        raise ValueError("x_km and y_km must be non-empty sequences of equal length")

    m_stat = x_arr.size
    x_deg = x_arr / DEG_TO_KM
    y_deg = y_arr / DEG_TO_KM

    g_size = max(3, int(n_grid))
    s_max = float(slowness_max)
    s_axis = np.linspace(-s_max, s_max, g_size, dtype=np.float64)
    sx_grid, sy_grid = np.meshgrid(s_axis, s_axis)

    tau = sx_grid[:, :, None] * x_deg[None, None, :] + sy_grid[:, :, None] * y_deg[None, None, :]

    if isinstance(freq_hz, (int, float)):
        freqs = [float(freq_hz)]
    else:
        freqs = [float(f) for f in freq_hz]
        if not freqs:
            freqs = [1.0]

    arf_sum = np.zeros((g_size, g_size), dtype=np.float64)
    inv_m2 = 1.0 / float(m_stat * m_stat)
    for f_val in freqs:
        omega = 2.0 * math.pi * f_val
        resp = np.sum(np.exp(1j * omega * tau), axis=2)
        arf_sum += (np.abs(resp) ** 2) * inv_m2

    arf_grid = arf_sum / float(len(freqs))
    arf_db = 10.0 * np.log10(np.maximum(1e-6, arf_grid))

    return {
        "s_axis": s_axis,
        "sx_grid": sx_grid,
        "sy_grid": sy_grid,
        "arf_grid": arf_grid,
        "arf_db": arf_db,
        "freq_hz": freq_hz,
        "slowness_max": s_max,
        "peak_value": float(np.max(arf_grid)),
    }


def build_fk_fit_diagnostic(
    fk_result: Dict[str, Any],
    beam_diag: Optional[Dict[str, Any]] = None,
    arf_result: Optional[Dict[str, Any]] = None,
    calib_info: Optional[Dict[str, Any]] = None,
    fit_type: FitType = FitType.FK_SEMBLANCE,
) -> FitDiagnosticResult:
    """Build a ``FitDiagnosticResult`` from FK/plane-wave results, beam diagnostics,
    2D ARF, and optional calibration metadata for ``FitDiagnosticWindow`` Tab 3.
    """
    src_cal = calib_info if calib_info is not None else fk_result
    calib_applied = bool(src_cal.get("calib_applied", False))

    cur_slo = float(fk_result.get("slowness", 0.0))
    cur_baz = float(fk_result.get("azimuth", 0.0)) % 360.0
    cur_sx = float(fk_result.get("sx", cur_slo * math.sin(math.radians(cur_baz))))
    cur_sy = float(fk_result.get("sy", cur_slo * math.cos(math.radians(cur_baz))))

    if calib_applied:
        raw_slo = float(src_cal.get("raw_slowness", cur_slo))
        raw_baz = float(src_cal.get("raw_azimuth", src_cal.get("raw_baz", cur_baz))) % 360.0
        raw_sx = float(src_cal.get("raw_sx", raw_slo * math.sin(math.radians(raw_baz))))
        raw_sy = float(src_cal.get("raw_sy", raw_slo * math.cos(math.radians(raw_baz))))
        cal_slo = float(src_cal.get("cal_slowness", cur_slo))
        cal_baz = float(src_cal.get("cal_azimuth", src_cal.get("cal_baz", cur_baz))) % 360.0
        cal_sx = float(src_cal.get("cal_sx", cal_slo * math.sin(math.radians(cal_baz))))
        cal_sy = float(src_cal.get("cal_sy", cal_slo * math.cos(math.radians(cal_baz))))
    else:
        raw_slo, raw_baz, raw_sx, raw_sy = cur_slo, cur_baz, cur_sx, cur_sy
        cal_slo, cal_baz, cal_sx, cal_sy = cur_slo, cur_baz, cur_sx, cur_sy

    residual_entries: List[ResidualEntry] = []
    channel_cc: List[float] = []
    channel_names: List[str] = []

    if beam_diag is not None:
        channel_cc = [float(c) for c in beam_diag.get("channel_cc", [])]
        channel_names = [str(n) for n in beam_diag.get("channel_names", [])]
        ch_rms_list = [float(r) for r in beam_diag.get("channel_rms", [0.0] * len(channel_cc))]
        for idx, (st_name, cc_val) in enumerate(zip(channel_names, channel_cc)):
            res_val = 1.0 - cc_val
            is_poor = cc_val < POOR_ELEMENT_CC_THRESHOLD
            ch_rms = ch_rms_list[idx] if idx < len(ch_rms_list) else 0.0
            residual_entries.append(
                ResidualEntry(
                    identifier=f"{st_name}.Z.BEAM",
                    observed=1.0,
                    calculated=cc_val,
                    residual=res_val,
                    normalized_residual=res_val / 0.30,
                    weight=0.0 if is_poor else 1.0,
                    uncertainty=0.30,
                    station=st_name,
                    phase="BEAM",
                    channel="Z",
                    is_outlier=is_poor,
                    extra_attributes={
                        "predicted": cc_val,
                        "cc": cc_val,
                        "channel_rms": ch_rms,
                    },
                )
            )

    res_vals = [e.residual for e in residual_entries] if residual_entries else [
        1.0 - float(fk_result.get("semblance", 1.0))
    ]
    rmse = compute_rmse(res_vals)
    mad_val = compute_mad(res_vals)
    outlier_count = sum(1 for e in residual_entries if e.is_outlier)
    dof = max(1, len(res_vals) - 2)
    chi2_red = compute_reduced_chi2(res_vals, uncertainties=[0.30] * len(res_vals), dof=dof)

    d_slo = float(fk_result.get("d_slowness", 0.5))
    d_baz = float(fk_result.get("d_azimuth", 3.0))
    params: Dict[str, ParameterUncertainty] = {
        "slowness": ParameterUncertainty(
            name="slowness",
            value=cal_slo if calib_applied else cur_slo,
            standard_error=d_slo,
            confidence_interval_95=(
                max(0.0, (cal_slo if calib_applied else cur_slo) - 1.96 * d_slo),
                (cal_slo if calib_applied else cur_slo) + 1.96 * d_slo,
            ),
            unit="s/deg",
        ),
        "backazimuth": ParameterUncertainty(
            name="backazimuth",
            value=cal_baz if calib_applied else cur_baz,
            standard_error=d_baz,
            confidence_interval_95=(
                ((cal_baz if calib_applied else cur_baz) - 1.96 * d_baz) % 360.0,
                ((cal_baz if calib_applied else cur_baz) + 1.96 * d_baz) % 360.0,
            ),
            unit="deg",
        ),
    }
    if "semblance" in fk_result:
        semb = float(fk_result["semblance"])
        params["semblance"] = ParameterUncertainty(
            name="semblance",
            value=semb,
            standard_error=0.0,
            confidence_interval_95=(semb, semb),
            unit="",
        )

    s_axis_arr = np.asarray(fk_result.get("s_axis", np.array([])), dtype=np.float64)
    fk_grid_arr = np.asarray(fk_result.get("grid", np.array([])), dtype=np.float64)
    arf_grid_arr = (
        np.asarray(arf_result.get("arf_grid", np.array([])), dtype=np.float64)
        if arf_result is not None
        else np.array([], dtype=np.float64)
    )
    if s_axis_arr.size == 0 and arf_result is not None and "s_axis" in arf_result:
        s_axis_arr = np.asarray(arf_result["s_axis"], dtype=np.float64)

    overlay_curves: Dict[str, np.ndarray] = {
        "s_axis": s_axis_arr,
        "fk_grid": fk_grid_arr,
        "arf_grid": arf_grid_arr,
        "beam_time": (
            np.asarray(beam_diag.get("beam_time", np.array([])), dtype=np.float64)
            if beam_diag is not None
            else np.array([], dtype=np.float64)
        ),
        "beam_waveform": (
            np.asarray(beam_diag.get("beam_waveform", np.array([])), dtype=np.float64)
            if beam_diag is not None
            else np.array([], dtype=np.float64)
        ),
        "mean_residual": (
            np.asarray(beam_diag.get("mean_residual", np.array([])), dtype=np.float64)
            if beam_diag is not None
            else np.array([], dtype=np.float64)
        ),
        "channel_residuals": (
            np.asarray(beam_diag.get("channel_residuals", np.array([])), dtype=np.float64)
            if beam_diag is not None
            else np.array([], dtype=np.float64)
        ),
    }

    context_data: Dict[str, Any] = {
        "calib_applied": calib_applied,
        "raw_sx": raw_sx,
        "raw_sy": raw_sy,
        "cal_sx": cal_sx,
        "cal_sy": cal_sy,
        "raw_slowness": raw_slo,
        "raw_baz": raw_baz,
        "cal_slowness": cal_slo,
        "cal_baz": cal_baz,
        "channel_cc": channel_cc,
        "channel_names": channel_names,
        "poor_channels": list(beam_diag.get("poor_channels", [])) if beam_diag is not None else [],
    }

    return FitDiagnosticResult(
        fit_type=fit_type,
        title="Array FK & Beamforming Diagnostics",
        converged=True,
        iterations=1,
        dof=dof,
        rmse=round(rmse, 4),
        wrmse=round(rmse, 4),
        mad=round(mad_val, 4),
        chi2_reduced=round(chi2_red, 4),
        parameters=params,
        residuals=residual_entries,
        outlier_count=outlier_count,
        overlay_curves=overlay_curves,
        context_data=context_data,
    )

