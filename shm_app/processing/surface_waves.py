"""Surface-wave (18-22 s Rayleigh) bandpass measurement and Prague Ms estimation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from shm_app.core.models import Trace
from shm_app.processing.filters import restitute_ground_displacement_nm


def _fft_zero_phase_bandpass(
    data: np.ndarray,
    dt: float,
    f_lo: float,
    f_hi: float,
    taper_frac: float = 0.05,
) -> np.ndarray:
    """Pure-NumPy zero-phase flat-top raised-cosine FFT bandpass filter."""
    x = np.asarray(data, dtype=np.float64)
    n = x.size
    if n < 8 or dt <= 0.0:
        return x.copy()

    # Linear detrend to prevent long-period baseline drift leakage into 18-22 s band
    idx = np.arange(n, dtype=np.float64)
    idx_mean = 0.5 * (n - 1)
    x_mean = float(np.mean(x))
    denom = float(np.sum((idx - idx_mean) ** 2))
    slope = float(np.sum((idx - idx_mean) * (x - x_mean)) / denom) if denom > 0.0 else 0.0
    x_dt = (x - x_mean) - slope * (idx - idx_mean)

    if taper_frac > 0.0 and n > 10:
        m = max(1, int(round(n * taper_frac)))
        win = 0.5 * (1.0 - np.cos(math.pi * np.arange(m, dtype=np.float64) / m))
        x_dt[:m] *= win
        x_dt[-m:] *= win[::-1]

    nfft = 8192
    while nfft < 2 * n:
        nfft <<= 1

    freqs = np.fft.rfftfreq(nfft, d=dt)
    nyq = 0.5 / dt
    fl_base = max(1e-5, min(f_lo, f_hi))
    fh_base = min(nyq * 0.96, max(f_lo, f_hi))
    fl = 0.97 * fl_base
    fh = min(nyq * 0.98, 1.03 * fh_base)
    fl_stop = 0.70 * fl_base
    fh_stop = min(nyq, 1.30 * fh_base)

    resp = np.zeros_like(freqs, dtype=np.float64)
    pass_mask = (freqs >= fl) & (freqs <= fh)
    resp[pass_mask] = 1.0

    lo_skirt = (freqs >= fl_stop) & (freqs < fl)
    if np.any(lo_skirt) and fl > fl_stop:
        resp[lo_skirt] = 0.5 * (1.0 - np.cos(math.pi * (freqs[lo_skirt] - fl_stop) / (fl - fl_stop)))

    hi_skirt = (freqs > fh) & (freqs <= fh_stop)
    if np.any(hi_skirt) and fh_stop > fh:
        resp[hi_skirt] = 0.5 * (1.0 + np.cos(math.pi * (freqs[hi_skirt] - fh) / (fh_stop - fh)))

    resp[0] = 0.0
    spec = np.fft.rfft(x_dt, n=nfft) * resp
    return np.fft.irfft(spec, n=nfft)[:n]


def _parabolic_extremum(seg: np.ndarray, idx: int) -> Tuple[float, float]:
    """Sub-sample parabolic refinement for an interior extremum index."""
    if idx <= 0 or idx >= seg.size - 1:
        return float(idx), float(seg[idx])
    y0, y1, y2 = float(seg[idx - 1]), float(seg[idx]), float(seg[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(idx), y1
    shift = 0.5 * (y0 - y2) / denom
    shift = max(-0.5, min(0.5, shift))
    val = y1 - 0.25 * (y0 - y2) * shift
    return float(idx) + shift, float(val)


def find_surface_wave_amplitude(
    tr: Trace,
    distance_deg: Optional[float] = None,
    period_lo: float = 18.0,
    period_hi: float = 22.0,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    input_type: str = "displacement",
) -> Dict[str, Any]:
    """M-22: 18-22 s surface wave amplitude finder and Prague Ms estimator.

    Bandpass-filters the trace in pure NumPy to [1/period_hi, 1/period_lo] Hz
    (default 0.04545 - 0.05556 Hz), locates the maximum peak-to-peak half-cycle
    (and zero-to-peak amplitude), estimates dominant period T in [period_lo, period_hi],
    and computes Prague Ms = log10(A_um / T) + 1.66 * log10(Delta_deg) + 3.3
    matching ``array_loc.compute_magnitude_ms`` unit conventions (ampl_nm in nm).
    """
    if not math.isfinite(tr.delta) or tr.delta <= 0.0 or tr.length < 8:
        raise ValueError("Trace must have positive finite delta and at least 8 samples.")

    if (
        not (math.isfinite(period_lo) and math.isfinite(period_hi))
        or float(period_lo) <= 0.0
        or float(period_hi) <= 0.0
    ):
        raise ValueError(
            f"period_lo and period_hi must be positive and finite, got ({period_lo}, {period_hi})"
        )

    p_lo = min(float(period_lo), float(period_hi))
    p_hi = max(float(period_lo), float(period_hi))
    f_lo = 1.0 / p_hi
    f_hi = 1.0 / p_lo

    raw_data = np.asarray(tr.data, dtype=np.float64)
    finite_mask = np.isfinite(raw_data)
    if not np.any(finite_mask):
        clean_data = np.zeros_like(raw_data)
    elif not np.all(finite_mask):
        idx_all = np.arange(raw_data.size, dtype=np.float64)
        clean_data = np.interp(idx_all, idx_all[finite_mask], raw_data[finite_mask])
    else:
        clean_data = raw_data

    filtered = _fft_zero_phase_bandpass(clean_data, tr.delta, f_lo, f_hi)

    if t_lo is not None and t_hi is not None:
        i0 = max(0, int(round(min(t_lo, t_hi) / tr.delta)))
        i1 = min(tr.length, int(round(max(t_lo, t_hi) / tr.delta)) + 1)
        if i1 <= i0 + 4:
            i0, i1 = 0, tr.length
    else:
        i0, i1 = 0, tr.length

    seg = filtered[i0:i1]
    n_seg = seg.size

    # Collect ordered local extrema (alternating peaks and troughs)
    extrema: List[Tuple[int, int]] = []  # (index_in_seg, +1 for max / -1 for min)
    for k in range(1, n_seg - 1):
        if seg[k] >= seg[k - 1] and seg[k] > seg[k + 1]:
            extrema.append((k, 1))
        elif seg[k] <= seg[k - 1] and seg[k] < seg[k + 1]:
            extrema.append((k, -1))

    best_p2p = -1.0
    best_pair: Optional[Tuple[int, int]] = None
    for idx_e in range(len(extrema) - 1):
        ia, ta = extrema[idx_e]
        ib, tb = extrema[idx_e + 1]
        if ta != tb:
            p2p = abs(float(seg[ia]) - float(seg[ib]))
            if p2p > best_p2p:
                best_p2p = p2p
                best_pair = (ia, ib)

    if best_pair is not None:
        ia, ib = best_pair
        fa, va = _parabolic_extremum(seg, ia)
        fb, vb = _parabolic_extremum(seg, ib)
        ampl_p2p_raw = abs(va - vb)
        ampl_half_p2p_raw = 0.5 * ampl_p2p_raw
        half_per = abs(fb - fa) * tr.delta
        period_raw = 2.0 * half_per
        # Choose larger-magnitude extremum as amplitude measurement time
        f_ext = fa if abs(va) >= abs(vb) else fb
        t_peak_s = (i0 + (fa if va >= vb else fb)) * tr.delta
        t_trough_s = (i0 + (fb if va >= vb else fa)) * tr.delta
    else:
        idx_max = int(np.argmax(seg))
        idx_min = int(np.argmin(seg))
        vmax = float(seg[idx_max])
        vmin = float(seg[idx_min])
        ampl_p2p_raw = abs(vmax - vmin)
        ampl_half_p2p_raw = 0.5 * ampl_p2p_raw
        half_per = abs(idx_max - idx_min) * tr.delta
        period_raw = 2.0 * half_per if half_per > 0.0 else 0.5 * (p_lo + p_hi)
        f_ext = float(idx_max if abs(vmax) >= abs(vmin) else idx_min)
        t_peak_s = (i0 + idx_max) * tr.delta
        t_trough_s = (i0 + idx_min) * tr.delta

    ampl_z2p_raw = float(np.max(np.abs(seg))) if n_seg > 0 else 0.0

    # Cross-check with adjacent zero-crossings around f_ext if half-cycle spacing is out of band
    if not (0.8 * p_lo <= period_raw <= 1.2 * p_hi):
        zc_indices: List[float] = []
        for k in range(n_seg - 1):
            if (seg[k] <= 0.0 < seg[k + 1]) or (seg[k] >= 0.0 > seg[k + 1]):
                frac = abs(seg[k]) / max(1e-18, abs(seg[k]) + abs(seg[k + 1]))
                zc_indices.append(k + frac)
        left_zc = [z for z in zc_indices if z <= f_ext]
        right_zc = [z for z in zc_indices if z >= f_ext]
        if left_zc and right_zc:
            zc_period = 2.0 * (right_zc[0] - left_zc[-1]) * tr.delta
            if p_lo * 0.8 <= zc_period <= p_hi * 1.2:
                period_raw = zc_period

    period_est = max(p_lo, min(p_hi, float(period_raw)))
    time_sec = (i0 + f_ext) * tr.delta

    is_vel = (input_type or "displacement").strip().lower().startswith("v")
    if is_vel:
        ampl_nm = restitute_ground_displacement_nm(ampl_half_p2p_raw, period_est, tr.filter_name)
        ampl_p2p_nm = restitute_ground_displacement_nm(ampl_p2p_raw, period_est, tr.filter_name)
        ampl_z2p_nm = restitute_ground_displacement_nm(ampl_z2p_raw, period_est, tr.filter_name)
    else:
        ampl_nm = float(ampl_half_p2p_raw)
        ampl_p2p_nm = float(ampl_p2p_raw)
        ampl_z2p_nm = float(ampl_z2p_raw)

    if not math.isfinite(ampl_nm):
        ampl_nm = 0.0
    if not math.isfinite(ampl_p2p_nm):
        ampl_p2p_nm = 0.0
    if not math.isfinite(ampl_z2p_nm):
        ampl_z2p_nm = 0.0

    ampl_um = ampl_nm / 1000.0
    dist = float(distance_deg) if distance_deg is not None else (float(tr.distance) if tr.distance > 0.0 else None)

    ms_val: Optional[float] = None
    if (
        dist is not None
        and math.isfinite(dist)
        and dist > 0.0
        and math.isfinite(ampl_nm)
        and ampl_nm > 0.0
        and math.isfinite(period_est)
        and period_est > 0.0
    ):
        ms_val = math.log10(ampl_um / period_est) + 1.66 * math.log10(dist) + 3.3

    return {
        "station": tr.station,
        "component": tr.comp,
        "ampl_nm": float(ampl_nm),
        "ampl_um": float(ampl_um),
        "ampl_p2p_nm": float(ampl_p2p_nm),
        "ampl_z2p_nm": float(ampl_z2p_nm),
        "period": float(period_est),
        "period_raw": float(period_raw),
        "time_sec": float(time_sec),
        "t_peak_sec": float(t_peak_s),
        "t_trough_sec": float(t_trough_s),
        "distance_deg": dist,
        "ms": ms_val,
        "band_hz": (float(f_lo), float(f_hi)),
    }


def scan_surface_waves(
    traces: Sequence[Trace],
    distances_deg: Optional[Any] = None,
    period_lo: float = 18.0,
    period_hi: float = 22.0,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    vertical_only: bool = True,
    input_type: str = "displacement",
) -> Dict[str, Any]:
    """Batch M-22 helper scanning multiple traces for 18-22 s surface-wave amplitudes and network Ms."""
    if (
        not (math.isfinite(period_lo) and math.isfinite(period_hi))
        or float(period_lo) <= 0.0
        or float(period_hi) <= 0.0
    ):
        raise ValueError(
            f"period_lo and period_hi must be positive and finite, got ({period_lo}, {period_hi})"
        )

    tr_list = list(traces)
    if vertical_only:
        z_traces = [tr for tr in tr_list if (tr.comp or "").upper() == "Z"]
        if z_traces:
            tr_list = z_traces

    results: List[Dict[str, Any]] = []
    ms_vals: List[float] = []

    for idx, tr in enumerate(tr_list):
        if isinstance(distances_deg, (list, tuple, np.ndarray)):
            dist = float(distances_deg[idx]) if idx < len(distances_deg) else None
        elif distances_deg is not None:
            dist = float(distances_deg)
        else:
            dist = float(tr.distance) if tr.distance > 0.0 else None

        try:
            res = find_surface_wave_amplitude(
                tr,
                distance_deg=dist,
                period_lo=period_lo,
                period_hi=period_hi,
                t_lo=t_lo,
                t_hi=t_hi,
                input_type=input_type,
            )
        except ValueError as exc:
            p_lo_eff = min(float(period_lo), float(period_hi))
            p_hi_eff = max(float(period_lo), float(period_hi))
            results.append(
                {
                    "station": tr.station,
                    "component": tr.comp,
                    "ampl_nm": 0.0,
                    "ampl_um": 0.0,
                    "ampl_p2p_nm": 0.0,
                    "ampl_z2p_nm": 0.0,
                    "period": 0.0,
                    "period_raw": 0.0,
                    "time_sec": 0.0,
                    "t_peak_sec": 0.0,
                    "t_trough_sec": 0.0,
                    "distance_deg": dist,
                    "ms": None,
                    "band_hz": (1.0 / p_hi_eff, 1.0 / p_lo_eff),
                    "skipped": True,
                    "error": str(exc),
                }
            )
            continue

        results.append(res)
        if res["ms"] is not None and math.isfinite(res["ms"]):
            ms_vals.append(float(res["ms"]))

    return {
        "stations": results,
        "count": len(results),
        "ms_mean": float(np.mean(ms_vals)) if ms_vals else None,
        "ms_std": float(np.std(ms_vals, ddof=1)) if len(ms_vals) > 1 else (0.0 if ms_vals else None),
        "ms_median": float(np.median(ms_vals)) if ms_vals else None,
    }
