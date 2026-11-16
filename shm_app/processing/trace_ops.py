"""Waveform trace operations, arithmetic, integration, differentiation, deconvolution,
Hilbert/envelope, mend, and synthetic trace generation matching shmenu*.c and shmath.c.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np

from shm_app.core.models import Trace


def demean_trace(tr: Trace, lo: Optional[float] = None, hi: Optional[float] = None) -> Trace:
    """Subtract the mean of tr (computed over [lo, hi] seconds relative to trace start, or full trace)."""
    out = tr.copy()
    if out.length == 0:
        return out
    seg = _slice_by_rel_time(out, lo, hi)
    mean_val = float(np.mean(seg)) if seg.size > 0 else 0.0
    out.data = out.data - mean_val
    return out


def detrend_trace(tr: Trace) -> Trace:
    """Remove least-squares linear trend y = a + b*x matching sh_trend in shmenu2.c."""
    out = tr.copy()
    n = out.length
    if n < 2:
        return out
    x = np.arange(n, dtype=np.float64)
    xm = 0.5 * (n - 1)
    ym = float(np.mean(out.data))
    dx = x - xm
    denom = float(np.dot(dx, dx))
    b = float(np.dot(dx, out.data - ym)) / denom if denom > 0.0 else 0.0
    a = ym - b * xm
    out.data = out.data - (a + b * x)
    return out


def despike_trace(tr: Trace, critmul: float = 5.0) -> Tuple[Trace, int, int]:
    """Remove single-sample spikes where |smp[i+1]-smp[i]| > critmul * mean_diff matching mt_despike."""
    out = tr.copy()
    smp = out.data
    lth = smp.size
    if lth < 3:
        return out, 0, 0

    diffs = np.abs(smp[1:] - smp[:-1])
    mean_diff = float(np.mean(diffs))
    if mean_diff <= 0.0:
        return out, 0, 0

    thresh = critmul * mean_diff
    isspike = True
    scnt = 0
    dcnt = 0
    for i in range(lth - 1):
        d = abs(smp[i + 1] - smp[i])
        if d > thresh:
            if isspike:
                smp[i] = 0.5 * (smp[i - 1] + smp[i + 1]) if i > 0 else smp[1]
                scnt += 1
                isspike = False
            else:
                isspike = True
        else:
            if isspike and i != 0:
                dcnt += 1
            isspike = False

    return out, scnt, dcnt


def resample_trace(tr: Trace, new_dt: float) -> Trace:
    """Linearly interpolate trace to new sample interval new_dt matching sh_resample."""
    if new_dt <= 0.0 or tr.length < 2:
        return tr.copy()
    out = tr.copy()
    dur = (tr.length - 1) * tr.delta
    new_len = max(1, int(math.floor(dur / new_dt)) + 1)
    old_t = np.arange(tr.length, dtype=np.float64) * tr.delta
    new_t = np.arange(new_len, dtype=np.float64) * new_dt
    out.data = np.interp(new_t, old_t, tr.data)
    out.delta = float(new_dt)
    return out


def decimate_trace(tr: Trace, factor: int, mean_window: bool = True) -> Trace:
    """Decimate trace by integer factor with optional moving-average anti-alias boxcar."""
    factor = max(1, int(factor))
    if factor == 1 or tr.length < factor:
        return tr.copy()
    out = tr.copy()
    n_blocks = tr.length // factor
    trimmed = tr.data[: n_blocks * factor].reshape(n_blocks, factor)
    if mean_window:
        out.data = np.mean(trimmed, axis=1)
    else:
        out.data = trimmed[:, 0].copy()
    out.delta = tr.delta * factor
    return out


def cut_trace(
    tr: Trace,
    t_lo: float,
    t_hi: float,
    relative_to_epoch: Optional[float] = None,
    pad_zeros: bool = True,
) -> Trace:
    """Cut trace to window [t_lo, t_hi] (seconds relative to trace start or relative_to_epoch)."""
    out = tr.copy()
    dt = out.delta
    if dt <= 0.0 or out.length == 0:
        return out

    base_shift = 0.0 if relative_to_epoch is None else (out.start_epoch + out.display_offset - relative_to_epoch)
    rel_lo = t_lo - base_shift
    rel_hi = t_hi - base_shift
    if rel_hi < rel_lo:
        rel_lo, rel_hi = rel_hi, rel_lo

    i0 = int(round(rel_lo / dt))
    i1 = int(round(rel_hi / dt))
    n_out = max(1, i1 - i0 + 1)

    if i0 >= 0 and i1 < out.length:
        out.data = out.data[i0 : i1 + 1].copy()
    elif pad_zeros:
        buf = np.zeros(n_out, dtype=np.float64)
        src_start = max(0, i0)
        src_end = min(out.length, i1 + 1)
        if src_end > src_start:
            dst_start = src_start - i0
            buf[dst_start : dst_start + (src_end - src_start)] = out.data[src_start:src_end]
        out.data = buf
    else:
        s0 = max(0, min(out.length - 1, i0))
        s1 = max(s0 + 1, min(out.length, i1 + 1))
        out.data = out.data[s0:s1].copy()
        i0 = s0

    out.start_epoch = out.start_epoch + i0 * dt
    return out


def integrate_trace(tr: Trace) -> Trace:
    """Trapezoidal numerical integration matching sh_int in shmenu2.c."""
    out = tr.copy()
    if out.length < 2:
        return out
    y = np.zeros_like(out.data)
    y[1:] = np.cumsum(0.5 * (out.data[:-1] + out.data[1:]) * out.delta)
    out.data = y
    return out


def differentiate_trace(tr: Trace, mode: int = 1) -> Trace:
    """Numerical differentiation matching sh_derive in shmenu2.c.

    mode=1: 5-point formula y'[n] = (2*(x[n+2]-x[n-2]) + x[n+1] - x[n-1]) / (10 * dt)
    mode=2: 2-point forward difference y'[n] = (x[n+1] - x[n]) / dt
    """
    out = tr.copy()
    n = out.length
    dt = out.delta
    if n < 2 or dt <= 0.0:
        return out

    x = out.data
    y = np.zeros_like(x)
    if mode == 1 and n >= 5:
        y[2 : n - 2] = (2.0 * (x[4:n] - x[0 : n - 4]) + (x[3 : n - 1] - x[1 : n - 3])) / (10.0 * dt)
        y[0] = (x[1] - x[0]) / dt
        y[1] = (x[2] - x[0]) / (2.0 * dt)
        y[-2] = (x[-1] - x[-3]) / (2.0 * dt)
        y[-1] = (x[-1] - x[-2]) / dt
    else:
        y[:-1] = (x[1:] - x[:-1]) / dt
        y[-1] = y[-2]
    out.data = y
    return out


def hilbert_transform(data: np.ndarray) -> np.ndarray:
    """90-degree phase-shifted Hilbert transform in the frequency domain matching sh_hilbert."""
    x = np.asarray(data, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()
    nfft = 1
    while nfft < n:
        nfft <<= 1
    spec = np.fft.rfft(x, n=nfft)
    # Multiply positive frequencies by -i (i.e. shift phase by -90 deg)
    h_spec = np.zeros_like(spec)
    h_spec[1:] = spec[1:] * (-1j)
    if nfft % 2 == 0:
        h_spec[-1] = 0.0
    return np.fft.irfft(h_spec, n=nfft)[:n]


def hilbert_trace(tr: Trace) -> Trace:
    """Return a copy of tr containing its Hilbert transform."""
    out = tr.copy()
    out.data = hilbert_transform(tr.data)
    out.comp = "H"
    return out


def envelope_trace(tr: Trace) -> Trace:
    """Analytic signal envelope sqrt(x^2 + H(x)^2) matching sh_envelope."""
    out = tr.copy()
    hx = hilbert_transform(tr.data)
    out.data = np.hypot(tr.data, hx)
    out.comp = "E"
    return out


def instantaneous_phase_trace(tr: Trace) -> Trace:
    """Instantaneous phase atan2(H(x), x) in degrees."""
    out = tr.copy()
    hx = hilbert_transform(tr.data)
    out.data = np.degrees(np.arctan2(hx, tr.data))
    return out


def resample_traces_to_common_grid(
    traces: List[Trace],
    offsets_sec: Optional[Sequence[float]] = None,
    include_display_offset: bool = True,
    shift_grid_bounds: bool = True,
) -> Tuple[float, float, np.ndarray, List[np.ndarray]]:
    """Interpolate traces onto a shared uniform time grid.

    Returns (t0_epoch, dt_sec, grid_t, aligned_arrays).
    """
    if not traces:
        raise ValueError("Cannot build common time grid for empty trace list")
    dt = traces[0].delta
    eff_offsets = (
        [
            float(off) + (tr.display_offset if include_display_offset else 0.0)
            for tr, off in zip(traces, offsets_sec)
        ]
        if offsets_sec is not None
        else [tr.display_offset if include_display_offset else 0.0 for tr in traces]
    )
    starts = [tr.start_epoch + off for tr, off in zip(traces, eff_offsets)]
    if shift_grid_bounds:
        t0 = min(starts)
        t1 = max(s + tr.duration for tr, s in zip(traces, starts))
    else:
        t0 = min(tr.start_epoch for tr in traces)
        t1 = max(tr.start_epoch + tr.duration for tr in traces)
    n_samps = max(1, int(round((t1 - t0) / dt)) + 1)
    grid_t = t0 + np.arange(n_samps, dtype=np.float64) * dt

    aligned: List[np.ndarray] = []
    for tr, s in zip(traces, starts):
        tr_t = s + np.arange(tr.length, dtype=np.float64) * tr.delta
        aligned.append(np.interp(grid_t, tr_t, tr.data, left=0.0, right=0.0))
    return t0, dt, grid_t, aligned


def sum_traces(traces: List[Trace], normalize_weights: bool = True) -> Optional[Trace]:
    """Weighted stack of traces aligned on their current display time axis."""
    if not traces:
        return None
    ref = traces[0]
    t0, _dt, grid_t, aligned = resample_traces_to_common_grid(traces, include_display_offset=True)

    accum = np.zeros(grid_t.size, dtype=np.float64)
    w_sum = 0.0
    for tr, interp_y in zip(traces, aligned):
        w = tr.weight if tr.weight != 0.0 else 1.0
        accum += w * interp_y
        w_sum += abs(w)

    if normalize_weights and w_sum > 0.0:
        accum /= w_sum

    out = ref.copy(copy_picks=False)
    out.data = accum
    out.start_epoch = t0
    out.display_offset = 0.0
    out.station = "SUM"
    return out


def phase_weighted_stack(traces: List[Trace], power: float = 2.0) -> Optional[Trace]:
    """Schimmel & Paulssen (1997) / SHM phase-weighted stack using instantaneous phase coherence."""
    if not traces:
        return None
    ref = traces[0]
    t0, _dt, grid_t, aligned = resample_traces_to_common_grid(traces, include_display_offset=True)

    lin_stack = np.zeros(grid_t.size, dtype=np.float64)
    phase_stack = np.zeros(grid_t.size, dtype=np.complex128)
    m = len(traces)

    for y in aligned:
        hy = hilbert_transform(y)
        amp = np.hypot(y, hy)
        analytic_unit = np.where(amp > 1e-30, (y + 1j * hy) / amp, 0.0 + 0.0j)
        lin_stack += y
        phase_stack += analytic_unit

    lin_stack /= m
    coherence = (np.abs(phase_stack) / m) ** power

    out = ref.copy(copy_picks=False)
    out.data = lin_stack * coherence
    out.start_epoch = t0
    out.display_offset = 0.0
    out.station = "PWS"
    return out


def mend_trace(tr: Trace, idx_lo: int, idx_hi: int, order: int = 3) -> Trace:
    """Mend/interpolate a corrupted sample segment [idx_lo, idx_hi] using polynomial fit."""
    out = tr.copy()
    n = out.length
    i0 = max(1, min(n - 2, idx_lo))
    i1 = max(i0, min(n - 2, idx_hi))
    left_idx = np.arange(max(0, i0 - 10), i0)
    right_idx = np.arange(i1 + 1, min(n, i1 + 11))
    fit_x = np.concatenate([left_idx, right_idx]).astype(np.float64)
    fit_y = out.data[fit_x.astype(int)]
    if fit_x.size >= 2:
        deg = min(order, fit_x.size - 1)
        poly = np.polyfit(fit_x, fit_y, deg)
        target_x = np.arange(i0, i1 + 1, dtype=np.float64)
        out.data[i0 : i1 + 1] = np.polyval(poly, target_x)
    return out


def attenuate_trace(tr: Trace, t_star: float = 1.0) -> Trace:
    """Apply Carpenter/Futterman causal Q attenuation operator with parameter t* (seconds) matching sh_attenuate."""
    out = tr.copy()
    n = out.length
    if n == 0 or t_star <= 0.0:
        return out
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    spec = np.fft.rfft(out.data, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=out.delta)
    f_ref = 1.0
    att = np.ones_like(spec, dtype=np.complex128)
    pos = freqs > 0.0
    f_p = freqs[pos]
    amp = np.exp(-math.pi * f_p * t_star)
    phase = 2.0 * f_p * t_star * np.log(f_p / f_ref)
    att[pos] = amp * np.exp(1j * phase)
    out.data = np.fft.irfft(spec * att, n=nfft)[:n]
    return out


def spiking_filter_levinson(
    wavelet: np.ndarray,
    filter_len: int = 100,
    prewhitening: float = 0.01,
    delay_samples: int = 0,
) -> np.ndarray:
    """Compute Wiener spiking deconvolution filter using Toeplitz autocorrelation and Levinson recursion."""
    w = np.asarray(wavelet, dtype=np.float64)
    m = max(2, int(filter_len))
    # Autocorrelation r[0..m-1]
    full_corr = np.correlate(w, w, mode="full")
    mid = w.size - 1
    r = full_corr[mid : mid + m].copy()
    if r.size < m:
        r = np.pad(r, (0, m - r.size))
    r[0] *= 1.0 + max(0.0, prewhitening)
    if abs(r[0]) < 1e-30:
        return np.zeros(m, dtype=np.float64)

    # Cross-correlation g[0..m-1] with spike at delay_samples
    g = np.zeros(m, dtype=np.float64)
    for k in range(m):
        idx = delay_samples - k
        if 0 <= idx < w.size:
            g[k] = w[idx]

    # Solve Toeplitz system R * f = g via Levinson algorithm matching mt_levinson in shmath.c
    f = np.zeros(m, dtype=np.float64)
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)

    f[0] = g[0] / r[0]
    if m > 1:
        a[0] = r[1] / r[0]
    for i in range(1, m):
        gn = r[0] - float(np.dot(r[1 : i + 1], a[:i][::-1]))
        if abs(gn) < 1e-30:
            break
        z1 = (r[i + 1] if i + 1 < m else 0.0) - float(np.dot(r[1 : i + 1], a[:i]))
        z2 = g[i] - float(np.dot(r[1 : i + 1], f[:i]))
        a[i] = z1 / gn
        f[i] = z2 / gn
        if i >= 1:
            b[:i] = a[:i] - a[i] * a[:i][::-1]
            f[:i] = f[:i] - f[i] * a[:i][::-1]
            a[:i] = b[:i]

    return f


def spectral_division_deconvolve(
    numerator_tr: Trace,
    denominator_tr: Trace,
    water_level: float = 0.01,
    gauss_width: float = 2.5,
) -> Trace:
    """Water-level frequency-domain deconvolution (receiver function / source time function)."""
    out = numerator_tr.copy()
    n = out.length
    nfft = 1
    while nfft < 2 * max(n, denominator_tr.length):
        nfft <<= 1

    num_f = np.fft.rfft(numerator_tr.data, n=nfft)
    den_f = np.fft.rfft(denominator_tr.data, n=nfft)
    den_pow = np.abs(den_f) ** 2
    wl = max(1e-12, water_level * float(np.max(den_pow)))
    denom_reg = np.maximum(den_pow, wl)

    freqs = np.fft.rfftfreq(nfft, d=out.delta)
    omega = 2.0 * math.pi * freqs
    gauss = np.exp(-(omega ** 2) / (4.0 * max(0.1, gauss_width) ** 2))

    deconv_f = (num_f * np.conj(den_f) / denom_reg) * gauss
    out.data = np.fft.irfft(deconv_f, n=nfft)[:n]
    return out


def create_synthetic_trace(
    kind: str,
    length_sec: float = 60.0,
    dt: float = 0.05,
    pos_sec: float = 10.0,
    ampl: float = 100.0,
    width: float = 1.0,
    freq_hz: float = 1.0,
    phase_deg: float = 0.0,
    order_m: int = 2,
) -> Trace:
    """Create synthetic test traces matching mt_cspike, mt_cgauss, mt_cexp, mt_csharp, mt_ckuepper, mt_csin, mt_crandom."""
    lth = max(1, int(round(length_sec / dt)))
    t = np.arange(lth, dtype=np.float64) * dt
    k = kind.strip().upper()
    a = np.zeros(lth, dtype=np.float64)

    if k == "SPIKE":
        idx = max(0, min(lth - 1, int(round(pos_sec / dt))))
        a[idx] = ampl
    elif k == "GAUSS":
        w = max(dt, width)
        a = ampl * np.exp(-((t - pos_sec) ** 2) / (2.0 * w * w))
    elif k == "EXP":
        w = max(dt, width)
        mask = t >= pos_sec
        a[mask] = ampl * np.exp(-(t[mask] - pos_sec) / w)
    elif k == "SHARP":
        w0 = max(0.01, width)
        w1 = 2.0 * math.pi * max(0.05, freq_hz)
        mask = t >= pos_sec
        tau = t[mask] - pos_sec
        raw = np.exp(-w0 * tau) * np.sin(w1 * tau) / w1
        mx = float(np.max(np.abs(raw))) if raw.size > 0 else 1.0
        a[mask] = raw * (ampl / mx if mx > 0 else 1.0)
    elif k == "KUEPPER":
        w = max(dt, width)
        m = max(1, int(order_m))
        mask = (t >= pos_sec) & (t <= pos_sec + w)
        tau = t[mask] - pos_sec
        raw = np.sin(m * math.pi / w * tau) - (m / (m + 2.0)) * np.sin((m + 2.0) * math.pi / w * tau)
        mx = float(np.max(np.abs(raw))) if raw.size > 0 else 1.0
        a[mask] = raw * (ampl / mx if mx > 0 else 1.0)
    elif k in ("SIN", "SINE"):
        ph = math.radians(phase_deg)
        a = ampl * np.sin(2.0 * math.pi * freq_hz * t + ph)
    elif k in ("RAND", "RANDOM"):
        rng = np.random.default_rng(42)
        a = rng.uniform(-ampl, ampl, size=lth)
    else:
        raise ValueError(f"Unknown synthetic trace kind: {kind}")

    return Trace(
        data=a,
        delta=dt,
        station=f"SYN{k[:2]}",
        comp="Z",
        chan1="B",
        chan2="H",
        start_epoch=1067739840.0,  # 02-NOV-2003_02:24:00.000
    )


def _slice_by_rel_time(tr: Trace, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if lo is None or hi is None or tr.delta <= 0:
        return tr.data
    i0 = max(0, int(round(lo / tr.delta)))
    i1 = min(tr.length, int(round(hi / tr.delta)) + 1)
    return tr.data[i0:i1] if i1 > i0 else tr.data
