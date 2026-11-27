"""Spectral estimation (FFT, Bartlett/Welch Power Spectrum, Burg MEM Autoregressive Spectrum),
log-log spectral slope/plateau fitting, and sliding-window spectrogram matching
spectr.c and spcwork.c.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np

from shm_app.core.models import Trace
from shm_app.processing.filters import burg_ar_coeffs


def compute_fft_spectrum(
    tr: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    squared: bool = False,
    taper_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute single-sided FFT amplitude (or squared amplitude) spectrum matching spectr.c.

    Returns (freqs_hz, spectrum_values).
    """
    seg = _extract_segment(tr, t_lo, t_hi)
    n = seg.size
    if n < 2 or tr.delta <= 0:
        return np.array([0.0]), np.array([0.0])

    x = seg - np.mean(seg)
    if taper_frac > 0.0 and n > 8:
        m = max(1, int(round(n * taper_frac)))
        win = 0.5 * (1.0 - np.cos(math.pi * np.arange(m) / m))
        x[:m] *= win
        x[-m:] *= win[::-1]

    nfft = 1
    while nfft < n:
        nfft <<= 1

    spec = np.fft.rfft(x, n=nfft) * tr.delta
    freqs = np.fft.rfftfreq(nfft, d=tr.delta)
    amp = np.abs(spec)
    if squared:
        amp = amp * amp
    return freqs[1:], amp[1:]


def compute_welch_power_spectrum(
    tr: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    num_windows: int = 4,
    overlap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Bartlett-windowed overlapped Welch power spectral density matching nr_spctrm in spectr.c."""
    seg = _extract_segment(tr, t_lo, t_hi)
    n = seg.size
    if n < 2 or tr.delta <= 0:
        return np.array([0.0]), np.array([0.0])
    if n < 8:
        x_s = seg - np.mean(seg)
        nfft_s = 1
        while nfft_s < n:
            nfft_s <<= 1
        half_s = 0.5 * (n - 1)
        bartlett_s = 1.0 - np.abs((np.arange(n, dtype=np.float64) - half_s) / max(1.0, half_s))
        w_norm_s = max(1e-30, float(np.sum(bartlett_s ** 2)))
        spec_s = np.fft.rfft(x_s * bartlett_s, n=nfft_s)
        psd_s = (np.abs(spec_s) ** 2) / (w_norm_s * (1.0 / tr.delta))
        freqs_s = np.fft.rfftfreq(nfft_s, d=tr.delta)
        return freqs_s[1:], psd_s[1:]

    x = seg - np.mean(seg)
    k = max(1, int(num_windows))
    win_len = max(8, n // k)
    nfft = 1
    while nfft < win_len:
        nfft <<= 1

    step = max(1, win_len // 2) if overlap else win_len
    # Bartlett triangular window matching Numerical Recipes spctrm
    half = 0.5 * (win_len - 1)
    bartlett = 1.0 - np.abs((np.arange(win_len, dtype=np.float64) - half) / max(1.0, half))
    w_norm = float(np.sum(bartlett ** 2))

    psd_accum = np.zeros(nfft // 2 + 1, dtype=np.float64)
    count = 0
    for start in range(0, max(1, n - win_len + 1), step):
        chunk = x[start : start + win_len]
        if chunk.size < win_len:
            chunk = np.pad(chunk, (0, win_len - chunk.size))
        w_chunk = (chunk - np.mean(chunk)) * bartlett
        spec = np.fft.rfft(w_chunk, n=nfft)
        psd_accum += (np.abs(spec) ** 2) / max(1e-30, w_norm * (1.0 / tr.delta))
        count += 1

    if count > 0:
        psd_accum /= count
    freqs = np.fft.rfftfreq(nfft, d=tr.delta)
    return freqs[1:], psd_accum[1:]


def compute_mem_spectrum(
    tr: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    order: int = 30,
    num_freqs: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Burg Maximum Entropy (Autoregressive) spectrum matching nr_memcof / nr_evlmem in spectr.c."""
    seg = _extract_segment(tr, t_lo, t_hi)
    n = seg.size
    if n < 8 or tr.delta <= 0:
        return compute_fft_spectrum(tr, t_lo, t_hi)

    x = seg - np.mean(seg)
    m = max(2, min(int(order), n // 3))
    coeffs = burg_ar_coeffs(x, m)

    # Estimate prediction error variance pm
    res = x[m:].copy()
    for k in range(1, m + 1):
        res -= coeffs[k - 1] * x[m - k : n - k]
    pm = float(np.mean(res ** 2)) if res.size > 0 else float(np.var(x))

    nyq = 0.5 / tr.delta
    freqs = np.linspace(nyq / num_freqs, nyq, num_freqs)
    fdt = freqs * tr.delta
    k_idx = np.arange(1, m + 1, dtype=np.float64)[:, None]
    exp_mat = np.exp(-2j * math.pi * k_idx * fdt[None, :])
    denom = np.abs(1.0 - np.dot(coeffs, exp_mat)) ** 2
    psd = (pm * tr.delta) / np.maximum(denom, 1e-30)
    return freqs, np.sqrt(psd)


def fit_log_log_spectrum(
    freqs: np.ndarray,
    amps: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> Dict[str, float]:
    """Fit log10(A) = a + b * log10(f) and Brune omega-square plateau / corner frequency in [f_lo, f_hi]."""
    mask = (freqs >= min(f_lo, f_hi)) & (freqs <= max(f_lo, f_hi)) & (freqs > 0) & (amps > 0)
    if np.count_nonzero(mask) < 2:
        mask = (freqs > 0) & (amps > 0)
    lf = np.log10(freqs[mask])
    la = np.log10(amps[mask])
    if lf.size < 2:
        return {"slope": 0.0, "intercept": 0.0, "plateau": 0.0, "f_lo": f_lo, "f_hi": f_hi}

    slope, intercept = np.polyfit(lf, la, 1)
    plateau = float(10.0 ** np.mean(la))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "plateau": plateau,
        "f_lo": float(np.min(freqs[mask])),
        "f_hi": float(np.max(freqs[mask])),
    }


def compute_spectrogram(
    tr: Trace,
    win_length_sec: float = 2.0,
    step_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sliding-window Hann spectrogram matching spcwork.c.

    Returns (times_sec, freqs_hz, power_db_2d) where power_db_2d has shape (n_freqs, n_times).
    """
    dt = tr.delta if tr.delta > 0 else 0.05
    n = tr.length
    win_samps = max(8, min(n, int(round(win_length_sec / dt))))
    step_samps = max(1, int(round(step_sec / dt))) if (step_sec and step_sec > 0) else max(1, win_samps // 4)

    nfft = 1
    while nfft < win_samps:
        nfft <<= 1

    # Hann window matching 0.5*(1 - cos(2*pi*i/(L-1))) in spcwork.c
    hann = 0.5 * (1.0 - np.cos(2.0 * math.pi * np.arange(win_samps, dtype=np.float64) / max(1, win_samps - 1)))
    starts = list(range(0, max(1, n - win_samps + 1), step_samps))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    times = np.array([(s + 0.5 * win_samps) * dt for s in starts], dtype=np.float64)

    spec_mat = np.zeros((freqs.size, len(starts)), dtype=np.float64)
    x = tr.data - np.mean(tr.data)
    for idx, s in enumerate(starts):
        chunk = x[s : s + win_samps]
        if chunk.size < win_samps:
            chunk = np.pad(chunk, (0, win_samps - chunk.size))
        w_chunk = (chunk - np.mean(chunk)) * hann
        f_val = np.abs(np.fft.rfft(w_chunk, n=nfft))
        spec_mat[:, idx] = 20.0 * np.log10(np.maximum(f_val, 1e-12))

    return times, freqs, spec_mat


def _extract_segment(tr: Trace, t_lo: Optional[float], t_hi: Optional[float]) -> np.ndarray:
    if t_lo is None or t_hi is None or tr.delta <= 0:
        return tr.data
    i0 = max(0, int(round(min(t_lo, t_hi) / tr.delta)))
    i1 = min(tr.length, int(round(max(t_lo, t_hi) / tr.delta)) + 1)
    return tr.data[i0:i1] if i1 > i0 + 2 else tr.data


# Backward-compatible re-exports (placed after compute_fft_spectrum/_extract_segment).
from shm_app.processing.source_inversion import (  # noqa: E402
    _bounded_lm_optimize,
    _bounded_nelder_mead,
    _compute_log_freq_weights,
    _eval_log10_residual_and_jac,
    _extract_window_strict,
    _log10_source_spectrum_model,
    _resolve_model_exponents,
    build_brune_fit_diagnostic,
    compute_physical_source_parameters,
    invert_brune_spectrum,
    invert_source_spectrum,
    source_spectrum_model,
    subtract_noise_quadrature,
)
from shm_app.processing.surface_waves import (  # noqa: E402
    _fft_zero_phase_bandpass,
    _parabolic_extremum,
    find_surface_wave_amplitude,
    scan_surface_waves,
)

