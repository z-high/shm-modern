"""Spectral estimation (FFT, Bartlett/Welch Power Spectrum, Burg MEM Autoregressive Spectrum),
log-log spectral slope/plateau fitting, and sliding-window spectrogram matching
spectr.c and spcwork.c.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import Trace
from shm_app.processing.filters import burg_ar_coeffs, restitute_ground_displacement_nm
from shm_app.seismology.diagnostics import (
    compute_condition_number,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_mad,
    compute_normality_stats,
    compute_r_squared,
    compute_reduced_chi2,
    compute_rmse,
    compute_wrmse,
    detect_outliers_3sigma,
)


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
    if n < 8 or tr.delta <= 0:
        return compute_fft_spectrum(tr, t_lo, t_hi, squared=True)

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


def _extract_window_strict(tr: Trace, t_lo: float, t_hi: float) -> Optional[np.ndarray]:
    """Extract window strictly without falling back to the full trace if bounds are empty."""
    if tr.delta <= 0 or tr.length < 4:
        return None
    i0 = max(0, int(round(min(t_lo, t_hi) / tr.delta)))
    i1 = min(tr.length, int(round(max(t_lo, t_hi) / tr.delta)) + 1)
    if i1 <= i0 + 4:
        return None
    return tr.data[i0:i1]


def _resolve_model_exponents(model: str) -> Tuple[str, float, float]:
    """Return canonical (model_name, gamma, n_exp) for Brune or Boatwright omega^-2 models."""
    m = (model or "brune").strip().lower()
    if m in ("brune", "brune_1970", "omega2"):
        return "brune", 1.0, 2.0
    if m in ("boatwright", "boatwright_1980"):
        return "boatwright", 2.0, 2.0
    raise ValueError(f"Unknown spectral source model {model!r}; expected 'brune' or 'boatwright'.")


def _log10_source_spectrum_model(
    freqs: np.ndarray,
    log10_omega0: float,
    log10_fc: float,
    t_star: float,
    gamma: float,
    n_exp: float,
) -> np.ndarray:
    """Overflow-safe log10 displacement source spectrum evaluation."""
    f = np.asarray(freqs, dtype=np.float64)
    fc = 10.0 ** float(log10_fc)
    ln10 = math.log(10.0)
    out = np.full_like(f, float(log10_omega0), dtype=np.float64)
    pos = f > 0.0
    if np.any(pos):
        fp = f[pos]
        # log(1 + (fp / fc)**(gamma * n_exp)) computed via logaddexp(0, x) to avoid overflow
        exponent = (gamma * n_exp) * np.log(np.maximum(fp / fc, 1e-300))
        log1p_term = np.logaddexp(0.0, exponent) / (gamma * ln10)
        atten_term = (math.pi * fp * float(t_star)) / ln10
        out[pos] = float(log10_omega0) - atten_term - log1p_term
    return out


def source_spectrum_model(
    freqs: np.ndarray,
    omega0: float,
    fc: float,
    t_star: float = 0.0,
    model: str = "brune",
) -> np.ndarray:
    """Forward Brune or Boatwright omega^-2 displacement source spectrum model.

    Omega(f) = Omega0 * exp(-pi * f * t*) / [1 + (f / fc)**(gamma * n)]**(1 / gamma)
    where:
    - Brune (1970): gamma = 1, n = 2
    - Boatwright (1980): gamma = 2, n = 2

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies in Hz.
    omega0 : float
        Low-frequency displacement spectral plateau (e.g., in nm*s or m*s). Must be > 0.
    fc : float
        Corner frequency in Hz. Must be > 0.
    t_star : float
        Attenuation operator t* = T / Q in seconds (default 0.0).
    model : str
        'brune' or 'boatwright'.
    """
    if omega0 <= 0.0:
        raise ValueError(f"omega0 must be positive, got {omega0}")
    if fc <= 0.0:
        raise ValueError(f"fc must be positive, got {fc}")
    _, gamma, n_exp = _resolve_model_exponents(model)
    log10_spec = _log10_source_spectrum_model(
        np.asarray(freqs, dtype=np.float64),
        math.log10(float(omega0)),
        math.log10(float(fc)),
        float(t_star),
        gamma,
        n_exp,
    )
    # Clamp exponent before 10**x to avoid overflow/underflow warnings
    return np.power(10.0, np.clip(log10_spec, -300.0, 300.0))


def subtract_noise_quadrature(
    obs_spec: np.ndarray,
    noise_spec: np.ndarray,
) -> np.ndarray:
    """Subtract noise amplitude spectrum in quadrature:
    |U_signal(f)| = sqrt(max(0, |U_obs(f)|^2 - |U_noise(f)|^2)).
    """
    obs = np.asarray(obs_spec, dtype=np.float64)
    noise = np.asarray(noise_spec, dtype=np.float64)
    if obs.shape != noise.shape:
        raise ValueError(f"Shape mismatch: obs_spec {obs.shape} vs noise_spec {noise.shape}")
    diff_sq = np.maximum(0.0, (obs * obs) - (noise * noise))
    return np.sqrt(diff_sq)


def compute_physical_source_parameters(
    omega0: float,
    fc: float,
    rho_kg_m3: float = 2700.0,
    vel_m_s: float = 3500.0,
    distance_m: float = 100000.0,
    fs: float = 2.0,
    r_theta_phi: float = 0.55,
    cp: float = 1.0,
    wave_type: str = "S",
    vs_m_s: Optional[float] = None,
    omega0_unit: str = "nm*s",
) -> Dict[str, float]:
    """Compute physical earthquake source parameters from Brune/Boatwright spectral fit.

    Unit convention
    ---------------
    By default ``omega0_unit="nm*s"`` (standard SHM displacement spectrum in nm*s),
    which is converted to SI (m*s) via ``omega0_si = omega0 * 1e-9`` before computing
    seismic moment M0 in N*m. Pass ``omega0_unit="m*s"`` (or ``"si"``) if ``omega0``
    is already in SI m*s.
    """
    unit_norm = (omega0_unit or "nm*s").strip().lower()
    if unit_norm in ("nm*s", "nms", "nm.s", "nm"):
        unit_scale = 1e-9
    elif unit_norm in ("um*s", "ums", "micron*s"):
        unit_scale = 1e-6
    elif unit_norm in ("m*s", "ms", "m.s", "si"):
        unit_scale = 1.0
    else:
        raise ValueError(f"Unsupported omega0_unit {omega0_unit!r}")

    omega0_si = max(0.0, float(omega0)) * unit_scale
    denom = max(1e-30, float(fs) * float(r_theta_phi) * float(cp))
    m0 = (4.0 * math.pi * float(rho_kg_m3) * (float(vel_m_s) ** 3) * float(distance_m) * omega0_si) / denom

    mw = (2.0 / 3.0) * (math.log10(m0) - 9.1) if m0 > 0.0 else 0.0

    wt = (wave_type or "S").strip().upper()
    kc = 0.21 if wt.startswith("P") else 0.37
    v_shear = float(vs_m_s) if vs_m_s is not None else float(vel_m_s)
    r0_m = (kc * v_shear / float(fc)) if fc > 0.0 else 0.0
    stress_drop_pa = ((7.0 / 16.0) * m0 / (r0_m ** 3)) if r0_m > 0.0 else 0.0
    stress_drop_bar = stress_drop_pa * 1e-5

    return {
        "omega0_si_m_s": float(omega0_si),
        "m0": float(m0),
        "mw": float(mw),
        "r0_m": float(r0_m),
        "stress_drop_pa": float(stress_drop_pa),
        "stress_drop_bar": float(stress_drop_bar),
        "kc": float(kc),
    }


def _compute_log_freq_weights(freqs: np.ndarray) -> np.ndarray:
    """Compute normalized d(log10 f) weights so each frequency decade contributes equally."""
    n = freqs.size
    if n <= 2:
        return np.ones(n, dtype=np.float64)
    lf = np.log10(np.maximum(freqs, 1e-30))
    dlf = np.empty(n, dtype=np.float64)
    dlf[1:-1] = 0.5 * (lf[2:] - lf[:-2])
    dlf[0] = lf[1] - lf[0]
    dlf[-1] = lf[-1] - lf[-2]
    dlf = np.maximum(dlf, 1e-12)
    mean_w = float(np.mean(dlf))
    return dlf / mean_w if mean_w > 0.0 else np.ones(n, dtype=np.float64)


def _eval_log10_residual_and_jac(
    theta: np.ndarray,
    freqs: np.ndarray,
    y_log10: np.ndarray,
    gamma: float,
    n_exp: float,
    fit_t_star: bool,
    fixed_t_star: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate residual r = y_log10 - model_log10 and Jacobian J = d(model_log10)/d(theta)
    where theta = [log10_omega0, log10_fc, (t_star)].
    """
    log10_w0 = float(theta[0])
    log10_fc = float(theta[1])
    t_star = float(theta[2]) if fit_t_star else float(fixed_t_star)

    pred = _log10_source_spectrum_model(freqs, log10_w0, log10_fc, t_star, gamma, n_exp)
    res = y_log10 - pred

    ln10 = math.log(10.0)
    fc = 10.0 ** log10_fc
    # d(log10_model)/d(log10_fc) = n_exp * sigmoid(gamma * n_exp * ln(f / fc))
    arg = np.clip((gamma * n_exp) * np.log(np.maximum(freqs / fc, 1e-300)), -500.0, 500.0)
    sigmoid = np.where(arg >= 0.0, 1.0 / (1.0 + np.exp(-arg)), np.exp(arg) / (1.0 + np.exp(arg)))

    n_obs = freqs.size
    n_params = 3 if fit_t_star else 2
    jac = np.empty((n_obs, n_params), dtype=np.float64)
    jac[:, 0] = 1.0
    jac[:, 1] = n_exp * sigmoid
    if fit_t_star:
        jac[:, 2] = -(math.pi * freqs) / ln10
    return res, jac


def _bounded_lm_optimize(
    theta0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    freqs: np.ndarray,
    y_log10: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    n_exp: float,
    fit_t_star: bool,
    fixed_t_star: float,
    max_iter: int = 80,
) -> Tuple[np.ndarray, float, int, bool]:
    """Pure-NumPy bounded Levenberg-Marquardt optimizer for log10 spectral fitting."""
    theta = np.clip(np.asarray(theta0, dtype=np.float64), lb, ub)
    w = np.asarray(weights, dtype=np.float64)
    lam = 1e-3
    n_p = theta.size

    res, jac = _eval_log10_residual_and_jac(
        theta, freqs, y_log10, gamma, n_exp, fit_t_star, fixed_t_star
    )
    cost = float(np.sum(w * (res * res)))
    converged = False
    iters = 0

    for it in range(1, max_iter + 1):
        iters = it
        jw = jac * w[:, np.newaxis]
        jt_w_j = jac.T @ jw
        jt_w_r = jw.T @ res

        diag = np.maximum(np.diag(jt_w_j), 1e-8)
        accepted = False
        for _ in range(12):
            mat = jt_w_j + lam * np.diag(diag)
            try:
                step = np.linalg.solve(mat, jt_w_r)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(mat) @ jt_w_r

            cand = np.clip(theta + step, lb, ub)
            actual_step = cand - theta
            if float(np.linalg.norm(actual_step)) < 1e-10:
                converged = True
                accepted = True
                break

            res_c, jac_c = _eval_log10_residual_and_jac(
                cand, freqs, y_log10, gamma, n_exp, fit_t_star, fixed_t_star
            )
            cost_c = float(np.sum(w * (res_c * res_c)))
            if cost_c <= cost:
                rel_impr = (cost - cost_c) / max(cost, 1e-18)
                theta = cand
                res = res_c
                jac = jac_c
                cost = cost_c
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                if rel_impr < 1e-10 and float(np.linalg.norm(actual_step)) < 1e-6:
                    converged = True
                break
            lam = min(lam * 5.0, 1e9)

        if converged or not accepted:
            if not accepted:
                converged = True
            break

    return theta, cost, iters, converged


def _bounded_nelder_mead(
    theta0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    freqs: np.ndarray,
    y_log10: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    n_exp: float,
    fit_t_star: bool,
    fixed_t_star: float,
    max_iter: int = 200,
) -> Tuple[np.ndarray, float, int]:
    """Pure-NumPy bounded Nelder-Mead simplex polish/fallback."""
    x0 = np.clip(np.asarray(theta0, dtype=np.float64), lb, ub)
    w = np.asarray(weights, dtype=np.float64)
    n = x0.size

    def obj(x: np.ndarray) -> float:
        xc = np.clip(x, lb, ub)
        t_val = float(xc[2]) if fit_t_star else float(fixed_t_star)
        pred = _log10_source_spectrum_model(freqs, float(xc[0]), float(xc[1]), t_val, gamma, n_exp)
        r = y_log10 - pred
        return float(np.sum(w * (r * r)))

    simplex = [x0.copy()]
    span = np.maximum(ub - lb, 1e-6)
    for i in range(n):
        v = x0.copy()
        delta = max(0.05 * abs(v[i]), 0.02 * span[i], 1e-3)
        if v[i] + delta <= ub[i]:
            v[i] += delta
        else:
            v[i] = max(lb[i], v[i] - delta)
        simplex.append(v)

    fvals = [obj(v) for v in simplex]
    iters = 0
    for it in range(1, max_iter + 1):
        iters = it
        order = sorted(range(n + 1), key=lambda idx: fvals[idx])
        simplex = [simplex[idx] for idx in order]
        fvals = [fvals[idx] for idx in order]

        if max(float(np.linalg.norm(simplex[i] - simplex[0])) for i in range(1, n + 1)) < 1e-7:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lb, ub)
        fr = obj(xr)

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = np.clip(centroid + 2.0 * (xr - centroid), lb, ub)
            fe = obj(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = np.clip(centroid + 0.5 * (xr - centroid), lb, ub)
                fc_val = obj(xc)
                if fc_val <= fr:
                    simplex[-1], fvals[-1] = xc, fc_val
                    continue
            else:
                xc = np.clip(centroid - 0.5 * (centroid - simplex[-1]), lb, ub)
                fc_val = obj(xc)
                if fc_val < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc_val
                    continue
            for i in range(1, n + 1):
                simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lb, ub)
                fvals[i] = obj(simplex[i])

    best_idx = int(np.argmin(fvals))
    return np.clip(simplex[best_idx], lb, ub), float(fvals[best_idx]), iters


def invert_source_spectrum(
    freqs: np.ndarray,
    obs_spec: np.ndarray,
    noise_spec: Optional[np.ndarray] = None,
    model: str = "brune",
    f_lo: Optional[float] = 0.1,
    f_hi: Optional[float] = None,
    fit_t_star: bool = True,
    t_star_init: float = 0.02,
    t_star_bounds: Tuple[float, float] = (0.0, 2.0),
    fc_bounds: Optional[Tuple[float, float]] = None,
    n_grid: int = 41,
    log_spacing_weights: bool = True,
    **phys_kwargs: Any,
) -> Dict[str, Any]:
    """Non-linear Brune/Boatwright omega^-2 spectral source parameter inversion.

    Works directly on frequency and amplitude spectrum arrays (in nm*s by default).
    """
    canon_model, gamma, n_exp = _resolve_model_exponents(model)
    f_all = np.asarray(freqs, dtype=np.float64)
    obs_all = np.asarray(obs_spec, dtype=np.float64)
    if f_all.shape != obs_all.shape or f_all.size < 3:
        raise ValueError("freqs and obs_spec must have identical shapes with at least 3 points")

    sort_idx = np.argsort(f_all)
    f_all = f_all[sort_idx]
    obs_all = obs_all[sort_idx]

    if noise_spec is not None:
        noise_arr = np.asarray(noise_spec, dtype=np.float64)[sort_idx]
        if noise_arr.shape != obs_all.shape:
            raise ValueError("noise_spec must have the same shape as obs_spec")
        signal_all = subtract_noise_quadrature(obs_all, noise_arr)
    else:
        noise_arr = np.zeros_like(obs_all)
        signal_all = np.maximum(0.0, obs_all)

    f_min_pos = float(np.min(f_all[f_all > 0.0])) if np.any(f_all > 0.0) else 0.01
    f_max_pos = float(np.max(f_all))
    flo_eff = max(f_min_pos, float(f_lo)) if f_lo is not None else f_min_pos
    fhi_eff = min(f_max_pos, float(f_hi)) if f_hi is not None else f_max_pos
    if fhi_eff <= flo_eff:
        flo_eff, fhi_eff = f_min_pos, f_max_pos

    band_mask = (f_all >= flo_eff) & (f_all <= fhi_eff) & (f_all > 0.0)
    if np.count_nonzero(band_mask) < 3:
        band_mask = f_all > 0.0

    # Active fitting mask: require positive signal after noise subtraction
    if noise_spec is not None:
        has_noise = noise_arr > 0.0
        snr_est = np.where(has_noise, signal_all / np.maximum(noise_arr, 1e-30), np.inf)
        # Prefer bins clearly above noise floor (SNR >= 1.0) to suppress noise-dominated high-freq bias
        high_snr_mask = band_mask & (signal_all > 0.0) & ((~has_noise) | (snr_est >= 1.0))
        mid_snr_mask = band_mask & (signal_all > 0.0) & ((~has_noise) | (snr_est >= 0.5))
        if np.count_nonzero(high_snr_mask) >= 10:
            active_mask = high_snr_mask
        elif np.count_nonzero(mid_snr_mask) >= 6:
            active_mask = mid_snr_mask
        else:
            active_mask = band_mask & (signal_all > 0.0)
    else:
        active_mask = band_mask & (signal_all > 0.0)

    if np.count_nonzero(active_mask) < 3:
        active_mask = (f_all > 0.0) & (obs_all > 0.0)
        y_active_src = obs_all[active_mask]
    else:
        y_active_src = signal_all[active_mask]

    f_act = f_all[active_mask]
    y_log = np.log10(np.maximum(y_active_src, 1e-30))

    # Weights: frequency decade balance * SNR taper (when noise is present)
    w_act = _compute_log_freq_weights(f_act) if log_spacing_weights else np.ones_like(f_act)
    if noise_spec is not None:
        n_act = noise_arr[active_mask]
        s_act = y_active_src
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_act = np.where(n_act > 0.0, s_act / np.maximum(n_act, 1e-30), 10.0)
        snr_w = np.where(n_act > 0.0, (snr_act ** 2) / (0.25 + snr_act ** 2), 1.0)
        w_act = w_act * np.maximum(snr_w, 0.05)
    w_act = w_act / max(1e-12, float(np.mean(w_act)))

    # Parameter bounds in [log10_omega0, log10_fc, (t_star)]
    y_min_l, y_max_l = float(np.min(y_log)), float(np.max(y_log))
    w0_lb = y_min_l - 3.0
    w0_ub = y_max_l + 4.0

    if fc_bounds is not None:
        fc_lo, fc_hi = max(1e-4, float(fc_bounds[0])), max(float(fc_bounds[0]) * 1.1, float(fc_bounds[1]))
    else:
        fc_lo = max(1e-3, 0.25 * flo_eff)
        fc_hi = max(fc_lo * 2.0, 1.5 * fhi_eff)
    lfc_lb, lfc_ub = math.log10(fc_lo), math.log10(fc_hi)

    ts_lo, ts_hi = float(min(t_star_bounds)), float(max(t_star_bounds))
    fixed_ts = float(np.clip(t_star_init, ts_lo, ts_hi))

    # Coarse multi-start search over (fc, t_star) with exact closed-form optimal log10(omega0)
    fc_seeds = np.geomspace(max(fc_lo, 0.5 * flo_eff), min(fc_hi, fhi_eff), 15)
    ts_seeds = np.linspace(ts_lo, min(ts_hi, max(ts_lo, 0.5)), 6) if fit_t_star else np.array([fixed_ts])

    candidates: List[Tuple[float, np.ndarray]] = []
    sum_w = max(1e-12, float(np.sum(w_act)))
    ln10 = math.log(10.0)
    for fc_s in fc_seeds:
        lfc_s = math.log10(float(fc_s))
        shape_only = _log10_source_spectrum_model(f_act, 0.0, lfc_s, 0.0, gamma, n_exp)
        for ts_s in ts_seeds:
            base = shape_only - (math.pi * f_act * float(ts_s)) / ln10
            lw0_s = float(np.clip(np.sum(w_act * (y_log - base)) / sum_w, w0_lb, w0_ub))
            pred = base + lw0_s
            sse = float(np.sum(w_act * ((y_log - pred) ** 2)))
            theta_cand = np.array([lw0_s, lfc_s, float(ts_s)], dtype=np.float64) if fit_t_star else np.array([lw0_s, lfc_s], dtype=np.float64)
            candidates.append((sse, theta_cand))

    candidates.sort(key=lambda item: item[0])
    lb = np.array([w0_lb, lfc_lb, ts_lo], dtype=np.float64) if fit_t_star else np.array([w0_lb, lfc_lb], dtype=np.float64)
    ub = np.array([w0_ub, lfc_ub, ts_hi], dtype=np.float64) if fit_t_star else np.array([w0_ub, lfc_ub], dtype=np.float64)

    best_theta = candidates[0][1]
    best_cost = candidates[0][0]
    total_iters = 0
    best_converged = False

    for _, start_theta in candidates[:3]:
        th_lm, c_lm, it_lm, conv_lm = _bounded_lm_optimize(
            start_theta, lb, ub, f_act, y_log, w_act, gamma, n_exp, fit_t_star, fixed_ts
        )
        total_iters += it_lm
        if c_lm < best_cost:
            best_cost = c_lm
            best_theta = th_lm
            best_converged = conv_lm

    # Nelder-Mead polish from best LM solution (also ensures convergence near boundaries)
    th_nm, c_nm, it_nm = _bounded_nelder_mead(
        best_theta, lb, ub, f_act, y_log, w_act, gamma, n_exp, fit_t_star, fixed_ts, max_iter=150
    )
    total_iters += it_nm
    if c_nm < best_cost:
        best_cost = c_nm
        best_theta = th_nm
        best_converged = True

    log10_w0_opt = float(best_theta[0])
    log10_fc_opt = float(best_theta[1])
    t_star_opt = float(best_theta[2]) if fit_t_star else fixed_ts
    omega0_opt = 10.0 ** log10_w0_opt
    fc_opt = 10.0 ** log10_fc_opt

    # Evaluate model on full grid and active bins
    model_all = source_spectrum_model(f_all, omega0_opt, fc_opt, t_star=t_star_opt, model=canon_model)
    pred_act_log = _log10_source_spectrum_model(f_act, log10_w0_opt, log10_fc_opt, t_star_opt, gamma, n_exp)
    res_act = y_log - pred_act_log

    # Parameter covariance & uncertainties with respect to [log10_omega0, fc, (t_star)]
    _, jac_log = _eval_log10_residual_and_jac(
        best_theta, f_act, y_log, gamma, n_exp, fit_t_star, fixed_ts
    )
    jac_phys = jac_log.copy()
    jac_phys[:, 1] = jac_log[:, 1] / (fc_opt * ln10)

    n_params = best_theta.size
    dof = max(1, f_act.size - n_params)
    s_sq = float(np.sum(w_act * (res_act ** 2))) / dof
    cov = compute_covariance_matrix(jac_phys, W=w_act, s_sq=s_sq, damping=1e-9)
    corr = compute_correlation_matrix(cov)
    cond_num = compute_condition_number(jac_phys)

    rmse = compute_rmse(res_act)
    wrmse = compute_wrmse(res_act, w_act)
    mad = compute_mad(res_act)
    r2 = compute_r_squared(y_log, pred_act_log)
    obs_span = max(1e-12, float(np.max(y_log) - np.min(y_log)))
    nrmse = rmse / obs_span
    chi2_red = compute_reduced_chi2(res_act, uncertainties=np.full_like(res_act, max(rmse, 1e-4)), dof=dof)
    norm_stats = compute_normality_stats(res_act)

    # Uncertainties (delta-method for omega0 from log10_omega0)
    se_log10_w0 = math.sqrt(max(0.0, float(cov[0, 0])))
    se_omega0 = omega0_opt * ln10 * se_log10_w0
    se_fc = math.sqrt(max(0.0, float(cov[1, 1])))
    se_t_star = math.sqrt(max(0.0, float(cov[2, 2]))) if fit_t_star else 0.0

    # Derived physical source parameters
    phys = compute_physical_source_parameters(omega0_opt, fc_opt, **phys_kwargs)

    # 2D misfit contour grid over (omega0_grid, fc_grid) at optimal t_star
    ng = max(11, int(n_grid))
    if ng % 2 == 0:
        ng += 1
    fc_min_g = max(1e-3, 0.35 * fc_opt)
    fc_max_g = max(fc_min_g * 2.0, 1.65 * fc_opt)
    fc_grid = np.linspace(fc_min_g, fc_max_g, ng, dtype=np.float64)
    fc_grid[ng // 2] = fc_opt

    w0_min_g = 0.35 * omega0_opt
    w0_max_g = 1.65 * omega0_opt
    omega0_grid = np.linspace(w0_min_g, w0_max_g, ng, dtype=np.float64)
    omega0_grid[ng // 2] = omega0_opt

    # Vectorized 2D RMS log10 misfit grid of shape (n_omega0, n_fc)
    lfc_grid = np.log10(fc_grid)
    lw0_grid = np.log10(omega0_grid)
    # Shape: (1, n_fc, n_act)
    arg_g = (gamma * n_exp) * np.log(np.maximum(f_act[None, :] / fc_grid[:, None], 1e-300))
    shape_fc = -(np.logaddexp(0.0, arg_g) / (gamma * ln10)) - (math.pi * f_act[None, :] * t_star_opt) / ln10
    # Residual shape: (n_omega0, n_fc, n_act)
    diff_g = y_log[None, None, :] - (lw0_grid[:, None, None] + shape_fc[None, :, :])
    misfit_grid = np.sqrt(np.sum(w_act[None, None, :] * (diff_g ** 2), axis=2) / max(1e-12, float(np.sum(w_act))))

    return {
        "model_type": canon_model,
        "gamma": gamma,
        "n_exp": n_exp,
        "omega0": float(omega0_opt),
        "fc": float(fc_opt),
        "t_star": float(t_star_opt),
        "se_omega0": float(se_omega0),
        "se_fc": float(se_fc),
        "se_t_star": float(se_t_star),
        "m0": phys["m0"],
        "mw": phys["mw"],
        "r0_m": phys["r0_m"],
        "stress_drop_pa": phys["stress_drop_pa"],
        "stress_drop_bar": phys["stress_drop_bar"],
        "kc": phys["kc"],
        "omega0_si_m_s": phys["omega0_si_m_s"],
        "rms_log10": float(rmse),
        "wrmse_log10": float(wrmse),
        "nrmse_log10": float(nrmse),
        "mad_log10": float(mad),
        "chi2_reduced": float(chi2_red),
        "r_squared": float(r2),
        "condition_number": float(cond_num),
        "covariance_matrix": cov,
        "correlation_matrix": corr,
        "normality_stats": norm_stats,
        "dof": int(dof),
        "iterations": int(total_iters),
        "converged": bool(best_converged),
        "freqs": f_all,
        "obs_spec": obs_all,
        "signal_spec": signal_all,
        "noise_spec": noise_arr,
        "model_spec": model_all,
        "active_freqs": f_act,
        "active_log10_obs": y_log,
        "active_log10_model": pred_act_log,
        "active_residuals": res_act,
        "active_weights": w_act,
        "fc_grid": fc_grid,
        "omega0_grid": omega0_grid,
        "misfit_grid": misfit_grid,
    }


def invert_brune_spectrum(
    tr: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    noise_lo: Optional[float] = None,
    noise_hi: Optional[float] = None,
    model: str = "brune",
    f_lo: float = 0.1,
    f_hi: Optional[float] = None,
    noise_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    input_type: str = "displacement",
    taper_frac: float = 0.05,
    **phys_kwargs: Any,
) -> Dict[str, Any]:
    """Perform non-linear Brune/Boatwright omega^-2 spectral source inversion on a Trace.

    Parameters
    ----------
    tr : Trace
        Input waveform trace (displacement in nm by default, or velocity in nm/s if
        ``input_type="velocity"``).
    t_lo, t_hi : Optional[float]
        Signal phase window in trace-relative seconds.
    noise_lo, noise_hi : Optional[float]
        Pre-phase noise window in trace-relative seconds for quadrature noise subtraction.
    model : str
        ``"brune"`` (gamma=1, n=2) or ``"boatwright"`` (gamma=2, n=2).
    f_lo, f_hi : float, Optional[float]
        Frequency band limits in Hz. If ``f_hi`` is None, defaults to ``0.9 * Nyquist``.
    noise_spectrum : Optional[Tuple[np.ndarray, np.ndarray]]
        Explicit ``(noise_freqs, noise_amps)`` pair overriding ``noise_lo/noise_hi``.
    input_type : str
        ``"displacement"`` (default) or ``"velocity"`` (divides spectrum by 2*pi*f).
    **phys_kwargs : Any
        Physical parameters forwarded to ``compute_physical_source_parameters`` and
        ``invert_source_spectrum`` (e.g. ``rho_kg_m3``, ``vel_m_s``, ``distance_m``,
        ``fs``, ``r_theta_phi``, ``cp``, ``wave_type``, ``fit_t_star``, ``t_star_bounds``).
    """
    if tr.delta <= 0.0 or tr.length < 4:
        raise ValueError("Trace must have positive delta and at least 4 samples.")

    sig_seg = _extract_segment(tr, t_lo, t_hi)
    freqs, obs_amps = compute_fft_spectrum(tr, t_lo=t_lo, t_hi=t_hi, squared=False, taper_frac=taper_frac)

    is_vel = (input_type or "displacement").strip().lower().startswith("v")
    if is_vel and not any(k in (tr.filter_name or "").upper() for k in ("DISPLACE", "KIRNOS", "WOODAND", "WWSSN", "SRO", "LRSM")):
        obs_amps = obs_amps / np.maximum(2.0 * math.pi * freqs, 1e-12)

    noise_interp: Optional[np.ndarray] = None
    if noise_spectrum is not None:
        nf, na = noise_spectrum
        nf_arr = np.asarray(nf, dtype=np.float64)
        na_arr = np.asarray(na, dtype=np.float64)
        order = np.argsort(nf_arr)
        noise_interp = np.interp(freqs, nf_arr[order], na_arr[order], left=na_arr[order][0], right=na_arr[order][-1])
    elif noise_lo is not None and noise_hi is not None:
        noise_seg = _extract_window_strict(tr, noise_lo, noise_hi)
        if noise_seg is not None:
            n_tr = Trace(data=noise_seg, delta=tr.delta, station=tr.station, comp=tr.comp, filter_name=tr.filter_name)
            nf, na = compute_fft_spectrum(n_tr, squared=False, taper_frac=taper_frac)
            if is_vel and not any(k in (tr.filter_name or "").upper() for k in ("DISPLACE", "KIRNOS", "WOODAND", "WWSSN", "SRO", "LRSM")):
                na = na / np.maximum(2.0 * math.pi * nf, 1e-12)
            # Scale noise FFT amplitude by sqrt(N_sig / N_noise) for equivalent Parseval window duration
            dur_scale = math.sqrt(max(1, sig_seg.size) / max(1, noise_seg.size))
            na_scaled = na * dur_scale
            noise_interp = np.interp(freqs, nf, na_scaled, left=na_scaled[0], right=na_scaled[-1])

    nyq = 0.5 / tr.delta
    fhi_eff = float(f_hi) if f_hi is not None else 0.9 * nyq

    result = invert_source_spectrum(
        freqs=freqs,
        obs_spec=obs_amps,
        noise_spec=noise_interp,
        model=model,
        f_lo=f_lo,
        f_hi=fhi_eff,
        **phys_kwargs,
    )
    result["station"] = tr.station
    result["channel"] = tr.channel3
    return result


def build_brune_fit_diagnostic(
    inv_result: Dict[str, Any],
    station: str = "STAT",
    phase: str = "P",
) -> FitDiagnosticResult:
    """Convert an ``invert_brune_spectrum`` / ``invert_source_spectrum`` result dict
    into a unified ``FitDiagnosticResult(fit_type=FitType.BRUNE_SPECTRUM)``.
    """
    st = station or str(inv_result.get("station", "STAT"))
    ph = phase or "P"

    omega0 = float(inv_result["omega0"])
    fc = float(inv_result["fc"])
    t_star = float(inv_result["t_star"])
    se_w0 = float(inv_result.get("se_omega0", 0.0))
    se_fc = float(inv_result.get("se_fc", 0.0))
    se_ts = float(inv_result.get("se_t_star", 0.0))

    parameters = {
        "omega0": ParameterUncertainty(
            name="omega0",
            value=omega0,
            standard_error=se_w0,
            confidence_interval_95=(max(0.0, omega0 - 1.96 * se_w0), omega0 + 1.96 * se_w0),
            unit="nm*s",
        ),
        "fc": ParameterUncertainty(
            name="fc",
            value=fc,
            standard_error=se_fc,
            confidence_interval_95=(max(0.0, fc - 1.96 * se_fc), fc + 1.96 * se_fc),
            unit="Hz",
        ),
        "t_star": ParameterUncertainty(
            name="t_star",
            value=t_star,
            standard_error=se_ts,
            confidence_interval_95=(max(0.0, t_star - 1.96 * se_ts), t_star + 1.96 * se_ts),
            unit="s",
        ),
    }

    f_act = np.asarray(inv_result["active_freqs"], dtype=np.float64)
    y_obs = np.asarray(inv_result["active_log10_obs"], dtype=np.float64)
    y_mod = np.asarray(inv_result["active_log10_model"], dtype=np.float64)
    res_arr = np.asarray(inv_result["active_residuals"], dtype=np.float64)
    w_arr = np.asarray(inv_result["active_weights"], dtype=np.float64)

    sigma_ref = max(float(inv_result.get("rms_log10", 0.05)), 1e-4)
    outlier_flags = detect_outliers_3sigma(res_arr) if res_arr.size >= 4 else np.zeros(res_arr.size, dtype=bool)
    residuals: List[ResidualEntry] = []
    for k in range(f_act.size):
        r_val = float(res_arr[k])
        residuals.append(
            ResidualEntry(
                identifier=f"Freq_{float(f_act[k]):.4g}Hz",
                observed=float(y_obs[k]),
                calculated=float(y_mod[k]),
                residual=r_val,
                normalized_residual=r_val / sigma_ref,
                weight=float(w_arr[k]),
                uncertainty=sigma_ref,
                station=st,
                phase=ph,
                channel=str(inv_result.get("channel", "")),
                is_outlier=bool(outlier_flags[k]),
            )
        )

    norm_stats = inv_result.get("normality_stats", {})
    overlay_curves = {
        "freqs": np.asarray(inv_result["freqs"], dtype=np.float64),
        "obs_spec": np.asarray(inv_result["obs_spec"], dtype=np.float64),
        "model_spec": np.asarray(inv_result["model_spec"], dtype=np.float64),
        "noise_spec": np.asarray(inv_result["noise_spec"], dtype=np.float64),
        "fc_grid": np.asarray(inv_result["fc_grid"], dtype=np.float64),
        "omega0_grid": np.asarray(inv_result["omega0_grid"], dtype=np.float64),
        "misfit_grid": np.asarray(inv_result["misfit_grid"], dtype=np.float64),
    }
    context_data = {
        "model_type": str(inv_result["model_type"]),
        "omega0": omega0,
        "fc": fc,
        "t_star": t_star,
        "m0": float(inv_result["m0"]),
        "mw": float(inv_result["mw"]),
        "r0_m": float(inv_result["r0_m"]),
        "stress_drop_pa": float(inv_result["stress_drop_pa"]),
        "stress_drop_bar": float(inv_result["stress_drop_bar"]),
        "rms_log10": float(inv_result["rms_log10"]),
        "r_squared": float(inv_result["r_squared"]),
        "station": st,
        "phase": ph,
    }

    return FitDiagnosticResult(
        fit_type=FitType.BRUNE_SPECTRUM,
        title=f"{str(inv_result['model_type']).title()} Spectral Source Inversion ({st} {ph})",
        converged=bool(inv_result.get("converged", True)),
        iterations=int(inv_result.get("iterations", 1)),
        dof=int(inv_result.get("dof", 1)),
        rmse=float(inv_result.get("rms_log10", 0.0)),
        wrmse=float(inv_result.get("wrmse_log10", 0.0)),
        nrmse=float(inv_result.get("nrmse_log10", 0.0)),
        mad=float(inv_result.get("mad_log10", 0.0)),
        chi2_reduced=float(inv_result.get("chi2_reduced", 0.0)),
        r_squared=float(inv_result.get("r_squared", 0.0)),
        condition_number=float(inv_result.get("condition_number", 1.0)),
        parameters=parameters,
        covariance_matrix=inv_result.get("covariance_matrix"),
        correlation_matrix=inv_result.get("correlation_matrix"),
        residuals=residuals,
        outlier_count=int(np.count_nonzero(outlier_flags)),
        overlay_curves=overlay_curves,
        normality_p_value=float(norm_stats.get("p_value", 1.0)),
        skewness=float(norm_stats.get("skewness", 0.0)),
        kurtosis=float(norm_stats.get("kurtosis", 0.0)),
        context_data=context_data,
    )


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
    if tr.delta <= 0.0 or tr.length < 8:
        raise ValueError("Trace must have positive delta and at least 8 samples.")

    p_lo = min(float(period_lo), float(period_hi))
    p_hi = max(float(period_lo), float(period_hi))
    f_lo = 1.0 / p_hi
    f_hi = 1.0 / p_lo

    filtered = _fft_zero_phase_bandpass(tr.data, tr.delta, f_lo, f_hi)

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

    ampl_um = ampl_nm / 1000.0
    dist = float(distance_deg) if distance_deg is not None else (float(tr.distance) if tr.distance > 0.0 else None)

    ms_val: Optional[float] = None
    if dist is not None:
        if dist > 0.0 and ampl_nm > 0.0 and period_est > 0.0:
            ms_val = math.log10(ampl_um / period_est) + 1.66 * math.log10(dist) + 3.3
        else:
            ms_val = 0.0

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

        res = find_surface_wave_amplitude(
            tr,
            distance_deg=dist,
            period_lo=period_lo,
            period_hi=period_hi,
            t_lo=t_lo,
            t_hi=t_hi,
            input_type=input_type,
        )
        results.append(res)
        if res["ms"] is not None and res["ms"] > 0.0:
            ms_vals.append(float(res["ms"]))

    return {
        "stations": results,
        "count": len(results),
        "ms_mean": float(np.mean(ms_vals)) if ms_vals else None,
        "ms_std": float(np.std(ms_vals, ddof=1)) if len(ms_vals) > 1 else (0.0 if ms_vals else None),
        "ms_median": float(np.median(ms_vals)) if ms_vals else None,
    }

