"""Non-linear Brune/Boatwright omega^-2 spectral source parameter inversion and diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import Trace
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

    from shm_app.processing.spectrum import _extract_segment, compute_fft_spectrum

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
        if nf_arr.shape != na_arr.shape:
            raise ValueError(
                f"noise_spectrum frequency and amplitude shapes must match, got {nf_arr.shape} vs {na_arr.shape}"
            )
        valid_noise = np.isfinite(nf_arr) & np.isfinite(na_arr)
        if np.any(valid_noise):
            nf_v = nf_arr[valid_noise]
            na_v = na_arr[valid_noise]
            order = np.argsort(nf_v)
            nf_s = nf_v[order]
            na_s = na_v[order]
            noise_interp = np.interp(freqs, nf_s, na_s, left=na_s[0], right=na_s[-1])
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
            if nf.size > 0 and na_scaled.size > 0:
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
