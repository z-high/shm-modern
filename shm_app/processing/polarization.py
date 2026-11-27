"""Covariance eigendecomposition (MDIR), particle motion hodograms, and
sliding-window polarization filtering matching pmotion.c and polfil.c.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import Trace
from shm_app.processing.rotation import (
    MAX_POLARIZATION_ALIGN_SAMPLES,
    _align_traces_for_polarization,
    rotate_2d,
    rotate_3d_lqt,
    rotate_STS2_zne_to_uvw,
    rotate_zne_to_zrt,
)
from shm_app.seismology.diagnostics import (
    compute_mad,
    compute_normality_stats,
    compute_reduced_chi2,
)


def _polarization_window_indices(
    lth: int,
    dt: float,
    t_lo: Optional[float],
    t_hi: Optional[float],
) -> Tuple[int, int]:
    """Resolve sample slice [i0, i1) for polarization analysis."""
    if lth <= 0:
        return 0, 0
    if t_lo is not None and t_hi is not None and dt > 0:
        lo = min(float(t_lo), float(t_hi))
        hi = max(float(t_lo), float(t_hi))
        i0 = max(0, int(round(lo / dt)))
        i1 = min(lth, int(round(hi / dt)) + 1)
    else:
        i0, i1 = 0, lth
    if i1 <= i0 + 1:
        i0, i1 = 0, lth
    return i0, i1


def _empty_mdir_result() -> Dict[str, float]:
    return {
        "azimuth": 0.0,
        "incidence": 0.0,
        "rectilinearity": 0.0,
        "planarity": 0.0,
        "lambda1": 0.0,
        "lambda2": 0.0,
        "lambda3": 0.0,
        "rms_pol": 0.0,
        "rms_pol_norm": 0.0,
        "sigma_theta_deg": 0.0,
        "sigma_phi_deg": 0.0,
    }


def compute_mdir(
    tr_z: Trace,
    tr_n: Trace,
    tr_e: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
) -> Dict[str, float]:
    """Compute 3D covariance eigendecomposition (MDIR) for P-wave backazimuth, incidence angle,
    rectilinearity, trajectory misfit RMS, and Jurkevics confidence cones.
    """
    t_start, dt, (z_all, n_all, e_all) = _align_traces_for_polarization(tr_z, tr_n, tr_e)
    shift = t_start - float(tr_z.start_epoch)
    lo_adj = (float(t_lo) - shift) if (t_lo is not None and t_hi is not None) else t_lo
    hi_adj = (float(t_hi) - shift) if (t_lo is not None and t_hi is not None) else t_hi
    lth = z_all.size
    i0, i1 = _polarization_window_indices(lth, dt, lo_adj, hi_adj)

    if i1 <= i0:
        return _empty_mdir_result()

    z_seg = z_all[i0:i1]
    n_seg = n_all[i0:i1]
    e_seg = e_all[i0:i1]
    finite_mask = np.isfinite(z_seg) & np.isfinite(n_seg) & np.isfinite(e_seg)
    if np.count_nonzero(finite_mask) == 0:
        return _empty_mdir_result()

    z_fin = z_seg[finite_mask]
    n_fin = n_seg[finite_mask]
    e_fin = e_seg[finite_mask]
    z = z_fin - np.mean(z_fin)
    n = n_fin - np.mean(n_fin)
    e = e_fin - np.mean(e_fin)

    mat = np.vstack([z, n, e])
    scale = float(np.max(np.abs(mat))) if mat.size > 0 else 0.0
    if not math.isfinite(scale):
        return _empty_mdir_result()

    if scale > 0.0:
        mat_scaled = mat / scale
        cov_scaled = (mat_scaled @ mat_scaled.T) / max(1, z.size)
        if not np.all(np.isfinite(cov_scaled)):
            return _empty_mdir_result()
        evals_scaled, evecs = np.linalg.eigh(cov_scaled)
        evals = evals_scaled * (scale * scale)
    else:
        evals = np.zeros(3, dtype=np.float64)
        evecs = np.eye(3, dtype=np.float64)

    # Sort eigenvalues descending: lam1 >= lam2 >= lam3
    order = np.argsort(evals)[::-1]
    evals = np.maximum(0.0, evals[order])
    evecs = evecs[:, order]

    v1 = evecs[:, 0]  # Principal eigenvector [e_Z, e_N, e_E]
    # For an upgoing compressional P wave, positive Z motion corresponds to radial motion away from source,
    # so backazimuth points in the opposite direction of (e_N, e_E) when e_Z > 0.
    sgn = 1.0 if v1[0] >= 0.0 else -1.0
    ez = abs(float(v1[0]))
    en = -sgn * float(v1[1])
    ee = -sgn * float(v1[2])

    incidence_deg = math.degrees(math.acos(max(-1.0, min(1.0, ez))))
    baz_deg = math.degrees(math.atan2(ee, en)) % 360.0

    lam1, lam2, lam3 = float(evals[0]), float(evals[1]), float(evals[2])
    rectilinearity = 1.0 - (lam2 + lam3) / (2.0 * lam1) if lam1 > 0.0 and lam1 > 1e-30 * max(1.0, scale * scale) else 0.0
    planarity = 1.0 - (2.0 * lam3) / (lam1 + lam2) if (lam1 + lam2) > 0.0 and (lam1 + lam2) > 1e-30 * max(1.0, scale * scale) else 0.0

    # Orthogonal trajectory distance d_k = ||u_k - (u_k . v1) v1|| = sqrt(||u_k||^2 - (u_k . v1)^2)
    u_sq = np.sum(mat ** 2, axis=0)
    proj = v1 @ mat
    perp = mat - np.outer(v1, proj)
    d_k_sq = np.sum(perp ** 2, axis=0)
    rms_pol = float(np.sqrt(np.mean(d_k_sq)))
    rms_total = float(np.sqrt(np.mean(u_sq)))
    rms_pol_norm = (rms_pol / rms_total) if rms_total > 1e-15 else 0.0

    # Jurkevics (1988) confidence cones
    sigma_theta_deg = math.degrees(math.atan(math.sqrt(lam2 / max(lam1, 1e-300)))) if lam1 > 0.0 else 0.0
    sigma_phi_deg = math.degrees(math.atan(math.sqrt(lam3 / max(lam1, 1e-300)))) if lam1 > 0.0 else 0.0

    return {
        "azimuth": baz_deg,
        "incidence": incidence_deg,
        "rectilinearity": rectilinearity,
        "planarity": planarity,
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda3": lam3,
        "rms_pol": rms_pol,
        "rms_pol_norm": rms_pol_norm,
        "sigma_theta_deg": sigma_theta_deg,
        "sigma_phi_deg": sigma_phi_deg,
    }


def analyze_polarization_diagnostic(
    tr_z: Trace,
    tr_n: Trace,
    tr_e: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
) -> FitDiagnosticResult:
    """Compute M-16 3C polarization trajectory diagnostics and package as FitDiagnosticResult."""
    t_start, dt, (z_all, n_all, e_all) = _align_traces_for_polarization(tr_z, tr_n, tr_e)
    shift = t_start - float(tr_z.start_epoch)
    lo_adj = (float(t_lo) - shift) if (t_lo is not None and t_hi is not None) else t_lo
    hi_adj = (float(t_hi) - shift) if (t_lo is not None and t_hi is not None) else t_hi
    lth = z_all.size
    i0, i1 = _polarization_window_indices(lth, dt, lo_adj, hi_adj)

    mdir = compute_mdir(tr_z, tr_n, tr_e, t_lo=t_lo, t_hi=t_hi)
    baz = mdir["azimuth"]
    incidence = mdir["incidence"]
    rect = mdir["rectilinearity"]
    plan = mdir["planarity"]
    lam1 = mdir["lambda1"]
    lam2 = mdir["lambda2"]
    lam3 = mdir["lambda3"]
    rms_pol = mdir["rms_pol"]
    rms_pol_norm = mdir["rms_pol_norm"]
    sigma_theta_deg = mdir["sigma_theta_deg"]
    sigma_phi_deg = mdir["sigma_phi_deg"]

    z_seg = z_all[i0:i1] if i1 > i0 else np.empty(0, dtype=np.float64)
    n_seg = n_all[i0:i1] if i1 > i0 else np.empty(0, dtype=np.float64)
    e_seg = e_all[i0:i1] if i1 > i0 else np.empty(0, dtype=np.float64)
    finite_mask = np.isfinite(z_seg) & np.isfinite(n_seg) & np.isfinite(e_seg)

    if i1 <= i0 or np.count_nonzero(finite_mask) == 0:
        empty = np.array([], dtype=np.float64)
        return FitDiagnosticResult(
            fit_type=FitType.POLARIZATION_3C,
            title=f"3C Polarization Analysis ({tr_z.station})",
            converged=False,
            iterations=1,
            dof=1,
            rmse=0.0,
            wrmse=0.0,
            nrmse=0.0,
            overlay_curves={
                "z": empty,
                "n": empty,
                "e": empty,
                "l": empty,
                "q": empty,
                "t": empty,
                "time_rel": empty,
            },
            context_data={
                "baz": 0.0,
                "incidence": 0.0,
                "rectilinearity": 0.0,
                "planarity": 0.0,
                "sigma_theta_deg": 0.0,
                "sigma_phi_deg": 0.0,
                "rms_pol": 0.0,
                "rms_pol_norm": 0.0,
                "lambda1": 0.0,
                "lambda2": 0.0,
                "lambda3": 0.0,
            },
        )

    mu_z = float(np.mean(z_seg[finite_mask]))
    mu_n = float(np.mean(n_seg[finite_mask]))
    mu_e = float(np.mean(e_seg[finite_mask]))
    z_win = np.where(finite_mask, z_seg - mu_z, 0.0)
    n_win = np.where(finite_mask, n_seg - mu_n, 0.0)
    e_win = np.where(finite_mask, e_seg - mu_e, 0.0)
    time_rel = np.arange(i0, i1, dtype=np.float64) * dt

    # Rotate demeaned windowed traces into ray-aligned (L, Q, T)
    tz_w = Trace(data=z_win, delta=dt, station=tr_z.station, comp="Z")
    tn_w = Trace(data=n_win, delta=dt, station=tr_n.station, comp="N")
    te_w = Trace(data=e_win, delta=dt, station=tr_e.station, comp="E")
    tl_w, tq_w, tt_w = rotate_3d_lqt(tz_w, tn_w, te_w, baz, incidence)
    l_arr = tl_w.data
    q_arr = tq_w.data
    t_arr = tt_w.data

    # Orthogonal trajectory misfit samples d_k = sqrt(q_k^2 + t_k^2)
    u_norm = np.sqrt(z_win ** 2 + n_win ** 2 + e_win ** 2)
    l_abs = np.abs(l_arr)
    d_k = np.sqrt(np.maximum(0.0, q_arr ** 2 + t_arr ** 2))

    n_samples = int(z_win.size)
    dof = max(1, int(np.count_nonzero(finite_mask)) - 3)
    tot_energy = lam1 + lam2 + lam3
    r_squared = (lam1 / tot_energy) if tot_energy > 1e-30 else 0.0
    sigma_scale = max(rms_pol, 1e-12)
    chi2_red = compute_reduced_chi2(d_k, uncertainties=np.full(n_samples, sigma_scale), dof=dof)
    mad_val = compute_mad(d_k)
    cond_num = math.sqrt(lam1 / max(lam3, 1e-15)) if lam1 > 1e-30 else 1.0
    norm_stats = compute_normality_stats(d_k)

    # Sample up to 200 residual entries evenly across the window for diagnostics table
    max_entries = 200
    step = max(1, int(math.ceil(n_samples / max_entries)))
    outlier_thresh = max(3.0 * sigma_scale, 1e-9)
    residual_entries: List[ResidualEntry] = []
    for idx in range(0, n_samples, step):
        dk_val = float(d_k[idx])
        norm_res = dk_val / sigma_scale if sigma_scale > 1e-12 else 0.0
        is_out = dk_val > outlier_thresh and rms_pol > 1e-12
        residual_entries.append(
            ResidualEntry(
                identifier=f"{tr_z.station}.3C.t={time_rel[idx]:.2f}s",
                observed=float(u_norm[idx]),
                calculated=float(l_abs[idx]),
                residual=dk_val,
                normalized_residual=float(norm_res),
                weight=1.0,
                uncertainty=float(sigma_scale),
                distance_deg=float(tr_z.distance),
                azimuth_deg=round(baz, 2),
                station=tr_z.station,
                phase="POL",
                channel="3C",
                is_outlier=is_out,
            )
        )

    outlier_count = int(sum(1 for r in residual_entries if r.is_outlier))

    parameters: Dict[str, ParameterUncertainty] = {
        "baz": ParameterUncertainty(
            name="baz",
            value=baz,
            standard_error=sigma_theta_deg,
            confidence_interval_95=(baz - 1.96 * sigma_theta_deg, baz + 1.96 * sigma_theta_deg),
            unit="deg",
        ),
        "incidence": ParameterUncertainty(
            name="incidence",
            value=incidence,
            standard_error=sigma_phi_deg,
            confidence_interval_95=(
                max(0.0, incidence - 1.96 * sigma_phi_deg),
                min(90.0, incidence + 1.96 * sigma_phi_deg),
            ),
            unit="deg",
        ),
        "rectilinearity": ParameterUncertainty(
            name="rectilinearity",
            value=rect,
            standard_error=rms_pol_norm,
            confidence_interval_95=(
                max(0.0, rect - 1.96 * rms_pol_norm),
                min(1.0, rect + 1.96 * rms_pol_norm),
            ),
            unit="",
        ),
        "planarity": ParameterUncertainty(
            name="planarity",
            value=plan,
            standard_error=0.0,
            confidence_interval_95=(plan, plan),
            unit="",
        ),
        "rms_pol": ParameterUncertainty(
            name="rms_pol",
            value=rms_pol,
            standard_error=rms_pol_norm,
            confidence_interval_95=(max(0.0, rms_pol * (1.0 - 1.96 * rms_pol_norm)), rms_pol * (1.0 + 1.96 * rms_pol_norm)),
            unit="",
        ),
    }

    p_cnt = len(parameters)
    corr_mat = np.eye(p_cnt, dtype=np.float64)

    return FitDiagnosticResult(
        fit_type=FitType.POLARIZATION_3C,
        title=f"3C Polarization Analysis ({tr_z.station})",
        converged=bool(lam1 > 0.0),
        iterations=1,
        dof=dof,
        rmse=rms_pol,
        wrmse=rms_pol,
        nrmse=rms_pol_norm,
        mad=mad_val,
        chi2_reduced=chi2_red,
        r_squared=r_squared,
        condition_number=cond_num,
        parameters=parameters,
        correlation_matrix=corr_mat,
        residuals=residual_entries,
        outlier_count=outlier_count,
        overlay_curves={
            "z": z_win,
            "n": n_win,
            "e": e_win,
            "l": l_arr,
            "q": q_arr,
            "t": t_arr,
            "time_rel": time_rel,
        },
        normality_p_value=norm_stats.get("p_value", 1.0),
        skewness=norm_stats.get("skewness", 0.0),
        kurtosis=norm_stats.get("kurtosis", 0.0),
        context_data={
            "baz": baz,
            "incidence": incidence,
            "rectilinearity": rect,
            "planarity": plan,
            "sigma_theta_deg": sigma_theta_deg,
            "sigma_phi_deg": sigma_phi_deg,
            "rms_pol": rms_pol,
            "rms_pol_norm": rms_pol_norm,
            "lambda1": lam1,
            "lambda2": lam2,
            "lambda3": lam3,
            "station": tr_z.station,
            "t_lo": float(time_rel[0]) if time_rel.size > 0 else 0.0,
            "t_hi": float(time_rel[-1]) if time_rel.size > 0 else 0.0,
        },
    )


def apply_polarization_filter(
    tr_z: Trace,
    tr_n: Trace,
    tr_e: Trace,
    coherence_win_sec: float = 1.0,
    linearity_power: float = 2.0,
    mode: int = 1,
    baz_deg: float = 0.0,
    inci_deg: float = 20.0,
) -> Tuple[Trace, Trace, Trace]:
    """Apply 3-component sliding-window centered-covariance polarization filter matching polfil.c.

    Modes:
    1 : Weight original (Z, N, E) by rectilinearity R(k)^p
    2 : Project (Z, N, E) onto instantaneous principal eigenvector e1(k) * R(k)^p
    3 : Weight by directional cosine |e1(k) . d_fixed|^p * R(k)^p
    4 : P-wave azimuth filter (weight by horizontal alignment with baz_deg)
    5 : S-wave transverse filter (weight by perpendicularity to P direction)
    """
    t_start, dt, (z, n, e) = _align_traces_for_polarization(tr_z, tr_n, tr_e)
    lth = z.size
    tz, tn, te = tr_z.copy(), tr_n.copy(), tr_e.copy()
    tz.start_epoch = t_start
    tn.start_epoch = t_start
    te.start_epoch = t_start
    tz.opinfo = f"POLFIL(M{mode})"
    tn.opinfo = f"POLFIL(M{mode})"
    te.opinfo = f"POLFIL(M{mode})"

    if lth == 0:
        tz.data = np.empty(0, dtype=np.float64)
        tn.data = np.empty(0, dtype=np.float64)
        te.data = np.empty(0, dtype=np.float64)
        return tz, tn, te

    half_w = max(1, int(round(0.5 * coherence_win_sec / dt)))
    finite = np.isfinite(z) & np.isfinite(n) & np.isfinite(e)
    if not np.any(finite):
        tz.data = np.zeros(lth, dtype=np.float64)
        tn.data = np.zeros(lth, dtype=np.float64)
        te.data = np.zeros(lth, dtype=np.float64)
        return tz, tn, te

    z_fin = np.where(finite, z, 0.0)
    n_fin = np.where(finite, n, 0.0)
    e_fin = np.where(finite, e, 0.0)

    # Subtract global finite mean first to eliminate catastrophic cancellation in prefix-sum covariance
    mu_z = float(np.mean(z[finite]))
    mu_n = float(np.mean(n[finite]))
    mu_e = float(np.mean(e[finite]))
    stacks_centered = np.vstack(
        [
            np.where(finite, z - mu_z, 0.0),
            np.where(finite, n - mu_n, 0.0),
            np.where(finite, e - mu_e, 0.0),
        ]
    )
    scale = float(np.max(np.abs(stacks_centered)))
    if scale > 0.0 and math.isfinite(scale):
        stacks = stacks_centered / scale
    else:
        tz.data = np.zeros(lth, dtype=np.float64)
        tn.data = np.zeros(lth, dtype=np.float64)
        te.data = np.zeros(lth, dtype=np.float64)
        return tz, tn, te

    # Precompute O(N) prefix sums for centered sliding-window 3x3 covariance matrices
    csum_cnt = np.zeros(lth + 1, dtype=np.float64)
    csum_cnt[1:] = np.cumsum(finite.astype(np.float64))

    csum1 = np.zeros((3, lth + 1), dtype=np.float64)
    csum1[:, 1:] = np.cumsum(stacks, axis=1)

    outer = stacks[:, None, :] * stacks[None, :, :]  # (3, 3, lth)
    csum2 = np.zeros((3, 3, lth + 1), dtype=np.float64)
    csum2[:, :, 1:] = np.cumsum(outer, axis=2)

    k_idx = np.arange(lth)
    k0 = np.maximum(0, k_idx - half_w)
    k1 = np.minimum(lth, k_idx + half_w + 1)
    cnt = csum_cnt[k1] - csum_cnt[k0]
    cnt_safe = np.maximum(1.0, cnt)

    sum1 = csum1[:, k1] - csum1[:, k0]  # (3, lth)
    mean1 = sum1 / cnt_safe[None, :]  # (3, lth)
    sum2 = csum2[:, :, k1] - csum2[:, :, k0]  # (3, 3, lth)
    covs = (sum2 / cnt_safe[None, None, :]) - (mean1[:, None, :] * mean1[None, :, :])  # (3, 3, lth)
    covs_batch = np.transpose(covs, (2, 0, 1))  # (lth, 3, 3)
    covs_batch = 0.5 * (covs_batch + np.swapaxes(covs_batch, 1, 2))
    covs_batch = np.where(np.isfinite(covs_batch), covs_batch, 0.0)

    # Vectorized eigendecomposition across all sliding windows
    evals, evecs = np.linalg.eigh(covs_batch)
    lam1 = np.maximum(0.0, evals[:, 2])
    lam2 = np.maximum(0.0, evals[:, 1])
    lam3 = np.maximum(0.0, evals[:, 0])
    e1 = evecs[:, :, 2]  # (lth, 3)

    valid_win = (cnt >= 1.0) & (lam1 > 1e-24)
    rect = np.where(valid_win, 1.0 - (lam2 + lam3) / np.maximum(2.0 * lam1, 1e-300), 0.0)
    w_rect = np.maximum(0.0, rect) ** linearity_power

    az_rad = math.radians(baz_deg)
    inc_rad = math.radians(inci_deg)
    d_p = np.array(
        [
            math.cos(inc_rad),
            -math.sin(inc_rad) * math.cos(az_rad),
            -math.sin(inc_rad) * math.sin(az_rad),
        ],
        dtype=np.float64,
    )

    vecs = np.column_stack([z_fin, n_fin, e_fin])  # (lth, 3)
    if mode == 1:
        res = vecs * w_rect[:, None]
    elif mode == 2:
        proj = np.sum(vecs * e1, axis=1)
        res = (proj * w_rect)[:, None] * e1
    elif mode == 3:
        align = np.abs(e1 @ d_p) ** linearity_power
        res = vecs * (w_rect * align)[:, None]
    elif mode == 4:
        h_norm = np.hypot(e1[:, 1], e1[:, 2])
        h_num = np.abs(-e1[:, 1] * math.cos(az_rad) - e1[:, 2] * math.sin(az_rad))
        h_align = np.where(h_norm > 1e-12, h_num / np.maximum(h_norm, 1e-30), 0.0)
        res = vecs * (w_rect * (h_align ** linearity_power))[:, None]
    else:
        s_align = (1.0 - np.abs(e1 @ d_p)) ** linearity_power
        res = vecs * (w_rect * s_align)[:, None]

    res[~finite] = 0.0
    tz.data = res[:, 0].copy()
    tn.data = res[:, 1].copy()
    te.data = res[:, 2].copy()
    return tz, tn, te
