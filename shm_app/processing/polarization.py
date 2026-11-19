"""2D & 3D coordinate rotations, covariance eigendecomposition (MDIR),
particle motion hodograms, and sliding-window polarization filtering matching
shmath.c, pmotion.c, and polfil.c.
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
from shm_app.seismology.diagnostics import (
    compute_mad,
    compute_normality_stats,
    compute_reduced_chi2,
)


def rotate_2d(
    tr_n: Trace,
    tr_e: Trace,
    angle_deg: float,
    labels: Tuple[str, str] = ("R", "T"),
) -> Tuple[Trace, Trace]:
    """2D horizontal rotation matching mt_rot2 in shmath.c.

    Note: In SHM, for backazimuth BAZ, rotation to Radial/Transverse (L->Q or R->T away from source)
    uses angle = BAZ - 180 (or for BAZ directly: R = -N*cos(BAZ) - E*sin(BAZ), T = N*sin(BAZ) - E*cos(BAZ)).
    Here angle_deg is the clockwise rotation angle of the North/East coordinate axes:
        r =  n * cos(a) + e * sin(a)
        t = -n * sin(a) + e * cos(a)
    """
    n_len = min(tr_n.length, tr_e.length)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)

    n_dat = tr_n.data[:n_len]
    e_dat = tr_e.data[:n_len]

    out_r = tr_n.copy()
    out_t = tr_e.copy()
    out_r.data = n_dat * ca + e_dat * sa
    out_t.data = -n_dat * sa + e_dat * ca
    out_r.comp = labels[0]
    out_t.comp = labels[1]
    out_r.azimuth = angle_deg
    out_t.azimuth = angle_deg
    return out_r, out_t


def rotate_zne_to_zrt(tr_z: Trace, tr_n: Trace, tr_e: Trace, baz_deg: float) -> Tuple[Trace, Trace, Trace]:
    """Rotate (Z, N, E) to (Z, R, T) given event backazimuth baz_deg (0..360)."""
    # Radial points from source to station (azimuth = baz_deg - 180)
    phi = baz_deg - 180.0
    out_z = tr_z.copy()
    out_r, out_t = rotate_2d(tr_n, tr_e, phi, labels=("R", "T"))
    return out_z, out_r, out_t


def rotate_3d_lqt(
    tr_z: Trace,
    tr_n: Trace,
    tr_e: Trace,
    azimuth_deg: float,
    incidence_deg: float,
) -> Tuple[Trace, Trace, Trace]:
    """3D rotation from (Z, N, E) into ray-aligned (L, Q, T) coordinates matching mt_rot3 in shmath.c.

    Parameters
    ----------
    azimuth_deg : backazimuth in degrees (or propagation azimuth following SHM convention)
    incidence_deg : incidence angle in degrees (0 = vertical up along Z)
    """
    lth = min(tr_z.length, tr_n.length, tr_e.length)
    az = math.radians(azimuth_deg)
    inc = math.radians(incidence_deg)

    ci, si = math.cos(inc), math.sin(inc)
    ca, sa = math.cos(az), math.sin(az)

    z = tr_z.data[:lth]
    n = tr_n.data[:lth]
    e = tr_e.data[:lth]

    # Exact mt_rot3 transformation matrix from shmath.c:
    # l =  z * cos(i) - n * sin(i)*cos(a) - e * sin(i)*sin(a)
    # q =  z * sin(i) + n * cos(i)*cos(a) + e * cos(i)*sin(a)
    # t =               n * sin(a)        - e * cos(a)
    l_dat = z * ci - n * si * ca - e * si * sa
    q_dat = z * si + n * ci * ca + e * ci * sa
    t_dat = n * sa - e * ca

    out_l = tr_z.copy()
    out_q = tr_n.copy()
    out_t = tr_e.copy()
    out_l.data, out_l.comp, out_l.azimuth, out_l.inci = l_dat, "L", azimuth_deg, incidence_deg
    out_q.data, out_q.comp, out_q.azimuth, out_q.inci = q_dat, "Q", azimuth_deg, incidence_deg
    out_t.data, out_t.comp, out_t.azimuth, out_t.inci = t_dat, "T", azimuth_deg, incidence_deg
    return out_l, out_q, out_t


def rotate_STS2_zne_to_uvw(tr_z: Trace, tr_n: Trace, tr_e: Trace, inverse: bool = False) -> Tuple[Trace, Trace, Trace]:
    """Transform between geographic (Z, N, E) and STS-2 triaxial sensor (U, V, W) coordinates."""
    lth = min(tr_z.length, tr_n.length, tr_e.length)
    s23 = math.sqrt(2.0 / 3.0)
    s16 = 1.0 / math.sqrt(6.0)
    s12 = 1.0 / math.sqrt(2.0)
    s13 = 1.0 / math.sqrt(3.0)

    # STS-2 matrix M where [U, V, W]^T = M * [E, N, Z]^T
    mat = np.array(
        [
            [-s23, 0.0, s13],
            [s16, s12, s13],
            [s16, -s12, s13],
        ],
        dtype=np.float64,
    )
    t1, t2, t3 = tr_z.copy(), tr_n.copy(), tr_e.copy()
    if not inverse:
        enz = np.vstack([tr_e.data[:lth], tr_n.data[:lth], tr_z.data[:lth]])
        res = mat @ enz
        t1.data, t1.comp = res[0], "U"
        t2.data, t2.comp = res[1], "V"
        t3.data, t3.comp = res[2], "W"
    else:
        uvw = np.vstack([tr_z.data[:lth], tr_n.data[:lth], tr_e.data[:lth]])
        res = mat.T @ uvw
        t1.data, t1.comp = res[2], "Z"
        t2.data, t2.comp = res[1], "N"
        t3.data, t3.comp = res[0], "E"
    return t1, t2, t3


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
    lth = min(tr_z.length, tr_n.length, tr_e.length)
    dt = tr_z.delta if tr_z.delta > 0 else 0.05
    i0, i1 = _polarization_window_indices(lth, dt, t_lo, t_hi)

    if i1 <= i0:
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

    z = tr_z.data[i0:i1] - np.mean(tr_z.data[i0:i1])
    n = tr_n.data[i0:i1] - np.mean(tr_n.data[i0:i1])
    e = tr_e.data[i0:i1] - np.mean(tr_e.data[i0:i1])

    mat = np.vstack([z, n, e])
    cov = (mat @ mat.T) / max(1, z.size)
    evals, evecs = np.linalg.eigh(cov)
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
    rectilinearity = 1.0 - (lam2 + lam3) / (2.0 * lam1) if lam1 > 1e-30 else 0.0
    planarity = 1.0 - (2.0 * lam3) / (lam1 + lam2) if (lam1 + lam2) > 1e-30 else 0.0

    # Orthogonal trajectory distance d_k = ||u_k - (u_k . v1) v1|| = sqrt(||u_k||^2 - (u_k . v1)^2)
    u_sq = np.sum(mat ** 2, axis=0)
    proj = v1 @ mat
    perp = mat - np.outer(v1, proj)
    d_k_sq = np.sum(perp ** 2, axis=0)
    rms_pol = float(np.sqrt(np.mean(d_k_sq)))
    rms_total = float(np.sqrt(np.mean(u_sq)))
    rms_pol_norm = (rms_pol / rms_total) if rms_total > 1e-15 else 0.0

    # Jurkevics (1988) confidence cones
    sigma_theta_deg = math.degrees(math.atan(math.sqrt(lam2 / max(lam1, 1e-15))))
    sigma_phi_deg = math.degrees(math.atan(math.sqrt(lam3 / max(lam1, 1e-15))))

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
    lth = min(tr_z.length, tr_n.length, tr_e.length)
    dt = tr_z.delta if tr_z.delta > 0 else 0.05
    i0, i1 = _polarization_window_indices(lth, dt, t_lo, t_hi)

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

    if i1 <= i0:
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

    z_win = np.asarray(tr_z.data[i0:i1] - np.mean(tr_z.data[i0:i1]), dtype=np.float64)
    n_win = np.asarray(tr_n.data[i0:i1] - np.mean(tr_n.data[i0:i1]), dtype=np.float64)
    e_win = np.asarray(tr_e.data[i0:i1] - np.mean(tr_e.data[i0:i1]), dtype=np.float64)
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
    dof = max(1, n_samples - 3)
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
        converged=bool(lam1 > 1e-30),
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
    """Apply 3-component sliding-window covariance polarization filter matching polfil.c.

    Modes:
    1 : Weight original (Z, N, E) by rectilinearity R(k)^p
    2 : Project (Z, N, E) onto instantaneous principal eigenvector e1(k) * R(k)^p
    3 : Weight by directional cosine |e1(k) . d_fixed|^p * R(k)^p
    4 : P-wave azimuth filter (weight by horizontal alignment with baz_deg)
    5 : S-wave transverse filter (weight by perpendicularity to P direction)
    """
    lth = min(tr_z.length, tr_n.length, tr_e.length)
    dt = tr_z.delta if tr_z.delta > 0 else 0.05
    half_w = max(1, int(round(0.5 * coherence_win_sec / dt)))

    z = tr_z.data[:lth]
    n = tr_n.data[:lth]
    e = tr_e.data[:lth]

    # Precompute cumulative products for O(N) sliding-window 3x3 covariance matrices
    stacks = np.vstack([z, n, e])  # (3, lth)
    # Outer products at each sample: (3, 3, lth)
    outer = stacks[:, None, :] * stacks[None, :, :]
    csum = np.zeros((3, 3, lth + 1), dtype=np.float64)
    csum[:, :, 1:] = np.cumsum(outer, axis=2)

    out_z = np.zeros(lth, dtype=np.float64)
    out_n = np.zeros(lth, dtype=np.float64)
    out_e = np.zeros(lth, dtype=np.float64)

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

    for k in range(lth):
        k0 = max(0, k - half_w)
        k1 = min(lth, k + half_w + 1)
        cov = (csum[:, :, k1] - csum[:, :, k0]) / max(1, k1 - k0)
        evals, evecs = np.linalg.eigh(cov)
        lam1 = max(0.0, float(evals[2]))
        lam2 = max(0.0, float(evals[1]))
        lam3 = max(0.0, float(evals[0]))
        e1 = evecs[:, 2]

        rect = (1.0 - (lam2 + lam3) / (2.0 * lam1)) if lam1 > 1e-30 else 0.0
        w_rect = max(0.0, rect) ** linearity_power

        vec_k = np.array([z[k], n[k], e[k]], dtype=np.float64)
        if mode == 1:
            res = vec_k * w_rect
        elif mode == 2:
            proj = float(np.dot(vec_k, e1))
            res = (proj * w_rect) * e1
        elif mode == 3:
            align = abs(float(np.dot(e1, d_p))) ** linearity_power
            res = vec_k * (w_rect * align)
        elif mode == 4:
            # P-azimuth weighting
            h_norm = math.hypot(e1[1], e1[2])
            h_align = abs((-e1[1] * math.cos(az_rad) - e1[2] * math.sin(az_rad)) / h_norm) if h_norm > 1e-12 else 0.0
            res = vec_k * (w_rect * (h_align ** linearity_power))
        else:
            # S-wave (perpendicular to P direction)
            s_align = (1.0 - abs(float(np.dot(e1, d_p)))) ** linearity_power
            res = vec_k * (w_rect * s_align)

        out_z[k], out_n[k], out_e[k] = res[0], res[1], res[2]

    tz, tn, te = tr_z.copy(), tr_n.copy(), tr_e.copy()
    tz.data, tn.data, te.data = out_z, out_n, out_e
    tz.opinfo = f"POLFIL(M{mode})"
    tn.opinfo = f"POLFIL(M{mode})"
    te.opinfo = f"POLFIL(M{mode})"
    return tz, tn, te
