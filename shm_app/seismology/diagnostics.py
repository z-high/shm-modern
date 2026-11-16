"""Pure Python/NumPy Statistical Diagnostics Engine for Seismic Inversions.

Implements M-05 requirements:
- RMSE, Weighted RMSE (WRMSE), MAD, and robust sigma_MAD (1.4826 * MAD).
- Huber M-estimator weights (k=1.345) and Tukey biweight (c=4.685).
- Reduced chi-square (chi2_red) and coefficient of determination (R^2).
- Model parameter covariance matrix C_m = s^2 (J^T W J + lambda I)^-1 via SVD / pseudo-inverse.
- Correlation matrix R_jk = C_jk / sqrt(C_jj * C_kk).
- Jacobian condition number kappa(J) = sigma_max / sigma_min.
- Horizontal 95% confidence error ellipse parameters (semi-major a, semi-minor b, strike azimuth theta)
  from 2x2 horizontal covariance submatrix using Chi2(2)=5.991 (scale=2.4477).
- Normality statistics (skewness, kurtosis, Jarque-Bera p-value) and 3-sigma outlier detection.

NO PyQt6 or GUI dependencies — pure mathematical calculations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def compute_rmse(residuals: Union[np.ndarray, Sequence[float]]) -> float:
    """Compute Root Mean Square Error: sqrt( (1/N) * sum(r_i^2) )."""
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_wrmse(
    residuals: Union[np.ndarray, Sequence[float]],
    weights: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute Weighted Root Mean Square Error: sqrt( sum(w_i * r_i^2) / sum(w_i) )."""
    r = np.asarray(residuals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if r.size == 0 or w.size == 0:
        return 0.0
    if r.shape != w.shape:
        raise ValueError(f"Shape mismatch: residuals {r.shape} vs weights {w.shape}")
    sum_w = float(np.sum(w))
    if sum_w <= 0.0:
        return compute_rmse(r)
    return float(np.sqrt(np.sum(w * (r ** 2)) / sum_w))


def compute_mad(residuals: Union[np.ndarray, Sequence[float]]) -> float:
    """Compute Median Absolute Deviation: median( |r_i - median(r)| )."""
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def compute_robust_sigma(residuals: Union[np.ndarray, Sequence[float]]) -> float:
    """Compute robust standard deviation estimator sigma_hat = 1.4826 * MAD."""
    mad = compute_mad(residuals)
    return float(1.4826 * mad)


def compute_reduced_chi2(
    residuals: Union[np.ndarray, Sequence[float]],
    uncertainties: Optional[Union[np.ndarray, Sequence[float]]] = None,
    dof: int = 1,
) -> float:
    """Compute reduced chi-square: (1 / nu) * sum( (r_i / sigma_i)^2 )."""
    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        return 0.0
    nu = max(1, int(dof))
    if uncertainties is None:
        return float(np.sum(r ** 2) / nu)
    sig = np.asarray(uncertainties, dtype=np.float64)
    if sig.shape != r.shape:
        raise ValueError(f"Shape mismatch: residuals {r.shape} vs uncertainties {sig.shape}")
    # Guard against zero uncertainties
    sig = np.where(sig > 1e-12, sig, 1e-12)
    return float(np.sum((r / sig) ** 2) / nu)


def compute_r_squared(
    observed: Union[np.ndarray, Sequence[float]],
    calculated: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute coefficient of determination R^2 = 1 - (SS_res / SS_tot)."""
    y_obs = np.asarray(observed, dtype=np.float64)
    y_calc = np.asarray(calculated, dtype=np.float64)
    if y_obs.size == 0 or y_calc.size == 0:
        return 0.0
    if y_obs.shape != y_calc.shape:
        raise ValueError(f"Shape mismatch: observed {y_obs.shape} vs calculated {y_calc.shape}")
    ss_res = float(np.sum((y_obs - y_calc) ** 2))
    mean_obs = float(np.mean(y_obs))
    ss_tot = float(np.sum((y_obs - mean_obs) ** 2))
    if ss_tot < 1e-14:
        return 1.0 if ss_res < 1e-14 else 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(-1.0, min(1.0, r2)))


def compute_huber_weights(
    residuals: Union[np.ndarray, Sequence[float]],
    delta: Optional[float] = None,
    k: float = 1.345,
) -> np.ndarray:
    """Compute Huber M-estimator weights:
    w(r) = 1.0 if |r| <= delta else delta / |r|
    where delta = k * sigma_hat (default k=1.345, standard 95% efficiency for normal distribution).
    """
    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        return np.array([], dtype=np.float64)
    if delta is None:
        sigma_hat = compute_robust_sigma(r)
        if sigma_hat < 1e-12:
            sigma_hat = float(np.std(r)) if r.size > 1 else 1.0
        if sigma_hat < 1e-12:
            sigma_hat = 1.0
        delta = k * sigma_hat
    delta = max(float(delta), 1e-12)

    abs_r = np.abs(r)
    weights = np.ones_like(r, dtype=np.float64)
    mask = abs_r > delta
    weights[mask] = delta / abs_r[mask]
    return weights


def compute_tukey_weights(
    residuals: Union[np.ndarray, Sequence[float]],
    c: float = 4.685,
) -> np.ndarray:
    """Compute Tukey biweight (bisquare) weights:
    w(r) = (1 - (r / (c * sigma_hat))^2)^2 if |r| <= c * sigma_hat else 0.0
    (default c=4.685, 95% efficiency on Gaussian errors).
    """
    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        return np.array([], dtype=np.float64)
    sigma_hat = compute_robust_sigma(r)
    if sigma_hat < 1e-12:
        sigma_hat = float(np.std(r)) if r.size > 1 else 1.0
    if sigma_hat < 1e-12:
        sigma_hat = 1.0
    cutoff = max(c * sigma_hat, 1e-12)

    weights = np.zeros_like(r, dtype=np.float64)
    abs_r = np.abs(r)
    mask = abs_r <= cutoff
    weights[mask] = (1.0 - (abs_r[mask] / cutoff) ** 2) ** 2
    return weights


def detect_outliers_3sigma(
    residuals: Union[np.ndarray, Sequence[float]],
    threshold: float = 3.0,
) -> np.ndarray:
    """Identify outliers beyond threshold * sigma_hat from median residual."""
    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        return np.array([], dtype=bool)
    med = float(np.median(r))
    sigma = compute_robust_sigma(r)
    if sigma < 1e-12:
        sigma = float(np.std(r)) if r.size > 1 else 1.0
    if sigma < 1e-12:
        sigma = 1.0
    return np.abs(r - med) > (threshold * sigma)


def compute_covariance_matrix(
    J: np.ndarray,
    W: Optional[Union[np.ndarray, Sequence[float]]] = None,
    s_sq: float = 1.0,
    damping: float = 1e-6,
) -> np.ndarray:
    """Compute model covariance matrix C_m = s^2 * (J^T W J + lambda I)^-1 via SVD / pseudo-inverse.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix of shape (N, P).
    W : Optional[np.ndarray]
        Weight vector (length N) or diagonal weight matrix (N, N). If None, identity.
    s_sq : float
        Scale factor s^2 = (sum w_i * r_i^2) / dof (sample residual variance).
    damping : float
        Levenberg-Marquardt / Tikhonov regularization parameter lambda >= 0.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape (P, P).
    """
    J_mat = np.asarray(J, dtype=np.float64)
    if J_mat.ndim != 2:
        raise ValueError(f"Jacobian must be a 2D matrix, got shape {J_mat.shape}")
    n_obs, n_params = J_mat.shape
    if n_params == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Compute normal matrix A = J^T W J
    if W is None:
        A = J_mat.T @ J_mat
    else:
        w_arr = np.asarray(W, dtype=np.float64)
        if w_arr.ndim == 1:
            if w_arr.shape[0] != n_obs:
                raise ValueError(f"Weight vector length {w_arr.shape[0]} != obs count {n_obs}")
            # Efficient diagonal multiplication: J^T @ (w[:, None] * J)
            A = J_mat.T @ (w_arr[:, np.newaxis] * J_mat)
        elif w_arr.ndim == 2:
            A = J_mat.T @ (w_arr @ J_mat)
        else:
            raise ValueError("W must be 1D vector or 2D matrix")

    # Add damping regularization
    if damping > 0.0:
        A += np.eye(n_params, dtype=np.float64) * float(damping)

    # Invert via Singular Value Decomposition for numerical stability
    u, s, vh = np.linalg.svd(A, hermitian=True)
    # Filter singular values
    max_s = float(np.max(s)) if s.size > 0 else 1.0
    thresh = max(1e-15, max_s * 1e-14)
    s_inv = np.zeros_like(s, dtype=np.float64)
    mask = s > thresh
    s_inv[mask] = 1.0 / s[mask]

    inv_A = (vh.T * s_inv) @ u.T
    s2 = max(float(s_sq), 1e-12)
    return inv_A * s2


def compute_correlation_matrix(cov: np.ndarray) -> np.ndarray:
    """Compute correlation matrix R_jk = C_jk / sqrt(C_jj * C_kk) from covariance matrix."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("Covariance matrix must be square (P, P)")
    p = c.shape[0]
    if p == 0:
        return np.zeros((0, 0), dtype=np.float64)

    diag = np.diag(c)
    # std deviations
    std = np.sqrt(np.maximum(0.0, diag))
    corr = np.zeros_like(c, dtype=np.float64)

    for i in range(p):
        for j in range(p):
            if i == j:
                corr[i, j] = 1.0
            elif std[i] > 1e-14 and std[j] > 1e-14:
                val = c[i, j] / (std[i] * std[j])
                corr[i, j] = max(-1.0, min(1.0, float(val)))
            else:
                corr[i, j] = 0.0
    return corr


def compute_condition_number(J: np.ndarray) -> float:
    """Compute condition number kappa(J) = sigma_max / sigma_min via SVD."""
    j_mat = np.asarray(J, dtype=np.float64)
    if j_mat.size == 0 or j_mat.ndim != 2:
        return 1.0
    s = np.linalg.svd(j_mat, compute_uv=False)
    if s.size == 0:
        return 1.0
    s_max = float(np.max(s))
    s_min = float(np.min(s))
    if s_min < 1e-15:
        return float("inf") if s_max > 1e-15 else 1.0
    return float(s_max / s_min)


def compute_confidence_ellipse_2d(
    cov_2x2: np.ndarray,
    dof: int = 20,
    confidence: float = 0.95,
    deg_to_km: float = 111.195,
) -> Tuple[float, float, float]:
    """Calculate horizontal confidence error ellipse parameters (semi_major, semi_minor, strike_deg).

    Parameters
    ----------
    cov_2x2 : np.ndarray
        2x2 horizontal covariance matrix [[C_yy, C_yx], [C_xy, C_xx]], where index 0 is Latitude
        (North/South, y) and index 1 is Longitude (East/West, x).
    dof : int
        Degrees of freedom (N - P). For Chi2(2)=5.991 at 95% confidence, scale = 2.4477.
    confidence : float
        Confidence level, default 0.95 (95%).
    deg_to_km : float
        Conversion factor from degrees to kilometers (111.195 km/deg). If cov_2x2 is already
        in km^2 or slowness^2, pass deg_to_km=1.0.

    Returns
    -------
    Tuple[float, float, float]
        (semi_major_km, semi_minor_km, strike_deg)
        - semi_major_km: semi-major axis length (km)
        - semi_minor_km: semi-minor axis length (km)
        - strike_deg: strike azimuth of the semi-major axis in degrees (0..180),
          measured clockwise from North towards East.
    """
    c = np.asarray(cov_2x2, dtype=np.float64)
    if c.shape != (2, 2):
        raise ValueError(f"Horizontal covariance must be 2x2, got {c.shape}")

    # Scale to kilometers if deg_to_km > 0 and != 1.0
    scale_dist = float(deg_to_km) if deg_to_km is not None else 1.0
    c_scaled = c * (scale_dist ** 2)

    # Symmetrize in case of minor numerical discrepancies
    c_sym = 0.5 * (c_scaled + c_scaled.T)

    # Eigendecomposition of symmetric 2x2 matrix
    evals, evecs = np.linalg.eigh(c_sym)

    # Sort descending so mu_1 >= mu_2
    if evals[1] >= evals[0]:
        mu1, mu2 = float(evals[1]), float(evals[0])
        v1 = evecs[:, 1]
    else:
        mu1, mu2 = float(evals[0]), float(evals[1])
        v1 = evecs[:, 0]

    mu1 = max(0.0, mu1)
    mu2 = max(0.0, mu2)

    # 95% confidence scaling factor:
    # Chi2(2) for 0.95 is 5.99146 -> sqrt(5.99146) = 2.447746...
    # If confidence != 0.95, quantile = -2 * ln(1 - confidence)
    conf = max(0.01, min(0.9999, float(confidence)))
    chi2_val = -2.0 * math.log(1.0 - conf)
    scale = math.sqrt(chi2_val)

    semi_major = float(scale * math.sqrt(mu1))
    semi_minor = float(scale * math.sqrt(mu2))

    # Strike azimuth theta of the semi-major axis:
    # v1 is [v_lat, v_lon] = [v_y, v_x].
    # Strike azimuth theta measured clockwise from North (+y) towards East (+x):
    # theta = atan2(v_x, v_y)
    v_y, v_x = float(v1[0]), float(v1[1])
    theta_rad = math.atan2(v_x, v_y)
    theta_deg = math.degrees(theta_rad) % 180.0
    if theta_deg < 0.0:
        theta_deg += 180.0

    return (round(semi_major, 3), round(semi_minor, 3), round(theta_deg, 1))


def compute_normality_stats(residuals: Union[np.ndarray, Sequence[float]]) -> Dict[str, float]:
    """Compute sample skewness, excess kurtosis, and Jarque-Bera normality test p-value.

    Under the null hypothesis of Gaussian residuals, the Jarque-Bera statistic
    JB = (N / 6) * (S^2 + (K^2 / 4)) asymptotically follows Chi2(2).
    For Chi2(2), the survival function has exact closed form: p = exp(-JB / 2).
    """
    r = np.asarray(residuals, dtype=np.float64)
    n = r.size
    if n < 4:
        return {"skewness": 0.0, "kurtosis": 0.0, "p_value": 1.0, "jb_stat": 0.0}

    mean_r = float(np.mean(r))
    diff = r - mean_r
    s2 = float(np.mean(diff ** 2))
    if s2 < 1e-14:
        return {"skewness": 0.0, "kurtosis": 0.0, "p_value": 1.0, "jb_stat": 0.0}

    s = math.sqrt(s2)
    m3 = float(np.mean(diff ** 3))
    m4 = float(np.mean(diff ** 4))

    skewness = m3 / (s ** 3)
    kurtosis = (m4 / (s ** 4)) - 3.0  # Excess kurtosis

    jb_stat = (n / 6.0) * (skewness ** 2 + (kurtosis ** 2) / 4.0)
    p_val = math.exp(-0.5 * max(0.0, jb_stat))

    return {
        "skewness": round(float(skewness), 4),
        "kurtosis": round(float(kurtosis), 4),
        "p_value": round(float(p_val), 4),
        "jb_stat": round(float(jb_stat), 4),
    }
