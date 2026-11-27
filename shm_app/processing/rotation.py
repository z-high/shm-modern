"""2D, 3D ray-aligned (LQT), ZRT, and triaxial STS-2 (UVW) coordinate rotations."""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np

from shm_app.core.models import Trace

MAX_POLARIZATION_ALIGN_SAMPLES = 10_000_000


def _align_traces_for_polarization(*traces: Trace) -> Tuple[float, float, List[np.ndarray]]:
    """Validate matching delta and align start_epoch across multi-component traces."""
    if not traces:
        raise ValueError("At least one trace is required")

    for tr in traces:
        if not math.isfinite(tr.delta) or tr.delta <= 0.0:
            raise ValueError(f"Trace {tr.station}.{tr.comp} has invalid delta={tr.delta}")
        if not math.isfinite(tr.start_epoch):
            raise ValueError(f"Trace {tr.station}.{tr.comp} has non-finite start_epoch={tr.start_epoch}")

    dt = float(traces[0].delta)
    for tr in traces[1:]:
        if not math.isclose(float(tr.delta), dt, rel_tol=1e-5, abs_tol=1e-9):
            raise ValueError(
                f"Mismatched sample intervals (delta) across components: "
                f"{[float(t.delta) for t in traces]}"
            )

    starts = [float(tr.start_epoch) for tr in traces]
    ref_start = starts[0]
    aligned_starts = all(abs(s - ref_start) <= 0.01 * dt for s in starts[1:])

    if any(tr.length == 0 for tr in traces):
        if not aligned_starts:
            raise ValueError("Cannot align empty trace with mismatched start_epoch")
        return ref_start, dt, [np.empty(0, dtype=np.float64) for _ in traces]

    if aligned_starts:
        n_len = min(tr.length for tr in traces)
        return ref_start, dt, [np.asarray(tr.data[:n_len], dtype=np.float64) for tr in traces]

    t_start = max(starts)
    t_end = min(float(tr.end_epoch) for tr in traces)
    if t_end < t_start - 0.01 * dt:
        raise ValueError(
            f"Traces have non-overlapping time windows (max start={t_start:.3f}, min end={t_end:.3f})"
        )

    n_ov = max(1, int(math.floor(max(0.0, t_end - t_start) / dt + 1e-9)) + 1)
    if n_ov > MAX_POLARIZATION_ALIGN_SAMPLES:
        raise ValueError(
            f"Aligned trace overlap requires {n_ov} samples, exceeding maximum {MAX_POLARIZATION_ALIGN_SAMPLES}"
        )

    grid_t = t_start + np.arange(n_ov, dtype=np.float64) * dt
    arrays: List[np.ndarray] = []
    for tr in traces:
        tr_t = float(tr.start_epoch) + np.arange(tr.length, dtype=np.float64) * float(tr.delta)
        arrays.append(np.interp(grid_t, tr_t, np.asarray(tr.data, dtype=np.float64)))
    return t_start, dt, arrays


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
    t_start, _dt, (n_dat, e_dat) = _align_traces_for_polarization(tr_n, tr_e)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)

    out_r = tr_n.copy()
    out_t = tr_e.copy()
    out_r.start_epoch = t_start
    out_t.start_epoch = t_start
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
    t_start, _dt, (z_dat, n_dat, e_dat) = _align_traces_for_polarization(tr_z, tr_n, tr_e)
    phi = baz_deg - 180.0
    a = math.radians(phi)
    ca, sa = math.cos(a), math.sin(a)
    out_z = tr_z.copy()
    out_r = tr_n.copy()
    out_t = tr_e.copy()
    out_z.start_epoch = t_start
    out_r.start_epoch = t_start
    out_t.start_epoch = t_start
    out_z.data = z_dat.copy()
    out_r.data = n_dat * ca + e_dat * sa
    out_t.data = -n_dat * sa + e_dat * ca
    out_r.comp = "R"
    out_t.comp = "T"
    out_r.azimuth = phi
    out_t.azimuth = phi
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
    t_start, _dt, (z, n, e) = _align_traces_for_polarization(tr_z, tr_n, tr_e)
    az = math.radians(azimuth_deg)
    inc = math.radians(incidence_deg)

    ci, si = math.cos(inc), math.sin(inc)
    ca, sa = math.cos(az), math.sin(az)

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
    out_l.start_epoch = t_start
    out_q.start_epoch = t_start
    out_t.start_epoch = t_start
    out_l.data, out_l.comp, out_l.azimuth, out_l.inci = l_dat, "L", azimuth_deg, incidence_deg
    out_q.data, out_q.comp, out_q.azimuth, out_q.inci = q_dat, "Q", azimuth_deg, incidence_deg
    out_t.data, out_t.comp, out_t.azimuth, out_t.inci = t_dat, "T", azimuth_deg, incidence_deg
    return out_l, out_q, out_t


def rotate_STS2_zne_to_uvw(tr_z: Trace, tr_n: Trace, tr_e: Trace, inverse: bool = False) -> Tuple[Trace, Trace, Trace]:
    """Transform between geographic (Z, N, E) and STS-2 triaxial sensor (U, V, W) coordinates."""
    t_start, _dt, (d1, d2, d3) = _align_traces_for_polarization(tr_z, tr_n, tr_e)
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
    t1.start_epoch = t_start
    t2.start_epoch = t_start
    t3.start_epoch = t_start
    if not inverse:
        enz = np.vstack([d3, d2, d1])
        res = mat @ enz
        t1.data, t1.comp = res[0], "U"
        t2.data, t2.comp = res[1], "V"
        t3.data, t3.comp = res[2], "W"
    else:
        uvw = np.vstack([d1, d2, d3])
        res = mat.T @ uvw
        t1.data, t1.comp = res[2], "Z"
        t2.data, t2.comp = res[1], "N"
        t3.data, t3.comp = res[0], "E"
    return t1, t2, t3
