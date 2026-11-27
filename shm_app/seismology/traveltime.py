"""Travel-time table (.TTT) loader, 2D (distance, depth) interpolator, slowness calculator,
slowness/differential-time distance inversion, and depth-phase solver matching ptusrdef.c.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np

from shm_app import SHM_DEFAULT_ROOT
from shm_app.seismology.geodesy import sphere_add, sphere_diff

logger = logging.getLogger(__name__)


@dataclass
class TravelTimeTable:
    """In-memory representation of an SH .TTT phase travel-time table."""

    phase: str
    comment: str
    min_dist: float
    max_dist: float
    depths: np.ndarray  # shape (M,)
    distances: np.ndarray  # shape (N,)
    times: np.ndarray  # shape (N, M), with <= 0 or NaN indicating shadow/unobserved

    def travel_time(self, dist_deg: float, depth_km: float) -> Optional[float]:
        """2D bilinear interpolation of travel time T(dist_deg, depth_km) in seconds."""
        if self.distances.size == 0 or self.depths.size == 0:
            return None
        if dist_deg < self.min_dist - 0.5 or dist_deg > self.max_dist + 0.5:
            return None

        d_clamped = max(float(self.distances[0]), min(float(self.distances[-1]), float(dist_deg)))
        z_clamped = max(float(self.depths[0]), min(float(self.depths[-1]), float(depth_km)))

        # Find bounding indices in distance
        i1 = int(np.searchsorted(self.distances, d_clamped, side="right"))
        i0 = max(0, min(self.distances.size - 1, i1 - 1))
        i1 = min(self.distances.size - 1, i1)

        # Find bounding indices in depth
        j1 = int(np.searchsorted(self.depths, z_clamped, side="right"))
        j0 = max(0, min(self.depths.size - 1, j1 - 1))
        j1 = min(self.depths.size - 1, j1)

        t00 = self.times[i0, j0]
        t10 = self.times[i1, j0]
        t01 = self.times[i0, j1]
        t11 = self.times[i1, j1]

        valid = [v for v in (t00, t10, t01, t11) if v > 0.0 and not math.isnan(v)]
        if not valid:
            return None

        # Replace any invalid corner with nearest valid corner
        fallback = valid[0]
        if t00 <= 0 or math.isnan(t00):
            t00 = fallback
        if t10 <= 0 or math.isnan(t10):
            t10 = t00
        if t01 <= 0 or math.isnan(t01):
            t01 = t00
        if t11 <= 0 or math.isnan(t11):
            t11 = t10

        d0, d1 = float(self.distances[i0]), float(self.distances[i1])
        z0, z1 = float(self.depths[j0]), float(self.depths[j1])
        u = (d_clamped - d0) / (d1 - d0) if d1 > d0 else 0.0
        v = (z_clamped - z0) / (z1 - z0) if z1 > z0 else 0.0

        val = (1.0 - u) * (1.0 - v) * t00 + u * (1.0 - v) * t10 + (1.0 - u) * v * t01 + u * v * t11
        return float(val) if val > 0.0 else None

    def slowness(self, dist_deg: float, depth_km: float) -> Optional[float]:
        """Compute horizontal slowness p = dT / dDelta (sec/deg) matching pt_slowness."""
        span = float(self.max_dist) - float(self.min_dist)
        if span <= 0.0:
            return None
        step = min(0.5, 0.25 * span)
        d_lo = max(self.min_dist, dist_deg - step)
        d_hi = min(self.max_dist, dist_deg + step)
        if d_hi <= d_lo:
            return None
        t_lo = self.travel_time(d_lo, depth_km)
        t_hi = self.travel_time(d_hi, depth_km)
        if t_lo is None or t_hi is None:
            return None
        return abs(t_hi - t_lo) / (d_hi - d_lo)


def encode_ttt_filename(phase: str) -> str:
    """Convert phase name into SHM's uppercase .TTT filename convention (lowercase 'c' -> 'VC').

    Examples:
        'P' -> 'P.TTT'
        'pP' -> 'VPP.TTT'
        'sP' -> 'VSP.TTT'
        'PcP' -> 'PVCVP.TTT'
        'PKPdf' -> 'PKPVDVF.TTT'
    """
    out: List[str] = []
    for ch in phase.strip():
        if ch.islower():
            out.append("V" + ch.upper())
        else:
            out.append(ch)
    return "".join(out) + ".TTT"


def decode_ttt_filename(filename: str) -> str:
    """Decode an SHM .TTT filename back into a case-sensitive phase name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    out: List[str] = []
    i = 0
    while i < len(base):
        if base[i] == "V" and i + 1 < len(base):
            out.append(base[i + 1].lower())
            i += 2
        else:
            out.append(base[i])
            i += 1
    return "".join(out)


def parse_ttt_file(filepath: str, phase_name: Optional[str] = None) -> TravelTimeTable:
    """Parse an SH .TTT travel-time table file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Travel-time table not found: {filepath}")

    with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    comments = [ln[1:].strip() for ln in lines if ln.startswith("!")]
    data_lines = [ln for ln in lines if not ln.startswith("!")]
    if len(data_lines) < 3 or not data_lines[0].upper().startswith("TTT"):
        raise ValueError(f"Invalid .TTT file header in {filepath}")

    # Tokenize everything after line 0 ("TTT")
    tokens: List[float] = []
    for ln in data_lines[1:]:
        for tok in ln.split():
            try:
                tokens.append(float(tok))
            except ValueError:
                pass

    if len(tokens) < 3:
        raise ValueError(f"Truncated .TTT header metadata in {filepath}")

    min_dist = tokens[0]
    max_dist = tokens[1]
    if not (math.isfinite(min_dist) and math.isfinite(max_dist) and math.isfinite(tokens[2])):
        raise ValueError(f"Non-finite .TTT header values in {filepath}")
    n_depths = int(round(tokens[2]))
    if n_depths <= 0:
        raise ValueError(f"Invalid depth count ({n_depths}) in .TTT file {filepath}")
    if len(tokens) < 3 + n_depths:
        raise ValueError(
            f"Truncated depth header in {filepath}: expected {n_depths} depths, got {len(tokens) - 3}"
        )
    depths = np.array(tokens[3 : 3 + n_depths], dtype=np.float64)

    rem = tokens[3 + n_depths :]
    row_size = 1 + n_depths
    n_rows = len(rem) // row_size
    if n_rows <= 0:
        raise ValueError(f"No travel-time rows found in .TTT file {filepath}")
    arr = np.array(rem[: n_rows * row_size], dtype=np.float64).reshape(n_rows, row_size)
    distances = arr[:, 0]
    times = arr[:, 1:]

    return TravelTimeTable(
        phase=phase_name or decode_ttt_filename(filepath),
        comment=" | ".join(comments),
        min_dist=min_dist,
        max_dist=max_dist,
        depths=depths,
        distances=distances,
        times=times,
    )


class TravelTimeManager:
    """Caches and queries all .TTT travel-time tables and .VEL velocity models in SHM."""

    STANDARD_PHASES = [
        "P", "S", "pP", "sP", "pS", "sS", "PP", "SS", "PPP", "SSS",
        "PcP", "ScS", "PcS", "ScP", "PKP", "PKPdf", "PKPab", "PKPbc", "PKiKP",
        "pPKPdf", "sPKPdf", "SKS", "SKKS", "SKKP", "PKKP", "Pdiff", "Sdiff",
        "Pg", "Sg", "Pn", "Sn", "Pb", "Sb", "Lg", "Rg",
    ]

    def __init__(self, inputs_dir: Optional[str] = None) -> None:
        self.inputs_dir = inputs_dir or os.path.join(SHM_DEFAULT_ROOT, "inputs")
        self._cache: Dict[str, TravelTimeTable] = {}
        self._dir_map: Dict[str, str] = {}
        self._scan_directory()

    def _scan_directory(self) -> None:
        if not os.path.isdir(self.inputs_dir):
            return
        for fn in os.listdir(self.inputs_dir):
            if fn.upper().endswith(".TTT"):
                self._dir_map[fn.upper()] = os.path.join(self.inputs_dir, fn)

    def available_phases(self) -> List[str]:
        phases = [decode_ttt_filename(fn) for fn in sorted(self._dir_map.keys())]
        # Ensure standard phases are listed first
        ordered = [p for p in self.STANDARD_PHASES if p in phases or self._resolve_path(p) is not None]
        for p in phases:
            if p not in ordered:
                ordered.append(p)
        return ordered

    def _resolve_path(self, phase: str) -> Optional[str]:
        enc = encode_ttt_filename(phase).upper()
        if enc in self._dir_map:
            return self._dir_map[enc]
        # Aliases: e.g. 'P' -> 'P+Pdiff.TTT' / 'P-PURE.TTT', 'S' -> 'S+Sdiff.TTT' / 'S-PURE.TTT'
        aliases = {
            "P": ["P.TTT", "P+PN+PDIFF.TTT", "P+PDIFF.TTT", "P-PURE.TTT", "P+PVDVI.TTT"],
            "S": ["S.TTT", "S+SDIFF.TTT", "S-PURE.TTT"],
            "PKP": ["PKPVDVF.TTT", "PKPAB.TTT", "PKPBC.TTT"],
            "SKS": ["SKSAC.TTT", "SKSVAVC.TTT", "SKS.TTT"],
        }
        for cand in aliases.get(phase, []):
            if cand.upper() in self._dir_map:
                return self._dir_map[cand.upper()]
        return None

    def get_table(self, phase: str) -> Optional[TravelTimeTable]:
        if phase in self._cache:
            return self._cache[phase]
        fpath = self._resolve_path(phase)
        if fpath and os.path.isfile(fpath):
            try:
                tbl = parse_ttt_file(fpath, phase_name=phase)
                self._cache[phase] = tbl
                return tbl
            except Exception as exc:
                msg = f"Failed to parse travel-time table {fpath} for phase {phase!r}: {exc}"
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return None
        return None

    def travel_time(self, phase: str, dist_deg: float, depth_km: float = 33.0) -> Optional[float]:
        """Return travel time in seconds for phase at (dist_deg, depth_km)."""
        # Local crustal phases Pg, Sg, Pn, Sn fallback if outside or missing table
        tbl = self.get_table(phase)
        if tbl is not None:
            t = tbl.travel_time(dist_deg, depth_km)
            if t is not None:
                return t
        p_up = phase.strip()
        if p_up == "P" and dist_deg > 98.0:
            pdiff_tbl = self.get_table("Pdiff")
            if pdiff_tbl is not None:
                t_diff = pdiff_tbl.travel_time(dist_deg, depth_km)
                if t_diff is not None:
                    return t_diff
            return None
        if p_up == "S" and dist_deg > 98.0:
            sdiff_tbl = self.get_table("Sdiff")
            if sdiff_tbl is not None:
                t_diff = sdiff_tbl.travel_time(dist_deg, depth_km)
                if t_diff is not None:
                    return t_diff
            return None
        # Analytic regional/local fallback for Pg, Sg, Pn, Sn, Lg (strictly local/regional distances)
        dist_km = dist_deg * 111.1949266
        hypo_km = math.hypot(dist_km, max(0.0, depth_km))
        if p_up in ("Pg", "P") and dist_deg <= 20.0:
            return hypo_km / 5.85
        if p_up in ("Sg", "S") and dist_deg <= 20.0:
            return hypo_km / 3.46
        if p_up == "Pn" and dist_deg <= 20.0:
            return 6.8 + dist_km / 8.04
        if p_up == "Sn" and dist_deg <= 20.0:
            return 11.5 + dist_km / 4.48
        if p_up == "Lg" and dist_deg <= 30.0:
            return hypo_km / 3.50
        if p_up == "Rg" and dist_deg <= 30.0:
            return dist_km / 3.05
        return None

    def slowness(self, phase: str, dist_deg: float, depth_km: float = 33.0) -> Optional[float]:
        """Return horizontal slowness p (s/deg) for phase at (dist_deg, depth_km)."""
        tbl = self.get_table(phase)
        if tbl is not None:
            s = tbl.slowness(dist_deg, depth_km)
            if s is not None:
                return s
        p_up = phase.strip()
        if p_up == "Pg":
            return 111.195 / 5.85
        if p_up == "Sg":
            return 111.195 / 3.46
        if p_up == "Pn":
            return 111.195 / 8.04
        if p_up == "Sn":
            return 111.195 / 4.48
        return None

    def distance_from_slowness(self, phase: str, slowness_s_deg: float, depth_km: float = 33.0) -> Optional[float]:
        """Invert slowness p(Delta, h) = slowness_s_deg for epicentral distance Delta (deg) matching pt_distance."""
        tbl = self.get_table(phase) or self.get_table("P")
        if tbl is None:
            return None

        span = float(tbl.max_dist) - float(tbl.min_dist)
        if span <= 0.0:
            return float(tbl.min_dist) if tbl.distances.size > 0 else None
        margin = min(0.5, 0.1 * span)
        d_lo = float(tbl.min_dist) + margin
        d_hi = float(tbl.max_dist) - margin
        if d_hi <= d_lo:
            d_lo, d_hi = float(tbl.min_dist), float(tbl.max_dist)

        dists = np.linspace(d_lo, d_hi, 360)
        slows: List[Tuple[float, float]] = []
        for d in dists:
            s = tbl.slowness(float(d), depth_km)
            if s is not None and s > 0:
                slows.append((float(d), s))
        if not slows:
            return None

        # Find best matching distance where slowness crosses slowness_s_deg (or closest extremum)
        best_d = slows[0][0]
        best_diff = abs(slows[0][1] - slowness_s_deg)
        for i in range(len(slows) - 1):
            d0, s0 = slows[i]
            d1, s1 = slows[i + 1]
            if (s0 - slowness_s_deg) * (s1 - slowness_s_deg) <= 0.0 and s0 != s1:
                frac = (slowness_s_deg - s0) / (s1 - s0)
                return d0 + frac * (d1 - d0)
            if abs(s1 - slowness_s_deg) < best_diff:
                best_diff = abs(s1 - slowness_s_deg)
                best_d = d1
        return best_d

    def distance_from_diff_time(
        self,
        phase1: str,
        phase2: str,
        dt_observed: float,
        depth_km: float = 33.0,
    ) -> Optional[float]:
        """Solve T(phase2, Delta, h) - T(phase1, Delta, h) = |dt_observed| for distance Delta (deg)."""
        dt_abs = abs(dt_observed)
        d_grid = np.linspace(0.5, 165.0, 660)
        pairs: List[Tuple[float, float]] = []
        for d in d_grid:
            t1 = self.travel_time(phase1, float(d), depth_km)
            t2 = self.travel_time(phase2, float(d), depth_km)
            if t1 is not None and t2 is not None:
                diff = abs(t2 - t1)
                if diff > 0:
                    pairs.append((float(d), diff))
        if not pairs:
            return None

        for i in range(len(pairs) - 1):
            d0, v0 = pairs[i]
            d1, v1 = pairs[i + 1]
            if (v0 - dt_abs) * (v1 - dt_abs) <= 0.0 and v1 != v0:
                return d0 + (dt_abs - v0) / (v1 - v0) * (d1 - d0)

        best = min(pairs, key=lambda item: abs(item[1] - dt_abs))
        return best[0]

    def depth_from_depth_phase(
        self,
        main_phase: str,
        depth_phase: str,
        dt_observed: float,
        dist_deg: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Solve T(depth_phase, dist_deg, h) - T(main_phase, dist_deg, h) = |dt_observed| for depth h (km)
        matching pt_depth in ptusrdef.c. Returns (depth_km, depth_error_km).
        """
        dt_abs = abs(dt_observed)
        z_grid = np.linspace(0.0, 700.0, 281)
        pairs: List[Tuple[float, float]] = []
        for z in z_grid:
            t_m = self.travel_time(main_phase, dist_deg, float(z))
            t_d = self.travel_time(depth_phase, dist_deg, float(z))
            if t_m is not None and t_d is not None and t_d > t_m:
                pairs.append((float(z), t_d - t_m))

        if not pairs:
            return None, None

        for i in range(len(pairs) - 1):
            z0, v0 = pairs[i]
            z1, v1 = pairs[i + 1]
            if (v0 - dt_abs) * (v1 - dt_abs) <= 0.0 and v1 != v0:
                slope = abs((z1 - z0) / (v1 - v0))
                z_fit = z0 + (dt_abs - v0) / (v1 - v0) * (z1 - z0)
                return z_fit, slope * 0.5

        best = min(pairs, key=lambda item: abs(item[1] - dt_abs))
        return best[0], 5.0

    def theoretical_arrivals(
        self,
        dist_deg: float,
        depth_km: float,
        origin_epoch: float,
        phases: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute theoretical arrival epochs for all valid phases at (dist_deg, depth_km)."""
        ph_list = phases if phases is not None else self.STANDARD_PHASES
        out: Dict[str, float] = {}
        for ph in ph_list:
            tt = self.travel_time(ph, dist_deg, depth_km)
            if tt is not None and tt > 0:
                out[ph] = origin_epoch + tt
        return out


DEFAULT_TTT: Optional[TravelTimeManager] = None


def get_ttt_manager(inputs_dir: Optional[str] = None) -> TravelTimeManager:
    global DEFAULT_TTT
    if DEFAULT_TTT is None or (inputs_dir and DEFAULT_TTT.inputs_dir != inputs_dir):
        DEFAULT_TTT = TravelTimeManager(inputs_dir)
    return DEFAULT_TTT
