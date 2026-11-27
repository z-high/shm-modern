"""Array slowness and backazimuth calibration tables (calib_*.dat)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import SHMConfig, get_config

CALC_STARTPOS = 26


@dataclass(frozen=True)
class SlownessCalibrationEntry:
    """Single empirical slowness-azimuth calibration vector."""

    s_meas: float
    az_meas: float
    s_corr: float
    az_corr: float
    comment: str = ""


class SlownessCalibrationTable:
    """Empirical slowness/backazimuth calibration table (e.g. ``calib_grf.dat``).

    Supports both:
    - Legacy SHM format (``CALC_STARTPOS=26``: 26-char prefix followed by
      ``s_beam az_beam s_corr az_corr`` and optional trailing text).
    - Delta format (``# slowness_in azimuth_in d_slowness d_azimuth``).

    Interpolation is performed in 2D Cartesian slowness space
    ``(sx, sy) = (s * sin(az), s * cos(az))`` using inverse-distance weighting
    (IDW) over the nearest calibration vectors, naturally handling angular
    wrap-around across 0/360 degrees.
    """

    def __init__(
        self,
        entries: Sequence[SlownessCalibrationEntry],
        source_path: str = "",
    ) -> None:
        self.entries: List[SlownessCalibrationEntry] = list(entries)
        self.source_path: str = source_path

        if self.entries:
            s_m = np.array([e.s_meas for e in self.entries], dtype=np.float64)
            az_m_r = np.radians([e.az_meas for e in self.entries])
            s_c = np.array([e.s_corr for e in self.entries], dtype=np.float64)
            az_c_r = np.radians([e.az_corr for e in self.entries])

            self._u_meas = s_m * np.sin(az_m_r)
            self._v_meas = s_m * np.cos(az_m_r)
            self._u_corr = s_c * np.sin(az_c_r)
            self._v_corr = s_c * np.cos(az_c_r)
            self._du = self._u_corr - self._u_meas
            self._dv = self._v_corr - self._v_meas
        else:
            empty = np.zeros(0, dtype=np.float64)
            self._u_meas = empty
            self._v_meas = empty
            self._u_corr = empty
            self._v_corr = empty
            self._du = empty
            self._dv = empty

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_text(
        cls,
        text: str,
        fmt: str = "auto",
        source_path: str = "",
    ) -> "SlownessCalibrationTable":
        """Parse calibration entries from legacy ``calib_grf.dat`` or delta-format text."""
        fmt_mode = fmt.strip().lower()
        header_hint: Optional[str] = None
        entries: List[SlownessCalibrationEntry] = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!"):
                continue
            if stripped.startswith("#"):
                hdr = stripped[1:].lower()
                if any(k in hdr for k in ("d_slowness", "d_azimuth", "dslow", "daz", "delta")):
                    header_hint = "delta"
                elif any(k in hdr for k in ("s_corr", "az_corr", "scorr", "azcorr")):
                    header_hint = "abs"
                continue

            entry = cls._parse_line(line, fmt_mode=fmt_mode, header_hint=header_hint)
            if entry is not None:
                entries.append(entry)

        return cls(entries=entries, source_path=source_path)

    @staticmethod
    def _parse_line(
        line: str,
        fmt_mode: str = "auto",
        header_hint: Optional[str] = None,
    ) -> Optional[SlownessCalibrationEntry]:
        if fmt_mode == "legacy":
            if len(line) <= CALC_STARTPOS:
                return None
            toks = line[CALC_STARTPOS:].split()
            if len(toks) < 4:
                return None
            try:
                s0, a0, sc, ac = (float(toks[0]), float(toks[1]), float(toks[2]), float(toks[3]))
            except ValueError:
                return None
            return SlownessCalibrationEntry(
                s_meas=s0,
                az_meas=a0 % 360.0,
                s_corr=sc,
                az_corr=ac % 360.0,
                comment=line[:CALC_STARTPOS].strip(),
            )

        if fmt_mode == "delta":
            clean = line.split("#", 1)[0].split("!", 1)[0].strip()
            toks = clean.split()
            if len(toks) < 4:
                return None
            try:
                s0, a0, ds, daz = (float(toks[0]), float(toks[1]), float(toks[2]), float(toks[3]))
            except ValueError:
                return None
            return SlownessCalibrationEntry(
                s_meas=s0,
                az_meas=a0 % 360.0,
                s_corr=max(0.0, s0 + ds),
                az_corr=(a0 + daz) % 360.0,
                comment="",
            )

        # Auto-detection: check if line starts directly with 4 float tokens (delta/compact format)
        clean = line.split("#", 1)[0].split("!", 1)[0].strip()
        toks = clean.split()
        direct_floats: Optional[Tuple[float, float, float, float]] = None
        if len(toks) >= 4:
            try:
                direct_floats = (float(toks[0]), float(toks[1]), float(toks[2]), float(toks[3]))
            except ValueError:
                direct_floats = None

        if direct_floats is not None:
            s0, a0, v3, v4 = direct_floats
            if header_hint == "abs":
                return SlownessCalibrationEntry(
                    s_meas=s0,
                    az_meas=a0 % 360.0,
                    s_corr=v3,
                    az_corr=v4 % 360.0,
                )
            return SlownessCalibrationEntry(
                s_meas=s0,
                az_meas=a0 % 360.0,
                s_corr=max(0.0, s0 + v3),
                az_corr=(a0 + v4) % 360.0,
            )

        # Otherwise fall back to legacy CALC_STARTPOS=26 format
        if len(line) > CALC_STARTPOS:
            legacy_toks = line[CALC_STARTPOS:].split()
            if len(legacy_toks) >= 4:
                try:
                    s0, a0, sc, ac = (
                        float(legacy_toks[0]),
                        float(legacy_toks[1]),
                        float(legacy_toks[2]),
                        float(legacy_toks[3]),
                    )
                    return SlownessCalibrationEntry(
                        s_meas=s0,
                        az_meas=a0 % 360.0,
                        s_corr=sc,
                        az_corr=ac % 360.0,
                        comment=line[:CALC_STARTPOS].strip(),
                    )
                except ValueError:
                    return None
        return None

    @classmethod
    def from_file(cls, filepath: str, fmt: str = "auto") -> "SlownessCalibrationTable":
        """Load slowness calibration table from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Slowness calibration file not found: {filepath}")
        with open(filepath, "r", encoding="latin-1", errors="ignore") as fh:
            text = fh.read()
        return cls.from_text(text, fmt=fmt, source_path=filepath)

    def calibrate(
        self,
        slowness_meas: float,
        azim_meas: float,
        k: int = 8,
        power: float = 2.0,
    ) -> Tuple[float, float, float, float]:
        """Interpolate empirical correction at ``(slowness_meas, azim_meas)``.

        Returns ``(slowness_cal, azim_cal, d_slowness, d_azim)`` where:
        - ``slowness_cal`` is in s/deg,
        - ``azim_cal`` is in [0, 360) deg,
        - ``d_slowness = slowness_cal - slowness_meas``,
        - ``d_azim`` is wrapped into [-180, 180) deg.
        """
        s_in = float(slowness_meas)
        az_in = float(azim_meas) % 360.0
        if not self.entries:
            return s_in, az_in, 0.0, 0.0

        az_r = math.radians(az_in)
        u_q = s_in * math.sin(az_r)
        v_q = s_in * math.cos(az_r)

        d2 = (self._u_meas - u_q) ** 2 + (self._v_meas - v_q) ** 2
        exact_mask = d2 <= 1e-12
        if np.any(exact_mask):
            du_interp = float(np.mean(self._du[exact_mask]))
            dv_interp = float(np.mean(self._dv[exact_mask]))
        else:
            n_pts = self._u_meas.size
            k_eff = min(max(1, int(k)), n_pts)
            if k_eff < n_pts:
                idx = np.argpartition(d2, k_eff - 1)[:k_eff]
            else:
                idx = np.arange(n_pts)
            dists = np.sqrt(np.maximum(1e-24, d2[idx]))
            weights = 1.0 / np.maximum(1e-12, dists ** float(power))
            w_sum = float(np.sum(weights))
            if w_sum <= 1e-30:
                du_interp = float(np.mean(self._du[idx]))
                dv_interp = float(np.mean(self._dv[idx]))
            else:
                weights = weights / w_sum
                du_interp = float(np.sum(weights * self._du[idx]))
                dv_interp = float(np.sum(weights * self._dv[idx]))

        u_cal = u_q + du_interp
        v_cal = v_q + dv_interp
        s_cal = math.hypot(u_cal, v_cal)
        az_cal = math.degrees(math.atan2(u_cal, v_cal)) % 360.0 if s_cal > 1e-12 else az_in
        d_slo = s_cal - s_in
        d_az = ((az_cal - az_in + 180.0) % 360.0) - 180.0
        return float(s_cal), float(az_cal), float(d_slo), float(d_az)


def load_calibration_for_array(
    stations_or_name: Union[str, Sequence[str]],
    inputs_dir: Optional[str] = None,
    config: Optional[SHMConfig] = None,
) -> Optional[SlownessCalibrationTable]:
    """Look up and load an empirical slowness calibration table via ``calib_lookup.txt``.

    Returns ``None`` gracefully if ``calib_lookup.txt`` or the target calibration
    file is missing, or if no matching array/station entry is found.
    """
    inp_dir = inputs_dir or (config.inputs_dir if config is not None else os.path.join(SHM_DEFAULT_ROOT, "inputs"))
    if not inp_dir or not os.path.isdir(inp_dir):
        return None

    lookup_path = os.path.join(inp_dir, "calib_lookup.txt")
    if not os.path.isfile(lookup_path):
        return None

    query_set: set[str] = set()
    if isinstance(stations_or_name, str):
        nm = stations_or_name.strip().upper()
        if nm:
            query_set.add(nm)
            if config is not None and nm in getattr(config, "station_groups", {}):
                query_set.update(s.upper() for s in config.station_groups[nm])
    else:
        for s in stations_or_name:
            su = str(s).strip().upper()
            if su:
                query_set.add(su)

    if not query_set:
        return None

    try:
        with open(lookup_path, "r", encoding="latin-1", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("!") or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        stat_tokens = {tok.strip().upper() for tok in parts[0].split(",") if tok.strip()}
        cal_filename = parts[1].strip()
        matched = bool(query_set & stat_tokens)
        if not matched and "GRF" in query_set and "grf" in cal_filename.lower():
            matched = True
        if matched:
            cal_path = cal_filename if os.path.isabs(cal_filename) else os.path.join(inp_dir, cal_filename)
            if os.path.isfile(cal_path):
                return SlownessCalibrationTable.from_file(cal_path)
            return None
    return None


def _resolve_calib_table(
    calib_table: Optional[SlownessCalibrationTable],
    apply_calibration: Optional[bool],
    stations: Optional[Sequence[str]] = None,
    config: Optional[SHMConfig] = None,
) -> Optional[SlownessCalibrationTable]:
    if apply_calibration is False:
        return None
    if calib_table is not None:
        return calib_table if len(calib_table) > 0 else None
    if apply_calibration is True and stations:
        tbl = load_calibration_for_array(stations, config=config)
        if tbl is not None and len(tbl) > 0:
            return tbl
    return None
