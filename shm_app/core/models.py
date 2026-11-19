"""Core data structures for traces, phase picks, stations, and SHM parameter sets."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple
import numpy as np

from shm_app.core.time_utils import format_sh_time, parse_sh_time

# Phase weight table from cblib.c: index = quality (0..10) -> weight (9,9,4,3,2,2,1,1,0,0,0)
QUALITY_TO_WEIGHT = [9, 9, 4, 3, 2, 2, 1, 1, 0, 0, 0]
WEIGHT_TO_QUALITY = {4: 2, 3: 3, 2: 4, 1: 6, 0: 8, 9: 0}
QUALITY_SIGMA_SEC = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 4.0, 8.0, 15.0]


def quality_from_sigma_t(sigma_sec: float) -> int:
    """Map timing uncertainty sigma_t (seconds) to nearest pick quality level 0..9."""
    try:
        val = float(sigma_sec)
    except (TypeError, ValueError):
        return 4
    if math.isnan(val):
        return 4
    if val <= 0.0:
        return 0
    if math.isinf(val):
        return len(QUALITY_SIGMA_SEC) - 1
    return min(range(len(QUALITY_SIGMA_SEC)), key=lambda idx: abs(QUALITY_SIGMA_SEC[idx] - val))

# Canonical SHM event-type codes and Parameter Box labels
SHM_EVENT_TYPES = (
    ("t", "t (Teleseismic Quake)"),
    ("r", "r (Regional Quake)"),
    ("l", "l (Local Quake)"),
    ("n", "n (Nuclear Explosion)"),
    ("b", "b (Quarry Blast)"),
    ("m", "m (Mining Induced)"),
)
_SHM_EVENT_CODES = {code for code, _ in SHM_EVENT_TYPES}


def normalize_shm_event_type(
    raw_type: str,
    tier: str = "",
    dist_deg: Optional[float] = None,
) -> str:
    """Map FDSN/EMSC/USGS/legacy event type strings onto SHM event codes (t/r/l/n/b/m)."""
    low = (raw_type or "").strip().lower()
    if low in _SHM_EVENT_CODES:
        return low
    if "nuclear" in low:
        return "n"
    if "quarry" in low or "blast" in low or "explosion" in low:
        return "b"
    if "mining" in low or "induced" in low or "collapse" in low or "rockburst" in low:
        return "m"
    if "local" in low:
        return "l"
    if "regional" in low:
        return "r"
    if "tele" in low:
        return "t"

    tier_l = (tier or "").strip().lower()
    if tier_l == "local":
        return "l"
    if tier_l == "regional":
        return "r"
    if tier_l in ("tele", "teleseismic"):
        return "t"
    if dist_deg is not None and dist_deg > 0.0:
        if dist_deg <= 2.0:
            return "l"
        if dist_deg <= 15.0:
            return "r"
    return "t"


@dataclass
class StationInfo:
    """Station metadata from STATINF.DAT / STATLOC.DAT."""

    code: str
    name: str = ""
    instrument: str = "STS-2"
    lat: float = 0.0
    lon: float = 0.0
    array_id: int = 0
    xrel: float = 0.0  # km East relative to array reference
    yrel: float = 0.0  # km North relative to array reference
    elevation: float = 0.0  # meters
    velocity: float = 5.8  # surface P velocity km/s
    s_velocity: float = 3.4  # surface S velocity km/s


@dataclass
class PhasePick:
    """Phase reading matching TPiPhase in phaseinf.h and .evt descriptors."""

    QUALITY_SIGMA_SEC: ClassVar[List[float]] = QUALITY_SIGMA_SEC

    name: str = "P"
    onset: str = ""  # SH time string DD-MMM-YYYY_HH:MM:SS.SSS
    onset_acc_l: float = 0.0
    onset_acc_r: float = 0.0
    spec: str = "e"  # 'e' (emergent) or 'i' (impulsive)
    sign: int = 0  # +1 ('c'/compression), -1 ('d'/dilatation), 0 (' '/none)
    source: str = "m"  # 'm' (manual), 'a' (auto), 't' (theoretical)
    weight: int = 2  # 0..4 (or 9)
    quality: int = 4  # 0..10
    filter_name: str = "NONE"
    ampl: float = 0.0  # Restituted displacement amplitude (nm)
    ampl_time: float = 0.0  # Relative time of amplitude measurement (s)
    ampl_veloc: float = 0.0  # Velocity amplitude (nm/s)
    period: float = 0.0  # Dominant period (s)
    ap_source: str = ""  # 'm' or 'a' or 'd'
    bb_ampl: float = 0.0  # Broadband velocity amplitude (nm/s)
    bb_period: float = 0.0
    bb_mbb: float = 0.0
    slowness: float = 0.0  # s/deg
    d_slowness: float = 0.0
    beam_azim: float = 0.0  # deg (backazimuth)
    d_beam_azim: float = 0.0
    l_slowness: float = 0.0  # empirical calibration corrected slowness
    l_azim: float = 0.0  # empirical calibration corrected backazimuth
    mag_ml: float = 0.0
    mag_mb: float = 0.0
    mag_ms: float = 0.0
    mag_mbb: float = 0.0
    snr: float = 0.0
    resid: float = 0.0
    resid_corr: float = 0.0
    comment: str = ""
    station: str = ""
    comp: str = "Z"
    chan1: str = "B"
    chan2: str = "H"
    flags: str = ""  # 'f' (flagged), 'r' (reference), 'l' (location), 'b' (beam)
    _cached_onset_str: str = field(default="", init=False, repr=False)
    _cached_onset_ep: float = field(default=0.0, init=False, repr=False)

    @property
    def onset_epoch(self) -> float:
        """Return onset time as UTC POSIX epoch float (cached for O(1) repeated access)."""
        if not self.onset:
            return 0.0
        if self.onset != self._cached_onset_str:
            self._cached_onset_str = self.onset
            self._cached_onset_ep = parse_sh_time(self.onset)
        return self._cached_onset_ep

    @onset_epoch.setter
    def onset_epoch(self, val: float) -> None:
        self.onset = format_sh_time(val)
        self._cached_onset_str = self.onset
        self._cached_onset_ep = float(val)

    @property
    def sign_char(self) -> str:
        """Return 'c' for +1, 'd' for -1, or '' for 0."""
        if self.sign > 0:
            return "c"
        if self.sign < 0:
            return "d"
        return ""

    @classmethod
    def sign_from_char(cls, ch: str) -> int:
        c = ch.strip().lower()
        if c in ("c", "+", "u", "1"):
            return 1
        if c in ("d", "-", "-1"):
            return -1
        return 0

    def set_quality(self, q: int) -> None:
        """Update quality (0..10) and synchronize weight via SHM's lookup table."""
        self.quality = max(0, min(10, int(q)))
        self.weight = QUALITY_TO_WEIGHT[self.quality]

    def set_weight(self, w: int) -> None:
        """Update weight and synchronize quality."""
        self.weight = int(w)
        self.quality = WEIGHT_TO_QUALITY.get(self.weight, 4)

    def sigma_t(self, dt: float = 0.05) -> float:
        """Return 1-sigma timing uncertainty (seconds) from quality (0..9) or dt*2^(q/2) fallback."""
        q = max(0, min(9, int(self.quality)))
        if 0 <= q < len(self.QUALITY_SIGMA_SEC):
            return float(self.QUALITY_SIGMA_SEC[q])
        dt_eff = max(1e-4, float(dt))
        return max(dt_eff, dt_eff * (2.0 ** (q / 2.0)))

    @staticmethod
    def quality_from_sigma_t(sigma_sec: float) -> int:
        """Inverse mapping from timing uncertainty sigma_sec to nearest quality level 0..9."""
        return quality_from_sigma_t(sigma_sec)


@dataclass
class AnalysisParamSet:
    """One of the 3 parameter sets in SHM's Parameter Box (TSypar)."""

    distance: float = 0.0  # deg
    d_distance: float = 0.0
    azimuth: float = 0.0  # backazimuth deg
    d_azimuth: float = 0.0
    slowness: float = 0.0  # s/deg
    d_slowness: float = 0.0
    depth: float = 33.0  # km
    d_depth: float = 0.0
    depth_type: int = 1  # 1=preset, 2=estimated, 3=free/fitted
    origin: str = ""  # DD-MMM-YYYY_HH:MM:SS.SSS
    lat: float = 0.0
    d_lat: float = 0.0
    lon: float = 0.0
    d_lon: float = 0.0
    phase: str = "P"
    mag_ml: float = 0.0
    d_ml: float = 0.0
    mag_ms: float = 0.0
    d_ms: float = 0.0
    mag_mb: float = 0.0
    d_mb: float = 0.0
    mag_mbb: float = 0.0
    d_mbb: float = 0.0
    mags_count: int = 0
    source: str = "SHM"
    quality: int = 1
    region_table: int = 1  # 1=Flinn-Engdahl
    region_id: int = 0
    region_name: str = ""
    sregion_id: int = 0
    sregion_name: str = ""
    event_id: int = 0
    event_type: str = "t"  # 't'=teleseismicqu, 'r'=regionalqu, 'l'=localqu, 'n'=nuclear, 'b'=blast, 'm'=mining
    flags: str = ""
    comment: str = ""
    ref_lat: float = 49.6919  # GRFO default
    ref_lon: float = 11.2217
    corner_freq: float = 0.0
    moment_m0: float = 0.0
    mag_mw: float = 0.0
    incidence: float = 0.0  # deg (adopted from 3C Particle Motion / MDIR)
    residuals: Optional[Dict[str, float]] = None
    error_ellipse: Optional[Tuple[float, float, float]] = None  # (semi_major_km, semi_minor_km, strike_deg)
    condition_number: float = 1.0
    chi2_reduced: float = 0.0
    rmse: float = 0.0
    wrmse: float = 0.0


@dataclass
class Trace:
    """In-memory waveform trace with full Q-file / SH header attributes."""

    data: np.ndarray
    delta: float = 0.05  # sample interval in seconds (dt)
    station: str = "STAT"
    comp: str = "Z"  # 1-char component ('Z', 'N', 'E', 'R', 'T', 'L', 'Q', 'U', 'V', 'W')
    chan1: str = "B"  # 1st char of 2-char channel ('B', 'L', 'S', 'H', 'V')
    chan2: str = "H"  # 2nd char of 2-char channel ('H', 'Z', 'N', 'E')
    auxid: str = ""
    network: str = ""
    locid: str = ""
    start_epoch: float = 0.0  # UTC POSIX timestamp of sample 0
    display_offset: float = 0.0  # relative display shift in seconds (for ALIGN/SHIFT/TIME_AL)
    distance: float = 0.0  # R011 deg
    azimuth: float = 0.0  # R012 deg
    inci: float = 0.0  # R013 deg
    depth: float = 0.0  # R014 km
    magnitude: float = 0.0  # R015
    lat: float = 0.0  # R016
    lon: float = 0.0  # R017
    slowness: float = 0.0  # R018
    torig: float = 0.0  # R019
    weight: float = 1.0  # R020
    norm: float = 1.0  # R021 trace zoom/norm factor
    calib: float = 1.0  # R026 calibration (nm/s per count)
    calper: float = 1.0
    instype: str = ""
    comment: str = ""  # S021
    opinfo: str = ""  # S023
    filter_name: str = "NONE"
    overlay_filter_name: str = "NONE"
    overlay_filter_color: str = ""
    filter_history: List[str] = field(default_factory=list)
    hidden: bool = False
    selected: bool = False
    picks: List[PhasePick] = field(default_factory=list)
    theo_picks: Dict[str, float] = field(default_factory=dict)  # phase_name -> epoch
    extra_headers: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        self.comp = (self.comp or "Z").strip().upper()[:1]
        if len(self.chan1) >= 2 and not self.chan2:
            self.chan2 = self.chan1[1:2].upper()
            self.chan1 = self.chan1[:1].upper()

    @property
    def length(self) -> int:
        return int(self.data.size)

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.delta if self.delta > 0 else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, (self.length - 1) * self.delta)

    @property
    def start_time_str(self) -> str:
        return format_sh_time(self.start_epoch)

    @start_time_str.setter
    def start_time_str(self, val: str) -> None:
        self.start_epoch = parse_sh_time(val)

    @property
    def end_epoch(self) -> float:
        return self.start_epoch + self.duration

    @property
    def end_time_str(self) -> str:
        return format_sh_time(self.end_epoch)

    @property
    def channel3(self) -> str:
        """Return 3-character SEED channel code, e.g. 'BHZ'."""
        c1 = (self.chan1 or "B")[:1].upper()
        c2 = (self.chan2 or "H")[:1].upper()
        cp = (self.comp or "Z")[:1].upper()
        return f"{c1}{c2}{cp}"

    @property
    def stream_name(self) -> str:
        """Return SHM stream string '<stat>-<chan1><chan2>-<comp>' in lowercase."""
        c1 = (self.chan1 or "b")[:1].lower()
        c2 = (self.chan2 or "h")[:1].lower()
        cp = (self.comp or "z")[:1].lower()
        return f"{self.station.lower()}-{c1}{c2}-{cp}"

    @property
    def label(self) -> str:
        """Display label in trace margin."""
        flt = f" [{self.filter_name}]" if self.filter_name and self.filter_name != "NONE" else ""
        return f"{self.station} {self.channel3}{flt}"

    def time_axis(self, ref_epoch: Optional[float] = None) -> np.ndarray:
        """Return time axis in seconds relative to ref_epoch (or self.start_epoch if None)."""
        base = 0.0 if ref_epoch is None else (self.start_epoch - ref_epoch)
        return base + self.display_offset + np.arange(self.length, dtype=np.float64) * self.delta

    def copy(self, copy_picks: bool = True) -> "Trace":
        """Create a deep copy of this trace."""
        new_tr = copy.copy(self)
        new_tr.data = self.data.copy()
        new_tr.picks = [copy.deepcopy(p) for p in self.picks] if copy_picks else []
        new_tr.theo_picks = dict(self.theo_picks)
        new_tr.extra_headers = dict(self.extra_headers)
        new_tr.filter_history = list(self.filter_history)
        new_tr.overlay_filter_name = str(self.overlay_filter_name)
        new_tr.overlay_filter_color = str(self.overlay_filter_color)
        return new_tr

    def get_pick(self, phase_name: str) -> Optional[PhasePick]:
        """Find first pick matching phase_name (case-sensitive for depth phases like pP vs PP)."""
        for p in self.picks:
            if p.name == phase_name:
                return p
        return None

    def add_or_replace_pick(self, pick: PhasePick) -> None:
        """Add or replace pick with the same phase name on this trace, or deduplicate within 0.1 s."""
        pick.station = self.station
        pick.comp = self.comp
        pick.chan1 = self.chan1
        pick.chan2 = self.chan2
        for idx, existing in enumerate(self.picks):
            if existing.name == pick.name:
                self.picks[idx] = pick
                return
        if pick.onset_epoch > 0:
            for idx, existing in enumerate(self.picks):
                if existing.onset_epoch > 0 and abs(existing.onset_epoch - pick.onset_epoch) < 0.1:
                    self.picks[idx] = pick
                    return
        self.picks.append(pick)
