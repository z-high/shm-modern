"""Configuration, station inventory, sensitivity, and filter lookup tables."""

from __future__ import annotations

import fnmatch
import logging
import math
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.flinn_engdahl import FlinnEngdahlLookup
from shm_app.core.models import StationInfo
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.stationxml import (
    StationXMLChannel,
    StationXMLInventory,
    parse_iso_epoch,
    parse_stationxml,
)

DEFAULT_REF_STATION = "GRFO"
DEFAULT_REF_LAT = 49.6919
DEFAULT_REF_LON = 11.2217

GRF_STATIONS: List[str] = [
    "GRA1", "GRA2", "GRA3", "GRA4",
    "GRB1", "GRB2", "GRB3", "GRB4", "GRB5",
    "GRC1", "GRC2", "GRC3", "GRC4",
]

GRSN_STATIONS: List[str] = [
    "BFO", "BRG", "BSEG", "BUG", "CLL", "CLZ", "FUR", "GEC2",
    "GRFO", "GSH", "GTTG", "HAM", "HLG", "IBBN", "LID", "MOX",
    "NRDL", "RGN", "RUE", "STU", "TNS", "UBBA", "WET",
]

GERESS_STATIONS_DEFAULT: List[str] = [
    "GEA0", "GEA1", "GEA2", "GEA3",
    "GEB1", "GEB2", "GEB3", "GEB4", "GEB5",
    "GEC1", "GEC2", "GEC3", "GEC4", "GEC5", "GEC6", "GEC7",
    "GED1", "GED2", "GED3", "GED4", "GED5", "GED6", "GED7", "GED8", "GED9",
]

# Legacy SHM Read-dialog regional matrix stations outside GRF
SHM_READ_REGIONAL_STATIONS: List[str] = [
    "BFO", "BRG", "BUG", "CLL", "CLZ", "FUR", "GEC2", "GRFO", "GSH", "GTTG",
    "HAM", "IBBN", "LID", "MOX", "NRDL", "RGN", "ROT", "STU", "TNS", "UBR",
    "WATA", "WET", "INZ",
]


def parse_station_list(spec: Union[str, Sequence[str]]) -> List[str]:
    """Parse comma/whitespace-separated station codes into an ordered unique list."""
    tokens = [spec] if isinstance(spec, str) else list(spec)
    out: List[str] = []
    seen = set()
    for item in tokens:
        for part in re.split(r"[\s,;]+", str(item).strip()):
            code = part.strip().upper()
            if code and code not in seen:
                seen.add(code)
                out.append(code)
    return out


class SHMConfig:
    """Manages SHM configuration, station database, sensitivities, and filter lookups."""

    def __init__(self, shm_root: str = SHM_DEFAULT_ROOT) -> None:
        self.shm_root = shm_root
        self.inputs_dir = os.path.join(shm_root, "inputs")
        self.filter_dir = os.path.join(shm_root, "filter")
        self.globals_dir = os.path.join(shm_root, "globals")
        self.data_examples_dir = os.path.join(shm_root, "data-examples")

        self.params: Dict[str, str] = {}
        self.stations: Dict[str, StationInfo] = {}
        self.sensitivities: List[Tuple[str, float, float, float]] = []
        self.filter_rules: List[Tuple[str, float, float, str]] = []
        self.station_groups: Dict[str, List[str]] = {}
        self.stationxml_channels: List[StationXMLChannel] = []
        self.fer = FlinnEngdahlLookup(self.inputs_dir, shm_root=self.shm_root)

        self.load_all()

    @property
    def root(self) -> str:
        return self.shm_root

    @root.setter
    def root(self, val: str) -> None:
        self.shm_root = val
        self.inputs_dir = os.path.join(val, "inputs")
        self.filter_dir = os.path.join(val, "filter")
        self.globals_dir = os.path.join(val, "globals")
        self.data_examples_dir = os.path.join(val, "data-examples")
        self.params.clear()
        self.stations.clear()
        self.sensitivities.clear()
        self.filter_rules.clear()
        self.station_groups.clear()
        self.stationxml_channels.clear()
        self.fer = FlinnEngdahlLookup(self.inputs_dir, shm_root=self.shm_root)
        self.load_all()

    def load_all(self) -> None:
        self._load_shm_config()
        self._load_stations()
        self._load_sensitivities()
        self._load_filter_lookup()
        self._load_station_groups()

    def _load_shm_config(self) -> None:
        cfg_path = os.path.join(self.inputs_dir, "shm-config.txt")
        if not os.path.isfile(cfg_path):
            return
        with open(cfg_path, "r", encoding="latin-1", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("!"):
                    continue
                parts = s.split(None, 1)
                if len(parts) == 2:
                    self.params[parts[0]] = parts[1].strip()

    def _load_stations(self) -> None:
        # 1. Load STATINF.DAT (newer format with key:value attributes)
        statinf_path = os.path.join(self.inputs_dir, "STATINF.DAT")
        if os.path.isfile(statinf_path):
            with open(statinf_path, "r", encoding="latin-1", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("!"):
                        continue
                    tokens = s.split()
                    code = tokens[0].upper()
                    kv: Dict[str, str] = {
                        k.lower(): v for k, v in re.findall(r"([A-Za-z0-9_]+):\s*([^\s]+)", s)
                    }
                    info = StationInfo(
                        code=code,
                        name=kv.get("name", code),
                        instrument=kv.get("instrument", "STS-2"),
                        lat=self._safe_float(kv.get("lat", "0"), 0.0),
                        lon=self._safe_float(kv.get("lon", "0"), 0.0),
                        array_id=int(self._safe_float(kv.get("array", "0"), 0.0)),
                        xrel=self._safe_float(kv.get("xrel", "0"), 0.0),
                        yrel=self._safe_float(kv.get("yrel", "0"), 0.0),
                        elevation=self._safe_float(kv.get("elevation", "0"), 0.0),
                        velocity=self._safe_float(kv.get("velocity", "5.8"), 5.8),
                    )
                    if info.velocity <= 0.0:
                        info.velocity = 5.8
                    self.stations[code] = info

        # 2. Load STATLOC.DAT as fallback for any additional stations
        statloc_path = os.path.join(self.inputs_dir, "STATLOC.DAT")
        if os.path.isfile(statloc_path):
            with open(statloc_path, "r", encoding="latin-1", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("!"):
                        continue
                    parts = s.split()
                    if len(parts) >= 3:
                        code = parts[0].upper()
                        if code in self.stations:
                            continue
                        lat = self._safe_float(parts[1], 0.0)
                        lon = self._safe_float(parts[2], 0.0)
                        arr_id = int(self._safe_float(parts[3], 0.0)) if len(parts) > 3 else 0
                        xrel = self._safe_float(parts[4], 0.0) if len(parts) > 4 else 0.0
                        yrel = self._safe_float(parts[5], 0.0) if len(parts) > 5 else 0.0
                        elev = self._safe_float(parts[6], 0.0) if len(parts) > 6 else 0.0
                        comment = " ".join(parts[7:]) if len(parts) > 7 else code
                        self.stations[code] = StationInfo(
                            code=code,
                            name=comment,
                            lat=lat,
                            lon=lon,
                            array_id=arr_id,
                            xrel=xrel,
                            yrel=yrel,
                            elevation=elev,
                        )

    def _load_sensitivities(self) -> None:
        sens_path = os.path.join(self.inputs_dir, "sensitivities.txt")
        if not os.path.isfile(sens_path):
            return
        with open(sens_path, "r", encoding="latin-1", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("!"):
                    continue
                parts = s.split()
                if len(parts) >= 4:
                    stream_pat = parts[0].lower()
                    try:
                        t_from = parse_sh_time(parts[1])
                    except ValueError:
                        t_from = 0.0
                    try:
                        t_to = parse_sh_time(parts[2])
                    except ValueError:
                        t_to = 4102444800.0
                    calib = self._safe_float(parts[3], 1.0)
                    self.sensitivities.append((stream_pat, t_from, t_to, calib))

    def _load_filter_lookup(self) -> None:
        fl_path = os.path.join(self.inputs_dir, "filter_lookup.txt")
        if not os.path.isfile(fl_path):
            return
        with open(fl_path, "r", encoding="latin-1", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("!"):
                    continue
                parts = s.split()
                if len(parts) >= 2:
                    pat = parts[0].lower()
                    tf_prefix = parts[1].upper()
                    t_from = 0.0
                    t_to = 4102444800.0
                    if len(parts) >= 4:
                        try:
                            t_from = parse_sh_time(parts[2])
                        except ValueError:
                            t_from = 0.0
                        try:
                            t_to = parse_sh_time(parts[3])
                        except ValueError:
                            t_to = 4102444800.0
                    self.filter_rules.append((pat, t_from, t_to, tf_prefix))

    def _load_station_groups(self) -> None:
        if not os.path.isdir(self.globals_dir):
            return
        for fname in sorted(os.listdir(self.globals_dir)):
            if fname.upper().endswith(".STX"):
                group_name = os.path.splitext(fname)[0]
                fpath = os.path.join(self.globals_dir, fname)
                stations: List[str] = []
                with open(fpath, "r", encoding="latin-1", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if s and not s.startswith("!"):
                            stations.append(s.split()[0].upper())
                if stations:
                    self.station_groups[group_name] = stations

    @staticmethod
    def _safe_float(val: str, default: float = 0.0) -> float:
        if not val or val.lower().startswith("undef"):
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def get_station(self, code: str) -> StationInfo:
        c = code.strip().upper()
        if c in self.stations:
            return self.stations[c]
        return StationInfo(code=c, name=c)

    def get_sensitivity(self, stream: str, epoch: Optional[float] = None) -> float:
        """Return calibration factor (nm/s per count) for stream '<stat>-<chan>-<comp>'."""
        st = stream.strip().lower()
        for pat, t_from, t_to, calib in self.sensitivities:
            if fnmatch.fnmatch(st, pat):
                if epoch is None or (t_from <= epoch <= t_to):
                    return calib
        # Second pass ignoring time if epoch fell outside listed window
        for pat, _, _, calib in self.sensitivities:
            if fnmatch.fnmatch(st, pat):
                return calib
        return 1.0

    def get_transfer_prefix(self, stream: str, epoch: Optional[float] = None) -> str:
        """Return instrument transfer function prefix (e.g. 'GRSN', 'GRF', 'STS2') for stream."""
        st = stream.strip().lower()
        for pat, t_from, t_to, prefix in self.filter_rules:
            if fnmatch.fnmatch(st, pat):
                if epoch is None or (t_from <= epoch <= t_to):
                    return prefix
        for pat, _, _, prefix in self.filter_rules:
            if fnmatch.fnmatch(st, pat):
                return prefix
        stat_code = st.split("-", 1)[0].upper()
        if stat_code.startswith(("GRA", "GRB", "GRC")) and len(stat_code) == 4:
            return "GRF"
        if stat_code in self.stations:
            return "GRSN"
        return "STS2"

    @staticmethod
    def _stationxml_calib_nm_per_count(ch: StationXMLChannel) -> Optional[float]:
        """Convert StationXML sensitivity (counts per physical unit) to SHM calib (nm/s per count)."""
        sens_val: Optional[float] = None
        in_units = ""
        if ch.instrument_sensitivity is not None and ch.instrument_sensitivity.value > 0.0:
            sens_val = float(ch.instrument_sensitivity.value)
            in_units = (ch.instrument_sensitivity.input_units or "").strip().upper()
        elif ch.stages:
            prod = 1.0
            has_gain = False
            for stg in ch.stages:
                if stg.stage_gain is not None and stg.stage_gain.value > 0.0:
                    prod *= float(stg.stage_gain.value)
                    has_gain = True
            if has_gain and prod > 0.0:
                sens_val = prod
                in_units = (ch.stages[0].input_units or "").strip().upper()

        if sens_val is None or sens_val <= 0.0:
            return None

        if in_units in ("NM/S", "NM/SEC", "NM"):
            return 1.0 / sens_val
        if in_units in ("UM/S", "UM/SEC", "UM"):
            return 1e3 / sens_val
        if in_units in ("MM/S", "MM/SEC", "MM"):
            return 1e6 / sens_val
        if in_units in ("CM/S", "CM/SEC", "CM"):
            return 1e7 / sens_val
        return 1e9 / sens_val

    def load_stationxml(self, path_or_xml: str) -> StationXMLInventory:
        """Parse FDSN StationXML and merge stations, epoch sensitivities, and channel metadata."""
        inv = parse_stationxml(path_or_xml)

        for sta in inv.stations:
            code = sta.code.strip().upper()
            if not code:
                continue
            sensor_inst = ""
            for ch in sta.channels:
                if ch.sensor_description:
                    sensor_inst = ch.sensor_description
                    break
            existing = self.stations.get(code)
            if existing is not None:
                if sta.latitude != 0.0 or sta.longitude != 0.0:
                    existing.lat = sta.latitude
                    existing.lon = sta.longitude
                    existing.elevation = sta.elevation
                if sta.name and sta.name != code:
                    existing.name = sta.name
                if sensor_inst:
                    existing.instrument = sensor_inst
            else:
                self.stations[code] = StationInfo(
                    code=code,
                    name=sta.name or code,
                    instrument=sensor_inst or "STS-2",
                    lat=sta.latitude,
                    lon=sta.longitude,
                    elevation=sta.elevation,
                )

        new_sens: List[Tuple[str, float, float, float]] = []
        for ch in inv.channels:
            key = (ch.network.upper(), ch.station.upper(), ch.location.strip().upper(), ch.code.upper(), ch.start_date, ch.end_date)
            replaced = False
            for idx, cur in enumerate(self.stationxml_channels):
                cur_key = (cur.network.upper(), cur.station.upper(), cur.location.strip().upper(), cur.code.upper(), cur.start_date, cur.end_date)
                if cur_key == key:
                    self.stationxml_channels[idx] = ch
                    replaced = True
                    break
            if not replaced:
                self.stationxml_channels.append(ch)

            calib = self._stationxml_calib_nm_per_count(ch)
            if calib is not None:
                pat = ch.stream_name
                t_from = ch.start_epoch if ch.start_epoch is not None else 0.0
                t_to = ch.end_epoch if ch.end_epoch is not None else 4102444800.0
                self.sensitivities = [
                    entry for entry in self.sensitivities
                    if not (entry[0] == pat and abs(entry[1] - t_from) < 1e-6 and abs(entry[2] - t_to) < 1e-6)
                ]
                new_sens.append((pat, t_from, t_to, calib))

        if new_sens:
            # Sort newest epochs first so epoch=None resolves to the most recent active epoch
            new_sens.sort(key=lambda item: (item[2], item[1]), reverse=True)
            self.sensitivities = new_sens + self.sensitivities

        return inv

    def get_channel_metadata(
        self,
        net: str,
        sta: str,
        loc: str = "",
        cha: str = "",
        epoch: Optional[Union[float, str]] = None,
    ) -> Optional[StationXMLChannel]:
        """Query stored StationXML channel orientation and response metadata by NSLC and optional epoch."""
        epoch_val: Optional[float]
        if isinstance(epoch, str):
            epoch_val = parse_iso_epoch(epoch)
            if epoch_val is None:
                try:
                    epoch_val = parse_sh_time(epoch)
                except ValueError:
                    epoch_val = None
        else:
            epoch_val = float(epoch) if epoch is not None else None

        net_u = (net or "").strip().upper()
        sta_u = (sta or "").strip().upper()
        loc_norm = "" if (loc or "").strip() in ("", "--") else (loc or "").strip().upper()
        cha_u = (cha or "").strip().upper()

        matches: List[StationXMLChannel] = []
        for ch in self.stationxml_channels:
            if net_u and net_u != "*" and ch.network.upper() != net_u:
                continue
            if sta_u and sta_u != "*" and ch.station.upper() != sta_u:
                continue
            ch_loc = "" if ch.location.strip() in ("", "--") else ch.location.strip().upper()
            if loc_norm != "*" and ch_loc != loc_norm:
                continue
            if cha_u and cha_u != "*" and ch.code.upper() != cha_u:
                continue
            if ch.matches_epoch(epoch_val):
                matches.append(ch)

        if not matches and loc_norm == "":
            for ch in self.stationxml_channels:
                if net_u and net_u != "*" and ch.network.upper() != net_u:
                    continue
                if sta_u and sta_u != "*" and ch.station.upper() != sta_u:
                    continue
                if cha_u and cha_u != "*" and ch.code.upper() != cha_u:
                    continue
                if ch.matches_epoch(epoch_val):
                    matches.append(ch)

        if not matches:
            return None

        if epoch_val is None:
            matches.sort(
                key=lambda c: (
                    1 if c.end_epoch is None else 0,
                    c.start_epoch if c.start_epoch is not None else float("-inf"),
                ),
                reverse=True,
            )
        else:
            matches.sort(
                key=lambda c: c.start_epoch if c.start_epoch is not None else float("-inf"),
                reverse=True,
            )
        return matches[0]

    def relative_xy_km(self, stations: List[str], ref_station: str = "CENTRE") -> List[Tuple[float, float, float, float]]:
        """Return [(x_km, y_km, elev_m, vp_kms), ...] for a list of station codes relative to ref_station."""
        infos = [self.get_station(s) for s in stations]
        if not infos:
            return []

        ref_u = (ref_station or "").strip().upper()
        use_centre = ref_u in ("CENTRE", "CENTER", "")
        ref_inf: Optional[StationInfo] = None
        if not use_centre:
            ref_inf = self.get_station(ref_u)
            ref_known = (
                ref_u in self.stations
                or abs(ref_inf.lat) > 1e-9
                or abs(ref_inf.lon) > 1e-9
                or ref_inf.array_id > 0
                or ref_inf.xrel != 0.0
                or ref_inf.yrel != 0.0
            )
            if not ref_known:
                msg = f"Unknown reference station {ref_station!r} in relative_xy_km; falling back to CENTRE"
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                use_centre = True
                ref_inf = None

        # Check if all stations belong to the same array with xrel/yrel populated
        array_ids = {inf.array_id for inf in infos if inf.array_id > 0}
        zero_xy_count = sum(1 for inf in infos if inf.xrel == 0.0 and inf.yrel == 0.0)
        has_nonzero_xy = any(inf.xrel != 0.0 or inf.yrel != 0.0 for inf in infos)
        use_array_xy = (
            len(array_ids) == 1
            and all(inf.array_id > 0 for inf in infos)
            and (len(infos) == 1 or (has_nonzero_xy and zero_xy_count <= 1))
        )
        if not use_centre and ref_inf is not None and use_array_xy:
            if ref_inf.array_id not in array_ids:
                use_array_xy = False

        deg_to_km = 111.1949266
        if use_array_xy:
            coords = [(inf.xrel, inf.yrel, inf.elevation, inf.velocity) for inf in infos]
        else:
            ref_lat = sum(inf.lat for inf in infos) / len(infos)
            sin_lon = sum(math.sin(math.radians(inf.lon)) for inf in infos)
            cos_lon = sum(math.cos(math.radians(inf.lon)) for inf in infos)
            ref_lon = math.degrees(math.atan2(sin_lon, cos_lon)) if (sin_lon != 0.0 or cos_lon != 0.0) else 0.0
            coords = []
            for inf in infos:
                dlon = ((inf.lon - ref_lon + 180.0) % 360.0) - 180.0
                x_km = dlon * deg_to_km * math.cos(math.radians(inf.lat))
                y_km = (inf.lat - ref_lat) * deg_to_km
                coords.append((x_km, y_km, inf.elevation, inf.velocity))

        if use_centre or ref_inf is None:
            xm = sum(c[0] for c in coords) / len(coords)
            ym = sum(c[1] for c in coords) / len(coords)
            em = sum(c[2] for c in coords) / len(coords)
        elif use_array_xy:
            xm, ym, em = ref_inf.xrel, ref_inf.yrel, ref_inf.elevation
        else:
            dlon_ref = ((ref_inf.lon - ref_lon + 180.0) % 360.0) - 180.0
            xm = dlon_ref * deg_to_km * math.cos(math.radians(ref_inf.lat))
            ym = (ref_inf.lat - ref_lat) * deg_to_km
            em = ref_inf.elevation

        return [(c[0] - xm, c[1] - ym, c[2] - em, c[3]) for c in coords]


# Singleton default configuration instance
DEFAULT_CONFIG: Optional[SHMConfig] = None


def get_config(shm_root: Optional[str] = None) -> SHMConfig:
    global DEFAULT_CONFIG
    root = shm_root or SHM_DEFAULT_ROOT
    if DEFAULT_CONFIG is None or DEFAULT_CONFIG.shm_root != root:
        DEFAULT_CONFIG = SHMConfig(root)
    return DEFAULT_CONFIG


def geress_stations(config: Optional[SHMConfig] = None) -> List[str]:
    """Return configured GERESS array station list with built-in fallback."""
    cfg = config or get_config()
    raw_items = cfg.station_groups.get("GERESS", [])
    if raw_items:
        parsed = parse_station_list(raw_items)
        if parsed:
            return parsed
    return list(GERESS_STATIONS_DEFAULT)


def resolve_reference_station(
    config: Optional[SHMConfig] = None,
    ref_station: Optional[str] = None,
    fallback_lat: Optional[float] = None,
    fallback_lon: Optional[float] = None,
    default_code: str = DEFAULT_REF_STATION,
) -> Tuple[str, float, float]:
    """Resolve observatory reference (code, lat, lon) from config/DB with explicit fallback."""
    cfg = config or get_config()
    code = (ref_station or cfg.params.get("ref_station", "") or default_code).strip().upper()
    if code:
        st = cfg.get_station(code)
        if abs(st.lat) > 1e-6 or abs(st.lon) > 1e-6:
            return code, float(st.lat), float(st.lon)
    if (
        fallback_lat is not None
        and fallback_lon is not None
        and (abs(fallback_lat) > 1e-6 or abs(fallback_lon) > 1e-6)
    ):
        return (code or default_code), float(fallback_lat), float(fallback_lon)
    return (code or default_code), DEFAULT_REF_LAT, DEFAULT_REF_LON
