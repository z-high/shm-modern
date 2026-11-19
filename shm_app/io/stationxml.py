"""FDSN StationXML 1.x parser and complex frequency response evaluator (pure Python stdlib + NumPy)."""

from __future__ import annotations

import datetime
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np


def _local_tag(tag: str) -> str:
    """Strip XML namespace URI from an ElementTree tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    """Return first direct child matching local tag name (case-sensitive, with case-insensitive fallback)."""
    for child in elem:
        if _local_tag(child.tag) == name:
            return child
    low = name.lower()
    for child in elem:
        if _local_tag(child.tag).lower() == low:
            return child
    return None


def _findall_children(elem: ET.Element, name: str) -> List[ET.Element]:
    """Return all direct children matching local tag name."""
    out = [child for child in elem if _local_tag(child.tag) == name]
    if not out:
        low = name.lower()
        out = [child for child in elem if _local_tag(child.tag).lower() == low]
    return out


def _child_text(elem: ET.Element, name: str, default: str = "") -> str:
    """Return stripped text of first matching direct child, or default."""
    child = _find_child(elem, name)
    if child is not None and child.text is not None:
        return child.text.strip()
    return default


def _child_float(elem: ET.Element, name: str, default: float = 0.0) -> float:
    """Return float value of first matching direct child, or default."""
    txt = _child_text(elem, name, "")
    if not txt:
        return default
    try:
        return float(txt)
    except ValueError:
        return default


def _child_int(elem: ET.Element, name: str, default: int = 0) -> int:
    """Return int value of first matching direct child, or default."""
    txt = _child_text(elem, name, "")
    if not txt:
        return default
    try:
        return int(float(txt))
    except ValueError:
        return default


def parse_iso_epoch(timestr: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 StationXML/QuakeML timestamp into a UTC POSIX epoch float."""
    if not timestr:
        return None
    s = str(timestr).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Truncate sub-microsecond precision if present (Python fromisoformat accepts up to 6 fractional digits)
    m = re.match(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(\.(\d+))?(.*)$", s)
    if m:
        base, _dot, frac, tz = m.groups()
        if frac:
            frac = frac[:6].ljust(6, "0")
            s = f"{base}.{frac}{tz or ''}"
        else:
            s = f"{base}{tz or ''}"
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        # Fallback for date-only strings like YYYY-MM-DD
        try:
            d = datetime.date.fromisoformat(s[:10])
            dt = datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


@dataclass
class StationXMLPolesZeros:
    """StationXML PolesZeros response element (Stage 1 or analog/digital stage)."""

    pz_transfer_function_type: str = "LAPLACE (RADIANS/SECOND)"
    normalization_factor: float = 1.0
    normalization_frequency: float = 1.0
    zeros: List[complex] = field(default_factory=list)
    poles: List[complex] = field(default_factory=list)
    input_units: str = ""
    output_units: str = ""
    resource_id: str = ""


@dataclass
class StationXMLCoefficients:
    """StationXML Coefficients response element (FIR / rational IIR stage)."""

    cf_transfer_function_type: str = "DIGITAL"
    numerator: List[float] = field(default_factory=list)
    denominator: List[float] = field(default_factory=list)
    input_units: str = ""
    output_units: str = ""


@dataclass
class StationXMLStageGain:
    """StationXML StageGain element."""

    value: float = 1.0
    frequency: float = 1.0


@dataclass
class StationXMLResponseStage:
    """Single StationXML Response Stage."""

    number: int = 1
    poles_zeros: Optional[StationXMLPolesZeros] = None
    coefficients: Optional[StationXMLCoefficients] = None
    stage_gain: Optional[StationXMLStageGain] = None
    decimation_input_sample_rate: Optional[float] = None
    decimation_factor: int = 1
    decimation_offset: int = 0
    decimation_delay: float = 0.0
    decimation_correction: float = 0.0
    input_units: str = ""
    output_units: str = ""


@dataclass
class StationXMLSensitivity:
    """Overall InstrumentSensitivity block."""

    value: float = 1.0
    frequency: float = 1.0
    input_units: str = ""
    output_units: str = ""
    input_units_description: str = ""
    output_units_description: str = ""


@dataclass
class StationXMLChannel:
    """Parsed FDSN StationXML Channel record including orientation and instrument response."""

    network: str
    station: str
    location: str
    code: str
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    depth: float = 0.0
    azimuth: float = 0.0
    dip: float = 0.0
    sample_rate: float = 20.0
    start_date: str = ""
    end_date: str = ""
    start_epoch: Optional[float] = None
    end_epoch: Optional[float] = None
    sensor_description: str = ""
    instrument_sensitivity: Optional[StationXMLSensitivity] = None
    poles_zeros: Optional[StationXMLPolesZeros] = None
    stages: List[StationXMLResponseStage] = field(default_factory=list)

    @property
    def stream_name(self) -> str:
        """Return SHM stream identifier '<stat>-<chan1><chan2>-<comp>' in lowercase."""
        ch = (self.code or "BHZ").strip().lower()
        c12 = ch[:2] if len(ch) >= 2 else (ch + "h")[:2]
        cp = ch[2:3] if len(ch) >= 3 else "z"
        return f"{self.station.strip().lower()}-{c12}-{cp}"

    @property
    def seed_id(self) -> str:
        """Return SEED NSLC identifier 'NET.STA.LOC.CHA'."""
        return f"{self.network}.{self.station}.{self.location}.{self.code}"

    def matches_epoch(self, epoch: Optional[float]) -> bool:
        """Return True if epoch falls within [start_epoch, end_epoch] (open bounds allowed)."""
        if epoch is None:
            return True
        if self.start_epoch is not None and epoch < self.start_epoch:
            return False
        if self.end_epoch is not None and epoch > self.end_epoch:
            return False
        return True


@dataclass
class StationXMLStation:
    """Parsed FDSN StationXML Station record."""

    network: str
    code: str
    name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    start_date: str = ""
    end_date: str = ""
    start_epoch: Optional[float] = None
    end_epoch: Optional[float] = None
    channels: List[StationXMLChannel] = field(default_factory=list)


@dataclass
class StationXMLInventory:
    """Top-level parsed FDSN StationXML Inventory."""

    source: str = ""
    sender: str = ""
    created: str = ""
    stations: List[StationXMLStation] = field(default_factory=list)
    channels: List[StationXMLChannel] = field(default_factory=list)

    def get_channel(
        self,
        net: str,
        sta: str,
        loc: str = "",
        cha: str = "",
        epoch: Optional[float] = None,
    ) -> Optional[StationXMLChannel]:
        """Find best-matching channel by NSLC and optional epoch."""
        net_u = (net or "").strip().upper()
        sta_u = (sta or "").strip().upper()
        loc_u = _norm_loc(loc)
        cha_u = (cha or "").strip().upper()

        candidates: List[StationXMLChannel] = []
        for ch in self.channels:
            if net_u and net_u != "*" and ch.network.upper() != net_u:
                continue
            if sta_u and sta_u != "*" and ch.station.upper() != sta_u:
                continue
            if loc_u != "*" and _norm_loc(ch.location) != loc_u:
                continue
            if cha_u and cha_u != "*" and ch.code.upper() != cha_u:
                continue
            if ch.matches_epoch(epoch):
                candidates.append(ch)

        if not candidates and loc_u == "":
            # Fallback if query omitted location code while inventory has a populated one
            for ch in self.channels:
                if net_u and net_u != "*" and ch.network.upper() != net_u:
                    continue
                if sta_u and sta_u != "*" and ch.station.upper() != sta_u:
                    continue
                if cha_u and cha_u != "*" and ch.code.upper() != cha_u:
                    continue
                if ch.matches_epoch(epoch):
                    candidates.append(ch)

        if not candidates:
            return None

        if epoch is None:
            # Prefer currently open epoch (end_epoch is None), otherwise latest start_epoch
            candidates.sort(
                key=lambda c: (
                    1 if c.end_epoch is None else 0,
                    c.start_epoch if c.start_epoch is not None else float("-inf"),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda c: c.start_epoch if c.start_epoch is not None else float("-inf"),
                reverse=True,
            )
        return candidates[0]


def _norm_loc(loc: str) -> str:
    s = (loc or "").strip()
    if s in ("--", ""):
        return ""
    return s.upper()


def _parse_units(elem: Optional[ET.Element]) -> Tuple[str, str]:
    if elem is None:
        return "", ""
    return _child_text(elem, "Name", ""), _child_text(elem, "Description", "")


def _parse_complex_list(parent: ET.Element, tag_name: str) -> List[complex]:
    items = _findall_children(parent, tag_name)
    # Sort by optional 'number' attribute if present to preserve SEED ordering
    def _sort_key(el: ET.Element) -> Tuple[int, int]:
        num_str = el.attrib.get("number", "")
        if num_str:
            try:
                return (0, int(num_str))
            except ValueError:
                pass
        return (1, 0)

    items_sorted = sorted(items, key=_sort_key)
    out: List[complex] = []
    for item in items_sorted:
        re_val = _child_float(item, "Real", 0.0)
        im_val = _child_float(item, "Imaginary", 0.0)
        out.append(complex(re_val, im_val))
    return out


def _parse_poles_zeros(elem: ET.Element) -> StationXMLPolesZeros:
    in_u, _ = _parse_units(_find_child(elem, "InputUnits"))
    out_u, _ = _parse_units(_find_child(elem, "OutputUnits"))
    pz_type = _child_text(elem, "PzTransferFunctionType", "LAPLACE (RADIANS/SECOND)")
    norm_factor = _child_float(elem, "NormalizationFactor", 1.0)
    norm_freq = _child_float(elem, "NormalizationFrequency", 1.0)
    zeros = _parse_complex_list(elem, "Zero")
    poles = _parse_complex_list(elem, "Pole")
    return StationXMLPolesZeros(
        pz_transfer_function_type=pz_type,
        normalization_factor=norm_factor,
        normalization_frequency=norm_freq,
        zeros=zeros,
        poles=poles,
        input_units=in_u,
        output_units=out_u,
        resource_id=elem.attrib.get("resourceId", ""),
    )


def _parse_coefficients(elem: ET.Element) -> StationXMLCoefficients:
    in_u, _ = _parse_units(_find_child(elem, "InputUnits"))
    out_u, _ = _parse_units(_find_child(elem, "OutputUnits"))
    cf_type = _child_text(elem, "CfTransferFunctionType", "DIGITAL")
    num: List[float] = []
    for node in _findall_children(elem, "Numerator"):
        if node.text is not None and node.text.strip():
            try:
                num.append(float(node.text.strip()))
            except ValueError:
                pass
    den: List[float] = []
    for node in _findall_children(elem, "Denominator"):
        if node.text is not None and node.text.strip():
            try:
                den.append(float(node.text.strip()))
            except ValueError:
                pass
    return StationXMLCoefficients(
        cf_transfer_function_type=cf_type,
        numerator=num,
        denominator=den,
        input_units=in_u,
        output_units=out_u,
    )


def _parse_response_stage(elem: ET.Element) -> StationXMLResponseStage:
    num_attr = elem.attrib.get("number", "1")
    try:
        stage_num = int(num_attr)
    except ValueError:
        stage_num = 1

    pz_el = _find_child(elem, "PolesZeros")
    pz = _parse_poles_zeros(pz_el) if pz_el is not None else None

    cf_el = _find_child(elem, "Coefficients")
    cf = _parse_coefficients(cf_el) if cf_el is not None else None

    sg_el = _find_child(elem, "StageGain")
    sg: Optional[StationXMLStageGain] = None
    if sg_el is not None:
        sg = StationXMLStageGain(
            value=_child_float(sg_el, "Value", 1.0),
            frequency=_child_float(sg_el, "Frequency", 1.0),
        )

    dec_el = _find_child(elem, "Decimation")
    dec_sr: Optional[float] = None
    dec_factor = 1
    dec_offset = 0
    dec_delay = 0.0
    dec_corr = 0.0
    if dec_el is not None:
        sr_txt = _child_text(dec_el, "InputSampleRate", "")
        if sr_txt:
            try:
                dec_sr = float(sr_txt)
            except ValueError:
                dec_sr = None
        dec_factor = _child_int(dec_el, "Factor", 1)
        dec_offset = _child_int(dec_el, "Offset", 0)
        dec_delay = _child_float(dec_el, "Delay", 0.0)
        dec_corr = _child_float(dec_el, "Correction", 0.0)

    in_u = pz.input_units if pz is not None else (cf.input_units if cf is not None else "")
    out_u = pz.output_units if pz is not None else (cf.output_units if cf is not None else "")

    return StationXMLResponseStage(
        number=stage_num,
        poles_zeros=pz,
        coefficients=cf,
        stage_gain=sg,
        decimation_input_sample_rate=dec_sr,
        decimation_factor=dec_factor,
        decimation_offset=dec_offset,
        decimation_delay=dec_delay,
        decimation_correction=dec_corr,
        input_units=in_u,
        output_units=out_u,
    )


def _extract_dates(elem: ET.Element) -> Tuple[str, str]:
    start_date = elem.attrib.get("startDate", "") or _child_text(elem, "StartDate", "")
    end_date = elem.attrib.get("endDate", "") or _child_text(elem, "EndDate", "")
    return start_date.strip(), end_date.strip()


def parse_stationxml(xml_or_path: str) -> StationXMLInventory:
    """Parse an FDSN StationXML 1.x document (string or filepath) into StationXMLInventory."""
    if not xml_or_path or not isinstance(xml_or_path, str):
        raise ValueError("StationXML input must be a non-empty XML string or file path")

    stripped = xml_or_path.lstrip()
    if stripped.startswith("<"):
        try:
            root = ET.fromstring(xml_or_path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid StationXML content: {exc}") from exc
    else:
        if not os.path.isfile(xml_or_path):
            raise FileNotFoundError(f"StationXML file not found: {xml_or_path}")
        try:
            tree = ET.parse(xml_or_path)
            root = tree.getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid StationXML file {xml_or_path}: {exc}") from exc

    if _local_tag(root.tag) != "FDSNStationXML":
        raise ValueError(f"Expected root element FDSNStationXML, got {_local_tag(root.tag)!r}")

    inv = StationXMLInventory(
        source=_child_text(root, "Source", ""),
        sender=_child_text(root, "Sender", ""),
        created=_child_text(root, "Created", ""),
    )

    for net_el in _findall_children(root, "Network"):
        net_code = net_el.attrib.get("code", "").strip().upper()
        for sta_el in _findall_children(net_el, "Station"):
            sta_code = sta_el.attrib.get("code", "").strip().upper()
            site_el = _find_child(sta_el, "Site")
            sta_name = _child_text(site_el, "Name", sta_code) if site_el is not None else sta_code
            sta_lat = _child_float(sta_el, "Latitude", 0.0)
            sta_lon = _child_float(sta_el, "Longitude", 0.0)
            sta_elev = _child_float(sta_el, "Elevation", 0.0)
            sta_sdate, sta_edate = _extract_dates(sta_el)

            sta_obj = StationXMLStation(
                network=net_code,
                code=sta_code,
                name=sta_name,
                latitude=sta_lat,
                longitude=sta_lon,
                elevation=sta_elev,
                start_date=sta_sdate,
                end_date=sta_edate,
                start_epoch=parse_iso_epoch(sta_sdate),
                end_epoch=parse_iso_epoch(sta_edate),
            )

            for cha_el in _findall_children(sta_el, "Channel"):
                cha_code = cha_el.attrib.get("code", "").strip().upper()
                loc_code = cha_el.attrib.get("locationCode", "").strip()
                cha_sdate, cha_edate = _extract_dates(cha_el)
                if not cha_sdate:
                    cha_sdate = sta_sdate
                if not cha_edate:
                    cha_edate = sta_edate

                sensor_el = _find_child(cha_el, "Sensor")
                sensor_desc = ""
                if sensor_el is not None:
                    sensor_desc = (
                        _child_text(sensor_el, "Type", "")
                        or _child_text(sensor_el, "Model", "")
                        or _child_text(sensor_el, "Description", "")
                    )

                resp_el = _find_child(cha_el, "Response")
                sens_obj: Optional[StationXMLSensitivity] = None
                stages: List[StationXMLResponseStage] = []
                first_pz: Optional[StationXMLPolesZeros] = None

                if resp_el is not None:
                    sens_el = _find_child(resp_el, "InstrumentSensitivity")
                    if sens_el is not None:
                        in_u, in_desc = _parse_units(_find_child(sens_el, "InputUnits"))
                        out_u, out_desc = _parse_units(_find_child(sens_el, "OutputUnits"))
                        sens_obj = StationXMLSensitivity(
                            value=_child_float(sens_el, "Value", 1.0),
                            frequency=_child_float(sens_el, "Frequency", 1.0),
                            input_units=in_u,
                            output_units=out_u,
                            input_units_description=in_desc,
                            output_units_description=out_desc,
                        )

                    for stg_el in _findall_children(resp_el, "Stage"):
                        stg = _parse_response_stage(stg_el)
                        stages.append(stg)
                    stages.sort(key=lambda s: s.number)
                    for stg in stages:
                        if stg.poles_zeros is not None:
                            first_pz = stg.poles_zeros
                            break

                cha_obj = StationXMLChannel(
                    network=net_code,
                    station=sta_code,
                    location=loc_code,
                    code=cha_code,
                    latitude=_child_float(cha_el, "Latitude", sta_lat),
                    longitude=_child_float(cha_el, "Longitude", sta_lon),
                    elevation=_child_float(cha_el, "Elevation", sta_elev),
                    depth=_child_float(cha_el, "Depth", 0.0),
                    azimuth=_child_float(cha_el, "Azimuth", 0.0),
                    dip=_child_float(cha_el, "Dip", 0.0),
                    sample_rate=_child_float(cha_el, "SampleRate", 20.0),
                    start_date=cha_sdate,
                    end_date=cha_edate,
                    start_epoch=parse_iso_epoch(cha_sdate),
                    end_epoch=parse_iso_epoch(cha_edate),
                    sensor_description=sensor_desc,
                    instrument_sensitivity=sens_obj,
                    poles_zeros=first_pz,
                    stages=stages,
                )
                sta_obj.channels.append(cha_obj)
                inv.channels.append(cha_obj)

            inv.stations.append(sta_obj)

    return inv


def eval_stationxml_response(
    channel: StationXMLChannel,
    freqs_hz: Union[np.ndarray, Sequence[float], float],
    include_sensitivity: bool = True,
) -> np.ndarray:
    """Evaluate complex instrument frequency response H(f) in NumPy at freqs_hz.

    Supports multi-stage StationXML responses containing:
    - PolesZeros (`LAPLACE (RADIANS/SECOND)`, `LAPLACE (HERTZ)`, `DIGITAL (Z-TRANSFORM)`)
    - Coefficients (FIR / IIR rational digital filter in z^-1)
    - StageGain and overall InstrumentSensitivity scaling.
    """
    f_arr = np.asarray(freqs_hz, dtype=np.float64)
    scalar_input = f_arr.ndim == 0
    freqs = np.atleast_1d(f_arr).astype(np.float64)

    resp = np.ones(freqs.shape, dtype=np.complex128)
    has_stage_gain = False

    if channel.stages:
        for stg in channel.stages:
            stg_resp = np.ones(freqs.shape, dtype=np.complex128)

            if stg.poles_zeros is not None:
                pz = stg.poles_zeros
                pz_type = (pz.pz_transfer_function_type or "").strip().upper()
                if "RADIANS" in pz_type:
                    s = 1j * 2.0 * math.pi * freqs
                elif "HERTZ" in pz_type:
                    s = 1j * freqs
                elif "DIGITAL" in pz_type or "Z-TRANSFORM" in pz_type:
                    sr = (
                        stg.decimation_input_sample_rate
                        if stg.decimation_input_sample_rate and stg.decimation_input_sample_rate > 0
                        else (channel.sample_rate if channel.sample_rate > 0 else 20.0)
                    )
                    dt = 1.0 / sr
                    s = np.exp(1j * 2.0 * math.pi * freqs * dt)
                else:
                    s = 1j * 2.0 * math.pi * freqs

                num = np.full(freqs.shape, complex(pz.normalization_factor, 0.0), dtype=np.complex128)
                for z in pz.zeros:
                    num = num * (s - z)
                den = np.ones(freqs.shape, dtype=np.complex128)
                for p in pz.poles:
                    den = den * (s - p)

                pz_val = np.zeros(freqs.shape, dtype=np.complex128)
                mask = np.abs(den) > 1e-30
                pz_val[mask] = num[mask] / den[mask]
                stg_resp = stg_resp * pz_val

            elif stg.coefficients is not None:
                cf = stg.coefficients
                sr = (
                    stg.decimation_input_sample_rate
                    if stg.decimation_input_sample_rate and stg.decimation_input_sample_rate > 0
                    else (channel.sample_rate if channel.sample_rate > 0 else 20.0)
                )
                dt = 1.0 / sr
                z_inv = np.exp(-1j * 2.0 * math.pi * freqs * dt)

                if cf.numerator:
                    num = np.zeros(freqs.shape, dtype=np.complex128)
                    for k, b_k in enumerate(cf.numerator):
                        num = num + b_k * (z_inv ** k)
                else:
                    num = np.ones(freqs.shape, dtype=np.complex128)

                if cf.denominator:
                    den = np.zeros(freqs.shape, dtype=np.complex128)
                    for m, a_m in enumerate(cf.denominator):
                        den = den + a_m * (z_inv ** m)
                else:
                    den = np.ones(freqs.shape, dtype=np.complex128)

                cf_val = np.zeros(freqs.shape, dtype=np.complex128)
                mask = np.abs(den) > 1e-30
                cf_val[mask] = num[mask] / den[mask]
                stg_resp = stg_resp * cf_val

            if stg.stage_gain is not None:
                has_stage_gain = True
                if include_sensitivity:
                    stg_resp = stg_resp * float(stg.stage_gain.value)

            resp = resp * stg_resp

    if include_sensitivity and not has_stage_gain and channel.instrument_sensitivity is not None:
        resp = resp * float(channel.instrument_sensitivity.value)

    if scalar_input:
        return resp[0]
    return resp
