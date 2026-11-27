"""QuakeML 1.2 schema-compliant XML exporter and parser (pure Python stdlib, XML ElementTree)."""

from __future__ import annotations

import datetime
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.io.stationxml import parse_iso_epoch

QML_NS = "http://quakeml.org/xmlns/quakeml/1.2"
BED_NS = "http://quakeml.org/xmlns/bed/1.2"

_QUALITY_SIGMA_TABLE = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 4.0, 8.0, 15.0]

_SHM_TO_QML_EVENT_TYPE = {
    "t": "earthquake",
    "r": "earthquake",
    "l": "earthquake",
    "n": "nuclear explosion",
    "b": "quarry blast",
    "m": "induced or triggered event",
}

_QML_TO_SHM_EVENT_TYPE = {
    "earthquake": "t",
    "nuclear explosion": "n",
    "quarry blast": "b",
    "explosion": "b",
    "induced or triggered event": "m",
    "mining explosion": "m",
    "rock burst": "m",
}


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    for child in elem:
        if _local_tag(child.tag) == name:
            return child
    low = name.lower()
    for child in elem:
        if _local_tag(child.tag).lower() == low:
            return child
    return None


def _findall_children(elem: ET.Element, name: str) -> List[ET.Element]:
    out = [child for child in elem if _local_tag(child.tag) == name]
    if not out:
        low = name.lower()
        out = [child for child in elem if _local_tag(child.tag).lower() == low]
    return out


def _child_text(elem: ET.Element, name: str, default: str = "") -> str:
    child = _find_child(elem, name)
    if child is not None and child.text is not None:
        return child.text.strip()
    return default


def _child_float(elem: ET.Element, name: str, default: Optional[float] = None) -> Optional[float]:
    txt = _child_text(elem, name, "")
    if not txt:
        return default
    try:
        return float(txt)
    except ValueError:
        return default


def _value_float(elem: ET.Element, name: str, default: Optional[float] = None) -> Optional[float]:
    """Read <name><value>...</value></name> float."""
    node = _find_child(elem, name)
    if node is None:
        return default
    return _child_float(node, "value", default)


def _value_and_unc(elem: ET.Element, name: str) -> Tuple[Optional[float], Optional[float]]:
    """Read <name><value>...</value><uncertainty>...</uncertainty></name>."""
    node = _find_child(elem, name)
    if node is None:
        return None, None
    return _child_float(node, "value", None), _child_float(node, "uncertainty", None)


def _epoch_to_iso(epoch: float) -> str:
    dt = datetime.datetime.fromtimestamp(float(epoch), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}Z"


def _sh_or_iso_to_iso(timestr: str, fallback_epoch: Optional[float] = None) -> str:
    s = (timestr or "").strip()
    if s:
        iso_ep = parse_iso_epoch(s)
        if iso_ep is not None:
            return _epoch_to_iso(iso_ep)
        try:
            ep = parse_sh_time(s)
            return _epoch_to_iso(ep)
        except ValueError:
            pass
    if fallback_epoch is not None and fallback_epoch > 0.0:
        return _epoch_to_iso(fallback_epoch)
    return _epoch_to_iso(0.0)


def _bed(parent: ET.Element, tag: str, text: Optional[str] = None, **attrib: str) -> ET.Element:
    el = ET.SubElement(parent, f"{{{BED_NS}}}{tag}", attrib)
    if text is not None:
        el.text = str(text)
    return el


def _bed_rv(parent: ET.Element, tag: str, value: float, uncertainty: Optional[float] = None) -> ET.Element:
    node = _bed(parent, tag)
    _bed(node, "value", f"{float(value):.6g}" if abs(value) >= 1e4 or (0 < abs(value) < 1e-4) else f"{float(value):.6f}".rstrip("0").rstrip(".") or "0")
    if uncertainty is not None and uncertainty > 0.0:
        _bed(node, "uncertainty", f"{float(uncertainty):.6g}")
    return node


def _pick_sigma_t(pick: PhasePick) -> float:
    q = max(0, min(9, int(pick.quality)))
    return _QUALITY_SIGMA_TABLE[q]


def export_quakeml(
    traces: Sequence[Trace],
    params: AnalysisParamSet,
    public_id_prefix: str = "smi:shm-modern",
    focal_mechanism: Optional[Dict[str, Any]] = None,
) -> str:
    """Export event parameters, picks, arrivals, uncertainties, and optional focal mechanism as QuakeML 1.2 XML."""
    prefix = (public_id_prefix or "smi:shm-modern").rstrip("/")

    ET.register_namespace("", BED_NS)
    ET.register_namespace("q", QML_NS)

    root = ET.Element(f"{{{QML_NS}}}quakeml")
    ev_uid = str(params.event_id) if getattr(params, "event_id", 0) else "event_1"
    ev_params_el = _bed(root, "eventParameters", publicID=f"{prefix}/eventParameters/{ev_uid}")
    event_el = _bed(ev_params_el, "event", publicID=f"{prefix}/event/{ev_uid}")

    origin_id = f"{prefix}/origin/{ev_uid}"
    _bed(event_el, "preferredOriginID", origin_id)

    # Determine magnitudes to serialize
    mag_specs: List[Tuple[str, float, float]] = []
    for m_type, m_val, m_err in (
        ("Mw", getattr(params, "mag_mw", 0.0), 0.0),
        ("mb", getattr(params, "mag_mb", 0.0), getattr(params, "d_mb", 0.0)),
        ("MS", getattr(params, "mag_ms", 0.0), getattr(params, "d_ms", 0.0)),
        ("ML", getattr(params, "mag_ml", 0.0), getattr(params, "d_ml", 0.0)),
        ("mB", getattr(params, "mag_mbb", 0.0), getattr(params, "d_mbb", 0.0)),
    ):
        if m_val and float(m_val) != 0.0:
            mag_specs.append((m_type, float(m_val), float(m_err or 0.0)))

    if mag_specs:
        _bed(event_el, "preferredMagnitudeID", f"{prefix}/magnitude/{ev_uid}/{mag_specs[0][0]}")

    ev_type_str = _SHM_TO_QML_EVENT_TYPE.get((params.event_type or "t").strip().lower(), "earthquake")
    _bed(event_el, "type", ev_type_str)

    if params.region_name:
        desc_el = _bed(event_el, "description")
        _bed(desc_el, "type", "Flinn-Engdahl region")
        _bed(desc_el, "text", params.region_name)

    cinfo = _bed(event_el, "creationInfo")
    _bed(cinfo, "agencyURI", f"{prefix}/agency/{params.source or 'SHM'}")
    _bed(cinfo, "creationTime", _epoch_to_iso(datetime.datetime.now(tz=datetime.timezone.utc).timestamp()))
    _bed(cinfo, "version", "1.2")

    # Collect picks with parent trace metadata
    pick_records: List[Tuple[str, Trace, PhasePick]] = []
    seen_pick_keys = set()
    pick_counter = 0
    for tr in traces or []:
        for pk in tr.picks or []:
            key = (
                (pk.station or tr.station).upper(),
                tr.channel3,
                pk.name,
                pk.onset,
            )
            if key in seen_pick_keys:
                continue
            seen_pick_keys.add(key)
            pick_counter += 1
            pick_id = f"{prefix}/pick/{ev_uid}/{pick_counter}"
            pick_records.append((pick_id, tr, pk))

    # Build <origin>
    origin_el = _bed(event_el, "origin", publicID=origin_id)
    fallback_origin_ep = traces[0].torig if traces and traces[0].torig > 0 else None
    _bed(_bed(origin_el, "time"), "value", _sh_or_iso_to_iso(params.origin, fallback_origin_ep))

    lat_unc = (params.d_lat / 111.1949266) if getattr(params, "d_lat", 0.0) > 0.0 else None
    lon_unc = None
    if getattr(params, "d_lon", 0.0) > 0.0:
        cos_lat = max(0.05, math.cos(math.radians(params.lat)))
        lon_unc = params.d_lon / (111.1949266 * cos_lat)
    _bed_rv(origin_el, "latitude", params.lat, lat_unc)
    _bed_rv(origin_el, "longitude", params.lon, lon_unc)

    depth_m = float(params.depth) * 1000.0
    d_depth_m = float(params.d_depth) * 1000.0 if getattr(params, "d_depth", 0.0) > 0.0 else None
    _bed_rv(origin_el, "depth", depth_m, d_depth_m)

    depth_type_map = {1: "operator assigned", 2: "from depth phases", 3: "from location"}
    _bed(origin_el, "depthType", depth_type_map.get(int(getattr(params, "depth_type", 1)), "from location"))
    _bed(origin_el, "methodID", f"{prefix}/method/{params.source or 'Geiger'}")

    # <quality>
    stations_used = {(pk.station or tr.station).upper() for _, tr, pk in pick_records if pk.weight > 0}
    if not stations_used and traces:
        stations_used = {tr.station.upper() for tr in traces if tr.station}
    dists = [float(tr.distance) for _, tr, _ in pick_records if tr.distance > 0.0]
    has_geometry = any(tr.azimuth != 0.0 or tr.distance > 0.0 for _, tr, _ in pick_records)
    azims = sorted([float(tr.azimuth) % 360.0 for _, tr, _ in pick_records]) if has_geometry else []
    az_gap = 0.0
    if len(azims) >= 2:
        gaps = [azims[i + 1] - azims[i] for i in range(len(azims) - 1)]
        gaps.append(360.0 - azims[-1] + azims[0])
        az_gap = max(gaps)

    qual_el = _bed(origin_el, "quality")
    _bed(qual_el, "associatedPhaseCount", str(len(pick_records)))
    _bed(qual_el, "usedPhaseCount", str(sum(1 for _, _, pk in pick_records if pk.weight > 0)))
    _bed(qual_el, "associatedStationCount", str(len(stations_used)))
    _bed(qual_el, "usedStationCount", str(len(stations_used)))
    std_err = getattr(params, "rmse", 0.0) or getattr(params, "wrmse", 0.0) or getattr(params, "chi2_reduced", 0.0)
    _bed(qual_el, "standardError", f"{float(std_err):.6f}")
    if az_gap > 0.0:
        _bed(qual_el, "azimuthalGap", f"{az_gap:.2f}")
    if dists:
        _bed(qual_el, "minimumDistance", f"{min(dists):.4f}")
        _bed(qual_el, "maximumDistance", f"{max(dists):.4f}")

    has_manual = any(pk.source != "a" for _, _, pk in pick_records) if pick_records else True
    _bed(origin_el, "evaluationMode", "manual" if has_manual else "automatic")

    # <originUncertainty>
    err_ell = getattr(params, "error_ellipse", None)
    if err_ell is None and (getattr(params, "d_lat", 0.0) > 0.0 or getattr(params, "d_lon", 0.0) > 0.0):
        a_km = max(float(params.d_lat), float(params.d_lon))
        b_km = min(float(params.d_lat), float(params.d_lon))
        strike_deg = 0.0 if params.d_lat >= params.d_lon else 90.0
        err_ell = (a_km, b_km, strike_deg)

    if err_ell is not None or getattr(params, "chi2_reduced", 0.0) > 0.0:
        ou_el = _bed(origin_el, "originUncertainty")
        if err_ell is not None:
            semi_major_km, semi_minor_km, strike_deg = float(err_ell[0]), float(err_ell[1]), float(err_ell[2])
            if semi_minor_km > semi_major_km:
                semi_major_km, semi_minor_km = semi_minor_km, semi_major_km
            semi_major_m = max(0.0, semi_major_km * 1000.0)
            semi_minor_m = max(0.0, semi_minor_km * 1000.0)
            semi_interm_m = min(
                semi_major_m,
                max(semi_minor_m, float(getattr(params, "d_depth", 0.0)) * 1000.0),
            )
            _bed(ou_el, "preferredDescription", "confidence ellipsoid")
            _bed(ou_el, "horizontalUncertainty", f"{semi_major_m:.3f}")
            _bed(ou_el, "minHorizontalUncertainty", f"{semi_minor_m:.3f}")
            _bed(ou_el, "maxHorizontalUncertainty", f"{semi_major_m:.3f}")
            _bed(ou_el, "azimuthMaxHorizontalUncertainty", f"{strike_deg % 180.0:.4f}")

            ce_el = _bed(ou_el, "confidenceEllipsoid")
            _bed(ce_el, "semiMajorAxisLength", f"{semi_major_m:.3f}")
            _bed(ce_el, "semiMinorAxisLength", f"{semi_minor_m:.3f}")
            _bed(ce_el, "semiIntermediateAxisLength", f"{semi_interm_m:.3f}")
            _bed(ce_el, "majorAxisPlunge", "0.0")
            _bed(ce_el, "majorAxisAzimuth", f"{strike_deg % 360.0:.4f}")
            _bed(ce_el, "majorAxisRotation", "0.0")

    # <arrival> elements inside <origin>
    for idx, (pick_id, tr, pk) in enumerate(pick_records, start=1):
        arr_el = _bed(origin_el, "arrival", publicID=f"{prefix}/arrival/{ev_uid}/{idx}")
        _bed(arr_el, "pickID", pick_id)
        _bed(arr_el, "phase", pk.name or "P")
        az_val = tr.azimuth if (tr.azimuth != 0.0 or tr.distance > 0.0) else pk.beam_azim
        _bed(arr_el, "azimuth", f"{float(az_val):.4f}")
        _bed(arr_el, "distance", f"{float(tr.distance):.4f}")
        _bed(arr_el, "timeResidual", f"{float(pk.resid):.4f}")
        _bed(arr_el, "timeWeight", f"{float(pk.weight):.4f}")

    # <magnitude> elements
    sta_count = int(getattr(params, "mags_count", 0)) or len(stations_used)
    for m_type, m_val, m_err in mag_specs:
        mag_el = _bed(event_el, "magnitude", publicID=f"{prefix}/magnitude/{ev_uid}/{m_type}")
        _bed_rv(mag_el, "mag", m_val, m_err if m_err > 0.0 else None)
        _bed(mag_el, "type", m_type)
        _bed(mag_el, "originID", origin_id)
        _bed(mag_el, "stationCount", str(sta_count))

    # <pick> elements
    for pick_id, tr, pk in pick_records:
        pk_el = _bed(event_el, "pick", publicID=pick_id)
        t_el = _bed(pk_el, "time")
        onset_iso = _sh_or_iso_to_iso(pk.onset, tr.start_epoch if tr.start_epoch > 0 else None)
        _bed(t_el, "value", onset_iso)
        _bed(t_el, "uncertainty", f"{_pick_sigma_t(pk):.4f}")

        net_code = (tr.network or "XX").strip().upper()
        sta_code = (pk.station or tr.station or "STAT").strip().upper()
        loc_code = (tr.locid or "").strip()
        cha_code = tr.channel3
        if pk.chan1 or pk.chan2 or pk.comp:
            c1 = (pk.chan1 or tr.chan1 or "B")[:1].upper()
            c2 = (pk.chan2 or tr.chan2 or "H")[:1].upper()
            cp = (pk.comp or tr.comp or "Z")[:1].upper()
            cha_code = f"{c1}{c2}{cp}"

        wf_attrib = {
            "networkCode": net_code,
            "stationCode": sta_code,
            "channelCode": cha_code,
        }
        if loc_code:
            wf_attrib["locationCode"] = loc_code
        _bed(pk_el, "waveformID", **wf_attrib)
        _bed(pk_el, "phaseHint", pk.name or "P")

        if pk.sign > 0:
            _bed(pk_el, "polarity", "positive")
        elif pk.sign < 0:
            _bed(pk_el, "polarity", "negative")
        else:
            _bed(pk_el, "polarity", "undecidable")

        onset_map = {"i": "impulsive", "e": "emergent"}
        _bed(pk_el, "onset", onset_map.get((pk.spec or "e").lower(), "emergent"))
        _bed(pk_el, "evaluationMode", "automatic" if pk.source == "a" else "manual")

    # Optional <focalMechanism>
    fm = focal_mechanism
    if fm is None:
        fm = getattr(params, "focal_mechanism", None)
    if isinstance(fm, dict) and (
        "strike" in fm or "nodal_plane1" in fm or "strike1" in fm
    ):
        fm_el = _bed(event_el, "focalMechanism", publicID=f"{prefix}/focalMechanism/{ev_uid}")
        _bed(fm_el, "triggeringOriginID", origin_id)
        np_el = _bed(fm_el, "nodalPlanes", preferredPlane="1")

        np1 = fm.get("nodal_plane1", fm)
        s1 = float(np1.get("strike", np1.get("strike1", 0.0)))
        d1 = float(np1.get("dip", np1.get("dip1", 0.0)))
        r1 = float(np1.get("rake", np1.get("rake1", 0.0)))
        n1_el = _bed(np_el, "nodalPlane1")
        _bed_rv(n1_el, "strike", s1)
        _bed_rv(n1_el, "dip", d1)
        _bed_rv(n1_el, "rake", r1)

        np2 = fm.get("nodal_plane2")
        if np2 is None and "strike2" in fm:
            np2 = {"strike": fm["strike2"], "dip": fm.get("dip2", 0.0), "rake": fm.get("rake2", 0.0)}
        if isinstance(np2, dict) and "strike" in np2:
            n2_el = _bed(np_el, "nodalPlane2")
            _bed_rv(n2_el, "strike", float(np2.get("strike", 0.0)))
            _bed_rv(n2_el, "dip", float(np2.get("dip", 0.0)))
            _bed_rv(n2_el, "rake", float(np2.get("rake", 0.0)))

    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")

    xml_body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_body}\n'


def write_quakeml(
    filepath: str,
    traces: Sequence[Trace],
    params: AnalysisParamSet,
    public_id_prefix: str = "smi:shm-modern",
    focal_mechanism: Optional[Dict[str, Any]] = None,
) -> None:
    """Write QuakeML 1.2 XML file to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    xml_str = export_quakeml(
        traces=traces,
        params=params,
        public_id_prefix=public_id_prefix,
        focal_mechanism=focal_mechanism,
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xml_str)


def parse_quakeml(xml_text_or_path: str) -> Dict[str, Any]:
    """Parse QuakeML 1.x XML string or file into a structured dictionary + AnalysisParamSet."""
    if not xml_text_or_path or not isinstance(xml_text_or_path, str):
        raise ValueError("QuakeML input must be a non-empty XML string or file path")

    stripped = xml_text_or_path.lstrip()
    if stripped.startswith("<"):
        try:
            root = ET.fromstring(xml_text_or_path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid QuakeML content: {exc}") from exc
    else:
        if not os.path.isfile(xml_text_or_path):
            raise FileNotFoundError(f"QuakeML file not found: {xml_text_or_path}")
        try:
            root = ET.parse(xml_text_or_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid QuakeML file {xml_text_or_path}: {exc}") from exc

    root_ns = ""
    if root.tag.startswith("{") and "}" in root.tag:
        root_ns = root.tag[1:].split("}", 1)[0]

    ep_el = _find_child(root, "eventParameters")
    if ep_el is None and _local_tag(root.tag) == "eventParameters":
        ep_el = root
    if ep_el is None:
        raise ValueError(f"Missing eventParameters in QuakeML document (root={_local_tag(root.tag)!r})")

    is_v12 = ("1.2" in root_ns) or any("1.2" in el.tag for el in root.iter())

    events_out: List[Dict[str, Any]] = []
    for ev_el in _findall_children(ep_el, "event"):
        ev_id = ev_el.attrib.get("publicID", "")
        pref_origin_id = _child_text(ev_el, "preferredOriginID", "")
        pref_mag_id = _child_text(ev_el, "preferredMagnitudeID", "")
        ev_type = _child_text(ev_el, "type", "earthquake")
        desc_el = _find_child(ev_el, "description")
        region_name = _child_text(desc_el, "text", "") if desc_el is not None else ""

        origins_out: List[Dict[str, Any]] = []
        for org_el in _findall_children(ev_el, "origin"):
            org_id = org_el.attrib.get("publicID", "")
            t_node = _find_child(org_el, "time")
            t_str = _child_text(t_node, "value", "") if t_node is not None else ""
            t_epoch = parse_iso_epoch(t_str)
            lat_val, lat_unc = _value_and_unc(org_el, "latitude")
            lon_val, lon_unc = _value_and_unc(org_el, "longitude")
            depth_raw, depth_unc_raw = _value_and_unc(org_el, "depth")

            depth_km = None
            d_depth_km = None
            if depth_raw is not None:
                depth_km = depth_raw / 1000.0 if is_v12 else depth_raw
            if depth_unc_raw is not None:
                d_depth_km = depth_unc_raw / 1000.0 if is_v12 else depth_unc_raw

            qual_el = _find_child(org_el, "quality")
            quality_dict: Dict[str, Any] = {}
            if qual_el is not None:
                for q_field in (
                    "associatedPhaseCount",
                    "usedPhaseCount",
                    "associatedStationCount",
                    "usedStationCount",
                    "standardError",
                    "azimuthalGap",
                    "minimumDistance",
                    "maximumDistance",
                ):
                    val_f = _child_float(qual_el, q_field, None)
                    if val_f is not None:
                        quality_dict[q_field] = int(val_f) if "Count" in q_field else val_f

            ou_el = _find_child(org_el, "originUncertainty")
            ou_dict: Dict[str, Any] = {}
            if ou_el is not None:
                ou_dict["preferredDescription"] = _child_text(ou_el, "preferredDescription", "")
                for u_field in (
                    "horizontalUncertainty",
                    "minHorizontalUncertainty",
                    "maxHorizontalUncertainty",
                    "azimuthMaxHorizontalUncertainty",
                ):
                    u_val = _child_float(ou_el, u_field, None)
                    if u_val is not None:
                        ou_dict[u_field] = u_val
                ce_el = _find_child(ou_el, "confidenceEllipsoid")
                if ce_el is not None:
                    ou_dict["confidenceEllipsoid"] = {
                        "semiMajorAxisLength": _child_float(ce_el, "semiMajorAxisLength", 0.0),
                        "semiMinorAxisLength": _child_float(ce_el, "semiMinorAxisLength", 0.0),
                        "semiIntermediateAxisLength": _child_float(ce_el, "semiIntermediateAxisLength", 0.0),
                        "majorAxisPlunge": _child_float(ce_el, "majorAxisPlunge", 0.0),
                        "majorAxisAzimuth": _child_float(ce_el, "majorAxisAzimuth", 0.0),
                        "majorAxisRotation": _child_float(ce_el, "majorAxisRotation", 0.0),
                    }

            arrivals_out: List[Dict[str, Any]] = []
            for arr_el in _findall_children(org_el, "arrival"):
                arrivals_out.append(
                    {
                        "public_id": arr_el.attrib.get("publicID", ""),
                        "pick_id": _child_text(arr_el, "pickID", ""),
                        "phase": _child_text(arr_el, "phase", ""),
                        "azimuth": _child_float(arr_el, "azimuth", 0.0),
                        "distance": _child_float(arr_el, "distance", 0.0),
                        "time_residual": _child_float(arr_el, "timeResidual", 0.0),
                        "time_weight": _child_float(arr_el, "timeWeight", 0.0),
                    }
                )

            origins_out.append(
                {
                    "public_id": org_id,
                    "time": t_str,
                    "time_epoch": t_epoch,
                    "latitude": lat_val,
                    "latitude_uncertainty": lat_unc,
                    "longitude": lon_val,
                    "longitude_uncertainty": lon_unc,
                    "depth_m": depth_raw,
                    "depth_km": depth_km,
                    "depth_uncertainty_m": depth_unc_raw,
                    "depth_uncertainty_km": d_depth_km,
                    "depth_type": _child_text(org_el, "depthType", ""),
                    "method_id": _child_text(org_el, "methodID", ""),
                    "evaluation_mode": _child_text(org_el, "evaluationMode", ""),
                    "quality": quality_dict,
                    "origin_uncertainty": ou_dict,
                    "arrivals": arrivals_out,
                }
            )

        mags_out: List[Dict[str, Any]] = []
        for mag_el in _findall_children(ev_el, "magnitude"):
            m_val, m_unc = _value_and_unc(mag_el, "mag")
            sc_f = _child_float(mag_el, "stationCount", 0.0)
            mags_out.append(
                {
                    "public_id": mag_el.attrib.get("publicID", ""),
                    "value": m_val,
                    "uncertainty": m_unc,
                    "type": _child_text(mag_el, "type", ""),
                    "origin_id": _child_text(mag_el, "originID", ""),
                    "station_count": int(sc_f) if sc_f is not None else 0,
                }
            )

        picks_out: List[Dict[str, Any]] = []
        for pk_el in _findall_children(ev_el, "pick"):
            pk_t_node = _find_child(pk_el, "time")
            pk_t_str = _child_text(pk_t_node, "value", "") if pk_t_node is not None else ""
            pk_t_unc = _child_float(pk_t_node, "uncertainty", None) if pk_t_node is not None else None
            wf_el = _find_child(pk_el, "waveformID")
            wf_attr = dict(wf_el.attrib) if wf_el is not None else {}
            picks_out.append(
                {
                    "public_id": pk_el.attrib.get("publicID", ""),
                    "time": pk_t_str,
                    "time_epoch": parse_iso_epoch(pk_t_str),
                    "time_uncertainty": pk_t_unc,
                    "network": wf_attr.get("networkCode", ""),
                    "station": wf_attr.get("stationCode", ""),
                    "location": wf_attr.get("locationCode", ""),
                    "channel": wf_attr.get("channelCode", ""),
                    "phase_hint": _child_text(pk_el, "phaseHint", ""),
                    "polarity": _child_text(pk_el, "polarity", ""),
                    "onset": _child_text(pk_el, "onset", ""),
                    "evaluation_mode": _child_text(pk_el, "evaluationMode", ""),
                }
            )

        fms_out: List[Dict[str, Any]] = []
        for fm_el in _findall_children(ev_el, "focalMechanism"):
            nps_el = _find_child(fm_el, "nodalPlanes")
            fm_dict: Dict[str, Any] = {
                "public_id": fm_el.attrib.get("publicID", ""),
                "triggering_origin_id": _child_text(fm_el, "triggeringOriginID", ""),
            }
            if nps_el is not None:
                for plane_key in ("nodalPlane1", "nodalPlane2"):
                    pl_el = _find_child(nps_el, plane_key)
                    if pl_el is not None:
                        fm_dict[plane_key] = {
                            "strike": _value_float(pl_el, "strike", 0.0),
                            "dip": _value_float(pl_el, "dip", 0.0),
                            "rake": _value_float(pl_el, "rake", 0.0),
                        }
            fms_out.append(fm_dict)

        events_out.append(
            {
                "public_id": ev_id,
                "type": ev_type,
                "preferred_origin_id": pref_origin_id,
                "preferred_magnitude_id": pref_mag_id,
                "region_name": region_name,
                "origins": origins_out,
                "magnitudes": mags_out,
                "picks": picks_out,
                "focal_mechanisms": fms_out,
            }
        )

    # Build top-level summary and reconstructed AnalysisParamSet from primary event
    primary_ev = events_out[0] if events_out else {}
    primary_origin: Optional[Dict[str, Any]] = None
    if primary_ev:
        pref_oid = primary_ev.get("preferred_origin_id", "")
        for org in primary_ev.get("origins", []):
            if org.get("public_id") == pref_oid:
                primary_origin = org
                break
        if primary_origin is None and primary_ev.get("origins"):
            primary_origin = primary_ev["origins"][0]

    params = AnalysisParamSet()
    phase_picks: List[PhasePick] = []
    if primary_ev:
        params.event_type = _QML_TO_SHM_EVENT_TYPE.get(primary_ev.get("type", "").lower(), "t")
        params.region_name = primary_ev.get("region_name", "")
        if primary_origin is not None:
            if primary_origin.get("time_epoch") is not None:
                params.origin = format_sh_time(primary_origin["time_epoch"])
            if primary_origin.get("latitude") is not None:
                params.lat = float(primary_origin["latitude"])
            if primary_origin.get("longitude") is not None:
                params.lon = float(primary_origin["longitude"])
            if primary_origin.get("depth_km") is not None:
                params.depth = float(primary_origin["depth_km"])
            if primary_origin.get("depth_uncertainty_km") is not None:
                params.d_depth = float(primary_origin["depth_uncertainty_km"])
            qual = primary_origin.get("quality", {})
            if "standardError" in qual:
                params.rmse = float(qual["standardError"])
            ou = primary_origin.get("origin_uncertainty", {})
            if "maxHorizontalUncertainty" in ou and "minHorizontalUncertainty" in ou:
                scale = 1000.0 if is_v12 else 1.0
                a_km = float(ou["maxHorizontalUncertainty"]) / scale
                b_km = float(ou["minHorizontalUncertainty"]) / scale
                az = float(ou.get("azimuthMaxHorizontalUncertainty", 0.0))
                params.error_ellipse = (a_km, b_km, az)
            elif isinstance(ou.get("confidenceEllipsoid"), dict):
                ce = ou["confidenceEllipsoid"]
                scale = 1000.0 if is_v12 else 1.0
                a_val = ce.get("semiMajorAxisLength")
                b_val = ce.get("semiMinorAxisLength")
                if a_val is not None and b_val is not None and (float(a_val) > 0.0 or float(b_val) > 0.0):
                    a_km = float(a_val) / scale
                    b_km = float(b_val) / scale
                    if b_km > a_km:
                        a_km, b_km = b_km, a_km
                    az = float(ce.get("majorAxisAzimuth") or 0.0)
                    params.error_ellipse = (a_km, b_km, az)

        for mag in primary_ev.get("magnitudes", []):
            mtype = (mag.get("type") or "").strip().lower()
            val = mag.get("value")
            if val is None:
                continue
            if mtype == "ml":
                params.mag_ml = float(val)
            elif mtype == "mb":
                params.mag_mb = float(val)
            elif mtype == "ms":
                params.mag_ms = float(val)
            elif mtype == "mw":
                params.mag_mw = float(val)
            elif mtype == "mbb":
                params.mag_mbb = float(val)
            if mag.get("station_count"):
                params.mags_count = max(params.mags_count, int(mag["station_count"]))

        arr_by_pick = {}
        if primary_origin is not None:
            for arr in primary_origin.get("arrivals", []):
                arr_by_pick[arr["pick_id"]] = arr

        for pk in primary_ev.get("picks", []):
            arr = arr_by_pick.get(pk["public_id"], {})
            pol = pk.get("polarity", "").lower()
            sign = 1 if pol == "positive" else (-1 if pol == "negative" else 0)
            onset_code = "i" if pk.get("onset", "").lower() == "impulsive" else "e"
            chan = pk.get("channel", "BHZ") or "BHZ"
            phase_pk = PhasePick(
                name=pk.get("phase_hint") or arr.get("phase") or "P",
                onset=format_sh_time(pk["time_epoch"]) if pk.get("time_epoch") is not None else "",
                spec=onset_code,
                sign=sign,
                source="a" if pk.get("evaluation_mode", "").lower() == "automatic" else "m",
                resid=float(arr.get("time_residual", 0.0) or 0.0),
                weight=int(round(float(arr.get("time_weight", 2.0) or 2.0))),
                station=pk.get("station", ""),
                chan1=chan[0:1] if len(chan) >= 1 else "B",
                chan2=chan[1:2] if len(chan) >= 2 else "H",
                comp=chan[2:3] if len(chan) >= 3 else "Z",
            )
            phase_picks.append(phase_pk)

    return {
        "schema_version": "1.2" if is_v12 else "1.0",
        "events": events_out,
        "origin": primary_origin,
        "magnitudes": primary_ev.get("magnitudes", []),
        "picks": primary_ev.get("picks", []),
        "arrivals": primary_origin.get("arrivals", []) if primary_origin else [],
        "focal_mechanisms": primary_ev.get("focal_mechanisms", []),
        "params": params,
        "phase_picks": phase_picks,
    }
