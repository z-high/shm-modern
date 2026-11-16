"""SHM .evt event file reader, writer, ev2view summary formatter, and LocSAT adapter."""

from __future__ import annotations

import os
from typing import List, Tuple

from shm_app.core.models import AnalysisParamSet, PhasePick

PHASE_END_MARKER = "--- End of Phase ---"


def read_evt_file(filepath: str) -> Tuple[AnalysisParamSet, List[PhasePick]]:
    """Parse an SHM .evt event parameter file into (AnalysisParamSet, List[PhasePick])."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Event file not found: {filepath}")

    param = AnalysisParamSet()
    picks: List[PhasePick] = []
    cur_pick = PhasePick()
    cur_comment_lines: List[str] = []
    in_comment = False

    with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line.strip() and not in_comment:
                continue

            if line.strip() == PHASE_END_MARKER:
                if cur_pick.station or cur_pick.onset:
                    picks.append(cur_pick)
                cur_pick = PhasePick()
                continue

            if in_comment:
                if "~" in line:
                    before, _ = line.split("~", 1)
                    cur_comment_lines.append(before)
                    param.comment = "\n".join(cur_comment_lines).strip()
                    in_comment = False
                else:
                    cur_comment_lines.append(line)
                continue

            if ":" not in line:
                continue

            key_raw, val_raw = line.split(":", 1)
            key = key_raw.strip().lower()
            val = val_raw.strip()

            # Check multi-line comment
            if key == "comment":
                if "~" in val:
                    param.comment = val.split("~", 1)[0].strip()
                else:
                    cur_comment_lines = [val]
                    in_comment = True
                continue

            # Phase-level attributes
            if key == "station code":
                cur_pick.station = val.upper()
            elif key == "component":
                cur_pick.comp = (val.upper() or "Z")[0]
            elif key == "onset time":
                cur_pick.onset = val
            elif key == "onset type":
                cur_pick.spec = (val.lower() or "e")[0]
            elif key == "phase name":
                cur_pick.name = val
            elif key == "sign":
                cur_pick.sign = PhasePick.sign_from_char(val)
            elif key == "PickType" or key == "pick type" or key == "source":
                cur_pick.source = (val.lower() or "m")[0]
            elif key == "quality number":
                try:
                    cur_pick.quality = int(float(val))
                except ValueError:
                    pass
            elif key == "weight":
                try:
                    cur_pick.weight = int(float(val))
                except ValueError:
                    pass
            elif key == "applied filter":
                cur_pick.filter_name = val
            elif key == "amplitude":
                cur_pick.ampl = _fval(val)
            elif key == "period":
                cur_pick.period = _fval(val)
            elif key == "broadband amplitude":
                cur_pick.bb_ampl = _fval(val)
            elif key == "broadband period":
                cur_pick.bb_period = _fval(val)
            elif key == "magnitude ml":
                cur_pick.mag_ml = _fval(val)
            elif key == "magnitude mb":
                cur_pick.mag_mb = _fval(val)
            elif key == "magnitude ms":
                cur_pick.mag_ms = _fval(val)
            elif key == "magnitude mbb":
                cur_pick.mag_mbb = _fval(val)
                cur_pick.bb_mbb = cur_pick.mag_mbb
            elif key == "phase flags":
                cur_pick.flags = val
            elif key == "signal/noise":
                cur_pick.snr = _fval(val)
            elif key == "residual time":
                cur_pick.resid = _fval(val)
            elif key == "residual corr":
                cur_pick.resid_corr = _fval(val)
            elif key == "phase slowness":
                cur_pick.slowness = _fval(val)
            elif key == "phase beam azimuth":
                cur_pick.beam_azim = _fval(val)

            # Event-level attributes (stored on param and also sometimes repeated per phase)
            elif key == "event id":
                try:
                    param.event_id = int(val)
                except ValueError:
                    pass
            elif key == "origin time":
                param.origin = val
            elif key == "latitude":
                param.lat = _fval(val)
            elif key == "error in latitude (km)":
                param.d_lat = _fval(val)
            elif key == "longitude":
                param.lon = _fval(val)
            elif key == "error in longitude (km)":
                param.d_lon = _fval(val)
            elif key == "depth (km)":
                param.depth = _fval(val)
            elif key == "error in depth (km)":
                param.d_depth = _fval(val)
            elif key == "depth type":
                try:
                    param.depth_type = int(val)
                except ValueError:
                    pass
            elif key == "distance (deg)":
                param.distance = _fval(val)
            elif key == "error in distance (deg)":
                param.d_distance = _fval(val)
            elif key == "back-azimuth (deg)" or key == "azimuth (deg)":
                param.azimuth = _fval(val)
                if cur_pick.beam_azim == 0.0:
                    cur_pick.beam_azim = param.azimuth
            elif key == "error in back-azimuth (deg)":
                param.d_azimuth = _fval(val)
            elif key == "observed slowness (sec/deg)" or key == "slowness (sec/deg)":
                param.slowness = _fval(val)
                if cur_pick.slowness == 0.0:
                    cur_pick.slowness = param.slowness
            elif key == "error in slowness (sec/deg)":
                param.d_slowness = _fval(val)
            elif key == "mean magnitude ml":
                param.mag_ml = _fval(val)
            elif key == "mean magnitude ms":
                param.mag_ms = _fval(val)
            elif key == "mean magnitude mb":
                param.mag_mb = _fval(val)
            elif key == "mean magnitude mbb":
                param.mag_mbb = _fval(val)
            elif key == "moment m0":
                param.moment_m0 = _fval(val)
            elif key == "magnitude mw":
                param.mag_mw = _fval(val)
            elif key == "corner frequency":
                param.corner_freq = _fval(val)
            elif key == "source of information":
                param.source = val
            elif key == "source region":
                param.region_name = val
            elif key == "region id":
                try:
                    param.region_id = int(val)
                except ValueError:
                    pass
            elif key == "region table":
                try:
                    param.region_table = int(val)
                except ValueError:
                    pass
            elif key == "event type":
                param.event_type = val
            elif key == "event flags":
                param.flags = val
            elif key == "reference latitude":
                param.ref_lat = _fval(val)
            elif key == "reference longitude":
                param.ref_lon = _fval(val)

    return param, picks


def write_evt_file(filepath: str, param: AnalysisParamSet, picks: List[PhasePick]) -> None:
    """Write an SHM .evt event parameter file matching phaseinf.c."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="latin-1") as out:
        for idx, p in enumerate(picks):
            if param.event_id > 0:
                out.write(f"Event ID                   : {param.event_id:09d}\n")
            out.write(f"Station code               : {p.station}\n")
            out.write(f"Onset time                 : {p.onset}\n")
            out.write(f"Onset type                 : {p.spec or 'e'}\n")
            out.write(f"Phase name                 : {p.name}\n")
            if p.sign != 0:
                out.write(f"Sign                       : {p.sign_char}\n")
            out.write(f"Component                  : {p.comp or 'Z'}\n")
            out.write(f"Quality number             : {p.quality}\n")
            out.write(f"Weight                     : {p.weight}\n")
            out.write(f"Pick type                  : {p.source or 'm'}\n")
            if p.filter_name and p.filter_name != "NONE":
                out.write(f"Applied filter             : {p.filter_name}\n")
            if p.ampl > 0.0:
                out.write(f"Amplitude                  : {p.ampl:.2f}\n")
            if p.period > 0.0:
                out.write(f"Period                     : {p.period:.2f}\n")
            if p.bb_ampl > 0.0:
                out.write(f"Broadband amplitude        : {p.bb_ampl:.2f}\n")
            if p.bb_period > 0.0:
                out.write(f"Broadband period           : {p.bb_period:.2f}\n")
            if p.mag_ml != 0.0:
                out.write(f"Magnitude ml               : {p.mag_ml:.2f}\n")
            if p.mag_mb != 0.0:
                out.write(f"Magnitude mb               : {p.mag_mb:.2f}\n")
            if p.mag_ms != 0.0:
                out.write(f"Magnitude ms               : {p.mag_ms:.2f}\n")
            if p.mag_mbb != 0.0:
                out.write(f"Magnitude mbb              : {p.mag_mbb:.2f}\n")
            if p.snr > 0.0:
                out.write(f"Signal/Noise               : {p.snr:.1f}\n")
            if p.resid != 0.0:
                out.write(f"Residual time              : {p.resid:.2f}\n")
            if p.slowness != 0.0:
                out.write(f"Phase slowness             : {p.slowness:.2f}\n")
            if p.beam_azim != 0.0:
                out.write(f"Phase beam azimuth         : {p.beam_azim:.2f}\n")
            if p.flags:
                out.write(f"Phase flags                : {p.flags}\n")

            # Attach event summary block on the last phase (or first reference phase)
            if idx == len(picks) - 1:
                _write_event_summary_lines(out, param)

            out.write(f"{PHASE_END_MARKER}\n")

        if not picks:
            _write_event_summary_lines(out, param)
            out.write(f"{PHASE_END_MARKER}\n")


def _write_event_summary_lines(out, param: AnalysisParamSet) -> None:
    if param.origin:
        out.write(f"Origin time                : {param.origin}\n")
    if param.lat != 0.0 or param.lon != 0.0:
        out.write(f"Latitude                   : {param.lat:+8.3f}\n")
        out.write(f"Longitude                  : {param.lon:+8.3f}\n")
    if param.d_lat > 0.0:
        out.write(f"Error in latitude (km)     : {param.d_lat:.2f}\n")
    if param.d_lon > 0.0:
        out.write(f"Error in longitude (km)    : {param.d_lon:.2f}\n")
    out.write(f"Depth (km)                 : {param.depth:.1f}\n")
    if param.d_depth > 0.0:
        out.write(f"Error in depth (km)        : {param.d_depth:.1f}\n")
    out.write(f"Depth type                 : {param.depth_type}\n")
    if param.distance > 0.0:
        out.write(f"Distance (deg)             : {param.distance:.2f}\n")
    if param.azimuth != 0.0:
        out.write(f"Back-azimuth (deg)         : {param.azimuth:.2f}\n")
    if param.slowness > 0.0:
        out.write(f"Observed slowness (sec/deg): {param.slowness:.2f}\n")
    if param.mag_ml != 0.0:
        out.write(f"Mean magnitude ml          : {param.mag_ml:.2f}\n")
    if param.mag_mb != 0.0:
        out.write(f"Mean magnitude mb          : {param.mag_mb:.2f}\n")
    if param.mag_ms != 0.0:
        out.write(f"Mean magnitude ms          : {param.mag_ms:.2f}\n")
    if param.mag_mbb != 0.0:
        out.write(f"Mean magnitude mbb         : {param.mag_mbb:.2f}\n")
    if param.region_id > 0:
        out.write(f"Region ID                  : {param.region_id}\n")
    if param.region_name:
        out.write(f"Source region              : {param.region_name}\n")
    if param.source:
        out.write(f"Source of information      : {param.source}\n")
    if param.event_type:
        out.write(f"Event type                 : {param.event_type}\n")
    if param.comment:
        out.write(f"Comment                    : {param.comment}~\n")


def format_ev2view_summary(param: AnalysisParamSet, picks: List[PhasePick]) -> str:
    """Format an event summary report matching SHM's ev2view utility."""
    lines = [
        "=" * 74,
        "                 SEISMIC HANDLER EVENT SUMMARY (ev2view)",
        "=" * 74,
        f"  Event ID       : {param.event_id if param.event_id else 'N/A'}",
        f"  Origin Time    : {param.origin or 'N/A'}",
        f"  Epicenter      : {param.lat:+8.3f} deg N,  {param.lon:+8.3f} deg E"
        + (f"  (+/- {param.d_lat:.1f} / {param.d_lon:.1f} km)" if param.d_lat or param.d_lon else ""),
        f"  Focal Depth    : {param.depth:6.1f} km"
        + (f" (+/- {param.d_depth:.1f} km)" if param.d_depth else ""),
        f"  Distance / BAZ : {param.distance:6.2f} deg   /  {param.azimuth:6.2f} deg",
        f"  Slowness       : {param.slowness:6.2f} s/deg",
        f"  Source Region  : [{param.region_id}] {param.region_name or 'Unknown'}",
        f"  Magnitudes     : ML={param.mag_ml:.2f}  mb={param.mag_mb:.2f}  MS={param.mag_ms:.2f}  mB={param.mag_mbb:.2f}",
        f"  Source / Type  : {param.source} / {param.event_type}",
    ]
    if param.comment:
        lines.append(f"  Comment        : {param.comment}")
    lines.extend(
        [
            "-" * 74,
            "  Stat  Comp  Phase     Onset Time                Q/W   Ampl(nm)   Per(s)   Mag",
            "-" * 74,
        ]
    )
    for p in picks:
        mag_val = p.mag_mb or p.mag_ml or p.mag_ms or p.mag_mbb
        spec_sign = f"{p.spec or 'e'}{p.sign_char or ' '}"
        lines.append(
            f"  {p.station:<5s}  {p.comp:<3s}  {spec_sign} {p.name:<6s} "
            f"{p.onset:<24s}  {p.quality:2d}/{p.weight:1d}  "
            f"{p.ampl:9.1f}   {p.period:5.2f}  {mag_val:5.2f}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)


def _fval(val: str, default: float = 0.0) -> float:
    if not val:
        return default
    try:
        return float(val.split()[0])
    except ValueError:
        return default
