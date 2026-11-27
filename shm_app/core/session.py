"""Workspace session autosave, atomic persistence, and crash recovery (M-23).

Pure Python + NumPy + stdlib implementation (GUI-independent).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from shm_app.core.models import AnalysisParamSet, PhasePick, Trace

logger = logging.getLogger(__name__)

DEFAULT_SESSION_FILENAME = ".shm_session.json"
SESSION_SCHEMA_VERSION = 1


class SessionLoadError(ValueError):
    """Raised when a session file cannot be parsed or fails schema checks."""


def _json_safe(obj: Any) -> Any:
    """Recursively convert Python/NumPy objects into strict RFC-8259 JSON-safe values."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return int(val) if isinstance(val, bool) else default
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default


def _safe_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    return bool(val)


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val)


def resolve_session_path(path: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve a target file or directory path to a concrete `.shm_session.json` file."""
    if path is None:
        env_path = os.environ.get("SHM_SESSION_PATH", "").strip()
        if not env_path:
            return Path.cwd() / DEFAULT_SESSION_FILENAME
        raw_str = env_path
    else:
        raw_str = os.fspath(path)
    p = Path(raw_str)
    if p.is_dir() or (isinstance(raw_str, str) and raw_str.endswith((os.sep, "/"))):
        return p / DEFAULT_SESSION_FILENAME
    return p


def serialize_pick(pick: PhasePick) -> Dict[str, Any]:
    """Serialize all fields of a PhasePick to a JSON-compatible dict."""
    return _json_safe(
        {
            "name": pick.name,
            "onset": pick.onset,
            "onset_acc_l": pick.onset_acc_l,
            "onset_acc_r": pick.onset_acc_r,
            "spec": pick.spec,
            "sign": pick.sign,
            "source": pick.source,
            "weight": pick.weight,
            "quality": pick.quality,
            "filter_name": pick.filter_name,
            "ampl": pick.ampl,
            "ampl_time": pick.ampl_time,
            "ampl_veloc": pick.ampl_veloc,
            "period": pick.period,
            "ap_source": pick.ap_source,
            "bb_ampl": pick.bb_ampl,
            "bb_period": pick.bb_period,
            "bb_mbb": pick.bb_mbb,
            "slowness": pick.slowness,
            "d_slowness": pick.d_slowness,
            "beam_azim": pick.beam_azim,
            "d_beam_azim": pick.d_beam_azim,
            "l_slowness": pick.l_slowness,
            "l_azim": pick.l_azim,
            "mag_ml": pick.mag_ml,
            "mag_mb": pick.mag_mb,
            "mag_ms": pick.mag_ms,
            "mag_mbb": pick.mag_mbb,
            "snr": pick.snr,
            "resid": pick.resid,
            "resid_corr": pick.resid_corr,
            "comment": pick.comment,
            "station": pick.station,
            "comp": pick.comp,
            "chan1": pick.chan1,
            "chan2": pick.chan2,
            "flags": pick.flags,
        }
    )


def deserialize_pick(raw: Dict[str, Any]) -> PhasePick:
    """Reconstruct a PhasePick from a serialized dictionary."""
    if not isinstance(raw, dict):
        raise SessionLoadError("Pick entry must be a JSON object")
    return PhasePick(
        name=_safe_str(raw.get("name"), "P"),
        onset=_safe_str(raw.get("onset"), ""),
        onset_acc_l=_safe_float(raw.get("onset_acc_l"), 0.0),
        onset_acc_r=_safe_float(raw.get("onset_acc_r"), 0.0),
        spec=_safe_str(raw.get("spec"), "e"),
        sign=_safe_int(raw.get("sign"), 0),
        source=_safe_str(raw.get("source"), "m"),
        weight=_safe_int(raw.get("weight"), 2),
        quality=_safe_int(raw.get("quality"), 4),
        filter_name=_safe_str(raw.get("filter_name"), "NONE"),
        ampl=_safe_float(raw.get("ampl"), 0.0),
        ampl_time=_safe_float(raw.get("ampl_time"), 0.0),
        ampl_veloc=_safe_float(raw.get("ampl_veloc"), 0.0),
        period=_safe_float(raw.get("period"), 0.0),
        ap_source=_safe_str(raw.get("ap_source"), ""),
        bb_ampl=_safe_float(raw.get("bb_ampl"), 0.0),
        bb_period=_safe_float(raw.get("bb_period"), 0.0),
        bb_mbb=_safe_float(raw.get("bb_mbb"), 0.0),
        slowness=_safe_float(raw.get("slowness"), 0.0),
        d_slowness=_safe_float(raw.get("d_slowness"), 0.0),
        beam_azim=_safe_float(raw.get("beam_azim"), 0.0),
        d_beam_azim=_safe_float(raw.get("d_beam_azim"), 0.0),
        l_slowness=_safe_float(raw.get("l_slowness"), 0.0),
        l_azim=_safe_float(raw.get("l_azim"), 0.0),
        mag_ml=_safe_float(raw.get("mag_ml"), 0.0),
        mag_mb=_safe_float(raw.get("mag_mb"), 0.0),
        mag_ms=_safe_float(raw.get("mag_ms"), 0.0),
        mag_mbb=_safe_float(raw.get("mag_mbb"), 0.0),
        snr=_safe_float(raw.get("snr"), 0.0),
        resid=_safe_float(raw.get("resid"), 0.0),
        resid_corr=_safe_float(raw.get("resid_corr"), 0.0),
        comment=_safe_str(raw.get("comment"), ""),
        station=_safe_str(raw.get("station"), ""),
        comp=_safe_str(raw.get("comp"), "Z"),
        chan1=_safe_str(raw.get("chan1"), "B"),
        chan2=_safe_str(raw.get("chan2"), "H"),
        flags=_safe_str(raw.get("flags"), ""),
    )


def serialize_param_set(params: AnalysisParamSet) -> Dict[str, Any]:
    """Serialize all fields of an AnalysisParamSet to a JSON-compatible dict."""
    return _json_safe(
        {
            "distance": params.distance,
            "d_distance": params.d_distance,
            "azimuth": params.azimuth,
            "d_azimuth": params.d_azimuth,
            "slowness": params.slowness,
            "d_slowness": params.d_slowness,
            "depth": params.depth,
            "d_depth": params.d_depth,
            "depth_type": params.depth_type,
            "origin": params.origin,
            "lat": params.lat,
            "d_lat": params.d_lat,
            "lon": params.lon,
            "d_lon": params.d_lon,
            "phase": params.phase,
            "mag_ml": params.mag_ml,
            "d_ml": params.d_ml,
            "mag_ms": params.mag_ms,
            "d_ms": params.d_ms,
            "mag_mb": params.mag_mb,
            "d_mb": params.d_mb,
            "mag_mbb": params.mag_mbb,
            "d_mbb": params.d_mbb,
            "mags_count": params.mags_count,
            "source": params.source,
            "quality": params.quality,
            "region_table": params.region_table,
            "region_id": params.region_id,
            "region_name": params.region_name,
            "sregion_id": params.sregion_id,
            "sregion_name": params.sregion_name,
            "event_id": params.event_id,
            "event_type": params.event_type,
            "flags": params.flags,
            "comment": params.comment,
            "ref_lat": params.ref_lat,
            "ref_lon": params.ref_lon,
            "corner_freq": params.corner_freq,
            "moment_m0": params.moment_m0,
            "mag_mw": params.mag_mw,
            "incidence": params.incidence,
            "residuals": dict(params.residuals) if params.residuals is not None else None,
            "error_ellipse": list(params.error_ellipse) if params.error_ellipse is not None else None,
            "condition_number": params.condition_number,
            "chi2_reduced": params.chi2_reduced,
            "rmse": params.rmse,
            "wrmse": params.wrmse,
        }
    )


def deserialize_param_set(raw: Dict[str, Any]) -> AnalysisParamSet:
    """Reconstruct an AnalysisParamSet from a serialized dictionary."""
    if not isinstance(raw, dict):
        raise SessionLoadError("Parameter set entry must be a JSON object")

    raw_res = raw.get("residuals")
    residuals: Optional[Dict[str, float]] = None
    if isinstance(raw_res, dict):
        residuals = {str(k): _safe_float(v, 0.0) for k, v in raw_res.items()}

    raw_ell = raw.get("error_ellipse")
    error_ellipse: Optional[tuple[float, float, float]] = None
    if isinstance(raw_ell, (list, tuple)) and len(raw_ell) == 3:
        error_ellipse = (
            _safe_float(raw_ell[0], 0.0),
            _safe_float(raw_ell[1], 0.0),
            _safe_float(raw_ell[2], 0.0),
        )

    return AnalysisParamSet(
        distance=_safe_float(raw.get("distance"), 0.0),
        d_distance=_safe_float(raw.get("d_distance"), 0.0),
        azimuth=_safe_float(raw.get("azimuth"), 0.0),
        d_azimuth=_safe_float(raw.get("d_azimuth"), 0.0),
        slowness=_safe_float(raw.get("slowness"), 0.0),
        d_slowness=_safe_float(raw.get("d_slowness"), 0.0),
        depth=_safe_float(raw.get("depth"), 33.0),
        d_depth=_safe_float(raw.get("d_depth"), 0.0),
        depth_type=_safe_int(raw.get("depth_type"), 1),
        origin=_safe_str(raw.get("origin"), ""),
        lat=_safe_float(raw.get("lat"), 0.0),
        d_lat=_safe_float(raw.get("d_lat"), 0.0),
        lon=_safe_float(raw.get("lon"), 0.0),
        d_lon=_safe_float(raw.get("d_lon"), 0.0),
        phase=_safe_str(raw.get("phase"), "P"),
        mag_ml=_safe_float(raw.get("mag_ml"), 0.0),
        d_ml=_safe_float(raw.get("d_ml"), 0.0),
        mag_ms=_safe_float(raw.get("mag_ms"), 0.0),
        d_ms=_safe_float(raw.get("d_ms"), 0.0),
        mag_mb=_safe_float(raw.get("mag_mb"), 0.0),
        d_mb=_safe_float(raw.get("d_mb"), 0.0),
        mag_mbb=_safe_float(raw.get("mag_mbb"), 0.0),
        d_mbb=_safe_float(raw.get("d_mbb"), 0.0),
        mags_count=_safe_int(raw.get("mags_count"), 0),
        source=_safe_str(raw.get("source"), "SHM"),
        quality=_safe_int(raw.get("quality"), 1),
        region_table=_safe_int(raw.get("region_table"), 1),
        region_id=_safe_int(raw.get("region_id"), 0),
        region_name=_safe_str(raw.get("region_name"), ""),
        sregion_id=_safe_int(raw.get("sregion_id"), 0),
        sregion_name=_safe_str(raw.get("sregion_name"), ""),
        event_id=_safe_int(raw.get("event_id"), 0),
        event_type=_safe_str(raw.get("event_type"), "t"),
        flags=_safe_str(raw.get("flags"), ""),
        comment=_safe_str(raw.get("comment"), ""),
        ref_lat=_safe_float(raw.get("ref_lat"), 49.6919),
        ref_lon=_safe_float(raw.get("ref_lon"), 11.2217),
        corner_freq=_safe_float(raw.get("corner_freq"), 0.0),
        moment_m0=_safe_float(raw.get("moment_m0"), 0.0),
        mag_mw=_safe_float(raw.get("mag_mw"), 0.0),
        incidence=_safe_float(raw.get("incidence"), 0.0),
        residuals=residuals,
        error_ellipse=error_ellipse,
        condition_number=_safe_float(raw.get("condition_number"), 1.0),
        chi2_reduced=_safe_float(raw.get("chi2_reduced"), 0.0),
        rmse=_safe_float(raw.get("rmse"), 0.0),
        wrmse=_safe_float(raw.get("wrmse"), 0.0),
    )


def serialize_trace(
    tr: Trace,
    *,
    include_inline_waveforms: bool = True,
    force_inline_waveforms: bool = False,
    store_inline: Optional[bool] = None,
) -> Dict[str, Any]:
    """Serialize Trace metadata, picks, filter state, and optional inline waveform fallback."""
    src_path = _safe_str(tr.source_path, "")
    has_disk_source = bool(src_path and os.path.isfile(src_path))
    is_modified = bool(
        (tr.filter_name and tr.filter_name != "NONE")
        or tr.extra_headers.get("waveform_modified")
    )
    if store_inline is None:
        store_inline_flag = bool(
            force_inline_waveforms
            or (include_inline_waveforms and (not has_disk_source or is_modified))
        )
    else:
        store_inline_flag = bool(store_inline)

    inline_data: Optional[List[Optional[float]]] = None
    if store_inline_flag:
        arr = np.asarray(tr.data, dtype=np.float64)
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            logger.warning(
                "Trace %s.%s contains non-finite samples during session serialization; sanitizing to 0.0",
                tr.station,
                tr.channel3,
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        inline_data = arr.tolist()

    return _json_safe(
        {
            "source_path": src_path,
            "length": int(tr.length),
            "has_inline_data": store_inline_flag,
            "data": inline_data,
            "delta": tr.delta,
            "station": tr.station,
            "comp": tr.comp,
            "chan1": tr.chan1,
            "chan2": tr.chan2,
            "auxid": tr.auxid,
            "network": tr.network,
            "locid": tr.locid,
            "start_epoch": tr.start_epoch,
            "display_offset": tr.display_offset,
            "distance": tr.distance,
            "azimuth": tr.azimuth,
            "inci": tr.inci,
            "depth": tr.depth,
            "magnitude": tr.magnitude,
            "lat": tr.lat,
            "lon": tr.lon,
            "slowness": tr.slowness,
            "torig": tr.torig,
            "weight": tr.weight,
            "norm": tr.norm,
            "calib": tr.calib,
            "calper": tr.calper,
            "instype": tr.instype,
            "comment": tr.comment,
            "opinfo": tr.opinfo,
            "filter_name": tr.filter_name,
            "overlay_filter_name": tr.overlay_filter_name,
            "overlay_filter_color": tr.overlay_filter_color,
            "filter_history": list(tr.filter_history),
            "hidden": tr.hidden,
            "selected": tr.selected,
            "picks": [serialize_pick(pk) for pk in tr.picks],
            "theo_picks": {str(k): _safe_float(v, 0.0) for k, v in tr.theo_picks.items()},
            "extra_headers": dict(tr.extra_headers),
        }
    )


def deserialize_trace(
    raw: Dict[str, Any],
    *,
    trace_loader: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = None,
) -> Trace:
    """Reconstruct a Trace from serialized metadata and inline or external waveform data."""
    if not isinstance(raw, dict):
        raise SessionLoadError("Trace entry must be a JSON object")

    raw_data = raw.get("data")
    if isinstance(raw_data, list):
        arr = np.asarray([_safe_float(v, 0.0) for v in raw_data], dtype=np.float64)
    elif trace_loader is not None:
        loaded = trace_loader(raw)
        arr = np.asarray(loaded, dtype=np.float64) if loaded is not None else np.zeros(0, dtype=np.float64)
    else:
        arr = np.zeros(0, dtype=np.float64)

    picks_raw = raw.get("picks")
    picks: List[PhasePick] = []
    if isinstance(picks_raw, list):
        for p_item in picks_raw:
            if isinstance(p_item, dict):
                picks.append(deserialize_pick(p_item))

    theo_raw = raw.get("theo_picks")
    theo_picks: Dict[str, float] = {}
    if isinstance(theo_raw, dict):
        theo_picks = {str(k): _safe_float(v, 0.0) for k, v in theo_raw.items()}

    extra_raw = raw.get("extra_headers")
    extra_headers: Dict[str, Any] = dict(extra_raw) if isinstance(extra_raw, dict) else {}

    hist_raw = raw.get("filter_history")
    filter_history = [str(item) for item in hist_raw] if isinstance(hist_raw, list) else []

    delta = _safe_float(raw.get("delta"), 0.05)
    if delta <= 0.0:
        logger.warning(
            "Invalid non-positive delta=%r for trace %r; falling back to 0.05s",
            raw.get("delta"),
            raw.get("station"),
        )
        delta = 0.05

    return Trace(
        data=arr,
        delta=delta,
        station=_safe_str(raw.get("station"), "STAT"),
        comp=_safe_str(raw.get("comp"), "Z"),
        chan1=_safe_str(raw.get("chan1"), "B"),
        chan2=_safe_str(raw.get("chan2"), "H"),
        auxid=_safe_str(raw.get("auxid"), ""),
        network=_safe_str(raw.get("network"), ""),
        locid=_safe_str(raw.get("locid"), ""),
        start_epoch=_safe_float(raw.get("start_epoch"), 0.0),
        display_offset=_safe_float(raw.get("display_offset"), 0.0),
        distance=_safe_float(raw.get("distance"), 0.0),
        azimuth=_safe_float(raw.get("azimuth"), 0.0),
        inci=_safe_float(raw.get("inci"), 0.0),
        depth=_safe_float(raw.get("depth"), 0.0),
        magnitude=_safe_float(raw.get("magnitude"), 0.0),
        lat=_safe_float(raw.get("lat"), 0.0),
        lon=_safe_float(raw.get("lon"), 0.0),
        slowness=_safe_float(raw.get("slowness"), 0.0),
        torig=_safe_float(raw.get("torig"), 0.0),
        weight=_safe_float(raw.get("weight"), 1.0),
        norm=_safe_float(raw.get("norm"), 1.0),
        calib=_safe_float(raw.get("calib"), 1.0),
        calper=_safe_float(raw.get("calper"), 1.0),
        instype=_safe_str(raw.get("instype"), ""),
        comment=_safe_str(raw.get("comment"), ""),
        opinfo=_safe_str(raw.get("opinfo"), ""),
        filter_name=_safe_str(raw.get("filter_name"), "NONE"),
        overlay_filter_name=_safe_str(raw.get("overlay_filter_name"), "NONE"),
        overlay_filter_color=_safe_str(raw.get("overlay_filter_color"), ""),
        filter_history=filter_history,
        hidden=_safe_bool(raw.get("hidden"), False),
        selected=_safe_bool(raw.get("selected"), False),
        picks=picks,
        theo_picks=theo_picks,
        extra_headers=extra_headers,
        source_path=_safe_str(raw.get("source_path"), ""),
    )


def serialize_workspace(
    traces: Sequence[Trace],
    param_sets: Optional[Sequence[AnalysisParamSet]] = None,
    active_set_index: int = 0,
    *,
    params: Optional[AnalysisParamSet] = None,
    selected_trace_indices: Optional[Sequence[int]] = None,
    active_trace_index: int = 0,
    display_settings: Optional[Dict[str, Any]] = None,
    clean_exit: bool = False,
    include_inline_waveforms: bool = True,
    force_inline_waveforms: bool = False,
    saved_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Serialize full workspace state into a JSON-ready dictionary."""
    if param_sets is None:
        param_list = [params] if params is not None else []
    else:
        param_list = list(param_sets)

    if selected_trace_indices is None:
        sel_indices = [idx for idx, tr in enumerate(traces) if tr.selected]
    else:
        sel_indices = [int(i) for i in selected_trace_indices]

    disp: Dict[str, Any] = dict(display_settings) if display_settings else {}
    if "global_filter_name" not in disp:
        active_filters = [
            tr.filter_name for tr in traces if tr.filter_name and tr.filter_name != "NONE"
        ]
        disp["global_filter_name"] = active_filters[0] if active_filters else "NONE"
    if "overlay_filters" not in disp:
        disp["overlay_filters"] = {
            str(idx): {
                "name": tr.overlay_filter_name,
                "color": tr.overlay_filter_color,
                "history": list(tr.filter_history),
            }
            for idx, tr in enumerate(traces)
            if (tr.overlay_filter_name and tr.overlay_filter_name != "NONE") or tr.filter_history
        }

    return _json_safe(
        {
            "version": SESSION_SCHEMA_VERSION,
            "saved_at": float(saved_at if saved_at is not None else time.time()),
            "clean_exit": bool(clean_exit),
            "active_trace_index": int(active_trace_index),
            "selected_trace_indices": sel_indices,
            "active_set_index": int(active_set_index),
            "traces": [
                serialize_trace(
                    tr,
                    include_inline_waveforms=include_inline_waveforms,
                    force_inline_waveforms=force_inline_waveforms,
                )
                for tr in traces
            ],
            "param_sets": [serialize_param_set(ps) for ps in param_list],
            "display_settings": disp,
        }
    )


def deserialize_workspace(
    payload: Dict[str, Any],
    *,
    trace_loader: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Deserialize a session dictionary back into live models and display settings."""
    if not isinstance(payload, dict):
        raise SessionLoadError("Session payload must be a JSON object")

    if not any(k in payload for k in ("traces", "param_sets", "version", "display_settings")):
        raise SessionLoadError("Invalid session payload: missing expected workspace keys")

    traces_raw = payload.get("traces", [])
    if not isinstance(traces_raw, list):
        raise SessionLoadError("Invalid session payload: 'traces' must be a list")

    traces: List[Trace] = []
    for item in traces_raw:
        traces.append(deserialize_trace(item, trace_loader=trace_loader))

    sel_raw = payload.get("selected_trace_indices", [])
    selected_indices: List[int] = []
    if isinstance(sel_raw, list):
        for idx in sel_raw:
            i = _safe_int(idx, -1)
            if 0 <= i < len(traces):
                selected_indices.append(i)
                traces[i].selected = True
    if not selected_indices:
        selected_indices = [idx for idx, tr in enumerate(traces) if tr.selected]

    ps_raw = payload.get("param_sets", [])
    if not isinstance(ps_raw, list):
        raise SessionLoadError("Invalid session payload: 'param_sets' must be a list")
    param_sets = [deserialize_param_set(item) for item in ps_raw]

    active_set_index = _safe_int(payload.get("active_set_index"), 0)
    if param_sets:
        active_set_index = max(0, min(len(param_sets) - 1, active_set_index))
        active_params = param_sets[active_set_index]
    else:
        active_set_index = 0
        active_params = AnalysisParamSet()

    active_trace_index = _safe_int(payload.get("active_trace_index"), 0)
    if traces:
        active_trace_index = max(0, min(len(traces) - 1, active_trace_index))
    else:
        active_trace_index = 0

    disp_raw = payload.get("display_settings", {})
    display_settings = dict(disp_raw) if isinstance(disp_raw, dict) else {}

    return {
        "version": _safe_int(payload.get("version"), SESSION_SCHEMA_VERSION),
        "saved_at": _safe_float(payload.get("saved_at"), 0.0),
        "clean_exit": _safe_bool(payload.get("clean_exit"), False),
        "traces": traces,
        "param_sets": param_sets,
        "params": active_params,
        "active_set_index": active_set_index,
        "active_trace_index": active_trace_index,
        "selected_trace_indices": selected_indices,
        "display_settings": display_settings,
    }


def atomic_write_json(path: Union[str, os.PathLike], payload: Dict[str, Any]) -> str:
    """Atomically write JSON payload to path via temp file in target directory + fsync + os.replace."""
    target = resolve_session_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(safe_payload, tmp_file, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, str(target))
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return str(target)


def has_unclean_session(path: Union[str, os.PathLike]) -> bool:
    """Return True if path points to a valid session file whose clean_exit flag is False."""
    target = resolve_session_path(path)
    if not target.is_file():
        return False
    try:
        raw_text = target.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    if bool(data.get("clean_exit", False)):
        return False
    try:
        deserialize_workspace(data)
    except SessionLoadError:
        return False
    return True


class SessionManager:
    """Manages atomic workspace autosaves, clean shutdown markers, and crash recovery."""

    def __init__(self, session_path: Optional[Union[str, os.PathLike]] = None) -> None:
        self.session_path: Path = resolve_session_path(session_path)

    def save(
        self,
        traces: Sequence[Trace],
        param_sets: Optional[Sequence[AnalysisParamSet]] = None,
        active_set_index: int = 0,
        *,
        params: Optional[AnalysisParamSet] = None,
        selected_trace_indices: Optional[Sequence[int]] = None,
        active_trace_index: int = 0,
        display_settings: Optional[Dict[str, Any]] = None,
        clean_exit: bool = False,
        include_inline_waveforms: bool = True,
        force_inline_waveforms: bool = False,
    ) -> str:
        """Serialize workspace state and write atomically to `self.session_path`."""
        payload = serialize_workspace(
            traces=traces,
            param_sets=param_sets,
            active_set_index=active_set_index,
            params=params,
            selected_trace_indices=selected_trace_indices,
            active_trace_index=active_trace_index,
            display_settings=display_settings,
            clean_exit=clean_exit,
            include_inline_waveforms=include_inline_waveforms,
            force_inline_waveforms=force_inline_waveforms,
        )
        return self.save_payload(payload)

    autosave = save

    def save_payload(self, payload: Dict[str, Any]) -> str:
        """Atomically persist a pre-serialized workspace dictionary."""
        return atomic_write_json(self.session_path, payload)

    def mark_clean_shutdown(
        self,
        traces: Optional[Sequence[Trace]] = None,
        param_sets: Optional[Sequence[AnalysisParamSet]] = None,
        active_set_index: int = 0,
        **kwargs: Any,
    ) -> bool:
        """Mark the current session as cleanly exited via atomic update."""
        if traces is not None:
            self.save(
                traces=traces,
                param_sets=param_sets,
                active_set_index=active_set_index,
                clean_exit=True,
                **kwargs,
            )
            return True

        if not self.session_path.is_file():
            return False
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        data["clean_exit"] = True
        data["saved_at"] = time.time()
        self.save_payload(data)
        return True

    def has_unclean_session(self) -> bool:
        """Return True if an unclosed session file exists at `self.session_path`."""
        return has_unclean_session(self.session_path)

    def load(
        self,
        *,
        trace_loader: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """Load and deserialize session state with safe fallback on missing or corrupt JSON."""
        fallback = {
            "ok": False,
            "error": None,
            "version": SESSION_SCHEMA_VERSION,
            "saved_at": 0.0,
            "clean_exit": True,
            "traces": [],
            "param_sets": [],
            "params": AnalysisParamSet(),
            "active_set_index": 0,
            "active_trace_index": 0,
            "selected_trace_indices": [],
            "display_settings": {},
        }

        if not self.session_path.is_file():
            if raise_on_error:
                raise FileNotFoundError(f"Session file not found: {self.session_path}")
            fallback["error"] = f"Session file not found: {self.session_path}"
            return fallback

        try:
            raw_text = self.session_path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
            restored = deserialize_workspace(payload, trace_loader=trace_loader)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, SessionLoadError, TypeError, ValueError) as exc:
            if raise_on_error:
                raise SessionLoadError(f"Failed to load session {self.session_path}: {exc}") from exc
            fallback["error"] = f"Corrupt or unreadable session file: {exc}"
            return fallback

        restored["ok"] = True
        restored["error"] = None
        return restored

    def clear(self) -> bool:
        """Remove the session file if present."""
        if self.session_path.is_file():
            try:
                self.session_path.unlink()
                return True
            except OSError:
                return False
        return False
