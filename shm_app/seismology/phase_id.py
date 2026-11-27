"""Automated seismic phase identification via IASP91 travel-time, slowness, and azimuth matching."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.seismology.traveltime import TravelTimeManager, get_ttt_manager

# ---------------------------------------------------------------------------
# M-18: Automated Phase Identification
# ---------------------------------------------------------------------------

DEFAULT_IDENTIFY_PHASES: Tuple[str, ...] = (
    "P",
    "Pdiff",
    "PKPdf",
    "PKPab",
    "PKPbc",
    "pP",
    "sP",
    "PP",
    "PPP",
    "PcP",
    "S",
    "Sdiff",
    "SKSdf",
    "sS",
    "SS",
    "ScS",
)


def identify_phase(
    t_obs_rel_sec: float,
    slowness_meas_s_deg: float,
    dist_deg: float,
    depth_km: float = 10.0,
    baz_meas_deg: Optional[float] = None,
    baz_theo_deg: Optional[float] = None,
    sigma_t: float = 2.0,
    sigma_s: float = 1.0,
    w_t: float = 1.0,
    w_s: float = 1.0,
    candidate_phases: Optional[Sequence[str]] = None,
    ttt: Optional[TravelTimeManager] = None,
    sigma_az: float = 10.0,
    w_az: float = 0.25,
) -> List[Dict[str, Any]]:
    """Identify seismic phases by matching observed relative travel time ``t_obs_rel_sec``
    and measured horizontal slowness ``slowness_meas_s_deg`` (plus optional backazimuth)
    against 1D velocity-model tables in ``TravelTimeManager``.

    Evaluates composite misfit:
        chi2_p = w_t * ((t_obs - T_p) / sigma_t)^2 + w_s * ((s_meas - p_p) / sigma_s)^2 [+ az term]
    and returns candidate dicts sorted by ascending ``chi2`` with normalized softmax probabilities.
    """
    tm = ttt or get_ttt_manager()
    phases = list(candidate_phases) if candidate_phases is not None else list(DEFAULT_IDENTIFY_PHASES)

    sig_t = max(1e-6, float(sigma_t))
    sig_s = max(1e-6, float(sigma_s))
    sig_az = max(1e-6, float(sigma_az))
    wt = max(0.0, float(w_t))
    ws = max(0.0, float(w_s))
    waz = max(0.0, float(w_az))

    use_baz = baz_meas_deg is not None and baz_theo_deg is not None and waz > 0.0
    if use_baz:
        baz_resid = ((float(baz_meas_deg) - float(baz_theo_deg) + 180.0) % 360.0) - 180.0
        chi2_baz = waz * ((baz_resid / sig_az) ** 2)
    else:
        baz_resid = 0.0
        chi2_baz = 0.0

    candidates: List[Dict[str, Any]] = []
    for ph in phases:
        t_pred = tm.travel_time(ph, float(dist_deg), float(depth_km))
        if t_pred is None or t_pred <= 0.0:
            continue

        s_pred = tm.slowness(ph, float(dist_deg), float(depth_km))
        if ws > 0.0 and s_pred is None:
            continue

        t_resid = float(t_obs_rel_sec) - float(t_pred)
        s_resid = (float(slowness_meas_s_deg) - float(s_pred)) if s_pred is not None else 0.0

        chi2 = wt * ((t_resid / sig_t) ** 2)
        if ws > 0.0 and s_pred is not None:
            chi2 += ws * ((s_resid / sig_s) ** 2)
        chi2 += chi2_baz

        entry: Dict[str, Any] = {
            "phase": ph,
            "chi2": float(chi2),
            "t_pred": float(t_pred),
            "s_pred": float(s_pred) if s_pred is not None else None,
            "t_resid": float(t_resid),
            "s_resid": float(s_resid),
            "probability": 0.0,
        }
        if use_baz:
            entry["baz_resid"] = float(baz_resid)
        candidates.append(entry)

    if not candidates:
        return []

    candidates.sort(key=lambda item: item["chi2"])
    min_chi2 = candidates[0]["chi2"]
    raw_likes = [math.exp(-0.5 * min(500.0, c["chi2"] - min_chi2)) for c in candidates]
    like_sum = sum(raw_likes)
    if like_sum > 0.0:
        for c, lk in zip(candidates, raw_likes):
            c["probability"] = float(lk / like_sum)
    else:
        candidates[0]["probability"] = 1.0

    return candidates


def identify_trace_picks(
    traces: Sequence[Trace],
    origin_epoch: float,
    dist_deg: float,
    depth_km: float = 10.0,
    baz_theo_deg: Optional[float] = None,
    default_slowness: Optional[float] = None,
    default_baz: Optional[float] = None,
    update_pick_names: bool = False,
    max_chi2: Optional[float] = None,
    sigma_t: float = 2.0,
    sigma_s: float = 1.0,
    w_t: float = 1.0,
    w_s: float = 1.0,
    candidate_phases: Optional[Sequence[str]] = None,
    ttt: Optional[TravelTimeManager] = None,
    config: Optional[SHMConfig] = None,
) -> List[Dict[str, Any]]:
    """Identify phase picks across a list of traces relative to a known event origin.

    Uses calibrated slowness/azimuth (``pk.l_slowness`` / ``pk.l_azim``) when present,
    falling back to measured pick slowness/azimuth, trace slowness/azimuth, or the
    supplied ``default_slowness`` / ``default_baz``. When no slowness is available,
    falls back to travel-time-only matching (``w_s=0.0``).
    """
    cfg = config or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)
    results: List[Dict[str, Any]] = []

    for tr in traces:
        for pk in tr.picks:
            if not pk.onset:
                continue
            t_obs_rel = float(pk.onset_epoch) - float(origin_epoch)

            if pk.l_slowness > 0.0:
                slo_val: Optional[float] = float(pk.l_slowness)
            elif pk.slowness > 0.0:
                slo_val = float(pk.slowness)
            elif tr.slowness > 0.0:
                slo_val = float(tr.slowness)
            else:
                slo_val = default_slowness

            if pk.l_azim > 0.0:
                baz_val: Optional[float] = float(pk.l_azim)
            elif pk.beam_azim > 0.0:
                baz_val = float(pk.beam_azim)
            elif tr.azimuth > 0.0:
                baz_val = float(tr.azimuth)
            else:
                baz_val = default_baz

            eff_ws = w_s if (slo_val is not None and slo_val > 0.0) else 0.0
            slo_arg = float(slo_val) if slo_val is not None else 0.0

            cands = identify_phase(
                t_obs_rel_sec=t_obs_rel,
                slowness_meas_s_deg=slo_arg,
                dist_deg=dist_deg,
                depth_km=depth_km,
                baz_meas_deg=baz_val,
                baz_theo_deg=baz_theo_deg,
                sigma_t=sigma_t,
                sigma_s=sigma_s,
                w_t=w_t,
                w_s=eff_ws,
                candidate_phases=candidate_phases,
                ttt=tm,
            )
            if not cands:
                continue

            best = cands[0]
            old_name = pk.name
            accepted = (max_chi2 is None) or (best["chi2"] <= float(max_chi2))
            if update_pick_names and accepted:
                pk.name = best["phase"]
                pk.resid = best["t_resid"]

            results.append(
                {
                    "station": tr.station,
                    "comp": tr.comp,
                    "old_phase": old_name,
                    "identified_phase": best["phase"],
                    "chi2": best["chi2"],
                    "probability": best["probability"],
                    "t_resid": best["t_resid"],
                    "s_resid": best["s_resid"],
                    "updated": bool(update_pick_names and accepted),
                    "candidates": cands,
                    "pick": pk,
                }
            )
    return results

