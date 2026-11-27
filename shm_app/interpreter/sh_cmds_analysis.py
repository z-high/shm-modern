"""Array processing, location, picking, magnitude, spectral, and FIT SH commands."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional
import numpy as np

from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.processing.polarization import analyze_polarization_diagnostic
from shm_app.processing.spectrum import (
    build_brune_fit_diagnostic,
    compute_fft_spectrum,
    compute_mem_spectrum,
    compute_welch_power_spectrum,
    find_surface_wave_amplitude,
    invert_brune_spectrum,
    scan_surface_waves,
)
from shm_app.seismology.array_loc import (
    SlownessCalibrationTable,
    align_traces_on_phase,
    compute_beam,
    compute_fk_analysis,
    compute_trace_window_magnitudes,
    compute_vespagram,
    correlate_traces,
    corrpick_array,
    fit_depth_phase,
    identify_phase,
    identify_trace_picks,
    load_calibration_for_array,
    locate_by_slowness_azimuth,
    locate_by_sp_diff,
    locate_network_geiger,
    pick_onset_baer_kradolfer,
    plane_wave_locate,
)
from shm_app.seismology.focal_mechanism import (
    extract_polarities_from_traces,
    solve_focal_mechanism,
)
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


class _AnalysisCommandsMixin:
    """Array beamforming/FK/vespa, location, phase-id, picking, magnitudes, spectrum, and FIT."""

    def locate_by_sp_diff(self) -> Optional[Dict[str, Any]]:
        res = locate_by_sp_diff(
            self.visible_traces(),
            depth_km=self.current_params.depth,
            ttt=self.ttt,
            config=self.config,
        )
        if res is not None:
            p = self.current_params
            p.distance = res["distance"]
            p.origin = res["origin"]
        return res

    def fit_depth_phase(self) -> Optional[Dict[str, Any]]:
        p = self.current_params
        dist = p.distance if p.distance > 0 else 65.0
        res = fit_depth_phase(
            self.visible_traces(),
            distance_deg=dist,
            ttt=self.ttt,
            config=self.config,
        )
        if res is not None:
            p.depth = res["depth"]
            p.d_depth = res["d_depth"]
            p.depth_type = 3
        return res

    @staticmethod
    def _apply_magnitudes_to_pick(pk: PhasePick, res: Dict[str, Any]) -> None:
        pk.ampl = res["ampl_nm"]
        pk.period = res["period"]
        pk.mag_ml = res["ml"]
        pk.mag_mb = res["mb"]
        pk.mag_ms = res["ms"]
        pk.mag_mbb = res["mbb"]

    def measure_amplitude_magnitudes(
        self,
        trace_idx: int,
        rel_lo: float,
        rel_hi: float,
        prefer_phase: str = "P",
    ) -> Optional[Dict[str, Any]]:
        if not (0 <= trace_idx < len(self.traces)):
            return None
        tr = self.traces[trace_idx]
        p = self.current_params
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        dist = p.distance if p.distance > 0 else 50.0
        if has_hypo:
            st = self.config.get_station(tr.station)
            if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                dist, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                tr.distance = dist
        elif tr.distance > 0.0:
            dist = tr.distance
        res = compute_trace_window_magnitudes(tr, rel_lo, rel_hi, dist, p.depth)
        pk = tr.get_pick(prefer_phase) or (tr.picks[0] if tr.picks else None)
        if pk is not None:
            self._apply_magnitudes_to_pick(pk, res)
        p.mag_ml = res["ml"]
        p.mag_mb = res["mb"]
        p.mag_ms = res["ms"]
        p.mag_mbb = res["mbb"]
        return {**res, "station": tr.station}


    def _cmd_beam(self, args: List[str], quals: Dict[str, str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        slo = float(args[1]) if len(args) > 1 else self.current_params.slowness
        ref = args[2] if len(args) > 2 else "CENTRE"
        vis = self.visible_traces()
        b_tr, _delays = compute_beam(vis, az, slo, ref_station=ref, config=self.config)
        self.traces.append(b_tr)
        msg = f"Created BEAM trace (az={az:.1f} deg, slo={slo:.2f} s/deg, ref={ref}) from {len(vis)} traces."
        self.emit(msg)
        return msg

    def _cmd_align(self, args: List[str]) -> str:
        ph = args[0] if args else (self.current_params.phase or "P")
        vis = self.visible_traces()
        a_tr = align_traces_on_phase(vis, phase_name=ph, create_sum_trace=True)
        if a_tr is not None:
            self.traces.append(a_tr)
            msg = f"Aligned traces on phase {ph} and appended ALIGN stack."
            self.emit(msg)
            return msg
        msg = f"No picks found for phase {ph} to align."
        self.emit(msg)
        return msg

    def _cmd_vespa(self, args: List[str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        s_lo = float(args[1]) if len(args) > 1 else 2.0
        s_hi = float(args[2]) if len(args) > 2 else 10.0
        s_step = float(args[3]) if len(args) > 3 else 0.5
        power = float(args[4]) if len(args) > 4 else 2.0
        vis = self.visible_traces()
        v_trs, _times, _slos, _mat = compute_vespagram(
            vis,
            azimuth_deg=az,
            slow_lo=s_lo,
            slow_hi=s_hi,
            slow_step=s_step,
            power=power,
            config=self.config,
        )
        self.traces.extend(v_trs)
        msg = f"Generated {len(v_trs)} Vespagram traces (az={az:.1f}, slo={s_lo}..{s_hi} s/deg, N={power})."
        self.emit(msg)
        return msg

    def _cmd_fk(self, args: List[str], quals: Optional[Dict[str, str]] = None) -> str:
        q = quals or {}
        f_lo = float(args[0]) if args else 0.5
        f_hi = float(args[1]) if len(args) > 1 else 3.0
        max_slo = float(args[2]) if len(args) > 2 else 12.0
        lo = self.time_window[0] if self.time_window else None
        hi = self.time_window[1] if self.time_window else None
        vis = self.visible_traces()
        use_calib = self.calib_table if ("NOCALIB" not in q) else None
        res = compute_fk_analysis(
            vis,
            t_lo=lo,
            t_hi=hi,
            f_lo=f_lo,
            f_hi=f_hi,
            max_slowness=max_slo,
            config=self.config,
            calib_table=use_calib,
            apply_calibration=True if use_calib is not None else None,
        )
        self.last_fk_result = res
        self.current_params.azimuth = res["azimuth"]
        self.current_params.slowness = res["slowness"]
        cal_tag = " [calibrated]" if res.get("calib_applied") else ""
        msg = (
            f"FK Result{cal_tag}: Azimuth = {res['azimuth']:.1f} deg, Slowness = {res['slowness']:.2f} s/deg "
            f"(Vel = {res['apparent_velocity_kms']:.2f} km/s, Semblance = {res['semblance']:.3f})"
        )
        self.emit(msg)
        return msg

    def _cmd_calib(self, args: List[str], quals: Dict[str, str]) -> str:
        if "OFF" in quals or "CLEAR" in quals or (args and args[0].upper() in ("OFF", "NONE", "CLEAR")):
            self.calib_table = None
            msg = "Slowness calibration cleared."
            self.emit(msg)
            return msg

        tbl: Optional[SlownessCalibrationTable] = None
        src_desc = ""

        if "FILE" in quals:
            fpath = quals["FILE"]
            tbl = SlownessCalibrationTable.from_file(fpath)
            src_desc = fpath
        elif args and (
            os.path.isfile(args[0])
            or args[0].lower().endswith((".dat", ".txt"))
            or os.sep in args[0]
        ):
            fpath = args[0]
            tbl = SlownessCalibrationTable.from_file(fpath)
            src_desc = fpath
        elif "ARRAY" in quals or args:
            arr_val = quals.get("ARRAY", "")
            if arr_val and arr_val != "TRUE":
                target: Any = arr_val
                src_desc = arr_val
            elif args:
                target = args[0]
                src_desc = args[0]
            else:
                vis_stas = [t.station for t in self.visible_traces() if t.station]
                target = vis_stas if vis_stas else "GRF"
                src_desc = ",".join(vis_stas[:4]) if vis_stas else "GRF"
            tbl = load_calibration_for_array(target, inputs_dir=self.config.inputs_dir, config=self.config)
            if tbl is None or len(tbl) == 0:
                raise ValueError(f"No slowness calibration table found for array/station {src_desc!r}")
        else:
            if self.calib_table is not None:
                msg = f"Active slowness calibration table: {len(self.calib_table)} entries ({self.calib_table.source_path or 'in-memory'})."
                self.emit(msg)
                return msg
            vis_stas = [t.station for t in self.visible_traces() if t.station]
            target = vis_stas if vis_stas else "GRF"
            tbl = load_calibration_for_array(target, inputs_dir=self.config.inputs_dir, config=self.config)
            if tbl is None or len(tbl) == 0:
                raise ValueError("No slowness calibration table found for current workspace")
            src_desc = tbl.source_path or "array"

        self.calib_table = tbl
        msg = f"Loaded slowness calibration table ({len(tbl)} entries) from {tbl.source_path or src_desc}."

        # If workspace already has a slowness solution, apply calibration immediately unless /NOAPPLY
        p = self.current_params
        if "NOAPPLY" not in quals and p.slowness > 0.0:
            raw_slo = float(p.residuals.get("raw_slowness", p.slowness)) if p.residuals else float(p.slowness)
            raw_az = float(p.residuals.get("raw_azimuth", p.azimuth)) if p.residuals else float(p.azimuth)
            ph = p.phase or "P"
            vis = self.visible_traces()
            ref_tr = vis[0] if vis else None
            pk = ref_tr.get_pick(ph) if ref_tr else None
            onset_ep = pk.onset_epoch if (pk and pk.onset) else (
                parse_sh_time(p.origin) if p.origin else (ref_tr.start_epoch if ref_tr else 0.0)
            )
            st_code = p.source if p.source in self.config.stations else (ref_tr.station if ref_tr else "GRA1")
            cal_param = locate_by_slowness_azimuth(
                st_code,
                ph,
                onset_ep,
                raw_slo,
                raw_az,
                p.depth,
                config=self.config,
                ttt=self.ttt,
                calib_table=self.calib_table,
                apply_calibration=True,
            )
            self.param_sets[self.active_set_index] = cal_param
            for tr in vis:
                pk_tr = tr.get_pick(ph)
                if pk_tr is not None:
                    pk_tr.l_slowness = cal_param.slowness
                    pk_tr.l_azim = cal_param.azimuth
            msg += (
                f" Applied calibration: slo {raw_slo:.2f}->{cal_param.slowness:.2f} s/deg, "
                f"baz {raw_az:.1f}->{cal_param.azimuth:.1f} deg, Dist={cal_param.distance:.2f} deg."
            )

        self.emit(msg)
        return msg

    def _cmd_locate(self, args: List[str], quals: Dict[str, str]) -> str:
        ph = args[0] if args else (self.current_params.phase or "P")
        vis = self.visible_traces()
        use_calib = self.calib_table if ("NOCALIB" not in quals) else None

        if "GEIGER" in quals or "NET" in quals:
            all_picks = [p for t in vis for p in t.picks]
            param, _resids, diag = locate_network_geiger(
                all_picks,
                depth_km=self.current_params.depth,
                free_depth=("FREE" in quals),
                config=self.config,
                ttt=self.ttt,
                return_diagnostics=True,
            )
            if self.last_focal_mechanism is not None:
                setattr(param, "focal_mechanism", self.last_focal_mechanism)
                diag.context_data["focal_mechanism"] = self.last_focal_mechanism
            self.param_sets[self.active_set_index] = param
            self._emit_fit_diagnostic(diag)
            msg = (
                f"Network Location: Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, "
                f"Origin={param.origin}, Region={param.region_name}"
            )
            self.emit(msg)
            return msg

        if "SLOWNESS" in quals:
            ref_tr = vis[0] if vis else None
            pk = ref_tr.get_pick(ph) if ref_tr else None
            onset_ep = pk.onset_epoch if (pk and pk.onset) else (ref_tr.start_epoch if ref_tr else 0.0)
            st_code = ref_tr.station if ref_tr else "GRA1"
            param = locate_by_slowness_azimuth(
                st_code,
                ph,
                onset_ep,
                self.current_params.slowness,
                self.current_params.azimuth,
                self.current_params.depth,
                config=self.config,
                ttt=self.ttt,
                calib_table=use_calib,
                apply_calibration=True if use_calib is not None else None,
            )
            self.param_sets[self.active_set_index] = param
            msg = (
                f"Slowness Location: Dist={param.distance:.2f} deg, Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, "
                f"Origin={param.origin}, Region=[{param.region_id}] {param.region_name}"
            )
            self.emit(msg)
            return msg

        # Default: Plane-wave array location across phase picks for `ph` (deduplicated by station)
        picked_traces: List[Trace] = []
        arr_times: List[float] = []
        seen_stations: set[str] = set()
        for tr in vis:
            pk = tr.get_pick(ph)
            if pk is not None and pk.onset and tr.station not in seen_stations:
                seen_stations.add(tr.station)
                picked_traces.append(tr)
                arr_times.append(pk.onset_epoch)

        if len(picked_traces) < 3:
            raise ValueError(f"Plane-wave location requires at least 3 picks for phase '{ph}' (found {len(picked_traces)})")

        coords = self.config.relative_xy_km([t.station for t in picked_traces], ref_station="CENTRE")
        res = plane_wave_locate(
            arr_times,
            [c[0] for c in coords],
            [c[1] for c in coords],
            elev_m=[c[2] for c in coords],
            vp_kms=[c[3] for c in coords],
            dt=picked_traces[0].delta,
            calib_table=use_calib,
            apply_calibration=True if use_calib is not None else None,
            stations=[t.station for t in picked_traces],
            config=self.config,
        )
        for tr, resid in zip(picked_traces, res["residuals"]):
            pk = tr.get_pick(ph)
            if pk:
                pk.resid = resid
                pk.slowness = res.get("raw_slowness", res["slowness"]) if res.get("calib_applied") else res["slowness"]
                pk.beam_azim = res.get("raw_azimuth", res["azimuth"]) if res.get("calib_applied") else res["azimuth"]
                if res.get("calib_applied"):
                    pk.l_slowness = res["slowness"]
                    pk.l_azim = res["azimuth"]

        # Pass raw slowness/azimuth when calib_table is passed so locate_by_slowness_azimuth calibrates once and populates metadata
        raw_slo = res.get("raw_slowness", res["slowness"]) if res.get("calib_applied") else res["slowness"]
        raw_az = res.get("raw_azimuth", res["azimuth"]) if res.get("calib_applied") else res["azimuth"]
        param = locate_by_slowness_azimuth(
            picked_traces[0].station,
            ph,
            arr_times[0],
            raw_slo,
            raw_az,
            self.current_params.depth,
            config=self.config,
            ttt=self.ttt,
            calib_table=use_calib,
            apply_calibration=True if res.get("calib_applied") else False,
        )
        param.d_azimuth = res["d_azimuth"]
        param.d_slowness = res["d_slowness"]
        self.param_sets[self.active_set_index] = param
        cal_note = " [calibrated]" if res.get("calib_applied") else ""
        msg = (
            f"Plane Wave Fit{cal_note} ({len(picked_traces)} stations): BAZ = {res['azimuth']:.1f} +/- {res['d_azimuth']:.1f} deg, "
            f"Slowness = {res['slowness']:.2f} +/- {res['d_slowness']:.2f} s/deg -> "
            f"Dist = {param.distance:.1f} deg ({param.region_name})"
        )
        self.emit(msg)
        return msg

    def _cmd_identify(self, args: List[str], quals: Dict[str, str]) -> str:
        p = self.current_params
        depth_km = float(quals["DEPTH"]) if "DEPTH" in quals else float(p.depth)
        dist_override = float(quals["DIST"]) if "DIST" in quals else None
        cand_phases: Optional[List[str]] = None
        if "PHASES" in quals:
            cand_phases = [x.strip() for x in quals["PHASES"].split(",") if x.strip()]

        sigma_t = float(quals.get("SIGMA_T", 2.0))
        sigma_s = float(quals.get("SIGMA_S", 1.0))

        # Direct travel-time + slowness phase identification via /TIME=<sec>
        if "TIME" in quals:
            t_obs_rel = float(quals["TIME"])
            slo_meas = float(quals.get("SLO", p.slowness))
            dist_val = dist_override if dist_override is not None else float(p.distance)
            if dist_val <= 0.0:
                raise ValueError("IDENTIFY /TIME requires /DIST=<deg> or a valid distance in params")
            baz_meas = float(quals["BAZ"]) if "BAZ" in quals else (float(p.azimuth) if p.azimuth > 0.0 else None)
            baz_theo = float(quals["BAZ_THEO"]) if "BAZ_THEO" in quals else None
            cands = identify_phase(
                t_obs_rel_sec=t_obs_rel,
                slowness_meas_s_deg=slo_meas,
                dist_deg=dist_val,
                depth_km=depth_km,
                baz_meas_deg=baz_meas,
                baz_theo_deg=baz_theo,
                sigma_t=sigma_t,
                sigma_s=sigma_s,
                w_s=1.0 if slo_meas > 0.0 else 0.0,
                candidate_phases=cand_phases,
                ttt=self.ttt,
            )
            self.last_identify_results = cands
            if not cands:
                msg = "No matching seismic phases found."
                self.emit(msg)
                return msg
            lines = [f"Phase Identification (t={t_obs_rel:.2f}s, slo={slo_meas:.2f}s/deg, dist={dist_val:.1f}deg):"]
            for rank, c in enumerate(cands[:5], start=1):
                lines.append(
                    f"  {rank}. {c['phase']:<7s} prob={c['probability']:.3f} chi2={c['chi2']:.2f} "
                    f"dt={c['t_resid']:+.2f}s ds={c['s_resid']:+.2f}s/deg"
                )
            msg = "\n".join(lines)
            self.emit(msg)
            return msg

        list_spec = quals.get("LIST", args[0] if args else "ALL")
        idxs = self.parse_trace_list(list_spec)
        target_traces = [self.traces[i] for i in idxs if self.traces[i].picks]
        if not target_traces:
            raise ValueError("IDENTIFY requires picked traces in workspace (or /TIME=... /DIST=...)")

        if "ORIGIN" in quals:
            try:
                orig_ep = float(quals["ORIGIN"])
            except ValueError:
                orig_ep = parse_sh_time(quals["ORIGIN"])
        elif p.origin:
            orig_ep = parse_sh_time(p.origin)
        elif target_traces[0].torig > 0.0:
            orig_ep = float(target_traces[0].torig)
        else:
            orig_ep = float(target_traces[0].start_epoch)

        update_names = ("NOUPDATE" not in quals) and ("DRY" not in quals)
        max_chi2 = float(quals["MAX_CHI2"]) if "MAX_CHI2" in quals else None
        def_slo = float(quals["SLO"]) if "SLO" in quals else (float(p.slowness) if p.slowness > 0.0 else None)
        def_baz = float(quals["BAZ"]) if "BAZ" in quals else (float(p.azimuth) if p.azimuth > 0.0 else None)
        baz_theo = float(quals["BAZ_THEO"]) if "BAZ_THEO" in quals else None

        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        all_results: List[Dict[str, Any]] = []
        for tr in target_traces:
            if dist_override is not None:
                dist_i = dist_override
            elif has_hypo:
                st = self.config.get_station(tr.station)
                if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                    dist_i, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = dist_i
                elif tr.lat != 0.0 or tr.lon != 0.0:
                    dist_i, _, _ = sphere_diff(tr.lat, tr.lon, p.lat, p.lon)
                    tr.distance = dist_i
                else:
                    dist_i = float(tr.distance) if tr.distance > 0.0 else float(p.distance)
            elif tr.distance > 0.0:
                dist_i = float(tr.distance)
            else:
                dist_i = float(p.distance)

            if dist_i <= 0.0:
                raise ValueError(f"IDENTIFY requires epicentral distance for trace {tr.station} (set params.distance, hypocenter, or /DIST=<deg>)")

            res_tr = identify_trace_picks(
                [tr],
                origin_epoch=orig_ep,
                dist_deg=dist_i,
                depth_km=depth_km,
                baz_theo_deg=baz_theo,
                default_slowness=def_slo,
                default_baz=def_baz,
                update_pick_names=update_names,
                max_chi2=max_chi2,
                sigma_t=sigma_t,
                sigma_s=sigma_s,
                candidate_phases=cand_phases,
                ttt=self.ttt,
                config=self.config,
            )
            all_results.extend(res_tr)

        self.last_identify_results = all_results
        if not all_results:
            msg = "No phase candidates matched."
            self.emit(msg)
            return msg

        lines = [f"Identified {len(all_results)} pick(s):"]
        for r in all_results:
            top3 = ", ".join(f"{c['phase']}({c['probability']:.2f})" for c in r["candidates"][:3])
            lines.append(
                f"  {r['station']} {r['old_phase']} -> {r['identified_phase']} "
                f"(prob={r['probability']:.3f}, chi2={r['chi2']:.2f}, dt={r['t_resid']:+.2f}s) | top: {top3}"
            )
        msg = "\n".join(lines)
        self.emit(msg)
        return msg

    def _cmd_pick(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        lo = self.time_window[0] if self.time_window else None
        hi = self.time_window[1] if self.time_window else None
        count = 0
        for i in idxs:
            pk = pick_onset_baer_kradolfer(self.traces[i], t_lo=lo, t_hi=hi)
            if pk is not None:
                self.traces[i].add_or_replace_pick(pk)
                count += 1
        msg = f"Auto-picked P onset on {count} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_corrpick(self, args: List[str]) -> str:
        ph = args[0] if args else "P"
        vis = self.visible_traces()
        pks = corrpick_array(vis, ref_trace_idx=0, phase_name=ph)
        msg = f"Cross-correlation picked '{ph}' across {len(pks)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_corr(self, args: List[str]) -> str:
        vis = self.visible_traces()
        if len(vis) < 2:
            raise ValueError("CORR requires at least 2 traces")
        if len(args) in (1, 3) or len(args) > 4:
            raise ValueError("Usage: CORR [<t1> <t2> [<lo_sec> <hi_sec>]]")
        if len(args) >= 2:
            if not args[0].isdigit() or not args[1].isdigit():
                raise ValueError("CORR trace indices must be positive integers")
            t1_idx = int(args[0]) - 1
            t2_idx = int(args[1]) - 1
            if not (0 <= t1_idx < len(vis) and 0 <= t2_idx < len(vis)):
                raise ValueError(f"CORR trace index out of range (1..{len(vis)})")
        else:
            t1_idx, t2_idx = 0, 1
        lo = float(args[2]) if len(args) == 4 else 0.0
        hi = float(args[3]) if len(args) == 4 else min(10.0, vis[t1_idx].duration)
        lags, corrs, best_lag, max_c = correlate_traces(vis[t1_idx], vis[t2_idx], lo, hi)
        tr_c = Trace(
            data=corrs,
            delta=vis[t1_idx].delta,
            station="CORR",
            comp="Z",
            start_epoch=vis[t1_idx].start_epoch + float(lags[0]),
            comment=f"Max corr={max_c:.3f} at lag={best_lag:+.3f}s",
        )
        self.traces.append(tr_c)
        msg = f"Cross-correlation peak = {max_c:.3f} at lag = {best_lag:+.3f} s."
        self.emit(msg)
        return msg

    def _cmd_mag(self, args: List[str]) -> str:
        ph = args[0] if args else "P"
        p = self.current_params
        global_dist = p.distance if p.distance > 0 else 50.0
        depth = p.depth
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        vis = self.visible_traces()
        mls, mbs, mss, mbbs = [], [], [], []
        dists_used = []

        for tr in vis:
            dist_i = global_dist
            if has_hypo:
                st = self.config.get_station(tr.station)
                if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                    dist_i, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = dist_i
            elif tr.distance > 0.0:
                dist_i = tr.distance

            pk = tr.get_pick(ph)
            if pk is not None and pk.onset:
                rel_t = pk.onset_epoch - tr.start_epoch
                lo, hi = max(0.0, rel_t), min(tr.duration, rel_t + 8.0)
            elif self.time_window:
                lo, hi = self.time_window[0], self.time_window[1]
            else:
                lo, hi = 0.0, tr.duration

            res = compute_trace_window_magnitudes(tr, lo, hi, dist_i, depth)
            if res["ampl_nm"] > 0:
                mls.append(res["ml"])
                mbs.append(res["mb"])
                mss.append(res["ms"])
                mbbs.append(res["mbb"])
                dists_used.append(dist_i)
                if pk is not None:
                    self._apply_magnitudes_to_pick(pk, res)

        n_mags = len(mbs)
        p.mags_count = n_mags
        if n_mags > 0:
            p.mag_ml = float(np.mean(mls))
            p.mag_mb = float(np.mean(mbs))
            p.mag_ms = float(np.mean(mss))
            p.mag_mbb = float(np.mean(mbbs))
            if n_mags >= 2:
                p.d_ml = float(np.std(mls, ddof=1))
                p.d_mb = float(np.std(mbs, ddof=1))
                p.d_ms = float(np.std(mss, ddof=1))
                p.d_mbb = float(np.std(mbbs, ddof=1))
            else:
                p.d_ml = 0.0
                p.d_mb = 0.0
                p.d_ms = 0.0
                p.d_mbb = 0.0
        avg_dist = float(np.mean(dists_used)) if dists_used else global_dist
        msg = (
            f"Mean Magnitudes ({n_mags} traces, Dist={avg_dist:.1f} deg): "
            f"ML={p.mag_ml:.2f}, mb={p.mag_mb:.2f}, "
            f"MS={p.mag_ms:.2f}, mB={p.mag_mbb:.2f}"
        )
        self.emit(msg)
        return msg

    def _cmd_surfmag(self, args: List[str], quals: Dict[str, str]) -> str:
        p = self.current_params
        list_spec = quals.get("LIST")
        if list_spec is None and args:
            tok = args[0]
            if tok.upper() in ("ALL", "SEL", "SELECTED", "*") or "," in tok or "-" in tok or tok.isdigit():
                list_spec = tok
        if list_spec is None:
            list_spec = "ALL"

        idxs = self.parse_trace_list(list_spec)
        selected_traces = [self.traces[i] for i in idxs]
        if not selected_traces:
            raise ValueError("SURFMAG requires at least one trace in workspace")

        vertical_only = ("ALLCOMP" not in quals)
        if vertical_only:
            z_traces = [tr for tr in selected_traces if (tr.comp or "").upper() == "Z"]
            eff_traces = z_traces if z_traces else selected_traces
        else:
            eff_traces = selected_traces

        p_lo = float(quals.get("PERIOD_LO", quals.get("TMIN", 18.0)))
        p_hi = float(quals.get("PERIOD_HI", quals.get("TMAX", 22.0)))
        lo = float(quals["T_LO"]) if "T_LO" in quals else (self.time_window[0] if self.time_window else None)
        hi = float(quals["T_HI"]) if "T_HI" in quals else (self.time_window[1] if self.time_window else None)
        input_type = (
            "velocity"
            if ("VELOCITY" in quals or quals.get("INPUT", "").lower().startswith("v"))
            else "displacement"
        )
        dist_override = float(quals["DIST"]) if "DIST" in quals else None
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        global_dist = p.distance if p.distance > 0.0 else 50.0

        dists: List[float] = []
        for tr in eff_traces:
            if dist_override is not None:
                dist_i = dist_override
            elif has_hypo:
                st = self.config.get_station(tr.station)
                if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                    dist_i, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = dist_i
                elif tr.lat != 0.0 or tr.lon != 0.0:
                    dist_i, _, _ = sphere_diff(tr.lat, tr.lon, p.lat, p.lon)
                    tr.distance = dist_i
                else:
                    dist_i = float(tr.distance) if tr.distance > 0.0 else global_dist
            elif tr.distance > 0.0:
                dist_i = float(tr.distance)
            else:
                dist_i = global_dist
            dists.append(dist_i)

        scan_res = scan_surface_waves(
            eff_traces,
            distances_deg=dists,
            period_lo=p_lo,
            period_hi=p_hi,
            t_lo=lo,
            t_hi=hi,
            vertical_only=False,
            input_type=input_type,
        )
        self.last_surfmag_result = scan_res

        valid_ms = [float(s["ms"]) for s in scan_res["stations"] if s.get("ms") is not None and s["ms"] > 0.0]
        if scan_res["ms_mean"] is not None:
            p.mag_ms = float(scan_res["ms_mean"])
            p.d_ms = float(scan_res["ms_std"] or 0.0)
            p.mags_count = max(p.mags_count, len(valid_ms))

        for tr, st_res in zip(eff_traces, scan_res["stations"]):
            pk = tr.get_pick("LR") or tr.get_pick("LQ") or tr.get_pick("R")
            if pk is None:
                pk = PhasePick(
                    name="LR",
                    onset=format_sh_time(tr.start_epoch + float(st_res["time_sec"])),
                    source="a",
                    filter_name=tr.filter_name,
                    station=tr.station,
                    comp=tr.comp,
                    chan1=tr.chan1,
                    chan2=tr.chan2,
                )
                tr.picks.insert(0, pk)
            pk.ampl = float(st_res["ampl_nm"])
            pk.ampl_time = float(st_res["time_sec"])
            pk.period = float(st_res["period"])
            if st_res["ms"] is not None:
                pk.mag_ms = float(st_res["ms"])

        ms_str = f"{p.mag_ms:.2f} +/- {p.d_ms:.2f}" if valid_ms else "N/A"
        max_ampl = max((float(s["ampl_nm"]) for s in scan_res["stations"]), default=0.0)
        msg = (
            f"Surface Wave Magnitude ({len(eff_traces)} Z-trace(s), {p_lo:.0f}-{p_hi:.0f}s): "
            f"MS={ms_str}, max A={max_ampl:.1f} nm"
        )
        self.emit(msg)
        return msg

    def _cmd_theo(self, args: List[str]) -> str:
        dist = float(args[0]) if args else self.current_params.distance
        depth = float(args[1]) if len(args) > 1 else self.current_params.depth
        vis = self.visible_traces()
        if not vis:
            msg = "No traces in workspace."
            self.emit(msg)
            return msg
        orig_ep = parse_sh_time(self.current_params.origin) if self.current_params.origin else vis[0].start_epoch
        arrs = self.ttt.theoretical_arrivals(dist, depth, orig_ep)
        for tr in vis:
            tr.theo_picks.update(arrs)
        msg = f"Computed {len(arrs)} theoretical phase arrivals for Dist={dist:.1f} deg, Depth={depth:.1f} km."
        self.emit(msg)
        return msg

    def _cmd_spectrum(self, args: List[str], quals: Dict[str, str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "1")
        if not idxs:
            msg = "No trace selected for spectrum."
            self.emit(msg)
            return msg
        tr = self.traces[idxs[0]]
        lo = self.time_window[0] if self.time_window else None
        hi = self.time_window[1] if self.time_window else None
        if "MEM" in quals:
            freqs, amps = compute_mem_spectrum(tr, lo, hi)
            kind = "MEM"
        elif "PSD" in quals:
            freqs, amps = compute_welch_power_spectrum(tr, lo, hi)
            kind = "PSD"
        else:
            freqs, amps = compute_fft_spectrum(tr, lo, hi)
            kind = "FFT"
        peak_i = int(np.argmax(amps)) if amps.size > 0 else 0
        f_peak = float(freqs[peak_i]) if freqs.size > 0 else 0.0
        a_peak = float(amps[peak_i]) if amps.size > 0 else 0.0
        msg = f"{kind} Spectrum ({tr.station} {tr.channel3}): Peak amplitude = {a_peak:.3e} at {f_peak:.3f} Hz"
        self.emit(msg)
        return msg

    def _cmd_fit(self, args: List[str], quals: Dict[str, str]) -> str:
        rem = list(args)
        mode: Optional[str] = None
        if "BOATWRIGHT" in quals:
            mode = "BOATWRIGHT"
        elif "BRUNE" in quals:
            mode = "BRUNE"
        elif any(k in quals for k in ("POL", "POLARIZATION", "MDIR")):
            mode = "POL"
        elif any(k in quals for k in ("FOCAL", "FM", "BEACHBALL")):
            mode = "FOCAL"
        elif any(k in quals for k in ("LOCATE", "GEIGER", "NET")):
            mode = "LOCATE"
        elif rem and rem[0].upper() in ("BRUNE", "BOATWRIGHT", "POL", "POLARIZATION", "MDIR", "FOCAL", "FM", "BEACHBALL", "LOCATE", "GEIGER", "NET"):
            tok = rem.pop(0).upper()
            if tok in ("POLARIZATION", "MDIR"):
                mode = "POL"
            elif tok in ("FM", "BEACHBALL"):
                mode = "FOCAL"
            elif tok in ("GEIGER", "NET"):
                mode = "LOCATE"
            else:
                mode = tok

        if mode is None:
            raise ValueError("FIT requires a mode qualifier: /BRUNE, /BOATWRIGHT, /POL, /FOCAL, or /LOCATE")

        if mode in ("BRUNE", "BOATWRIGHT"):
            model_name = "boatwright" if mode == "BOATWRIGHT" else "brune"
            list_spec = quals.get("LIST")
            t_lo = float(quals["T_LO"]) if "T_LO" in quals else (self.time_window[0] if self.time_window else None)
            t_hi = float(quals["T_HI"]) if "T_HI" in quals else (self.time_window[1] if self.time_window else None)

            if list_spec is None and rem:
                if len(rem) >= 3:
                    list_spec = rem[0]
                    t_lo = float(rem[1])
                    t_hi = float(rem[2])
                elif len(rem) == 2:
                    if "." in rem[0] or not rem[0].isdigit() or int(rem[0]) > len(self.traces):
                        t_lo = float(rem[0])
                        t_hi = float(rem[1])
                    else:
                        list_spec = rem[0]
                else:
                    list_spec = rem[0]

            if list_spec is None:
                list_spec = "SEL"
            idxs = self.parse_trace_list(list_spec)
            if not idxs:
                raise ValueError("FIT /BRUNE requires at least one trace in workspace")
            tr = self.traces[idxs[0]]

            noise_lo = float(quals["NOISE_LO"]) if "NOISE_LO" in quals else None
            noise_hi = float(quals["NOISE_HI"]) if "NOISE_HI" in quals else None
            f_lo = float(quals.get("F_LO", 0.1))
            f_hi = float(quals["F_HI"]) if "F_HI" in quals else None
            input_type = (
                "velocity"
                if ("VELOCITY" in quals or quals.get("INPUT", "").lower().startswith("v"))
                else "displacement"
            )
            phase = quals.get("PHASE", self.current_params.phase or "P")

            phys_kwargs: Dict[str, Any] = {}
            if "DIST_M" in quals:
                phys_kwargs["distance_m"] = float(quals["DIST_M"])
            elif "DIST" in quals:
                phys_kwargs["distance_m"] = float(quals["DIST"]) * 111194.9266
            elif tr.distance > 0.0:
                phys_kwargs["distance_m"] = float(tr.distance) * 111194.9266
            elif self.current_params.distance > 0.0:
                phys_kwargs["distance_m"] = float(self.current_params.distance) * 111194.9266

            if "VEL" in quals or "VEL_M_S" in quals:
                phys_kwargs["vel_m_s"] = float(quals.get("VEL", quals.get("VEL_M_S")))
            if "RHO" in quals or "RHO_KG_M3" in quals:
                phys_kwargs["rho_kg_m3"] = float(quals.get("RHO", quals.get("RHO_KG_M3")))
            if "WAVE" in quals:
                phys_kwargs["wave_type"] = quals["WAVE"]
            if "NOTSTAR" in quals:
                phys_kwargs["fit_t_star"] = False

            inv_res = invert_brune_spectrum(
                tr,
                t_lo=t_lo,
                t_hi=t_hi,
                noise_lo=noise_lo,
                noise_hi=noise_hi,
                model=model_name,
                f_lo=f_lo,
                f_hi=f_hi,
                input_type=input_type,
                **phys_kwargs,
            )
            diag = build_brune_fit_diagnostic(inv_res, station=tr.station, phase=phase)

            p = self.current_params
            p.corner_freq = float(inv_res["fc"])
            p.moment_m0 = float(inv_res["m0"])
            p.mag_mw = float(inv_res["mw"])
            p.rmse = float(inv_res["rms_log10"])
            p.wrmse = float(inv_res.get("wrmse_log10", inv_res["rms_log10"]))
            p.chi2_reduced = float(inv_res.get("chi2_reduced", 0.0))
            p.condition_number = float(inv_res.get("condition_number", 1.0))

            self.last_brune_result = inv_res
            self._emit_fit_diagnostic(diag)
            msg = (
                f"{model_name.title()} Fit ({tr.station} {tr.channel3}): "
                f"Omega0={inv_res['omega0']:.3e} nm*s, fc={inv_res['fc']:.3f} Hz, "
                f"t*={inv_res['t_star']:.3f} s, M0={inv_res['m0']:.3e} N*m, "
                f"Mw={inv_res['mw']:.2f}, dSigma={inv_res['stress_drop_bar']:.2f} bar, "
                f"RMS={inv_res['rms_log10']:.4f}"
            )
            self.emit(msg)
            return msg

        if mode == "POL":
            prefer_station = quals.get("STATION")
            if prefer_station is None and rem and not rem[0].replace(".", "").replace("-", "").isdigit():
                prefer_station = rem.pop(0)
            tr_z, tr_n, tr_e = self._find_zne_triplet(prefer_station=prefer_station)
            lo = float(quals["T_LO"]) if "T_LO" in quals else (
                float(rem[0]) if len(rem) >= 2 else (self.time_window[0] if self.time_window else None)
            )
            hi = float(quals["T_HI"]) if "T_HI" in quals else (
                float(rem[1]) if len(rem) >= 2 else (self.time_window[1] if self.time_window else None)
            )
            diag = analyze_polarization_diagnostic(tr_z, tr_n, tr_e, t_lo=lo, t_hi=hi)
            ctx = diag.context_data
            p = self.current_params
            p.azimuth = float(ctx.get("baz", 0.0))
            p.d_azimuth = float(ctx.get("sigma_theta_deg", 0.0))
            p.incidence = float(ctx.get("incidence", 0.0))
            p.rmse = float(diag.rmse)
            p.wrmse = float(diag.wrmse)
            p.chi2_reduced = float(diag.chi2_reduced)
            p.condition_number = float(diag.condition_number)
            for t in (tr_z, tr_n, tr_e):
                t.inci = p.incidence
            self.last_mdir_result = {
                "azimuth": p.azimuth,
                "incidence": p.incidence,
                "rectilinearity": float(ctx.get("rectilinearity", 0.0)),
                "planarity": float(ctx.get("planarity", 0.0)),
                "lambda1": float(ctx.get("lambda1", 0.0)),
                "lambda2": float(ctx.get("lambda2", 0.0)),
                "lambda3": float(ctx.get("lambda3", 0.0)),
                "rms_pol": float(ctx.get("rms_pol", 0.0)),
                "sigma_theta_deg": float(ctx.get("sigma_theta_deg", 0.0)),
                "sigma_phi_deg": float(ctx.get("sigma_phi_deg", 0.0)),
            }
            self._emit_fit_diagnostic(diag)
            msg = (
                f"Polarization Fit ({tr_z.station}): BAZ={p.azimuth:.1f} +/- {ctx.get('sigma_theta_deg', 0.0):.1f} deg, "
                f"Inc={p.incidence:.1f} +/- {ctx.get('sigma_phi_deg', 0.0):.1f} deg, "
                f"Rect={ctx.get('rectilinearity', 0.0):.3f}, RMS_pol={diag.rmse:.4f}"
            )
            self.emit(msg)
            return msg

        if mode == "FOCAL":
            vis = self.visible_traces()
            p = self.current_params
            ev_lat = float(quals["LAT"]) if "LAT" in quals else float(p.lat)
            ev_lon = float(quals["LON"]) if "LON" in quals else float(p.lon)
            depth_km = float(quals["DEPTH"]) if "DEPTH" in quals else float(p.depth)
            grid_step = float(quals.get("STEP", 10.0))
            pols = extract_polarities_from_traces(
                vis,
                ev_lat,
                ev_lon,
                depth_km=depth_km,
                config=self.config,
                ttt=self.ttt,
            )
            if not pols:
                raise ValueError(
                    "FIT /FOCAL requires at least one trace pick with non-zero polarity (sign +1/-1 or 'c'/'d')"
                )
            fm_res = solve_focal_mechanism(pols, grid_step_deg=grid_step, refine=("NOREFINE" not in quals))
            pol_tuples = [
                (float(m["azimuth"]), float(m["takeoff"]), int(m["observed"]))
                for m in fm_res["station_matches"]
            ]
            fm_payload: Dict[str, Any] = {
                **fm_res,
                "misfit": float(fm_res["misfit_ratio"]),
                "polarities": pol_tuples,
            }
            self.last_focal_mechanism = fm_payload
            setattr(p, "focal_mechanism", fm_payload)

            residuals: List[ResidualEntry] = []
            for m in fm_res["station_matches"]:
                obs_v = float(m["observed"])
                pred_v = float(m["predicted_sign"])
                res_v = obs_v - pred_v
                residuals.append(
                    ResidualEntry(
                        identifier=f"{m['station']}.P.FM",
                        observed=obs_v,
                        calculated=pred_v,
                        residual=res_v,
                        normalized_residual=abs(res_v),
                        weight=float(m["weight"]),
                        uncertainty=1.0,
                        azimuth_deg=float(m["azimuth"]),
                        station=str(m["station"]),
                        phase="P",
                        channel="Z",
                        is_outlier=not bool(m["matched"]),
                    )
                )

            fm_params = {
                "strike": ParameterUncertainty(
                    name="strike",
                    value=float(fm_res["strike"]),
                    standard_error=grid_step,
                    confidence_interval_95=(float(fm_res["strike"]) - grid_step, float(fm_res["strike"]) + grid_step),
                    unit="deg",
                ),
                "dip": ParameterUncertainty(
                    name="dip",
                    value=float(fm_res["dip"]),
                    standard_error=grid_step,
                    confidence_interval_95=(max(0.0, float(fm_res["dip"]) - grid_step), min(90.0, float(fm_res["dip"]) + grid_step)),
                    unit="deg",
                ),
                "rake": ParameterUncertainty(
                    name="rake",
                    value=float(fm_res["rake"]),
                    standard_error=grid_step,
                    confidence_interval_95=(float(fm_res["rake"]) - grid_step, float(fm_res["rake"]) + grid_step),
                    unit="deg",
                ),
            }
            diag = FitDiagnosticResult(
                fit_type=FitType.FOCAL_MECHANISM,
                title=f"Double-Couple Focal Mechanism Fit ({fm_res['n_obs']} polarities)",
                converged=bool(fm_res["converged"]),
                iterations=1,
                dof=max(1, int(fm_res["n_obs"]) - 3),
                rmse=float(fm_res["misfit_ratio"]),
                wrmse=float(fm_res["misfit_ratio"]),
                parameters=fm_params,
                residuals=residuals,
                outlier_count=int(fm_res["n_misfit"]),
                context_data={"focal_mechanism": fm_payload},
            )
            self._emit_fit_diagnostic(diag)
            msg = (
                f"Focal Mechanism Fit ({fm_res['n_obs']} polarities): "
                f"Strike={fm_res['strike']:.1f} deg, Dip={fm_res['dip']:.1f} deg, Rake={fm_res['rake']:.1f} deg "
                f"(Aux: {fm_res['strike2']:.1f}/{fm_res['dip2']:.1f}/{fm_res['rake2']:.1f}, "
                f"Misfit={fm_res['n_misfit']}/{fm_res['n_obs']} [{fm_res['misfit_ratio']:.2f}])"
            )
            self.emit(msg)
            return msg

        # mode == "LOCATE"
        vis = self.visible_traces()
        all_picks = [pk for t in vis for pk in t.picks]
        depth_km = float(quals["DEPTH"]) if "DEPTH" in quals else float(self.current_params.depth)
        free_depth = ("FREE" in quals)
        init_lat = float(quals["LAT"]) if "LAT" in quals else None
        init_lon = float(quals["LON"]) if "LON" in quals else None
        excl = [s.strip() for s in quals["EXCLUDE"].split(",") if s.strip()] if "EXCLUDE" in quals else None
        param, _resids, diag = locate_network_geiger(
            all_picks,
            depth_km=depth_km,
            free_depth=free_depth,
            initial_lat=init_lat,
            initial_lon=init_lon,
            config=self.config,
            ttt=self.ttt,
            excluded_stations=excl,
            return_diagnostics=True,
        )
        if self.last_focal_mechanism is not None:
            setattr(param, "focal_mechanism", self.last_focal_mechanism)
            diag.context_data["focal_mechanism"] = self.last_focal_mechanism
        self.param_sets[self.active_set_index] = param
        self._emit_fit_diagnostic(diag)
        msg = (
            f"Geiger Location Fit: Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, "
            f"RMSE={param.rmse:.3f} s, Origin={param.origin}, Region={param.region_name}"
        )
        self.emit(msg)
        return msg

