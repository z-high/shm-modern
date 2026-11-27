"""Array, location, fit-diagnostic, spectrum, polarization, and focal-mechanism actions."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from PyQt6 import sip
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitModificationRequest,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import AnalysisParamSet, Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    ParticleMotionDialog,
    SpectrumDialog,
    VespagramDialog,
)
from shm_app.gui.dialog_event_summary import EventSummaryDialog
from shm_app.gui.dialogs_location import NetworkLocationDialog
from shm_app.gui.main_window_workers import _RecomputeFitWorker
from shm_app.gui.world_map import WorldMapDialog
from shm_app.seismology.focal_mechanism import (
    extract_polarities_from_traces,
    p_wave_radiation_pattern,
    solve_focal_mechanism,
)
from shm_app.seismology.geodesy import sphere_diff


class _MainWindowAnalysisMixin:
    """Seismological analysis, interactive fit-diagnostic recomputation, and summary actions."""

    def _spectrum_window_for(self, tr: Trace) -> Tuple[float, float]:
        """Magnifier window converted to trace-relative seconds (clamped to the trace)."""
        cv = self.main_canvas
        rel_start = tr.start_epoch + tr.display_offset - cv.reference_epoch()
        lo = max(0.0, cv.mag_t_lo - rel_start)
        hi = min(tr.duration, cv.mag_t_hi - rel_start)
        return lo, hi

    def _add_spectra(self, traces: List[Trace]) -> None:
        cv = self.main_canvas
        items = []
        skipped = []
        for tr in traces:
            lo, hi = self._spectrum_window_for(tr)
            if hi - lo < 8 * tr.delta:
                skipped.append(f"{tr.station} {tr.channel3}")
                continue
            label = f"{tr.station} {tr.channel3}  [{cv.mag_t_lo:.1f}..{cv.mag_t_hi:.1f} s]"
            items.append((tr, lo, hi, label))
        if not items:
            self.status_bar.showMessage(
                "The yellow window contains no data of the chosen trace(s): move it over the signal.", 6000
            )
            return
        dlg = self._find_open_dialog(SpectrumDialog)
        is_new = dlg is None
        if is_new:
            dlg = SpectrumDialog(None, parent=self)
            dlg.fitDiagnosticReady.connect(self.fit_bridge.fitDiagnosticAvailable.emit)
        added = sum(1 for tr, lo, hi, label in items if dlg.add_curve(tr, lo, hi, label))
        if is_new:
            self._show_nonmodal(dlg)
        else:
            self._raise_nonmodal(dlg)
        msg = f"Spectrum: {added} curve(s) added ({len(dlg.curves)} total)."
        if added == 0:
            msg = "Spectrum for this window and trace is already in the graph."
        if skipped:
            msg += " No data in the window for: " + ", ".join(skipped)
        self.status_bar.showMessage(msg, 6000)

    def action_open_spectrum_dialog(self) -> None:
        idx = self._resolve_visible_mag_trace_idx()
        if idx is None:
            return
        self.main_canvas.mag_trace_idx = idx
        self._add_spectra([self.engine.traces[idx]])

    def action_open_spectrum_selected(self) -> None:
        sel = [t for t in self.engine.traces if t.selected and not t.hidden]
        if not sel:
            self.status_bar.showMessage("No traces selected: click the left margin of a trace first.", 5000)
            return
        self._add_spectra(sel)


    def action_form_beam(self) -> None:
        """Toggle: add a beam trace; run again to remove it."""
        if any(t.station == "BEAM" for t in self.engine.traces):
            self.engine.traces = [t for t in self.engine.traces if t.station != "BEAM"]
            self.refresh_views()
            self._append_console_log("BEAM trace removed.")
            return
        p = self.engine.current_params
        az = p.azimuth if p.azimuth != 0 else 45.0
        slo = p.slowness if p.slowness > 0 else 6.5
        self.engine.execute_line(f"BEAM {az} {slo} CENTRE")
        self.refresh_views()

    @staticmethod
    def _offset_key(tr: Trace) -> Tuple[str, str, float]:
        return (tr.station, tr.channel3, tr.start_epoch)

    def action_align_phase(self) -> None:
        """Toggle: align on the active phase and add a sum trace; run again to restore and remove it."""
        if any(t.station == "ALIGN" for t in self.engine.traces):
            self.engine.traces = [t for t in self.engine.traces if t.station != "ALIGN"]
            for tr in self.engine.traces:
                tr.display_offset = self._align_saved_offsets.get(self._offset_key(tr), 0.0)
            self._align_saved_offsets = {}
            self.refresh_views(reset_time=True)
            self._append_console_log("ALIGN removed: original trace positions restored.")
            return
        self._align_saved_offsets = {self._offset_key(t): t.display_offset for t in self.engine.traces}
        self.engine.execute_line(f"ALIGN {self.phase_panel.active_phase}")
        if not any(t.station == "ALIGN" for t in self.engine.traces):
            self._align_saved_offsets = {}
        self.refresh_views(reset_time=True)

    def action_sum_traces(self) -> None:
        self.engine.execute_line("SUM ALL")
        self.refresh_views()

    def action_pws_traces(self) -> None:
        self.engine.execute_line("PWS ALL 2.0")
        self.refresh_views()

    def action_open_fk_dialog(self) -> None:
        vis = self.engine.visible_traces()
        dlg = FKAnalysisDialog(
            vis,
            t_lo=self.main_canvas.mag_t_lo,
            t_hi=self.main_canvas.mag_t_hi,
            config=self.config,
            parent=self,
        )
        dlg.adoptParameters.connect(self._on_adopt_az_slo)
        dlg.adoptAndLocate.connect(self._on_adopt_az_slo_and_locate)
        dlg.fitDiagnosticReady.connect(self.fit_bridge.fitDiagnosticAvailable.emit)
        self._show_nonmodal(dlg)

    def _on_adopt_az_slo(self, az: float, slo: float) -> None:
        p = self.engine.current_params
        if az is not None:
            p.azimuth = az
        if slo is not None:
            p.slowness = slo
        az_fmt = f"{az:.1f}" if az is not None else "N/A"
        slo_fmt = f"{slo:.2f}" if slo is not None else "N/A"
        had_beam = any(t.station == "BEAM" for t in self.engine.traces)
        if had_beam:
            source_traces = [t for t in self.engine.traces if t.station != "BEAM"]
            self.engine.traces = source_traces
            if source_traces and az is not None and slo is not None:
                self.engine.execute_line(f"BEAM {az} {slo} CENTRE")
                self._append_console_log(f"Re-formed BEAM with updated slowness: BAZ={az_fmt} deg, Slowness={slo_fmt} s/deg.")
        self.refresh_views()
        self._append_console_log(f"Adopted FK parameters: BAZ={az_fmt} deg, Slowness={slo_fmt} s/deg.")
        self.status_bar.showMessage(f"Adopted FK BAZ={az_fmt}°, Slow={slo_fmt} s/deg", 5000)

    def _on_adopt_az_slo_and_locate(self, az: float, slo: float) -> None:
        p = self.engine.current_params
        if az is not None:
            p.azimuth = az
        if slo is not None:
            p.slowness = slo
        az_fmt = f"{az:.1f}" if az is not None else "N/A"
        slo_fmt = f"{slo:.2f}" if slo is not None else "N/A"
        had_beam = any(t.station == "BEAM" for t in self.engine.traces)
        if had_beam:
            source_traces = [t for t in self.engine.traces if t.station != "BEAM"]
            self.engine.traces = source_traces
            if source_traces and az is not None and slo is not None:
                self._exec_engine_checked(f"BEAM {az} {slo} CENTRE", title="Beam")
                self._append_console_log(f"Re-formed BEAM with updated slowness: BAZ={az_fmt} deg, Slowness={slo_fmt} s/deg.")
        self._exec_engine_checked(f"LOCATE /SLOWNESS {self.phase_panel.active_phase}", title="Locate by Slowness")
        self.refresh_views()

    def action_open_vespa_dialog(self) -> None:
        vis = self.engine.visible_traces()
        dlg = VespagramDialog(
            vis,
            azimuth_deg=self.engine.current_params.azimuth or 45.0,
            config=self.config,
            parent=self,
        )
        dlg.appendVespaTraces.connect(self._on_append_vespa_traces)
        self._show_nonmodal(dlg)

    def _on_append_vespa_traces(self, v_trs: List[Trace]) -> None:
        self.engine.traces.extend(v_trs)
        self.refresh_views()

    def action_locate_plane_wave(self) -> None:
        ph = self.phase_panel.active_phase
        # If fewer than 3 picks exist, auto-pick first so user gets immediate result
        vis = self.engine.visible_traces()
        picked = [t for t in vis if t.get_pick(ph) is not None]
        if len(picked) < 3 and len(vis) >= 3:
            msg = f"Fewer than 3 '{ph}' picks found: running the automatic picker first."
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 6000)
            self._exec_engine_checked("PICK ALL", title="Auto Pick")
        self._exec_engine_checked(f"LOCATE {ph}", title="Plane Wave Location")
        self.refresh_views()

    def action_locate_by_slowness(self) -> None:
        self._exec_engine_checked(f"LOCATE /SLOWNESS {self.phase_panel.active_phase}", title="Locate by Slowness")
        self.refresh_views()

    def action_locate_by_sp_diff(self) -> None:
        res = self.engine.locate_by_sp_diff()
        if res is not None:
            self.refresh_views()
            self._append_console_log(
                f"S-P Differential Location ({res['station']}): dt={res['dt_sp']:.2f}s -> Dist={res['distance']:.2f} deg"
            )
            return
        self._append_console_log("Pick both P (or Pg/Pn) and S (or Sg/Sn) on a trace to compute S-P distance.")

    def action_fit_depth_phase(self) -> None:
        res = self.engine.fit_depth_phase()
        if res is not None:
            self.refresh_views()
            self._append_console_log(
                f"Depth Phase Fit ({res['phase']}-P = {res['dt_dp']:.2f} s at {res['distance']:.1f} deg): "
                f"Depth = {res['depth']:.1f} +/- {res['d_depth']:.1f} km"
            )
            return
        self._append_console_log("Pick both 'P' and 'pP' (or 'sP') on a trace to invert for focal depth.")

    def action_open_network_loc_dialog(self) -> None:
        all_picks = [pk for tr in self.engine.visible_traces() for pk in tr.picks]
        dlg = NetworkLocationDialog(
            all_picks,
            initial_depth=self.engine.current_params.depth,
            config=self.config,
            parent=self,
        )
        dlg.adoptLocation.connect(self._on_adopt_network_loc)
        if hasattr(dlg, "diagnosticAvailable"):
            dlg.diagnosticAvailable.connect(self.fit_bridge.fitDiagnosticAvailable)
        self._show_nonmodal(dlg)

    def action_open_fit_diagnostic_window(self) -> None:
        """Open or show the dockable Fit Diagnostics Window and compute initial diagnostics."""
        self.fit_dock.show()
        self.fit_dock.raise_()
        all_picks = [pk for tr in self.engine.visible_traces() for pk in tr.picks if pk.onset and pk.station]
        if not all_picks:
            all_picks = [pk for tr in self.engine.traces for pk in tr.picks if pk.onset and pk.station]
        if len(all_picks) >= 3:
            req = FitModificationRequest(fit_type=FitType.HYPOCENTER_GEIGER)
            self._on_recompute_fit_requested(req)

    def _on_fit_diagnostic_received(self, diag: Any) -> None:
        self._last_fit_diagnostic = diag
        self.engine.last_fit_diagnostic = diag

    def _cancel_active_recompute_workers(self) -> int:
        alive: List[_RecomputeFitWorker] = []
        for w in getattr(self, "_recompute_workers", []):
            if w is not None and not sip.isdeleted(w) and w.isRunning():
                w.cancel()
                alive.append(w)
        if getattr(self, "_recompute_busy_active", False):
            self._recompute_busy_active = False
            self._end_busy()
        self._recompute_workers = alive
        self._recompute_seq = getattr(self, "_recompute_seq", 0) + 1
        return self._recompute_seq

    def _launch_recompute_worker(self, worker: _RecomputeFitWorker) -> _RecomputeFitWorker:
        worker.finishedDiagnostic.connect(self._on_recompute_worker_finished)
        worker.failed.connect(self._on_recompute_worker_failed)
        self._recompute_workers.append(worker)
        if not getattr(self, "_recompute_busy_active", False):
            self._recompute_busy_active = True
            self._begin_busy("Recomputing fit diagnostics...")
        worker.start()
        return worker

    def _on_recompute_worker_finished(self, seq_id: int, diag: Any, log_msg: str) -> None:
        if seq_id != self._recompute_seq:
            return
        if getattr(self, "_recompute_busy_active", False):
            self._recompute_busy_active = False
            self._end_busy()
        self.fit_bridge.fitDiagnosticAvailable.emit(diag)
        if log_msg:
            self._append_console_log(log_msg)
            self.status_bar.showMessage(log_msg, 4000)

    def _on_recompute_worker_failed(self, seq_id: int, err_msg: str) -> None:
        if seq_id != self._recompute_seq:
            return
        if getattr(self, "_recompute_busy_active", False):
            self._recompute_busy_active = False
            self._end_busy()
        if err_msg:
            self._append_console_log(err_msg)
            self.status_bar.showMessage(err_msg, 5000)

    def _build_focal_mech_diagnostic(
        self,
        fm_res: Dict[str, Any],
        all_pols: Sequence[Dict[str, Any]],
        excluded_upper: Optional[Set[str]] = None,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, Any], FitDiagnosticResult]:
        excl = excluded_upper or set()
        w_map = weight_overrides or {}
        pol_tuples = [
            (float(m["azimuth"]), float(m["takeoff"]), int(m["observed"]))
            for m in fm_res.get("station_matches", [])
        ]
        fm_payload: Dict[str, Any] = {
            **fm_res,
            "misfit": float(fm_res["misfit_ratio"]),
            "polarities": pol_tuples,
        }
        strike = float(fm_res["strike"])
        dip = float(fm_res["dip"])
        rake = float(fm_res["rake"])
        residuals: List[ResidualEntry] = []
        for obs in all_pols:
            sta = str(obs.get("station", ""))
            ph = str(obs.get("phase", "P") or "P")
            az = float(obs.get("azimuth", 0.0))
            ih = float(obs.get("takeoff", 0.0))
            obs_sign = 1.0 if int(obs.get("polarity", 0)) > 0 else -1.0
            rad = float(p_wave_radiation_pattern(strike, dip, rake, az, ih))
            pred_sign = 1.0 if rad >= 0.0 else -1.0
            ident = f"{sta}.{ph}.FM"
            is_excl = (
                sta.strip().upper() in excl
                or ident.upper() in excl
                or (ident in w_map and float(w_map[ident]) <= 0.0)
            )
            w = 0.0 if is_excl else float(w_map.get(ident, obs.get("weight", 1.0)))
            residuals.append(
                ResidualEntry(
                    identifier=ident,
                    observed=obs_sign,
                    calculated=pred_sign,
                    residual=obs_sign - pred_sign,
                    weight=w,
                    distance_deg=float(obs.get("distance_deg", 0.0)),
                    azimuth_deg=az,
                    station=sta,
                    phase=ph,
                    channel="Z",
                    is_outlier=(obs_sign != pred_sign),
                )
            )
        diag = FitDiagnosticResult(
            fit_type=FitType.FOCAL_MECHANISM,
            title="First-Motion Focal Mechanism Grid Search",
            converged=bool(fm_res.get("converged", True)),
            iterations=int(fm_res.get("n_obs", len(all_pols))),
            dof=max(0, int(fm_res.get("n_obs", len(all_pols))) - 3),
            rmse=float(fm_res.get("misfit_ratio", 0.0)),
            wrmse=float(fm_res.get("weighted_misfit", 0.0)),
            r_squared=max(0.0, 1.0 - float(fm_res.get("misfit_ratio", 0.0))),
            outlier_count=sum(1 for r in residuals if r.is_outlier),
            parameters={
                "strike": ParameterUncertainty(
                    name="Strike", value=strike, standard_error=0.0, confidence_interval_95=(strike, strike), unit="deg"
                ),
                "dip": ParameterUncertainty(
                    name="Dip", value=dip, standard_error=0.0, confidence_interval_95=(dip, dip), unit="deg"
                ),
                "rake": ParameterUncertainty(
                    name="Rake", value=rake, standard_error=0.0, confidence_interval_95=(rake, rake), unit="deg"
                ),
            },
            residuals=residuals,
            context_data={"focal_mechanism": fm_payload},
        )
        return fm_payload, diag

    def _on_recompute_fit_requested(self, req: Any) -> None:
        """Re-compute diagnostics / re-invert according to req.fit_type with exclusions and weights."""
        fit_type = getattr(req, "fit_type", FitType.HYPOCENTER_GEIGER)
        excluded = list(getattr(req, "excluded_identifiers", None) or [])
        excluded_upper = set()
        for x in excluded:
            s = str(x).strip().upper()
            if s:
                excluded_upper.add(s)
                excluded_upper.add(s.split(".")[0])
        weight_overrides = getattr(req, "weight_overrides", None)
        seq_id = self._cancel_active_recompute_workers()

        if fit_type in (
            FitType.HYPOCENTER_GEIGER,
            FitType.ORIGIN_TIME,
            FitType.TTFIT_SIMPLEX,
            FitType.SP_DIFF_DEPTH,
        ):
            from shm_app.seismology.array_loc import locate_network_geiger

            all_picks = [pk for tr in self.engine.visible_traces() for pk in tr.picks if pk.onset and pk.station]
            if not all_picks:
                all_picks = [pk for tr in self.engine.traces for pk in tr.picks if pk.onset and pk.station]
            if len(all_picks) < 3:
                self._append_console_log("Cannot re-invert: need at least 3 phase picks across network.")
                return

            try:
                with self._busy_cursor("Re-inverting hypocenter..."):
                    param, _station_residuals, diag = locate_network_geiger(
                        all_picks,
                        depth_km=self.engine.current_params.depth,
                        free_depth=(self.engine.current_params.depth_type == 3),
                        config=self.config,
                        ttt=self.engine.ttt,
                        excluded_stations=excluded or None,
                        weight_overrides=weight_overrides,
                        return_diagnostics=True,
                    )
                    self._on_adopt_network_loc(param)
                    self.fit_bridge.fitDiagnosticAvailable.emit(diag)
                self._append_console_log(
                    f"Re-inverted Geiger hypocenter: Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, "
                    f"Depth={param.depth:.1f} km, RMSE={diag.rmse:.3f} s"
                )
            except Exception as exc:
                self._append_console_log(f"Re-fit error: {exc}")
            return

        if fit_type in (FitType.FK_SEMBLANCE, FitType.PLANE_WAVE):
            vis = [
                tr.copy()
                for tr in self.engine.visible_traces()
                if tr.station.strip().upper() not in excluded_upper
            ]
            if len(vis) < 2:
                self._append_console_log("Cannot recompute FK/Array fit: need at least 2 active traces.")
                return
            worker = _RecomputeFitWorker(
                seq_id,
                fit_type,
                traces=vis,
                t_lo=self.main_canvas.mag_t_lo,
                t_hi=self.main_canvas.mag_t_hi,
                config=self.config,
                calib_table=getattr(self.engine, "calib_table", None),
                parent=self,
            )
            self._launch_recompute_worker(worker)
            return

        if fit_type == FitType.BRUNE_SPECTRUM:
            idx = self._resolve_visible_mag_trace_idx()
            if idx is None:
                self._append_console_log("Cannot recompute Brune spectrum: no visible traces loaded.")
                return
            tr = self.engine.traces[idx].copy()
            lo, hi = self._spectrum_window_for(tr)
            if hi - lo < 8 * tr.delta:
                lo, hi = 0.0, tr.duration
            dist_deg = tr.distance if tr.distance > 0 else (self.engine.current_params.distance or 1.0)
            worker = _RecomputeFitWorker(
                seq_id,
                fit_type,
                trace=tr,
                t_lo=lo,
                t_hi=hi,
                dist_deg=float(dist_deg),
                parent=self,
            )
            self._launch_recompute_worker(worker)
            return

        if fit_type == FitType.POLARIZATION_3C:
            try:
                from shm_app.processing.polarization import analyze_polarization_diagnostic

                prefer_sta = None
                idx = self._resolve_visible_mag_trace_idx()
                if idx is not None:
                    prefer_sta = self.engine.traces[idx].station
                with self._busy_cursor("Analyzing 3C polarization..."):
                    tz, tn, te = self.engine._find_zne_triplet(prefer_station=prefer_sta)
                    lo, hi = self._spectrum_window_for(tz)
                    if hi - lo <= 2.0 * max(1e-6, tz.delta):
                        lo, hi = 0.0, tz.duration
                    diag = analyze_polarization_diagnostic(tz, tn, te, t_lo=lo, t_hi=hi)
                    self.fit_bridge.fitDiagnosticAvailable.emit(diag)
            except Exception as exc:
                self._append_console_log(f"Polarization re-fit error: {exc}")
            return

        if fit_type == FitType.FOCAL_MECHANISM:
            p = self.engine.current_params
            vis = self.engine.visible_traces()
            try:
                with self._busy_cursor("Solving focal mechanism..."):
                    all_pols = extract_polarities_from_traces(
                        vis,
                        p.lat,
                        p.lon,
                        depth_km=p.depth,
                        config=self.config,
                        ttt=self.ttt,
                    )
                    if not all_pols:
                        self._append_console_log(
                            "Cannot recompute focal mechanism: need polarity picks ('c'/'d') with station coordinates."
                        )
                        return
                    w_map = dict(weight_overrides or {})
                    active_pols: List[Dict[str, Any]] = []
                    for obs in all_pols:
                        sta = str(obs.get("station", ""))
                        ph = str(obs.get("phase", "P") or "P")
                        ident = f"{sta}.{ph}.FM"
                        if sta.strip().upper() in excluded_upper or ident.upper() in excluded_upper:
                            continue
                        w_val = float(w_map.get(ident, obs.get("weight", 1.0)))
                        if w_val <= 0.0:
                            continue
                        obs_copy = dict(obs)
                        obs_copy["weight"] = w_val
                        active_pols.append(obs_copy)
                    if not active_pols:
                        self._append_console_log("Cannot recompute focal mechanism: all polarity observations are excluded.")
                        return
                    fm_res = solve_focal_mechanism(active_pols)
                    fm_payload, diag = self._build_focal_mech_diagnostic(
                        fm_res,
                        all_pols,
                        excluded_upper=excluded_upper,
                        weight_overrides=w_map,
                    )
                    self._last_focal_mechanism = fm_payload
                    self.engine.last_focal_mechanism = fm_payload
                    setattr(p, "focal_mechanism", fm_payload)
                    self.fit_bridge.fitDiagnosticAvailable.emit(diag)
                    self._append_console_log(
                        f"Recomputed Focal Mechanism ({fm_res['n_obs']} active polarities): "
                        f"Strike={fm_res['strike']:.1f}°, Dip={fm_res['dip']:.1f}°, Rake={fm_res['rake']:.1f}° "
                        f"(misfit={fm_res['misfit_ratio']:.2f})"
                    )
            except Exception as exc:
                self._append_console_log(f"Focal mechanism re-fit error: {exc}")
            return

        # Other domains (RESTITUTION_FIT, RECEIVER_FUNC, SURFACE_DISPERSION): re-emit cached result if matching
        last_diag = self._last_fit_diagnostic
        if last_diag is not None and getattr(last_diag, "fit_type", None) == fit_type:
            self.fit_bridge.fitDiagnosticAvailable.emit(last_diag)
        self._append_console_log(f"Re-fit handled for domain {getattr(fit_type, 'value', fit_type)}.")

    def _on_highlight_fit_trace(self, station: str) -> None:
        """Highlight and select matching station trace(s) in main canvas when clicked in diagnostic table."""
        st_upper = (station or "").strip().upper()
        if not st_upper:
            return
        selected_indices = set()
        first_idx: Optional[int] = None
        for idx, tr in enumerate(self.engine.traces):
            if tr.hidden:
                continue
            matched = tr.station.strip().upper() == st_upper
            tr.selected = matched
            if matched:
                selected_indices.add(idx)
                if first_idx is None:
                    first_idx = idx
        self.engine.selected_trace_indices = selected_indices
        if first_idx is not None:
            self.main_canvas.mag_trace_idx = first_idx
        self.main_canvas.invalidate_cache()
        self.main_canvas.update()
        self.refresh_views()

    def _on_zoom_to_fit_phase(self, station: str, phase: str) -> None:
        """Focus main trace & magnifier canvases onto the picked or theoretical phase for (station, phase)."""
        st_upper = (station or "").strip().upper()
        ph_name = (phase or "").strip()
        if not st_upper:
            return
        selected_indices = set()
        first_idx: Optional[int] = None
        target_tr: Optional[Trace] = None
        target_epoch: Optional[float] = None
        for idx, tr in enumerate(self.engine.traces):
            if tr.hidden:
                continue
            matched = tr.station.strip().upper() == st_upper
            tr.selected = matched
            if matched:
                selected_indices.add(idx)
                if first_idx is None:
                    first_idx = idx
                    target_tr = tr
                if ph_name:
                    pk = tr.get_pick(ph_name)
                    if pk is not None and np.isfinite(pk.onset_epoch):
                        first_idx = idx
                        target_tr = tr
                        target_epoch = float(pk.onset_epoch)
        if target_tr is not None and target_epoch is None and ph_name and ph_name in target_tr.theo_picks:
            ep = target_tr.theo_picks[ph_name]
            if np.isfinite(ep):
                target_epoch = float(ep)
        self.engine.selected_trace_indices = selected_indices
        if first_idx is not None:
            self.main_canvas.mag_trace_idx = first_idx
        if target_tr is not None and target_epoch is not None:
            ref_ep = self.main_canvas.reference_epoch()
            rel_t = target_epoch + target_tr.display_offset - ref_ep
            cur_span = self.main_canvas.mag_t_hi - self.main_canvas.mag_t_lo
            if not (np.isfinite(cur_span) and 1.0 <= cur_span <= 60.0):
                cur_span = max(6.0, min(20.0, 0.15 * max(10.0, target_tr.duration)))
            half_w = 0.5 * cur_span
            self.main_canvas.mag_t_lo = rel_t - half_w
            self.main_canvas.mag_t_hi = rel_t + half_w
            v_lo = self.main_canvas.view_t_lo
            v_hi = self.main_canvas.view_t_hi
            if v_lo is not None and v_hi is not None and (rel_t < v_lo or rel_t > v_hi):
                v_span = max(cur_span * 2.0, v_hi - v_lo)
                self.main_canvas.view_t_lo = rel_t - 0.5 * v_span
                self.main_canvas.view_t_hi = rel_t + 0.5 * v_span
                self.engine.view_window = (self.main_canvas.view_t_lo, self.main_canvas.view_t_hi)
                self._synced_view_window_version = self.engine._view_window_version
        self.refresh_views(reset_time=False)
        if ph_name:
            self.status_bar.showMessage(f"Zoomed to {station} {ph_name}", 4000)

    def _on_adopt_focal_mechanism(self, fm_payload: Any) -> None:
        """Persist focal mechanism parameters from FitDiagnosticWindow into workspace parameters."""
        if not isinstance(fm_payload, dict) or not fm_payload:
            return
        self._push_undo("Adopt focal mechanism")
        payload = dict(fm_payload)
        self._last_focal_mechanism = payload
        self.engine.last_focal_mechanism = payload
        p = self.engine.current_params
        setattr(p, "focal_mechanism", payload)
        self.refresh_views(reset_time=False)
        msg = (
            f"Adopted focal mechanism: Strike={float(payload.get('strike', 0.0)):.1f}°, "
            f"Dip={float(payload.get('dip', 0.0)):.1f}°, Rake={float(payload.get('rake', 0.0)):.1f}°"
        )
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)

    def _on_exclude_fit_station(self, station: str, exclude: bool) -> None:
        """Handle explicit station exclusion toggle from external plugins."""
        req = FitModificationRequest(
            fit_type=FitType.HYPOCENTER_GEIGER,
            excluded_identifiers=[station] if exclude else [],
        )
        self._on_recompute_fit_requested(req)

    def _on_adopt_network_loc(self, param: AnalysisParamSet) -> None:
        self.engine.param_sets[self.engine.active_set_index] = param
        orig_ep = parse_sh_time(param.origin) if param.origin else 0.0

        for tr in self.engine.traces:
            st = self.config.get_station(tr.station)
            if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                dist_i, _, _ = sphere_diff(st.lat, st.lon, param.lat, param.lon)
            else:
                dist_i = param.distance
            tr.distance = dist_i

            if orig_ep > 0:
                arrs = self.engine.ttt.theoretical_arrivals(dist_i, param.depth, orig_ep)
                tr.theo_picks = arrs
                for pk in tr.picks:
                    if param.residuals and f"{tr.station}:{pk.name}" in param.residuals:
                        pk.resid = param.residuals[f"{tr.station}:{pk.name}"]
                    elif pk.name in arrs:
                        pk.resid = pk.onset_epoch - arrs[pk.name]

        self.refresh_views()
        self._append_console_log(
            f"Adopted location: {param.lat:+.3f} N, {param.lon:+.3f} E, Depth {param.depth:.1f} km, "
            f"Origin {param.origin}. Updated station distances, theoretical arrivals, and residuals."
        )


    def _copy_param_set(self, src: int, dst: int) -> None:
        self.engine.param_sets[dst] = copy.deepcopy(self.engine.param_sets[src])
        self.refresh_views()

    def action_clear_params(self) -> None:
        self.engine.param_sets[self.engine.active_set_index] = AnalysisParamSet()
        for tr in self.engine.traces:
            tr.picks.clear()
            tr.theo_picks.clear()
        self._last_fit_diagnostic = None
        self._last_focal_mechanism = None
        self.engine.last_fit_diagnostic = None
        self.engine.last_focal_mechanism = None
        if hasattr(self, "fit_dock") and self.fit_dock is not None:
            self.fit_dock.clear_diagnostics()
        if hasattr(self, "fit_bridge") and self.fit_bridge is not None:
            self.fit_bridge.workspaceCleared.emit()
        self.refresh_views()

    def action_open_event_summary(self) -> None:
        all_picks = [pk for tr in self.engine.traces for pk in tr.picks]
        dlg = EventSummaryDialog(self.engine.current_params, all_picks, config=self.config, parent=self)
        dlg.openWorldMapRequested.connect(self._on_event_summary_open_world_map)
        self._show_nonmodal(dlg)

    def _on_event_summary_open_world_map(self) -> None:
        self.action_open_world_map()
        wmap = self._find_open_dialog(WorldMapDialog)
        if wmap is not None:
            wmap.set_view_preset("focus_event")

    def action_compute_magnitudes(self, prefer: str = "ALL") -> None:
        self._exec_engine_checked(f"MAG {self.phase_panel.active_phase}", title="Compute Magnitudes")
        self.refresh_views()


    def action_open_pmotion_dialog(self) -> None:
        prefer_sta = None
        if 0 <= self.main_canvas.mag_trace_idx < len(self.engine.traces):
            prefer_sta = self.engine.traces[self.main_canvas.mag_trace_idx].station
        try:
            tz, tn, te = self.engine._find_zne_triplet(prefer_station=prefer_sta)
        except ValueError:
            vis = self.engine.visible_traces()
            if len(vis) >= 3:
                tz, tn, te = vis[0], vis[1], vis[2]
            else:
                QMessageBox.information(self, "Particle Motion", "Need at least 3 traces (Z, N, E) for 3D particle motion.")
                return
        self._last_pmotion_station = tz.station
        lo, hi = self._spectrum_window_for(tz)
        if hi - lo <= 2.0 * max(1e-6, tz.delta):
            lo, hi = 0.0, tz.duration
        dlg = ParticleMotionDialog(tz, tn, te, t_lo=lo, t_hi=hi, parent=self)
        dlg.adoptAngles.connect(self._on_adopt_pmotion_angles)
        dlg.fitDiagnosticReady.connect(self.fit_bridge.fitDiagnosticAvailable.emit)
        dlg.emit_diagnostic()
        self._show_nonmodal(dlg)

    def _on_adopt_pmotion_angles(self, az: float, inc: float) -> None:
        self.engine.current_params.azimuth = az
        self.engine.current_params.incidence = inc
        target_sta = (self._last_pmotion_station or "").strip().upper()
        for tr in self.engine.traces:
            if target_sta:
                if tr.station.strip().upper() == target_sta and tr.comp in ("Z", "N", "E"):
                    tr.inci = inc
            elif tr.comp in ("Z", "N", "E"):
                tr.inci = inc
        self.refresh_views()
        self._append_console_log(f"Adopted Particle Motion angles: BAZ={az:.1f} deg, Incidence={inc:.1f} deg.")
        self.status_bar.showMessage(f"Adopted BAZ={az:.1f}°, Inc={inc:.1f}°", 5000)

    def action_apply_polfil_interactive(self) -> None:
        try:
            self.engine.execute_line("POLFIL 1.0 2.0 1")
            self.refresh_views()
        except Exception as exc:
            QMessageBox.warning(self, "Polarization Filter", str(exc))


    def action_identify_phases(self, dist_deg: Optional[float] = None) -> None:
        picked_traces = [tr for tr in self.engine.traces if tr.picks]
        if not picked_traces:
            self.status_bar.showMessage("Pick at least one phase before identifying phases.", 4000)
            self._append_console_log("IDENTIFY: no picked traces in workspace.")
            return

        p = self.engine.current_params
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        if dist_deg is None and not has_hypo and p.distance <= 0.0 and not any(tr.distance > 0.0 for tr in picked_traces):
            val, ok = QInputDialog.getDouble(
                self,
                "Identify Phases (IASP91)",
                "Epicentral Distance (degrees):",
                65.0,
                0.1,
                180.0,
                2,
            )
            if not ok:
                return
            dist_deg = float(val)

        try:
            self._push_undo("Identify phases")
            cmd = f"IDENTIFY /DIST={float(dist_deg):.4f}" if dist_deg is not None else "IDENTIFY"
            msg = self.engine.execute_line(cmd)
            self.refresh_views()
            if msg:
                self.status_bar.showMessage(msg.splitlines()[0], 5000)
        except Exception as exc:
            self._append_console_log(f"Phase identification error: {exc}")
            self.status_bar.showMessage(f"Identify error: {exc}", 5000)

    def action_scan_surface_waves(self) -> None:
        if not self.engine.traces:
            self.status_bar.showMessage("Load traces before scanning surface waves.", 4000)
            return
        try:
            self._push_undo("Scan surface waves (18-22s Ms)")
            with self._busy_cursor("Scanning surface waves (Ms)..."):
                ok, msg = self._exec_engine_checked("SURFMAG", title="Surface Wave Scan")
                self.refresh_views()
                if ok and msg:
                    self.status_bar.showMessage(msg, 6000)
        except Exception as exc:
            self._append_console_log(f"Surface wave scan error: {exc}")
            self.status_bar.showMessage(f"Surface wave scan error: {exc}", 5000)

    def action_solve_focal_mechanism(self) -> Optional[Dict[str, Any]]:
        p = self.engine.current_params
        vis = self.engine.visible_traces()
        try:
            with self._busy_cursor("Solving focal mechanism..."):
                pols = extract_polarities_from_traces(
                    vis,
                    p.lat,
                    p.lon,
                    depth_km=p.depth,
                    config=self.config,
                    ttt=self.ttt,
                )
                if not pols:
                    msg = "Need at least one pick with +/- polarity ('c'/'d') to solve focal mechanism."
                    self._append_console_log(msg)
                    self.status_bar.showMessage(msg, 5000)
                    return None
                fm_res = solve_focal_mechanism(pols)
                fm_payload, diag = self._build_focal_mech_diagnostic(fm_res, pols)
                self._last_focal_mechanism = fm_payload
                self.engine.last_focal_mechanism = fm_payload
                setattr(p, "focal_mechanism", fm_payload)
                self.fit_dock.show()
                self.fit_dock.raise_()
                self.fit_bridge.fitDiagnosticAvailable.emit(diag)
                self.fit_dock.display_focal_mechanism(fm_payload, switch_tab=True)
            msg = (
                f"Focal Mechanism ({fm_res['n_obs']} polarities): "
                f"Strike={fm_res['strike']:.1f}°, Dip={fm_res['dip']:.1f}°, Rake={fm_res['rake']:.1f}° "
                f"(misfit={fm_res['misfit_ratio']:.2f})"
            )
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 6000)
            return fm_payload
        except Exception as exc:
            self._append_console_log(f"Focal mechanism solver error: {exc}")
            self.status_bar.showMessage(f"Focal mechanism error: {exc}", 5000)
            return None

