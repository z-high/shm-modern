"""BGR EIDA waveform fetch and Live Event Mode actions for SHMMainWindow."""

from __future__ import annotations

from typing import List, Optional, Sequence

from shm_app.gui.help_texts import apply_tutorial_tooltips, register_tutorial_help
from shm_app.gui.world_map import WorldMapDialog
from shm_app.plugins.bgr_dialog import BGRFetchDialog
from shm_app.plugins.live_dialog import LiveEventDialog
from shm_app.plugins.live_service import build_analysis_param_set


class _MainWindowLiveMixin:
    """Live Event Mode, World Map synchronization, and BGR FDSNWS fetch actions."""

    def _workspace_stations(self) -> List[str]:
        selected = [tr for tr in self.engine.traces if tr.selected]
        source = selected if selected else self.engine.traces
        out: List[str] = []
        seen = set()
        for tr in source:
            st = (tr.station or "").strip().upper()
            if st and st not in seen:
                seen.add(st)
                out.append(st)
        return out

    def _open_bgr_fetch_prefilled(
        self,
        stations: Optional[Sequence[str]] = None,
        start_epoch: Optional[float] = None,
        duration_sec: Optional[float] = None,
    ) -> None:
        dlg = self._find_open_dialog(BGRFetchDialog)
        if dlg is None:
            view_win = None
            if self.engine.traces:
                lo, hi = self.main_canvas.active_time_span()
                if hi > lo:
                    view_win = (self.main_canvas.reference_epoch() + lo, hi - lo)
            dlg = BGRFetchDialog(
                config=self.config,
                view_window=view_win,
                origin_str=self.engine.current_params.origin,
                workspace_stations=self._workspace_stations(),
                parent=self,
            )
            dlg.waveformsDownloaded.connect(self._on_bgr_waveforms_downloaded)
            self._show_nonmodal(dlg)
        else:
            self._raise_nonmodal(dlg)

        if stations:
            dlg._apply_station_preset([str(s) for s in stations])
        if start_epoch is not None and duration_sec is not None:
            dlg.set_time_window(float(start_epoch), float(duration_sec))

    def action_open_bgr_fetch_dialog(self) -> None:
        self._open_bgr_fetch_prefilled()

    def _on_bgr_waveforms_downloaded(self, traces: list, append_mode: bool, msg: str) -> None:
        if not traces:
            return
        self._load_waveforms(traces, append_mode, "BGR Download", msg)

    # ------------------------------------------------------------------
    # Live Event Mode
    # ------------------------------------------------------------------

    def _is_workspace_idle_for_live(self) -> bool:
        if not self.engine.traces:
            return True
        return not any(pk.source != "t" for tr in self.engine.traces for pk in tr.picks)

    def action_toggle_live_mode(self) -> None:
        self.live_coordinator.set_enabled(not self.live_coordinator.is_enabled)

    def _on_live_enabled_changed(self, enabled: bool) -> None:
        if hasattr(self, "act_live_toggle"):
            self.act_live_toggle.blockSignals(True)
            self.act_live_toggle.setChecked(enabled)
            self.act_live_toggle.setText("LIVE: ON" if enabled else "Live: OFF")
            self.act_live_toggle.blockSignals(False)
        self.status_bar.showMessage(
            "Live Event Mode enabled (monitoring real-time seismic feeds)." if enabled else "Live Event Mode paused.",
            5000,
        )

    def action_open_live_event_dialog(self) -> None:
        existing = self._find_open_dialog(LiveEventDialog)
        if existing is not None:
            self._raise_nonmodal(existing)
            return
        dlg = LiveEventDialog(self.live_coordinator, parent=self)
        dlg.openInBGRRequested.connect(self._on_live_open_in_bgr)
        dlg.showOnMapRequested.connect(self._on_live_show_on_map)
        self._show_nonmodal(dlg)

    def _on_live_open_in_bgr(self, entry) -> None:
        if entry.plan is not None:
            self._open_bgr_fetch_prefilled(
                stations=[t.station for t in entry.plan.targets],
                start_epoch=entry.plan.start_epoch,
                duration_sec=entry.plan.duration_sec,
            )
        else:
            self._open_bgr_fetch_prefilled(start_epoch=entry.event.origin_epoch - 30.0, duration_sec=600.0)

    def _on_live_show_on_map(self, event_id: str) -> None:
        self.action_open_world_map()
        wmap = self._find_open_dialog(WorldMapDialog)
        if wmap is not None:
            wmap.focus_event_by_id(event_id)

    def _on_live_event_ready(self, event_id: str) -> None:
        entry = self.live_coordinator.entries.get(event_id)
        if entry is None:
            return
        self._pending_ready_event_id = event_id
        n_tr = len(entry.fetched_traces)
        label = f"Load Live M{entry.event.magnitude:.1f} {entry.evaluation.region_display[:28]} ({n_tr} tr)"
        self.live_notify_btn.setText(label)
        register_tutorial_help(
            self.live_notify_btn,
            f"Load Live {entry.event.event_id}",
            "live_notify",
            "",
            f"Click to load {entry.event.event_id} waveforms into SHM",
        )
        apply_tutorial_tooltips(self.live_notify_btn, self.tutorial_mode)
        self.live_notify_btn.setVisible(True)

    def _on_live_notify_clicked(self) -> None:
        eid = self._pending_ready_event_id
        self._pending_ready_event_id = None
        self.live_notify_btn.setVisible(False)
        if eid:
            self.live_coordinator.request_load_event(eid, auto_triggered=False)

    def _on_live_load_requested(self, event_id: str, auto_triggered: bool) -> None:
        entry = self.live_coordinator.entries.get(event_id)
        if entry is None or not entry.fetched_traces:
            return
        if self._pending_ready_event_id == event_id:
            self._pending_ready_event_id = None
            self.live_notify_btn.setVisible(False)

        traces = [tr.copy() for tr in entry.fetched_traces]
        mode_tag = "auto-loaded" if auto_triggered else "loaded"
        msg = (
            f"Live Event {entry.event.event_id} {mode_tag}: M{entry.event.magnitude:.1f} "
            f"{entry.evaluation.region_display} ({len(traces)} traces, Δ={entry.evaluation.dist_deg:.1f}°)"
        )
        self._load_waveforms(traces, append_mode=False, undo_label=f"Live M{entry.event.magnitude:.1f}", log_msg=msg)

        # Populate Analysis Parameter Set cleanly inside active slot
        pset = build_analysis_param_set(entry, self.config, self.live_coordinator.live_config)
        self.engine.param_sets[self.engine.active_set_index] = pset

        lcfg = self.live_coordinator.live_config
        filt_expr = (lcfg.auto_filter_expr or "").strip()
        if filt_expr:
            if filt_expr.upper().startswith(("BUTTER", "FILTER", "SIMULATE")):
                self.engine.execute_line(filt_expr)
            else:
                self.engine.execute_line(f"FILTER {filt_expr}")

        if lcfg.auto_theo_phases and pset.distance > 0.0:
            self.engine.execute_line(f"THEO {pset.distance} {pset.depth}")

        self.refresh_views(reset_time=True)

