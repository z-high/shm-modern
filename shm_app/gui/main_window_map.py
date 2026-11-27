"""World Map dialog and trace/event synchronization mixin for SHMMainWindow."""

from __future__ import annotations

from shm_app.gui.world_map import WorldMapDialog


class _MainWindowMapMixin:
    """Interactive World Map actions and signal synchronization."""

    def action_open_world_map(self) -> None:
        existing = self._find_open_dialog(WorldMapDialog)
        if existing is not None:
            existing.refresh_data(preserve_view=True)
            self._raise_nonmodal(existing)
            return
        dlg = WorldMapDialog(
            self.engine,
            config=self.config,
            live_coordinator=self.live_coordinator,
            parent=self,
        )
        dlg.selectStationTracesRequested.connect(self._on_world_map_select_traces)
        dlg.openBGRFetchRequested.connect(self._on_world_map_open_bgr)
        dlg.adoptEventParamsRequested.connect(self._on_world_map_adopt_event)
        dlg.loadLiveEventRequested.connect(lambda eid: self.live_coordinator.request_load_event(eid, auto_triggered=False))
        dlg.openLiveMonitorRequested.connect(self.action_open_live_event_dialog)
        self._show_nonmodal(dlg)

    def _sync_open_world_map(self) -> None:
        dlg = self._find_open_dialog(WorldMapDialog)
        if dlg is not None:
            dlg.refresh_data(preserve_view=True)

    def _on_world_map_select_traces(self, station_code: str) -> None:
        target = (station_code or "").strip().upper()
        matched = 0
        for tr in self.engine.traces:
            hit = (tr.station or "").strip().upper() == target
            tr.selected = hit
            if hit:
                matched += 1
        self.main_canvas.invalidate_cache()
        self.status_bar.showMessage(f"Selected {matched} trace(s) for station {target}.", 4000)

    def _on_world_map_open_bgr(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._open_bgr_fetch_prefilled(
            stations=payload.get("stations") or [],
            start_epoch=payload.get("start_epoch"),
            duration_sec=payload.get("duration_sec"),
        )

    def _on_world_map_adopt_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        pset = self.engine.param_sets[self.engine.active_set_index]
        pset.lat = float(payload.get("lat", pset.lat))
        pset.lon = float(payload.get("lon", pset.lon))
        pset.depth = float(payload.get("depth", pset.depth))
        if payload.get("origin"):
            pset.origin = str(payload["origin"])
        pset.distance = float(payload.get("distance", pset.distance))
        pset.azimuth = float(payload.get("azimuth", pset.azimuth))
        if payload.get("region_name"):
            pset.region_name = str(payload["region_name"])
        mag = float(payload.get("magnitude", 0.0) or 0.0)
        if mag > 0.0:
            pset.mag_mb = mag
        if payload.get("source"):
            pset.source = str(payload["source"])
        self.param_panel.refresh_from_model()
        self._sync_open_world_map()
        self.status_bar.showMessage(
            f"Adopted event into Parameter Box: {pset.region_name or 'Unknown'} (Δ={pset.distance:.1f}°, BAZ={pset.azimuth:.1f}°)",
            5000,
        )
