"""File import/export, vector bulletin reporting, and SFD index generation for SHMMainWindow."""

from __future__ import annotations

import copy
import sys
import numpy as np

from PyQt6 import sip
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from shm_app.gui.main_window_utils import _quote_sh_arg
from shm_app.gui.reporting import BulletinWorker
from shm_app.io.geojson import write_geojson
from shm_app.io.quakeml import parse_quakeml, write_quakeml


class _MainWindowIOMixin:
    """Waveform/event I/O, vector bulletin export, and SFD index generation."""

    def action_read_evt(self) -> None:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getOpenFileName(
            self, "Read .evt Event File", self.config.shm_root, "Event Files (*.evt);;All Files (*)"
        )
        if not fpath:
            return
        snap = self._snapshot("Read event file")
        try:
            ok, _out = self._exec_engine_checked(
                f"READEVT {_quote_sh_arg(fpath)}",
                title="Read .evt Event File",
            )
            if ok:
                self._undo_stack.append(snap)
                del self._undo_stack[: -self.UNDO_LIMIT]
                self._redo_stack.clear()
                self._update_undo_actions()
                self._register_unfiltered_traces(self.engine.traces)
                self.refresh_views()
        except Exception as exc:
            QMessageBox.warning(self, "Read .evt Event File", str(exc))
            self.status_bar.showMessage(f"Failed to read .evt file: {exc}", 5000)


    def action_export_plot(self) -> None:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(self, "Export Waveform Canvas Screenshot", "shm_waveforms.png", "PNG Images (*.png)")
        if not fpath:
            return
        try:
            pix = self.main_canvas.grab()
            if not pix.save(fpath):
                QMessageBox.warning(self, "Export Waveform Screenshot", f"Failed to write image file:\n{fpath}")
                self.status_bar.showMessage(f"Failed to export plot image to {fpath}", 5000)
                return
            self._append_console_log(f"Saved waveform plot image to {fpath}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Waveform Screenshot", str(exc))
            self.status_bar.showMessage(f"Failed to export plot image: {exc}", 5000)


    def action_save_evt(self) -> bool:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(self, "Save Event Parameters (.evt)", "shm_event.evt", "Event Files (*.evt)")
        if not fpath:
            return False
        try:
            ok, _out = self._exec_engine_checked(
                f"WRITEEVT {_quote_sh_arg(fpath)}",
                title="Save Event Parameters (.evt)",
            )
            return bool(ok)
        except Exception as exc:
            QMessageBox.warning(self, "Save Event Parameters (.evt)", str(exc))
            self.status_bar.showMessage(f"Failed to save .evt file: {exc}", 5000)
            return False

    def action_save_evt_and_clear(self) -> None:
        if self.action_save_evt():
            self.action_clear_params()

    def action_export_gse2(self) -> None:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(self, "Export GSE2 File", "export.GSE", "GSE Files (*.GSE *.gse)")
        if fpath:
            try:
                self._exec_engine_checked(f"WRITEGSE {_quote_sh_arg(fpath)}", title="Export GSE2")
            except Exception as exc:
                QMessageBox.warning(self, "Export GSE2", str(exc))

    def action_export_mseed(self) -> None:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(self, "Export Mini-SEED File", "export.mseed", "Mini-SEED Files (*.mseed *.bhz)")
        if fpath:
            try:
                self._exec_engine_checked(f"WRITESEED {_quote_sh_arg(fpath)}", title="Export Mini-SEED")
            except Exception as exc:
                QMessageBox.warning(self, "Export Mini-SEED", str(exc))

    def action_export_qfile(self) -> None:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(self, "Export SH Q-File", "export.QHD", "Q-Files (*.QHD)")
        if fpath:
            try:
                self._exec_engine_checked(f"WRITE {_quote_sh_arg(fpath)}", title="Export SH Q-File")
            except Exception as exc:
                QMessageBox.warning(self, "Export SH Q-File", str(exc))

    def action_export_ascii(self) -> None:
        fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(self, "Export ASCII Waveforms", "export.asc", "ASCII Files (*.asc *.txt)")
        if fpath:
            try:
                self._exec_engine_checked(f"WRITEA {_quote_sh_arg(fpath)}", title="Export ASCII")
            except Exception as exc:
                QMessageBox.warning(self, "Export ASCII", str(exc))


    def action_import_stationxml(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getOpenFileName(
                self,
                "Import FDSN StationXML",
                self.config.shm_root,
                "StationXML Files (*.xml *.stationxml);;All Files (*)",
            )
        if not fpath:
            return
        try:
            inv = self.config.load_stationxml(fpath)
            self.engine.last_stationxml_inventory = inv
            for tr in self.engine.traces:
                ch_meta = self.config.get_channel_metadata(
                    tr.network, tr.station, tr.locid, tr.channel3, tr.start_epoch
                )
                if ch_meta is not None:
                    if ch_meta.latitude != 0.0 or ch_meta.longitude != 0.0:
                        tr.lat = float(ch_meta.latitude)
                        tr.lon = float(ch_meta.longitude)
                    if ch_meta.sensor_description:
                        tr.instype = ch_meta.sensor_description
                    tr.extra_headers["azimuth"] = float(ch_meta.azimuth)
                    tr.extra_headers["dip"] = float(ch_meta.dip)
                    tr.extra_headers["station_lat"] = float(tr.lat)
                    tr.extra_headers["station_lon"] = float(tr.lon)
                    tr.extra_headers["station_elev"] = float(ch_meta.elevation)
                    sxml_calib = self.config._stationxml_calib_nm_per_count(ch_meta)
                    if sxml_calib is not None and sxml_calib > 0:
                        tr.calib = float(sxml_calib)
                else:
                    st = self.config.get_station(tr.station)
                    if st is not None and (st.lat != 0.0 or st.lon != 0.0) and tr.lat == 0.0 and tr.lon == 0.0:
                        tr.lat = st.lat
                        tr.lon = st.lon
            self.refresh_views()
            n_sta = len(inv.stations)
            n_cha = sum(len(s.channels) for s in inv.stations)
            msg = f"Imported StationXML ({n_sta} station(s), {n_cha} channel(s)) from {fpath}."
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"StationXML import error: {exc}")
            QMessageBox.warning(self, "StationXML Import Error", str(exc))

    def action_import_quakeml(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getOpenFileName(
                self,
                "Import QuakeML Event",
                self.config.shm_root,
                "QuakeML Files (*.xml *.qml);;All Files (*)",
            )
        if not fpath:
            return
        try:
            self._push_undo("Import QuakeML")
            qml = parse_quakeml(fpath)
            param = qml["params"]
            picks = qml.get("phase_picks", [])
            fms = qml.get("focal_mechanisms", [])
            if fms:
                fm0 = fms[0]
                np1 = fm0.get("nodalPlane1")
                if isinstance(np1, dict):
                    fm_dict: Dict[str, Any] = {
                        "strike": float(np1.get("strike", 0.0)),
                        "dip": float(np1.get("dip", 0.0)),
                        "rake": float(np1.get("rake", 0.0)),
                    }
                    np2 = fm0.get("nodalPlane2")
                    if isinstance(np2, dict):
                        fm_dict["strike2"] = float(np2.get("strike", 0.0))
                        fm_dict["dip2"] = float(np2.get("dip", 0.0))
                        fm_dict["rake2"] = float(np2.get("rake", 0.0))
                    self.engine.last_focal_mechanism = fm_dict
                    self._last_focal_mechanism = fm_dict
                    setattr(param, "focal_mechanism", fm_dict)
                    self.fit_dock.display_focal_mechanism(
                        fm_dict,
                        switch_tab=bool(hasattr(self, "fit_dock") and self.fit_dock.isVisible()),
                    )

            self.engine.param_sets[self.engine.active_set_index] = param
            for p in picks:
                for tr in self.engine.traces:
                    if tr.station.upper() == p.station.upper() and (not p.comp or tr.comp.upper() == p.comp.upper()):
                        tr.add_or_replace_pick(p)
                        break
                else:
                    for tr in self.engine.traces:
                        if tr.station.upper() == p.station.upper():
                            tr.add_or_replace_pick(p)
                            break
            self.refresh_views()
            msg = (
                f"Imported QuakeML {qml.get('schema_version', '1.2')} from {fpath}: "
                f"Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, {len(picks)} pick(s)."
            )
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"QuakeML import error: {exc}")
            QMessageBox.warning(self, "QuakeML Import Error", str(exc))

    def action_export_quakeml(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(
                self,
                "Export QuakeML 1.2",
                "event.quakeml.xml",
                "QuakeML XML (*.xml *.qml)",
            )
        if not fpath:
            return
        try:
            fm = self._last_focal_mechanism or getattr(self.engine, "last_focal_mechanism", None)
            write_quakeml(
                fpath,
                self.engine.traces,
                self.engine.current_params,
                focal_mechanism=fm,
            )
            msg = f"Exported QuakeML 1.2 to {fpath}"
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"QuakeML export error: {exc}")
            QMessageBox.warning(self, "QuakeML Export Error", str(exc))

    def action_export_geojson(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(
                self,
                "Export GeoJSON (RFC 7946)",
                "event.geojson",
                "GeoJSON Files (*.geojson *.json)",
            )
        if not fpath:
            return
        try:
            fm = self._last_focal_mechanism or getattr(self.engine, "last_focal_mechanism", None)
            write_geojson(
                fpath,
                self.engine.traces,
                self.engine.current_params,
                focal_mechanism=fm,
            )
            msg = f"Exported RFC 7946 GeoJSON to {fpath}"
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"GeoJSON export error: {exc}")
            QMessageBox.warning(self, "GeoJSON Export Error", str(exc))

    def action_export_bulletin(
        self,
        fpath: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> Optional[BulletinWorker]:
        if fpath is None:
            fpath, selected_filter = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getSaveFileName(
                self,
                "Export Observatory Bulletin (PDF / SVG)",
                "observatory_bulletin.pdf",
                "Vector PDF (*.pdf);;Vector SVG (*.svg)",
            )
            if fpath and fmt is None and "SVG" in selected_filter and not fpath.lower().endswith(".svg"):
                fpath = fpath + ".svg"
        if not fpath:
            return None

        self._bulletin_workers = [
            w for w in self._bulletin_workers if w is not None and not sip.isdeleted(w) and w.isRunning()
        ]
        fm = self._last_focal_mechanism or getattr(self.engine, "last_focal_mechanism", None)
        fit_res = self._last_fit_diagnostic or getattr(self.engine, "last_fit_diagnostic", None)
        worker = BulletinWorker(
            output_path=fpath,
            traces=[t.copy() for t in self.engine.traces],
            params=copy.deepcopy(self.engine.current_params),
            fit_result=fit_res,
            focal_mech=fm,
            format=fmt,
            config=self.config,
            parent=self,
        )
        worker.finishedReport.connect(self._on_bulletin_finished)
        worker.failed.connect(self._on_bulletin_failed)
        self._bulletin_workers.append(worker)
        worker.start()
        self.status_bar.showMessage(f"Generating observatory bulletin: {fpath}...", 4000)
        return worker

    def _on_bulletin_finished(self, out_path: str) -> None:
        msg = f"Generated vector observatory bulletin: {out_path}"
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)

    def _on_bulletin_failed(self, err: str) -> None:
        msg = f"Observatory bulletin export failed: {err}"
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)

    def action_generate_sfd(self) -> None:
        dpath = getattr(sys.modules.get("shm_app.gui.main_window"), "QFileDialog", QFileDialog).getExistingDirectory(self, "Select Directory of Mini-SEED Files", self.config.data_examples_dir)
        if dpath:
            try:
                self._exec_engine_checked(f"SFD {_quote_sh_arg(dpath)}", title="Generate SFD Index")
            except Exception as exc:
                QMessageBox.warning(self, "Generate SFD Index", str(exc))

