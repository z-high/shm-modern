"""Regression tests for M4 GUI core, crash resilience, data-loss prevention, session recovery, and fit diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

from shm_app.core.config_station import SHMConfig
from shm_app.core.fit_models import FitDiagnosticResult, FitType, ParameterUncertainty, ResidualEntry
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.main_window import SHMMainWindow, _quote_sh_arg
from shm_app.io.mseed import read_mseed_file, write_mseed_file


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_trace(
    station: str = "GRA1",
    comp: str = "Z",
    npts: int = 1200,
    dt: float = 0.05,
    start_epoch: float = 1_700_000_000.0,
    freq: float = 1.5,
) -> Trace:
    t = np.arange(npts, dtype=np.float64) * dt
    data = 500.0 * np.sin(2.0 * np.pi * freq * t) + 10.0 * np.cos(0.4 * t)
    return Trace(
        data=data,
        delta=dt,
        start_epoch=start_epoch,
        station=station,
        chan1="B",
        chan2="H",
        comp=comp,
        network="GR",
        locid="",
    )


class TestM4CrashResilienceAndPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _get_app()

    def test_autoload_example_missing_shm_root_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_root = os.path.join(tmpdir, "nonexistent_shm_root")
            cfg = SHMConfig(shm_root=missing_root)
            with (
                mock.patch.dict(os.environ, {"SHM_ROOT": missing_root}),
                mock.patch("shm_app.gui.main_window.get_config", return_value=cfg),
            ):
                win = SHMMainWindow(autoload_example=True)
                try:
                    self.assertEqual(len(win.engine.traces), 0)
                    self.assertIn("not found", win.console_log.toPlainText().lower())
                finally:
                    win.close()

    def test_interactive_load_example_missing_warns_without_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_root = os.path.join(tmpdir, "missing_examples")
            cfg = SHMConfig(shm_root=missing_root)
            with mock.patch("shm_app.gui.main_window.get_config", return_value=cfg):
                win = SHMMainWindow(autoload_example=False)
                try:
                    self.assertEqual(len(win._undo_stack), 0)
                    with mock.patch.object(QMessageBox, "warning") as warn_mock:
                        ok = win.load_example_dataset("example_mseed", interactive=True)
                    self.assertFalse(ok)
                    self.assertTrue(warn_mock.called)
                    self.assertEqual(len(win._undo_stack), 0)
                finally:
                    win.close()

    def test_action_cut_to_magnifier_empty_and_invalid_window(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            win.action_cut_to_magnifier()
            self.assertEqual(len(win._undo_stack), 0)

            tr = _make_trace("TST1")
            win.engine.traces = [tr]
            win.refresh_views(reset_time=True)
            win.main_canvas.mag_t_lo = 20.0
            win.main_canvas.mag_t_hi = 10.0
            with mock.patch.object(QMessageBox, "warning") as warn_mock:
                win.action_cut_to_magnifier()
            self.assertTrue(warn_mock.called)
            self.assertEqual(len(win._undo_stack), 0)
        finally:
            win.close()

    def test_canvases_sanitize_nan_inf_waveforms(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            tr = _make_trace("NAN1", npts=400)
            tr.data[10] = np.nan
            tr.data[20] = np.inf
            tr.data[30] = -np.inf
            win.engine.traces = [tr]
            win.resize(900, 650)
            win.refresh_views(reset_time=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                pix1 = win.main_canvas.grab()
                pix2 = win.mag_canvas.grab()
                snapped = win.main_canvas.snap_time_to_feature(tr, tr.start_epoch + 1.0)
            self.assertFalse(pix1.isNull())
            self.assertFalse(pix2.isNull())
            self.assertTrue(np.isfinite(snapped))
            runtime_msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
            self.assertFalse(any("mean of empty" in m.lower() or "invalid value" in m.lower() for m in runtime_msgs))
        finally:
            win.close()

    def test_quote_sh_arg_and_save_evt_and_clear_cancel_or_error(self) -> None:
        quoted = _quote_sh_arg("/tmp/dir with spaces/my event.evt")
        self.assertTrue(quoted.startswith('"') or quoted.startswith("'"))

        win = SHMMainWindow(autoload_example=False)
        try:
            tr = _make_trace("EVT1")
            tr.picks.append(PhasePick(name="P", onset=format_sh_time(tr.start_epoch + 5.0)))
            win.engine.traces = [tr]
            win.engine.current_params.lat = 42.5
            win.refresh_views(reset_time=True)

            # 1. Cancel dialog -> must NOT clear picks or params
            with mock.patch("shm_app.gui.main_window.QFileDialog.getSaveFileName", return_value=("", "")):
                win.action_save_evt_and_clear()
            self.assertEqual(len(tr.picks), 1)
            self.assertAlmostEqual(win.engine.current_params.lat, 42.5)

            # 2. SH Error during WRITEEVT -> must warn and NOT clear picks or params
            with (
                mock.patch(
                    "shm_app.gui.main_window.QFileDialog.getSaveFileName",
                    return_value=("/bad/out.evt", "Event Files (*.evt)"),
                ),
                mock.patch.object(win.engine, "execute_line", return_value="SH Error: permission denied"),
                mock.patch.object(QMessageBox, "warning") as warn_mock,
            ):
                win.action_save_evt_and_clear()
            self.assertTrue(warn_mock.called)
            self.assertEqual(len(tr.picks), 1)
            self.assertAlmostEqual(win.engine.current_params.lat, 42.5)

            # 3. Valid path WITH spaces -> succeeds AND clears picks/params
            with tempfile.TemporaryDirectory() as tmpdir:
                spaced_dir = Path(tmpdir) / "path with spaces"
                spaced_dir.mkdir()
                out_evt = spaced_dir / "saved event.evt"
                with mock.patch(
                    "shm_app.gui.main_window.QFileDialog.getSaveFileName",
                    return_value=(str(out_evt), "Event Files (*.evt)"),
                ):
                    win.action_save_evt_and_clear()
                self.assertTrue(out_evt.is_file())
                self.assertEqual(len(tr.picks), 0)
                self.assertAlmostEqual(win.engine.current_params.lat, 0.0)
        finally:
            win.close()

    def test_read_evt_cancel_and_error_no_spurious_undo_and_export_plot_warn(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            with mock.patch("shm_app.gui.main_window.QFileDialog.getOpenFileName", return_value=("", "")):
                win.action_read_evt()
            self.assertEqual(len(win._undo_stack), 0)

            with (
                mock.patch(
                    "shm_app.gui.main_window.QFileDialog.getOpenFileName",
                    return_value=("/no/such file.evt", "Event Files (*.evt)"),
                ),
                mock.patch.object(QMessageBox, "warning") as warn_mock,
            ):
                win.action_read_evt()
            self.assertTrue(warn_mock.called)
            self.assertEqual(len(win._undo_stack), 0)

            fake_pix = mock.MagicMock()
            fake_pix.save.return_value = False
            with (
                mock.patch(
                    "shm_app.gui.main_window.QFileDialog.getSaveFileName",
                    return_value=("/bad/img.png", "PNG Images (*.png)"),
                ),
                mock.patch.object(win.main_canvas, "grab", return_value=fake_pix),
                mock.patch.object(QMessageBox, "warning") as warn_mock,
            ):
                win.action_export_plot()
            self.assertTrue(warn_mock.called)
        finally:
            win.close()

    def test_sh_error_surfaced_on_processing_actions(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            with (
                mock.patch.object(win.engine, "execute_line", return_value="SH Error: no traces selected"),
                mock.patch.object(QMessageBox, "warning") as warn_mock,
            ):
                win.action_demean()
                win.action_auto_pick()
            self.assertEqual(warn_mock.call_count, 2)
        finally:
            win.close()


class TestM4SessionPersistenceAndIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _get_app()

    def test_offscreen_isolation_does_not_touch_cwd_session_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with mock.patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}):
                    os.environ.pop("SHM_SESSION_PATH", None)
                    win = SHMMainWindow(autoload_example=False)
                    win.action_create_synthetics()
                    win.autosave_session(clean_exit=False)
                    win.close()
                self.assertFalse((Path(tmpdir) / ".shm_session.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_session_restore_filtered_disk_trace_unfilter_and_time_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mseed_path = Path(tmpdir) / "disk_trace.mseed"
            sess_path = Path(tmpdir) / "session.json"

            seed_tr = _make_trace("ZZ99", npts=800, dt=0.05)
            write_mseed_file(mseed_path, [seed_tr])

            win1 = SHMMainWindow(autoload_example=False, session_path=sess_path)
            try:
                tr = read_mseed_file(mseed_path, config=win1.config)[0]
                raw_copy = tr.data.copy()
                win1.engine.traces = [tr]
                win1._register_unfiltered_traces(win1.engine.traces)
                win1.engine.apply_filter_to_traces("BP_1_5HZ", mode="FFT")
                win1.engine.time_window = (5.0, 25.0)
                win1.refresh_views()
                win1.autosave_session(clean_exit=True)
            finally:
                win1.close()

            win2 = SHMMainWindow(autoload_example=False, session_path=sess_path)
            try:
                self.assertTrue(win2.restore_session(sess_path))
                self.assertEqual(win2.engine.time_window, (5.0, 25.0))
                self.assertEqual(len(win2.engine.traces), 1)
                self.assertEqual(win2.engine.traces[0].filter_name, "BP_1_5HZ")
                win2.action_unfilter()
                self.assertEqual(win2.engine.traces[0].filter_name, "NONE")
                np.testing.assert_allclose(win2.engine.traces[0].data, raw_copy, atol=1e-4)

                # Disk-only (non-inline) filtered session replays filter + keeps raw backup
                raw_json = json.loads(sess_path.read_text(encoding="utf-8"))
                raw_json["traces"][0].pop("data", None)
                sess_disk_only = Path(tmpdir) / "disk_only.json"
                sess_disk_only.write_text(json.dumps(raw_json), encoding="utf-8")

                self.assertTrue(win2.restore_session(sess_disk_only))
                self.assertEqual(len(win2.engine.traces), 1)
                self.assertEqual(win2.engine.traces[0].filter_name, "BP_1_5HZ")
                self.assertFalse(np.allclose(win2.engine.traces[0].data, raw_copy))
                win2.action_unfilter()
                np.testing.assert_allclose(win2.engine.traces[0].data, raw_copy, atol=1e-4)

                # Missing disk source -> pruned & recorded warning
                mseed_path.unlink()
                self.assertTrue(win2.restore_session(sess_disk_only))
                self.assertEqual(len(win2.engine.traces), 0)
                self.assertTrue(win2._last_session_restore_warnings)
            finally:
                win2.close()

    def test_action_restore_session_warns_on_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sess_path = Path(tmpdir) / "does_not_exist.json"
            win = SHMMainWindow(autoload_example=False, session_path=sess_path)
            try:
                with mock.patch.object(QMessageBox, "warning") as warn_mock:
                    win.action_restore_session()
                self.assertTrue(warn_mock.called)
            finally:
                win.close()

    def test_action_clear_params_clears_fit_dock(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            diag = FitDiagnosticResult(
                fit_type=FitType.HYPOCENTER_GEIGER,
                title="Test Geiger",
                converged=True,
                iterations=3,
                dof=1,
                rmse=0.25,
                wrmse=0.25,
                residuals=[
                    ResidualEntry(
                        identifier="GRA1.P",
                        observed=10.0,
                        calculated=9.8,
                        residual=0.2,
                        station="GRA1",
                        phase="P",
                    )
                ],
            )
            win.fit_bridge.fitDiagnosticAvailable.emit(diag)
            self.assertIsNotNone(win.fit_dock.current_result)
            self.assertEqual(win.fit_dock.table_residuals.rowCount(), 1)

            win.action_clear_params()
            self.assertIsNone(win.fit_dock.current_result)
            self.assertEqual(win.fit_dock.table_residuals.rowCount(), 0)
        finally:
            win.close()

    def test_shortcuts_help_dialog_layout_and_size(self) -> None:
        from PyQt6.QtWidgets import QTextEdit

        win = SHMMainWindow(autoload_example=False)
        try:
            dlg = win._build_shortcuts_help_dialog()
            try:
                self.assertEqual(dlg.windowTitle(), "SHM Keyboard & Mouse Shortcuts Reference")
                self.assertGreaterEqual(dlg.minimumWidth(), 720)
                self.assertGreaterEqual(dlg.minimumHeight(), 520)
                self.assertGreaterEqual(dlg.width(), 800)
                self.assertGreaterEqual(dlg.height(), 600)
                viewers = dlg.findChildren(QTextEdit)
                self.assertEqual(len(viewers), 1)
                self.assertTrue(viewers[0].isReadOnly())
                plain = viewers[0].toPlainText()
                self.assertIn("Mouse Controls (Main Trace Canvas)", plain)
                self.assertIn("Mouse Controls (Magnifier Window)", plain)
                self.assertIn("Single-Key Accelerators", plain)
                self.assertIn("Ctrl Accelerators", plain)
                self.assertIn("Ctrl+Shift+Del", plain)
            finally:
                dlg.close()
        finally:
            win.close()

    def test_world_map_active_event_baz_points_station_to_epicenter(self) -> None:
        from shm_app.gui.world_map import WorldMapDialog

        win = SHMMainWindow(autoload_example=False)
        try:
            # Place active event due east of GRA1 (49.69N, 11.22E) at (49.762N, 12.879E)
            win.engine.current_params.lat = 49.762
            win.engine.current_params.lon = 12.879
            win.engine.current_params.depth = 33.0
            dlg = WorldMapDialog(win.engine, win.config, parent=win)
            try:
                active = next((e for e in dlg._event_items if e.is_active_param), None)
                self.assertIsNotNone(active)
                # Station->epicenter BAZ must be ~85.6 deg (East), NOT ~266.9 deg (West)
                self.assertAlmostEqual(active.baz_deg, 85.6, delta=2.0)
            finally:
                dlg.close()
        finally:
            win.close()

    def test_parameter_box_shows_mbb_and_mw_and_netloc_outlier_controls(self) -> None:
        from shm_app.gui.dialogs_location import NetworkLocationDialog
        from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff

        win = SHMMainWindow(autoload_example=False)
        try:
            p = win.engine.current_params
            p.mag_ml = 3.10
            p.mag_mb = 6.20
            p.mag_mbb = 7.45
            p.mag_ms = 7.10
            p.mag_mw = 7.40
            win.param_panel.refresh_from_model()
            txt = win.param_panel.mags_lbl.text()
            self.assertIn("mB:7.45", txt)
            self.assertIn("Mw:7.40", txt)

            # NetworkLocationDialog robust mode + one-click outlier exclusion
            cfg = win.config
            ttt = get_ttt_manager(cfg.inputs_dir)
            t0 = 1753831492.0
            stas = ["GRA1", "GRA2", "GRA3", "GRB1", "GRC1", "BFO", "BRG", "FUR", "WET"]
            picks = []
            for code in stas:
                st = cfg.get_station(code)
                d, _, _ = sphere_diff(st.lat, st.lon, 52.49, 160.24)
                tt = ttt.travel_time("P", d, 35.0)
                err = 16.0 if code == "WET" else 0.0
                picks.append(PhasePick(name="P", onset=format_sh_time(t0 + tt + err), station=code, comp="Z"))

            ndlg = NetworkLocationDialog(picks, initial_depth=33.0, config=cfg, parent=win)
            try:
                self.assertTrue(ndlg.chk_robust.isChecked())
                self.assertIsNotNone(ndlg.result_param)
                self.assertTrue(ndlg.btn_exclude_outliers.isEnabled())
                self.assertIn("WET", ndlg._last_outlier_stations)
                ndlg._on_exclude_outliers()
                self.assertIn("WET", ndlg._excluded_stations)
                self.assertTrue(ndlg.btn_reset_exclusions.isEnabled())
            finally:
                ndlg.close()
        finally:
            win.close()


class TestM4CanvasesPanelsAndFitDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _get_app()

    def test_hidden_mag_trace_resolution_right_press_and_state_key(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            tr0 = _make_trace("HID0")
            tr0.hidden = True
            tr1 = _make_trace("VIS1")
            win.engine.traces = [tr0, tr1]
            win.main_canvas.mag_trace_idx = 0
            win.resize(900, 650)
            win.refresh_views(reset_time=True)

            self.assertEqual(win.main_canvas.mag_trace_idx, 1)
            self.assertIs(win.mag_canvas.trace, tr1)

            k1 = win.main_canvas._current_state_key()
            tr1.display_offset = 3.5
            k2 = win.main_canvas._current_state_key()
            self.assertNotEqual(k1, k2)

            tr1.theo_picks["P"] = tr1.start_epoch + 12.0
            k3 = win.main_canvas._current_state_key()
            self.assertNotEqual(k2, k3)

            win.main_canvas.norm_mode = "SW"
            k4 = win.main_canvas._current_state_key()
            self.assertNotEqual(k3, k4)

            # Right-press on slot 0 (even if unhidden) without dragging must NOT switch mag_trace_idx
            tr0.hidden = False
            win.refresh_views(reset_time=False)
            win.main_canvas.mag_trace_idx = 1
            y0 = max(30.0, 0.25 * win.main_canvas.height())
            ev = QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPointF(200.0, y0),
                QPointF(200.0, y0),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier,
            )
            win.main_canvas.mousePressEvent(ev)
            self.assertEqual(win.main_canvas.mag_trace_idx, 1)
        finally:
            win.close()

    def test_key_press_focus_guard_and_magnifier_spacebar_and_error_bar_undo(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            tr = _make_trace("KEY1")
            win.engine.traces = [tr]
            win.resize(900, 650)
            win.show()
            win.refresh_views(reset_time=True)
            self.app.processEvents()

            # Spacebar in Magnifier places pick at crosshair
            win.mag_canvas._crosshair_x = 120.0
            sp_ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
            win.mag_canvas.keyPressEvent(sp_ev)
            self.assertEqual(len(tr.picks), 1)

            # Focused spinbox/lineedit prevents SHMMainWindow keyPressEvent from modifying pick quality
            win.phase_panel.qual_spin.setFocus()
            self.app.processEvents()
            old_q = tr.picks[0].quality
            ev_q9 = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_9, Qt.KeyboardModifier.NoModifier, "9")
            win.keyPressEvent(ev_q9)
            self.assertEqual(tr.picks[0].quality, old_q)

            # Error-bar drag start emits pickEditStarted and pushes undo
            before_undo = len(win._undo_stack)
            win.main_canvas.pickEditStarted.emit(0, tr.picks[0].name)
            self.assertEqual(len(win._undo_stack), before_undo + 1)
        finally:
            win.close()

    def test_panels_qual_range_single_emit_and_depth_rays_refresh(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            self.assertEqual(win.phase_panel.qual_spin.minimum(), 0)
            self.assertEqual(win.phase_panel.qual_spin.maximum(), 9)

            emits = []
            win.phase_panel.phaseConfigChanged.connect(lambda: emits.append(1))
            win.phase_panel.set_phase_name("Sn")
            self.assertEqual(len(emits), 1)

            tr = _make_trace("DEP1")
            win.engine.traces = [tr]
            win.engine.current_params.origin = "02-NOV-2003_12:00:00.000"
            win.refresh_views(reset_time=True)
            win.phase_panel.chk_depth_rays.setChecked(True)
            win.param_panel.depth_spin.setValue(45.0)
            win.param_panel.dist_spin.setValue(55.0)
            self.assertTrue(len(win.mag_canvas.depth_phase_cursors) > 0)
        finally:
            win.close()

    def test_zoom_to_phase_requested_and_focal_mechanism_fit_diagnostic(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            tr0 = _make_trace("AAA1", npts=2000, dt=0.05)
            tr1 = _make_trace("BBB2", npts=2000, dt=0.05)
            pk_ep = tr1.start_epoch + 42.0
            tr1.picks.append(PhasePick(name="P", onset=format_sh_time(pk_ep), sign=1, station="BBB2"))
            win.engine.traces = [tr0, tr1]
            win.resize(900, 650)
            win.refresh_views(reset_time=True)

            # zoomToPhaseRequested focuses BBB2 and centers magnifier near 42.0s
            win.fit_bridge.zoomToPhaseRequested.emit("BBB2", "P")
            self.assertEqual(win.main_canvas.mag_trace_idx, 1)
            self.assertLessEqual(win.main_canvas.mag_t_lo, 42.0)
            self.assertGreaterEqual(win.main_canvas.mag_t_hi, 42.0)

            # FitType.FOCAL_MECHANISM domain, tab, KPI, adopt, and weight validation
            fm_payload = {
                "strike": 120.0,
                "dip": 55.0,
                "rake": -40.0,
                "strike2": 236.0,
                "dip2": 58.0,
                "rake2": -138.0,
                "misfit": 0.125,
                "misfit_ratio": 0.125,
                "n_obs": 8,
                "polarities": [(10.0, 30.0, 1), (100.0, 35.0, -1)],
            }
            fm_diag = FitDiagnosticResult(
                fit_type=FitType.FOCAL_MECHANISM,
                title="First-Motion Focal Mechanism Grid Search",
                converged=True,
                iterations=8,
                dof=5,
                rmse=0.125,
                wrmse=0.125,
                parameters={
                    "strike": ParameterUncertainty(
                        name="Strike", value=120.0, standard_error=0.0, confidence_interval_95=(120.0, 120.0), unit="deg"
                    ),
                    "dip": ParameterUncertainty(
                        name="Dip", value=55.0, standard_error=0.0, confidence_interval_95=(55.0, 55.0), unit="deg"
                    ),
                    "rake": ParameterUncertainty(
                        name="Rake", value=-40.0, standard_error=0.0, confidence_interval_95=(-40.0, -40.0), unit="deg"
                    ),
                },
                residuals=[
                    ResidualEntry(
                        identifier="BBB2.P.FM",
                        observed=1.0,
                        calculated=1.0,
                        residual=0.0,
                        weight=1.0,
                        station="BBB2",
                        phase="P",
                    )
                ],
                context_data={"focal_mechanism": fm_payload},
            )
            win.fit_dock.set_fit_result(fm_diag)
            self.assertEqual(win.fit_dock.tabs.currentIndex(), 4)
            self.assertIn("misfit", win.fit_dock.kpi_rmse.lbl_value.text())
            self.assertTrue(win.fit_dock.btn_adopt.isEnabled())

            # Adopt emits adoptFocalMechanismRequested -> stored & undoable
            undo_before = len(win._undo_stack)
            win.fit_dock._on_adopt_clicked()
            self.assertEqual(len(win._undo_stack), undo_before + 1)
            self.assertEqual(win.engine.last_focal_mechanism["strike"], 120.0)

            # Weight validation rejects invalid and negative entries
            w_item = win.fit_dock.table_residuals.item(0, 7)
            self.assertIsNotNone(w_item)
            recomp_reqs = []
            win.fit_bridge.recomputeFitRequested.connect(recomp_reqs.append)
            w_item.setText("abc")
            self.assertEqual(w_item.text(), "1.00")
            self.assertEqual(len(recomp_reqs), 0)
            w_item.setText("-2.0")
            self.assertEqual(w_item.text(), "1.00")

            # Non-adoptable domain disables btn_adopt
            sp_diag = FitDiagnosticResult(
                fit_type=FitType.BRUNE_SPECTRUM,
                title="Brune",
                converged=True,
                iterations=10,
                dof=7,
                rmse=0.05,
                wrmse=0.05,
            )
            win.fit_dock.set_fit_result(sp_diag)
            self.assertFalse(win.fit_dock.btn_adopt.isEnabled())

            # Tab 4 PNG export uses canvas_beachball when displaying focal mechanism
            with tempfile.TemporaryDirectory() as tmpdir:
                out_png = os.path.join(tmpdir, "beachball.png")
                win.fit_dock.set_fit_result(fm_diag)
                with mock.patch(
                    "shm_app.gui.fit_dialog.QFileDialog.getSaveFileName",
                    return_value=(out_png, "PNG (*.png)"),
                ):
                    win.fit_dock._on_export_png()
                self.assertTrue(os.path.isfile(out_png) and os.path.getsize(out_png) > 0)

                # Write error surfaces QMessageBox.warning
                with (
                    mock.patch(
                        "shm_app.gui.fit_dialog.QFileDialog.getSaveFileName",
                        return_value=(os.path.join(tmpdir, "bad", "x.csv"), "CSV (*.csv)"),
                    ),
                    mock.patch.object(QMessageBox, "warning") as warn_mock,
                ):
                    win.fit_dock._on_export_csv()
                self.assertTrue(warn_mock.called)
        finally:
            win.close()

    def test_quakeml_hidden_fit_dock_does_not_force_tab_switch(self) -> None:
        win = SHMMainWindow(autoload_example=False)
        try:
            win.fit_dock.hide()
            win.fit_dock.tabs.setCurrentIndex(0)
            qml_sample = """<?xml version="1.0" encoding="utf-8"?>
<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2">
  <eventParameters publicID="smi:local/ep">
    <event publicID="smi:local/ev1">
      <origin publicID="smi:local/org1">
        <time><value>2023-11-15T12:00:00.000000Z</value></time>
        <latitude><value>40.1</value></latitude>
        <longitude><value>22.3</value></longitude>
        <depth><value>12000.0</value></depth>
      </origin>
      <focalMechanism publicID="smi:local/fm1">
        <nodalPlanes>
          <nodalPlane1>
            <strike><value>95.0</value></strike>
            <dip><value>60.0</value></dip>
            <rake><value>-30.0</value></rake>
          </nodalPlane1>
        </nodalPlanes>
      </focalMechanism>
    </event>
  </eventParameters>
</q:quakeml>"""
            with tempfile.TemporaryDirectory() as tmpdir:
                qml_path = os.path.join(tmpdir, "ev.qml")
                Path(qml_path).write_text(qml_sample, encoding="utf-8")
                win.action_import_quakeml(qml_path)
            self.assertFalse(win.fit_dock.isVisible())
            self.assertEqual(win.fit_dock.tabs.currentIndex(), 0)
            self.assertEqual(win._last_focal_mechanism["strike"], 95.0)
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
