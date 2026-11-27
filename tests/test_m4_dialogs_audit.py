"""Regression tests for M4 GUI dialogs, external-machine crash hardening, canvas teardown, and polish."""

from __future__ import annotations

import gc
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

app = QApplication.instance()
if app is None:
    app = QApplication(["shm_test_m4_dialogs_audit"])

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    FilterDialog,
    ParticleMotionDialog,
    SpectrumDialog,
    VespagramDialog,
)
from shm_app.gui.dialogs_io import ReadDataDialog
from shm_app.gui.dialogs_location import EventSummaryDialog, NetworkLocationDialog
from shm_app.gui.help_texts import HELP_TEXTS
from shm_app.gui.ui_helpers import cleanup_mpl_canvases
from shm_app.gui.world_map import WorldMapDialog
from shm_app.io.gse import write_gse2_file
from shm_app.io.mseed import read_from_sfd, write_mseed_file
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.plugins.bgr_dialog import BGRFetchDialog
from shm_app.plugins.live_dialog import LiveEventDialog
from shm_app.plugins.live_service import LiveEventCoordinator


def _make_3c_traces(station: str = "GRA1", n: int = 200, dt: float = 0.05) -> tuple[Trace, Trace, Trace]:
    t = np.arange(n) * dt
    sig = 500.0 * np.sin(2.0 * np.pi * 1.2 * t) * np.exp(-0.5 * ((t - 4.0) / 1.0) ** 2)
    t0 = parse_sh_time("02-NOV-2003_02:25:00.000")
    return (
        Trace(data=sig.copy(), delta=dt, station=station, chan1="B", chan2="H", comp="Z", start_epoch=t0),
        Trace(data=(0.5 * sig).copy(), delta=dt, station=station, chan1="B", chan2="H", comp="N", start_epoch=t0),
        Trace(data=(-0.3 * sig).copy(), delta=dt, station=station, chan1="B", chan2="H", comp="E", start_epoch=t0),
    )


class TestM4DialogsAndExternalHardening(unittest.TestCase):
    def setUp(self) -> None:
        self._sess_dir = tempfile.TemporaryDirectory()
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "SHM_SESSION_PATH": os.path.join(self._sess_dir.name, "session.json"),
                "SHM_NO_SESSION_PROMPT": "1",
            },
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        app.processEvents()
        self._env_patch.stop()
        self._sess_dir.cleanup()

    def test_read_data_dialog_external_missing_shm_root_and_full_duration_widgets(self) -> None:
        with tempfile.TemporaryDirectory() as fake_root:
            cfg = SHMConfig(shm_root=fake_root)
            dlg = ReadDataDialog(config=cfg)
            try:
                for btn in (dlg.btn_ex_mseed, dlg.btn_ex_grf_gse, dlg.btn_ex_grsn1, dlg.btn_ex_grsn2):
                    self.assertFalse(btn.isEnabled())
                    self.assertIn("SHM_ROOT", btn.toolTip())

                self.assertEqual(dlg.path_edit.text(), "")
                self.assertTrue(os.path.isdir(dlg._default_browse_dir()))

                # Full duration disables start_edit and len_spin
                self.assertTrue(dlg.chk_full_dur.isChecked())
                self.assertFalse(dlg.start_edit.isEnabled())
                self.assertFalse(dlg.len_spin.isEnabled())
                dlg.chk_full_dur.setChecked(False)
                self.assertTrue(dlg.start_edit.isEnabled())
                self.assertTrue(dlg.len_spin.isEnabled())

                # Clicking Read with empty path warns cleanly without crashing
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    dlg._execute_read()
                    mock_warn.assert_called_once()
                    self.assertNotEqual(dlg.result(), QDialog.DialogCode.Accepted)
            finally:
                dlg.close()
                app.processEvents()

    def test_read_data_dialog_error_resilience_sfd_fallback_dir_dispatch_and_station_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SHMConfig(shm_root=tmp)
            dlg = ReadDataDialog(config=cfg)
            try:
                # 1. Missing file warns cleanly
                dlg.path_edit.setText(os.path.join(tmp, "missing.QHD"))
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    dlg._execute_read()
                    mock_warn.assert_called_once()
                    self.assertNotEqual(dlg.result(), QDialog.DialogCode.Accepted)

                # 2. Invalid start time warns cleanly
                gse_dir = os.path.join(tmp, "gse_dir")
                os.makedirs(gse_dir, exist_ok=True)
                tz, _, _ = _make_3c_traces(station="CUSTOMSTA")
                gse_path = os.path.join(gse_dir, "custom.GSE")
                write_gse2_file(gse_path, [tz])

                dlg.path_edit.setText(gse_path)
                dlg.chk_full_dur.setChecked(False)
                dlg.start_edit.setText("BAD_TIME_TOKEN")
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    dlg._execute_read()
                    mock_warn.assert_called_once()
                dlg.chk_full_dur.setChecked(True)

                # 3. Directory mode dispatches .GSE files and unchecking GRA1 preserves non-GRF station CUSTOMSTA
                dlg.path_edit.setText(gse_dir)
                dlg.station_checks["GRA1"].setChecked(False)
                dlg._execute_read()
                self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)
                self.assertEqual(len(dlg.loaded_traces), 1)
                self.assertEqual(dlg.loaded_traces[0].station, "CUSTOMSTA")

                # 4. 0-trace filter match keeps dialog open and does not overwrite loaded_traces
                dlg.setResult(0)
                dlg.chan_combo.setCurrentText("LH")
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    dlg._execute_read()
                    mock_warn.assert_called_once()
                    self.assertNotEqual(dlg.result(), QDialog.DialogCode.Accepted)
                    self.assertEqual(len(dlg.loaded_traces), 1)
                dlg.chan_combo.setCurrentText("BH")

                # 5. Broken SFD warns in ReadDataDialog & raises clear FileNotFoundError in read_from_sfd
                broken_sfd = os.path.join(tmp, "broken.sfd")
                with open(broken_sfd, "w", encoding="ascii") as fp:
                    fp.write(
                        "s>gra1-bh-z   n>/nonexistent/machine/path/gra1.bhz "
                        "b>02-NOV-2003_02:25:00.000 e>02-NOV-2003_02:25:10.000 "
                        "r>     1 h>0 l>  512 o>      0 c>1.0000e+00 a>GR  \n"
                    )
                with self.assertRaises(FileNotFoundError):
                    read_from_sfd(broken_sfd, config=cfg)

                dlg.path_edit.setText(broken_sfd)
                dlg.station_checks["GRA1"].setChecked(True)
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    dlg._execute_read()
                    mock_warn.assert_called_once()
                    self.assertNotEqual(dlg.result(), QDialog.DialogCode.Accepted)

                # 6. SFD relative-dir fallback finds Mini-SEED file placed alongside SFD
                mseed_local = os.path.join(tmp, "gra1_local.bhz")
                tr_gra1, _, _ = _make_3c_traces(station="GRA1")
                write_mseed_file(mseed_local, [tr_gra1])
                fallback_sfd = os.path.join(tmp, "fallback.sfd")
                with open(fallback_sfd, "w", encoding="ascii") as fp:
                    fp.write(
                        "s>gra1-bh-z   n>/old_host/archive/gra1_local.bhz "
                        "b>02-NOV-2003_02:24:59.000 e>02-NOV-2003_02:25:15.000 "
                        "r>     1 h>0 l>  512 o>      0 c>1.0000e+00 a>GR  \n"
                    )
                trs_ok = read_from_sfd(fallback_sfd, config=cfg)
                self.assertEqual(len(trs_ok), 1)
                self.assertEqual(trs_ok[0].station, "GRA1")
            finally:
                dlg.close()
                app.processEvents()

    def test_canvas_teardown_cleanup_on_all_matplotlib_dialogs(self) -> None:
        cfg = get_config()
        tz, tn, te = _make_3c_traces("GRA1")
        tr2, _, _ = _make_3c_traces("GRA2")
        tr3, _, _ = _make_3c_traces("GRA3")
        eng = SHCommandEngine(config=cfg)
        eng.traces = [tz, tr2, tr3]

        dialogs = [
            FilterDialog(dt=0.05, config=cfg),
            FKAnalysisDialog([tz, tr2, tr3], t_lo=1.0, t_hi=8.0, config=cfg),
            VespagramDialog([tz, tr2, tr3], azimuth_deg=45.0, config=cfg),
            SpectrumDialog(tz, t_lo=1.0, t_hi=8.0),
            ParticleMotionDialog(tz, tn, te, t_lo=1.0, t_hi=8.0),
            EventSummaryDialog(AnalysisParamSet(lat=50.0, lon=11.0), [], config=cfg),
            WorldMapDialog(eng, config=cfg),
        ]
        for dlg in dialogs:
            dlg.show()
            for attr in ("canvas", "canvas_tt"):
                cv = getattr(dlg, attr, None)
                if cv is not None:
                    cv.draw_idle()
            dlg.close()
            app.processEvents()
            for attr in ("canvas", "canvas_tt"):
                cv = getattr(dlg, attr, None)
                if cv is not None:
                    self.assertFalse(getattr(cv, "_draw_pending", False))
            self.assertEqual(getattr(dlg, "_mpl_cids", []), [])
        gc.collect()
        app.processEvents()
        cleanup_mpl_canvases(None)

    def test_vespagram_fk_spectrum_and_particle_motion_polish(self) -> None:
        cfg = get_config()
        tz, tn, te = _make_3c_traces("GRA1")
        tr2, _, _ = _make_3c_traces("GRA2")
        tr3, _, _ = _make_3c_traces("GRA3")

        # 1. Vespagram error display & slowness validation
        vdlg = VespagramDialog([tz, tr2, tr3], config=cfg)
        try:
            vdlg._on_vespa_error("synthetic error")
            self.assertIn("synthetic error", vdlg.status_lbl.text())
            self.assertIn("#f85149", vdlg.status_lbl.styleSheet())

            vdlg.slo_lo_spin.setValue(10.0)
            vdlg.slo_hi_spin.setValue(5.0)
            vdlg.run_vespa(sync=True)
            self.assertIn("must be less than", vdlg.status_lbl.text())
        finally:
            vdlg.close()
            app.processEvents()

        # 2. FKAnalysisDialog frequency range validation & sync exception guard
        fk_dlg = FKAnalysisDialog([tz, tr2, tr3], config=cfg)
        try:
            fk_dlg.flo_spin.setValue(4.0)
            fk_dlg.fhi_spin.setValue(1.0)
            fk_dlg.run_fk(sync=True)
            self.assertIn("must be less than", fk_dlg.res_label.text())
            self.assertIn("#f85149", fk_dlg.res_label.styleSheet())
        finally:
            fk_dlg.close()
            app.processEvents()

        # 3. SpectrumDialog band validation & tutorial help coverage
        sdlg = SpectrumDialog(tz, t_lo=1.0, t_hi=8.0)
        try:
            for w in (
                sdlg.chk_fit_tstar,
                sdlg.chk_use_noise,
                sdlg.noise_lo_spin,
                sdlg.noise_hi_spin,
                sdlg.btn_fit_brune,
                sdlg.btn_clear,
            ):
                self.assertIsNotNone(w.property("_shm_tutorial_meta"), f"Missing tutorial help on {w}")
            self.assertIn("dlg_spec_method", HELP_TEXTS)

            sdlg.fit_f1.setValue(6.0)
            sdlg.fit_f2.setValue(2.0)
            app.processEvents()
            self.assertIn("Fit Band Error", sdlg.fit_label.text())
            sdlg.run_brune_inversion(sync=True)
            self.assertIn("must be less than", sdlg.brune_label.text())
        finally:
            sdlg.close()
            app.processEvents()

        # 4. ParticleMotionDialog: spinboxes do not trigger compute_mdir or emit_diagnostic
        pdlg = ParticleMotionDialog(tz, tn, te, t_lo=1.0, t_hi=8.0)
        try:
            emitted = []
            pdlg.fitDiagnosticReady.connect(emitted.append)
            with mock.patch("shm_app.gui.dialogs_analysis.compute_mdir") as mock_mdir:
                pdlg.az_spin.setValue(120.0)
                pdlg.inc_spin.setValue(40.0)
                pdlg.chk_rotated.setChecked(True)
                app.processEvents()
                mock_mdir.assert_not_called()
            self.assertEqual(emitted, [])

            pdlg.btn_emit_diag.click()
            app.processEvents()
            self.assertEqual(len(emitted), 1)
        finally:
            pdlg.close()
            app.processEvents()

    def test_network_location_event_summary_world_map_bgr_and_live_validation(self) -> None:
        cfg = get_config()
        # 1. NetworkLocationDialog error styling and stale row clearing
        bad_picks = [PhasePick(name="P", onset="02-NOV-2003_02:25:00.000", station="GRA1")]
        ndlg = NetworkLocationDialog(bad_picks, config=cfg)
        try:
            self.assertEqual(ndlg.table.rowCount(), 0)
            self.assertIn("#f85149", ndlg.summary_lbl.styleSheet())
            self.assertIn("Location Error", ndlg.summary_lbl.text())
        finally:
            ndlg.close()
            app.processEvents()

        # 2. EventSummaryDialog read-only text_edit & _save_evt try/except
        pset = AnalysisParamSet(lat=50.1, lon=11.5, depth=10.0, origin="02-NOV-2003_02:25:00.000")
        ev_dlg = EventSummaryDialog(pset, bad_picks, config=cfg)
        try:
            self.assertTrue(ev_dlg.text_edit.isReadOnly())
            with mock.patch.object(QFileDialog := __import__("PyQt6.QtWidgets", fromlist=["QFileDialog"]).QFileDialog, "getSaveFileName", return_value=("/nonexistent_dir_xyz/test.evt", "Event Files (*.evt)")):
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    ev_dlg._save_evt()
                    mock_warn.assert_called_once()
        finally:
            ev_dlg.close()
            app.processEvents()

        # 3. WorldMapDialog parses pset.origin into origin_epoch for BGR prefill
        eng = SHCommandEngine(config=cfg)
        eng.param_sets[eng.active_set_index] = pset
        wmap = WorldMapDialog(eng, config=cfg)
        try:
            active_ev = next(e for e in wmap._event_items if e.is_active_param)
            expected_ep = parse_sh_time("02-NOV-2003_02:25:00.000")
            self.assertAlmostEqual(active_ev.origin_epoch, expected_ep, places=3)

            emitted_bgr = []
            wmap.openBGRFetchRequested.connect(emitted_bgr.append)
            wmap.focus_event_by_id("ACTIVE_PARAM_SET")
            wmap._on_open_selected_in_bgr()
            self.assertEqual(len(emitted_bgr), 1)
            self.assertAlmostEqual(emitted_bgr[0]["start_epoch"], expected_ep - 30.0, places=3)
            self.assertEqual(emitted_bgr[0]["duration_sec"], 600.0)
        finally:
            wmap.close()
            app.processEvents()

        # 4. BGRFetchDialog origin timestamp error & URL/out_dir validation
        bgr = BGRFetchDialog(config=cfg)
        try:
            bgr._origin_str = "NOT_A_VALID_DATE"
            with mock.patch.object(QMessageBox, "warning") as mock_warn:
                bgr._fill_from_origin()
                mock_warn.assert_called_once()
            self.assertIn("Invalid origin timestamp", bgr.status_label.text())

            bgr._apply_station_preset(["GRA1"])
            bgr.url_edit.setText("ftp://invalid-scheme.example.org")
            with mock.patch.object(QMessageBox, "warning") as mock_warn:
                bgr._start_download()
                mock_warn.assert_called_once()
                self.assertIsNone(bgr._worker)

            bgr.url_edit.setText("https://eida.bgr.de/fdsnws/dataselect/1/query")
            with tempfile.NamedTemporaryFile() as file_as_dir:
                bgr.out_dir_edit.setText(file_as_dir.name)
                with mock.patch.object(QMessageBox, "warning") as mock_warn:
                    bgr._start_download()
                    mock_warn.assert_called_once()
                    self.assertIsNone(bgr._worker)
        finally:
            bgr.close()
            app.processEvents()

        # 5. LiveEventDialog unknown ref station & empty components warning
        coord = LiveEventCoordinator(shm_config=cfg)
        ldlg = LiveEventDialog(coord)
        try:
            ldlg.edit_ref_station.setText("UNKNOWN_STA_XYZ")
            ldlg.chk_z.setChecked(False)
            ldlg.chk_n.setChecked(False)
            ldlg.chk_e.setChecked(False)
            with mock.patch.object(QMessageBox, "warning") as mock_warn:
                ldlg._apply_config_from_widgets()
                mock_warn.assert_called_once()
            self.assertTrue(ldlg.chk_z.isChecked())
            self.assertIn("UNKNOWN_STA_XYZ", ldlg.lbl_footer_hint.text())
        finally:
            ldlg.close()
            coord.shutdown()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
