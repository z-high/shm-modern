"""Headless unit tests for Phase 3 upgrades in shm_app/gui/dialogs_analysis.py (M-09, M-10, M-11, M-15, M-16, M-17)."""

from __future__ import annotations

import ast
import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.collections import LineCollection
from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(["shm_test_dialogs_p3"])

from shm_app.core.config_station import get_config
from shm_app.core.fit_models import FitDiagnosticResult, FitType
from shm_app.core.models import Trace
from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    FilterDialog,
    ParticleMotionDialog,
    SpectrumDialog,
)
from shm_app.processing.spectrum import source_spectrum_model


def _make_array_traces_with_outlier() -> list[Trace]:
    """Create 5 GRF array traces with a coherent 1.2 Hz pulse and 1 corrupted/poor channel."""
    cfg = get_config()
    stations = ["GRA1", "GRA2", "GRA3", "GRB1", "GRB2"]
    dt = 0.05
    n_samp = 400
    t_arr = np.arange(n_samp) * dt
    baz_rad = math.radians(45.0)
    slo_s_deg = 7.5
    deg_to_km = 111.19492664455873
    sx_km = (slo_s_deg / deg_to_km) * math.sin(baz_rad)
    sy_km = (slo_s_deg / deg_to_km) * math.cos(baz_rad)

    rel_coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
    traces: list[Trace] = []
    for idx, (st, (xk, yk, _el, _vp)) in enumerate(zip(stations, rel_coords)):
        # Wave arriving from baz_rad: arrival delay tau = -(xk*sx_km + yk*sy_km)
        t_center = 10.0 - (xk * sx_km + yk * sy_km)
        env = np.exp(-0.5 * ((t_arr - t_center) / 0.8) ** 2)
        sig = 2000.0 * env * np.sin(2.0 * math.pi * 1.2 * (t_arr - t_center))
        if idx == 4:
            # Make GRB2 inverted/uncorrelated so CC_m < 0.70 without swamping the beam
            rng = np.random.RandomState(42)
            sig = -0.8 * sig + 150.0 * rng.randn(n_samp)
        traces.append(Trace(data=sig, delta=dt, station=st, comp="Z", start_epoch=1000000.0))
    return traces


def _make_synthetic_brune_trace(
    fc: float = 1.5,
    omega0: float = 1000.0,
    t_star: float = 0.03,
    dt: float = 0.05,
) -> Trace:
    """Create a synthetic displacement trace containing pre-signal noise and a Brune pulse."""
    n_sig = 512
    freqs = np.fft.rfftfreq(n_sig, d=dt)
    amp = np.zeros_like(freqs)
    pos = freqs > 0.0
    amp[pos] = source_spectrum_model(freqs[pos], omega0=omega0, fc=fc, t_star=t_star, model="brune")
    # Linear phase shift to place pulse inside signal window (t=15..40s)
    t_shift = 5.0
    spec_c = amp * np.exp(-1j * 2.0 * math.pi * freqs * t_shift)
    pulse = np.fft.irfft(spec_c, n=n_sig) / dt

    # Pre-signal noise (0..15 s = 300 samples) + signal window (15..40.6 s = 512 samples)
    rng = np.random.RandomState(7)
    noise_pre = 2.0 * rng.randn(300)
    sig_plus_noise = pulse + 2.0 * rng.randn(n_sig)
    full_data = np.concatenate([noise_pre, sig_plus_noise])
    return Trace(data=full_data, delta=dt, station="GRA1", comp="Z", start_epoch=1000000.0, distance=1.0)


def _make_3c_synthetic_traces(baz_deg: float = 60.0, inc_deg: float = 30.0) -> tuple[Trace, Trace, Trace]:
    """Create 3C (Z, N, E) rectilinear P-wave traces with slight noise."""
    dt = 0.05
    n_samp = 300
    t = np.arange(n_samp) * dt
    pulse = 1000.0 * np.exp(-0.5 * ((t - 5.0) / 0.5) ** 2) * np.sin(2.0 * math.pi * 1.5 * (t - 5.0))

    inc_r = math.radians(inc_deg)
    baz_r = math.radians(baz_deg)
    # Upgoing P-wave particle motion: +Z away from source => horizontal radial = -sin(inc) * (cos(baz), sin(baz))
    ez = math.cos(inc_r)
    en = -math.sin(inc_r) * math.cos(baz_r)
    ee = -math.sin(inc_r) * math.sin(baz_r)

    rng = np.random.RandomState(123)
    z = ez * pulse + 15.0 * rng.randn(n_samp)
    n = en * pulse + 15.0 * rng.randn(n_samp)
    e = ee * pulse + 15.0 * rng.randn(n_samp)

    tr_z = Trace(data=z, delta=dt, station="GRA1", comp="Z", start_epoch=1000000.0)
    tr_n = Trace(data=n, delta=dt, station="GRA1", comp="N", start_epoch=1000000.0)
    tr_e = Trace(data=e, delta=dt, station="GRA1", comp="E", start_epoch=1000000.0)
    return tr_z, tr_n, tr_e


class TestPhase3Dialogs(unittest.TestCase):
    """Verify Phase 3 GUI dialog features, workers, plots, and diagnostic signals headlessly."""

    def test_no_forbidden_imports_in_dialogs_analysis(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(repo_root, "shm_app", "gui", "dialogs_analysis.py")
        with open(target, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=target)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("scipy", imported)
        self.assertNotIn("obspy", imported)

    def test_filter_dialog_order_16_autocut_and_zero_phase(self) -> None:
        cfg = get_config()
        dlg = FilterDialog(dt=0.05, config=cfg)
        try:
            self.assertEqual(dlg.but_order.minimum(), 1)
            self.assertEqual(dlg.but_order.maximum(), 16)
            self.assertTrue(hasattr(dlg, "chk_autocut"))
            self.assertTrue(hasattr(dlg, "chk_zero_phase"))

            dlg.tabs.setCurrentIndex(1)  # Butterworth Designer
            dlg.but_type.setCurrentIndex(0)  # BP
            dlg.but_f1.setValue(0.7)
            dlg.but_f2.setValue(2.5)
            dlg.but_order.setValue(16)
            self.assertEqual(dlg.but_order.value(), 16)
            self.assertEqual(dlg._current_expr(), "BUTTER BP 0.700 2.500 16")

            dlg.chk_autocut.setChecked(True)
            dlg.chk_zero_phase.setChecked(True)
            dlg.mode_combo.setCurrentIndex(2)  # RECURSIVE (IIR)
            dlg._on_apply()

            self.assertTrue(dlg.autocut)
            self.assertTrue(dlg.zero_phase)
            opts = dlg.get_filter_options()
            self.assertEqual(opts["filter_expr"], "BUTTER BP 0.700 2.500 16")
            self.assertEqual(opts["mode"], "REC")
            self.assertTrue(opts["autocut"])
            self.assertTrue(opts["zero_phase"])

            # Verify zero-phase Bode plot has zero phase across frequencies
            self.assertEqual(len(dlg.fig.axes), 2)
            phase_line = dlg.fig.axes[1].lines[0]
            np.testing.assert_allclose(phase_line.get_ydata(), 0.0, atol=1e-9)
        finally:
            dlg.close()

    def test_fk_dialog_calibration_toggle_arf_and_diagnostic_signal(self) -> None:
        cfg = get_config()
        traces = _make_array_traces_with_outlier()
        dlg = FKAnalysisDialog(traces, t_lo=5.0, t_hi=15.0, config=cfg)
        try:
            self.assertTrue(hasattr(dlg, "chk_apply_calib"))
            self.assertIsNotNone(dlg.calib_table)
            self.assertTrue(dlg.chk_apply_calib.isEnabled())
            self.assertTrue(dlg.chk_apply_calib.isChecked())
            self.assertIsNotNone(dlg.last_result)
            self.assertTrue(dlg.last_result.get("calib_applied", False))
            self.assertIn("Raw BAZ:", dlg.res_label.text())
            self.assertIn("Cal BAZ:", dlg.res_label.text())
            self.assertIn("GRB2", dlg.diag_label.text())

            emitted: list[FitDiagnosticResult] = []
            dlg.fitDiagnosticReady.connect(emitted.append)

            # Toggle calibration off -> must revert to raw slowness/BAZ and emit FitDiagnosticResult
            dlg.chk_apply_calib.setChecked(False)
            app.processEvents()
            self.assertEqual(len(emitted), 1)
            diag_off = emitted[-1]
            self.assertIsInstance(diag_off, FitDiagnosticResult)
            self.assertEqual(diag_off.fit_type, FitType.FK_SEMBLANCE)
            self.assertFalse(diag_off.context_data["calib_applied"])
            self.assertIn("arf_grid", diag_off.overlay_curves)
            self.assertGreater(diag_off.overlay_curves["arf_grid"].size, 0)
            self.assertIn("GRB2", diag_off.context_data["poor_channels"])

            raw_az = diag_off.context_data["raw_baz"]
            raw_slo = diag_off.context_data["raw_slowness"]
            self.assertAlmostEqual(dlg.last_result["azimuth"], raw_az, places=4)
            self.assertAlmostEqual(dlg.last_result["slowness"], raw_slo, places=4)

            # Toggle calibration back on + run async FKWorker
            dlg.chk_apply_calib.setChecked(True)
            app.processEvents()
            self.assertGreaterEqual(len(emitted), 2)
            self.assertTrue(emitted[-1].context_data["calib_applied"])

            dlg.run_fk(sync=False)
            self.assertIsNotNone(dlg.worker)
            self.assertTrue(dlg.worker.wait(5000))
            app.processEvents()
            self.assertGreaterEqual(len(emitted), 3)
            self.assertTrue(emitted[-1].context_data["calib_applied"])
        finally:
            dlg.close()

    def test_spectrum_dialog_brune_and_boatwright_inversion_and_signal(self) -> None:
        tr = _make_synthetic_brune_trace(fc=1.5, omega0=1000.0, t_star=0.03)
        dlg = SpectrumDialog(tr, t_lo=15.0, t_hi=40.0)
        try:
            self.assertTrue(hasattr(dlg, "brune_model_combo"))
            self.assertTrue(hasattr(dlg, "brune_tstar_spin"))
            self.assertTrue(hasattr(dlg, "dist_km_spin"))
            self.assertTrue(hasattr(dlg, "vel_kms_spin"))
            self.assertTrue(hasattr(dlg, "noise_lo_spin"))
            self.assertTrue(hasattr(dlg, "noise_hi_spin"))
            self.assertTrue(hasattr(dlg, "btn_fit_brune"))

            dlg.noise_lo_spin.setValue(0.0)
            dlg.noise_hi_spin.setValue(14.0)
            dlg.chk_use_noise.setChecked(True)
            dlg.dist_km_spin.setValue(120.0)
            dlg.vel_kms_spin.setValue(3.5)

            emitted: list[FitDiagnosticResult] = []
            dlg.fitDiagnosticReady.connect(emitted.append)

            # Async Brune inversion via button click
            dlg.brune_model_combo.setCurrentText("brune")
            dlg.btn_fit_brune.click()
            self.assertIsNotNone(dlg.brune_worker)
            self.assertTrue(dlg.brune_worker.wait(5000))
            app.processEvents()

            self.assertEqual(len(emitted), 1)
            diag = emitted[0]
            self.assertIsInstance(diag, FitDiagnosticResult)
            self.assertEqual(diag.fit_type, FitType.BRUNE_SPECTRUM)
            self.assertIn("fc", diag.context_data)
            self.assertIn("omega0", diag.context_data)
            self.assertIn("m0", diag.context_data)
            self.assertIn("mw", diag.context_data)
            self.assertIn("stress_drop_bar", diag.context_data)
            self.assertAlmostEqual(diag.context_data["fc"], 1.5, delta=0.5)

            lbl_txt = dlg.brune_label.text()
            for token in ("fc =", "Omega0 =", "t* =", "M0 =", "Mw =", "Delta_sigma ="):
                self.assertIn(token, lbl_txt)

            # Verify observed, fitted model, and noise floor lines exist on the plot
            ax = dlg.fig.axes[0]
            line_labels = [ln.get_label() for ln in ax.lines]
            self.assertTrue(any("Fit" in lbl for lbl in line_labels))
            self.assertIn("Noise Floor", line_labels)

            # Also test Boatwright model synchronously
            dlg.brune_model_combo.setCurrentText("boatwright")
            dlg.run_brune_inversion(sync=True)
            app.processEvents()
            self.assertEqual(len(emitted), 2)
            self.assertEqual(emitted[-1].context_data["model_type"], "boatwright")
        finally:
            dlg.close()

    def test_particle_motion_dialog_color_hodogram_cones_and_signal(self) -> None:
        tr_z, tr_n, tr_e = _make_3c_synthetic_traces(baz_deg=60.0, inc_deg=30.0)
        dlg = ParticleMotionDialog(tr_z, tr_n, tr_e, t_lo=3.0, t_hi=7.0)
        try:
            emitted: list[FitDiagnosticResult] = []
            dlg.fitDiagnosticReady.connect(emitted.append)

            dlg.replot()
            app.processEvents()

            self.assertEqual(len(emitted), 1)
            diag = emitted[0]
            self.assertIsInstance(diag, FitDiagnosticResult)
            self.assertEqual(diag.fit_type, FitType.POLARIZATION_3C)
            self.assertIn("rms_pol", diag.context_data)
            self.assertIn("sigma_theta_deg", diag.context_data)
            self.assertIn("sigma_phi_deg", diag.context_data)

            # Verify metrics readout contains RMS_pol and confidence cones
            info_txt = dlg.info_lbl.text()
            self.assertIn("RMS_pol", info_txt)
            self.assertIn("sigma_theta, sigma_phi", info_txt)

            # Verify all 3 subplots render time-colored LineCollection (Green -> Yellow -> Red)
            self.assertEqual(len(dlg.fig.axes), 3)
            for ax in dlg.fig.axes:
                lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
                self.assertEqual(len(lcs), 1)
                self.assertEqual(lcs[0].cmap.name, "gyr_hodogram")
                self.assertGreater(len(lcs[0].get_segments()), 10)

            # Toggle L-Q-T rotated mode and verify re-emission
            dlg.chk_rotated.setChecked(True)
            app.processEvents()
            self.assertEqual(len(emitted), 2)
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main()
