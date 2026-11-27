"""Headless unit tests for Phase 3/4 FitDiagnosticWindow Tabs 3-6 & M-25 Beachball."""

from __future__ import annotations

import ast
import math
import os
import sys
import unittest
import warnings

import numpy as np

from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(["shm_test_fit_dialog_p3"])

from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
from matplotlib.text import Annotation

from shm_app.core.config_station import get_config
from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.core.fit_models import FitDiagnosticResult, FitType, ParameterUncertainty, ResidualEntry
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.gui.fit_dialog import FitDiagnosticWindow
from shm_app.processing.polarization import analyze_polarization_diagnostic
from shm_app.processing.spectrum import (
    build_brune_fit_diagnostic,
    invert_brune_spectrum,
    source_spectrum_model,
)
from shm_app.processing.trace_ops import create_synthetic_trace
from shm_app.seismology.array_loc import (
    SlownessCalibrationTable,
    beam_delays_sec,
    build_fk_fit_diagnostic,
    compute_array_response_function,
    compute_beam_diagnostics,
    compute_fk_analysis,
)
from shm_app.seismology.focal_mechanism import p_wave_radiation_pattern, solve_focal_mechanism


def _make_brune_trace(
    omega0: float = 1200.0,
    fc: float = 2.2,
    t_star: float = 0.04,
    dt: float = 0.02,
    duration_sec: float = 40.0,
) -> Trace:
    n = int(round(duration_sec / dt))
    n_sig = int(round(12.0 / dt))
    freqs = np.fft.rfftfreq(n_sig, d=dt)
    amp = np.zeros_like(freqs)
    pos = freqs > 0.0
    amp[pos] = source_spectrum_model(freqs[pos], omega0=omega0, fc=fc, t_star=t_star, model="brune")
    # Zero-phase source pulse centered at 5.0 s inside signal window [15, 27] s
    phases = np.exp(-2j * math.pi * freqs * 5.0)
    pulse = np.fft.irfft(amp * phases / dt, n=n_sig)

    rng = np.random.RandomState(7)
    data = rng.randn(n) * 1.5
    i_sig = int(round(15.0 / dt))
    data[i_sig : i_sig + n_sig] += pulse
    return Trace(data=data, delta=dt, station="GRA1", comp="Z")


class TestFitDialogPhase3And4(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FitDiagnosticBridge()
        self.win = FitDiagnosticWindow(bridge=self.bridge)

    def tearDown(self) -> None:
        self.win.close()
        app.processEvents()

    def test_tab3_array_fk_real_data_with_arf_calibration_and_poor_channel(self) -> None:
        cfg = get_config()
        stations = ["GRA1", "GRA2", "GRA3", "GRA4", "GRB1", "GRB2", "GRB3", "GRB4"]
        coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
        x_km = [c[0] for c in coords]
        y_km = [c[1] for c in coords]
        delays = beam_delays_sec(x_km, y_km, azimuth_deg=110.0, slowness_s_deg=7.5)
        base_ep = parse_sh_time("02-NOV-2003_02:25:00.000")

        traces = []
        for st_code, d_sec in zip(stations, delays):
            tr = create_synthetic_trace(
                "KUEPPER",
                length_sec=40.0,
                dt=0.05,
                pos_sec=15.0 - float(d_sec),
                ampl=200.0,
                width=1.0,
            )
            tr.station = st_code
            tr.start_epoch = base_ep
            traces.append(tr)

        # Invert polarity on GRB2 so its coherence CC_m < 0.70 (highlighted in red)
        traces[5].data = -traces[5].data

        calib_tbl = SlownessCalibrationTable.from_text(
            "# slowness_in azimuth_in d_slowness d_azimuth\n"
            "7.5 110.0 -0.6 +4.5\n"
        )
        fk_res = compute_fk_analysis(
            traces,
            t_lo=10.0,
            t_hi=22.0,
            f_lo=0.4,
            f_hi=2.5,
            max_slowness=12.0,
            grid_points=31,
            config=cfg,
            calib_table=calib_tbl,
            apply_calibration=True,
        )
        beam_diag = compute_beam_diagnostics(
            traces, fk_res["azimuth"], fk_res["slowness"], t_lo=12.0, t_hi=22.0, config=cfg
        )
        arf_res = compute_array_response_function(x_km, y_km, slowness_max=12.0, n_grid=31, freq_hz=1.0)
        fk_diag = build_fk_fit_diagnostic(fk_res, beam_diag=beam_diag, arf_result=arf_res)

        self.win.display_result(fk_diag)
        app.processEvents()

        self.assertEqual(self.win.tabs.currentIndex(), 2)
        self.assertEqual(self.win.table_residuals.rowCount(), len(traces))
        self.assertGreaterEqual(len(self.win.fig_fk.axes), 1)
        ax_fk = self.win.fig_fk.axes[0]
        self.assertGreaterEqual(len(ax_fk.images), 1)
        self.assertIn("Calibrated", ax_fk.get_title())
        # Verify calibration arrow annotation is drawn
        arrows = [c for c in ax_fk.get_children() if isinstance(c, Annotation)]
        self.assertGreaterEqual(len(arrows), 1)

        # Verify beam waveform + CC_m bar subplot highlighting poor elements in red
        self.assertEqual(len(self.win.fig_beam_res.axes), 2)
        ax_cc = self.win.fig_beam_res.axes[1]
        self.assertEqual(len(ax_cc.patches), len(traces))
        bar_hexes = [to_hex(p.get_facecolor()).lower() for p in ax_cc.patches]
        self.assertIn("#f85149", bar_hexes)
        self.assertIn("#3fb950", bar_hexes)

        # Verify PLANE_WAVE routes to Tab 3 cleanly too
        pw_diag = build_fk_fit_diagnostic(fk_res, beam_diag=beam_diag, arf_result=arf_res, fit_type=FitType.PLANE_WAVE)
        self.win.display_result(pw_diag)
        self.assertEqual(self.win.tabs.currentIndex(), 2)

    def test_tab4_brune_spectrum_real_inversion_and_misfit_grid(self) -> None:
        tr = _make_brune_trace(omega0=1200.0, fc=2.2, t_star=0.04)
        inv = invert_brune_spectrum(
            tr,
            t_lo=14.0,
            t_hi=28.0,
            noise_lo=0.0,
            noise_hi=10.0,
            model="brune",
            f_lo=0.2,
            f_hi=12.0,
        )
        brune_diag = build_brune_fit_diagnostic(inv, station="GRA1", phase="P")

        self.win.display_result(brune_diag)
        app.processEvents()

        self.assertEqual(self.win.tabs.currentIndex(), 3)
        self.assertGreaterEqual(len(self.win.fig_spec.axes), 1)
        ax_s = self.win.fig_spec.axes[0]
        # Observed, Model, Noise + vertical fc line
        self.assertGreaterEqual(len(ax_s.lines), 3)
        # Annotation box should report real fc, t*, M0, Mw, r0, stress drop
        ann_texts = "\n".join(t.get_text() for t in ax_s.texts)
        self.assertIn("fc =", ann_texts)
        self.assertIn("Mw =", ann_texts)
        self.assertIn("Δσ =", ann_texts)

        # Misfit contour subplot should have optimal marker
        self.assertGreaterEqual(len(self.win.fig_misfit.axes), 1)
        ax_m = self.win.fig_misfit.axes[0]
        self.assertGreaterEqual(len(ax_m.collections), 1)
        self.assertGreaterEqual(len(ax_m.lines), 1)

        lbl_txt = self.win.lbl_source_params.text()
        self.assertNotIn("M0=--", lbl_txt)
        self.assertIn(f"fc={inv['fc']:.3f} Hz", lbl_txt)
        self.assertIn("log10", self.win.kpi_rmse.lbl_value.text())

    def test_tab5_polarization_gradient_hodograms_and_m25_beachball(self) -> None:
        dt = 0.05
        t = np.arange(400) * dt
        baz_true = 65.0
        inc_true = 30.0
        az_r = math.radians(baz_true)
        inc_r = math.radians(inc_true)

        main_sig = 80.0 * np.sin(2.0 * math.pi * 1.0 * t)
        orth_q = 18.0 * np.cos(2.0 * math.pi * 1.0 * t)
        orth_t = 6.0 * np.cos(2.0 * math.pi * 2.0 * t)

        z_dat = main_sig * math.cos(inc_r) + orth_q * math.sin(inc_r)
        n_dat = (
            -main_sig * math.sin(inc_r) * math.cos(az_r)
            + orth_q * math.cos(inc_r) * math.cos(az_r)
            + orth_t * math.sin(az_r)
        )
        e_dat = (
            -main_sig * math.sin(inc_r) * math.sin(az_r)
            + orth_q * math.cos(inc_r) * math.sin(az_r)
            - orth_t * math.cos(az_r)
        )
        tz = Trace(data=z_dat, delta=dt, station="BFO", comp="Z")
        tn = Trace(data=n_dat, delta=dt, station="BFO", comp="N")
        te = Trace(data=e_dat, delta=dt, station="BFO", comp="E")

        pol_diag = analyze_polarization_diagnostic(tz, tn, te, t_lo=2.0, t_hi=12.0)

        # Solve a real focal mechanism and attach to context_data
        obs = []
        for az in np.arange(0.0, 360.0, 30.0):
            for ih in (25.0, 55.0, 75.0):
                rad = float(p_wave_radiation_pattern(40.0, 55.0, -90.0, ih, az))
                if abs(rad) >= 0.12:
                    obs.append((float(az), float(ih), 1 if rad > 0 else -1))
        fm_sol = solve_focal_mechanism(obs, grid_step_deg=15.0, refine=True)
        pol_diag.context_data["focal_mechanism"] = fm_sol

        self.win.display_result(pol_diag)
        app.processEvents()

        self.assertEqual(self.win.tabs.currentIndex(), 4)
        self.assertEqual(len(self.win.fig_pol.axes), 2)
        for ax in self.win.fig_pol.axes:
            lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
            self.assertEqual(len(lcs), 1)
            self.assertGreater(len(lcs[0].get_segments()), 10)

        pol_txt = self.win.lbl_pol_metrics.text()
        self.assertIn("Trajectory RMS", pol_txt)
        self.assertIn(f"{pol_diag.context_data['baz']:.1f}°", pol_txt)
        self.assertIn(f"{pol_diag.context_data['sigma_theta_deg']:.2f}°", pol_txt)

        # Beachball canvas rendered inside Tab 5
        self.assertEqual(len(self.win.fig_beachball.axes), 1)
        ax_bb = self.win.fig_beachball.axes[0]
        self.assertGreaterEqual(len(ax_bb.collections), 1)
        self.assertGreaterEqual(len(ax_bb.lines), 2)  # two nodal planes
        self.assertIn("NP1", self.win.lbl_fm_summary.text())
        self.assertIn("Misfit:", self.win.lbl_fm_summary.text())

        # Direct push via display_focal_mechanism on a fresh window
        win2 = FitDiagnosticWindow()
        try:
            win2.display_focal_mechanism(fm_sol)
            app.processEvents()
            self.assertEqual(win2.tabs.currentIndex(), 4)
            self.assertGreaterEqual(len(win2.fig_beachball.axes[0].lines), 2)
            self.assertIn(f"{fm_sol['strike']:.1f}°", win2.lbl_fm_summary.text())
        finally:
            win2.close()
            app.processEvents()

    def test_tab6_restitution_receiver_func_and_dispersion(self) -> None:
        freqs = np.logspace(-2, 1.5, 60)
        obs_amp = 1000.0 / (1.0 + (freqs / 0.5) ** 2)
        resp_amp = 1500.0 * (freqs / 0.0083) ** 2 / np.sqrt((1.0 - (freqs / 0.0083) ** 2) ** 2 + 2.0 * (freqs / 0.0083) ** 2)
        snr_db = 20.0 * np.exp(-((np.log10(freqs) - 0.0) ** 2) / 0.8) + 2.0

        rest_res = FitDiagnosticResult(
            fit_type=FitType.RESTITUTION_FIT,
            title="STS-2 Restitution Fit",
            converged=True,
            iterations=1,
            dof=58,
            rmse=0.04,
            wrmse=0.04,
            chi2_reduced=0.95,
            overlay_curves={
                "freqs": freqs,
                "obs_amp": obs_amp,
                "resp_amp": resp_amp,
                "snr_freqs": freqs,
                "snr_db": snr_db,
            },
            context_data={
                "station": "GRA1",
                "status": "OK",
                "passband_snr_db": 22.0,
                "waterlevel": 0.01,
            },
        )
        self.win.display_result(rest_res)
        self.assertEqual(self.win.tabs.currentIndex(), 5)
        self.assertGreaterEqual(len(self.win.fig_rest.axes[0].lines), 2)
        self.assertGreaterEqual(len(self.win.fig_rf.axes[0].lines), 2)
        self.assertIn("22.0 dB", self.win.lbl_rest_info.text())

        t_rf = np.linspace(-5.0, 25.0, 150)
        rf_syn = np.exp(-((t_rf - 0.0) ** 2) / 0.8) - 0.25 * np.exp(-((t_rf - 4.5) ** 2) / 1.2)
        rf_obs = rf_syn + 0.02 * np.sin(2.0 * math.pi * t_rf)
        rf_res = FitDiagnosticResult(
            fit_type=FitType.RECEIVER_FUNC,
            title="P Receiver Function",
            converged=True,
            iterations=4,
            dof=145,
            rmse=0.02,
            wrmse=0.02,
            overlay_curves={
                "rf_time": t_rf,
                "rf_obs": rf_obs,
                "rf_syn": rf_syn,
                "snr_freqs": freqs,
                "snr_db": snr_db,
            },
            context_data={"station": "BFO", "fit_pct": 93.4},
        )
        self.win.display_result(rf_res)
        self.assertEqual(self.win.tabs.currentIndex(), 5)
        # Without transfer/dispersion curves, SNR renders on left and RF gets full right canvas
        self.assertGreaterEqual(len(self.win.fig_rest.axes[0].lines), 2)
        self.assertEqual(len(self.win.fig_rf.axes), 1)
        self.assertGreaterEqual(len(self.win.fig_rf.axes[0].lines), 2)
        self.assertIn("93.4%", self.win.lbl_rest_info.text())

        # When transfer function + SNR + RF are all present, right canvas shows both SNR & RF subplots
        combined_res = FitDiagnosticResult(
            fit_type=FitType.RECEIVER_FUNC,
            title="Combined Restitution + RF",
            converged=True,
            iterations=4,
            dof=145,
            rmse=0.02,
            wrmse=0.02,
            overlay_curves={
                "freqs": freqs,
                "resp_amp": resp_amp,
                "snr_freqs": freqs,
                "snr_db": snr_db,
                "rf_time": t_rf,
                "rf_obs": rf_obs,
                "rf_syn": rf_syn,
            },
            context_data={"station": "BFO", "fit_pct": 93.4},
        )
        self.win.display_result(combined_res)
        self.assertEqual(len(self.win.fig_rf.axes), 2)

        periods = np.linspace(5.0, 50.0, 20)
        disp_res = FitDiagnosticResult(
            fit_type=FitType.SURFACE_DISPERSION,
            title="Rayleigh Dispersion",
            converged=True,
            iterations=6,
            dof=17,
            rmse=0.03,
            wrmse=0.03,
            overlay_curves={
                "periods": periods,
                "obs_vel": 3.0 + 0.02 * periods,
                "model_vel": 3.01 + 0.02 * periods,
            },
            context_data={"station": "FUR"},
        )
        self.win.display_result(disp_res)
        self.assertEqual(self.win.tabs.currentIndex(), 5)
        self.assertGreaterEqual(len(self.win.fig_rest.axes[0].lines), 2)

    def test_empty_overlay_curves_fallback_for_all_fit_types(self) -> None:
        all_types = [
            FitType.HYPOCENTER_GEIGER,
            FitType.ORIGIN_TIME,
            FitType.PLANE_WAVE,
            FitType.FK_SEMBLANCE,
            FitType.SP_DIFF_DEPTH,
            FitType.TTFIT_SIMPLEX,
            FitType.BRUNE_SPECTRUM,
            FitType.POLARIZATION_3C,
            FitType.RESTITUTION_FIT,
            FitType.RECEIVER_FUNC,
            FitType.SURFACE_DISPERSION,
        ]
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            for ftype in all_types:
                minimal = FitDiagnosticResult(
                    fit_type=ftype,
                    title=f"Minimal {ftype.value}",
                    converged=True,
                    iterations=1,
                    dof=1,
                    rmse=0.1,
                    wrmse=0.1,
                    residuals=[
                        ResidualEntry(
                            identifier="STA1.Z.P",
                            observed=1.0,
                            calculated=0.9,
                            residual=0.1,
                            station="STA1",
                            phase="P",
                        )
                    ],
                    parameters={
                        "p1": ParameterUncertainty("p1", 1.0, 0.1, (0.8, 1.2), ""),
                    },
                    overlay_curves={},
                    context_data={},
                )
                self.win.display_result(minimal)
                app.processEvents()
                self.assertEqual(self.win.table_residuals.rowCount(), 1)

        # Verify placeholder texts are present on empty Tabs 3-6
        self.assertIn("No FK", self.win.fig_fk.axes[0].texts[0].get_text())
        self.assertIn("No Spectral", self.win.fig_spec.axes[0].texts[0].get_text())
        self.assertIn("No 3C Polarization", self.win.fig_pol.axes[0].texts[0].get_text())
        self.assertIn("No Transfer Function", self.win.fig_rest.axes[0].texts[0].get_text())
        empty_mean_warns = [w for w in ws if "Mean of empty slice" in str(w.message)]
        self.assertEqual(empty_mean_warns, [])

    def test_zero_scipy_or_obspy_imports(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in ("shm_app/gui/fit_dialog.py", "tests/test_fit_dialog_p3.py"):
            fpath = os.path.join(base_dir, rel)
            with open(fpath, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=fpath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertNotIn(top, ("scipy", "obspy"), msg=f"Forbidden import {top} in {rel}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    self.assertNotIn(top, ("scipy", "obspy"), msg=f"Forbidden import {top} in {rel}")
        self.assertNotIn("scipy", sys.modules)
        self.assertNotIn("obspy", sys.modules)


if __name__ == "__main__":
    unittest.main()
