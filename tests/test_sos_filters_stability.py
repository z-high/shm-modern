"""Unit and adversarial stability tests for M-10 (SOS Butterworth up to N=16,
pure-NumPy sosfilt/sosfiltfilt) and M-17 (filter transient autocut).
"""

from __future__ import annotations

import ast
import math
import os
import unittest
import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.models import Trace
from shm_app.processing import filters as filters_mod
from shm_app.processing.filters import (
    FilterStage,
    apply_fft_filter,
    apply_recursive_filter,
    autocut_array,
    autocut_trace,
    butterworth_sos,
    compute_filter_settle_sec,
    design_butterworth_stages,
    evaluate_stage_freq,
    filter_trace,
    load_named_filter,
    parse_filter_file,
    sosfilt,
    sosfiltfilt,
)


class TestSOSButterworthStability(unittest.TestCase):
    """M-10: Numerical stability, unit-circle poles, exact response, and zero-phase symmetry."""

    def test_poles_strictly_inside_unit_circle_up_to_order_16(self) -> None:
        for ftype in ("LP", "HP", "BP", "BS"):
            for order in (1, 3, 4, 8, 12, 16):
                for dt in (0.05, 0.01):  # 20 Hz and 100 Hz
                    with self.subTest(ftype=ftype, order=order, dt=dt):
                        sos = butterworth_sos(ftype, order=order, freq_lo=0.5, freq_hi=1.5, dt=dt)
                        expected_sections = (order + 1) // 2 if ftype in ("LP", "HP") else order
                        self.assertEqual(sos.shape, (expected_sections, 6))
                        self.assertTrue(np.all(np.isfinite(sos)))
                        for sec in sos:
                            roots = np.roots([sec[3], sec[4], sec[5]])
                            self.assertTrue(
                                all(abs(r) < 1.0 for r in roots),
                                f"Pole outside unit circle for {ftype} N={order} dt={dt}: {roots}",
                            )

    def test_order_clamp_1_to_16(self) -> None:
        sos_low = butterworth_sos("LP", order=0, freq_lo=2.0, dt=0.05)
        self.assertEqual(sos_low.shape[0], 1)
        sos_high = butterworth_sos("LP", order=25, freq_lo=2.0, dt=0.05)
        self.assertEqual(sos_high.shape[0], 8)  # 16 poles -> 8 biquads

        stgs = design_butterworth_stages("BP", order=16, freq_lo=0.5, freq_hi=1.5, dt=0.05, recursive=True)
        self.assertEqual(len(stgs), 1)
        self.assertEqual(stgs[0].store_id, 3)
        self.assertEqual(len(stgs[0].sos), 16)

    def test_narrow_bandpass_and_all_types_time_domain_stability_orders_4_8_12_16(self) -> None:
        rng = np.random.default_rng(42)
        for dt in (0.05, 0.01):  # 20 Hz and 100 Hz
            n_samples = int(40.0 / dt)
            t = np.arange(n_samples) * dt
            # Composite signal: 0.1 Hz (low stopband), 1.0 Hz (passband), 7.5 Hz (high stopband) + noise
            x = (
                100.0 * np.sin(2.0 * math.pi * 0.1 * t)
                + 100.0 * np.sin(2.0 * math.pi * 1.0 * t)
                + 100.0 * np.sin(2.0 * math.pi * 7.5 * t)
                + rng.normal(0.0, 2.0, size=n_samples)
            )
            in_energy = float(np.sum(x * x))

            for order in (4, 8, 12, 16):
                # Narrow bandpass 0.5 - 1.5 Hz
                with self.subTest(dt=dt, order=order, kind="narrow_BP"):
                    stgs = design_butterworth_stages("BP", order=order, freq_lo=0.5, freq_hi=1.5, dt=dt, recursive=True)
                    y = apply_recursive_filter(x, dt, stgs, demean_first=True)
                    self.assertEqual(y.shape, x.shape)
                    self.assertTrue(np.all(np.isfinite(y)))
                    out_energy = float(np.sum(y * y))
                    self.assertLess(out_energy, 1.1 * in_energy)
                    # Check steady-state middle segment preserves 1.0 Hz passband amplitude (~70.7 RMS)
                    mid = y[n_samples // 2 : -n_samples // 8]
                    rms_mid = float(np.sqrt(np.mean(mid * mid)))
                    self.assertGreater(rms_mid, 55.0)
                    self.assertLess(rms_mid, 85.0)

                # LP / HP / BS orders 4, 8, 12, 16
                for ftype, f1, f2 in (("LP", 2.0, 2.0), ("HP", 2.0, 2.0), ("BS", 0.5, 1.5)):
                    with self.subTest(dt=dt, order=order, kind=ftype):
                        stgs = design_butterworth_stages(ftype, order=order, freq_lo=f1, freq_hi=f2, dt=dt, recursive=True)
                        y = apply_recursive_filter(x, dt, stgs, demean_first=True)
                        self.assertEqual(y.shape, x.shape)
                        self.assertTrue(np.all(np.isfinite(y)))
                        self.assertLess(float(np.sum(y * y)), 1.2 * in_energy)

    def test_exact_frequency_response_passband_cutoff_and_stopband(self) -> None:
        dt = 0.01  # 100 Hz
        for order in (4, 8, 12, 16):
            # LP at 5.0 Hz
            stgs_lp = design_butterworth_stages("LP", order=order, freq_lo=5.0, dt=dt, recursive=True)
            freqs = np.array([0.5, 5.0, 15.0])
            h_lp = evaluate_stage_freq(stgs_lp[0], 2.0 * math.pi * freqs)
            self.assertAlmostEqual(abs(h_lp[0]), 1.0, places=4)
            self.assertAlmostEqual(abs(h_lp[1]), 1.0 / math.sqrt(2.0), places=4)
            # Theoretical Butterworth stopband gain at 15 Hz (pre-warped)
            w_ratio = math.tan(math.pi * 15.0 * dt) / math.tan(math.pi * 5.0 * dt)
            expected_stop = 1.0 / math.sqrt(1.0 + w_ratio ** (2 * order))
            self.assertAlmostEqual(abs(h_lp[2]), expected_stop, places=6)

            # HP at 5.0 Hz
            stgs_hp = design_butterworth_stages("HP", order=order, freq_lo=5.0, dt=dt, recursive=True)
            h_hp = evaluate_stage_freq(stgs_hp[0], 2.0 * math.pi * np.array([0.5, 5.0, 25.0]))
            self.assertLess(abs(h_hp[0]), 1e-3)
            self.assertAlmostEqual(abs(h_hp[1]), 1.0 / math.sqrt(2.0), places=4)
            self.assertAlmostEqual(abs(h_hp[2]), 1.0, places=4)

            # Narrow BP 0.5 - 1.5 Hz
            stgs_bp = design_butterworth_stages("BP", order=order, freq_lo=0.5, freq_hi=1.5, dt=dt, recursive=True)
            f_center = math.sqrt(0.5 * 1.5)
            h_bp = evaluate_stage_freq(stgs_bp[0], 2.0 * math.pi * np.array([0.1, 0.5, f_center, 1.5, 8.0]))
            self.assertLess(abs(h_bp[0]), 1e-3)
            self.assertAlmostEqual(abs(h_bp[1]), 1.0 / math.sqrt(2.0), places=3)
            self.assertAlmostEqual(abs(h_bp[2]), 1.0, places=3)
            self.assertAlmostEqual(abs(h_bp[3]), 1.0 / math.sqrt(2.0), places=3)
            self.assertLess(abs(h_bp[4]), 1e-3)

            # BS 0.5 - 1.5 Hz
            stgs_bs = design_butterworth_stages("BS", order=order, freq_lo=0.5, freq_hi=1.5, dt=dt, recursive=True)
            h_bs = evaluate_stage_freq(stgs_bs[0], 2.0 * math.pi * np.array([0.05, 0.5, f_center, 1.5, 10.0]))
            self.assertAlmostEqual(abs(h_bs[0]), 1.0, places=3)
            self.assertAlmostEqual(abs(h_bs[1]), 1.0 / math.sqrt(2.0), places=3)
            self.assertLess(abs(h_bs[2]), 1e-6)
            self.assertAlmostEqual(abs(h_bs[3]), 1.0 / math.sqrt(2.0), places=3)
            self.assertAlmostEqual(abs(h_bs[4]), 1.0, places=3)

    def test_sosfiltfilt_zero_phase_symmetry_and_edge_suppression(self) -> None:
        dt = 0.02
        n = 501
        t = (np.arange(n) - n // 2) * dt
        # Symmetric Gaussian pulse at exact center index 250
        pulse = np.exp(-0.5 * (t / 0.3) ** 2)

        for order in (4, 8, 12):
            sos = butterworth_sos("LP", order=order, freq_lo=3.0, dt=dt)
            y_causal = sosfilt(sos, pulse)
            y_zp = sosfiltfilt(sos, pulse)

            # Causal filter shifts peak forward in time; sosfiltfilt preserves exact symmetry & peak index
            self.assertGreater(int(np.argmax(y_causal)), n // 2)
            self.assertEqual(int(np.argmax(y_zp)), n // 2)
            np.testing.assert_allclose(y_zp, y_zp[::-1], atol=1e-11)

        # Zero-phase preserves phase of passband sinusoid while attenuating stopband
        t_sin = np.arange(1000) * dt
        sig_pass = np.sin(2.0 * math.pi * 1.0 * t_sin)
        sig_stop = np.sin(2.0 * math.pi * 12.0 * t_sin)
        sos_bp = butterworth_sos("BP", order=8, freq_lo=0.5, freq_hi=3.0, dt=dt)
        out = sosfiltfilt(sos_bp, sig_pass + sig_stop)
        corr = np.corrcoef(sig_pass[100:-100], out[100:-100])[0, 1]
        self.assertGreater(corr, 0.999)

        # Edge suppression on linear ramp + offset under LP
        ramp = 50.0 + 2.5 * t_sin + 0.5 * np.sin(2.0 * math.pi * 1.0 * t_sin)
        sos_lp = butterworth_sos("LP", order=8, freq_lo=3.0, dt=dt)
        ramp_out = sosfiltfilt(sos_lp, ramp)
        np.testing.assert_allclose(ramp_out, ramp, atol=1e-2)

    def test_sos_stage_inversion_and_edge_cases(self) -> None:
        dt = 0.02
        stg = design_butterworth_stages("BP", order=12, freq_lo=0.8, freq_hi=3.0, dt=dt, recursive=True)[0]
        inv = stg.inverted()
        omega = 2.0 * math.pi * np.linspace(0.2, 10.0, 100)
        prod = evaluate_stage_freq(stg, omega) * evaluate_stage_freq(inv, omega)
        np.testing.assert_allclose(np.abs(prod), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.angle(prod), 0.0, atol=1e-10)

        # Empty and tiny arrays
        sos = butterworth_sos("LP", order=8, freq_lo=2.0, dt=dt)
        self.assertEqual(sosfilt(sos, np.array([])).size, 0)
        self.assertEqual(sosfiltfilt(sos, np.array([])).size, 0)
        np.testing.assert_array_equal(sosfiltfilt(sos, np.array([3.5])), np.array([3.5]))
        self.assertEqual(sosfiltfilt(sos, np.array([1.0, 2.0])).size, 2)

        # Invalid inputs
        with self.assertRaises(ValueError):
            sosfilt(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.ones(10))
        with self.assertRaises(ValueError):
            sosfilt(np.ones((2, 5)), np.ones(10))


class TestFilterAutocutAndLegacyCompatibility(unittest.TestCase):
    """M-17: Settling time estimation, transient autocut, and .FLF/.FLR backward compatibility."""

    def test_compute_filter_settle_sec_formula_and_impulse_fallback(self) -> None:
        # Formula tau ~= N / (2 * pi * f_c)
        tau_4 = compute_filter_settle_sec("BUTTER LP 1.0 4", dt=0.05)
        tau_8 = compute_filter_settle_sec("BUTTER LP 1.0 8", dt=0.05)
        tau_lowf = compute_filter_settle_sec("BP 0.2 2.0 8", dt=0.05)
        self.assertAlmostEqual(tau_4, 4.0 / (2.0 * math.pi * 1.0), places=6)
        self.assertAlmostEqual(tau_8, 8.0 / (2.0 * math.pi * 1.0), places=6)
        self.assertAlmostEqual(tau_lowf, 8.0 / (2.0 * math.pi * 0.2), places=6)

        # Stage-based Butterworth settle matches string formula
        stgs = design_butterworth_stages("BP", order=8, freq_lo=0.5, freq_hi=2.0, dt=0.05)
        self.assertAlmostEqual(
            compute_filter_settle_sec(stgs, dt=0.05),
            8.0 / (2.0 * math.pi * 0.5),
            places=6,
        )

        # Synthetic simulation stage fallback (non-Butterworth)
        tau_wwssn = compute_filter_settle_sec("WWSSN_SP", dt=0.05)
        self.assertGreater(tau_wwssn, 0.2)
        self.assertLess(tau_wwssn, 20.0)
        self.assertEqual(compute_filter_settle_sec("NONE", dt=0.05), 0.0)

    def test_autocut_trace_epoch_advancement_and_short_trace_clamping(self) -> None:
        dt = 0.05
        t0 = 1700000000.0
        data = np.arange(1000, dtype=np.float64)  # 50 s
        tr = Trace(data=data, delta=dt, station="GRA1", start_epoch=t0)

        # Causal trim (startup only)
        cut_causal = autocut_trace(tr, settle_sec=2.0, zero_phase=False, in_place=False)
        self.assertEqual(tr.length, 1000)  # original untouched
        self.assertEqual(tr.start_epoch, t0)
        self.assertEqual(cut_causal.length, 960)  # 40 samples cut from front
        self.assertAlmostEqual(cut_causal.start_epoch, t0 + 2.0, places=9)
        self.assertEqual(cut_causal.data[0], 40.0)
        self.assertEqual(cut_causal.data[-1], 999.0)

        # Zero-phase trim (startup + tail)
        cut_zp = autocut_trace(tr, settle_sec=2.0, zero_phase=True, in_place=False)
        self.assertEqual(cut_zp.length, 920)  # 40 front + 40 back
        self.assertAlmostEqual(cut_zp.start_epoch, t0 + 2.0, places=9)
        self.assertEqual(cut_zp.data[0], 40.0)
        self.assertEqual(cut_zp.data[-1], 959.0)

        # Short trace clamping (< 2 * settle_sec): must never produce empty trace
        short_tr = Trace(data=np.arange(30, dtype=np.float64), delta=dt, start_epoch=t0)
        cut_short = autocut_trace(short_tr, settle_sec=5.0, zero_phase=True)
        self.assertGreaterEqual(cut_short.length, 1)
        self.assertGreater(cut_short.start_epoch, t0)

        tiny_tr = Trace(data=np.array([1.0, 2.0, 3.0]), delta=dt, start_epoch=t0)
        cut_tiny = autocut_trace(tiny_tr, settle_sec=5.0, zero_phase=True)
        self.assertEqual(cut_tiny.length, 3)
        self.assertEqual(cut_tiny.start_epoch, t0)

        # autocut_array
        arr_cut = autocut_array(data, dt=dt, settle_sec=2.0, zero_phase=True)
        self.assertEqual(arr_cut.size, 920)
        np.testing.assert_array_equal(arr_cut, cut_zp.data)

    def test_filter_trace_autocut_removes_startup_ringing_and_preserves_default(self) -> None:
        dt = 0.05
        t0 = 1000000.0
        n = 1200  # 60 s
        t = np.arange(n) * dt
        # Step discontinuity at sample 0 followed by quiet 1 Hz sine wave after t=20s
        # First sample jump creates startup ringing in BP filter without LPC
        sig = 500.0 * np.exp(-t / 0.2) + 5.0 * np.sin(2.0 * math.pi * 1.0 * t)
        tr = Trace(data=sig, delta=dt, station="BFO", start_epoch=t0)

        tr_no_cut = filter_trace(tr, "BP 0.5 2.0 8", mode="REC", autocut=False)
        self.assertEqual(tr_no_cut.length, n)
        self.assertEqual(tr_no_cut.start_epoch, t0)

        tr_cut = filter_trace(tr, "BP 0.5 2.0 8", mode="REC", autocut=True)
        expected_settle = compute_filter_settle_sec("BP 0.5 2.0 8", dt=dt)
        expected_cut_samples = int(round(expected_settle / dt))
        self.assertEqual(tr_cut.length, n - expected_cut_samples)
        self.assertAlmostEqual(tr_cut.start_epoch, t0 + expected_cut_samples * dt, places=9)
        # Startup ringing spike in tr_no_cut is much larger than peak in tr_cut
        self.assertLess(float(np.max(np.abs(tr_cut.data))), float(np.max(np.abs(tr_no_cut.data))))

        # Zero-phase + autocut
        tr_zp_cut = filter_trace(tr, "BP 0.5 2.0 8", mode="REC", zero_phase=True, autocut=True)
        self.assertEqual(tr_zp_cut.length, n - 2 * expected_cut_samples)
        self.assertAlmostEqual(tr_zp_cut.start_epoch, t0 + expected_cut_samples * dt, places=9)

    def test_legacy_flf_flr_files_and_architectural_invariants(self) -> None:
        cfg = get_config()
        if os.path.isdir(cfg.filter_dir):
            flf_path = os.path.join(cfg.filter_dir, "G_WWSSN_SP.FLF")
            flr_path = os.path.join(cfg.filter_dir, "20HZ_GRSN_S+G_WWSSN_SP.FLR")
            tr = Trace(data=np.sin(2.0 * math.pi * 1.0 * np.arange(800) * 0.05), delta=0.05, station="GRA1")
            if os.path.isfile(flf_path):
                stgs_flf = parse_filter_file(flf_path)
                self.assertGreaterEqual(len(stgs_flf), 1)
                self.assertEqual(stgs_flf[0].store_id, 1)
                self.assertGreater(compute_filter_settle_sec(stgs_flf, dt=0.05), 0.0)
                out_flf = apply_fft_filter(tr.data, 0.05, stgs_flf)
                self.assertTrue(np.all(np.isfinite(out_flf)))
            if os.path.isfile(flr_path):
                stgs_flr = parse_filter_file(flr_path)
                self.assertGreaterEqual(len(stgs_flr), 1)
                self.assertEqual(stgs_flr[0].store_id, 3)
                self.assertEqual(stgs_flr[0].sos, [])
                self.assertGreater(compute_filter_settle_sec(stgs_flr, dt=0.05), 0.0)
                out_flr = apply_recursive_filter(tr.data, 0.05, stgs_flr)
                self.assertTrue(np.all(np.isfinite(out_flr)))
                out_flr_zp = apply_recursive_filter(tr.data, 0.05, stgs_flr, zero_phase=True)
                self.assertTrue(np.all(np.isfinite(out_flr_zp)))

        # Verify architectural invariant: ZERO PyQt6, scipy, or obspy imports in filters.py
        filters_path = os.path.abspath(filters_mod.__file__)
        with open(filters_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filters_path)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        for forbidden in ("PyQt6", "scipy", "obspy"):
            self.assertNotIn(forbidden, imported_roots)


if __name__ == "__main__":
    unittest.main()
