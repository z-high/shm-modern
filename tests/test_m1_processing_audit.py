"""Regression and edge-case tests for M1 signal processing & spectral analysis fixes
across filters.py, trace_ops.py, polarization.py, and spectrum.py.
"""

from __future__ import annotations

import math
import unittest
import warnings
import numpy as np

from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.processing.filters import (
    FilterStage,
    apply_fft_filter,
    apply_recursive_filter,
    butterworth_sos,
    design_butterworth_stages,
    evaluate_stage_freq,
    filter_trace,
)
from shm_app.processing.polarization import (
    analyze_polarization_diagnostic,
    apply_polarization_filter,
    compute_mdir,
    rotate_2d,
    rotate_3d_lqt,
    rotate_STS2_zne_to_uvw,
    rotate_zne_to_zrt,
)
from shm_app.processing.spectrum import (
    compute_welch_power_spectrum,
    find_surface_wave_amplitude,
    invert_brune_spectrum,
    scan_surface_waves,
)
from shm_app.processing.trace_ops import (
    create_synthetic_trace,
    cut_trace,
    resample_traces_to_common_grid,
    sum_traces,
)


class TestFiltersAudit(unittest.TestCase):
    def test_apply_fft_filter_preserves_dc_for_lowpass_and_gain(self) -> None:
        dt = 0.05
        n = 256
        x = np.full(n, 5.0, dtype=np.float64)

        lp_cont = design_butterworth_stages("LP", order=4, freq_lo=2.0, dt=dt, recursive=False)
        out_cont = apply_fft_filter(x, dt, lp_cont, taper_frac=0.0, demean_first=False)
        # Interior steady-state away from startup/zero-pad edges preserves DC 5.0 (previously decayed toward 0 with mean ~2.4)
        np.testing.assert_allclose(out_cont[n // 4 : 3 * n // 4], 5.0, rtol=1e-4, atol=1e-4)

        lp_rec = design_butterworth_stages("LP", order=4, freq_lo=2.0, dt=dt, recursive=True)
        out_rec = apply_fft_filter(x, dt, lp_rec, taper_frac=0.0, demean_first=False)
        np.testing.assert_allclose(out_rec[n // 4 : 3 * n // 4], 5.0, rtol=1e-4, atol=1e-4)

        gain_stage = [FilterStage(store_id=1, norm=2.5, zeros=[], poles=[], comment="GAIN")]
        out_gain = apply_fft_filter(x, dt, gain_stage, taper_frac=0.0, demean_first=False)
        np.testing.assert_allclose(out_gain, 12.5, rtol=1e-10, atol=1e-10)

    def test_apply_fft_filter_zeros_dc_for_hp_bp_and_origin_zero_or_pole(self) -> None:
        dt = 0.05
        n = 1024
        x = np.full(n, 5.0, dtype=np.float64)

        for ftype, f1, f2 in (("HP", 1.0, 1.0), ("BP", 0.5, 2.0)):
            for rec in (False, True):
                stgs = design_butterworth_stages(ftype, order=3, freq_lo=f1, freq_hi=f2, dt=dt, recursive=rec)
                out = apply_fft_filter(x, dt, stgs, taper_frac=0.0, demean_first=False)
                self.assertAlmostEqual(float(np.mean(out[n // 2 : 3 * n // 4])), 0.0, places=4)
                if rec:
                    np.testing.assert_allclose(out[n // 2 : 3 * n // 4], 0.0, atol=1e-5)

        zero_origin = [FilterStage(store_id=1, norm=1.0, zeros=[0j], poles=[complex(-2.0 * math.pi, 0.0)], comment="HP1")]
        out_zero = apply_fft_filter(x, dt, zero_origin, taper_frac=0.0, demean_first=False)
        np.testing.assert_allclose(out_zero[n // 4 : 3 * n // 4], 0.0, atol=1e-3)

        pole_origin = [FilterStage(store_id=1, norm=1.0, zeros=[], poles=[0j], comment="INT")]
        out_pole = apply_fft_filter(x, dt, pole_origin, taper_frac=0.0, demean_first=False)
        self.assertTrue(np.all(np.isfinite(out_pole)))
        self.assertAlmostEqual(float(np.mean(out_pole)), 0.0, places=10)

    def test_filter_validates_positive_finite_dt(self) -> None:
        x = np.ones(32, dtype=np.float64)
        stgs = design_butterworth_stages("LP", order=2, freq_lo=1.0, dt=0.05, recursive=False)
        for bad_dt in (0.0, -0.05, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                apply_fft_filter(x, bad_dt, stgs)
            with self.assertRaises(ValueError):
                apply_recursive_filter(x, bad_dt, stgs)

    def test_filter_trace_fft_mode_supports_zero_phase(self) -> None:
        dt = 0.02
        n = 501
        t = (np.arange(n) - n // 2) * dt
        pulse = np.exp(-0.5 * (t / 0.3) ** 2)
        tr = Trace(data=pulse, delta=dt, station="GRA1", comp="Z")

        # Continuous s-plane LP stage forces FFT mode
        tr_causal = filter_trace(tr, "LP 3.0 4", mode="FFT", taper_frac=0.0, zero_phase=False)
        tr_zp = filter_trace(tr, "LP 3.0 4", mode="FFT", taper_frac=0.0, zero_phase=True)

        self.assertGreater(int(np.argmax(tr_causal.data)), n // 2)
        self.assertEqual(int(np.argmax(tr_zp.data)), n // 2)
        np.testing.assert_allclose(tr_zp.data, tr_zp.data[::-1], atol=1e-10)

    def test_evaluate_stage_freq_tabulated_unwraps_phase_across_180_deg(self) -> None:
        stg = FilterStage(
            store_id=2,
            norm=1.0,
            freqs_hz=[1.0, 2.0],
            amplitudes=[1.0, 1.0],
            phases_deg=[170.0, -170.0],
        )
        w_mid = np.array([2.0 * math.pi * 1.5])
        h_mid = evaluate_stage_freq(stg, w_mid)[0]
        self.assertAlmostEqual(abs(h_mid), 1.0, places=9)
        # Wrapped midpoint without unwrapping would be 0 deg (+1 + 0j); unwrapped is 180 deg (-1 + 0j)
        self.assertAlmostEqual(h_mid.real, -1.0, places=9)
        self.assertAlmostEqual(abs(h_mid.imag), 0.0, places=9)

    def test_direct_iir_preserves_tiny_physical_displacement_signals(self) -> None:
        # 1e-35 m signal would be flushed to 0.0 by an overly aggressive 1e-30 cutoff
        x = np.full(32, 1e-35, dtype=np.float64)
        stg = FilterStage(store_id=3, norm=1.0, dt=0.05, num_a=[1.0], den_b=[1.0])
        out = apply_recursive_filter(x, 0.05, [stg], demean_first=False)
        np.testing.assert_allclose(out, x, rtol=1e-12)

    def test_butterworth_sos_warns_when_clamping_at_nyquist(self) -> None:
        dt = 0.05  # Nyquist = 10.0 Hz
        for ftype, f1, f2 in (
            ("LP", 10.0, 10.0),
            ("HP", 12.0, 12.0),
            ("BP", 2.0, 15.0),
            ("BS", 11.0, 15.0),
        ):
            with self.subTest(ftype=ftype):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sos = butterworth_sos(ftype, order=4, freq_lo=f1, freq_hi=f2, dt=dt)
                self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))
                self.assertTrue(np.all(np.isfinite(sos)))


class TestTraceOpsAudit(unittest.TestCase):
    def test_sum_traces_respects_explicit_zero_weight(self) -> None:
        t0 = 1000.0
        tr_bad = Trace(data=np.array([999.0, 999.0, 999.0]), delta=0.05, start_epoch=t0, weight=0.0, station="BAD")
        tr_good = Trace(data=np.array([2.0, 4.0, 6.0]), delta=0.05, start_epoch=t0, weight=2.0, station="GOOD")

        res = sum_traces([tr_bad, tr_good], normalize_weights=True)
        self.assertIsNotNone(res)
        assert res is not None
        np.testing.assert_allclose(res.data, np.array([2.0, 4.0, 6.0]))

        res_unnorm = sum_traces([tr_bad, tr_good], normalize_weights=False)
        assert res_unnorm is not None
        np.testing.assert_allclose(res_unnorm.data, np.array([4.0, 8.0, 12.0]))

        # All-zero weights produce zeros safely
        tr_bad2 = tr_bad.copy()
        res_all_zero = sum_traces([tr_bad, tr_bad2], normalize_weights=True)
        assert res_all_zero is not None
        np.testing.assert_array_equal(res_all_zero.data, np.zeros(3))

    def test_resample_traces_to_common_grid_validates_dt_and_max_grid_size(self) -> None:
        tr_bad_dt = Trace(data=np.ones(10), delta=0.0, start_epoch=1000.0)
        with self.assertRaises(ValueError):
            resample_traces_to_common_grid([tr_bad_dt])

        tr_a = Trace(data=np.ones(10), delta=0.01, start_epoch=1000.0)
        tr_b = Trace(data=np.ones(10), delta=0.01, start_epoch=1000.0 + 100 * 86400.0)
        with self.assertRaises(ValueError):
            resample_traces_to_common_grid([tr_a, tr_b])

    def test_cut_trace_out_of_bounds_pad_zeros_false_returns_empty_and_prunes_picks(self) -> None:
        t0 = 1_700_000_000.0
        dt = 0.05
        tr = Trace(data=np.arange(201, dtype=np.float64), delta=dt, start_epoch=t0, station="GRA1")
        tr.picks = [
            PhasePick(name="P", onset=format_sh_time(t0 + 2.0), station="GRA1"),
            PhasePick(name="S", onset=format_sh_time(t0 + 7.0), station="GRA1"),
        ]
        tr.theo_picks = {"P": t0 + 2.0, "S": t0 + 7.0}

        # Completely after trace end
        cut_after = cut_trace(tr, 20.0, 30.0, pad_zeros=False)
        self.assertEqual(cut_after.length, 0)
        self.assertEqual(cut_after.picks, [])
        self.assertEqual(cut_after.theo_picks, {})

        # Completely before trace start
        cut_before = cut_trace(tr, -15.0, -5.0, pad_zeros=False)
        self.assertEqual(cut_before.length, 0)
        self.assertEqual(cut_before.picks, [])
        self.assertEqual(cut_before.theo_picks, {})

        # Partial window keeps in-bounds picks and prunes out-of-bounds picks
        cut_partial = cut_trace(tr, 0.0, 4.0, pad_zeros=False)
        self.assertEqual([p.name for p in cut_partial.picks], ["P"])
        self.assertEqual(list(cut_partial.theo_picks.keys()), ["P"])


class TestPolarizationAudit(unittest.TestCase):
    def _make_p_wave_3c(
        self,
        baz_deg: float = 120.0,
        inc_deg: float = 30.0,
        dt: float = 0.05,
        start_epoch: float = 1000.0,
        duration_sec: float = 20.0,
        dc_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[Trace, Trace, Trace]:
        n = int(round(duration_sec / dt))
        t = np.arange(n) * dt
        wave = 100.0 * np.exp(-0.5 * ((t - 10.0) / 1.0) ** 2) * np.sin(2.0 * math.pi * 1.2 * t)
        az_r = math.radians(baz_deg)
        inc_r = math.radians(inc_deg)
        z = wave * math.cos(inc_r) + dc_offsets[0]
        n_d = -wave * math.sin(inc_r) * math.cos(az_r) + dc_offsets[1]
        e = -wave * math.sin(inc_r) * math.sin(az_r) + dc_offsets[2]
        return (
            Trace(data=z, delta=dt, start_epoch=start_epoch, station="GRA1", comp="Z"),
            Trace(data=n_d, delta=dt, start_epoch=start_epoch, station="GRA1", comp="N"),
            Trace(data=e, delta=dt, start_epoch=start_epoch, station="GRA1", comp="E"),
        )

    def test_mismatched_delta_rejected_across_polarization_apis(self) -> None:
        tz, tn, te = self._make_p_wave_3c()
        te_bad = te.copy()
        te_bad.delta = 0.01

        with self.assertRaises(ValueError):
            rotate_2d(tn, te_bad, 45.0)
        with self.assertRaises(ValueError):
            rotate_zne_to_zrt(tz, tn, te_bad, 120.0)
        with self.assertRaises(ValueError):
            rotate_3d_lqt(tz, tn, te_bad, 120.0, 30.0)
        with self.assertRaises(ValueError):
            rotate_STS2_zne_to_uvw(tz, tn, te_bad)
        with self.assertRaises(ValueError):
            compute_mdir(tz, tn, te_bad)
        with self.assertRaises(ValueError):
            analyze_polarization_diagnostic(tz, tn, te_bad)
        with self.assertRaises(ValueError):
            apply_polarization_filter(tz, tn, te_bad)

    def test_staggered_start_epoch_aligned_and_disjoint_rejected(self) -> None:
        dt = 0.05
        t0 = 1000.0
        baz_true, inc_true = 120.0, 30.0
        az_r, inc_r = math.radians(baz_true), math.radians(inc_true)

        # Build continuous analytical P-wave sampled at staggered start epochs
        def _sample_wave(start: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            tt = (start - t0) + np.arange(n) * dt
            w = 100.0 * np.sin(2.0 * math.pi * 0.8 * tt)
            return w * math.cos(inc_r), -w * math.sin(inc_r) * math.cos(az_r), -w * math.sin(inc_r) * math.sin(az_r)

        z_a, _, _ = _sample_wave(t0, 400)
        _, n_b, _ = _sample_wave(t0 + 1.0, 400)
        _, _, e_c = _sample_wave(t0 + 2.0, 400)

        tz = Trace(data=z_a, delta=dt, start_epoch=t0, station="GRA1", comp="Z")
        tn = Trace(data=n_b, delta=dt, start_epoch=t0 + 1.0, station="GRA1", comp="N")
        te = Trace(data=e_c, delta=dt, start_epoch=t0 + 2.0, station="GRA1", comp="E")

        tl, tq, tt = rotate_3d_lqt(tz, tn, te, baz_true, inc_true)
        self.assertAlmostEqual(tl.start_epoch, t0 + 2.0, places=9)
        self.assertLess(float(np.max(np.abs(tq.data))), 1e-6)
        self.assertLess(float(np.max(np.abs(tt.data))), 1e-6)

        mdir = compute_mdir(tz, tn, te)
        self.assertAlmostEqual(mdir["azimuth"], baz_true, delta=0.5)
        self.assertAlmostEqual(mdir["incidence"], inc_true, delta=0.5)

        pz, pn, pe = apply_polarization_filter(tz, tn, te, coherence_win_sec=0.5, mode=1)
        self.assertAlmostEqual(pz.start_epoch, t0 + 2.0, places=9)
        self.assertEqual(pz.length, tl.length)

        # Completely disjoint start epochs raise ValueError
        te_disjoint = Trace(data=e_c, delta=dt, start_epoch=t0 + 100.0, station="GRA1", comp="E")
        with self.assertRaises(ValueError):
            rotate_3d_lqt(tz, tn, te_disjoint, baz_true, inc_true)
        with self.assertRaises(ValueError):
            compute_mdir(tz, tn, te_disjoint)

    def test_mdir_and_polfil_guard_nan_and_inf(self) -> None:
        tz, tn, te = self._make_p_wave_3c()
        tz.data[195] = np.nan
        tn.data[205] = np.inf
        te.data[215] = -np.inf

        mdir = compute_mdir(tz, tn, te, 7.0, 13.0)
        self.assertTrue(all(math.isfinite(v) for v in mdir.values()))
        self.assertAlmostEqual(mdir["azimuth"], 120.0, delta=2.0)

        diag = analyze_polarization_diagnostic(tz, tn, te, 7.0, 13.0)
        self.assertTrue(diag.converged)
        self.assertTrue(np.all(np.isfinite(diag.overlay_curves["l"])))

        for mode in (1, 2, 3, 4, 5):
            pz, pn, pe = apply_polarization_filter(tz, tn, te, mode=mode, baz_deg=120.0, inci_deg=30.0)
            self.assertTrue(np.all(np.isfinite(pz.data)))
            self.assertTrue(np.all(np.isfinite(pn.data)))
            self.assertTrue(np.all(np.isfinite(pe.data)))

        # All-NaN traces return safe finite zeros
        tz_nan = tz.copy()
        tz_nan.data[:] = np.nan
        mdir_nan = compute_mdir(tz_nan, tn, te)
        self.assertEqual(mdir_nan["lambda1"], 0.0)
        pz_nan, _, _ = apply_polarization_filter(tz_nan, tn, te)
        np.testing.assert_array_equal(pz_nan.data, np.zeros(tz_nan.length))

    def test_polfil_centered_covariance_invariant_to_dc_offset(self) -> None:
        # With centered covariance, mode=1 weights on (Z, N, E) with a huge DC offset
        # yield the exact same rectilinearity weights as zero-DC traces.
        tz0, tn0, te0 = self._make_p_wave_3c(baz_deg=60.0, inc_deg=25.0)
        dc = (1e6, -5e5, 2.5e5)
        tz_dc, tn_dc, te_dc = self._make_p_wave_3c(baz_deg=60.0, inc_deg=25.0, dc_offsets=dc)

        # In mode=2 (projection onto e1), uncentered covariance would align e1 along the DC vector!
        pz0, pn0, pe0 = apply_polarization_filter(tz0, tn0, te0, coherence_win_sec=1.0, mode=2)
        pz_dc, pn_dc, pe_dc = apply_polarization_filter(tz_dc, tn_dc, te_dc, coherence_win_sec=1.0, mode=2)

        # At the signal peak (t=10s -> idx=200), demeaned output matches centered projection direction
        peak_idx = int(np.argmax(np.abs(pz0.data)))
        self.assertGreater(abs(pz0.data[peak_idx]), 50.0)
        # Demeaned Z/N/E ratio in mode=2 output matches zero-DC mode=2 at peak
        ratio_0 = pn0.data[peak_idx] / pz0.data[peak_idx]
        # Subtract projected DC component by checking correlation of demeaned filtered waveforms
        corr_z = np.corrcoef(pz0.data[150:250], (pz_dc.data - np.mean(pz_dc.data))[150:250])[0, 1]
        self.assertGreater(corr_z, 0.9999)
        self.assertTrue(math.isfinite(ratio_0))


class TestSpectrumAudit(unittest.TestCase):
    def test_compute_welch_power_spectrum_short_signal_units_consistent(self) -> None:
        dt = 0.05
        rng = np.random.default_rng(123)
        sigma = 10.0
        # For white noise with variance sigma^2, PSD(f) ~= sigma^2 * dt = 5.0 amp^2/Hz
        long_tr = Trace(data=rng.normal(0.0, sigma, size=2000), delta=dt)
        _, psd_long = compute_welch_power_spectrum(long_tr)

        short_meds = []
        for seed in range(50):
            r = np.random.default_rng(seed)
            short_tr = Trace(data=r.normal(0.0, sigma, size=6), delta=dt)
            _, psd_short = compute_welch_power_spectrum(short_tr)
            short_meds.append(float(np.mean(psd_short)))

        mean_long = float(np.mean(psd_long))
        mean_short = float(np.mean(short_meds))
        # Both should be around sigma^2 * dt = 5.0 (previous bug gave ~0.25, 20x lower)
        self.assertAlmostEqual(mean_long, sigma * sigma * dt, delta=1.5)
        self.assertAlmostEqual(mean_short, sigma * sigma * dt, delta=1.5)

    def test_invert_brune_spectrum_handles_empty_and_bad_noise_arrays(self) -> None:
        tr = create_synthetic_trace("SHARP", length_sec=20.0, dt=0.05, pos_sec=5.0, ampl=1000.0, freq_hz=1.5)
        res = invert_brune_spectrum(
            tr,
            f_lo=0.2,
            f_hi=8.0,
            noise_spectrum=(np.array([]), np.array([])),
        )
        self.assertTrue(math.isfinite(res["omega0"]))
        self.assertTrue(math.isfinite(res["fc"]))

        with self.assertRaises(ValueError):
            invert_brune_spectrum(
                tr,
                noise_spectrum=(np.array([1.0, 2.0]), np.array([1.0])),
            )

    def test_find_surface_wave_amplitude_guards_period_nan_and_zero_amp(self) -> None:
        dt = 0.25
        t = np.arange(0.0, 300.0, dt)
        sig = 20_000.0 * np.sin(2.0 * math.pi * t / 20.0)
        tr = Trace(data=sig, delta=dt, station="GRA1", comp="Z", distance=50.0)

        with self.assertRaises(ValueError):
            find_surface_wave_amplitude(tr, period_lo=0.0, period_hi=22.0)
        with self.assertRaises(ValueError):
            find_surface_wave_amplitude(tr, period_lo=-5.0, period_hi=22.0)

        # Partial NaN/Inf spike still yields finite amplitude & Ms
        tr_nan = tr.copy()
        tr_nan.data[100] = np.nan
        tr_nan.data[200] = np.inf
        res_nan = find_surface_wave_amplitude(tr_nan, distance_deg=50.0)
        self.assertTrue(math.isfinite(res_nan["ampl_nm"]))
        self.assertIsNotNone(res_nan["ms"])
        assert res_nan["ms"] is not None
        self.assertTrue(math.isfinite(res_nan["ms"]))

        # Zero / all-NaN trace returns ms=None (not 0.0)
        tr_zero = Trace(data=np.zeros(400), delta=dt, station="ZERO", comp="Z", distance=50.0)
        res_zero = find_surface_wave_amplitude(tr_zero, distance_deg=50.0)
        self.assertEqual(res_zero["ampl_nm"], 0.0)
        self.assertIsNone(res_zero["ms"])

        tr_all_nan = Trace(data=np.full(400, np.nan), delta=dt, station="NAN", comp="Z", distance=50.0)
        res_all_nan = find_surface_wave_amplitude(tr_all_nan, distance_deg=50.0)
        self.assertEqual(res_all_nan["ampl_nm"], 0.0)
        self.assertIsNone(res_all_nan["ms"])

    def test_scan_surface_waves_resilient_to_short_trace(self) -> None:
        dt = 0.25
        t = np.arange(0.0, 300.0, dt)
        ray = 30_000.0 * np.exp(-0.5 * ((t - 150.0) / 30.0) ** 2) * np.sin(2.0 * math.pi * t / 20.0)
        tr1 = Trace(data=ray, delta=dt, station="GRA1", comp="Z", distance=60.0)
        tr_short = Trace(data=np.ones(4), delta=dt, station="SHORT", comp="Z", distance=60.0)
        tr2 = Trace(data=ray * 1.2, delta=dt, station="GRA2", comp="Z", distance=60.0)

        batch = scan_surface_waves([tr1, tr_short, tr2])
        self.assertEqual(batch["count"], 3)
        self.assertTrue(batch["stations"][1].get("skipped"))
        self.assertIsNone(batch["stations"][1]["ms"])
        self.assertIsNotNone(batch["ms_mean"])
        self.assertIsNotNone(batch["ms_median"])


if __name__ == "__main__":
    unittest.main()
