"""Adversarial & unit tests for M-09 (Brune & Boatwright omega^-2 spectral inversion)
and M-22 (18-22 s surface wave amplitude finder & Prague Ms) in shm_app.processing.spectrum.
"""

from __future__ import annotations

import ast
import math
import os
import unittest
import numpy as np

from shm_app.core.fit_models import FitDiagnosticResult, FitType, ParameterUncertainty, ResidualEntry
from shm_app.core.models import Trace
from shm_app.processing.spectrum import (
    build_brune_fit_diagnostic,
    compute_physical_source_parameters,
    find_surface_wave_amplitude,
    invert_brune_spectrum,
    invert_source_spectrum,
    scan_surface_waves,
    source_spectrum_model,
    subtract_noise_quadrature,
)
from shm_app.seismology.array_loc import compute_magnitude_ms


def _synthesize_brune_pulse_trace(
    omega0: float,
    fc: float,
    t_star: float = 0.03,
    model: str = "brune",
    dt: float = 0.02,
    duration_sec: float = 40.0,
    pulse_center_sec: float = 25.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> Trace:
    """Synthesize a causal-like displacement pulse whose amplitude spectrum matches
    the Brune/Boatwright source_spectrum_model inside the signal window.
    """
    n = int(round(duration_sec / dt))
    nfft = 1
    while nfft < n:
        nfft <<= 1

    freqs = np.fft.rfftfreq(nfft, d=dt)
    amp_spec = source_spectrum_model(freqs, omega0=omega0, fc=fc, t_star=t_star, model=model)
    amp_spec[0] = 0.0  # zero DC

    # Linear phase shifts pulse to pulse_center_sec; minimum-phase-like shift keeps pulse compact
    phase = -2.0 * math.pi * freqs * pulse_center_sec
    complex_spec = (amp_spec / dt) * np.exp(1j * phase)
    data = np.fft.irfft(complex_spec, n=nfft)[:n]

    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        data = data + rng.normal(0.0, noise_std, size=n)

    return Trace(data=data, delta=dt, station="GRA1", comp="Z", chan1="B", chan2="H")


class TestSpectralSourceInversionM09(unittest.TestCase):
    def test_architectural_imports_pure_numpy_no_pyqt_scipy_obspy(self) -> None:
        spec_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "shm_app", "processing", "spectrum.py")
        )
        with open(spec_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=spec_path)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for mod in imported:
            root = mod.split(".")[0]
            self.assertNotIn(root, ("PyQt6", "PyQt5", "PySide6", "scipy", "obspy"))

    def test_forward_brune_and_boatwright_models(self) -> None:
        freqs = np.array([0.0, 0.01, 1.0, 2.0, 10.0, 20.0])
        omega0 = 2000.0
        fc = 2.0

        brune = source_spectrum_model(freqs, omega0=omega0, fc=fc, t_star=0.0, model="brune")
        self.assertAlmostEqual(brune[0], omega0, places=6)
        self.assertAlmostEqual(brune[3], omega0 / 2.0, places=6)  # At f=fc, Brune = Omega0 / 2
        self.assertAlmostEqual(brune[5], omega0 / (1.0 + 100.0), places=6)

        boat = source_spectrum_model(freqs, omega0=omega0, fc=fc, t_star=0.0, model="boatwright")
        self.assertAlmostEqual(boat[0], omega0, places=6)
        self.assertAlmostEqual(boat[3], omega0 / math.sqrt(2.0), places=6)  # At f=fc, Boatwright = Omega0 / sqrt(2)
        self.assertAlmostEqual(boat[5], omega0 / math.sqrt(1.0 + 10.0 ** 4), places=6)

        # Attenuation factor exp(-pi * f * t*)
        brune_att = source_spectrum_model(freqs, omega0=omega0, fc=fc, t_star=0.05, model="brune")
        np.testing.assert_allclose(brune_att, brune * np.exp(-math.pi * freqs * 0.05), rtol=1e-10)

        with self.assertRaises(ValueError):
            source_spectrum_model(freqs, omega0=-1.0, fc=1.0)
        with self.assertRaises(ValueError):
            source_spectrum_model(freqs, omega0=100.0, fc=0.0)
        with self.assertRaises(ValueError):
            source_spectrum_model(freqs, omega0=100.0, fc=1.0, model="invalid")

    def test_direct_brune_and_boatwright_inversion_recovery(self) -> None:
        freqs = np.linspace(0.05, 20.0, 400)
        for model_name, w0_true, fc_true, ts_true in (
            ("brune", 3500.0, 1.4, 0.035),
            ("boatwright", 1800.0, 2.2, 0.025),
            ("brune", 12000.0, 0.65, 0.0),
        ):
            obs = source_spectrum_model(freqs, w0_true, fc_true, t_star=ts_true, model=model_name)
            res = invert_source_spectrum(freqs, obs, model=model_name, f_lo=0.05, f_hi=20.0)
            self.assertTrue(res["converged"])
            self.assertEqual(res["model_type"], model_name)
            self.assertAlmostEqual(res["omega0"], w0_true, delta=0.02 * w0_true)
            self.assertAlmostEqual(res["fc"], fc_true, delta=0.02 * fc_true)
            self.assertAlmostEqual(res["t_star"], ts_true, delta=0.005)
            self.assertLess(res["rms_log10"], 1e-3)

    def test_waveform_trace_inversion_with_quadrature_noise_and_low_snr_high_freq(self) -> None:
        """Adversarial test: recover Brune (Omega0, fc, t*) from a time-domain Trace with
        a pre-phase noise window where high-frequency SNR drops well below 1.0.
        """
        w0_true = 4000.0
        fc_true = 1.5
        ts_true = 0.03
        # Noise std chosen so high-frequency (> 5 Hz) signal falls below noise floor (SNR < 1.0)
        tr = _synthesize_brune_pulse_trace(
            omega0=w0_true,
            fc=fc_true,
            t_star=ts_true,
            model="brune",
            dt=0.02,
            duration_sec=50.0,
            pulse_center_sec=32.0,
            noise_std=45.0,
            seed=123,
        )

        # Verify SNR at 8 Hz is indeed < 1.0 in the signal vs noise windows
        res = invert_brune_spectrum(
            tr,
            t_lo=26.0,
            t_hi=42.0,
            noise_lo=2.0,
            noise_hi=18.0,
            model="brune",
            f_lo=0.15,
            f_hi=15.0,
        )
        # Check high-freq SNR < 1.0 was actually exercised
        hf_mask = res["freqs"] >= 7.0
        self.assertTrue(np.any(hf_mask))
        mean_hf_obs = float(np.mean(res["obs_spec"][hf_mask]))
        mean_hf_noise = float(np.mean(res["noise_spec"][hf_mask]))
        self.assertLess(mean_hf_obs / max(mean_hf_noise, 1e-12), 1.5)

        # Should recover parameters cleanly despite low-SNR high frequencies
        self.assertAlmostEqual(res["omega0"], w0_true, delta=0.20 * w0_true)
        self.assertAlmostEqual(res["fc"], fc_true, delta=0.20 * fc_true)
        self.assertAlmostEqual(res["t_star"], ts_true, delta=0.035)
        self.assertGreater(res["r_squared"], 0.85)

    def test_adversarial_low_snr_quadrature_subtraction_array(self) -> None:
        """Adversarial spectral test where high frequencies have SNR < 0.3 and random fluctuations
        cause obs < noise on many bins.
        """
        rng = np.random.default_rng(7)
        freqs = np.linspace(0.1, 20.0, 250)
        w0_true, fc_true, ts_true = 5000.0, 1.2, 0.04
        clean_sig = source_spectrum_model(freqs, w0_true, fc_true, t_star=ts_true, model="brune")

        # Flat noise floor = 120 nm*s (so above ~6 Hz, clean_sig < 120 -> SNR < 1.0; at 15 Hz SNR < 0.05)
        noise_true = np.full_like(freqs, 120.0)
        # Verify true SNR < 1.0 for > 65% of the band
        self.assertGreater(np.mean((clean_sig / noise_true) < 1.0), 0.65)

        obs_power = clean_sig ** 2 + (noise_true * (1.0 + 0.08 * rng.normal(size=freqs.size))) ** 2
        obs_spec = np.sqrt(np.maximum(0.0, obs_power))
        noise_est = noise_true * (1.0 + 0.05 * rng.normal(size=freqs.size))

        sub = subtract_noise_quadrature(obs_spec, noise_est)
        self.assertTrue(np.any(sub == 0.0))  # high-freq bins clamp cleanly to 0

        res = invert_source_spectrum(freqs, obs_spec, noise_spec=noise_est, model="brune", f_lo=0.1, f_hi=20.0)
        self.assertAlmostEqual(res["omega0"], w0_true, delta=0.10 * w0_true)
        self.assertAlmostEqual(res["fc"], fc_true, delta=0.12 * fc_true)
        self.assertAlmostEqual(res["t_star"], ts_true, delta=0.025)

    def test_physical_parameters_and_2d_misfit_contour_and_diagnostic_contract(self) -> None:
        w0 = 2450.0  # nm*s
        fc = 1.45    # Hz
        phys_s = compute_physical_source_parameters(
            omega0=w0,
            fc=fc,
            rho_kg_m3=2700.0,
            vel_m_s=3500.0,
            distance_m=100000.0,
            fs=2.0,
            r_theta_phi=0.55,
            cp=1.0,
            wave_type="S",
        )
        expected_m0 = (4.0 * math.pi * 2700.0 * (3500.0 ** 3) * 100000.0 * (w0 * 1e-9)) / (2.0 * 0.55 * 1.0)
        expected_mw = (2.0 / 3.0) * (math.log10(expected_m0) - 9.1)
        expected_r0 = 0.37 * 3500.0 / fc
        expected_ds_pa = (7.0 / 16.0) * expected_m0 / (expected_r0 ** 3)

        self.assertAlmostEqual(phys_s["m0"], expected_m0, places=4)
        self.assertAlmostEqual(phys_s["mw"], expected_mw, places=6)
        self.assertAlmostEqual(phys_s["r0_m"], expected_r0, places=6)
        self.assertAlmostEqual(phys_s["stress_drop_pa"], expected_ds_pa, places=4)
        self.assertAlmostEqual(phys_s["stress_drop_bar"], expected_ds_pa * 1e-5, places=6)

        # P-wave kc=0.21 check & SI unit check
        phys_p = compute_physical_source_parameters(
            omega0=w0 * 1e-9,
            fc=fc,
            vel_m_s=6000.0,
            vs_m_s=3500.0,
            wave_type="P",
            omega0_unit="m*s",
        )
        self.assertAlmostEqual(phys_p["r0_m"], 0.21 * 3500.0 / fc, places=6)

        # Inversion + 2D contour + FitDiagnosticResult contract
        freqs = np.logspace(-1, 1.3, 120)
        obs = source_spectrum_model(freqs, w0, fc, t_star=0.02, model="boatwright")
        inv = invert_source_spectrum(freqs, obs, model="boatwright", f_lo=0.1, f_hi=20.0, n_grid=31)

        fc_g = inv["fc_grid"]
        w0_g = inv["omega0_grid"]
        mg = inv["misfit_grid"]
        self.assertEqual(mg.shape, (w0_g.size, fc_g.size))
        min_ij = np.unravel_index(int(np.argmin(mg)), mg.shape)
        self.assertAlmostEqual(float(w0_g[min_ij[0]]), inv["omega0"], delta=0.02 * inv["omega0"])
        self.assertAlmostEqual(float(fc_g[min_ij[1]]), inv["fc"], delta=0.02 * inv["fc"])

        diag = build_brune_fit_diagnostic(inv, station="GRFO", phase="S")
        self.assertIsInstance(diag, FitDiagnosticResult)
        self.assertEqual(diag.fit_type, FitType.BRUNE_SPECTRUM)
        self.assertTrue(diag.converged)

        required_curves = {"freqs", "obs_spec", "model_spec", "noise_spec", "fc_grid", "omega0_grid", "misfit_grid"}
        self.assertEqual(set(diag.overlay_curves.keys()), required_curves)
        for k, arr in diag.overlay_curves.items():
            self.assertIsInstance(arr, np.ndarray, msg=f"{k} must be np.ndarray")

        required_ctx = {
            "model_type", "omega0", "fc", "t_star", "m0", "mw",
            "r0_m", "stress_drop_pa", "stress_drop_bar", "rms_log10",
        }
        self.assertTrue(required_ctx.issubset(set(diag.context_data.keys())))
        self.assertEqual(diag.context_data["model_type"], "boatwright")

        for pname in ("omega0", "fc", "t_star"):
            self.assertIn(pname, diag.parameters)
            pu = diag.parameters[pname]
            self.assertIsInstance(pu, ParameterUncertainty)
            self.assertTrue(pu.confidence_interval_95[0] <= pu.value <= pu.confidence_interval_95[1])

        self.assertGreater(len(diag.residuals), 10)
        self.assertIsInstance(diag.residuals[0], ResidualEntry)
        self.assertEqual(diag.residuals[0].station, "GRFO")
        self.assertEqual(diag.residuals[0].phase, "S")


class TestSurfaceWaveFinderM22(unittest.TestCase):
    def test_recover_20s_rayleigh_amplitude_period_and_prague_ms_with_clutter(self) -> None:
        dt = 0.25  # 4 Hz sampling, 400 s long trace
        t = np.arange(0.0, 400.0, dt)
        true_period = 20.0
        true_ampl_nm = 50_000.0  # 50 microns zero-to-peak
        dist_deg = 65.0

        # Gaussian-modulated 20 s Rayleigh wave packet centered at t=220 s
        env = np.exp(-0.5 * ((t - 220.0) / 35.0) ** 2)
        rayleigh = true_ampl_nm * env * np.sin(2.0 * math.pi * t / true_period)

        # Strong out-of-band clutter: 1 Hz body wave (200,000 nm), 4 s microseism (80,000 nm), linear drift
        clutter = (
            200_000.0 * np.exp(-0.5 * ((t - 70.0) / 5.0) ** 2) * np.sin(2.0 * math.pi * 1.0 * t)
            + 80_000.0 * np.sin(2.0 * math.pi * 0.25 * t)
            + 10_000.0
            + 50.0 * t
        )

        tr = Trace(data=rayleigh + clutter, delta=dt, station="GRFO", comp="Z")
        res = find_surface_wave_amplitude(tr, distance_deg=dist_deg)

        self.assertAlmostEqual(res["ampl_nm"], true_ampl_nm, delta=0.05 * true_ampl_nm)
        self.assertAlmostEqual(res["ampl_p2p_nm"], 2.0 * true_ampl_nm, delta=0.10 * true_ampl_nm)
        self.assertAlmostEqual(res["period"], true_period, delta=0.5)
        self.assertTrue(18.0 <= res["period"] <= 22.0)
        self.assertAlmostEqual(res["time_sec"], 220.0, delta=15.0)

        expected_ms = compute_magnitude_ms(true_ampl_nm, true_period, dist_deg)
        self.assertIsNotNone(res["ms"])
        self.assertAlmostEqual(res["ms"], expected_ms, delta=0.04)

    def test_surface_wave_velocity_input_window_and_batch_scan(self) -> None:
        dt = 0.5
        t = np.arange(0.0, 360.0, dt)
        period = 19.0
        disp_ampl_nm = 25_000.0
        # Velocity amplitude for displacement A * sin(w t) is A * w
        vel_ampl = disp_ampl_nm * (2.0 * math.pi / period)
        env = np.exp(-0.5 * ((t - 180.0) / 30.0) ** 2)
        vel_sig = vel_ampl * env * np.sin(2.0 * math.pi * t / period)

        tr_z = Trace(data=vel_sig, delta=dt, station="BFO", comp="Z", distance=50.0)
        tr_n = Trace(data=vel_sig * 0.3, delta=dt, station="BFO", comp="N", distance=50.0)

        res_vel = find_surface_wave_amplitude(tr_z, t_lo=120.0, t_hi=250.0, input_type="velocity")
        self.assertAlmostEqual(res_vel["ampl_nm"], disp_ampl_nm, delta=0.06 * disp_ampl_nm)
        self.assertAlmostEqual(res_vel["period"], period, delta=0.6)
        self.assertIsNotNone(res_vel["ms"])

        # None distance returns ms=None without error
        tr_nodist = Trace(data=vel_sig, delta=dt, station="WET", comp="Z")
        res_nodist = find_surface_wave_amplitude(tr_nodist, distance_deg=None, input_type="velocity")
        self.assertIsNone(res_nodist["ms"])

        # Batch scan defaults to vertical Z traces
        batch = scan_surface_waves([tr_z, tr_n], input_type="velocity")
        self.assertEqual(batch["count"], 1)
        self.assertEqual(batch["stations"][0]["component"], "Z")
        self.assertIsNotNone(batch["ms_mean"])
        self.assertAlmostEqual(batch["ms_mean"], res_vel["ms"], places=6)

        # Multi-station explicit distances list
        batch_list = scan_surface_waves([tr_z, tr_z], distances_deg=[40.0, 60.0], input_type="velocity")
        self.assertEqual(batch_list["count"], 2)
        self.assertGreater(batch_list["stations"][1]["ms"], batch_list["stations"][0]["ms"])

    def test_extreme_frequency_ratios_velocity_brune_and_all_noise_fallback(self) -> None:
        # 1. Extreme f/fc ratios without overflow/underflow warnings or NaNs
        f_ext = np.array([1e-5, 1e-2, 1.0, 1e3, 1e6])
        spec_ext = source_spectrum_model(f_ext, omega0=1000.0, fc=1.0, t_star=0.0, model="boatwright")
        self.assertTrue(np.all(np.isfinite(spec_ext)))
        self.assertAlmostEqual(spec_ext[0], 1000.0, places=6)

        # 2. All-noise spectrum where obs < noise everywhere: graceful fallback without NaN/crash
        freqs = np.linspace(0.1, 10.0, 50)
        obs_low = np.full_like(freqs, 10.0)
        noise_high = np.full_like(freqs, 50.0)
        res_fallback = invert_source_spectrum(freqs, obs_low, noise_spec=noise_high, f_lo=0.1, f_hi=10.0)
        self.assertTrue(math.isfinite(res_fallback["omega0"]))
        self.assertTrue(math.isfinite(res_fallback["fc"]))
        self.assertTrue(math.isfinite(res_fallback["t_star"]))

        # 3. Velocity-input Brune waveform inversion (input_type="velocity")
        dt = 0.02
        n = 2048
        nfft = 2048
        f_v = np.fft.rfftfreq(nfft, d=dt)
        w0_true, fc_true, ts_true = 3000.0, 1.5, 0.02
        disp_spec = source_spectrum_model(f_v, w0_true, fc_true, t_star=ts_true, model="brune")
        vel_spec = disp_spec * (2.0 * math.pi * f_v)
        vel_spec[0] = 0.0
        phase = -2.0 * math.pi * f_v * 20.0
        vel_data = np.fft.irfft((vel_spec / dt) * np.exp(1j * phase), n=nfft)[:n]
        tr_vel = Trace(data=vel_data, delta=dt, station="MOX", comp="Z")

        res_v = invert_brune_spectrum(
            tr_vel, t_lo=14.0, t_hi=28.0, model="brune", f_lo=0.2, f_hi=15.0, input_type="velocity"
        )
        self.assertAlmostEqual(res_v["omega0"], w0_true, delta=0.12 * w0_true)
        self.assertAlmostEqual(res_v["fc"], fc_true, delta=0.12 * fc_true)
        self.assertAlmostEqual(res_v["t_star"], ts_true, delta=0.02)


if __name__ == "__main__":
    unittest.main()

