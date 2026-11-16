"""Intensive tests for signal processing, filters, 3-component polarization, spectrum,
travel-time tables, array processing (BEAM, VESPA, FK), plane-wave & Geiger location, and magnitudes.
"""

from __future__ import annotations

import math
import os
import unittest
import numpy as np

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.io.gse import read_gse_file
from shm_app.processing.filters import (
    FilterStage,
    STANDARD_SIMULATION_FILTERS,
    apply_fft_filter,
    design_butterworth_stages,
    evaluate_stage_freq,
    filter_trace,
    load_named_filter,
    parse_filter_file,
    restitute_ground_displacement_nm,
)
from shm_app.processing.polarization import (
    apply_polarization_filter,
    compute_mdir,
    rotate_2d,
    rotate_3d_lqt,
    rotate_STS2_zne_to_uvw,
)
from shm_app.processing.spectrum import (
    compute_fft_spectrum,
    compute_mem_spectrum,
    compute_spectrogram,
    compute_welch_power_spectrum,
    fit_log_log_spectrum,
)
from shm_app.processing.trace_ops import (
    attenuate_trace,
    create_synthetic_trace,
    cut_trace,
    decimate_trace,
    demean_trace,
    despike_trace,
    detrend_trace,
    differentiate_trace,
    envelope_trace,
    hilbert_trace,
    instantaneous_phase_trace,
    integrate_trace,
    mend_trace,
    phase_weighted_stack,
    resample_trace,
    spectral_division_deconvolve,
    spiking_filter_levinson,
    sum_traces,
)
from shm_app.seismology.array_loc import (
    align_traces_on_phase,
    beam_delays_sec,
    compute_beam,
    compute_fk_analysis,
    compute_magnitude_mb,
    compute_magnitude_mbb,
    compute_magnitude_ml,
    compute_magnitude_ms,
    compute_vespagram,
    correlate_traces,
    corrpick_array,
    locate_by_slowness_azimuth,
    locate_network_geiger,
    measure_amplitude_and_period,
    pick_onset_baer_kradolfer,
    plane_wave_locate,
)
from shm_app.seismology.traveltime import get_ttt_manager, sphere_add, sphere_diff


class TestFiltersAndTraceOps(unittest.TestCase):
    def test_parse_real_flf_and_flr_files_and_simulation_filters(self) -> None:
        cfg = get_config()
        flf_path = os.path.join(cfg.filter_dir, "G_WWSSN_SP.FLF")
        stages_flf = parse_filter_file(flf_path)
        self.assertGreaterEqual(len(stages_flf), 1)
        self.assertEqual(stages_flf[0].store_id, 1)

        flr_path = os.path.join(cfg.filter_dir, "20HZ_GRSN_S+G_WWSSN_SP.FLR")
        stages_flr = parse_filter_file(flr_path)
        self.assertGreaterEqual(len(stages_flr), 1)
        self.assertEqual(stages_flr[0].store_id, 3)

        tr = create_synthetic_trace("KUEPPER", length_sec=40.0, dt=0.05, pos_sec=10.0, ampl=500.0, width=1.5)
        tr.station = "GRA1"
        for sim_name in STANDARD_SIMULATION_FILTERS:
            f_tr = filter_trace(tr, sim_name, config=cfg)
            self.assertEqual(f_tr.length, tr.length)
            self.assertFalse(np.any(np.isnan(f_tr.data)))
            self.assertGreater(float(np.max(np.abs(f_tr.data))), 0.0)

    def test_butterworth_bandpass_and_lpc_recursive(self) -> None:
        # 1 Hz signal + 8 Hz noise sampled at 20 Hz
        dt = 0.05
        t = np.arange(800) * dt
        sig_1hz = 100.0 * np.sin(2.0 * math.pi * 1.0 * t)
        sig_8hz = 100.0 * np.sin(2.0 * math.pi * 8.0 * t)
        tr = Trace(data=sig_1hz + sig_8hz, delta=dt, station="BFO")

        # Lowpass at 2.5 Hz should strongly attenuate 8 Hz while preserving 1 Hz
        lp_tr = filter_trace(tr, "BUTTER LP 2.5 4", mode="REC", use_lpc=True)
        residual_mid = lp_tr.data[200:600]
        self.assertGreater(float(np.std(residual_mid)), 50.0)
        self.assertLess(float(np.std(residual_mid)), 85.0)

    def test_all_trace_operations(self) -> None:
        tr = create_synthetic_trace("GAUSS", length_sec=30.0, dt=0.05, pos_sec=15.0, ampl=200.0, width=1.0)
        tr.data += 25.0 + np.linspace(0.0, 10.0, tr.length)

        dm = demean_trace(tr)
        self.assertAlmostEqual(float(np.mean(dm.data)), 0.0, places=5)

        dt_tr = detrend_trace(tr)
        self.assertAlmostEqual(float(np.mean(dt_tr.data)), 0.0, places=4)

        # Inject spike and despike
        sp_tr = dt_tr.copy()
        sp_tr.data[100] = 50000.0
        dsp_tr, scnt, _ = despike_trace(sp_tr, critmul=5.0)
        self.assertGreaterEqual(scnt, 1)
        self.assertLess(abs(dsp_tr.data[100]), 500.0)

        # Resample & decimate
        res = resample_trace(dt_tr, 0.025)
        self.assertAlmostEqual(res.delta, 0.025)
        dec = decimate_trace(res, 2)
        self.assertAlmostEqual(dec.delta, 0.05)

        # Cut
        cut = cut_trace(dt_tr, 10.0, 20.0)
        self.assertAlmostEqual(cut.duration, 10.0, delta=0.1)

        # Integrate & differentiate
        integ = integrate_trace(dt_tr)
        deriv = differentiate_trace(integ, mode=1)
        corr = np.corrcoef(dt_tr.data[10:-10], deriv.data[10:-10])[0, 1]
        self.assertGreater(corr, 0.98)

        # Hilbert, envelope, instantaneous phase
        env = envelope_trace(dt_tr)
        self.assertTrue(np.all(env.data >= 0.0))
        hilb = hilbert_trace(dt_tr)
        sph = instantaneous_phase_trace(dt_tr)
        self.assertEqual(hilb.length, dt_tr.length)
        self.assertEqual(sph.length, dt_tr.length)

        # Sum, PWS, mend, attenuate, spiking, deconv
        st_sum = sum_traces([dt_tr, dt_tr])
        self.assertIsNotNone(st_sum)
        pws = phase_weighted_stack([dt_tr, dt_tr])
        self.assertIsNotNone(pws)
        mended = mend_trace(dt_tr, 50, 55)
        self.assertEqual(mended.length, dt_tr.length)
        att = attenuate_trace(dt_tr, t_star=0.5)
        self.assertEqual(att.length, dt_tr.length)
        spk_f = spiking_filter_levinson(dt_tr.data, filter_len=30)
        self.assertEqual(spk_f.size, 30)
        deconv = spectral_division_deconvolve(dt_tr, dt_tr)
        self.assertEqual(deconv.length, dt_tr.length)


class TestPolarizationAndSpectrum(unittest.TestCase):
    def test_3d_rotation_mdir_and_polarization_filter(self) -> None:
        # Synthesize a pure rectilinear P wave arriving from backazimuth=60 deg, incidence=25 deg
        baz_true = 60.0
        inc_true = 25.0
        wave = create_synthetic_trace("KUEPPER", length_sec=20.0, dt=0.05, pos_sec=8.0, ampl=100.0, width=1.5).data

        az_r = math.radians(baz_true)
        inc_r = math.radians(inc_true)
        # Upgoing P-wave particle motion points UP (+Z) and AWAY from source (-N*cos(baz), -E*sin(baz))
        z_dat = wave * math.cos(inc_r)
        n_dat = -wave * math.sin(inc_r) * math.cos(az_r)
        e_dat = -wave * math.sin(inc_r) * math.sin(az_r)

        tz = Trace(data=z_dat, delta=0.05, station="BFO", comp="Z")
        tn = Trace(data=n_dat, delta=0.05, station="BFO", comp="N")
        te = Trace(data=e_dat, delta=0.05, station="BFO", comp="E")

        mdir = compute_mdir(tz, tn, te, 7.0, 10.0)
        self.assertAlmostEqual(mdir["azimuth"], baz_true, delta=0.5)
        self.assertAlmostEqual(mdir["incidence"], inc_true, delta=0.5)
        self.assertGreater(mdir["rectilinearity"], 0.99)

        # Rotate into L-Q-T at (baz_true, inc_true): all energy should concentrate in L, while Q and T vanish
        tl, tq, tt = rotate_3d_lqt(tz, tn, te, baz_true, inc_true)
        self.assertGreater(float(np.max(np.abs(tl.data))), 95.0)
        self.assertLess(float(np.max(np.abs(tq.data))), 1e-6)
        self.assertLess(float(np.max(np.abs(tt.data))), 1e-6)

        # STS-2 ZNE <-> UVW round-trip
        tu, tv, tw = rotate_STS2_zne_to_uvw(tz, tn, te, inverse=False)
        bz, bn, be = rotate_STS2_zne_to_uvw(tu, tv, tw, inverse=True)
        np.testing.assert_allclose(bz.data, tz.data, atol=1e-9)
        np.testing.assert_allclose(bn.data, tn.data, atol=1e-9)
        np.testing.assert_allclose(be.data, te.data, atol=1e-9)

        # Polarization filter modes 1..5
        for mode in (1, 2, 3, 4, 5):
            pz, pn, pe = apply_polarization_filter(tz, tn, te, coherence_win_sec=0.5, mode=mode, baz_deg=baz_true, inci_deg=inc_true)
            self.assertEqual(pz.length, tz.length)

    def test_spectral_estimators_and_spectrogram(self) -> None:
        tr = create_synthetic_trace("SIN", length_sec=40.0, dt=0.05, freq_hz=2.5, ampl=50.0)
        f_fft, a_fft = compute_fft_spectrum(tr)
        f_peak_fft = float(f_fft[int(np.argmax(a_fft))])
        self.assertAlmostEqual(f_peak_fft, 2.5, delta=0.05)

        f_psd, a_psd = compute_welch_power_spectrum(tr)
        self.assertAlmostEqual(float(f_psd[int(np.argmax(a_psd))]), 2.5, delta=0.1)

        f_mem, a_mem = compute_mem_spectrum(tr, order=20)
        self.assertAlmostEqual(float(f_mem[int(np.argmax(a_mem))]), 2.5, delta=0.1)

        fit = fit_log_log_spectrum(f_fft, a_fft, 0.5, 5.0)
        self.assertIn("slope", fit)

        times, freqs, spec_db = compute_spectrogram(tr, win_length_sec=2.0)
        self.assertEqual(spec_db.shape, (freqs.size, times.size))


class TestSeismologyArrayAndLocation(unittest.TestCase):
    def test_ttt_tables_slowness_diff_and_depth_phases(self) -> None:
        ttt = get_ttt_manager()
        self.assertIn("P", ttt.available_phases())
        self.assertIn("pP", ttt.available_phases())
        self.assertIn("S", ttt.available_phases())

        t_p = ttt.travel_time("P", 60.0, 33.0)
        self.assertIsNotNone(t_p)
        self.assertTrue(580.0 <= t_p <= 630.0)

        slo_p = ttt.slowness("P", 60.0, 33.0)
        self.assertIsNotNone(slo_p)
        self.assertTrue(6.0 <= slo_p <= 7.5)

        d_inv = ttt.distance_from_slowness("P", slo_p, 33.0)
        self.assertAlmostEqual(d_inv, 60.0, delta=1.5)

        # S-P differential distance
        t_s = ttt.travel_time("S", 45.0, 33.0)
        t_p45 = ttt.travel_time("P", 45.0, 33.0)
        d_sp = ttt.distance_from_diff_time("P", "S", t_s - t_p45, 33.0)
        self.assertAlmostEqual(d_sp, 45.0, delta=1.0)

        # Depth phase pP-P inversion for depth=120 km at dist=70 deg
        t_p70 = ttt.travel_time("P", 70.0, 120.0)
        t_pp70 = ttt.travel_time("pP", 70.0, 120.0)
        z_fit, _ = ttt.depth_from_depth_phase("P", "pP", t_pp70 - t_p70, 70.0)
        self.assertAlmostEqual(z_fit, 120.0, delta=3.0)

    def test_spherical_geometry_and_array_plane_wave_beam_fk_vespa(self) -> None:
        cfg = get_config()
        lat2, lon2 = sphere_add(49.69, 11.22, 55.0, 42.0)
        dist, az12, _baz = sphere_diff(49.69, 11.22, lat2, lon2)
        self.assertAlmostEqual(dist, 55.0, delta=0.3)
        self.assertAlmostEqual(az12, 42.0, delta=0.5)

        # Synthesize GRF array traces for a plane wave coming from BAZ=115 deg, slowness=7.2 s/deg
        grf_stations = ["GRA1", "GRA2", "GRA3", "GRA4", "GRB1", "GRB2", "GRB3", "GRB4", "GRB5", "GRC1", "GRC2", "GRC3", "GRC4"]
        coords = cfg.relative_xy_km(grf_stations, ref_station="CENTRE")
        x_km = [c[0] for c in coords]
        y_km = [c[1] for c in coords]
        true_baz = 115.0
        true_slo = 7.2
        delays = beam_delays_sec(x_km, y_km, true_baz, true_slo)

        base_ep = parse_sh_time("02-NOV-2003_02:25:00.000")
        ref_onset_rel = 15.0
        arrival_epochs = [base_ep + ref_onset_rel - float(d) for d in delays]

        loc_res = plane_wave_locate(arrival_epochs, x_km, y_km)
        self.assertAlmostEqual(loc_res["azimuth"], true_baz, delta=0.2)
        self.assertAlmostEqual(loc_res["slowness"], true_slo, delta=0.1)

        # Build synthetic waveform array shifted by -delays and test FK, Beam, Vespa, CorrPick
        traces: List[Trace] = []
        for st_code, d_sec, arr_ep in zip(grf_stations, delays, arrival_epochs):
            tr = create_synthetic_trace(
                "KUEPPER",
                length_sec=40.0,
                dt=0.05,
                pos_sec=ref_onset_rel - float(d_sec),
                ampl=200.0,
                width=1.0,
            )
            tr.station = st_code
            tr.start_epoch = base_ep
            if st_code == "GRA1":
                tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(arr_ep), station=st_code))
            traces.append(tr)

        # Cross-correlation pick across array starting from GRA1
        pks = corrpick_array(traces, ref_trace_idx=0, phase_name="P", max_lag_sec=5.0)
        self.assertEqual(len(pks), len(grf_stations))

        # Wideband FK analysis
        fk_res = compute_fk_analysis(traces, t_lo=10.0, t_hi=22.0, f_lo=0.4, f_hi=2.5, max_slowness=12.0, grid_points=41, config=cfg)
        self.assertAlmostEqual(fk_res["azimuth"], true_baz, delta=3.0)
        self.assertAlmostEqual(fk_res["slowness"], true_slo, delta=0.5)
        self.assertGreater(fk_res["semblance"], 0.85)

        # Beamforming & Vespagram
        beam_tr, _ = compute_beam(traces, true_baz, true_slo, config=cfg)
        self.assertGreater(float(np.max( np.abs(beam_tr.data) )), 180.0)

        v_trs, _times, _slos, v_mat = compute_vespagram(traces, true_baz, slow_lo=4.0, slow_hi=10.0, slow_step=1.0, config=cfg)
        self.assertEqual(len(v_trs), 7)

    def test_network_geiger_location_and_magnitudes(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager()
        true_lat, true_lon, true_z = 50.2, 12.4, 15.0
        t0_ep = parse_sh_time("02-NOV-2003_02:24:00.000")

        stations = ["BFO", "WET", "CLZ", "FUR", "MOX", "GRFO", "TNS", "BRG"]
        picks: List[PhasePick] = []
        for st_code in stations:
            st = cfg.get_station(st_code)
            d_deg, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = ttt.travel_time("P", d_deg, true_z)
            picks.append(PhasePick(name="P", onset=format_sh_time(t0_ep + tt), weight=3, station=st_code))

        sol = locate_network_geiger(picks, depth_km=true_z, free_depth=False, config=cfg, ttt=ttt)
        self.assertAlmostEqual(sol.lat, true_lat, delta=0.15)
        self.assertAlmostEqual(sol.lon, true_lon, delta=0.15)

        # Magnitudes ML, mb, MS, mB
        ml = compute_magnitude_ml(1500.0, 3.0)
        mb = compute_magnitude_mb(500.0, 1.2, 65.0, 33.0)
        ms = compute_magnitude_ms(2000.0, 20.0, 65.0)
        mbb = compute_magnitude_mbb(3000.0, 65.0, 33.0)
        self.assertTrue(2.0 <= ml <= 6.5)
        self.assertTrue(4.0 <= mb <= 7.5)
        self.assertTrue(4.0 <= ms <= 7.5)
        self.assertTrue(4.0 <= mbb <= 7.5)


class TestFilterDSPAndRestitutionEdgeCases(unittest.TestCase):
    """Direct verification of FFT filtering edge cases, +/- filter chain parsing,
    tabulated/inverted frequency responses, and displacement restitution accuracy.
    """

    def test_apply_fft_filter_parameters_and_edge_conditions(self) -> None:
        """Verify apply_fft_filter:
        - Passband transmission and stopband attenuation
        - Parameter variations: taper_frac (0.0, 0.05, 0.1), demean_first (True, False)
        - Edge conditions: empty array, empty stages, short array (n <= 10)
        - DC bin suppression: output mean is near 0
        """
        dt = 0.02  # 50 Hz
        t = np.arange(500) * dt
        # Composite signal: 1.0 Hz in passband, 10.0 Hz in stopband
        sig = np.sin(2.0 * math.pi * 1.0 * t) + np.sin(2.0 * math.pi * 10.0 * t)
        stgs = design_butterworth_stages("LP", order=4, freq_lo=2.5, dt=dt)

        # 1. Passband vs stopband transmission
        out = apply_fft_filter(sig, dt, stgs, taper_frac=0.05, demean_first=True)
        self.assertEqual(out.size, sig.size)
        spec = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(out.size, d=dt)
        idx_1 = np.argmin(np.abs(freqs - 1.0))
        idx_10 = np.argmin(np.abs(freqs - 10.0))
        # 10.0 Hz component attenuated by > 95% compared to 1.0 Hz component
        self.assertLess(spec[idx_10], 0.05 * spec[idx_1])

        # 2. Parameter variations: taper_frac
        out_notaper = apply_fft_filter(sig, dt, stgs, taper_frac=0.0, demean_first=True)
        out_taper10 = apply_fft_filter(sig, dt, stgs, taper_frac=0.1, demean_first=True)
        self.assertEqual(out_notaper.size, sig.size)
        self.assertEqual(out_taper10.size, sig.size)

        # 3. Parameter variations: demean_first
        # With DC offset + sinusoid: demean_first=True centers trace at 0
        sig_dc = 100.0 + 5.0 * np.sin(2.0 * math.pi * 1.0 * t)
        out_dm = apply_fft_filter(sig_dc, dt, stgs, demean_first=True)
        self.assertAlmostEqual(float(np.mean(out_dm)), 0.0, places=1)

        # 4. DC bin suppression (total_resp[0] = 0.0):
        # A zero-mean sinusoid filtered with demean_first=False has mean near zero (~0.0015)
        out_nodm = apply_fft_filter(sig, dt, stgs, demean_first=False)
        self.assertAlmostEqual(float(np.mean(out_nodm)), 0.0, places=2)
        # And a pure DC signal with demean_first=True results in all zeros
        out_pure_dc = apply_fft_filter(np.full(200, 100.0), dt, stgs, demean_first=True)
        np.testing.assert_array_equal(out_pure_dc, np.zeros(200))

        # 5. Edge cases: empty array
        empty_in = np.array([])
        empty_out = apply_fft_filter(empty_in, dt, stgs)
        self.assertEqual(empty_out.size, 0)

        # 6. Edge cases: empty stages
        no_stg_out = apply_fft_filter(sig, dt, [])
        np.testing.assert_array_equal(no_stg_out, sig)

        # 7. Edge cases: short array (n <= 10) bypassing cosine taper
        short_in = np.array([1.0, 2.5, 4.0, 3.0, 1.5, 0.0, -1.0, -0.5])
        short_out = apply_fft_filter(short_in, dt, stgs)
        self.assertEqual(short_out.size, short_in.size)
        self.assertFalse(np.isnan(short_out).any())

    def test_load_named_filter_chains_with_plus_and_minus(self) -> None:
        """Verify load_named_filter chain parsing:
        - Exact file match precedence (e.g. 20HZ_GRSN_S+G_WWSSN_SP.FLR)
        - Inverted tokens (-GRSN+WWSSN_SP, WWSSN_SP-WWSSN_LP, -WWSSN_SP-WWSSN_LP)
        - Butterworth shorthand (BP 0.5 2.0 3, BUTTER LP 2.0 4, HP 1.0 2)
        - Shorthand 'NONE' and empty string returning empty list
        """
        # 1. Exact catalog file match with '+' in name (not split)
        exact_stages = load_named_filter("20HZ_GRSN_S+G_WWSSN_SP.FLR")
        self.assertGreater(len(exact_stages), 0)
        self.assertTrue(all(s.store_id == 3 for s in exact_stages))

        # 2. Chain expression with leading '-' (deconvolve GRSN, simulate WWSSN_SP)
        chain_inv_fwd = load_named_filter("-GRSN+WWSSN_SP")
        self.assertGreaterEqual(len(chain_inv_fwd), 2)
        self.assertIn("INV", chain_inv_fwd[0].comment)

        # 3. Chain expression with middle '-' (forward WWSSN_SP, deconvolve WWSSN_LP)
        chain_fwd_inv = load_named_filter("WWSSN_SP-WWSSN_LP")
        self.assertGreaterEqual(len(chain_fwd_inv), 2)
        self.assertIn("INV", chain_fwd_inv[-1].comment)

        # 4. Chain expression with both '-' (both inverted)
        chain_both_inv = load_named_filter("-WWSSN_SP-WWSSN_LP")
        self.assertGreaterEqual(len(chain_both_inv), 2)
        self.assertTrue(all("INV" in s.comment for s in chain_both_inv))

        # 5. Butterworth shorthand expressions
        bw_bp = load_named_filter("BP 0.5 2.0 3")
        self.assertGreater(len(bw_bp), 0)
        bw_lp = load_named_filter("BUTTER LP 2.0 4")
        self.assertGreater(len(bw_lp), 0)
        bw_hp = load_named_filter("HP 1.0 2")
        self.assertGreater(len(bw_hp), 0)

        # 6. 'NONE' and empty expression
        self.assertEqual(load_named_filter("NONE"), [])
        self.assertEqual(load_named_filter(""), [])

    def test_evaluate_stage_freq_store_id_2_and_inverted_stages(self) -> None:
        """Verify evaluate_stage_freq:
        - store_id == 2: tabulated frequency response linear interpolation and out-of-band clamping
        - store_id == 2: FilterStage.inverted() satisfies H * H_inv == 1.0 + 0j at tabulated frequencies
        - store_id == 1 (.FLF): FilterStage.inverted() satisfies H * H_inv == 1.0 + 0j across continuous passband
        - store_id == 3 (.FLR): FilterStage.inverted() satisfies H * H_inv == 1.0 + 0j across continuous passband
        - Real .FLT file (KS54_S+G_WWSSN_SP.FLT) evaluation and inversion
        """
        cfg = get_config()

        # 1. Synthetic store_id == 2 stage
        stg2 = FilterStage(
            store_id=2,
            norm=1.0,
            freqs_hz=[0.1, 1.0, 10.0],
            amplitudes=[0.5, 2.0, 0.2],
            phases_deg=[90.0, 0.0, -90.0],
            comment="SYNTH_TAB",
        )
        # Midpoint interpolation at 0.55 Hz
        w_mid = np.array([2.0 * math.pi * 0.55])
        h_mid = evaluate_stage_freq(stg2, w_mid)
        expected_amp = (0.5 + 2.0) / 2.0  # 1.25
        expected_phase = math.radians((90.0 + 0.0) / 2.0)  # 45 deg
        self.assertAlmostEqual(abs(h_mid[0]), expected_amp, places=6)
        self.assertAlmostEqual(np.angle(h_mid[0]), expected_phase, places=6)

        # Out-of-band clamping to 0.0
        w_oob = 2.0 * math.pi * np.array([0.01, 25.0])
        h_oob = evaluate_stage_freq(stg2, w_oob)
        self.assertEqual(h_oob[0], 0.0)
        self.assertEqual(h_oob[1], 0.0)

        # Inversion of synthetic store_id == 2 stage at tabulated frequencies
        inv2 = stg2.inverted()
        self.assertEqual(inv2.store_id, 2)
        self.assertIn("INV", inv2.comment)
        w_nodes = 2.0 * math.pi * np.array(stg2.freqs_hz)
        h_nodes = evaluate_stage_freq(stg2, w_nodes)
        h_inv_nodes = evaluate_stage_freq(inv2, w_nodes)
        prod2 = h_nodes * h_inv_nodes
        np.testing.assert_allclose(np.abs(prod2), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.angle(prod2), 0.0, atol=1e-12)

        # 2. Real catalog .FLT file (KS54_S+G_WWSSN_SP.FLT)
        flt_path = os.path.join(cfg.filter_dir, "KS54_S+G_WWSSN_SP.FLT")
        if os.path.isfile(flt_path):
            flt_stages = parse_filter_file(flt_path)
            self.assertEqual(len(flt_stages), 1)
            flt_stg = flt_stages[0]
            self.assertEqual(flt_stg.store_id, 2)
            flt_inv = flt_stg.inverted()
            active_nodes = [f for f, a in zip(flt_stg.freqs_hz, flt_stg.amplitudes) if a > 1e-10]
            w_real_nodes = 2.0 * math.pi * np.array(active_nodes)
            h_real = evaluate_stage_freq(flt_stg, w_real_nodes)
            h_real_inv = evaluate_stage_freq(flt_inv, w_real_nodes)
            np.testing.assert_allclose(np.abs(h_real * h_real_inv), 1.0, atol=1e-12)
            np.testing.assert_allclose(np.angle(h_real * h_real_inv), 0.0, atol=1e-12)

        # 3. store_id == 1 (.FLF s-plane poles/zeros) inversion
        flf_path = os.path.join(cfg.filter_dir, "G_WWSSN_SP.FLF")
        if os.path.isfile(flf_path):
            flf_stages = parse_filter_file(flf_path)
            flf_stg = flf_stages[0]
            self.assertEqual(flf_stg.store_id, 1)
            flf_inv = flf_stg.inverted()
            w_cont = 2.0 * math.pi * np.linspace(0.1, 10.0, 50)
            h1 = evaluate_stage_freq(flf_stg, w_cont)
            h1_inv = evaluate_stage_freq(flf_inv, w_cont)
            np.testing.assert_allclose(np.abs(h1 * h1_inv), 1.0, atol=1e-12)
            np.testing.assert_allclose(np.angle(h1 * h1_inv), 0.0, atol=1e-12)

        # 4. store_id == 3 (.FLR discrete recursive IIR) inversion
        flr_path = os.path.join(cfg.filter_dir, "20HZ_G_WWSSN_SP.FLR")
        if os.path.isfile(flr_path):
            flr_stages = parse_filter_file(flr_path)
            flr_stg = flr_stages[0]
            self.assertEqual(flr_stg.store_id, 3)
            flr_inv = flr_stg.inverted()
            w_flr = 2.0 * math.pi * np.linspace(0.1, 9.0, 50)
            h3 = evaluate_stage_freq(flr_stg, w_flr)
            h3_inv = evaluate_stage_freq(flr_inv, w_flr)
            np.testing.assert_allclose(np.abs(h3 * h3_inv), 1.0, atol=1e-12)
            np.testing.assert_allclose(np.angle(h3 * h3_inv), 0.0, atol=1e-12)

    def test_restitute_ground_displacement_nm_and_flf_vs_flr_consistency(self) -> None:
        """Verify restitute_ground_displacement_nm and continuous/discrete consistency:
        - Velocity-to-displacement conversion: A_disp = A_vel * T / (2*pi)
        - Displacement simulation filter bypass (WWSSN_SP, KIRNOS, etc.) returns |A_raw|
        - Non-positive period fallback returns |A_raw|
        - Amplitude & phase consistency between G_WWSSN_SP.FLF and 20HZ_G_WWSSN_SP.FLR (< 1% amp, < 1.5 deg phase)
        """
        cfg = get_config()

        # 1. Velocity-to-displacement conversion
        disp_1s = restitute_ground_displacement_nm(100.0, 1.0, "NONE")
        self.assertAlmostEqual(disp_1s, 100.0 / (2.0 * math.pi), places=5)

        disp_4s = restitute_ground_displacement_nm(250.0, 4.0, "BP 0.5 2.0 3")
        self.assertAlmostEqual(disp_4s, 250.0 * 4.0 / (2.0 * math.pi), places=5)

        # 2. Non-positive period fallback
        self.assertAlmostEqual(restitute_ground_displacement_nm(100.0, 0.0, "NONE"), 100.0)
        self.assertAlmostEqual(restitute_ground_displacement_nm(-80.0, -2.5, "NONE"), 80.0)

        # 3. Simulation filter bypass
        for sim_filter in ("WWSSN_SP", "WWSSN_LP", "WOODAND", "KIRNOS", "DISPLACE", "SRO", "LRSM"):
            disp_sim = restitute_ground_displacement_nm(500.0, 1.5, sim_filter)
            self.assertAlmostEqual(disp_sim, 500.0, places=5, msg=f"Failed for {sim_filter}")

        # 4. Continuous (.FLF) vs Discrete IIR (.FLR) response consistency
        flf_path = os.path.join(cfg.filter_dir, "G_WWSSN_SP.FLF")
        flr_path = os.path.join(cfg.filter_dir, "20HZ_G_WWSSN_SP.FLR")
        self.assertTrue(os.path.isfile(flf_path), f"Missing {flf_path}")
        self.assertTrue(os.path.isfile(flr_path), f"Missing {flr_path}")

        flf_stages = parse_filter_file(flf_path)
        flr_stages = parse_filter_file(flr_path)

        f_pass = np.linspace(0.5, 1.5, 50)  # seismic passband of WWSSN-SP (peak ~ 1 Hz)
        w_pass = 2.0 * math.pi * f_pass

        h_flf = np.ones_like(w_pass, dtype=complex)
        for stg in flf_stages:
            h_flf *= evaluate_stage_freq(stg, w_pass)

        h_flr = np.ones_like(w_pass, dtype=complex)
        for stg in flr_stages:
            h_flr *= evaluate_stage_freq(stg, w_pass)

        # Amplitude ratio within 1% (0.99 .. 1.01)
        amp_ratio = np.abs(h_flr) / np.abs(h_flf)
        np.testing.assert_allclose(amp_ratio, 1.0, rtol=0.01)

        # Phase difference within 1.5 degrees
        ph_flf_deg = np.degrees(np.unwrap(np.angle(h_flf)))
        ph_flr_deg = np.degrees(np.unwrap(np.angle(h_flr)))
        max_ph_diff = np.max(np.abs(ph_flr_deg - ph_flf_deg))
        self.assertLess(max_ph_diff, 1.5)

        # 5. End-to-end restitution check on synthetic harmonic velocity trace v(t) = V0 * sin(omega * t)
        # For v(t) = V0 * sin(omega * t), true displacement is x(t) = -V0/omega * cos(omega * t),
        # so peak displacement amplitude is D0 = V0 / omega = V0 * T / (2*pi).
        v0 = 1250.0  # nm/s
        dt = 0.001   # fine time resolution
        for period_target in (0.2, 0.5, 1.0, 2.0, 5.0):
            omega = 2.0 * math.pi / period_target
            d0_true = v0 / omega

            # Direct mathematical restitution verification
            disp_direct = restitute_ground_displacement_nm(v0, period_target, "NONE")
            self.assertAlmostEqual(disp_direct, d0_true, places=7)

            # End-to-end trace with measure_amplitude_and_period
            t_vec = np.arange(0.0, 10.0 * period_target, dt)
            v_vec = v0 * np.sin(omega * t_vec)
            tr_vel = Trace(data=v_vec, delta=dt, station="SYNTH", comp="Z", filter_name="NONE")

            # Zero-to-peak amplitude measurement
            res_zero = measure_amplitude_and_period(
                tr_vel, t_lo=2.0 * period_target, t_hi=8.0 * period_target, mode="ZERO"
            )
            np.testing.assert_allclose(res_zero["ampl_raw"], v0, rtol=1e-3)
            np.testing.assert_allclose(res_zero["period"], period_target, rtol=5e-3)
            np.testing.assert_allclose(res_zero["ampl_nm"], d0_true, rtol=5e-3)

            # Peak-to-peak / 2 amplitude measurement
            res_peak = measure_amplitude_and_period(
                tr_vel, t_lo=2.0 * period_target, t_hi=8.0 * period_target, mode="PEAK"
            )
            np.testing.assert_allclose(res_peak["ampl_raw"], v0, rtol=1e-3)
            np.testing.assert_allclose(res_peak["period"], period_target, rtol=5e-3)
            np.testing.assert_allclose(res_peak["ampl_nm"], d0_true, rtol=5e-3)

            # Simulation filter bypass (output already represents displacement nm)
            tr_vel_sim = Trace(data=v_vec, delta=dt, station="SYNTH", comp="Z", filter_name="WWSSN_SP")
            res_sim = measure_amplitude_and_period(
                tr_vel_sim, t_lo=2.0 * period_target, t_hi=8.0 * period_target, mode="ZERO"
            )
            np.testing.assert_allclose(res_sim["ampl_nm"], v0, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()

