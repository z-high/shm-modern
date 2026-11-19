"""Tests for M-15 (Beamforming channel residuals, coherence CC_m, 2D ARF, FitDiagnosticResult)
and M-18 (Automated phase identification via TravelTimeManager).
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.fit_models import FitType
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.processing.trace_ops import create_synthetic_trace
from shm_app.seismology.array_loc import (
    SlownessCalibrationTable,
    beam_delays_sec,
    build_fk_fit_diagnostic,
    compute_array_response_function,
    compute_beam_diagnostics,
    compute_fk_analysis,
    identify_phase,
    identify_trace_picks,
)
from shm_app.seismology.traveltime import get_ttt_manager


class TestBeamDiagnosticsAndARF(unittest.TestCase):
    def _build_synthetic_grf_traces(
        self,
        baz: float = 115.0,
        slowness: float = 7.2,
    ) -> List[Trace]:
        cfg = get_config()
        stations = ["GRA1", "GRA2", "GRA3", "GRA4", "GRB1", "GRB2", "GRB3", "GRB4", "GRB5"]
        coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
        x_km = [c[0] for c in coords]
        y_km = [c[1] for c in coords]
        delays = beam_delays_sec(x_km, y_km, baz, slowness)
        base_ep = parse_sh_time("02-NOV-2003_02:25:00.000")

        traces: List[Trace] = []
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
        return traces

    def test_beam_diagnostics_clean_and_corrupted_channels(self) -> None:
        cfg = get_config()
        baz, slo = 115.0, 7.2
        traces = self._build_synthetic_grf_traces(baz=baz, slowness=slo)

        # Clean array -> all channels highly coherent (CC_m > 0.90), zero poor elements
        diag_clean = compute_beam_diagnostics(traces, baz, slo, t_lo=12.0, t_hi=22.0, config=cfg)
        self.assertEqual(len(diag_clean["channel_cc"]), len(traces))
        self.assertEqual(diag_clean["poor_channels"], [])
        self.assertTrue(all(cc > 0.90 for cc in diag_clean["channel_cc"]))
        self.assertEqual(diag_clean["channel_residuals"].shape[0], len(traces))
        self.assertEqual(diag_clean["beam_waveform"].shape, diag_clean["mean_residual"].shape)
        self.assertGreater(float(np.max(np.abs(diag_clean["beam_waveform"]))), 150.0)
        self.assertLess(float(np.max(diag_clean["mean_residual"])), 25.0)

        # Corrupt GRB3 with noise of comparable RMS amplitude and invert polarity on GRB5
        rng = np.random.RandomState(1234)
        traces_bad = [tr.copy() for tr in traces]
        # Index 6 is GRB3, index 8 is GRB5
        traces_bad[6].data = rng.randn(traces_bad[6].length) * 60.0
        traces_bad[8].data = -traces_bad[8].data

        diag_bad = compute_beam_diagnostics(traces_bad, baz, slo, t_lo=12.0, t_hi=22.0, config=cfg)
        self.assertIn("GRB3", diag_bad["poor_channels"])
        self.assertIn("GRB5", diag_bad["poor_channels"])
        self.assertNotIn("GRA1", diag_bad["poor_channels"])
        self.assertLess(diag_bad["channel_cc"][6], 0.70)
        self.assertLess(diag_bad["channel_cc"][8], -0.50)

    def test_array_response_function_peak_and_symmetry(self) -> None:
        cfg = get_config()
        stations = ["GRA1", "GRA2", "GRA3", "GRA4", "GRB1", "GRB2", "GRB3", "GRB4", "GRB5", "GRC1", "GRC2", "GRC3", "GRC4"]
        coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
        x_km = [c[0] for c in coords]
        y_km = [c[1] for c in coords]

        arf = compute_array_response_function(x_km, y_km, slowness_max=20.0, n_grid=101, freq_hz=1.0)
        grid = arf["arf_grid"]
        self.assertEqual(grid.shape, (101, 101))
        # Center (50, 50) corresponds to (sx=0, sy=0) where |A(0)|^2 == 1.0
        self.assertAlmostEqual(float(grid[50, 50]), 1.0, places=12)
        self.assertTrue(np.all((grid >= 0.0) & (grid <= 1.0 + 1e-12)))
        # Point-symmetric ARF: |A(-k)|^2 == |A(k)|^2
        np.testing.assert_allclose(grid, grid[::-1, ::-1], atol=1e-12)

        # Wideband ARF over multiple frequencies also peaks at 1.0 at origin
        arf_wb = compute_array_response_function(x_km, y_km, slowness_max=20.0, n_grid=101, freq_hz=[0.8, 1.2, 1.6])
        self.assertAlmostEqual(float(arf_wb["arf_grid"][50, 50]), 1.0, places=12)

    def test_build_fk_fit_diagnostic_contract(self) -> None:
        cfg = get_config()
        traces = self._build_synthetic_grf_traces(baz=115.0, slowness=7.2)
        # Invert polarity on GRA4 so we have one outlier residual entry
        traces[3].data = -traces[3].data

        calib_tbl = SlownessCalibrationTable.from_text(
            "# slowness_in azimuth_in d_slowness d_azimuth\n"
            "7.2 115.0 -0.4 +3.0\n"
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
        beam_diag = compute_beam_diagnostics(traces, fk_res["azimuth"], fk_res["slowness"], t_lo=12.0, t_hi=22.0, config=cfg)

        coords = cfg.relative_xy_km([t.station for t in traces], ref_station="CENTRE")
        arf_res = compute_array_response_function(
            [c[0] for c in coords], [c[1] for c in coords], slowness_max=12.0, n_grid=31, freq_hz=1.0
        )

        diag = build_fk_fit_diagnostic(fk_res, beam_diag=beam_diag, arf_result=arf_res)
        self.assertEqual(diag.fit_type, FitType.FK_SEMBLANCE)
        self.assertTrue(diag.converged)

        for key in ("s_axis", "fk_grid", "arf_grid", "beam_time", "beam_waveform", "mean_residual", "channel_residuals"):
            self.assertIn(key, diag.overlay_curves)
            self.assertGreater(diag.overlay_curves[key].size, 0)

        self.assertEqual(len(diag.residuals), len(traces))
        gra4_entry = next(e for e in diag.residuals if e.station == "GRA4")
        self.assertEqual(gra4_entry.observed, 1.0)
        self.assertAlmostEqual(gra4_entry.residual, 1.0 - gra4_entry.calculated, places=6)
        self.assertTrue(gra4_entry.is_outlier)
        self.assertGreaterEqual(diag.outlier_count, 1)

        for ctx_key in (
            "calib_applied",
            "raw_sx",
            "raw_sy",
            "cal_sx",
            "cal_sy",
            "raw_slowness",
            "raw_baz",
            "cal_slowness",
            "cal_baz",
            "channel_cc",
            "channel_names",
        ):
            self.assertIn(ctx_key, diag.context_data)
        self.assertTrue(diag.context_data["calib_applied"])
        self.assertEqual(len(diag.context_data["channel_cc"]), len(traces))


class TestAutomatedPhaseIdentification(unittest.TestCase):
    def test_identify_phase_distinguishes_p_pp_s_and_pkp(self) -> None:
        ttt = get_ttt_manager()

        # Teleseismic distance 60 deg, depth 10 km (shallow: pP/sP arrive ~2.8s after P)
        for true_phase in ("P", "PP", "S"):
            t_true = ttt.travel_time(true_phase, 60.0, 10.0)
            s_true = ttt.slowness(true_phase, 60.0, 10.0)
            self.assertIsNotNone(t_true)
            self.assertIsNotNone(s_true)

            # Add small realistic measurement noise (+0.4 s, -0.15 s/deg)
            cands = identify_phase(
                t_obs_rel_sec=t_true + 0.4,
                slowness_meas_s_deg=s_true - 0.15,
                dist_deg=60.0,
                depth_km=10.0,
                baz_meas_deg=45.0,
                baz_theo_deg=43.0,
                ttt=ttt,
            )
            self.assertGreater(len(cands), 3)
            self.assertEqual(cands[0]["phase"], true_phase)
            self.assertGreater(cands[0]["probability"], 0.5)
            # Sorted by ascending chi2 and probabilities sum to 1.0
            chi2_vals = [c["chi2"] for c in cands]
            self.assertEqual(chi2_vals, sorted(chi2_vals))
            self.assertAlmostEqual(sum(c["probability"] for c in cands), 1.0, places=6)

        # Deep focus (120 km) separates P from pP cleanly -> P probability > 0.99
        t_p120 = ttt.travel_time("P", 60.0, 120.0)
        s_p120 = ttt.slowness("P", 60.0, 120.0)
        cands_deep = identify_phase(t_p120, s_p120, dist_deg=60.0, depth_km=120.0, ttt=ttt)
        self.assertEqual(cands_deep[0]["phase"], "P")
        self.assertGreater(cands_deep[0]["probability"], 0.99)

        # Core phase distance 150 deg, depth 10 km -> PKPdf vs PKPab vs PKPbc vs PP
        t_pkpdf = ttt.travel_time("PKPdf", 150.0, 10.0)
        s_pkpdf = ttt.slowness("PKPdf", 150.0, 10.0)
        cands_core = identify_phase(
            t_obs_rel_sec=t_pkpdf + 0.3,
            slowness_meas_s_deg=s_pkpdf + 0.1,
            dist_deg=150.0,
            depth_km=10.0,
            ttt=ttt,
        )
        self.assertGreater(len(cands_core), 0)
        self.assertEqual(cands_core[0]["phase"], "PKPdf")

        # Slowness discriminates PKPab (~4.1 s/deg) from PKPdf (~1.6 s/deg)
        t_pkpab = ttt.travel_time("PKPab", 150.0, 10.0)
        s_pkpab = ttt.slowness("PKPab", 150.0, 10.0)
        cands_ab = identify_phase(
            t_obs_rel_sec=t_pkpab,
            slowness_meas_s_deg=s_pkpab,
            dist_deg=150.0,
            depth_km=10.0,
            ttt=ttt,
        )
        self.assertEqual(cands_ab[0]["phase"], "PKPab")

    def test_identify_trace_picks_helper_updates_picks(self) -> None:
        ttt = get_ttt_manager()
        origin_ep = parse_sh_time("02-NOV-2003_02:15:00.000")
        dist_deg = 60.0
        depth_km = 10.0

        t_p = ttt.travel_time("P", dist_deg, depth_km)
        s_p = ttt.slowness("P", dist_deg, depth_km)
        t_s = ttt.travel_time("S", dist_deg, depth_km)
        s_s = ttt.slowness("S", dist_deg, depth_km)

        tr = Trace(data=np.zeros(200), delta=0.05, station="GRA1", start_epoch=origin_ep)
        tr.add_or_replace_pick(
            PhasePick(name="X1", onset=format_sh_time(origin_ep + t_p + 0.2), slowness=s_p, station="GRA1")
        )
        tr.add_or_replace_pick(
            PhasePick(name="X2", onset=format_sh_time(origin_ep + t_s - 0.3), slowness=s_s, station="GRA1")
        )

        id_res = identify_trace_picks(
            [tr], origin_epoch=origin_ep, dist_deg=dist_deg, depth_km=depth_km, update_pick_names=True, ttt=ttt
        )
        self.assertEqual(len(id_res), 2)
        self.assertEqual(id_res[0]["identified_phase"], "P")
        self.assertEqual(id_res[1]["identified_phase"], "S")
        self.assertEqual(tr.picks[0].name, "P")
        self.assertEqual(tr.picks[1].name, "S")
        self.assertAlmostEqual(tr.picks[0].resid, 0.2, delta=0.05)


if __name__ == "__main__":
    unittest.main()
