"""Empirical adversarial stress testing for fit_depth_phase (M-01) and _cmd_mag (M-02).

Executed by challenger_m1_1.
"""

import math
import unittest
import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.seismology.array_loc import fit_depth_phase
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


class TestAdversarialDepthPhase(unittest.TestCase):
    """Adversarial stress testing of fit_depth_phase in seismology/array_loc.py."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.ttt = get_ttt_manager(self.cfg.inputs_dir)
        self.ref_epoch = 1000000.0

    def test_zero_traces_and_empty_picks(self) -> None:
        """Edge case 1: 0 traces or traces without any depth phase picks."""
        # Completely empty trace list
        self.assertIsNone(fit_depth_phase([], distance_deg=60.0, ttt=self.ttt, config=self.cfg))

        # Trace with no picks at all
        tr1 = Trace(data=np.zeros(50), station="GRA1", distance=60.0, start_epoch=self.ref_epoch)
        self.assertIsNone(fit_depth_phase([tr1], distance_deg=60.0, ttt=self.ttt, config=self.cfg))

        # Trace with P pick only (no depth phase)
        tr2 = Trace(data=np.zeros(50), station="GRA1", distance=60.0, start_epoch=self.ref_epoch)
        tr2.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(self.ref_epoch + 10.0), station="GRA1"))
        self.assertIsNone(fit_depth_phase([tr2], distance_deg=60.0, ttt=self.ttt, config=self.cfg))

        # Trace with depth phase (pP) only, but missing reference P pick
        tr3 = Trace(data=np.zeros(50), station="GRA1", distance=60.0, start_epoch=self.ref_epoch)
        tr3.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(self.ref_epoch + 25.0), station="GRA1"))
        self.assertIsNone(fit_depth_phase([tr3], distance_deg=60.0, ttt=self.ttt, config=self.cfg))

        # Trace with P pick having empty onset string
        tr4 = Trace(data=np.zeros(50), station="GRA1", distance=60.0, start_epoch=self.ref_epoch)
        tr4.add_or_replace_pick(PhasePick(name="P", onset="", station="GRA1"))
        tr4.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(self.ref_epoch + 25.0), station="GRA1"))
        self.assertIsNone(fit_depth_phase([tr4], distance_deg=60.0, ttt=self.ttt, config=self.cfg))

    def test_single_trace_pP_only(self) -> None:
        """Edge case 2: Single trace with pP only."""
        dist = 67.0
        target_z = 75.0
        t_p = self.ttt.travel_time("P", dist, target_z)
        t_pp = self.ttt.travel_time("pP", dist, target_z)
        self.assertIsNotNone(t_p)
        self.assertIsNotNone(t_pp)

        tr = Trace(data=np.zeros(50), station="GRB1", distance=dist, start_epoch=self.ref_epoch)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(self.ref_epoch + t_p), station="GRB1"))
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(self.ref_epoch + t_pp), station="GRB1", quality=4))

        res = fit_depth_phase([tr], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["station"], "GRB1")
        self.assertEqual(res["phase"], "pP")
        self.assertAlmostEqual(res["depth"], target_z, delta=2.0)
        # For n=1, variance = sigma_z^2 (unrounded), sigma_z = d_depth
        self.assertEqual(res["std_depth"], res["d_depth"])
        self.assertGreater(res["std_depth"], 0.0)
        self.assertAlmostEqual(res["variance"], round(res["details"][0]["d_depth"] ** 2, 4), places=4)
        self.assertAlmostEqual(math.sqrt(res["variance"]), res["std_depth"], delta=0.05)

    def test_conflicting_depth_phases_and_dispersion(self) -> None:
        """Edge case 3: Multiple traces with strongly conflicting depth phases."""
        dist = 65.0
        t0 = self.ref_epoch
        # Station 1: shallow event (30 km) -> small dt_dp
        dt1 = self.ttt.travel_time("pP", dist, 30.0) - self.ttt.travel_time("P", dist, 30.0)
        # Station 2: deep event (450 km) -> large dt_dp
        dt2 = self.ttt.travel_time("pP", dist, 450.0) - self.ttt.travel_time("P", dist, 450.0)

        tr1 = Trace(data=np.zeros(50), station="GRA1", distance=dist, start_epoch=t0)
        tr1.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1"))
        tr1.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + dt1), station="GRA1", quality=4))

        tr2 = Trace(data=np.zeros(50), station="BFO", distance=dist, start_epoch=t0)
        tr2.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="BFO"))
        tr2.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + dt2), station="BFO", quality=4))

        res = fit_depth_phase([tr1, tr2], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 2)

        # Depth must be between 30 and 450 km
        self.assertGreater(res["depth"], 30.0)
        self.assertLess(res["depth"], 450.0)

        # Standard deviation must reflect the large discrepancy (> 100 km)
        self.assertGreater(res["std_depth"], 100.0)
        self.assertGreater(res["variance"], 10000.0)

        # Verify weighted mean formula
        w1 = res["details"][0]["weight"]
        w2 = res["details"][1]["weight"]
        z1 = res["details"][0]["depth"]
        z2 = res["details"][1]["depth"]
        expected_mean = (w1 * z1 + w2 * z2) / (w1 + w2)
        self.assertAlmostEqual(res["depth"], round(expected_mean, 2), places=2)

    def test_identical_arrival_times_dt_zero(self) -> None:
        """Edge case 4: Identical arrival times (dt_dp == 0.0)."""
        dist = 60.0
        t0 = self.ref_epoch

        # When dt_dp == 0, pP arrival coincides with P arrival
        tr = Trace(data=np.zeros(50), station="GRA1", distance=dist, start_epoch=t0)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1"))
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0), station="GRA1"))

        # TravelTimeManager cannot invert dt=0 to a valid focal depth, so it returns None
        res = fit_depth_phase([tr], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNone(res)

        # If mixed with a valid trace, the dt=0 trace is safely discarded
        dt_valid = self.ttt.travel_time("pP", dist, 60.0) - self.ttt.travel_time("P", dist, 60.0)
        tr_valid = Trace(data=np.zeros(50), station="BFO", distance=dist, start_epoch=t0)
        tr_valid.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="BFO"))
        tr_valid.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + dt_valid), station="BFO", quality=4))

        res_mixed = fit_depth_phase([tr, tr_valid], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res_mixed)
        self.assertEqual(res_mixed["count"], 1)
        self.assertEqual(res_mixed["station"], "BFO")
        self.assertAlmostEqual(res_mixed["depth"], 60.0, delta=2.0)

    def test_extreme_outlier_depths_and_clamping(self) -> None:
        """Edge case 5: Extreme differential times (e.g. 5000 s) capped safely."""
        dist = 60.0
        t0 = self.ref_epoch

        tr = Trace(data=np.zeros(50), station="GRA1", distance=dist, start_epoch=t0)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1"))
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + 5000.0), station="GRA1", quality=4))

        res = fit_depth_phase([tr], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        # Depth is clamped to 700.0 km (mantle boundary)
        self.assertEqual(res["depth"], 700.0)
        self.assertGreater(res["variance"], 0.0)
        self.assertGreater(res["std_depth"], 0.0)

    def test_missing_station_distances_fallbacks(self) -> None:
        """Edge case 6: Missing station distance with fallback to distance_deg."""
        t0 = self.ref_epoch
        dt = self.ttt.travel_time("pP", 65.0, 50.0) - self.ttt.travel_time("P", 65.0, 50.0)

        # tr.distance <= 0 (e.g. 0.0 or -1.0) -> must fallback to distance_deg (65.0)
        tr_nodist = Trace(data=np.zeros(50), station="GRA1", distance=-1.0, start_epoch=t0)
        tr_nodist.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1"))
        tr_nodist.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + dt), station="GRA1", quality=4))

        res = fit_depth_phase([tr_nodist], distance_deg=65.0, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["distance"], 65.0)
        self.assertAlmostEqual(res["depth"], 50.0, delta=2.0)

        # Both tr.distance <= 0 and distance_deg <= 0 -> returns None
        res_zero = fit_depth_phase([tr_nodist], distance_deg=0.0, ttt=self.ttt, config=self.cfg)
        self.assertIsNone(res_zero)

    def test_formal_variance_positive_definite_bounds(self) -> None:
        """Verify weighted mean formula, positive variance, and formal sigma_z bounds."""
        dist = 70.0
        t0 = self.ref_epoch
        target_z = 100.0
        dt = self.ttt.travel_time("pP", dist, target_z) - self.ttt.travel_time("P", dist, target_z)

        # 3 traces with IDENTICAL arrival times (sample_var = 0)
        stations = ["GRA1", "GRB1", "GRC1"]
        traces = []
        for st in stations:
            tr = Trace(data=np.zeros(50), station=st, distance=dist, start_epoch=t0)
            tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station=st))
            tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + dt), station=st, quality=4))
            traces.append(tr)

        res = fit_depth_phase(traces, distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 3)
        self.assertAlmostEqual(res["depth"], target_z, delta=2.0)

        # Even though sample variance among estimates is 0, formal sigma_z > 0 because of max(sqrt(formal_var), sem)
        self.assertGreater(res["std_depth"], 0.0)
        self.assertGreater(res["variance"], 0.0)

        # Calculate theoretical formal variance: 1 / sum(weights)
        total_w = sum(f["weight"] for f in res["details"])
        formal_sigma = math.sqrt(1.0 / total_w)
        self.assertAlmostEqual(res["std_depth"], round(formal_sigma, 2), places=2)

    def test_dual_phase_quality_weighting(self) -> None:
        """Verify that quality factor correctly scales weights between pP and sP on the same trace."""
        dist = 65.0
        t0 = self.ref_epoch
        dt_pp = self.ttt.travel_time("pP", dist, 50.0) - self.ttt.travel_time("P", dist, 50.0)
        dt_sp = self.ttt.travel_time("sP", dist, 120.0) - self.ttt.travel_time("P", dist, 120.0)

        tr = Trace(data=np.zeros(50), station="GRA1", distance=dist, start_epoch=t0)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1"))
        # High quality pP (quality=5 -> depth 50)
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(t0 + dt_pp), station="GRA1", quality=5))
        # Low quality sP (quality=0 -> depth 120)
        tr.add_or_replace_pick(PhasePick(name="sP", onset=format_sh_time(t0 + dt_sp), station="GRA1", quality=0))

        res = fit_depth_phase([tr], dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 2)
        # Quality 5 weight vs Quality 0 weight (>20x ratio)
        w_pp = res["details"][0]["weight"]
        w_sp = res["details"][1]["weight"]
        self.assertGreater(w_pp, w_sp * 20)
        # Mean depth must be pulled heavily toward the high-quality pick (near 50 km)
        self.assertLess(res["depth"], 55.0)


class TestAdversarialMagnitude(unittest.TestCase):
    """Adversarial stress testing of _cmd_mag and measure_amplitude_magnitudes."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.engine = SHCommandEngine(config=self.cfg)
        self.ref_epoch = 1000000.0
        self.dt = 0.05

    def _make_synth_trace(self, stat: str, amp: float, p_offset: float = 2.0) -> Trace:
        t_arr = np.arange(300) * self.dt
        sig = np.sin(2.0 * np.pi * 1.0 * t_arr) * amp
        tr = Trace(
            data=sig,
            delta=self.dt,
            station=stat,
            comp="Z",
            start_epoch=self.ref_epoch,
            calib=1.0,
        )
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(self.ref_epoch + p_offset), station=stat))
        return tr

    def test_station_coordinate_lookup_failure(self) -> None:
        """Adversarial test 1: Station coordinate lookup fails for unconfigured station."""
        p = self.engine.current_params
        p.lat = 38.0
        p.lon = 22.0
        p.depth = 15.0
        p.distance = 55.0

        tr_nonexist = self._make_synth_trace("NONEXISTENT_STATION", amp=4000.0)
        self.engine.traces = [tr_nonexist]

        # Should fall back to global distance without crashing
        msg = self.engine.execute_line("MAG P")
        self.assertIn("Dist=55.0 deg", msg)
        self.assertEqual(p.mags_count, 1)
        self.assertGreater(p.mag_mb, 0.0)
        self.assertEqual(p.d_mb, 0.0)

    def test_missing_event_coordinates(self) -> None:
        """Adversarial test 2: Missing event lat/lon (both zero)."""
        p = self.engine.current_params
        p.lat = 0.0
        p.lon = 0.0
        p.distance = 42.0

        tr = self._make_synth_trace("GRA1", amp=3500.0)
        tr.distance = 0.0
        self.engine.traces = [tr]

        msg = self.engine.execute_line("MAG P")
        self.assertIn("Dist=42.0 deg", msg)
        self.assertEqual(p.mags_count, 1)

    def test_single_vs_multistation_dispersion(self) -> None:
        """Adversarial test 3: Single vs multi-station magnitude dispersion (ddof=1)."""
        p = self.engine.current_params
        p.lat = 40.0
        p.lon = 20.0
        p.depth = 10.0

        # 1 Station -> stddev MUST be 0.0
        self.engine.traces = [self._make_synth_trace("GRA1", amp=2000.0)]
        self.engine.execute_line("MAG P")
        self.assertEqual(p.mags_count, 1)
        self.assertEqual(p.d_mb, 0.0)
        self.assertEqual(p.d_ml, 0.0)
        self.assertEqual(p.d_ms, 0.0)
        self.assertEqual(p.d_mbb, 0.0)

        # 2 Stations with different amplitudes -> stddev MUST be > 0.0 (ddof=1)
        tr1 = self._make_synth_trace("GRA1", amp=2000.0)
        tr2 = self._make_synth_trace("BFO", amp=8000.0)
        self.engine.traces = [tr1, tr2]
        self.engine.execute_line("MAG P")
        self.assertEqual(p.mags_count, 2)
        self.assertGreater(p.d_mb, 0.0)
        self.assertGreater(p.d_ml, 0.0)

        # 3 Stations -> ddof=1 calculation matches numpy std(ddof=1)
        tr3 = self._make_synth_trace("FUR", amp=15000.0)
        self.engine.traces = [tr1, tr2, tr3]
        self.engine.execute_line("MAG P")
        self.assertEqual(p.mags_count, 3)
        self.assertGreater(p.d_mb, 0.0)

    def test_extreme_distances(self) -> None:
        """Adversarial test 4: Extreme distances (0 deg co-located and 180 deg antipodal)."""
        p = self.engine.current_params
        st = self.cfg.get_station("GRA1")

        # 0 deg distance (co-located epicenter)
        p.lat = st.lat
        p.lon = st.lon
        p.depth = 10.0
        tr_co = self._make_synth_trace("GRA1", amp=3000.0)
        self.engine.traces = [tr_co]

        msg0 = self.engine.execute_line("MAG P")
        self.assertIn("Dist=0.0 deg", msg0)
        # At dist 0.0, magnitude is safe 0.0 without division by zero
        self.assertEqual(p.mag_mb, 0.0)
        self.assertEqual(p.mag_ml, 0.0)

        # 180 deg antipodal distance
        p.lat = -st.lat
        p.lon = st.lon - 180.0
        tr_anti = self._make_synth_trace("GRA1", amp=3000.0)
        self.engine.traces = [tr_anti]

        msg180 = self.engine.execute_line("MAG P")
        self.assertIn("Dist=180.0 deg", msg180)
        self.assertGreater(p.mag_mb, 0.0)
        self.assertGreater(p.mag_ms, 0.0)

    def test_zero_amplitude_and_empty_workspace(self) -> None:
        """Adversarial test 5: Zero-amplitude traces and empty workspace."""
        p = self.engine.current_params
        p.lat = 40.0
        p.lon = 20.0
        p.depth = 10.0

        # Zero amplitude signal
        tr_zero = Trace(data=np.zeros(200), delta=self.dt, station="GRA1", comp="Z", start_epoch=self.ref_epoch)
        tr_zero.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(self.ref_epoch + 1.0), station="GRA1"))
        self.engine.traces = [tr_zero]

        msg_zero = self.engine.execute_line("MAG P")
        self.assertIn("0 traces", msg_zero)
        self.assertEqual(p.mags_count, 0)
        self.assertEqual(p.mag_mb, 0.0)

        # Empty workspace
        self.engine.traces = []
        msg_empty = self.engine.execute_line("MAG P")
        self.assertIn("0 traces", msg_empty)
        self.assertEqual(p.mags_count, 0)

    def test_fallback_when_trace_lacks_picks(self) -> None:
        """Adversarial test 6: Trace lacking P pick falls back to time_window or full duration."""
        p = self.engine.current_params
        p.lat = 40.0
        p.lon = 20.0
        p.depth = 10.0

        # Trace without any picks
        t_arr = np.arange(400) * self.dt
        sig = np.sin(2.0 * np.pi * 1.0 * t_arr) * 4000.0
        tr_nopick = Trace(data=sig, delta=self.dt, station="GRA1", comp="Z", start_epoch=self.ref_epoch)
        self.engine.traces = [tr_nopick]

        msg = self.engine.execute_line("MAG P")
        self.assertIn("1 traces", msg)
        self.assertEqual(p.mags_count, 1)
        self.assertGreater(p.mag_mb, 0.0)

    def test_nan_in_trace_data_handling(self) -> None:
        """Adversarial test 7: Trace data with NaN does not cause uncaught crash."""
        p = self.engine.current_params
        p.lat = 40.0
        p.lon = 20.0
        p.depth = 10.0

        sig_nan = np.ones(200) * 1000.0
        sig_nan[50] = np.nan
        tr_nan = Trace(data=sig_nan, delta=self.dt, station="GRA1", comp="Z", start_epoch=self.ref_epoch)
        tr_nan.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(self.ref_epoch + 1.0), station="GRA1"))
        self.engine.traces = [tr_nan]

        msg = self.engine.execute_line("MAG P")
        self.assertIn("0 traces", msg)
        self.assertEqual(p.mags_count, 0)

    def test_measure_amplitude_magnitudes_invalid_indices(self) -> None:
        """Adversarial test 8: Out of bounds indices return None."""
        self.assertIsNone(self.engine.measure_amplitude_magnitudes(-1, 0.0, 5.0))
        self.assertIsNone(self.engine.measure_amplitude_magnitudes(999, 0.0, 5.0))


if __name__ == "__main__":
    unittest.main()
