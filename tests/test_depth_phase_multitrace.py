"""Unit tests for multi-station depth phase analysis in seismology/array_loc.py (M-01)."""

import math
import unittest
import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.seismology.array_loc import fit_depth_phase
from shm_app.seismology.traveltime import get_ttt_manager


class TestDepthPhaseMultiTrace(unittest.TestCase):
    """Test multi-station depth phase focal depth inversion and uncertainty estimation."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.ttt = get_ttt_manager(self.cfg.inputs_dir)

    def test_fit_depth_phase_evaluates_all_traces(self) -> None:
        """Verify that all traces with depth phases are evaluated and aggregated."""
        true_depth = 120.0
        stations = ["GRA1", "GRB1", "GRC1", "BFO", "FUR"]
        distances = [65.0, 66.0, 67.0, 68.0, 69.0]
        traces = []

        ref_epoch = 1000000.0
        for st, dist in zip(stations, distances):
            tr = Trace(data=np.zeros(200), station=st, distance=dist, start_epoch=ref_epoch)
            t_p = self.ttt.travel_time("P", dist, true_depth)
            t_pp = self.ttt.travel_time("pP", dist, true_depth)
            self.assertIsNotNone(t_p)
            self.assertIsNotNone(t_pp)

            tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + t_p), station=st))
            tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(ref_epoch + t_pp), station=st, quality=4))
            traces.append(tr)

        res = fit_depth_phase(traces, distance_deg=65.0, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 5)
        self.assertAlmostEqual(res["depth"], true_depth, delta=3.0)
        self.assertIn("std_depth", res)
        self.assertIn("d_depth", res)
        self.assertEqual(res["std_depth"], res["d_depth"])
        self.assertGreater(res["std_depth"], 0.0)

        # Check details structure
        self.assertEqual(len(res["details"]), 5)
        for detail in res["details"]:
            self.assertIn("station", detail)
            self.assertIn("phase", detail)
            self.assertEqual(detail["phase"], "pP")
            self.assertIn("residual", detail)
            self.assertIn("weight", detail)
            self.assertIn("distance", detail)

    def test_fit_depth_phase_both_pP_and_sP_on_same_station(self) -> None:
        """Verify that both pP and sP picks on the same station contribute independently."""
        true_depth = 90.0
        dist = 70.0
        ref_epoch = 1000000.0

        tr = Trace(data=np.zeros(200), station="GRA1", distance=dist, start_epoch=ref_epoch)
        t_p = self.ttt.travel_time("P", dist, true_depth)
        t_pp = self.ttt.travel_time("pP", dist, true_depth)
        t_sp = self.ttt.travel_time("sP", dist, true_depth)

        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + t_p), station="GRA1"))
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(ref_epoch + t_pp), station="GRA1", quality=4))
        self.assertGreater(abs(t_sp - t_pp), 1.0)
        tr.add_or_replace_pick(PhasePick(name="sP", onset=format_sh_time(ref_epoch + t_sp), station="GRA1", quality=4))

        res = fit_depth_phase([tr], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 2)
        self.assertIn("pP", res["phases"])
        self.assertIn("sP", res["phases"])
        self.assertAlmostEqual(res["depth"], true_depth, delta=4.0)

    def test_fit_depth_phase_single_trace_fallback(self) -> None:
        """Verify single-trace depth phase calculation behaves predictably."""
        true_depth = 100.0
        dist = 70.0
        ref_epoch = 1000000.0

        tr = Trace(data=np.zeros(200), station="BFO", distance=dist, start_epoch=ref_epoch)
        t_p = self.ttt.travel_time("P", dist, true_depth)
        t_pp = self.ttt.travel_time("pP", dist, true_depth)

        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + t_p), station="BFO"))
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(ref_epoch + t_pp), station="BFO", quality=3))

        res = fit_depth_phase([tr], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 1)
        self.assertAlmostEqual(res["depth"], true_depth, delta=3.0)
        self.assertEqual(res["station"], "BFO")
        self.assertEqual(res["phase"], "pP")

    def test_fit_depth_phase_no_picks_returns_none(self) -> None:
        """Verify None is returned when traces lack depth phases."""
        tr1 = Trace(data=np.zeros(100), station="GRA1")
        tr1.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(1000000.0)))
        tr2 = Trace(data=np.zeros(100), station="GRB1")
        tr2.add_or_replace_pick(PhasePick(name="S", onset=format_sh_time(1000020.0)))

        res = fit_depth_phase([tr1, tr2], distance_deg=50.0, ttt=self.ttt, config=self.cfg)
        self.assertIsNone(res)

    def test_fit_depth_phase_backward_compatibility(self) -> None:
        """Verify legacy callers expecting ['depth'], ['d_depth'], ['station'], ['phase'], ['dt_dp'], ['distance'] work."""
        true_depth = 80.0
        dist = 60.0
        ref_epoch = 1000000.0

        tr = Trace(data=np.zeros(100), station="FUR", distance=dist, start_epoch=ref_epoch)
        t_p = self.ttt.travel_time("P", dist, true_depth)
        t_pp = self.ttt.travel_time("pP", dist, true_depth)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + t_p), station="FUR"))
        tr.add_or_replace_pick(PhasePick(name="pP", onset=format_sh_time(ref_epoch + t_pp), station="FUR"))

        res = fit_depth_phase([tr], distance_deg=dist, ttt=self.ttt, config=self.cfg)
        self.assertIsNotNone(res)
        for key in ["depth", "std_depth", "d_depth", "count", "details", "station", "phase", "dt_dp", "distance"]:
            self.assertIn(key, res)


if __name__ == "__main__":
    unittest.main()
