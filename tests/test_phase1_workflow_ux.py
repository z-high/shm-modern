"""Unit tests for Phase 1 workflow, asynchronous workers, and UX enhancements (M-03, M-04, UX-B01 to UX-B04)."""

import math
import os
import unittest
import numpy as np

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists for Qt widget/canvas/worker testing
app = QApplication.instance()
if app is None:
    app = QApplication(["shm_test"])

from shm_app.core.config_station import get_config
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.dialogs_analysis import FKAnalysisDialog, FKWorker, VespagramDialog, VespagramWorker
from shm_app.gui.main_window import MainWindow
from shm_app.gui.waveform_canvases import MagnifierCanvas, MainTraceCanvas
from shm_app.seismology.array_loc import locate_network_geiger
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


def _create_synthetic_grf_traces() -> list[Trace]:
    """Create 5 synthetic array traces with 1 Hz pulse."""
    cfg = get_config()
    stations = ["GRA1", "GRA2", "GRA3", "GRB1", "GRB2"]
    dt = 0.05
    t_arr = np.arange(300) * dt
    traces = []
    for i, st in enumerate(stations):
        sig = np.zeros_like(t_arr)
        # Shift pulse by station index
        idx_p = 50 + i * 2
        sig[idx_p : idx_p + 20] = np.sin(np.linspace(0, np.pi, 20)) * 2000.0
        tr = Trace(
            data=sig,
            delta=dt,
            station=st,
            comp="Z",
            start_epoch=1000000.0,
        )
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(1000000.0 + idx_p * dt), station=st))
        traces.append(tr)
    return traces


class TestAsyncWorkers(unittest.TestCase):
    """Test background QThread execution, progress reporting, and cancellation for FK and Vespagram."""

    def setUp(self) -> None:
        self.traces = _create_synthetic_grf_traces()
        self.cfg = get_config()

    def test_fk_worker_execution(self) -> None:
        """Verify FKWorker computes FK analysis asynchronously and emits progress and finishedResult."""
        worker = FKWorker(
            self.traces,
            t_lo=1.0,
            t_hi=10.0,
            f_lo=0.5,
            f_hi=3.0,
            max_slowness=10.0,
            grid_points=21,
            config=self.cfg,
        )
        progress_values = []
        worker.progress.connect(progress_values.append)

        result_container = []
        worker.finishedResult.connect(result_container.append)

        worker.start()
        self.assertTrue(worker.wait(5000))
        app.processEvents()

        self.assertGreater(len(progress_values), 0)
        self.assertEqual(len(result_container), 1)
        res = result_container[0]
        self.assertIn("azimuth", res)
        self.assertIn("slowness", res)
        self.assertIn("grid", res)

    def test_fk_worker_cancellation(self) -> None:
        """Verify FKWorker cancels cleanly when cancel() is invoked."""
        worker = FKWorker(
            self.traces,
            t_lo=1.0,
            t_hi=10.0,
            f_lo=0.5,
            f_hi=3.0,
            max_slowness=10.0,
            grid_points=51,
            config=self.cfg,
        )
        worker.cancel()
        worker.start()
        self.assertTrue(worker.wait(3000))
        app.processEvents()

    def test_vespagram_worker_execution(self) -> None:
        """Verify VespagramWorker executes asynchronously and emits progress and finishedResult."""
        worker = VespagramWorker(
            self.traces,
            azimuth_deg=45.0,
            slow_lo=2.0,
            slow_hi=8.0,
            slow_step=1.0,
            power=2.0,
            config=self.cfg,
        )
        progress_values = []
        worker.progress.connect(progress_values.append)

        results = []
        worker.finishedResult.connect(results.append)

        worker.start()
        self.assertTrue(worker.wait(5000))
        app.processEvents()

        self.assertGreater(len(progress_values), 0)
        self.assertEqual(len(results), 1)
        v_trs, times, slos, mat = results[0]
        self.assertEqual(len(v_trs), len(slos))
        self.assertEqual(mat.shape[0], len(slos))

    def test_vespagram_worker_cancellation(self) -> None:
        """Verify VespagramWorker cancels cleanly when cancel() is invoked."""
        worker = VespagramWorker(
            self.traces,
            azimuth_deg=45.0,
            slow_lo=2.0,
            slow_hi=12.0,
            slow_step=0.2,
            power=2.0,
            config=self.cfg,
        )
        worker.cancel()
        worker.start()
        self.assertTrue(worker.wait(3000))

    def test_fk_dialog_async_and_sync(self) -> None:
        """Verify FKAnalysisDialog supports both synchronous and asynchronous execution."""
        dlg = FKAnalysisDialog(self.traces, t_lo=1.0, t_hi=10.0, config=self.cfg)
        # Synchronous execution
        dlg.run_fk(sync=True)
        self.assertIsNotNone(dlg.last_result)
        self.assertFalse(dlg.progress_bar.isVisible())

        # Asynchronous execution trigger
        dlg.run_fk(sync=False)
        self.assertTrue(dlg.worker is not None)
        self.assertTrue(dlg.worker.wait(5000))
        dlg.close()

    def test_vespa_dialog_async_and_sync(self) -> None:
        """Verify VespagramDialog supports both synchronous and asynchronous execution."""
        dlg = VespagramDialog(self.traces, azimuth_deg=45.0, config=self.cfg)
        # Synchronous execution
        dlg.run_vespa(sync=True)
        self.assertGreater(len(dlg.vespa_traces), 0)

        # Asynchronous execution trigger
        dlg.run_vespa(sync=False)
        self.assertTrue(dlg.worker is not None)
        self.assertTrue(dlg.worker.wait(5000))
        dlg.close()


class TestClosedLoopLocation(unittest.TestCase):
    """Test closed-loop location residuals and arrival realignment."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.ttt = get_ttt_manager(self.cfg.inputs_dir)

    def test_locate_network_geiger_residuals(self) -> None:
        """Verify locate_network_geiger returns final residuals on picks and attached to AnalysisParamSet."""
        true_lat, true_lon, true_z = 50.1, 12.3, 10.0
        stations = ["GRA1", "BFO", "FUR", "CLZ", "MOX"]
        t0 = 1000000.0
        picks = []

        for st_code in stations:
            st = self.cfg.get_station(st_code)
            dist, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = self.ttt.travel_time("P", max(0.05, dist), true_z)
            self.assertIsNotNone(tt)
            pk = PhasePick(name="P", onset=format_sh_time(t0 + tt), station=st_code)
            picks.append(pk)

        param, resids = locate_network_geiger(picks, depth_km=true_z, free_depth=False, config=self.cfg, ttt=self.ttt, return_residuals=True)
        self.assertIsNotNone(param)
        self.assertEqual(len(resids), len(picks))
        self.assertIsNotNone(param.residuals)

        # Residuals for synthetic perfect picks should be very small (< 0.1 s)
        for pk in picks:
            self.assertAlmostEqual(pk.resid, 0.0, delta=0.1)

    def test_adopt_network_location_feedback(self) -> None:
        """Verify _on_adopt_network_loc updates tr.theo_picks, tr.distance, and pk.resid on workspace traces."""
        win = MainWindow()
        traces = _create_synthetic_grf_traces()
        win.engine.traces = traces

        # Formulate an adopted location solution
        param = AnalysisParamSet(
            lat=50.0,
            lon=11.5,
            depth=15.0,
            origin="02-NOV-2003_02:25:00.000",
            distance=2.0,
            residuals={"GRA1:P": 0.05, "GRA2:P": -0.03},
        )

        win._on_adopt_network_loc(param)

        # Verify theoretical arrivals and residuals are set
        for tr in win.engine.traces:
            self.assertGreater(tr.distance, 0.0)
            self.assertGreater(len(tr.theo_picks), 0)
            p_pk = tr.get_pick("P")
            if p_pk:
                self.assertIsNotNone(p_pk.resid)

        win.close()


class TestUXFrictionFixes(unittest.TestCase):
    """Test automatic beam reforming, amplitude scaling protection, and temporal pick deduplication."""

    def setUp(self) -> None:
        self.cfg = get_config()

    def test_beam_auto_reform_on_slowness_adopt(self) -> None:
        """Verify that adopting FK parameters automatically reforms an existing BEAM trace (UX-B04)."""
        win = MainWindow()
        win.engine.traces = _create_synthetic_grf_traces()

        # Create initial BEAM trace
        win.action_form_beam()
        beam_trace = next((t for t in win.engine.traces if t.station == "BEAM"), None)
        self.assertIsNotNone(beam_trace)
        initial_data = beam_trace.data.copy()

        # Adopt new FK parameters with a distinct backazimuth and slowness
        win._on_adopt_az_slo(az=180.0, slo=9.5)

        reformed_beam = next((t for t in win.engine.traces if t.station == "BEAM"), None)
        self.assertIsNotNone(reformed_beam)
        # Beam data should be reformed with the new steering delays
        self.assertFalse(np.array_equal(reformed_beam.data, initial_data))
        win.close()

    def test_amplitude_normalization_zero_flattening_protection(self) -> None:
        """Verify tiered normalization prevents zero-flattening when mixing raw counts and physical nm (UX-B03)."""
        canvas = MainTraceCanvas()
        canvas.norm_mode = "C"

        # Trace 1: Raw counts digitizer signal (amplitude ~ 5,000,000 counts)
        raw_data = np.sin(np.linspace(0, 10 * np.pi, 500)) * 5_000_000.0
        tr_raw = Trace(data=raw_data, station="RAW", delta=0.05, comp="Z")

        # Trace 2: Restituted true physical displacement (amplitude ~ 25 nm)
        phys_data = np.sin(np.linspace(0, 10 * np.pi, 500)) * 25.0
        tr_phys = Trace(data=phys_data, station="PHYS", delta=0.05, comp="Z")

        canvas.set_traces([tr_raw, tr_phys])
        # Trigger backing store pixmap rendering
        canvas._ensure_static_pixmap()

        self.assertTrue(getattr(canvas, "_has_mixed_tiers", False))
        self.assertGreater(canvas._cached_tier_high_max, 1_000_000.0)
        self.assertLess(canvas._cached_tier_low_max, 1000.0)

    def test_temporal_pick_deduplication(self) -> None:
        """Verify Trace.add_or_replace_pick deduplicates picks within 0.1 s on same trace."""
        tr = Trace(data=np.zeros(200), station="GRA1")
        t0 = 1000000.0

        # Add initial P pick
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1", quality=4))
        self.assertEqual(len(tr.picks), 1)

        # Add another pick 'Pg' at t0 + 0.05 s (< 0.1 s apart)
        tr.add_or_replace_pick(PhasePick(name="Pg", onset=format_sh_time(t0 + 0.05), station="GRA1", quality=5))
        # Must replace rather than duplicate!
        self.assertEqual(len(tr.picks), 1)
        self.assertEqual(tr.picks[0].name, "Pg")

        # Add distinct pick 'S' at t0 + 15.0 s (> 0.1 s apart)
        tr.add_or_replace_pick(PhasePick(name="S", onset=format_sh_time(t0 + 15.0), station="GRA1", quality=3))
        self.assertEqual(len(tr.picks), 2)

    def test_magnifier_canvas_pick_at(self) -> None:
        """Verify MagnifierCanvas.pick_at hit-tests existing phase picks accurately."""
        mag = MagnifierCanvas()
        tr = Trace(data=np.zeros(500), station="GRA1", delta=0.05, start_epoch=1000000.0)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(1000010.0), station="GRA1"))

        mag.set_magnifier_view(tr, 0, 1000000.0, 5.0, 15.0)
        mag.resize(800, 300)

        # Compute pixel position of the pick
        px = mag.time_to_x(10.0)
        hit_pk = mag.pick_at(QPointF(px, 150.0))
        self.assertIsNotNone(hit_pk)
        self.assertEqual(hit_pk.name, "P")

        # Point far away
        miss_pk = mag.pick_at(QPointF(px + 50.0, 150.0))
        self.assertIsNone(miss_pk)


if __name__ == "__main__":
    unittest.main()
