"""Adversarial stress testing and empirical validation harness for Milestone 1.

Covers:
1. Async Workers (FKWorker, VespagramWorker, FKAnalysisDialog, VespagramDialog):
   - Fast cancellation right after start
   - Cancellation during long iteration
   - Dialog destroyed / closed while worker is running
   - Zero trace input (empty trace list)
   - All zero trace data (numerical stability)
2. Closed-Loop Location feedback & Theoretical Arrivals:
   - Solution adoption with 0 picks
   - Picks without stations / unlinked stations
   - Non-converged hypocenter adoption
   - Verification of PhasePick.resid and theoretical marker rendering
3. UX Friction Points:
   - Beam auto-reforming with empty trace list, single trace, and missing azimuth/slowness
   - Amplitude scaling: combine raw counts (1e7) and physical displacement (0.5 nm) in NORM C and AW
   - Rapid double-click pick creation and deduplication (<0.1s threshold)
"""

import math
import os
import sys
import unittest
import numpy as np

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(["adversarial_m1_test"])

from shm_app.core.config_station import get_config
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    FKWorker,
    VespagramDialog,
    VespagramWorker,
)
from shm_app.gui.dialogs_location import NetworkLocationDialog
from shm_app.gui.main_window import MainWindow
from shm_app.gui.waveform_canvases import MainTraceCanvas
from shm_app.seismology.array_loc import (
    compute_beam,
    compute_fk_analysis,
    compute_vespagram,
    locate_network_geiger,
)
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


def _create_synthetic_grf_traces(n: int = 5, amp: float = 2000.0) -> list[Trace]:
    """Create synthetic array traces for testing."""
    stations = ["GRA1", "GRA2", "GRA3", "GRB1", "GRB2", "GRC1", "GRC2"][:n]
    dt = 0.05
    t_arr = np.arange(400) * dt
    traces = []
    for i, st in enumerate(stations):
        sig = np.zeros_like(t_arr)
        idx_p = 60 + i * 2
        sig[idx_p : idx_p + 30] = np.sin(np.linspace(0, 2 * np.pi, 30)) * amp
        tr = Trace(
            data=sig,
            delta=dt,
            station=st,
            comp="Z",
            start_epoch=1000000.0,
        )
        tr.add_or_replace_pick(
            PhasePick(
                name="P",
                onset=format_sh_time(1000000.0 + idx_p * dt),
                station=st,
                quality=4,
            )
        )
        traces.append(tr)
    return traces


class TestAdversarialAsyncWorkers(unittest.TestCase):
    """Adversarial stress testing of FKWorker and VespagramWorker."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.traces = _create_synthetic_grf_traces(5)

    def test_fk_worker_fast_cancellation(self) -> None:
        """Stress: Fast cancellation right after start() without blocking or deadlocking."""
        worker = FKWorker(
            self.traces,
            t_lo=1.0,
            t_hi=15.0,
            f_lo=0.5,
            f_hi=4.0,
            max_slowness=12.0,
            grid_points=41,
            config=self.cfg,
        )
        results = []
        worker.finishedResult.connect(results.append)

        worker.start()
        # Immediately cancel right after starting
        worker.cancel()
        finished = worker.wait(3000)
        app.processEvents()

        self.assertTrue(finished, "FKWorker failed to terminate within timeout after fast cancel")
        self.assertEqual(len(results), 0, "FKWorker should not emit finishedResult when cancelled")

    def test_vespagram_worker_fast_cancellation(self) -> None:
        """Stress: Fast cancellation right after start() for VespagramWorker."""
        worker = VespagramWorker(
            self.traces,
            azimuth_deg=60.0,
            slow_lo=1.0,
            slow_hi=12.0,
            slow_step=0.2,
            power=2.0,
            config=self.cfg,
        )
        results = []
        worker.finishedResult.connect(results.append)

        worker.start()
        worker.cancel()
        finished = worker.wait(3000)
        app.processEvents()

        self.assertTrue(finished, "VespagramWorker failed to terminate within timeout after fast cancel")
        self.assertEqual(len(results), 0, "VespagramWorker should not emit finishedResult when cancelled")

    def test_fk_worker_cancellation_during_long_iteration(self) -> None:
        """Stress: Cancellation mid-computation on an intensive grid search."""
        worker = FKWorker(
            self.traces,
            t_lo=0.0,
            t_hi=20.0,
            f_lo=0.2,
            f_hi=8.0,
            max_slowness=15.0,
            grid_points=81,  # Large grid
            config=self.cfg,
        )
        progress_log = []
        results = []
        errors = []

        def on_prog(p: int) -> None:
            progress_log.append(p)
            if p >= 5 and not worker._is_cancelled:
                worker.cancel()

        worker.progress.connect(on_prog, Qt.ConnectionType.DirectConnection)
        worker.finishedResult.connect(results.append)
        worker.error.connect(errors.append)

        worker.start()
        finished = worker.wait(6000)
        app.processEvents()

        self.assertTrue(finished, "FKWorker did not abort cleanly during mid-iteration cancel")
        self.assertEqual(len(results), 0, "No result should be emitted on cancellation")
        self.assertEqual(len(errors), 0, "Cancellation should not raise an unhandled error")

    def test_vespagram_worker_cancellation_during_long_iteration(self) -> None:
        """Stress: Cancellation mid-computation during fine-step vespagram slant stacks."""
        worker = VespagramWorker(
            self.traces,
            azimuth_deg=45.0,
            slow_lo=0.0,
            slow_hi=20.0,
            slow_step=0.05,  # 400 steps
            power=3.0,
            config=self.cfg,
        )
        progress_log = []
        results = []

        def on_prog(p: int) -> None:
            progress_log.append(p)
            if len(progress_log) >= 3 and not worker._is_cancelled:
                worker.cancel()

        worker.progress.connect(on_prog, Qt.ConnectionType.DirectConnection)
        worker.finishedResult.connect(results.append)

        worker.start()
        finished = worker.wait(6000)
        app.processEvents()

        self.assertTrue(finished, "VespagramWorker failed to cancel mid-iteration")
        self.assertEqual(len(results), 0)

    def test_dialog_destroyed_while_worker_running(self) -> None:
        """Stress: Dialog closed and deleted while worker is actively executing in background thread."""
        # 1. FK Dialog destroyed
        fk_dlg = FKAnalysisDialog(self.traces, t_lo=1.0, t_hi=15.0, config=self.cfg)
        fk_dlg.run_fk(sync=False)
        self.assertIsNotNone(fk_dlg.worker)
        self.assertTrue(fk_dlg.worker.isRunning())
        # Close dialog immediately
        fk_dlg.close()
        app.processEvents()
        self.assertFalse(fk_dlg.worker.isRunning(), "FK worker must be stopped upon dialog close")

        # 2. Vespagram Dialog destroyed
        v_dlg = VespagramDialog(self.traces, azimuth_deg=50.0, config=self.cfg)
        v_dlg.run_vespa(sync=False)
        self.assertIsNotNone(v_dlg.worker)
        self.assertTrue(v_dlg.worker.isRunning())
        v_dlg.reject()
        app.processEvents()
        self.assertFalse(v_dlg.worker.isRunning(), "Vespagram worker must be stopped upon dialog reject")

    def test_zero_trace_input(self) -> None:
        """Stress: Passing zero traces to FKWorker and VespagramWorker."""
        # FKWorker with 0 traces
        worker_fk = FKWorker(
            [],
            t_lo=0.0,
            t_hi=10.0,
            f_lo=0.5,
            f_hi=3.0,
            max_slowness=10.0,
            grid_points=21,
            config=self.cfg,
        )
        fk_errors = []
        worker_fk.error.connect(fk_errors.append)
        worker_fk.start()
        self.assertTrue(worker_fk.wait(3000))
        app.processEvents()
        self.assertEqual(len(fk_errors), 1)
        self.assertIn("at least 3 array traces", fk_errors[0])

        # VespagramWorker with 0 traces
        worker_v = VespagramWorker(
            [],
            azimuth_deg=45.0,
            slow_lo=2.0,
            slow_hi=8.0,
            slow_step=1.0,
            power=2.0,
            config=self.cfg,
        )
        v_errors = []
        worker_v.error.connect(v_errors.append)
        worker_v.start()
        self.assertTrue(worker_v.wait(3000))
        app.processEvents()
        self.assertEqual(len(v_errors), 1)
        self.assertIn("empty trace list", v_errors[0])

    def test_all_zero_trace_data_stability(self) -> None:
        """Stress: All-zero waveform data does not cause ZeroDivisionError, NaN, or crash."""
        dt = 0.05
        zero_traces = [
            Trace(data=np.zeros(300), station=st, comp="Z", delta=dt, start_epoch=1000000.0)
            for st in ["GRA1", "GRA2", "GRA3"]
        ]

        # 1. compute_fk_analysis with all zeros
        res_fk = compute_fk_analysis(
            zero_traces,
            t_lo=0.0,
            t_hi=10.0,
            f_lo=0.5,
            f_hi=3.0,
            max_slowness=10.0,
            grid_points=21,
            config=self.cfg,
        )
        self.assertIsNotNone(res_fk)
        self.assertFalse(math.isnan(res_fk["azimuth"]))
        self.assertFalse(math.isnan(res_fk["slowness"]))
        self.assertFalse(np.any(np.isnan(res_fk["grid"])))

        # 2. compute_vespagram with all zeros
        v_trs, times, slos, mat = compute_vespagram(
            zero_traces,
            azimuth_deg=45.0,
            slow_lo=2.0,
            slow_hi=8.0,
            slow_step=1.0,
            power=2.0,
            config=self.cfg,
        )
        self.assertEqual(len(v_trs), len(slos))
        self.assertFalse(np.any(np.isnan(mat)))

        # 3. FKDialog with all zeros renders cleanly without exception
        dlg_fk = FKAnalysisDialog(zero_traces, t_lo=0.0, t_hi=10.0, config=self.cfg)
        dlg_fk.run_fk(sync=True)
        self.assertIsNotNone(dlg_fk.last_result)
        dlg_fk.close()

        # 4. VespagramDialog with all zeros renders cleanly without exception
        dlg_v = VespagramDialog(zero_traces, azimuth_deg=45.0, config=self.cfg)
        dlg_v.run_vespa(sync=True)
        self.assertGreater(len(dlg_v.vespa_traces), 0)
        dlg_v.close()


class TestAdversarialClosedLoopLocation(unittest.TestCase):
    """Adversarial stress testing of Closed-Loop Location feedback and theoretical arrivals."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.ttt = get_ttt_manager(self.cfg.inputs_dir)
        self.win = MainWindow()

    def tearDown(self) -> None:
        self.win.close()
        app.processEvents()

    def test_solution_adoption_with_zero_picks(self) -> None:
        """Stress: Solution adopted when workspace traces have zero phase picks."""
        traces = _create_synthetic_grf_traces(3)
        for tr in traces:
            tr.picks.clear()
        self.win.engine.traces = traces

        param = AnalysisParamSet(
            lat=48.5,
            lon=12.0,
            depth=20.0,
            origin="02-NOV-2003_02:25:00.000",
            distance=1.8,
            residuals={},
        )

        # Must not raise exceptions
        self.win._on_adopt_network_loc(param)
        self.win.refresh_views()
        app.processEvents()

        for tr in self.win.engine.traces:
            self.assertGreater(len(tr.theo_picks), 0, "Theoretical arrivals should still be populated")
            self.assertEqual(len(tr.picks), 0, "Trace picks should remain empty without error")

    def test_solution_adoption_picks_without_stations(self) -> None:
        """Stress: Solution adopted when picks have blank station codes or unlisted stations."""
        traces = _create_synthetic_grf_traces(3)
        # Give unlisted stations
        traces[0].station = "UNKNOWN1"
        traces[1].station = ""
        traces[2].station = "GRA1"
        for tr in traces:
            for pk in tr.picks:
                pk.station = tr.station

        self.win.engine.traces = traces

        param = AnalysisParamSet(
            lat=50.0,
            lon=11.2,
            depth=10.0,
            origin="02-NOV-2003_02:25:00.000",
            distance=2.5,
            residuals={"GRA1:P": 0.12},
        )

        # Must cleanly fall back to global distance for UNKNOWN stations without crashing
        self.win._on_adopt_network_loc(param)
        self.win.refresh_views()
        app.processEvents()

        # UNKNOWN1 should have fallen back to param.distance
        self.assertAlmostEqual(self.win.engine.traces[0].distance, 2.5)
        # GRA1 should have station-specific distance
        self.assertNotEqual(self.win.engine.traces[2].distance, 2.5)
        self.assertAlmostEqual(self.win.engine.traces[2].get_pick("P").resid, 0.12)

    def test_non_converged_hypocenter_adoption(self) -> None:
        """Stress: Adopting non-converged hypocenter parameters (missing origin, extreme depth)."""
        traces = _create_synthetic_grf_traces(3)
        self.win.engine.traces = traces

        # Case 1: Missing origin time (blank string)
        param_no_orig = AnalysisParamSet(
            lat=50.0,
            lon=11.2,
            depth=10.0,
            origin="",
            distance=2.5,
        )
        self.win._on_adopt_network_loc(param_no_orig)
        self.win.refresh_views()
        app.processEvents()

        # Case 2: Deep hypocenter (700 km) and zero residuals dict
        param_deep = AnalysisParamSet(
            lat=50.0,
            lon=11.2,
            depth=700.0,
            origin="02-NOV-2003_02:25:00.000",
            distance=2.5,
            residuals=None,
        )
        self.win._on_adopt_network_loc(param_deep)
        self.win.refresh_views()
        app.processEvents()

        for tr in self.win.engine.traces:
            p_pk = tr.get_pick("P")
            if p_pk:
                self.assertIsNotNone(p_pk.resid)

    def test_theoretical_marker_rendering_without_exception(self) -> None:
        """Stress: Canvas re-renders theoretical arrivals (theo_picks) and pick residuals smoothly."""
        canvas = self.win.main_canvas
        traces = _create_synthetic_grf_traces(3)
        canvas.set_traces(traces)

        # Add theoretical picks to all traces
        for tr in traces:
            tr.theo_picks = {
                "P": 1000003.0,
                "pP": 1000007.5,
                "sP": 1000010.2,
                "S": 1000018.0,
            }
            pk = tr.get_pick("P")
            if pk:
                pk.resid = 0.42

        canvas.invalidate_cache()
        # Force backing-store pixmap recreation and painting
        canvas._ensure_static_pixmap()
        app.processEvents()

        self.assertIsNotNone(canvas._static_pixmap)
        self.assertFalse(canvas._static_pixmap.isNull())


class TestAdversarialUXFrictionPoints(unittest.TestCase):
    """Adversarial stress testing of UX friction points (beam reform, scaling, pick dedup)."""

    def setUp(self) -> None:
        self.win = MainWindow()

    def tearDown(self) -> None:
        self.win.close()
        app.processEvents()

    def test_beam_auto_reform_empty_trace_list(self) -> None:
        """Stress: Beam auto-reform when trace list is completely empty or has only BEAM."""
        # 1. Empty trace list
        self.win.engine.traces = []
        self.win._on_adopt_az_slo(az=45.0, slo=6.5)
        self.assertEqual(len(self.win.engine.traces), 0)

        # 2. Only a BEAM trace exists
        beam_only = Trace(data=np.sin(np.linspace(0, 10, 100)), station="BEAM")
        self.win.engine.traces = [beam_only]
        # Adopting az/slo should remove stale BEAM and handle empty remainder gracefully
        self.win._on_adopt_az_slo(az=90.0, slo=8.0)
        # Should not crash, and BEAM cannot be re-formed from empty trace list
        self.assertEqual(len(self.win.engine.traces), 0)

    def test_beam_auto_reform_single_trace(self) -> None:
        """Stress: Beam auto-reform with a single trace in the workspace."""
        single_tr = _create_synthetic_grf_traces(1)[0]
        self.win.engine.traces = [single_tr]
        self.win.action_form_beam()

        beam_tr = next((t for t in self.win.engine.traces if t.station == "BEAM"), None)
        self.assertIsNotNone(beam_tr)

        # Auto-reform
        self.win._on_adopt_az_slo(az=120.0, slo=5.0)
        reformed = next((t for t in self.win.engine.traces if t.station == "BEAM"), None)
        self.assertIsNotNone(reformed)
        self.assertEqual(reformed.azimuth, 120.0)
        self.assertEqual(reformed.slowness, 5.0)

    def test_beam_auto_reform_missing_azimuth_slowness(self) -> None:
        """Stress: Beam auto-reform with 0.0 or negative slowness values."""
        self.win.engine.traces = _create_synthetic_grf_traces(3)
        self.win.action_form_beam()
        self.win._on_adopt_az_slo(az=0.0, slo=0.0)
        reformed = next((t for t in self.win.engine.traces if t.station == "BEAM"), None)
        self.assertIsNotNone(reformed)
        self.assertEqual(reformed.azimuth, 0.0)
        self.assertEqual(reformed.slowness, 0.0)

    def test_beam_auto_reform_none_azimuth_slowness(self) -> None:
        """Stress: Beam auto-reform when azimuth and slowness are None."""
        self.win.engine.traces = _create_synthetic_grf_traces(3)
        self.win.action_form_beam()
        # Adopting with None should not crash on string formatting or beam reforming
        self.win._on_adopt_az_slo(az=None, slo=None)
        # Old BEAM should have been removed and not re-formed with None
        beam_tr = next((t for t in self.win.engine.traces if t.station == "BEAM"), None)
        self.assertIsNone(beam_tr)


    def test_amplitude_scaling_zero_flattening_prevention_norm_c_and_aw(self) -> None:
        """Stress: Combine raw counts (1e7) and physical displacement (0.5 nm) in NORM C and NORM AW.

        Verify that neither trace is zero-flattened (both have distinct non-zero pixel height).
        """
        for mode in ("C", "AW"):
            canvas = MainTraceCanvas()
            canvas.norm_mode = mode
            canvas.resize(800, 400)

            # Trace 1: Raw digital counts (amplitude 1e7)
            t_axis = np.linspace(0, 10 * np.pi, 500)
            raw_data = np.sin(t_axis) * 10_000_000.0
            tr_raw = Trace(data=raw_data, station="RAW", delta=0.05, comp="Z")

            # Trace 2: Restituted ground displacement (amplitude 0.5 nm)
            phys_data = np.sin(t_axis) * 0.5
            tr_phys = Trace(data=phys_data, station="PHYS", delta=0.05, comp="Z")

            canvas.set_traces([tr_raw, tr_phys])
            canvas._ensure_static_pixmap()

            # Verify tiered normalization activated
            self.assertTrue(
                getattr(canvas, "_has_mixed_tiers", False),
                f"Mixed tiers should be active in mode NORM {mode}",
            )
            self.assertGreater(canvas._cached_tier_high_max, 1_000_000.0)
            self.assertLessEqual(canvas._cached_tier_low_max, 1000.0)

            # Check scale factors computed for both traces
            slot_h = canvas.height() / 2.0
            scale_raw = (0.44 * slot_h * canvas.global_zoom * tr_raw.norm) / canvas._cached_tier_high_max
            scale_phys = (0.44 * slot_h * canvas.global_zoom * tr_phys.norm) / canvas._cached_tier_low_max

            peak_px_raw = scale_raw * 10_000_000.0
            peak_px_phys = scale_phys * 0.5

            # Assert neither trace collapses to 0 or 1 pixel flatline
            self.assertGreaterEqual(
                peak_px_phys,
                3.0,
                f"Physical trace (0.5 nm) zero-flattened in NORM {mode}! peak_px={peak_px_phys}",
            )
            self.assertGreaterEqual(
                peak_px_raw,
                3.0,
                f"Raw count trace (1e7) zero-flattened in NORM {mode}! peak_px={peak_px_raw}",
            )

    def test_rapid_double_click_pick_deduplication(self) -> None:
        """Stress: Rapid clicks at identical and close timestamps (<0.1s) do not duplicate markers."""
        tr = Trace(data=np.zeros(200), station="GRA1", delta=0.05, start_epoch=1000000.0)
        self.win.engine.traces = [tr]

        t0 = 1000010.0
        # 1. Click 1 at t0
        self.win._on_main_trace_clicked(trace_idx=0, epoch_time=t0, rel_t=10.0)
        self.assertEqual(len(tr.picks), 1)
        first_pick = tr.picks[0]
        self.assertEqual(first_pick.name, "P")

        # 2. Rapid Click 2 at t0 + 0.01 s (<0.1s threshold)
        self.win._on_main_trace_clicked(trace_idx=0, epoch_time=t0 + 0.01, rel_t=10.01)
        self.assertEqual(
            len(tr.picks),
            1,
            "Rapid double-click at +0.01s should be deduplicated, not appended",
        )
        self.assertAlmostEqual(tr.picks[0].onset_epoch, t0 + 0.01, delta=0.001)

        # 3. Rapid Click 3 at identical timestamp t0 + 0.01 s
        self.win._on_main_trace_clicked(trace_idx=0, epoch_time=t0 + 0.01, rel_t=10.01)
        self.assertEqual(len(tr.picks), 1, "Click at identical timestamp must not duplicate pick")

        # 4. Rapid Click 4 with another phase name (e.g. 'Pg') at t0 + 0.03 s (<0.1s)
        self.win.phase_panel.active_phase = "Pg"
        self.win._on_main_trace_clicked(trace_idx=0, epoch_time=t0 + 0.03, rel_t=10.03)
        self.assertEqual(
            len(tr.picks),
            1,
            "Different phase within <0.1s should replace previous pick at same arrival",
        )
        self.assertEqual(tr.picks[0].name, "Pg")

        # 5. Re-picking same phase beyond threshold (t0 + 5.0 s) updates arrival time without duplicating
        self.win._on_main_trace_clicked(trace_idx=0, epoch_time=t0 + 5.0, rel_t=15.0)
        self.assertEqual(len(tr.picks), 1, "Re-picking same phase should update existing pick position")
        self.assertAlmostEqual(tr.picks[0].onset_epoch, t0 + 5.0, delta=0.001)

        # 6. Distinct pick with different phase 'S' beyond threshold: t0 + 15.0 s
        self.win.phase_panel.active_phase = "S"
        self.win._on_main_trace_clicked(trace_idx=0, epoch_time=t0 + 15.0, rel_t=25.0)
        self.assertEqual(len(tr.picks), 2, "Distinct phase beyond 0.1s should be appended as distinct arrival")
        self.assertEqual(set(p.name for p in tr.picks), {"Pg", "S"})


if __name__ == "__main__":
    unittest.main()
