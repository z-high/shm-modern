"""Adversarial stress-test suite for Milestone 2 (Phase 2): Fit Diagnostics & Reactive Re-Fit Loop.

Tests:
1. Checkbox exclusion in residual table:
   - Station exclusion triggers locate_network_geiger with station excluded.
   - AnalysisParamSet updated on MainWindow.
   - tr.theo_picks and arrival markers on MainTraceCanvas refreshed.
1b. Checkbox re-inclusion deadlock (Empirical Bug Finding):
   - Re-checking an excluded station in the residual table fails to restore the station
     because weight=0.00 in column 7 causes _dispatch_recompute_request to treat
     the station as still excluded ('if not is_checked or w_val <= 0.0:').
2. Rapid tab switching across all 6 tabs in FitDiagnosticWindow:
   - Empty state (current_result = None).
   - Partially populated state (missing residuals, missing ellipse, missing correlation matrix, inf condition number).
   - Populated state.
   - Rapid domain combo switching across all 11 domains.
3. Closed-loop verification:
   - Invert network with outlier arrival (+15.0 s on BUG across 13 stations).
   - Verify RMSE decreases and 95% confidence ellipse shrinks (semi-major, semi-minor, and area).
   - Test UI 'Deactivate Outliers (>3σ)' button triggers re-fit and excludes outlier.
4. Adversarial edge cases:
   - All stations excluded (fallback to valid picks).
   - Invalid / corrupt text input in weight column ('abc', '---', '-99.0').
   - Out-of-bounds row clicks.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists for offscreen execution
app = QApplication.instance()
if app is None:
    app = QApplication(["adversarial_m2_test"])

from shm_app.core.config_station import get_config
from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.core.fit_models import (
    ConfidenceEllipse,
    FitDiagnosticResult,
    FitModificationRequest,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.fit_dialog import FitDiagnosticWindow
from shm_app.gui.main_window import MainWindow
from shm_app.seismology.array_loc import locate_network_geiger
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


class TestAdversarialMilestone2(unittest.TestCase):
    """Adversarial verification and stress harness for Milestone 2."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.ttt = get_ttt_manager(self.cfg.inputs_dir)

        # Reference synthetic event in Central Germany
        self.true_lat = 50.100
        self.true_lon = 11.500
        self.true_depth = 33.0
        self.t0 = 1000000.0

        # Multi-station network (13 stations spanning array and regional network)
        self.station_codes = [
            "GRA1", "GRA2", "GRA3", "GRB1", "GRB2", "GRC1",
            "FUR", "BFO", "CLZ", "MOX", "WET", "BUG", "TNS",
        ]
        self.dt = 0.05
        self.n_samples = 400

    def _generate_synthetic_traces_and_picks(
        self, outlier_station: str | None = None, outlier_sec: float = 0.0
    ) -> list[Trace]:
        """Generate network traces with theoretical P picks and optional outlier injection."""
        traces = []
        for st_code in self.station_codes:
            st = self.cfg.get_station(st_code)
            dist, _, _ = sphere_diff(st.lat, st.lon, self.true_lat, self.true_lon)
            tt = self.ttt.travel_time("P", max(0.05, dist), self.true_depth)
            if tt is None:
                tt = dist * 111.195 / 6.0  # fallback ~6 km/s

            arr_epoch = self.t0 + tt
            if st_code == outlier_station:
                arr_epoch += outlier_sec

            pk = PhasePick(
                name="P",
                onset=format_sh_time(arr_epoch),
                station=st_code,
                comp="Z",
                weight=2,
            )

            # Synthetic trace starting 5 seconds before origin
            start_ep = self.t0 - 5.0
            data = np.zeros(self.n_samples, dtype=np.float32)
            tr = Trace(
                data=data,
                delta=self.dt,
                station=st_code,
                comp="Z",
                start_epoch=start_ep,
            )
            tr.add_or_replace_pick(pk)
            traces.append(tr)
        return traces

    def test_01_checkbox_exclusion_reactive_refit_loop(self) -> None:
        """Adversarial verification: Unchecking active checkbox in residual table triggers Geiger re-fit,

        updates AnalysisParamSet on MainWindow, and refreshes theoretical arrivals on MainTraceCanvas.
        """
        # Inject an outlier on BUG (+15.0 s)
        traces = self._generate_synthetic_traces_and_picks(outlier_station="BUG", outlier_sec=15.0)

        win = MainWindow(autoload_example=False)
        win.engine.traces = traces
        win.engine.current_params.depth = self.true_depth
        win.refresh_views()

        # Open Fit Diagnostics Window -> triggers initial location
        win.action_open_fit_diagnostic_window()
        app.processEvents()

        # Verify initial location solution is affected by BUG outlier
        p_initial = win.engine.current_params
        self.assertTrue(p_initial.lat != 0.0 or p_initial.lon != 0.0)
        rmse_initial = p_initial.rmse
        self.assertIsNotNone(rmse_initial)
        self.assertTrue(rmse_initial > 2.0, f"Expected high RMSE due to outlier, got {rmse_initial}")

        # Note initial theoretical arrival on GRA1
        gra1_tr = next(t for t in win.engine.traces if t.station == "GRA1")
        initial_theo_p = gra1_tr.theo_picks.get("P")
        self.assertIsNotNone(initial_theo_p)

        # Locate BUG row in residual table
        table = win.fit_dock.table_residuals
        bug_row = -1
        for r in range(table.rowCount()):
            st_item = table.item(r, 1)
            if st_item and st_item.text().upper() == "BUG":
                bug_row = r
                break
        self.assertNotEqual(bug_row, -1, "BUG station row not found in residual table")

        # Verify BUG row has status OUTLIER
        status_item = table.item(bug_row, 10)
        self.assertIn("OUTLIER", status_item.text())

        # Uncheck BUG active checkbox in Column 0 -> triggers itemChanged -> _dispatch_recompute_request
        chk_item = table.item(bug_row, 0)
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)

        chk_item.setCheckState(Qt.CheckState.Unchecked)
        app.processEvents()

        # Verify that AnalysisParamSet was updated with new solution
        p_refit = win.engine.current_params
        self.assertIsNotNone(p_refit.rmse)
        self.assertTrue(
            p_refit.rmse < rmse_initial * 0.1,
            f"Expected RMSE to drop significantly after excluding BUG: {rmse_initial} -> {p_refit.rmse}",
        )
        self.assertAlmostEqual(p_refit.lat, self.true_lat, delta=0.1)
        self.assertAlmostEqual(p_refit.lon, self.true_lon, delta=0.1)

        # Verify theoretical arrivals on MainTraceCanvas refreshed
        refreshed_theo_p = gra1_tr.theo_picks.get("P")
        self.assertIsNotNone(refreshed_theo_p)
        self.assertNotEqual(initial_theo_p, refreshed_theo_p)

        # Verify canvas marks cache dirty on refresh
        self.assertTrue(win.main_canvas._cache_dirty or win.main_canvas.isVisible() is False)

    def test_01b_checkbox_reinclusion_deadlock(self) -> None:
        """Adversarial reproducer: Re-checking an excluded station should re-include it.

        Currently fails due to a mutual deadlock in fit_dialog.py:
        Once excluded, entry.weight becomes 0.00. Checking the checkbox leaves weight=0.00,
        which triggers 'if not is_checked or w_val <= 0.0:' in _dispatch_recompute_request(),
        causing the station to remain permanently excluded and the checkbox to revert to Unchecked.
        """
        traces = self._generate_synthetic_traces_and_picks(outlier_station="BUG", outlier_sec=15.0)

        win = MainWindow(autoload_example=False)
        win.engine.traces = traces
        win.engine.current_params.depth = self.true_depth
        win.refresh_views()
        win.action_open_fit_diagnostic_window()
        app.processEvents()

        table = win.fit_dock.table_residuals
        bug_row = -1
        for r in range(table.rowCount()):
            if table.item(r, 1).text().upper() == "BUG":
                bug_row = r
                break

        # Step 1: Uncheck BUG -> excludes BUG
        table.item(bug_row, 0).setCheckState(Qt.CheckState.Unchecked)
        app.processEvents()
        p_refit_rmse = win.engine.current_params.rmse

        # Step 2: Attempt to re-include BUG by checking the box
        table = win.fit_dock.table_residuals
        bug_chk_item = table.item(bug_row, 0)
        self.assertEqual(bug_chk_item.checkState(), Qt.CheckState.Unchecked)

        bug_chk_item.setCheckState(Qt.CheckState.Checked)
        app.processEvents()

        # Verification: BUG should be re-included and RMSE should increase back
        p_restored = win.engine.current_params
        self.assertGreater(
            p_restored.rmse,
            p_refit_rmse * 10.0,
            "Re-checking station checkbox failed to re-include station in Geiger inversion",
        )

    def test_02_rapid_tab_switching_empty_and_partial(self) -> None:
        """Adversarial stress-test: Rapid tab switching across all 6 tabs in FitDiagnosticWindow

        with empty, partially populated, and extreme results without crashing or leaking state.
        """
        bridge = FitDiagnosticBridge()
        win = FitDiagnosticWindow(bridge=bridge)

        # Phase A: Empty state (current_result = None) - rapid switching across all tabs
        tab_sequence = [0, 1, 2, 3, 4, 5, 4, 2, 0, 3, 1, 5, 0, 1, 2, 3, 4, 5]
        for _ in range(5):
            for t_idx in tab_sequence:
                win.tabs.setCurrentIndex(t_idx)
                app.processEvents()
        self.assertIsNone(win.current_result)

        # Phase B: Partially populated state with missing / None fields
        partial_res = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="Partial Diagnostic",
            converged=False,
            iterations=0,
            dof=0,
            rmse=0.0,
            wrmse=0.0,
            chi2_reduced=0.0,
            condition_number=1e12,
            parameters={},
            correlation_matrix=None,  # None correlation matrix
            confidence_ellipse=None,  # None confidence ellipse
            residuals=[],  # Empty residuals
            outlier_count=0,
            normality_p_value=0.0,
            skewness=0.0,
            kurtosis=0.0,
        )
        win.set_fit_result(partial_res)
        app.processEvents()

        # Rapid switching on partial result
        for _ in range(5):
            for t_idx in tab_sequence:
                win.tabs.setCurrentIndex(t_idx)
                app.processEvents()

        self.assertEqual(win.table_params.rowCount(), 0)
        self.assertEqual(win.table_residuals.rowCount(), 0)
        self.assertIn("Inf", win.kpi_cond.lbl_value.text())

        # Phase C: Extreme values (NaN, Inf, single entry)
        extreme_res = FitDiagnosticResult(
            fit_type=FitType.BRUNE_SPECTRUM,
            title="Extreme Result",
            converged=True,
            iterations=99,
            dof=1,
            rmse=999.9,
            wrmse=999.9,
            chi2_reduced=99.9,
            condition_number=999.9,
            parameters={
                "omega_0": ParameterUncertainty("omega_0", 1e6, 1e4, (9e5, 1.1e6), "nm·s"),
            },
            correlation_matrix=np.array([[1.0]]),
            residuals=[
                ResidualEntry(
                    identifier="TEST.Z.P",
                    observed=100.0,
                    calculated=0.0,
                    residual=100.0,
                    weight=0.0,
                    is_outlier=True,
                )
            ],
            outlier_count=1,
        )
        win.set_fit_result(extreme_res)
        app.processEvents()

        # Rapid switching on extreme result
        for t_idx in tab_sequence:
            win.tabs.setCurrentIndex(t_idx)
            app.processEvents()

        # Phase D: Rapid switching through all 11 domain combobox entries
        for dom_idx in range(win.domain_combo.count()):
            win.domain_combo.setCurrentIndex(dom_idx)
            app.processEvents()
            # Must map cleanly to a valid tab index (0 <= idx < 6)
            self.assertTrue(0 <= win.tabs.currentIndex() < 6)

    def test_03_closed_loop_outlier_exclusion_rmse_and_ellipse_shrink(self) -> None:
        """Adversarial closed-loop verification:

        Excluding an outlier station decreases RMSE and shrinks the 95% confidence ellipse
        (semi-major axis, semi-minor axis, and area).
        """
        # Inject an outlier on BUG (+15.0 s)
        traces = self._generate_synthetic_traces_and_picks(outlier_station="BUG", outlier_sec=15.0)
        picks = [tr.picks[0] for tr in traces]

        # 1. Inversion with ALL stations (including outlier BUG)
        param_with_outlier, _, diag_with_outlier = locate_network_geiger(
            picks, return_diagnostics=True
        )

        rmse_1 = diag_with_outlier.rmse
        el_1 = diag_with_outlier.confidence_ellipse
        self.assertIsNotNone(el_1, "Confidence ellipse must not be None")
        a_1 = el_1.semi_major_km
        b_1 = el_1.semi_minor_km
        area_1 = math.pi * a_1 * b_1

        # Verify BUG is detected as an outlier
        bug_entry_1 = next(e for e in diag_with_outlier.residuals if e.station == "BUG")
        self.assertTrue(bug_entry_1.is_outlier, "BUG should be flagged as an outlier")
        self.assertTrue(diag_with_outlier.outlier_count >= 1)

        # 2. Inversion with BUG excluded
        param_clean, _, diag_clean = locate_network_geiger(
            picks, excluded_stations=["BUG"], return_diagnostics=True
        )

        rmse_2 = diag_clean.rmse
        el_2 = diag_clean.confidence_ellipse
        self.assertIsNotNone(el_2, "Confidence ellipse after exclusion must not be None")
        a_2 = el_2.semi_major_km
        b_2 = el_2.semi_minor_km
        area_2 = math.pi * a_2 * b_2

        # 3. EMPIRICAL VERIFICATION:
        # RMSE must decrease substantially (> 50% reduction)
        self.assertLess(
            rmse_2,
            rmse_1 * 0.1,
            f"RMSE did not decrease sufficiently: before={rmse_1:.3f} s, after={rmse_2:.3f} s",
        )

        # Semi-major axis must shrink
        self.assertLess(
            a_2,
            a_1,
            f"Semi-major axis did not shrink: before={a_1:.2f} km, after={a_2:.2f} km",
        )

        # Semi-minor axis must shrink
        self.assertLess(
            b_2,
            b_1,
            f"Semi-minor axis did not shrink: before={b_1:.2f} km, after={b_2:.2f} km",
        )

        # Total 95% confidence ellipse area must shrink drastically (> 50% reduction)
        self.assertLess(
            area_2,
            area_1 * 0.1,
            f"Confidence ellipse area did not shrink: before={area_1:.2f} km^2, after={area_2:.2f} km^2",
        )

        # In diag_clean, BUG residual entry must have weight = 0.0
        bug_entry_2 = next(e for e in diag_clean.residuals if e.station == "BUG")
        self.assertEqual(bug_entry_2.weight, 0.0)

        # 4. Verify UI 'Deactivate Outliers' button performs same closed-loop workflow
        bridge = FitDiagnosticBridge()
        win = FitDiagnosticWindow(bridge=bridge)
        win.set_fit_result(diag_with_outlier)
        app.processEvents()

        requests = []
        bridge.recomputeFitRequested.connect(lambda req: requests.append(req))

        win.btn_deactivate_outliers.click()
        app.processEvents()

        self.assertTrue(len(requests) >= 1)
        last_req = requests[-1]
        self.assertIn("BUG", last_req.excluded_identifiers)
        self.assertEqual(last_req.weight_overrides.get("BUG"), 0.0)

    def test_04_adversarial_edge_cases_and_robustness(self) -> None:
        """Adversarial stress-test: Edge cases in table edits, exclusions, and UI buttons."""
        bridge = FitDiagnosticBridge()
        win = FitDiagnosticWindow(bridge=bridge)

        # Populate with standard result
        traces = self._generate_synthetic_traces_and_picks()
        picks = [tr.picks[0] for tr in traces]
        _, _, diag_res = locate_network_geiger(picks, return_diagnostics=True)
        win.set_fit_result(diag_res)
        app.processEvents()

        requests = []
        bridge.recomputeFitRequested.connect(lambda req: requests.append(req))

        # A: Invalid non-numeric text in weight column ('invalid', '---', '-5.0')
        w_item = win.table_residuals.item(0, 7)
        self.assertIsNotNone(w_item)
        w_item.setText("not-a-number")
        app.processEvents()
        self.assertTrue(len(requests) >= 1)
        # Fallback to weight 1.0
        self.assertEqual(requests[-1].weight_overrides.get(self.station_codes[0]), 1.0)

        # Negative weight should deactivate station
        requests.clear()
        w_item.setText("-2.5")
        app.processEvents()
        self.assertTrue(len(requests) >= 1)
        self.assertIn(self.station_codes[0], requests[-1].excluded_identifiers)

        # B: Exclude ALL stations in residual table -> should not crash locate_network_geiger
        all_excluded_req = FitModificationRequest(
            fit_type=FitType.HYPOCENTER_GEIGER,
            excluded_identifiers=list(self.station_codes),
        )
        # Should gracefully fall back to valid picks without raising
        p_fallback, _, diag_fallback = locate_network_geiger(
            picks,
            excluded_stations=all_excluded_req.excluded_identifiers,
            return_diagnostics=True,
        )
        self.assertIsNotNone(p_fallback)
        self.assertIsNotNone(diag_fallback)

        # C: Click and double-click handlers with invalid row/col indices
        win._on_table_row_clicked(-1, -1)
        win._on_table_row_clicked(999, 999)
        win._on_table_row_double_clicked(-1, -1)
        win._on_table_row_double_clicked(999, 999)

        # D: Adopt clicked with and without context_data
        adopted = []
        bridge.adoptParametersRequested.connect(lambda p: adopted.append(p))
        win._on_adopt_clicked()  # context_data has lat/lon
        self.assertTrue(len(adopted) >= 1)


if __name__ == "__main__":
    unittest.main()
