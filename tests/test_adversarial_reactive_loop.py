"""Adversarial Stress Test: UI Reactive Loop & Column Synchronization in FitDiagnosticWindow.

Empirical verification of:
1. Checkbox uncheck -> recheck cycle (checkState, Column 7 weight, and emitted signal).
2. Weight column (Col 7) direct text entry -> Col 0 sync (positive values, zero, negative, non-numeric).
3. Rapid toggling stress harness (50 rapid iterations).
4. Partial / Multi-station exclusion and re-inclusion permutations.
5. Auto-refit disabled vs manual recompute.
"""

from __future__ import annotations

import unittest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(["adversarial_reactive_loop"])

from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitModificationRequest,
    FitType,
    ResidualEntry,
)
from shm_app.gui.fit_dialog import FitDiagnosticWindow


def make_test_result(stations: list[str]) -> FitDiagnosticResult:
    residuals = []
    for st in stations:
        residuals.append(
            ResidualEntry(
                identifier=f"{st}.Z.P",
                observed=10.0,
                calculated=10.0,
                residual=0.0,
                weight=1.0,
                distance_deg=5.0,
                azimuth_deg=45.0,
                phase="P",
                station=st,
                channel="Z",
                is_outlier=False,
            )
        )
    return FitDiagnosticResult(
        fit_type=FitType.HYPOCENTER_GEIGER,
        title="Test Inversion",
        converged=True,
        iterations=3,
        dof=len(stations) - 4,
        rmse=0.1,
        wrmse=0.1,
        chi2_reduced=1.0,
        condition_number=10.0,
        parameters={},
        residuals=residuals,
    )


class TestReactiveLoopAdversarial(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FitDiagnosticBridge()
        self.win = FitDiagnosticWindow(bridge=self.bridge)
        self.stations = ["GRA1", "BUG", "BFO", "CLZ", "FUR"]
        self.res = make_test_result(self.stations)
        self.win.set_fit_result(self.res)
        app.processEvents()

        self.received_requests: list[FitModificationRequest] = []
        self.bridge.recomputeFitRequested.connect(self.received_requests.append)

    def tearDown(self) -> None:
        self.win.close()
        app.processEvents()

    def test_01_uncheck_then_recheck_station(self) -> None:
        """Requirement 1: Unchecking, then re-checking a station in table_residuals

        correctly re-enables the station (checkState remains Checked, weight resets to 1.00,
        and recomputeFitRequested is emitted with weight=1.0).
        """
        table = self.win.table_residuals
        # Station BUG is at index 1
        bug_row = 1
        self.assertEqual(table.item(bug_row, 1).text(), "BUG")

        # Initial state: Checked, weight = 1.00
        chk_item = table.item(bug_row, 0)
        weight_item = table.item(bug_row, 7)
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        self.assertEqual(weight_item.text(), "1.00")

        # Step 1: Uncheck station
        self.received_requests.clear()
        chk_item.setCheckState(Qt.CheckState.Unchecked)
        app.processEvents()

        # Check Col 0 is Unchecked
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Unchecked)
        # Check Col 7 synchronized to 0.00
        self.assertEqual(weight_item.text(), "0.00")
        # Check recomputeFitRequested emitted
        self.assertEqual(len(self.received_requests), 1)
        req1 = self.received_requests[-1]
        self.assertIn("BUG", req1.excluded_identifiers)
        self.assertIn("BUG.Z.P", req1.excluded_identifiers)
        self.assertEqual(req1.weight_overrides.get("BUG"), 0.0)

        # Step 2: Re-check station
        self.received_requests.clear()
        chk_item.setCheckState(Qt.CheckState.Checked)
        app.processEvents()

        # Check Col 0 remains Checked (no deadlock / reverting to Unchecked)
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        # Check Col 7 weight resets to 1.00
        self.assertEqual(weight_item.text(), "1.00")
        # Check recomputeFitRequested emitted with station re-included and weight=1.0
        self.assertEqual(len(self.received_requests), 1)
        req2 = self.received_requests[-1]
        self.assertNotIn("BUG", req2.excluded_identifiers)
        self.assertNotIn("BUG.Z.P", req2.excluded_identifiers)
        self.assertEqual(req2.weight_overrides.get("BUG"), 1.0)
        self.assertEqual(req2.weight_overrides.get("BUG.Z.P"), 1.0)

    def test_02_weight_column_positive_and_zero_sync(self) -> None:
        """Requirement 2: Typing a positive weight into Column 7 checks Column 0,

        and typing 0.00 unchecks Column 0.
        """
        table = self.win.table_residuals
        chk_item = table.item(0, 0)
        weight_item = table.item(0, 7)

        # 2a. Type 0.00 -> unchecks Col 0
        self.received_requests.clear()
        weight_item.setText("0.00")
        app.processEvents()

        self.assertEqual(chk_item.checkState(), Qt.CheckState.Unchecked)
        self.assertEqual(len(self.received_requests), 1)
        self.assertIn("GRA1", self.received_requests[-1].excluded_identifiers)
        self.assertEqual(self.received_requests[-1].weight_overrides.get("GRA1"), 0.0)

        # 2b. Type positive weight "0.75" -> checks Col 0
        self.received_requests.clear()
        weight_item.setText("0.75")
        app.processEvents()

        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        self.assertEqual(len(self.received_requests), 1)
        self.assertNotIn("GRA1", self.received_requests[-1].excluded_identifiers)
        self.assertEqual(self.received_requests[-1].weight_overrides.get("GRA1"), 0.75)

        # 2c. Type "0" (without decimal places) -> unchecks Col 0
        self.received_requests.clear()
        weight_item.setText("0")
        app.processEvents()

        self.assertEqual(chk_item.checkState(), Qt.CheckState.Unchecked)
        self.assertIn("GRA1", self.received_requests[-1].excluded_identifiers)

        # 2d. Type "2.5" -> checks Col 0
        self.received_requests.clear()
        weight_item.setText("2.5")
        app.processEvents()

        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        self.assertNotIn("GRA1", self.received_requests[-1].excluded_identifiers)
        self.assertEqual(self.received_requests[-1].weight_overrides.get("GRA1"), 2.5)

    def test_03_weight_column_adversarial_inputs(self) -> None:
        """Adversarial stress-test on Column 7 text inputs:

        - Negative values ("-1.5") -> reverted to previous valid weight (1.00), keeps Col 0 Checked, no spurious recompute
        - Non-numeric strings ("CORRUPT_INPUT") -> reverted to previous valid weight (1.00), keeps Col 0 Checked, no spurious recompute
        - Whitespace padded numbers ("   0.42   ") -> normalized to "0.42", emits weight 0.42, keeps Col 0 Checked
        """
        table = self.win.table_residuals
        chk_item = table.item(2, 0)  # BFO
        weight_item = table.item(2, 7)

        # Negative value -> rejected and reverted to 1.00
        self.received_requests.clear()
        weight_item.setText("-1.5")
        app.processEvents()
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        self.assertEqual(weight_item.text(), "1.00")
        self.assertEqual(len(self.received_requests), 0)

        # Non-numeric string -> rejected and reverted to 1.00
        self.received_requests.clear()
        weight_item.setText("CORRUPT_INPUT")
        app.processEvents()
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        self.assertEqual(weight_item.text(), "1.00")
        self.assertEqual(len(self.received_requests), 0)

        # Whitespace padded valid number -> accepted and formatted
        self.received_requests.clear()
        weight_item.setText("   0.42   ")
        app.processEvents()
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)
        self.assertEqual(weight_item.text(), "0.42")
        self.assertAlmostEqual(self.received_requests[-1].weight_overrides.get("BFO"), 0.42)

    def test_04_rapid_oscillation_stress_harness(self) -> None:
        """Rapidly toggle check/uncheck 50 times on multiple stations to ensure

        no race conditions, no recursive signal cascade crashes, and clean final state.
        """
        table = self.win.table_residuals
        chk_bug = table.item(1, 0)
        chk_clz = table.item(3, 0)

        for i in range(50):
            target_state = Qt.CheckState.Unchecked if (i % 2 == 0) else Qt.CheckState.Checked
            chk_bug.setCheckState(target_state)
            chk_clz.setCheckState(target_state)
            app.processEvents()

        # Both ended on Checked (i=49 -> Checked)
        self.assertEqual(chk_bug.checkState(), Qt.CheckState.Checked)
        self.assertEqual(chk_clz.checkState(), Qt.CheckState.Checked)
        self.assertEqual(table.item(1, 7).text(), "1.00")
        self.assertEqual(table.item(3, 7).text(), "1.00")

        # Verify last emitted request has both included
        last_req = self.received_requests[-1]
        self.assertNotIn("BUG", last_req.excluded_identifiers)
        self.assertNotIn("CLZ", last_req.excluded_identifiers)
        self.assertEqual(last_req.weight_overrides.get("BUG"), 1.0)
        self.assertEqual(last_req.weight_overrides.get("CLZ"), 1.0)

    def test_05_multi_station_selective_exclusion(self) -> None:
        """Test selectively excluding 3 stations and re-including only 1 station."""
        table = self.win.table_residuals

        # Exclude GRA1 (0), BUG (1), BFO (2)
        table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
        table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
        table.item(2, 0).setCheckState(Qt.CheckState.Unchecked)
        app.processEvents()

        last_req = self.received_requests[-1]
        self.assertIn("GRA1", last_req.excluded_identifiers)
        self.assertIn("BUG", last_req.excluded_identifiers)
        self.assertIn("BFO", last_req.excluded_identifiers)
        self.assertNotIn("CLZ", last_req.excluded_identifiers)
        self.assertNotIn("FUR", last_req.excluded_identifiers)

        # Now re-include only BUG (1)
        table.item(1, 0).setCheckState(Qt.CheckState.Checked)
        app.processEvents()

        last_req = self.received_requests[-1]
        self.assertIn("GRA1", last_req.excluded_identifiers)
        self.assertNotIn("BUG", last_req.excluded_identifiers)
        self.assertIn("BFO", last_req.excluded_identifiers)
        self.assertEqual(last_req.weight_overrides.get("BUG"), 1.0)
        self.assertEqual(last_req.weight_overrides.get("GRA1"), 0.0)
        self.assertEqual(last_req.weight_overrides.get("BFO"), 0.0)

    def test_06_auto_refit_toggle_behavior(self) -> None:
        """When auto-refit is unchecked, edits should update table display but not emit

        until manual 'Recompute Inversion' button is clicked.
        """
        self.win.chk_auto_refit.setChecked(False)
        self.received_requests.clear()

        table = self.win.table_residuals
        # Uncheck FUR (row 4)
        table.item(4, 0).setCheckState(Qt.CheckState.Unchecked)
        app.processEvents()

        # Weight text is synced to 0.00
        self.assertEqual(table.item(4, 7).text(), "0.00")
        # BUT no request emitted because auto_refit is disabled
        self.assertEqual(len(self.received_requests), 0)

        # Now click manual recompute button
        self.win.btn_recompute.click()
        app.processEvents()

        self.assertEqual(len(self.received_requests), 1)
        self.assertIn("FUR", self.received_requests[-1].excluded_identifiers)


if __name__ == "__main__":
    unittest.main()
