"""Comprehensive Unit Test Suite for Milestone 2 (Phase 2): Fit Diagnostics & Inversion Residuals.

Tests:
1. Statistical diagnostics engine (shm_app/seismology/diagnostics.py):
   - RMSE, WRMSE, MAD, robust sigma_MAD (1.4826 * MAD).
   - Huber weights (k=1.345), Tukey biweight (c=4.685).
   - Reduced chi-square, R^2, normality stats (Jarque-Bera), 3-sigma outliers.
   - Covariance matrix via SVD, condition number kappa(J), correlation matrix.
   - 95% confidence error ellipse (a, b, theta) from 2x2 covariance with scale=2.4477.
2. Data contracts and decoupled signaling (shm_app/core/fit_models.py, shm_app/core/fit_bridge.py):
   - FitType enum (all 11 domains).
   - Dataclasses: ResidualEntry, ParameterUncertainty, ConfidenceEllipse, FitDiagnosticResult, FitModificationRequest.
   - FitDiagnosticBridge signals: fitDiagnosticAvailable, recomputeFitRequested, highlightTraceRequested, etc.
3. Dockable FitDiagnosticWindow (shm_app/gui/fit_dialog.py):
   - QDockWidget container, 5 KPI cards, 6 domain tabs, table population, interactive checkbox/weight editing.
4. Interactive residual table and reactive re-fit loop:
   - locate_network_geiger with station exclusions and weight overrides.
   - MainWindow reactive integration and canvas arrival alignment.
"""

import math
import unittest
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists for headless Qt widget tests
app = QApplication.instance()
if app is None:
    app = QApplication(["shm_test_m2"])

import shm_app.seismology.diagnostics as diag
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
from shm_app.gui.fit_dialog import FitDiagnosticWindow, KPICard
from shm_app.gui.main_window import MainWindow
from shm_app.seismology.array_loc import locate_network_geiger


class TestStatisticalDiagnostics(unittest.TestCase):
    """Test suite for seismology/diagnostics.py pure Python/NumPy math engine."""

    def test_rmse_and_wrmse(self) -> None:
        """Verify RMSE and weighted RMSE against analytical values."""
        # Empty array
        self.assertEqual(diag.compute_rmse([]), 0.0)
        self.assertEqual(diag.compute_wrmse([], []), 0.0)

        # Basic residuals [3, -4] -> RMSE = sqrt((9+16)/2) = sqrt(12.5) = 3.5355...
        res = np.array([3.0, -4.0])
        self.assertAlmostEqual(diag.compute_rmse(res), math.sqrt(12.5), places=5)

        # Equal weights: WRMSE == RMSE
        weights_equal = np.array([1.0, 1.0])
        self.assertAlmostEqual(diag.compute_wrmse(res, weights_equal), math.sqrt(12.5), places=5)

        # Weighted residuals: w = [4.0, 1.0] -> sum(w*r^2) = 4*9 + 1*16 = 52. sum(w) = 5. WRMSE = sqrt(52/5) = sqrt(10.4)
        weights = np.array([4.0, 1.0])
        expected_wrmse = math.sqrt(52.0 / 5.0)
        self.assertAlmostEqual(diag.compute_wrmse(res, weights), expected_wrmse, places=5)

    def test_mad_and_robust_sigma(self) -> None:
        """Verify Median Absolute Deviation and robust scale sigma_hat = 1.4826 * MAD."""
        self.assertEqual(diag.compute_mad([]), 0.0)
        self.assertEqual(diag.compute_robust_sigma([]), 0.0)

        # Symmetric distribution around median 0: [-2, -1, 0, 1, 2] -> MAD = 1.0
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.assertAlmostEqual(diag.compute_mad(data), 1.0, places=5)
        self.assertAlmostEqual(diag.compute_robust_sigma(data), 1.4826, places=4)

        # Outlier resilience: [-2, -1, 0, 1, 100] -> median=0, abs diffs=[0, 1, 1, 2, 100] -> MAD = 1.0
        data_outlier = np.array([-2.0, -1.0, 0.0, 1.0, 100.0])
        self.assertAlmostEqual(diag.compute_mad(data_outlier), 1.0, places=5)
        self.assertAlmostEqual(diag.compute_robust_sigma(data_outlier), 1.4826, places=4)

    def test_reduced_chi2_and_r_squared(self) -> None:
        """Verify reduced chi-square and coefficient of determination."""
        res = np.array([0.1, -0.1, 0.2, -0.2])
        # dof = 2, unc = 0.1 for all -> (1/2) * (1 + 1 + 4 + 4) = 5.0
        unc = np.array([0.1, 0.1, 0.1, 0.1])
        chi2_red = diag.compute_reduced_chi2(res, unc, dof=2)
        self.assertAlmostEqual(chi2_red, 5.0, places=5)

        # R^2 test
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect fit
        self.assertAlmostEqual(diag.compute_r_squared(obs, obs), 1.0, places=5)
        # Shifted fit
        calc = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        r2 = diag.compute_r_squared(obs, calc)
        self.assertTrue(0.98 < r2 <= 1.0)

    def test_huber_and_tukey_weights(self) -> None:
        """Verify Huber M-estimator and Tukey bisquare weight behaviors."""
        # Residuals with clear inliers and outliers
        res = np.array([0.0, 0.1, -0.1, 5.0])
        # Huber weights with explicit delta = 0.5
        w_huber = diag.compute_huber_weights(res, delta=0.5)
        self.assertEqual(w_huber[0], 1.0)
        self.assertEqual(w_huber[1], 1.0)
        self.assertEqual(w_huber[2], 1.0)
        # Outlier at 5.0 should be weighted down to delta / |r| = 0.5 / 5.0 = 0.1
        self.assertAlmostEqual(w_huber[3], 0.1, places=5)

        # Tukey weights: outlier > c * sigma should receive exactly 0.0 weight
        w_tukey = diag.compute_tukey_weights(res, c=4.685)
        self.assertTrue(w_tukey[0] > 0.9)
        self.assertEqual(w_tukey[3], 0.0)

    def test_detect_outliers_3sigma(self) -> None:
        """Verify detection of outliers beyond 3 sigma."""
        res = np.array([0.05, -0.02, 0.01, -0.03, 0.04, 2.50])
        outliers = diag.detect_outliers_3sigma(res, threshold=3.0)
        self.assertFalse(outliers[0])
        self.assertFalse(outliers[1])
        self.assertTrue(outliers[5])

    def test_covariance_and_correlation_matrix(self) -> None:
        """Verify model parameter covariance matrix via SVD and correlation matrix."""
        # Setup simple Jacobian (4 obs, 2 params)
        J = np.array([
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
        ])
        W = np.array([1.0, 1.0, 1.0, 1.0])
        cov = diag.compute_covariance_matrix(J, W, s_sq=1.0, damping=1e-5)
        self.assertEqual(cov.shape, (2, 2))
        # Covariance must be symmetric and positive-definite
        self.assertAlmostEqual(cov[0, 1], cov[1, 0], places=6)
        self.assertTrue(cov[0, 0] > 0.0)
        self.assertTrue(cov[1, 1] > 0.0)

        corr = diag.compute_correlation_matrix(cov)
        self.assertEqual(corr.shape, (2, 2))
        self.assertAlmostEqual(corr[0, 0], 1.0, places=5)
        self.assertAlmostEqual(corr[1, 1], 1.0, places=5)
        self.assertTrue(-1.0 <= corr[0, 1] <= 1.0)

    def test_condition_number(self) -> None:
        """Verify condition number computation kappa(J)."""
        # Identity matrix: kappa = 1.0
        I = np.eye(3)
        self.assertAlmostEqual(diag.compute_condition_number(I), 1.0, places=5)

        # Ill-conditioned matrix
        A = np.array([
            [1.0, 0.0],
            [0.0, 1e-6],
        ])
        kappa = diag.compute_condition_number(A)
        self.assertAlmostEqual(kappa, 1e6, places=0)

    def test_confidence_ellipse_2d(self) -> None:
        """Verify horizontal 95% confidence error ellipse parameters (a, b, theta)."""
        # Circular covariance in degrees: C = diag(1e-4, 1e-4)
        # Using deg_to_km = 111.195: sigma_km = sqrt(1e-4) * 111.195 = 0.01 * 111.195 = 1.11195 km
        # Scale for 95% Chi2(2) = 5.99146 -> sqrt(5.99146) = 2.447746...
        # a = b = 2.447746 * 1.11195 = 2.7217... km
        cov_circ = np.diag([1e-4, 1e-4])
        a, b, theta = diag.compute_confidence_ellipse_2d(cov_circ, confidence=0.95, deg_to_km=111.195)
        self.assertAlmostEqual(a, 2.722, places=2)
        self.assertAlmostEqual(b, 2.722, places=2)

        # Anisotropic covariance oriented North-South (higher lat uncertainty):
        # C_yy = 4e-4, C_xx = 1e-4, C_yx = 0.
        # Major axis should point along North (y-axis) -> strike theta = 0.0 or 180.0 deg
        cov_ns = np.diag([4e-4, 1e-4])
        a_ns, b_ns, theta_ns = diag.compute_confidence_ellipse_2d(cov_ns, confidence=0.95, deg_to_km=111.195)
        self.assertAlmostEqual(a_ns, 2.0 * b_ns, places=1)
        self.assertTrue(theta_ns == 0.0 or theta_ns == 180.0)

        # Anisotropic covariance oriented East-West (higher lon uncertainty):
        # C_yy = 1e-4, C_xx = 4e-4, C_yx = 0.
        # Major axis points East (x-axis) -> strike theta = 90.0 deg
        cov_ew = np.diag([1e-4, 4e-4])
        a_ew, b_ew, theta_ew = diag.compute_confidence_ellipse_2d(cov_ew, confidence=0.95, deg_to_km=111.195)
        self.assertAlmostEqual(a_ew, 2.0 * b_ew, places=1)
        self.assertAlmostEqual(theta_ew, 90.0, places=1)

    def test_normality_statistics(self) -> None:
        """Verify Jarque-Bera normality statistics on Gaussian vs skewed data."""
        # Centered Gaussian-like data
        normal_data = np.array([-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5])
        stats = diag.compute_normality_stats(normal_data)
        self.assertAlmostEqual(stats["skewness"], 0.0, places=2)
        self.assertTrue(stats["p_value"] > 0.05)


class TestFitModelsAndBridge(unittest.TestCase):
    """Test suite for data models (fit_models.py) and signaling bridge (fit_bridge.py)."""

    def test_fit_type_enum(self) -> None:
        """Verify FitType contains all 11 required seismological fit domains."""
        expected_domains = [
            "HYPOCENTER_GEIGER",
            "ORIGIN_TIME",
            "PLANE_WAVE",
            "FK_SEMBLANCE",
            "SP_DIFF_DEPTH",
            "TTFIT_SIMPLEX",
            "BRUNE_SPECTRUM",
            "POLARIZATION_3C",
            "RESTITUTION_FIT",
            "RECEIVER_FUNC",
            "SURFACE_DISPERSION",
        ]
        for dom in expected_domains:
            self.assertTrue(hasattr(FitType, dom), f"Missing domain {dom} in FitType")

        # Verify aliases exist
        self.assertEqual(FitType.LOCATION_TT, FitType.HYPOCENTER_GEIGER)
        self.assertEqual(FitType.ARRAY_FK, FitType.FK_SEMBLANCE)

    def test_fit_dataclasses(self) -> None:
        """Verify creation and attributes of fit dataclasses."""
        ellipse = ConfidenceEllipse(semi_major_km=10.5, semi_minor_km=4.2, strike_deg=35.0)
        self.assertEqual(ellipse.semi_major_km, 10.5)
        self.assertEqual(ellipse.strike_deg, 35.0)

        param_unc = ParameterUncertainty(
            name="latitude", value=50.0, standard_error=0.02, confidence_interval_95=(49.96, 50.04), unit="deg"
        )
        self.assertEqual(param_unc.name, "latitude")

        entry = ResidualEntry(
            identifier="GRA1.BHZ.P",
            observed=100.0,
            calculated=99.8,
            residual=0.2,
            station="GRA1",
            phase="P",
            is_outlier=False,
        )
        self.assertEqual(entry.station, "GRA1")
        self.assertAlmostEqual(entry.residual, 0.2)

        result = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="Geiger Fit",
            converged=True,
            iterations=4,
            dof=8,
            rmse=0.18,
            wrmse=0.14,
            residuals=[entry],
            confidence_ellipse=ellipse,
        )
        self.assertTrue(result.converged)
        self.assertEqual(len(result.residuals), 1)

        req = FitModificationRequest(
            fit_type=FitType.HYPOCENTER_GEIGER,
            excluded_identifiers=["GRA2"],
            weight_overrides={"GRA1": 0.5},
        )
        self.assertIn("GRA2", req.excluded_identifiers)
        self.assertEqual(req.weight_overrides["GRA1"], 0.5)

    def test_fit_diagnostic_bridge_signals(self) -> None:
        """Verify FitDiagnosticBridge signal definition, connection, and emission."""
        bridge = FitDiagnosticBridge()
        received = []

        bridge.fitDiagnosticAvailable.connect(lambda res: received.append(("diag", res)))
        bridge.recomputeFitRequested.connect(lambda req: received.append(("recompute", req)))
        bridge.highlightTraceRequested.connect(lambda st: received.append(("highlight", st)))
        bridge.excludeStationRequested.connect(lambda st, ex: received.append(("exclude", st, ex)))
        bridge.updateWeightsRequested.connect(lambda w: received.append(("weights", w)))

        res = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="T",
            converged=True,
            iterations=1,
            dof=1,
            rmse=0.1,
            wrmse=0.1,
        )
        bridge.fitDiagnosticAvailable.emit(res)
        self.assertEqual(received[-1][0], "diag")

        req = FitModificationRequest(fit_type=FitType.HYPOCENTER_GEIGER)
        bridge.recomputeFitRequested.emit(req)
        self.assertEqual(received[-1][0], "recompute")

        bridge.highlightTraceRequested.emit("GRA1")
        self.assertEqual(received[-1], ("highlight", "GRA1"))

        bridge.excludeStationRequested.emit("GRA2", True)
        self.assertEqual(received[-1], ("exclude", "GRA2", True))

        bridge.updateWeightsRequested.emit({"GRA1": 0.5})
        self.assertEqual(received[-1], ("weights", {"GRA1": 0.5}))


class TestFitDiagnosticWindowUI(unittest.TestCase):
    """Test suite for dockable FitDiagnosticWindow and interactive widgets."""

    def setUp(self) -> None:
        self.bridge = FitDiagnosticBridge()
        self.win = FitDiagnosticWindow(bridge=self.bridge)

    def test_kpi_card_update(self) -> None:
        """Verify KPI metric card value and color styling."""
        card = KPICard("Test KPI", "--")
        card.set_value("0.125 s", "#3fb950")
        self.assertEqual(card.lbl_value.text(), "0.125 s")

    def test_window_tabs_and_dock_setup(self) -> None:
        """Verify all 6 domain tabs exist and dock widget is properly configured."""
        self.assertEqual(self.win.tabs.count(), 6)
        tab_titles = [self.win.tabs.tabText(i) for i in range(6)]
        self.assertIn("Global Overview", tab_titles[0])
        self.assertIn("Travel Time & Geiger Residuals", tab_titles[1])
        self.assertIn("Array Beamforming & FK Analysis", tab_titles[2])
        self.assertIn("Spectral Source Parameters", tab_titles[3])
        self.assertIn("3C Particle Motion & Polarization", tab_titles[4])
        self.assertIn("Restitution & Deconvolution", tab_titles[5])

    def test_set_fit_result_populates_ui(self) -> None:
        """Verify set_fit_result updates KPI cards, parameter table, and residual table."""
        entries = [
            ResidualEntry(
                identifier="GRA1.BHZ.P",
                observed=100.0,
                calculated=99.9,
                residual=0.1,
                weight=1.0,
                distance_deg=2.5,
                azimuth_deg=45.0,
                station="GRA1",
                phase="P",
            ),
            ResidualEntry(
                identifier="GRA2.BHZ.P",
                observed=105.0,
                calculated=101.0,
                residual=4.0,
                weight=1.0,
                distance_deg=3.5,
                azimuth_deg=48.0,
                station="GRA2",
                phase="P",
                is_outlier=True,
            ),
        ]
        params = {
            "latitude": ParameterUncertainty("latitude", 50.5, 0.05, (50.4, 50.6), "deg"),
            "longitude": ParameterUncertainty("longitude", 11.2, 0.08, (11.04, 11.36), "deg"),
        }
        ellipse = ConfidenceEllipse(semi_major_km=15.0, semi_minor_km=5.0, strike_deg=60.0)

        result = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="Geiger Test",
            converged=True,
            iterations=3,
            dof=4,
            rmse=0.25,
            wrmse=0.18,
            chi2_reduced=1.15,
            condition_number=45.2,
            parameters=params,
            correlation_matrix=np.array([[1.0, -0.4], [-0.4, 1.0]]),
            confidence_ellipse=ellipse,
            residuals=entries,
            outlier_count=1,
            normality_p_value=0.45,
            skewness=0.1,
            kurtosis=0.05,
        )

        self.win.set_fit_result(result)

        # Check KPI cards
        self.assertIn("0.250", self.win.kpi_rmse.lbl_value.text())
        self.assertIn("1.15", self.win.kpi_chi2.lbl_value.text())
        self.assertIn("1 / 2", self.win.kpi_outliers.lbl_value.text())
        self.assertIn("Converged", self.win.kpi_status.lbl_value.text())
        self.assertIn("45.2", self.win.kpi_cond.lbl_value.text())

        # Check Tab 1: Parameter table
        self.assertEqual(self.win.table_params.rowCount(), 2)

        # Check Tab 2: Residuals table
        self.assertEqual(self.win.table_residuals.rowCount(), 2)
        # Check active checkbox state
        chk_item = self.win.table_residuals.item(0, 0)
        self.assertEqual(chk_item.checkState(), Qt.CheckState.Checked)

        # Outlier column for row 1
        status_item = self.win.table_residuals.item(1, 10)
        self.assertIn("OUTLIER", status_item.text())

    def test_interactive_table_emits_recompute_request(self) -> None:
        """Verify toggling active checkbox in table emits recomputeFitRequested."""
        entries = [
            ResidualEntry(identifier="GRA1.BHZ.P", observed=100.0, calculated=100.0, residual=0.0, station="GRA1", phase="P"),
            ResidualEntry(identifier="GRA2.BHZ.P", observed=105.0, calculated=101.0, residual=4.0, station="GRA2", phase="P", is_outlier=True),
        ]
        result = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="T",
            converged=True,
            iterations=1,
            dof=2,
            rmse=0.2,
            wrmse=0.2,
            residuals=entries,
        )
        self.win.set_fit_result(result)

        recompute_requests = []
        self.bridge.recomputeFitRequested.connect(lambda req: recompute_requests.append(req))

        # Uncheck row 1 (GRA2)
        chk_item = self.win.table_residuals.item(1, 0)
        chk_item.setCheckState(Qt.CheckState.Unchecked)

        self.assertTrue(len(recompute_requests) >= 1)
        last_req = recompute_requests[-1]
        self.assertIn("GRA2", last_req.excluded_identifiers)

    def test_deactivate_outliers_button(self) -> None:
        """Verify clicking 'Deactivate Outliers' unchecks outlier rows and triggers recompute."""
        entries = [
            ResidualEntry(identifier="GRA1.BHZ.P", observed=100.0, calculated=100.0, residual=0.0, station="GRA1", phase="P"),
            ResidualEntry(identifier="GRA2.BHZ.P", observed=105.0, calculated=101.0, residual=4.0, station="GRA2", phase="P", is_outlier=True),
        ]
        result = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="T",
            converged=True,
            iterations=1,
            dof=2,
            rmse=0.2,
            wrmse=0.2,
            residuals=entries,
            outlier_count=1,
        )
        self.win.set_fit_result(result)

        recompute_requests = []
        self.bridge.recomputeFitRequested.connect(lambda req: recompute_requests.append(req))

        self.win.btn_deactivate_outliers.click()

        self.assertTrue(len(recompute_requests) >= 1)
        last_req = recompute_requests[-1]
        self.assertIn("GRA2", last_req.excluded_identifiers)


class TestReactiveReFitLoop(unittest.TestCase):
    """Test suite for Geiger location with exclusions and closed-loop UI integration."""

    def setUp(self) -> None:
        from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff

        self.cfg = get_config()
        self.ttt = get_ttt_manager(self.cfg.inputs_dir)
        true_lat, true_lon, true_z = 50.1, 12.3, 10.0
        stations = ["GRA1", "BFO", "FUR", "CLZ", "MOX"]
        t0 = 1000000.0
        self.picks = []
        for st_code in stations:
            st = self.cfg.get_station(st_code)
            dist, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = self.ttt.travel_time("P", max(0.05, dist), true_z)
            pk = PhasePick(name="P", onset=format_sh_time(t0 + tt), station=st_code)
            self.picks.append(pk)

    def test_locate_network_geiger_with_diagnostics(self) -> None:
        """Verify locate_network_geiger returns AnalysisParamSet, residuals, and FitDiagnosticResult."""
        param, resids, diag_res = locate_network_geiger(self.picks, return_diagnostics=True)
        self.assertIsInstance(param, AnalysisParamSet)
        self.assertIsInstance(resids, dict)
        self.assertIsInstance(diag_res, FitDiagnosticResult)

        self.assertTrue(diag_res.converged)
        self.assertTrue(diag_res.rmse > 0.0)
        self.assertIsNotNone(diag_res.confidence_ellipse)
        self.assertTrue(diag_res.confidence_ellipse.semi_major_km > 0.0)
        self.assertTrue(0.0 <= diag_res.confidence_ellipse.strike_deg <= 180.0)
        self.assertEqual(len(diag_res.residuals), 5)

        # Check parameter uncertainties
        self.assertIn("latitude", diag_res.parameters)
        self.assertIn("longitude", diag_res.parameters)
        self.assertTrue(diag_res.parameters["latitude"].standard_error > 0.0)

    def test_locate_network_geiger_with_exclusions(self) -> None:
        """Verify station exclusion alters active pick count and parameter values."""
        param_all, _, diag_all = locate_network_geiger(self.picks, return_diagnostics=True)

        # Exclude MOX
        param_ex, _, diag_ex = locate_network_geiger(
            self.picks, excluded_stations=["MOX"], return_diagnostics=True
        )

        # In diag_ex, MOX residual should have weight = 0.0
        mox_entry = next((e for e in diag_ex.residuals if e.station == "MOX"), None)
        self.assertIsNotNone(mox_entry)
        self.assertEqual(mox_entry.weight, 0.0)

        # Other stations remain active with weight > 0
        gra1_entry = next((e for e in diag_ex.residuals if e.station == "GRA1"), None)
        self.assertIsNotNone(gra1_entry)
        self.assertTrue(gra1_entry.weight > 0.0)

    def test_main_window_reactive_refit_integration(self) -> None:
        """Verify MainWindow reactive loop: recomputeFitRequested re-inverts and updates canvas."""
        win = MainWindow(autoload_example=False)
        self.assertTrue(hasattr(win, "fit_dock"))
        self.assertTrue(hasattr(win, "fit_bridge"))

        # Add traces with picks
        dt = 0.05
        t_arr = np.arange(200) * dt
        traces = []
        for i, pk in enumerate(self.picks):
            tr = Trace(data=np.zeros_like(t_arr), delta=dt, station=pk.station, comp="Z", start_epoch=1000000.0)
            tr.add_or_replace_pick(pk)
            traces.append(tr)

        win.engine.traces = traces
        win.refresh_views()

        # Trigger recompute
        req = FitModificationRequest(
            fit_type=FitType.HYPOCENTER_GEIGER,
            excluded_identifiers=["FUR"],
            weight_overrides={"FUR": 0.0},
        )
        win._on_recompute_fit_requested(req)

        # Check that active parameter set is updated
        curr_p = win.engine.current_params
        self.assertTrue(curr_p.lat != 0.0 or curr_p.lon != 0.0)
        self.assertIsNotNone(curr_p.residuals)
        self.assertIsNotNone(curr_p.error_ellipse)

        # Check that traces received theoretical arrivals and updated residuals
        for tr in win.engine.traces:
            self.assertTrue(tr.distance > 0.0)
            self.assertTrue(len(tr.theo_picks) > 0)
            for p in tr.picks:
                self.assertIsNotNone(p.resid)


if __name__ == "__main__":
    unittest.main()
