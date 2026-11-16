"""Comprehensive Empirical Challenger Test Suite for Milestone 2 Fix.

Tests:
1. Condition number kappa(J) and SVD covariance inversion under nearly collinear / rank-deficient geometries:
   - Rank-deficient Jacobians (1D collinear stations, duplicate rows, zero matrix).
   - Numerical condition numbers across condition spectrum (1 to 1e16).
   - Regularization stability with damping=0.0 and damping>0.
   - Moore-Penrose pseudo-inverse identity checks.
2. Confidence ellipse parameters (a, b, theta) under isotropic and highly eccentric residual structures:
   - Isotropic circular structures across multiple scales.
   - Highly eccentric structures (ratio 1e-6 to 1e6).
   - Full 180-degree strike azimuth sweep (every 1 degree).
   - Zero-variance covariance matrix.
   - Non-positive-definite / perturbed numerical matrices with negative eigenvalues.
3. Huber and Tukey weights under zero MAD and extreme outlier residuals:
   - MAD = 0 with nonzero variance (majority identical, one outlier).
   - MAD = 0 and variance = 0 (all zeros).
   - Extreme astronomical outliers (1e12, 1e30).
   - Invariance to sign of residuals.
4. Parameter uncertainty degree units in array_loc.py:
   - Synthetic multi-station Geiger inversion.
   - Exact mathematical check that latitude/longitude standard errors are in degrees.
   - 95% confidence intervals consistency check in degrees.
   - AnalysisParamSet d_lat / d_lon in kilometers consistency check.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

import shm_app.seismology.diagnostics as diag
from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick
from shm_app.core.time_utils import format_sh_time
from shm_app.seismology.array_loc import DEG_TO_KM, locate_network_geiger
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


class TestChallengerConditionNumberAndSVD(unittest.TestCase):
    """Stress testing condition number computation and SVD covariance inversion."""

    def test_svd_collinear_geometric_singularity(self) -> None:
        """Adversarial challenge: 10 collinear stations where dt/dlat and dt/dlon are linearly dependent."""
        # Collinear direction: 45 degrees. dt/dlat = dt/dlon for all stations
        n_obs = 10
        J_collinear = np.zeros((n_obs, 3), dtype=np.float64)
        J_collinear[:, 0] = 1.0  # origin time column
        slope = 2.5
        for i in range(n_obs):
            dlat = 0.1 * (i + 1)
            dlon = slope * dlat
            J_collinear[i, 1] = dlat
            J_collinear[i, 2] = dlon

        # Condition number must be infinity or > 1e14
        kappa = diag.compute_condition_number(J_collinear)
        self.assertTrue(np.isinf(kappa) or kappa > 1e14, f"Expected near-infinite kappa, got {kappa}")

        # SVD inversion with damping=0.0 must NOT raise LinAlgError
        cov_zero_damp = diag.compute_covariance_matrix(J_collinear, damping=0.0)
        self.assertEqual(cov_zero_damp.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(cov_zero_damp)), "Covariance contains non-finite values!")

        # Verify symmetry of cov matrix
        np.testing.assert_allclose(cov_zero_damp, cov_zero_damp.T, atol=1e-12)

        # Diagonals must be non-negative
        self.assertTrue(np.all(np.diag(cov_zero_damp) >= 0.0), "Covariance diagonal contains negative variance!")

        # Moore-Penrose pseudo-inverse property: A @ C @ A = A * s^2
        A = J_collinear.T @ J_collinear
        s_sq = 1.0
        # A @ (inv_A) @ A == A
        A_invA_A = A @ cov_zero_damp @ A
        np.testing.assert_allclose(A_invA_A, A * s_sq, atol=1e-8, rtol=1e-8)

    def test_svd_progressive_ill_conditioning(self) -> None:
        """Adversarial challenge: sweep condition number from 1 to 1e16 via perturbation epsilon."""
        base_J = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 2.0],
        ], dtype=np.float64)

        for log_eps in range(1, 16):
            eps = 10.0 ** (-log_eps)
            J_perturbed = base_J.copy()
            # Make column 2 almost identical to column 1
            J_perturbed[:, 2] = J_perturbed[:, 1] + eps

            kappa = diag.compute_condition_number(J_perturbed)
            self.assertTrue(kappa > 1.0)
            if log_eps >= 14:
                self.assertTrue(np.isinf(kappa) or kappa > 1e12)

            cov = diag.compute_covariance_matrix(J_perturbed, damping=1e-6)
            self.assertTrue(np.all(np.isfinite(cov)))
            self.assertTrue(np.all(np.diag(cov) >= 0.0))

    def test_svd_all_zeros_and_underdetermined(self) -> None:
        """Adversarial challenge: zero matrix and underdetermined systems."""
        # Zero matrix
        J_zero = np.zeros((8, 4), dtype=np.float64)
        self.assertEqual(diag.compute_condition_number(J_zero), 1.0)
        cov_zero = diag.compute_covariance_matrix(J_zero, damping=1e-4, s_sq=2.0)
        # With zero J and damping lambda, (0 + lambda I)^-1 = (1/lambda) I
        expected_diag = 2.0 / 1e-4
        np.testing.assert_allclose(np.diag(cov_zero), expected_diag, rtol=1e-5)

        # Underdetermined (2 observations, 4 parameters)
        J_under = np.array([[1.0, 0.5, 0.2, 0.1], [1.0, -0.5, -0.2, -0.1]])
        cov_under = diag.compute_covariance_matrix(J_under, damping=0.0)
        self.assertEqual(cov_under.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(cov_under)))
        self.assertTrue(np.all(np.diag(cov_under) >= 0.0))


class TestChallengerConfidenceEllipse(unittest.TestCase):
    """Stress testing horizontal 95% confidence ellipse parameters under extreme geometries."""

    def test_isotropic_circular_covariance(self) -> None:
        """Adversarial challenge: Isotropic covariance across 8 orders of magnitude."""
        scale_95 = math.sqrt(-2.0 * math.log(0.05))  # ~ 2.44774677
        sigmas = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]

        for s in sigmas:
            C = np.array([[s ** 2, 0.0], [0.0, s ** 2]], dtype=np.float64)
            a, b, theta = diag.compute_confidence_ellipse_2d(C, confidence=0.95, deg_to_km=1.0)

            expected_len = round(scale_95 * s, 3)
            self.assertAlmostEqual(a, expected_len, places=2)
            self.assertAlmostEqual(b, expected_len, places=2)
            # For a circle, strike azimuth is arbitrary but must be a valid angle [0, 180)
            self.assertTrue(0.0 <= theta < 180.0)

    def test_highly_eccentric_full_strike_sweep(self) -> None:
        """Adversarial challenge: Highly eccentric ellipse (ratio 1000:1) swept across all 180 degrees in 1-deg steps."""
        scale_95 = math.sqrt(-2.0 * math.log(0.05))
        sigma_major = 50.0
        sigma_minor = 0.05  # 1000:1 ratio

        expected_a = round(scale_95 * sigma_major, 3)
        expected_b = round(scale_95 * sigma_minor, 3)

        for target_deg in range(0, 180):
            theta_rad = math.radians(target_deg)
            # Unit vector along major axis clockwise from North (+y) towards East (+x)
            # Note: index 0 is North (y), index 1 is East (x)
            u_major = np.array([math.cos(theta_rad), math.sin(theta_rad)])
            u_minor = np.array([-math.sin(theta_rad), math.cos(theta_rad)])

            C = (sigma_major ** 2) * np.outer(u_major, u_major) + (sigma_minor ** 2) * np.outer(u_minor, u_minor)

            a, b, theta = diag.compute_confidence_ellipse_2d(C, confidence=0.95, deg_to_km=1.0)

            self.assertAlmostEqual(a, expected_a, places=2)
            self.assertAlmostEqual(b, expected_b, places=2)

            # Check azimuth match modulo 180 degrees
            diff = min(
                abs(theta - target_deg),
                abs(theta - target_deg - 180.0),
                abs(theta - target_deg + 180.0),
            )
            self.assertLess(diff, 0.25, f"Azimuth mismatch for target {target_deg} deg: got {theta}")

    def test_degenerate_zero_and_near_negative_eigenvalues(self) -> None:
        """Adversarial challenge: Exact zero covariance and numerical matrices with tiny negative eigenvalues."""
        # Exact zero
        C_zero = np.zeros((2, 2))
        a, b, theta = diag.compute_confidence_ellipse_2d(C_zero)
        self.assertEqual(a, 0.0)
        self.assertEqual(b, 0.0)
        self.assertTrue(0.0 <= theta < 180.0)

        # Numerical artifact: negative eigenvalue -1e-15
        C_neg = np.array([[1.0, 0.0], [0.0, -1e-15]])
        a, b, theta = diag.compute_confidence_ellipse_2d(C_neg, deg_to_km=1.0)
        self.assertTrue(np.isfinite(a) and a > 0.0)
        self.assertEqual(b, 0.0)


class TestChallengerRobustWeights(unittest.TestCase):
    """Stress testing Huber and Tukey weights under zero MAD and extreme outliers."""

    def test_zero_mad_nonzero_variance_fallback(self) -> None:
        """Adversarial challenge: MAD == 0 because majority of residuals are identical, but one outlier exists."""
        # In this dataset, median is 0.0, 4 out of 5 residuals are 0.0 -> MAD is exactly 0.0!
        resids = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        mad = diag.compute_mad(resids)
        self.assertEqual(mad, 0.0)

        # Robust sigma is 0.0
        self.assertEqual(diag.compute_robust_sigma(resids), 0.0)

        # compute_huber_weights MUST fall back to std(resids) > 0 and NOT divide by zero
        w_huber = diag.compute_huber_weights(resids)
        self.assertEqual(len(w_huber), 5)
        self.assertTrue(np.all(np.isfinite(w_huber)))
        self.assertTrue(np.all(w_huber > 0.0))
        # Outlier (10.0) must be severely downweighted relative to normal picks
        self.assertTrue(w_huber[-1] < w_huber[0])

        # compute_tukey_weights must also fall back safely to std
        w_tukey = diag.compute_tukey_weights(resids)
        self.assertEqual(len(w_tukey), 5)
        self.assertTrue(np.all(np.isfinite(w_tukey)))
        self.assertTrue(np.all(w_tukey >= 0.0))

    def test_zero_mad_and_zero_variance(self) -> None:
        """Adversarial challenge: All residuals are identically zero."""
        resids = np.zeros(15, dtype=np.float64)
        self.assertEqual(diag.compute_mad(resids), 0.0)
        self.assertEqual(diag.compute_robust_sigma(resids), 0.0)

        w_huber = diag.compute_huber_weights(resids)
        np.testing.assert_allclose(w_huber, 1.0)

        w_tukey = diag.compute_tukey_weights(resids)
        np.testing.assert_allclose(w_tukey, 1.0)

    def test_astronomical_outlier_rejection(self) -> None:
        """Adversarial challenge: Outliers of scale 1e12 and 1e30 seconds."""
        resids = np.array([0.01, -0.01, 0.02, -0.02, 1e12, 1e30])

        # MAD must stay immune to the two astronomical outliers
        mad = diag.compute_mad(resids)
        self.assertAlmostEqual(mad, 0.03, places=3)

        # Huber weights for outliers must be practically zero without underflow exception
        w_huber = diag.compute_huber_weights(resids)
        self.assertTrue(w_huber[4] < 1e-10)
        self.assertTrue(w_huber[5] < 1e-25)
        self.assertTrue(w_huber[0] > 0.99)

        # Tukey weights for outliers must be strictly 0.0
        w_tukey = diag.compute_tukey_weights(resids)
        self.assertEqual(w_tukey[4], 0.0)
        self.assertEqual(w_tukey[5], 0.0)
        self.assertTrue(w_tukey[0] > 0.99)


class TestChallengerParameterUncertaintyUnits(unittest.TestCase):
    """Stress testing and mathematical verification of degree units in Geiger inversion."""

    def test_geiger_parameter_uncertainty_degree_units(self) -> None:
        """Empirical verification: ParameterUncertainty latitude/longitude are strictly in degrees."""
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)

        event_lat = 50.100
        event_lon = 11.500
        event_depth = 25.0
        event_t0 = 1700000000.0

        stations = ["GRA1", "GRA2", "GRA3", "GRB1", "GRB2", "GRC1", "FUR", "BFO", "CLZ", "MOX", "WET"]
        picks = []
        for st_name in stations:
            st = cfg.get_station(st_name)
            dist, _, _ = sphere_diff(st.lat, st.lon, event_lat, event_lon)
            tt = ttt.travel_time("P", max(0.05, dist), event_depth) or (dist * 111.195 / 6.0)
            pk = PhasePick(
                name="P",
                onset=format_sh_time(event_t0 + tt),
                station=st_name,
                comp="Z",
                weight=2,
            )
            picks.append(pk)

        # Run Geiger inversion
        param, _res, diag_res = locate_network_geiger(picks, free_depth=True, return_diagnostics=True)

        self.assertIsNotNone(diag_res)
        self.assertTrue(diag_res.converged)

        params = diag_res.parameters
        self.assertIn("latitude", params)
        self.assertIn("longitude", params)
        self.assertIn("origin_time", params)
        self.assertIn("depth", params)

        lat_unc = params["latitude"]
        lon_unc = params["longitude"]
        t0_unc = params["origin_time"]
        z_unc = params["depth"]

        # 1. Units check
        self.assertEqual(lat_unc.unit, "deg")
        self.assertEqual(lon_unc.unit, "deg")
        self.assertEqual(t0_unc.unit, "s")
        self.assertEqual(z_unc.unit, "km")

        # 2. Standard error degree scaling check:
        # Latitude standard error must be reasonable for regional network (< 0.5 degrees, NOT 50+ degrees!)
        self.assertLess(lat_unc.standard_error, 0.5, f"Latitude SE {lat_unc.standard_error} appears scaled in km!")
        self.assertLess(lon_unc.standard_error, 0.5, f"Longitude SE {lon_unc.standard_error} appears scaled in km!")

        # 3. Exact relationship with covariance matrix:
        # diag_res.covariance_matrix index 1 is lat (deg^2), index 2 is lon (deg^2)
        cov_m = diag_res.covariance_matrix
        expected_lat_se = float(np.sqrt(max(0.0, cov_m[1, 1])))
        expected_lon_se = float(np.sqrt(max(0.0, cov_m[2, 2])))
        self.assertAlmostEqual(lat_unc.standard_error, expected_lat_se, places=7)
        self.assertAlmostEqual(lon_unc.standard_error, expected_lon_se, places=7)

        # 4. Consistency with AnalysisParamSet d_lat / d_lon in km:
        # param.d_lat is in km = lat_se * DEG_TO_KM
        self.assertAlmostEqual(lat_unc.standard_error * DEG_TO_KM, param.d_lat, places=2)
        self.assertAlmostEqual(lon_unc.standard_error * DEG_TO_KM, param.d_lon, places=2)

        # 5. 95% confidence interval sanity check
        self.assertAlmostEqual(
            lat_unc.confidence_interval_95[1] - lat_unc.confidence_interval_95[0],
            2.0 * 1.96 * lat_unc.standard_error,
            places=6,
        )
        self.assertAlmostEqual(
            (lat_unc.confidence_interval_95[0] + lat_unc.confidence_interval_95[1]) / 2.0,
            lat_unc.value,
            places=6,
        )

    def test_fixed_depth_inversion_units(self) -> None:
        """Empirical verification: ParameterUncertainty with fixed depth (free_depth=False)."""
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)

        picks = []
        for st_name in ["GRA1", "GRA2", "GRA3", "FUR", "BFO"]:
            st = cfg.get_station(st_name)
            dist, _, _ = sphere_diff(st.lat, st.lon, 48.0, 10.0)
            tt = ttt.travel_time("P", max(0.05, dist), 10.0) or (dist * 111.195 / 6.0)
            picks.append(
                PhasePick(
                    name="P",
                    onset=format_sh_time(1000000.0 + tt),
                    station=st_name,
                    comp="Z",
                    weight=1,
                )
            )

        param, _res, diag_res = locate_network_geiger(picks, free_depth=False, return_diagnostics=True)
        self.assertNotIn("depth", diag_res.parameters)
        self.assertEqual(param.depth_type, 1)

        lat_unc = diag_res.parameters["latitude"]
        self.assertEqual(lat_unc.unit, "deg")
        self.assertLess(lat_unc.standard_error, 0.5)
        self.assertAlmostEqual(lat_unc.standard_error * DEG_TO_KM, param.d_lat, places=2)


class TestChallengerAdvancedAdversarialEdgeCases(unittest.TestCase):
    """Extreme adversarial edge cases."""

    def test_azimuth_invariance_under_km_scaling(self) -> None:
        """Adversarial check: Strike azimuth theta MUST be invariant to deg_to_km scaling."""
        for angle in [10.0, 45.0, 90.0, 135.0, 170.0]:
            rad = math.radians(angle)
            u = np.array([math.cos(rad), math.sin(rad)])
            v = np.array([-math.sin(rad), math.cos(rad)])
            C = 10.0 * np.outer(u, u) + 0.1 * np.outer(v, v)

            _, _, theta_deg = diag.compute_confidence_ellipse_2d(C, deg_to_km=1.0)
            _, _, theta_km = diag.compute_confidence_ellipse_2d(C, deg_to_km=111.195)
            self.assertAlmostEqual(theta_deg, theta_km, places=1)

    def test_asymmetric_noise_floating_point_covariance(self) -> None:
        """Adversarial check: Symmetrizing handles asymmetric numerical float discrepancies."""
        C_asym = np.array([[4.0, 1.000000001], [0.999999999, 2.0]])
        a, b, theta = diag.compute_confidence_ellipse_2d(C_asym)
        self.assertTrue(np.isfinite(a) and np.isfinite(b) and np.isfinite(theta))
        self.assertTrue(a >= b)

    def test_huge_residual_array_robustness(self) -> None:
        """Adversarial check: 10,000 residuals with mixed outliers and noise."""
        np.random.seed(42)
        n = 10000
        # 95% Gaussian noise + 5% Cauchy extreme outliers
        normal_noise = np.random.normal(0.0, 0.1, int(n * 0.95))
        extreme_outliers = np.random.standard_cauchy(int(n * 0.05)) * 1000.0
        res = np.concatenate([normal_noise, extreme_outliers])

        mad = diag.compute_mad(res)
        # Normal std is 0.1, MAD for normal is 0.1 * 0.6745 ~ 0.067
        self.assertTrue(0.05 < mad < 0.10)

        w_huber = diag.compute_huber_weights(res)
        w_tukey = diag.compute_tukey_weights(res)
        self.assertTrue(np.all(np.isfinite(w_huber)))
        self.assertTrue(np.all(np.isfinite(w_tukey)))
        self.assertTrue(np.all(w_huber >= 0.0) and np.all(w_huber <= 1.0))
        self.assertTrue(np.all(w_tukey >= 0.0) and np.all(w_tukey <= 1.0))


if __name__ == "__main__":
    unittest.main()
