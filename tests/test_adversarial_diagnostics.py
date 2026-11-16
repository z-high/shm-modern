"""Adversarial stress tests for Milestone 2 seismology diagnostics engine.

Empirical verification of:
1. Collinear, rank-deficient, zero, and underdetermined Jacobians J (SVD regularization & condition number).
2. Degenerate residuals (empty arrays, zero-variance, identical values, negative residuals, massive outliers).
3. Mathematical accuracy of 95% confidence error ellipse (a, b, theta) against analytical 2x2 covariance matrices.
"""

import math
import unittest
import numpy as np

import shm_app.seismology.diagnostics as diag
from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick, StationInfo
from shm_app.core.time_utils import format_sh_time
from shm_app.seismology.array_loc import locate_network_geiger
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


class TestAdversarialJacobiansAndSVD(unittest.TestCase):
    """Stress tests for Jacobian condition numbers and SVD covariance regularization."""

    def test_zero_jacobian(self) -> None:
        """Verify handling of completely zero Jacobian matrix."""
        J_zero = np.zeros((10, 4))
        cov_zero = diag.compute_covariance_matrix(J_zero, damping=1e-5)
        self.assertEqual(cov_zero.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(cov_zero)))
        # Covariance diagonal with zero J and damping lambda is 1/lambda
        np.testing.assert_allclose(np.diag(cov_zero), 1e5, rtol=1e-3)

        kappa = diag.compute_condition_number(J_zero)
        self.assertEqual(kappa, 1.0)

    def test_collinear_jacobian(self) -> None:
        """Verify condition number and SVD pseudo-inverse on rank-deficient collinear Jacobian."""
        J_collinear = np.ones((12, 4))
        J_collinear[:, 1] = J_collinear[:, 0] * 2.0
        J_collinear[:, 2] = J_collinear[:, 0] * -3.5
        J_collinear[:, 3] = J_collinear[:, 0] + J_collinear[:, 1]

        kappa = diag.compute_condition_number(J_collinear)
        self.assertTrue(np.isinf(kappa) or kappa > 1e12)

        # Inversion with damping > 0
        cov_damped = diag.compute_covariance_matrix(J_collinear, damping=1e-4)
        self.assertTrue(np.all(np.isfinite(cov_damped)))
        self.assertEqual(cov_damped.shape, (4, 4))

        # Inversion with damping = 0.0 (pure SVD pseudo-inverse thresholding)
        cov_nodamp = diag.compute_covariance_matrix(J_collinear, damping=0.0)
        self.assertTrue(np.all(np.isfinite(cov_nodamp)))
        self.assertEqual(cov_nodamp.shape, (4, 4))

    def test_underdetermined_jacobian(self) -> None:
        """Verify behavior when observations N < parameters P."""
        J_under = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 1.0, 3.0],
        ])
        cov_under = diag.compute_covariance_matrix(J_under, damping=0.0)
        self.assertTrue(np.all(np.isfinite(cov_under)))
        self.assertEqual(cov_under.shape, (4, 4))

    def test_degenerate_dimensions(self) -> None:
        """Verify empty and degenerate dimension matrices."""
        self.assertEqual(diag.compute_condition_number(np.zeros((0, 0))), 1.0)
        self.assertEqual(diag.compute_condition_number(np.zeros((5, 0))), 1.0)
        self.assertEqual(diag.compute_covariance_matrix(np.zeros((0, 4))).shape, (4, 4))
        self.assertEqual(diag.compute_covariance_matrix(np.zeros((4, 0))).shape, (0, 0))


class TestAdversarialResiduals(unittest.TestCase):
    """Stress tests for residual edge cases, zero-variance, outliers, and weights."""

    def test_empty_and_single_element_residuals(self) -> None:
        """Verify behavior on empty arrays and single observations."""
        self.assertEqual(diag.compute_rmse([]), 0.0)
        self.assertEqual(diag.compute_wrmse([], []), 0.0)
        self.assertEqual(diag.compute_mad([]), 0.0)
        self.assertEqual(diag.compute_robust_sigma([]), 0.0)
        self.assertEqual(diag.compute_reduced_chi2([]), 0.0)
        self.assertEqual(len(diag.compute_huber_weights([])), 0)
        self.assertEqual(len(diag.compute_tukey_weights([])), 0)
        self.assertEqual(len(diag.detect_outliers_3sigma([])), 0)

        # Single element
        res_single = [5.0]
        self.assertEqual(diag.compute_rmse(res_single), 5.0)
        self.assertEqual(diag.compute_mad(res_single), 0.0)
        self.assertEqual(diag.compute_robust_sigma(res_single), 0.0)
        self.assertEqual(diag.compute_huber_weights(res_single)[0], 1.345 / 5.0)

    def test_zero_variance_residuals(self) -> None:
        """Verify all identical residuals (zero variance)."""
        res_identical = np.full(20, 10.0)
        self.assertAlmostEqual(diag.compute_rmse(res_identical), 10.0)
        self.assertEqual(diag.compute_mad(res_identical), 0.0)
        self.assertEqual(diag.compute_robust_sigma(res_identical), 0.0)
        # Huber weights should fall back safely without division by zero
        w_huber = diag.compute_huber_weights(res_identical)
        self.assertTrue(np.all(np.isfinite(w_huber)))
        self.assertTrue(np.all(w_huber > 0.0))

        # Detect outliers should find 0 outliers
        outliers = diag.detect_outliers_3sigma(res_identical)
        self.assertFalse(np.any(outliers))

        # Normality stats should return clean fallback
        stats = diag.compute_normality_stats(res_identical)
        self.assertEqual(stats["skewness"], 0.0)
        self.assertEqual(stats["kurtosis"], 0.0)
        self.assertEqual(stats["p_value"], 1.0)

    def test_massive_outliers(self) -> None:
        """Verify robust isolation of extreme unassociated pick outliers (e.g. 1e6 seconds)."""
        res_with_outlier = np.array([0.02, -0.01, 0.03, -0.02, 1e6])
        # MAD and robust sigma must remain completely unaffected by the 1e6 outlier
        mad = diag.compute_mad(res_with_outlier)
        self.assertAlmostEqual(mad, 0.03, places=2)
        sigma_hat = diag.compute_robust_sigma(res_with_outlier)
        self.assertTrue(sigma_hat < 0.1)

        # Huber weight for the outlier must be suppressed to ~ 1e-7
        w_huber = diag.compute_huber_weights(res_with_outlier)
        self.assertTrue(w_huber[-1] < 1e-6)
        self.assertTrue(w_huber[0] > 0.9)

        # Tukey weight for outlier must be exactly 0.0
        w_tukey = diag.compute_tukey_weights(res_with_outlier)
        self.assertEqual(w_tukey[-1], 0.0)
        self.assertTrue(w_tukey[0] > 0.9)

        # 3-sigma outlier mask must flag the outlier
        outlier_mask = diag.detect_outliers_3sigma(res_with_outlier)
        self.assertTrue(outlier_mask[-1])
        self.assertEqual(np.sum(outlier_mask), 1)

    def test_negative_residuals_and_weights(self) -> None:
        """Verify symmetric handling of negative residuals and weight bounds."""
        res_pos = np.array([1.0, 2.0, 3.0])
        res_neg = -res_pos
        self.assertAlmostEqual(diag.compute_rmse(res_pos), diag.compute_rmse(res_neg))
        self.assertAlmostEqual(diag.compute_mad(res_pos), diag.compute_mad(res_neg))

        # Huber and Tukey weights must be identical for positive and negative residuals
        np.testing.assert_allclose(
            diag.compute_huber_weights(res_pos), diag.compute_huber_weights(res_neg)
        )
        np.testing.assert_allclose(
            diag.compute_tukey_weights(res_pos), diag.compute_tukey_weights(res_neg)
        )


class TestMathematicalConfidenceEllipse(unittest.TestCase):
    """Rigorous analytical validation of horizontal 95% confidence ellipse geometry."""

    def test_analytical_covariance_matrices(self) -> None:
        """Verify semi-major a, semi-minor b, and strike azimuth theta across all orientations."""
        scale_95 = math.sqrt(-2.0 * math.log(0.05))  # Exact Chi2(2) scale ~ 2.44774677
        test_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0]
        test_ratios = [
            (10.0, 2.0),
            (5.0, 5.0),
            (25.0, 1.0),
            (100.0, 0.01),
            (8.0, 0.0),
        ]

        for sigma1, sigma2 in test_ratios:
            for target_theta in test_angles:
                theta_rad = math.radians(target_theta)
                # u1 points along target_theta clockwise from North (+y) towards East (+x)
                u1 = np.array([math.cos(theta_rad), math.sin(theta_rad)])
                u2 = np.array([-math.sin(theta_rad), math.cos(theta_rad)])

                # Analytical covariance matrix C = sigma1^2 * u1 u1^T + sigma2^2 * u2 u2^T
                C = (sigma1 ** 2) * np.outer(u1, u1) + (sigma2 ** 2) * np.outer(u2, u2)

                a, b, theta = diag.compute_confidence_ellipse_2d(C, confidence=0.95, deg_to_km=1.0)
                expected_a = round(scale_95 * sigma1, 3)
                expected_b = round(scale_95 * sigma2, 3)

                self.assertAlmostEqual(a, expected_a, places=2)
                self.assertAlmostEqual(b, expected_b, places=2)

                if sigma1 > sigma2 + 1e-4:
                    diff_theta = min(
                        abs(theta - target_theta),
                        abs(theta - target_theta - 180.0),
                        abs(theta - target_theta + 180.0),
                    )
                    self.assertLess(diff_theta, 0.2)


if __name__ == "__main__":
    unittest.main()
