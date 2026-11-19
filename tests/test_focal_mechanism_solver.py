"""Unit tests for M-25 double-couple focal mechanism solver and Schmidt stereonet helpers."""

from __future__ import annotations

import math
import unittest
import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick, Trace
from shm_app.seismology.focal_mechanism import (
    auxiliary_nodal_plane,
    compute_beachball_contours,
    extract_polarities_from_traces,
    fault_normal_and_slip,
    p_wave_radiation_pattern,
    principal_axes,
    project_schmidt,
    solve_focal_mechanism,
)
from shm_app.seismology.traveltime import sphere_diff


def _moment_tensor(strike: float, dip: float, rake: float) -> np.ndarray:
    n, d = fault_normal_and_slip(strike, dip, rake)
    m = np.outer(n, d) + np.outer(d, n)
    return m / np.linalg.norm(m)


def _generate_synthetic_polarities(
    strike: float,
    dip: float,
    rake: float,
    azimuths: np.ndarray,
    takeoffs: np.ndarray,
    min_abs_rad: float = 0.08,
) -> list[tuple[float, float, int]]:
    obs: list[tuple[float, float, int]] = []
    for az in azimuths:
        for ih in takeoffs:
            rad = float(p_wave_radiation_pattern(strike, dip, rake, float(ih), float(az)))
            if abs(rad) >= min_abs_rad:
                obs.append((float(az), float(ih), 1 if rad > 0.0 else -1))
    return obs


class TestFocalMechanismSolver(unittest.TestCase):
    def test_radiation_pattern_and_auxiliary_plane_symmetry(self) -> None:
        # Pure thrust (strike=0, dip=45, rake=90): straight down (ih=0) is compressional (+1)
        r_down = p_wave_radiation_pattern(0.0, 45.0, 90.0, 0.0, 0.0)
        self.assertAlmostEqual(r_down, 1.0, places=6)

        # Pure normal (strike=0, dip=45, rake=-90): straight down (ih=0) is dilatational (-1)
        r_norm = p_wave_radiation_pattern(0.0, 45.0, -90.0, 0.0, 0.0)
        self.assertAlmostEqual(r_norm, -1.0, places=6)

        # Pure vertical strike-slip (strike=0, dip=90, rake=0): straight down is nodal (0)
        r_ss_down = p_wave_radiation_pattern(0.0, 90.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(r_ss_down, 0.0, places=9)
        self.assertAlmostEqual(p_wave_radiation_pattern(0.0, 90.0, 0.0, 90.0, 45.0), 1.0, places=6)
        self.assertAlmostEqual(p_wave_radiation_pattern(0.0, 90.0, 0.0, 90.0, 135.0), -1.0, places=6)

        # Auxiliary plane yields identical radiation pattern everywhere
        s2, d2, r2 = auxiliary_nodal_plane(35.0, 58.0, -125.0)
        az_grid = np.linspace(0.0, 345.0, 24)
        ih_grid = np.linspace(10.0, 85.0, 8)
        az_m, ih_m = np.meshgrid(az_grid, ih_grid)
        rad1 = p_wave_radiation_pattern(35.0, 58.0, -125.0, ih_m, az_m)
        rad2 = p_wave_radiation_pattern(s2, d2, r2, ih_m, az_m)
        np.testing.assert_allclose(rad1, rad2, atol=1e-9)

    def test_solve_pure_strike_slip_normal_and_thrust(self) -> None:
        azimuths = np.arange(0.0, 360.0, 20.0)
        takeoffs = np.array([15.0, 30.0, 50.0, 70.0, 85.0])

        cases = [
            ("strike_slip", 30.0, 90.0, 0.0),
            ("normal", 115.0, 45.0, -90.0),
            ("thrust", 210.0, 45.0, 90.0),
            ("oblique", 65.0, 55.0, -135.0),
        ]
        for name, true_s, true_d, true_r in cases:
            with self.subTest(mechanism=name):
                obs = _generate_synthetic_polarities(true_s, true_d, true_r, azimuths, takeoffs)
                self.assertGreaterEqual(len(obs), 30)

                sol = solve_focal_mechanism(obs, grid_step_deg=10.0, refine=True)
                self.assertTrue(sol["converged"])
                self.assertEqual(sol["n_misfit"], 0)
                self.assertAlmostEqual(sol["misfit_ratio"], 0.0, places=9)

                m_true = _moment_tensor(true_s, true_d, true_r)
                m_est = _moment_tensor(sol["strike"], sol["dip"], sol["rake"])
                frob_err = float(np.linalg.norm(m_true - m_est))
                self.assertLess(frob_err, 0.25, msg=f"{name}: sol={sol}")

                # Also verify auxiliary plane has identical moment tensor
                m_aux = _moment_tensor(sol["strike2"], sol["dip2"], sol["rake2"])
                np.testing.assert_allclose(m_est, m_aux, atol=1e-9)

    def test_solve_with_noisy_outlier_polarity(self) -> None:
        azimuths = np.arange(0.0, 360.0, 30.0)
        takeoffs = np.array([25.0, 55.0, 75.0])
        obs = _generate_synthetic_polarities(0.0, 45.0, 90.0, azimuths, takeoffs, min_abs_rad=0.15)
        # Flip one polarity
        az0, ih0, p0 = obs[0]
        obs[0] = (az0, ih0, -p0)

        sol = solve_focal_mechanism(obs, grid_step_deg=10.0, refine=True)
        self.assertEqual(sol["n_misfit"], 1)
        self.assertAlmostEqual(sol["misfit_ratio"], 1.0 / len(obs), places=6)

        m_true = _moment_tensor(0.0, 45.0, 90.0)
        m_est = _moment_tensor(sol["strike"], sol["dip"], sol["rake"])
        self.assertLess(float(np.linalg.norm(m_true - m_est)), 0.35)

    def test_schmidt_projection_bounds_and_beachball_contours(self) -> None:
        # Bounds check across all trends and plunges (including negative upper-hemisphere plunges)
        for tr in np.linspace(0.0, 360.0, 37):
            for pl in np.linspace(-90.0, 90.0, 19):
                x, y = project_schmidt(float(tr), float(pl))
                self.assertLessEqual(x * x + y * y, 1.0 + 1e-12)

        x0, y0 = project_schmidt(42.0, 90.0)
        self.assertAlmostEqual(x0, 0.0, places=12)
        self.assertAlmostEqual(y0, 0.0, places=12)

        xh, yh = project_schmidt(90.0, 0.0)
        self.assertAlmostEqual(xh, 1.0, places=9)
        self.assertAlmostEqual(yh, 0.0, places=9)

        bb = compute_beachball_contours(40.0, 60.0, -90.0, n_grid=101)
        self.assertEqual(bb["sign_grid"].shape, (101, 101))
        inside = bb["mask"]
        self.assertTrue(np.all(np.isnan(bb["sign_grid"][~inside])))
        valid_signs = bb["sign_grid"][inside]
        self.assertTrue(np.any(valid_signs > 0.0))
        self.assertTrue(np.any(valid_signs < 0.0))

        # Nodal plane curves lie within unit disk
        self.assertTrue(np.all(bb["plane1_x"] ** 2 + bb["plane1_y"] ** 2 <= 1.0 + 1e-10))
        self.assertTrue(np.all(bb["plane2_x"] ** 2 + bb["plane2_y"] ** 2 <= 1.0 + 1e-10))

        # Principal axes P, T, B orthogonality check via thrust/normal plunge expectations
        axes_thrust = principal_axes(0.0, 45.0, 90.0)
        self.assertAlmostEqual(axes_thrust["T"][1], 90.0, places=6)
        self.assertAlmostEqual(axes_thrust["P"][1], 0.0, places=6)

    def test_extract_polarities_from_traces_end_to_end(self) -> None:
        cfg = get_config()
        ev_lat, ev_lon, ev_depth = 49.5, 11.5, 15.0
        true_s, true_d, true_r = 25.0, 50.0, 85.0

        # Create synthetic station network surrounding the epicenter at ~1-15 deg distance
        stations = []
        for idx in range(18):
            az_deg = idx * 20.0
            dist_deg = 2.0 + (idx % 4) * 3.5
            # Rough spherical offset for test stations
            st_lat = ev_lat + dist_deg * math.cos(math.radians(az_deg))
            st_lon = ev_lon + dist_deg * math.sin(math.radians(az_deg)) / math.cos(math.radians(ev_lat))
            stations.append((f"TST{idx:02d}", st_lat, st_lon))

        traces = []
        for code, slat, slon in stations:
            tr = Trace(data=np.zeros(20), delta=0.05, station=code, comp="Z")
            tr.extra_headers["station_lat"] = slat
            tr.extra_headers["station_lon"] = slon
            d_deg, az, _ = sphere_diff(ev_lat, ev_lon, slat, slon)
            # Use takeoff from focal_mechanism helper via extract after assigning sign
            # Compute expected takeoff ~ 55 deg for local/regional or extract directly
            from shm_app.seismology.focal_mechanism import estimate_takeoff_angle

            ih = estimate_takeoff_angle(d_deg, depth_km=ev_depth, phase="P")
            rad = float(p_wave_radiation_pattern(true_s, true_d, true_r, ih, az))
            if abs(rad) < 0.08:
                continue
            tr.add_or_replace_pick(PhasePick(name="P", sign=(1 if rad > 0 else -1), weight=3))
            traces.append(tr)

        obs = extract_polarities_from_traces(traces, ev_lat, ev_lon, depth_km=ev_depth, config=cfg)
        self.assertGreaterEqual(len(obs), 12)
        sol = solve_focal_mechanism(obs, grid_step_deg=10.0, refine=True)
        self.assertEqual(sol["n_misfit"], 0)
        self.assertAlmostEqual(sol["misfit_ratio"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
