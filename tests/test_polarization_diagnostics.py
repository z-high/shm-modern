"""Unit tests for M-16 3C polarization trajectory misfit, confidence cones, and FitDiagnosticResult."""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import unittest
import numpy as np

from shm_app.core.fit_models import FitDiagnosticResult, FitType
from shm_app.core.models import Trace
from shm_app.processing.polarization import (
    analyze_polarization_diagnostic,
    compute_mdir,
)
from shm_app.processing.trace_ops import create_synthetic_trace


def _make_linear_3c_traces(
    baz_deg: float,
    inc_deg: float,
    length_sec: float = 20.0,
    dt: float = 0.05,
    pos_sec: float = 8.0,
    ampl: float = 100.0,
) -> tuple[Trace, Trace, Trace]:
    wave = create_synthetic_trace(
        "KUEPPER", length_sec=length_sec, dt=dt, pos_sec=pos_sec, ampl=ampl, width=1.5
    ).data
    az_r = math.radians(baz_deg)
    inc_r = math.radians(inc_deg)
    z_dat = wave * math.cos(inc_r)
    n_dat = -wave * math.sin(inc_r) * math.cos(az_r)
    e_dat = -wave * math.sin(inc_r) * math.sin(az_r)

    tz = Trace(data=z_dat, delta=dt, station="GRA1", comp="Z")
    tn = Trace(data=n_dat, delta=dt, station="GRA1", comp="N")
    te = Trace(data=e_dat, delta=dt, station="GRA1", comp="E")
    return tz, tn, te


class TestPolarizationDiagnostics(unittest.TestCase):
    def test_linear_3c_wave_near_zero_misfit_and_cones(self) -> None:
        baz_true = 132.0
        inc_true = 34.0
        tz, tn, te = _make_linear_3c_traces(baz_true, inc_true)

        res = compute_mdir(tz, tn, te, 7.0, 10.0)

        # Preserve all legacy keys
        for key in (
            "azimuth",
            "incidence",
            "rectilinearity",
            "planarity",
            "lambda1",
            "lambda2",
            "lambda3",
            "rms_pol",
            "rms_pol_norm",
            "sigma_theta_deg",
            "sigma_phi_deg",
        ):
            self.assertIn(key, res)

        self.assertAlmostEqual(res["azimuth"], baz_true, delta=0.2)
        self.assertAlmostEqual(res["incidence"], inc_true, delta=0.2)
        self.assertGreater(res["rectilinearity"], 0.999)
        self.assertLess(res["rms_pol"], 1e-8)
        self.assertLess(res["rms_pol_norm"], 1e-10)
        self.assertLess(res["sigma_theta_deg"], 1e-4)
        self.assertLess(res["sigma_phi_deg"], 1e-4)

    def test_elliptical_motion_misfit_and_jurkevics_cones(self) -> None:
        baz_true = 45.0
        inc_true = 25.0
        dt = 0.05
        t = np.arange(400) * dt
        # Major axis signal + smaller orthogonal transverse/radial phase-shifted motion
        main_sig = 100.0 * np.sin(2.0 * math.pi * 1.0 * t)
        orth_q = 20.0 * np.cos(2.0 * math.pi * 1.0 * t)
        orth_t = 5.0 * np.cos(2.0 * math.pi * 2.0 * t)

        az_r = math.radians(baz_true)
        inc_r = math.radians(inc_true)
        # L, Q, T -> Z, N, E (inverse of rotate_3d_lqt)
        z_dat = main_sig * math.cos(inc_r) + orth_q * math.sin(inc_r)
        n_dat = (
            -main_sig * math.sin(inc_r) * math.cos(az_r)
            + orth_q * math.cos(inc_r) * math.cos(az_r)
            + orth_t * math.sin(az_r)
        )
        e_dat = (
            -main_sig * math.sin(inc_r) * math.sin(az_r)
            + orth_q * math.cos(inc_r) * math.sin(az_r)
            - orth_t * math.cos(az_r)
        )

        tz = Trace(data=z_dat, delta=dt, station="BFO", comp="Z")
        tn = Trace(data=n_dat, delta=dt, station="BFO", comp="N")
        te = Trace(data=e_dat, delta=dt, station="BFO", comp="E")

        res = compute_mdir(tz, tn, te, 2.0, 12.0)

        # Mathematical identity: rms_pol == sqrt(lambda2 + lambda3)
        expected_rms = math.sqrt(res["lambda2"] + res["lambda3"])
        self.assertAlmostEqual(res["rms_pol"], expected_rms, places=6)
        self.assertGreater(res["rms_pol"], 10.0)
        self.assertTrue(0.0 < res["rms_pol_norm"] < 1.0)

        # Jurkevics cone definitions
        exp_theta = math.degrees(math.atan(math.sqrt(res["lambda2"] / res["lambda1"])))
        exp_phi = math.degrees(math.atan(math.sqrt(res["lambda3"] / res["lambda1"])))
        self.assertAlmostEqual(res["sigma_theta_deg"], exp_theta, places=6)
        self.assertAlmostEqual(res["sigma_phi_deg"], exp_phi, places=6)
        self.assertGreater(res["sigma_theta_deg"], res["sigma_phi_deg"])
        self.assertGreater(res["sigma_phi_deg"], 1.0)

    def test_analyze_polarization_diagnostic_contract(self) -> None:
        tz, tn, te = _make_linear_3c_traces(240.0, 40.0)
        diag = analyze_polarization_diagnostic(tz, tn, te, 7.0, 10.0)

        self.assertIsInstance(diag, FitDiagnosticResult)
        self.assertEqual(diag.fit_type, FitType.POLARIZATION_3C)
        self.assertTrue(diag.converged)

        expected_curves = {"z", "n", "e", "l", "q", "t", "time_rel"}
        self.assertTrue(expected_curves.issubset(set(diag.overlay_curves.keys())))
        lengths = {diag.overlay_curves[k].size for k in expected_curves}
        self.assertEqual(len(lengths), 1)
        self.assertGreater(next(iter(lengths)), 10)

        expected_ctx = {
            "baz",
            "incidence",
            "rectilinearity",
            "planarity",
            "sigma_theta_deg",
            "sigma_phi_deg",
            "rms_pol",
            "lambda1",
            "lambda2",
            "lambda3",
        }
        self.assertTrue(expected_ctx.issubset(set(diag.context_data.keys())))
        self.assertAlmostEqual(diag.context_data["baz"], 240.0, delta=0.2)
        self.assertAlmostEqual(diag.context_data["incidence"], 40.0, delta=0.2)
        self.assertLess(diag.context_data["rms_pol"], 1e-8)

        # Q and T rotated curves carry orthogonal trajectory energy
        q = diag.overlay_curves["q"]
        t = diag.overlay_curves["t"]
        rms_qt = float(np.sqrt(np.mean(q ** 2 + t ** 2)))
        self.assertAlmostEqual(rms_qt, diag.context_data["rms_pol"], places=6)

        self.assertIn("baz", diag.parameters)
        self.assertIn("incidence", diag.parameters)
        self.assertGreater(len(diag.residuals), 0)

    def test_zero_and_empty_traces_safe(self) -> None:
        tz0 = Trace(data=np.zeros(50), delta=0.05, station="ZERO", comp="Z")
        tn0 = Trace(data=np.zeros(50), delta=0.05, station="ZERO", comp="N")
        te0 = Trace(data=np.zeros(50), delta=0.05, station="ZERO", comp="E")
        res0 = compute_mdir(tz0, tn0, te0)
        self.assertEqual(res0["rms_pol"], 0.0)
        self.assertEqual(res0["sigma_theta_deg"], 0.0)
        self.assertEqual(res0["sigma_phi_deg"], 0.0)

        tz_e = Trace(data=np.array([]), delta=0.05, station="EMP", comp="Z")
        tn_e = Trace(data=np.array([]), delta=0.05, station="EMP", comp="N")
        te_e = Trace(data=np.array([]), delta=0.05, station="EMP", comp="E")
        diag_e = analyze_polarization_diagnostic(tz_e, tn_e, te_e)
        self.assertFalse(diag_e.converged)
        self.assertEqual(diag_e.fit_type, FitType.POLARIZATION_3C)

    def test_no_pyqt_scipy_obspy_imported(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pol_path = os.path.join(repo_root, "shm_app", "processing", "polarization.py")
        with open(pol_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=pol_path)

        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

        forbidden = ("PyQt6", "scipy", "obspy")
        self.assertTrue(
            imported_roots.isdisjoint(forbidden),
            f"Forbidden imports in polarization.py AST: {imported_roots.intersection(forbidden)}",
        )

        # Also verify in a fresh Python interpreter that importing polarization pulls none of them in
        probe = (
            "import sys; "
            "import shm_app.processing.polarization; "
            "bad = [m for m in ('PyQt6', 'scipy', 'obspy') if m in sys.modules]; "
            "assert not bad, f'Forbidden modules loaded: {bad}'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Subprocess import isolation check failed: {proc.stderr or proc.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
