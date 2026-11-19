"""Unit tests for M-13/M-24 model fields and M-23 workspace session autosave & crash recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from shm_app.core.models import (
    QUALITY_SIGMA_SEC,
    AnalysisParamSet,
    PhasePick,
    Trace,
    quality_from_sigma_t,
)
from shm_app.core.session import (
    DEFAULT_SESSION_FILENAME,
    SessionLoadError,
    SessionManager,
    deserialize_workspace,
    has_unclean_session,
    serialize_workspace,
)
from shm_app.core.time_utils import format_sh_time


class TestModelContractsM13M24(unittest.TestCase):
    """Tests for PhasePick.sigma_t / quality_from_sigma_t and Trace overlay filter fields."""

    def test_phase_pick_sigma_t_table_and_clamping(self) -> None:
        self.assertEqual(
            PhasePick.QUALITY_SIGMA_SEC,
            [0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 4.0, 8.0, 15.0],
        )
        self.assertEqual(QUALITY_SIGMA_SEC, PhasePick.QUALITY_SIGMA_SEC)

        for q, expected in enumerate(PhasePick.QUALITY_SIGMA_SEC):
            pk = PhasePick(name="P", quality=q)
            self.assertAlmostEqual(pk.sigma_t(), expected, places=6)

        # Clamping out-of-range quality values
        self.assertAlmostEqual(PhasePick(quality=-5).sigma_t(), 0.02, places=6)
        self.assertAlmostEqual(PhasePick(quality=10).sigma_t(), 15.0, places=6)
        self.assertAlmostEqual(PhasePick(quality=99).sigma_t(), 15.0, places=6)

    def test_quality_from_sigma_t_inverse_mapping(self) -> None:
        for q, sigma in enumerate(PhasePick.QUALITY_SIGMA_SEC):
            self.assertEqual(quality_from_sigma_t(sigma), q)
            self.assertEqual(PhasePick.quality_from_sigma_t(sigma), q)

        self.assertEqual(quality_from_sigma_t(0.01), 0)
        self.assertEqual(quality_from_sigma_t(-1.0), 0)
        self.assertEqual(quality_from_sigma_t(0.025), 0)
        self.assertEqual(quality_from_sigma_t(0.045), 1)
        self.assertEqual(quality_from_sigma_t(100.0), 9)
        self.assertEqual(quality_from_sigma_t(float("inf")), 9)
        self.assertEqual(quality_from_sigma_t(float("nan")), 4)

    def test_trace_overlay_filter_fields_and_copy(self) -> None:
        tr = Trace(
            data=np.arange(10, dtype=np.float64),
            station="GRA1",
            filter_name="WWSSN_SP",
            overlay_filter_name="BP_1_5HZ",
            overlay_filter_color="red",
            filter_history=["NONE", "WWSSN_SP", "BP_1_5HZ"],
        )
        copied = tr.copy()
        self.assertEqual(copied.overlay_filter_name, "BP_1_5HZ")
        self.assertEqual(copied.overlay_filter_color, "red")
        self.assertEqual(copied.filter_history, ["NONE", "WWSSN_SP", "BP_1_5HZ"])

        # Mutating copy's filter_history must not mutate original
        copied.filter_history.append("LP_2HZ")
        copied.overlay_filter_name = "LP_2HZ"
        self.assertEqual(tr.filter_history, ["NONE", "WWSSN_SP", "BP_1_5HZ"])
        self.assertEqual(tr.overlay_filter_name, "BP_1_5HZ")


class TestSessionCrashRecovery(unittest.TestCase):
    """Tests for M-23 workspace serialization, atomic writes, and crash recovery."""

    def _build_sample_workspace(self, mseed_path: str = ""):
        t0 = 1700000000.0
        pk1 = PhasePick(
            name="P",
            onset=format_sh_time(t0 + 12.345),
            spec="i",
            sign=1,
            source="m",
            weight=3,
            quality=3,
            filter_name="WWSSN_SP",
            ampl=125.4,
            ampl_time=14.2,
            period=0.85,
            slowness=6.2,
            d_slowness=0.3,
            beam_azim=42.5,
            d_beam_azim=1.8,
            l_slowness=6.0,
            l_azim=44.1,
            mag_mb=5.4,
            snr=18.2,
            resid=-0.14,
            station="GRA1",
            comp="Z",
            flags="l",
        )
        pk2 = PhasePick(
            name="S",
            onset=format_sh_time(t0 + 28.9),
            spec="e",
            sign=-1,
            source="a",
            weight=2,
            quality=5,
            resid=0.31,
            station="GRB1",
            comp="N",
        )
        tr1 = Trace(
            data=np.linspace(-1.0, 1.0, 200),
            delta=0.05,
            station="GRA1",
            comp="Z",
            chan1="B",
            chan2="H",
            network="GR",
            start_epoch=t0,
            display_offset=1.5,
            distance=14.2,
            azimuth=42.5,
            lat=49.6919,
            lon=11.2217,
            filter_name="WWSSN_SP",
            overlay_filter_name="BP_1_5HZ",
            overlay_filter_color="red",
            filter_history=["WWSSN_SP", "BP_1_5HZ"],
            selected=True,
            picks=[pk1],
            theo_picks={"P": t0 + 12.5, "pP": t0 + 22.1},
            extra_headers={"custom_tag": "array_ref"},
            source_path="",
        )
        tr2 = Trace(
            data=np.sin(np.linspace(0.0, 4.0 * np.pi, 100)),
            delta=0.05,
            station="GRB1",
            comp="N",
            start_epoch=t0 + 2.0,
            filter_name="NONE",
            overlay_filter_name="HP_05HZ",
            overlay_filter_color="blue",
            filter_history=["HP_05HZ"],
            selected=False,
            picks=[pk2],
            source_path=mseed_path,
        )
        params = AnalysisParamSet(
            distance=14.2,
            azimuth=42.5,
            slowness=6.2,
            depth=18.5,
            d_depth=2.3,
            depth_type=3,
            origin=format_sh_time(t0 - 190.0),
            lat=38.42,
            d_lat=0.05,
            lon=21.75,
            d_lon=0.07,
            phase="P",
            mag_ml=5.1,
            mag_mb=5.4,
            mag_ms=5.0,
            mag_mw=5.3,
            event_id=20261118,
            event_type="r",
            region_id=364,
            region_name="GREECE",
            residuals={"GRA1.P": -0.14, "GRB1.S": 0.31},
            error_ellipse=(8.4, 4.1, 35.0),
            condition_number=4.7,
            chi2_reduced=1.18,
            rmse=0.225,
            wrmse=0.198,
        )
        display = {
            "view_t_lo": 5.0,
            "view_t_hi": 55.0,
            "global_zoom": 2.5,
            "norm_mode": "AW",
            "mag_trace_idx": 0,
            "mag_t_lo": 10.0,
            "mag_t_hi": 20.0,
            "global_filter_name": "WWSSN_SP",
        }
        return [tr1, tr2], [params], display

    def test_roundtrip_serialization_and_disk_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_file = Path(tmpdir) / "grb1.mseed"
            ext_file.write_bytes(b"DUMMY_EXTERNAL_SEED")

            traces, param_sets, display = self._build_sample_workspace(mseed_path=str(ext_file))
            mgr = SessionManager(tmpdir)
            self.assertEqual(mgr.session_path, Path(tmpdir) / DEFAULT_SESSION_FILENAME)

            saved_path = mgr.save(
                traces=traces,
                param_sets=param_sets,
                active_set_index=0,
                selected_trace_indices=[0],
                active_trace_index=0,
                display_settings=display,
                clean_exit=False,
            )
            self.assertTrue(os.path.isfile(saved_path))
            self.assertTrue(has_unclean_session(saved_path))
            self.assertTrue(mgr.has_unclean_session())

            # Load back (provide trace_loader for external file trace)
            def _loader(entry):
                if entry.get("source_path") == str(ext_file):
                    return np.sin(np.linspace(0.0, 4.0 * np.pi, 100))
                return None

            restored = mgr.load(trace_loader=_loader)
            self.assertTrue(restored["ok"])
            self.assertIsNone(restored["error"])
            self.assertFalse(restored["clean_exit"])
            self.assertEqual(restored["active_trace_index"], 0)
            self.assertEqual(restored["selected_trace_indices"], [0])

            r_traces = restored["traces"]
            self.assertEqual(len(r_traces), 2)
            # Inline waveform restored for tr1 (no source_path)
            np.testing.assert_allclose(r_traces[0].data, traces[0].data)
            # External waveform restored via trace_loader for tr2
            np.testing.assert_allclose(r_traces[1].data, traces[1].data)
            self.assertEqual(r_traces[1].source_path, str(ext_file))

            # Filter state & history preserved
            self.assertEqual(r_traces[0].filter_name, "WWSSN_SP")
            self.assertEqual(r_traces[0].overlay_filter_name, "BP_1_5HZ")
            self.assertEqual(r_traces[0].overlay_filter_color, "red")
            self.assertEqual(r_traces[0].filter_history, ["WWSSN_SP", "BP_1_5HZ"])
            self.assertEqual(r_traces[1].overlay_filter_name, "HP_05HZ")
            self.assertEqual(r_traces[1].overlay_filter_color, "blue")

            # Picks & theoretical picks preserved
            self.assertEqual(len(r_traces[0].picks), 1)
            rpk = r_traces[0].picks[0]
            self.assertEqual(rpk.name, "P")
            self.assertEqual(rpk.sign, 1)
            self.assertEqual(rpk.quality, 3)
            self.assertEqual(rpk.weight, 3)
            self.assertAlmostEqual(rpk.resid, -0.14)
            self.assertAlmostEqual(rpk.ampl, 125.4)
            self.assertAlmostEqual(rpk.l_slowness, 6.0)
            self.assertAlmostEqual(rpk.l_azim, 44.1)
            self.assertAlmostEqual(r_traces[0].theo_picks["pP"], traces[0].theo_picks["pP"])

            # AnalysisParamSet preserved
            rp = restored["params"]
            self.assertAlmostEqual(rp.lat, 38.42)
            self.assertAlmostEqual(rp.lon, 21.75)
            self.assertAlmostEqual(rp.depth, 18.5)
            self.assertAlmostEqual(rp.slowness, 6.2)
            self.assertAlmostEqual(rp.azimuth, 42.5)
            self.assertAlmostEqual(rp.mag_mw, 5.3)
            self.assertEqual(rp.error_ellipse, (8.4, 4.1, 35.0))
            self.assertAlmostEqual(rp.rmse, 0.225)
            self.assertAlmostEqual(rp.chi2_reduced, 1.18)
            self.assertEqual(rp.residuals, {"GRA1.P": -0.14, "GRB1.S": 0.31})

            # Display settings preserved
            rdisp = restored["display_settings"]
            self.assertEqual(rdisp["view_t_lo"], 5.0)
            self.assertEqual(rdisp["view_t_hi"], 55.0)
            self.assertEqual(rdisp["global_zoom"], 2.5)
            self.assertEqual(rdisp["norm_mode"], "AW")
            self.assertEqual(rdisp["global_filter_name"], "WWSSN_SP")

    def test_clean_shutdown_marker_and_unclean_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / ".shm_session.json"
            self.assertFalse(has_unclean_session(session_file))

            traces, param_sets, display = self._build_sample_workspace()
            mgr = SessionManager(session_file)
            mgr.save(traces, param_sets, display_settings=display, clean_exit=False)
            self.assertTrue(has_unclean_session(session_file))

            self.assertTrue(mgr.mark_clean_shutdown())
            self.assertFalse(has_unclean_session(session_file))
            self.assertFalse(mgr.has_unclean_session())

            loaded = mgr.load()
            self.assertTrue(loaded["ok"])
            self.assertTrue(loaded["clean_exit"])

    def test_atomic_write_leaves_no_temp_files_and_protects_prior_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / ".shm_session.json"
            traces, param_sets, display = self._build_sample_workspace()
            mgr = SessionManager(session_file)
            mgr.save(traces, param_sets, display_settings=display, clean_exit=False)
            original_bytes = session_file.read_bytes()

            # Simulate crash during os.replace -> previous valid file stays intact & temp file cleaned up
            with mock.patch("shm_app.core.session.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    mgr.save(traces, param_sets, display_settings={"global_zoom": 9.9})

            self.assertEqual(session_file.read_bytes(), original_bytes)
            leftover_tmps = [p for p in os.listdir(tmpdir) if p.endswith(".tmp")]
            self.assertEqual(leftover_tmps, [])

    def test_corrupt_and_truncated_json_resilience(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / ".shm_session.json"
            mgr = SessionManager(session_file)

            # 1. Missing file
            missing_res = mgr.load()
            self.assertFalse(missing_res["ok"])
            self.assertIn("not found", missing_res["error"].lower())

            # 2. Truncated JSON
            session_file.write_text('{"version": 1, "traces": [{"station": "GRA1"', encoding="utf-8")
            self.assertFalse(has_unclean_session(session_file))
            trunc_res = mgr.load()
            self.assertFalse(trunc_res["ok"])
            self.assertIsNotNone(trunc_res["error"])
            self.assertEqual(trunc_res["traces"], [])
            with self.assertRaises(SessionLoadError):
                mgr.load(raise_on_error=True)

            # 3. Non-dict JSON root
            session_file.write_text("[1, 2, 3]", encoding="utf-8")
            self.assertFalse(has_unclean_session(session_file))
            list_res = mgr.load()
            self.assertFalse(list_res["ok"])
            with self.assertRaises(SessionLoadError):
                mgr.load(raise_on_error=True)

            # 4. Binary garbage / invalid UTF-8
            session_file.write_bytes(b"\xff\xfe\x00\x80corrupt")
            self.assertFalse(has_unclean_session(session_file))
            bin_res = mgr.load()
            self.assertFalse(bin_res["ok"])

            # 5. Direct deserialize_workspace with NaN/Inf sanitized during serialize_workspace
            tr_nan = Trace(data=np.array([1.0, float("nan"), float("inf")]), station="NAN1")
            payload = serialize_workspace([tr_nan], display_settings={"zoom": float("nan")})
            dumped = json.dumps(payload, allow_nan=False)
            restored = deserialize_workspace(json.loads(dumped))
            self.assertEqual(len(restored["traces"]), 1)
            self.assertTrue(np.all(np.isfinite(restored["traces"][0].data)))


if __name__ == "__main__":
    unittest.main()
