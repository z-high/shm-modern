"""Tests for M-11: Empirical slowness vector calibration (calib_grf.dat, calib_lookup.txt,
delta format, 2D Cartesian IDW interpolation, 0/360 wrap-around, and array location integration).
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from typing import List

import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.processing.trace_ops import create_synthetic_trace
from shm_app.seismology.array_loc import (
    SlownessCalibrationTable,
    beam_delays_sec,
    compute_fk_analysis,
    load_calibration_for_array,
    locate_by_slowness_azimuth,
    plane_wave_locate,
)
from shm_app.seismology.traveltime import get_ttt_manager


class TestSlownessCalibration(unittest.TestCase):
    def test_parse_real_calib_grf_and_lookup(self) -> None:
        cfg = get_config()
        calib_path = os.path.join(cfg.inputs_dir, "calib_grf.dat")
        if not os.path.isfile(calib_path):
            self.skipTest(f"Legacy calibration file not present at {calib_path}")

        tbl = SlownessCalibrationTable.from_file(calib_path)
        self.assertGreater(len(tbl), 600)

        # First entry in calib_grf.dat:
        # 12.12.81 MALDIVE ISL.R.   7.8  118.3   6.4  112.0 ...
        first = tbl.entries[0]
        self.assertAlmostEqual(first.s_meas, 7.8, places=3)
        self.assertAlmostEqual(first.az_meas, 118.3, places=3)
        self.assertAlmostEqual(first.s_corr, 6.4, places=3)
        self.assertAlmostEqual(first.az_corr, 112.0, places=3)

        # Querying near an exact entry reproduces the table's calibrated vector
        s_cal, az_cal, ds, daz = tbl.calibrate(7.8, 118.3)
        self.assertAlmostEqual(s_cal, 6.4, delta=0.1)
        self.assertAlmostEqual(az_cal, 112.0, delta=0.5)
        self.assertLess(ds, 0.0)
        self.assertLess(daz, 0.0)

        # calib_lookup.txt resolution for GRF stations and array name
        tbl_sta = load_calibration_for_array("GRA1", config=cfg)
        self.assertIsNotNone(tbl_sta)
        self.assertEqual(len(tbl_sta), len(tbl))

        tbl_arr = load_calibration_for_array(["GRA1", "GRB2", "GRC1"], config=cfg)
        self.assertIsNotNone(tbl_arr)

        tbl_grf = load_calibration_for_array("GRF", config=cfg)
        self.assertIsNotNone(tbl_grf)

        # Unknown station returns None gracefully
        self.assertIsNone(load_calibration_for_array("NONEXISTENT_XYZ", config=cfg))

    def test_delta_format_and_wraparound_interpolation(self) -> None:
        delta_text = """
        # slowness_in  azimuth_in  d_slowness  d_azimuth
        4.50           359.0       -0.20       +2.4
        4.50             1.0       -0.20       +2.4
        5.50           359.0       -0.40       +2.4
        5.50             1.0       -0.40       +2.4
        """
        tbl = SlownessCalibrationTable.from_text(delta_text)
        self.assertEqual(len(tbl), 4)

        # Query right across the 0/360 boundary at az=0.0 deg, slowness=4.5 s/deg
        s_cal, az_cal, d_slo, d_az = tbl.calibrate(4.50, 0.0)
        self.assertAlmostEqual(s_cal, 4.30, delta=0.02)
        self.assertAlmostEqual(az_cal, 2.4, delta=0.15)
        self.assertAlmostEqual(d_slo, -0.20, delta=0.02)
        self.assertAlmostEqual(d_az, 2.4, delta=0.15)

        # Query at midpoint slowness=5.0, az=0.0 -> interpolated d_slowness ~ -0.30
        s_mid, az_mid, d_slo_mid, d_az_mid = tbl.calibrate(5.00, 360.0)
        self.assertAlmostEqual(s_mid, 4.70, delta=0.05)
        self.assertAlmostEqual(az_mid, 2.4, delta=0.25)
        self.assertAlmostEqual(d_slo_mid, -0.30, delta=0.05)
        self.assertAlmostEqual(d_az_mid, 2.4, delta=0.25)

        # Negative azimuth wrap-around (1.0 - 3.0 -> 358.0 deg, d_azim = -3.0)
        wrap_neg = SlownessCalibrationTable.from_text(
            "# slowness_in azimuth_in d_slowness d_azimuth\n"
            "6.0 1.0 +0.10 -3.0\n"
        )
        s_w, az_w, ds_w, daz_w = wrap_neg.calibrate(6.0, 1.0)
        self.assertAlmostEqual(s_w, 6.10, places=5)
        self.assertAlmostEqual(az_w, 358.0, places=4)
        self.assertAlmostEqual(ds_w, 0.10, places=5)
        self.assertAlmostEqual(daz_w, -3.0, places=4)

    def test_missing_lookup_and_empty_table_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # No calib_lookup.txt
            self.assertIsNone(load_calibration_for_array("GRA1", inputs_dir=tmpdir))

            # calib_lookup.txt pointing to non-existent file
            lookup = os.path.join(tmpdir, "calib_lookup.txt")
            with open(lookup, "w", encoding="utf-8") as fh:
                fh.write("! comment\nGRA1,GRA2 missing_calib.dat\n")
            self.assertIsNone(load_calibration_for_array("GRA1", inputs_dir=tmpdir))

        # Empty table acts as identity
        empty_tbl = SlownessCalibrationTable.from_text("# empty\n")
        self.assertEqual(len(empty_tbl), 0)
        s_out, az_out, ds, daz = empty_tbl.calibrate(6.5, 355.0)
        self.assertAlmostEqual(s_out, 6.5)
        self.assertAlmostEqual(az_out, 355.0)
        self.assertAlmostEqual(ds, 0.0)
        self.assertAlmostEqual(daz, 0.0)

    def test_integration_in_plane_wave_fk_and_locate(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        stations = ["GRA1", "GRA2", "GRA3", "GRA4", "GRB1", "GRB2", "GRB3", "GRB4", "GRB5"]
        coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
        x_km = [c[0] for c in coords]
        y_km = [c[1] for c in coords]

        meas_baz = 120.0
        meas_slo = 7.0
        delays = beam_delays_sec(x_km, y_km, meas_baz, meas_slo)
        base_ep = parse_sh_time("02-NOV-2003_02:25:00.000")
        arr_times = [base_ep + 15.0 - float(d) for d in delays]

        calib_tbl = SlownessCalibrationTable.from_text(
            "# slowness_in azimuth_in d_slowness d_azimuth\n"
            "7.0 120.0 -0.50 +6.0\n"
        )

        # 1. plane_wave_locate default (uncalibrated) vs calibrated
        pw_raw = plane_wave_locate(arr_times, x_km, y_km)
        self.assertFalse(pw_raw["calib_applied"])
        self.assertAlmostEqual(pw_raw["azimuth"], meas_baz, delta=0.2)
        self.assertAlmostEqual(pw_raw["slowness"], meas_slo, delta=0.1)

        pw_disabled = plane_wave_locate(
            arr_times, x_km, y_km, calib_table=calib_tbl, apply_calibration=False
        )
        self.assertFalse(pw_disabled["calib_applied"])
        self.assertAlmostEqual(pw_disabled["slowness"], pw_raw["slowness"], places=6)

        pw_cal = plane_wave_locate(
            arr_times, x_km, y_km, calib_table=calib_tbl, apply_calibration=True
        )
        self.assertTrue(pw_cal["calib_applied"])
        self.assertAlmostEqual(pw_cal["raw_azimuth"], meas_baz, delta=0.2)
        self.assertAlmostEqual(pw_cal["raw_slowness"], meas_slo, delta=0.1)
        self.assertAlmostEqual(pw_cal["cal_azimuth"], 126.0, delta=0.3)
        self.assertAlmostEqual(pw_cal["cal_slowness"], 6.50, delta=0.15)
        self.assertAlmostEqual(pw_cal["azimuth"], pw_cal["cal_azimuth"], places=6)
        self.assertAlmostEqual(pw_cal["slowness"], pw_cal["cal_slowness"], places=6)

        # 2. compute_fk_analysis default vs calibrated
        traces: List[Trace] = []
        for st_code, d_sec in zip(stations, delays):
            tr = create_synthetic_trace(
                "KUEPPER", length_sec=40.0, dt=0.05, pos_sec=15.0 - float(d_sec), ampl=200.0, width=1.0
            )
            tr.station = st_code
            tr.start_epoch = base_ep
            traces.append(tr)

        fk_raw = compute_fk_analysis(
            traces, t_lo=10.0, t_hi=22.0, f_lo=0.4, f_hi=2.5, max_slowness=12.0, grid_points=31, config=cfg
        )
        self.assertFalse(fk_raw["calib_applied"])

        fk_cal = compute_fk_analysis(
            traces,
            t_lo=10.0,
            t_hi=22.0,
            f_lo=0.4,
            f_hi=2.5,
            max_slowness=12.0,
            grid_points=31,
            config=cfg,
            calib_table=calib_tbl,
            apply_calibration=True,
        )
        self.assertTrue(fk_cal["calib_applied"])
        self.assertAlmostEqual(fk_cal["raw_azimuth"], fk_raw["azimuth"], places=6)
        self.assertAlmostEqual(fk_cal["raw_slowness"], fk_raw["slowness"], places=6)
        self.assertAlmostEqual(fk_cal["cal_slowness"], fk_raw["slowness"] - 0.50, delta=0.2)
        self.assertAlmostEqual(fk_cal["cal_azimuth"], fk_raw["azimuth"] + 6.0, delta=0.5)
        self.assertAlmostEqual(
            fk_cal["cal_sx"],
            fk_cal["cal_slowness"] * math.sin(math.radians(fk_cal["cal_azimuth"])),
            places=6,
        )

        # 3. locate_by_slowness_azimuth default vs calibrated
        loc_raw = locate_by_slowness_azimuth(
            "GRA1", "P", base_ep + 600.0, meas_slo, meas_baz, depth_km=10.0, config=cfg, ttt=ttt
        )
        self.assertIsNone(loc_raw.residuals)

        loc_cal = locate_by_slowness_azimuth(
            "GRA1",
            "P",
            base_ep + 600.0,
            meas_slo,
            meas_baz,
            depth_km=10.0,
            config=cfg,
            ttt=ttt,
            calib_table=calib_tbl,
            apply_calibration=True,
        )
        self.assertAlmostEqual(loc_cal.slowness, 6.50, places=4)
        self.assertAlmostEqual(loc_cal.azimuth, 126.0, places=4)
        self.assertNotAlmostEqual(loc_cal.lat, loc_raw.lat, places=2)
        self.assertNotAlmostEqual(loc_cal.lon, loc_raw.lon, places=2)
        self.assertIsNotNone(loc_cal.residuals)
        self.assertAlmostEqual(loc_cal.residuals["raw_slowness"], meas_slo, places=4)
        self.assertAlmostEqual(loc_cal.residuals["cal_slowness"], 6.50, places=4)


if __name__ == "__main__":
    unittest.main()
