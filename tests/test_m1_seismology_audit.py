"""Comprehensive regression tests for Milestone 1 Core, Seismology & Station Config fixes."""

from __future__ import annotations

import logging
import math
import os
import tempfile
import unittest
import warnings
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import PhasePick, StationInfo, Trace
from shm_app.core.time_utils import (
    epoch_to_ntime,
    format_gse2_time,
    format_sh_time,
    parse_sh_time,
)
from shm_app.processing.trace_ops import create_synthetic_trace
from shm_app.seismology import diagnostics as diag
from shm_app.seismology.array_loc import (
    DEG_TO_KM,
    beam_delays_sec,
    compute_beam,
    compute_fk_analysis,
    compute_vespagram,
    correlate_traces,
    locate_network_geiger,
    plane_wave_locate,
)
from shm_app.seismology.focal_mechanism import (
    estimate_takeoff_angle,
    extract_polarities_from_traces,
)
from shm_app.seismology.traveltime import (
    TravelTimeManager,
    TravelTimeTable,
    get_ttt_manager,
    parse_ttt_file,
    sphere_diff,
)


class TestTimeUtilsAudit(unittest.TestCase):
    def test_parse_sh_time_invalid_month_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_sh_time("02-XYZ-2024_12:00:00.000")
        with self.assertRaises(ValueError):
            parse_sh_time("15-FOO-2023")
        with self.assertRaises(ValueError):
            parse_sh_time("nan")
        with self.assertRaises(ValueError):
            parse_sh_time("inf")

    def test_format_sh_time_guards_non_finite_epochs(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                format_sh_time(bad)
            with self.assertRaises(ValueError):
                format_gse2_time(bad)
            with self.assertRaises(ValueError):
                epoch_to_ntime(bad)


class TestConfigStationAudit(unittest.TestCase):
    def test_relative_xy_km_unknown_ref_station_no_5500km_shift(self) -> None:
        cfg = get_config()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            coords = cfg.relative_xy_km(["GRA1", "GRA2", "GRB1"], ref_station="UNKNOWN_REF")
        self.assertEqual(len(coords), 3)
        # All coordinates must remain local (< 100 km from centroid), never shifted by ~5,500 km
        for x_km, y_km, _, _ in coords:
            self.assertLess(abs(x_km), 100.0)
            self.assertLess(abs(y_km), 100.0)
        self.assertTrue(any("UNKNOWN_REF" in str(w.message) for w in ws))

    def test_relative_xy_km_arbitrary_array_origin_and_antimeridian_wrap(self) -> None:
        cfg = SHMConfig(shm_root=tempfile.mkdtemp())
        # Custom array without GRA1/BFO hardcoded names, with origin station CUST0 at (0, 0)
        cfg.stations["CUST0"] = StationInfo(code="CUST0", lat=40.0, lon=10.0, array_id=7, xrel=0.0, yrel=0.0)
        cfg.stations["CUST1"] = StationInfo(code="CUST1", lat=40.0, lon=10.1, array_id=7, xrel=8.5, yrel=0.0)
        cfg.stations["CUST2"] = StationInfo(code="CUST2", lat=40.1, lon=10.0, array_id=7, xrel=0.0, yrel=11.1)
        xy = cfg.relative_xy_km(["CUST0", "CUST1", "CUST2"], ref_station="CUST0")
        self.assertAlmostEqual(xy[0][0], 0.0, places=6)
        self.assertAlmostEqual(xy[1][0], 8.5, places=6)
        self.assertAlmostEqual(xy[2][1], 11.1, places=6)

        # Antimeridian spanning stations: 179.9E and -179.9E (180.1E) are 0.2 deg apart (~22 km)
        cfg.stations["AM_W"] = StationInfo(code="AM_W", lat=0.0, lon=179.9, array_id=0)
        cfg.stations["AM_E"] = StationInfo(code="AM_E", lat=0.0, lon=-179.9, array_id=0)
        xy_am = cfg.relative_xy_km(["AM_W", "AM_E"], ref_station="CENTRE")
        dx_km = xy_am[1][0] - xy_am[0][0]
        self.assertAlmostEqual(dx_km, 0.2 * 111.1949266, delta=0.5)

    def test_root_setter_updates_dirs_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            for d, sta_code, lat in ((d1, "TST1", 45.0), (d2, "TST2", 52.0)):
                os.makedirs(os.path.join(d, "inputs"), exist_ok=True)
                os.makedirs(os.path.join(d, "filter"), exist_ok=True)
                with open(os.path.join(d, "inputs", "STATLOC.DAT"), "w", encoding="utf-8") as fh:
                    fh.write(f"{sta_code} {lat:.4f} 10.0 0 0.0 0.0 !Test\n")

            cfg = SHMConfig(d1)
            self.assertIn("TST1", cfg.stations)
            self.assertNotIn("TST2", cfg.stations)

            cfg.root = d2
            self.assertEqual(cfg.inputs_dir, os.path.join(d2, "inputs"))
            self.assertEqual(cfg.filter_dir, os.path.join(d2, "filter"))
            self.assertIn("TST2", cfg.stations)
            self.assertNotIn("TST1", cfg.stations)


class TestDiagnosticsAudit(unittest.TestCase):
    def test_non_finite_filtering_in_stats_and_weights(self) -> None:
        r = [1.0, -1.0, float("nan"), float("inf"), 1.0, -1.0]
        w = [1.0, 1.0, 2.0, 1.0, float("nan"), 1.0]
        self.assertAlmostEqual(diag.compute_rmse(r), 1.0, places=9)
        self.assertAlmostEqual(diag.compute_wrmse(r, w), 1.0, places=9)
        self.assertAlmostEqual(diag.compute_mad([0.0, 0.0, 2.0, float("nan"), float("-inf")]), 0.0, places=9)
        self.assertAlmostEqual(diag.compute_reduced_chi2(r, dof=2), 2.0, places=9)

        hw = diag.compute_huber_weights(r)
        tw = diag.compute_tukey_weights(r)
        self.assertEqual(hw.size, len(r))
        self.assertEqual(tw.size, len(r))
        self.assertTrue(np.all(np.isfinite(hw)))
        self.assertTrue(np.all(np.isfinite(tw)))
        self.assertEqual(hw[2], 0.0)
        self.assertEqual(tw[3], 0.0)

        norm = diag.compute_normality_stats([-1.0, -0.5, 0.5, 1.0, float("nan"), float("inf")])
        self.assertTrue(math.isfinite(norm["p_value"]))
        self.assertTrue(math.isfinite(norm["skewness"]))

    def test_confidence_ellipse_respects_dof_vs_chi2_none(self) -> None:
        cov = np.diag([1.0, 1.0])
        a_chi2, b_chi2, _ = diag.compute_confidence_ellipse_2d(cov, dof=None, confidence=0.95, deg_to_km=1.0)
        self.assertAlmostEqual(a_chi2, math.sqrt(-2.0 * math.log(0.05)), places=3)
        self.assertAlmostEqual(a_chi2, b_chi2, places=3)

        a_dof3, _, _ = diag.compute_confidence_ellipse_2d(cov, dof=3, confidence=0.95, deg_to_km=1.0)
        expected_dof3 = math.sqrt(3.0 * ((0.05 ** (-2.0 / 3.0)) - 1.0))
        self.assertAlmostEqual(a_dof3, expected_dof3, places=3)
        self.assertGreater(a_dof3, a_chi2)


class TestTravelTimeAudit(unittest.TestCase):
    def test_parse_ttt_bounds_and_get_table_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_ndepths = os.path.join(tmp, "BAD1.TTT")
            with open(bad_ndepths, "w", encoding="utf-8") as fh:
                fh.write("TTT\n1.0 10.0 0\n0.0\n")
            with self.assertRaises(ValueError):
                parse_ttt_file(bad_ndepths)

            trunc = os.path.join(tmp, "BAD2.TTT")
            with open(trunc, "w", encoding="utf-8") as fh:
                fh.write("TTT\n1.0 10.0 3 0.0 33.0\n")
            with self.assertRaises(ValueError):
                parse_ttt_file(trunc)

            # TravelTimeManager.get_table warns on parse error instead of silent swallow
            p_path = os.path.join(tmp, "P.TTT")
            with open(p_path, "w", encoding="utf-8") as fh:
                fh.write("TTT\n1.0 10.0 0\n")
            tm = TravelTimeManager(tmp)
            with self.assertLogs("shm_app.seismology.traveltime", level=logging.WARNING), warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                self.assertIsNone(tm.get_table("P"))
            self.assertTrue(len(ws) >= 1)

    def test_teleseismic_p_missing_pdiff_does_not_fallback_to_crustal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p_path = os.path.join(tmp, "P.TTT")
            with open(p_path, "w", encoding="utf-8") as fh:
                fh.write("TTT\n1.0 98.0 1\n0.0\n1.0 15.0\n50.0 535.0\n98.0 818.0\n")
            tm = TravelTimeManager(tmp)
            self.assertAlmostEqual(tm.travel_time("P", 50.0, 0.0), 535.0, places=3)
            # > 98 deg without P+Pdiff.TTT or Pdiff.TTT must return None (never hypo_km / 5.85 ~ 1995 s)
            self.assertIsNone(tm.travel_time("P", 105.0, 33.0))

    def test_narrow_distance_from_slowness_and_north_pole_sphere_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p_path = os.path.join(tmp, "PNARROW.TTT")
            with open(p_path, "w", encoding="utf-8") as fh:
                fh.write("TTT\n10.0 10.6 1\n0.0\n10.0 140.0\n10.3 144.0\n10.6 148.0\n")
            tm = TravelTimeManager(tmp)
            d_est = tm.distance_from_slowness("PNARROW", 13.3, 0.0)
            self.assertIsNotNone(d_est)
            self.assertTrue(10.0 <= d_est <= 10.6)

        dist, az_12, az_21 = sphere_diff(90.0, 0.0, 45.0, 10.0)
        self.assertAlmostEqual(dist, 45.0, places=5)
        self.assertAlmostEqual(az_12, 180.0, places=5)
        self.assertAlmostEqual(az_21, 0.0, places=5)


class TestFocalMechanismAudit(unittest.TestCase):
    def test_shallow_local_pg_sg_takeoff_is_upgoing(self) -> None:
        tm = get_ttt_manager()
        for ph in ("Pg", "Sg"):
            for z in (5.0, 25.0):
                ih = estimate_takeoff_angle(0.1, depth_km=z, phase=ph, ttt=tm)
                self.assertGreater(ih, 90.0, f"{ph} at dist=0.1, depth={z} should be upgoing (>90), got {ih}")

    def test_extract_polarities_warns_and_skips_unresolved_zero_coords(self) -> None:
        cfg = get_config()
        tr_bad = Trace(data=np.zeros(20), delta=0.05, station="UNRES_STA", comp="Z")
        tr_bad.add_or_replace_pick(PhasePick(name="P", sign=1, weight=3, station="UNRES_STA"))

        tr_ok = Trace(data=np.zeros(20), delta=0.05, station="GRA1", comp="Z")
        tr_ok.add_or_replace_pick(PhasePick(name="P", sign=-1, weight=3, station="GRA1"))

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            obs = extract_polarities_from_traces([tr_bad, tr_ok], 48.0, 12.0, depth_km=10.0, config=cfg)

        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["station"], "GRA1")
        self.assertTrue(any("UNRES_STA" in str(w.message) for w in ws))


class TestArrayLocAndGeigerAudit(unittest.TestCase):
    def test_geiger_quality_weights_and_zero_weight_exclusion(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        true_lat, true_lon, true_z = 50.1, 11.5, 10.0
        t0 = 1000000.0
        stations = ["GRA1", "BFO", "FUR", "CLZ", "MOX"]
        picks = []
        for idx, st_code in enumerate(stations):
            st = cfg.get_station(st_code)
            d, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = ttt.travel_time("P", max(0.05, d), true_z)
            # Make last station (MOX) Q8 (weight=0) with huge +25s error; GRA1 Q0 (weight=9)
            err = 25.0 if st_code == "MOX" else 0.0
            pk = PhasePick(name="P", onset=format_sh_time(t0 + tt + err), station=st_code)
            pk.set_quality(0 if idx == 0 else (8 if st_code == "MOX" else 2))
            picks.append(pk)

        param, _, diag_res = locate_network_geiger(picks, depth_km=true_z, config=cfg, ttt=ttt, return_diagnostics=True)
        by_sta = {e.station: e for e in diag_res.residuals}
        self.assertGreater(by_sta["GRA1"].weight, by_sta["BFO"].weight)
        self.assertEqual(by_sta["MOX"].weight, 0.0)
        # Q8 outlier must not corrupt solution
        self.assertAlmostEqual(param.lat, true_lat, delta=0.05)
        self.assertAlmostEqual(param.lon, true_lon, delta=0.05)

    def test_geiger_cos_lat_scaling_and_dof_and_exclusion_retention(self) -> None:
        # High-latitude synthetic network around 60N (cos(60) = 0.5)
        cfg = SHMConfig(shm_root=get_config().shm_root)
        ttt = get_ttt_manager(cfg.inputs_dir)
        true_lat, true_lon, true_z = 60.0, 20.0, 15.0
        t0 = 1000000.0
        sta_coords = [
            ("HI_N", 62.5, 20.0),
            ("HI_S", 57.5, 20.0),
            ("HI_E", 60.0, 25.0),
            ("HI_W", 60.0, 15.0),
            ("HI_NE", 61.8, 23.6),
        ]
        picks = []
        traces = []
        for idx, (code, sla, slo) in enumerate(sta_coords):
            d, _, _ = sphere_diff(sla, slo, true_lat, true_lon)
            tt = ttt.travel_time("P", max(0.05, d), true_z)
            # Put station coords only on traces for HI_NE to also test trace metadata fallback!
            if code != "HI_NE":
                cfg.stations[code] = StationInfo(code=code, lat=sla, lon=slo)
            tr = Trace(data=np.zeros(10), delta=0.05, station=code)
            tr.extra_headers["station_lat"] = sla
            tr.extra_headers["station_lon"] = slo
            traces.append(tr)
            err = 0.12 if idx == 0 else (-0.08 if idx == 1 else 0.02)
            picks.append(PhasePick(name="P", onset=format_sh_time(t0 + tt + err), station=code, weight=2))

        param, _, diag_res = locate_network_geiger(
            picks, depth_km=true_z, config=cfg, ttt=ttt, return_diagnostics=True, traces=traces
        )
        # All 5 picks (including HI_NE resolved from traces) must be present
        self.assertEqual(len(diag_res.residuals), 5)
        self.assertEqual(diag_res.dof, 5 - 3)
        lon_se_deg = diag_res.parameters["longitude"].standard_error
        expected_d_lon_km = lon_se_deg * DEG_TO_KM * math.cos(math.radians(param.lat))
        self.assertAlmostEqual(param.d_lon, expected_d_lon_km, delta=0.02)

        # Verify 60N isotropic covariance ellipse is not inflated in E-W by 1/cos(60)=2x
        # With symmetric N/S/E/W station geometry at 60N:
        # cov_m[2,2] (deg_lon^2) ~ 4 * cov_m[1,1] (deg_lat^2), so after cos(60)=0.5 scaling
        # semi_major and semi_minor in km should be comparable (~1.0-1.5x, NOT >2.5x).
        el = diag_res.confidence_ellipse
        self.assertIsNotNone(el)
        self.assertLess(el.semi_major_km / max(1e-6, el.semi_minor_km), 1.6)

        # Verify invalid phase pick is excluded from dof (dof uses actual valid travel times)
        picks_with_bad_ph = list(picks) + [
            PhasePick(name="BOGUS_PH", onset=format_sh_time(t0 + 20.0), station="HI_N", weight=2)
        ]
        _, _, diag_bad = locate_network_geiger(
            picks_with_bad_ph, depth_km=true_z, config=cfg, ttt=ttt, return_diagnostics=True, traces=traces
        )
        self.assertEqual(diag_bad.dof, 2)

        # Verify exclusions are NOT wiped when < 3 active picks remain
        param_few, _, diag_few = locate_network_geiger(
            picks,
            excluded_stations=["HI_S", "HI_E", "HI_W"],
            config=cfg,
            ttt=ttt,
            return_diagnostics=True,
            traces=traces,
        )
        self.assertFalse(diag_few.converged)
        self.assertIn("Insufficient", param_few.comment)
        self.assertIn("error", diag_few.context_data)
        few_weights = {e.station: e.weight for e in diag_few.residuals}
        self.assertEqual(few_weights["HI_S"], 0.0)
        self.assertEqual(few_weights["HI_E"], 0.0)
        self.assertEqual(few_weights["HI_W"], 0.0)

    def test_geiger_backtracking_prevents_sse_divergence_on_poor_geometry(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        true_lat, true_lon, true_z = 50.0, 11.4, 10.0
        t0 = 1000000.0
        # Narrow aperture / nearly collinear stations with slight arrival noise
        stations = ["GRA1", "GRA2", "GRA3", "GRA4"]
        picks = []
        for idx, st_code in enumerate(stations):
            st = cfg.get_station(st_code)
            d, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = ttt.travel_time("P", max(0.05, d), true_z)
            picks.append(PhasePick(name="P", onset=format_sh_time(t0 + tt + (0.2 if idx == 0 else 0.0)), station=st_code, weight=2))
        param, _, diag_res = locate_network_geiger(
            picks, depth_km=true_z, initial_lat=53.0, initial_lon=14.0, config=cfg, ttt=ttt, return_diagnostics=True
        )
        self.assertTrue(math.isfinite(param.lat) and math.isfinite(param.lon))
        self.assertTrue(diag_res.rmse < 2.0)

    def test_geiger_teleseismic_plane_wave_init_and_baz_convention(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        true_lat, true_lon, true_z = 38.0, 143.0, 33.0
        t0 = 1067739240.0  # 02-NOV-2003_02:14:00.000
        grf_stas = [
            "GRA1", "GRA2", "GRA3", "GRA4",
            "GRB1", "GRB2", "GRB3", "GRB4", "GRB5",
            "GRC1", "GRC2", "GRC3", "GRC4",
        ]
        picks = []
        for code in grf_stas:
            st = cfg.get_station(code)
            d, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = ttt.travel_time("P", d, true_z)
            picks.append(PhasePick(name="P", onset=format_sh_time(t0 + tt), station=code, comp="Z"))

        param = locate_network_geiger(picks, depth_km=true_z, config=cfg, ttt=ttt)
        # Should converge near Japan (~38N, 143E), NOT trapped locally near Germany (~50N, 11E)
        self.assertAlmostEqual(param.lat, true_lat, delta=1.0)
        self.assertAlmostEqual(param.lon, true_lon, delta=1.0)
        # param.azimuth must be station->epicenter BAZ (~36.3 deg NNE), NOT epicenter->station (~330.9 deg)
        gra1 = cfg.get_station("GRA1")
        _, expected_baz, _ = sphere_diff(gra1.lat, gra1.lon, param.lat, param.lon)
        self.assertAlmostEqual(param.azimuth, expected_baz, delta=0.5)
        self.assertLess(param.azimuth, 90.0)

    def test_fk_aligns_staggered_start_epochs_and_deltas(self) -> None:
        cfg = get_config()
        stations = ["GRA1", "GRA2", "GRA3", "GRA4", "GRB1", "GRB2", "GRB3", "GRC1"]
        coords = cfg.relative_xy_km(stations, ref_station="CENTRE")
        x_km = [c[0] for c in coords]
        y_km = [c[1] for c in coords]
        true_baz = 70.0
        true_slo = 6.5
        delays = beam_delays_sec(x_km, y_km, true_baz, true_slo)
        base_ep = parse_sh_time("02-NOV-2003_02:25:00.000")

        # Stagger start_epoch by +/- up to 3.0s and mix delta=0.05 and delta=0.025
        epoch_offsets = [0.0, 2.5, -1.5, 3.0, -2.0, 1.25, -0.75, 2.0]
        deltas = [0.05, 0.025, 0.05, 0.025, 0.05, 0.05, 0.025, 0.05]
        traces = []
        for st_code, d_sec, ep_off, dt in zip(stations, delays, epoch_offsets, deltas):
            tr_start = base_ep + ep_off
            # Wave arrives at absolute epoch base_ep + 18.0 - d_sec
            arr_rel = (18.0 - float(d_sec)) - ep_off
            tr = create_synthetic_trace("KUEPPER", length_sec=40.0, dt=dt, pos_sec=arr_rel, ampl=150.0, width=1.0)
            tr.station = st_code
            tr.start_epoch = tr_start
            traces.append(tr)

        # Window specified in absolute epoch seconds covering the wave across all staggered traces
        fk_res = compute_fk_analysis(
            traces,
            t_lo=base_ep + 13.0,
            t_hi=base_ep + 24.0,
            f_lo=0.4,
            f_hi=2.5,
            max_slowness=12.0,
            grid_points=41,
            config=cfg,
        )
        self.assertAlmostEqual(fk_res["azimuth"], true_baz, delta=4.0)
        self.assertAlmostEqual(fk_res["slowness"], true_slo, delta=0.6)
        self.assertGreater(fk_res["semblance"], 0.8)

    def test_plane_wave_validation_correlate_no_overlap_and_vespagram(self) -> None:
        with self.assertRaises(ValueError):
            plane_wave_locate([1.0, 2.0, 3.0], [0.0, 1.0], [0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            plane_wave_locate([1.0, float("nan"), 3.0], [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])

        tr1 = Trace(data=np.sin(np.linspace(0, 10, 200)), delta=0.05, start_epoch=1000.0)
        tr2 = Trace(data=np.sin(np.linspace(0, 10, 200)), delta=0.05, start_epoch=5000.0)
        _lags, _corrs, best_lag, best_val = correlate_traces(tr1, tr2, 1.0, 5.0, max_lag_sec=3.0)
        self.assertEqual(best_lag, 0.0)
        self.assertEqual(best_val, 0.0)

        with self.assertRaises(ValueError):
            compute_vespagram([], azimuth_deg=45.0)

        cfg = get_config()
        grf = ["GRA1", "GRA2", "GRA3", "GRB1"]
        traces = []
        for idx, st in enumerate(grf):
            tr = create_synthetic_trace("KUEPPER", length_sec=20.0, dt=0.05, pos_sec=10.0, ampl=100.0)
            tr.station = st
            tr.start_epoch = 1000000.0 + idx * 0.1
            traces.append(tr)
        v_trs, _t, slos, mat = compute_vespagram(traces, azimuth_deg=90.0, slow_lo=4.0, slow_hi=6.0, slow_step=1.0, power=2.0, config=cfg)
        self.assertEqual(len(v_trs), len(slos))
        b0, _ = compute_beam(traces, azimuth_deg=90.0, slowness_s_deg=float(slos[0]), nth_root=2.0, config=cfg)
        np.testing.assert_allclose(mat[0], b0.data, atol=1e-12)

        # Cancellation callback stops cleanly
        v_cancel, _, slos_cancel, _ = compute_vespagram(
            traces, azimuth_deg=90.0, slow_lo=4.0, slow_hi=8.0, slow_step=1.0, config=cfg,
            progress_callback=lambda i, _tot: i < 2,
        )
        self.assertEqual(len(v_cancel), 2)
        self.assertEqual(len(slos_cancel), 2)

    def test_baer_kradolfer_noisy_trace_onset_and_quality_weight(self) -> None:
        from shm_app.seismology.array_loc import pick_onset_baer_kradolfer

        rng = np.random.default_rng(42)
        dt = 0.05
        n = int(120.0 / dt)
        t = np.arange(n) * dt
        # Realistic ambient background noise + impulsive P arrival at t = 60.0 s
        noise = rng.normal(0.0, 1.0, size=n)
        sig = np.zeros(n, dtype=np.float64)
        mask = t >= 60.0
        sig[mask] = 25.0 * np.exp(-0.25 * (t[mask] - 60.0)) * np.sin(2.0 * np.pi * 1.2 * (t[mask] - 60.0))
        tr = Trace(data=noise + sig, delta=dt, start_epoch=1700000000.0, station="GRA1", comp="Z")

        pk = pick_onset_baer_kradolfer(tr, t_lo=5.0, t_hi=115.0, thresh_sigma=3.5)
        self.assertIsNotNone(pk)
        picked_rel = pk.onset_epoch - tr.start_epoch
        # Must NOT false-trigger in early background noise (t ~ 5..20 s); must land near 60.0 s
        self.assertAlmostEqual(picked_rel, 60.0, delta=0.8)
        # High-SNR pick must receive good SHM quality (1..2) and positive weight (>= 4), never quality=8/weight=0
        self.assertLessEqual(pk.quality, 2)
        self.assertGreaterEqual(pk.weight, 4)

    def test_geiger_teleseismic_plane_wave_init_with_outlier(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        true_lat, true_lon, true_z = 37.23, 37.01, 10.0
        t0 = 1675646254.0
        stas = ["GRA1", "GRA2", "GRA3", "GRB1", "GRB2", "GRC1", "BFO", "FUR", "WET", "CLZ"]
        picks = []
        for idx, code in enumerate(stas):
            st = cfg.get_station(code)
            d, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = ttt.travel_time("P", d, true_z)
            # Inject 1 outlier pick (+8.0 s) on one station to test trimmed plane-wave init
            err = 8.0 if idx == 0 else 0.0
            picks.append(PhasePick(name="P", onset=format_sh_time(t0 + tt + err), station=code, comp="Z"))

        param = locate_network_geiger(picks, depth_km=33.0, config=cfg, ttt=ttt)
        epi_err, _, _ = sphere_diff(true_lat, true_lon, param.lat, param.lon)
        self.assertLess(epi_err, 4.0)

    def test_corrpick_and_fit_depth_phase_non_inverted_shm_weights(self) -> None:
        from shm_app.seismology.array_loc import corrpick_array, fit_depth_phase

        # 1. corrpick_array: near-identical traces (max_c ~ 1.0) must get Q1 / weight=9, never Q8 / weight=0
        t0 = 1700000000.0
        tr_ref = create_synthetic_trace("KUEPPER", length_sec=30.0, dt=0.05, pos_sec=10.0, ampl=100.0)
        tr_ref.station = "GRA1"
        tr_ref.start_epoch = t0
        p_ref = PhasePick(name="P", onset=format_sh_time(t0 + 10.0), station="GRA1", comp="Z")
        p_ref.set_quality(1)
        tr_ref.add_or_replace_pick(p_ref)

        tr_b = create_synthetic_trace("KUEPPER", length_sec=30.0, dt=0.05, pos_sec=10.5, ampl=100.0)
        tr_b.station = "GRA2"
        tr_b.start_epoch = t0
        pks = corrpick_array([tr_ref, tr_b], ref_trace_idx=0, phase_name="P", max_lag_sec=2.5, min_corr=0.5)
        self.assertEqual(len(pks), 2)
        self.assertLessEqual(pks[1].quality, 2)
        self.assertGreaterEqual(pks[1].weight, 4)

        # 2. fit_depth_phase: Q1 pick (weight=9) must dominate Q6 pick (weight=1), not vice versa
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        dist = 70.0
        dt_150 = ttt.travel_time("pP", dist, 150.0) - ttt.travel_time("P", dist, 150.0)
        dt_50 = ttt.travel_time("pP", dist, 50.0) - ttt.travel_time("P", dist, 50.0)

        tr1 = Trace(data=np.zeros(40), delta=0.05, station="GRA1", distance=dist, start_epoch=t0)
        tr1.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="GRA1"))
        dp1 = PhasePick(name="pP", onset=format_sh_time(t0 + dt_150), station="GRA1")
        dp1.set_quality(1)  # best quality -> weight 9
        tr1.add_or_replace_pick(dp1)

        tr2 = Trace(data=np.zeros(40), delta=0.05, station="BFO", distance=dist, start_epoch=t0)
        tr2.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0), station="BFO"))
        dp2 = PhasePick(name="pP", onset=format_sh_time(t0 + dt_50), station="BFO")
        dp2.set_quality(6)  # poor quality -> weight 1
        tr2.add_or_replace_pick(dp2)

        res = fit_depth_phase([tr1, tr2], distance_deg=dist, ttt=ttt, config=cfg)
        self.assertIsNotNone(res)
        # Weighted mean must be strongly pulled toward 150 km (> 125 km), NOT toward 50 km
        self.assertGreater(res["depth"], 125.0)

    def test_geiger_robust_irls_suppresses_gross_outlier_and_trace_quality_flags_spiking(self) -> None:
        from shm_app.seismology.array_loc import assess_trace_quality

        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        true_lat, true_lon, true_z = 52.49, 160.24, 35.0
        t0 = 1753831492.0
        stas = ["GRA1", "GRA2", "GRA3", "GRB1", "GRB2", "GRC1", "BFO", "BRG", "CLZ", "FUR", "MOX", "WET"]
        picks = []
        for code in stas:
            st = cfg.get_station(code)
            d, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt = ttt.travel_time("P", d, true_z)
            # Corrupt WET with a +18s gross timing error (simulating broken/spiking station)
            err = 18.0 if code == "WET" else 0.0
            picks.append(PhasePick(name="P", onset=format_sh_time(t0 + tt + err), station=code, comp="Z"))

        param_rob, _, diag_rob = locate_network_geiger(
            picks, depth_km=33.0, config=cfg, ttt=ttt, return_diagnostics=True, robust=True
        )
        epi_err, _, _ = sphere_diff(true_lat, true_lon, param_rob.lat, param_rob.lon)
        self.assertLess(epi_err, 1.0)
        wet_entries = [e for e in diag_rob.residuals if e.station.upper() == "WET"]
        self.assertTrue(wet_entries and wet_entries[0].is_outlier)

        # assess_trace_quality: clean earthquake trace OK, flat/NaN/chronically spiking flagged
        rng = np.random.default_rng(7)
        dt = 0.05
        n = int(120.0 / dt)
        clean_data = rng.normal(0.0, 1.0, size=n)
        t_arr = np.arange(n) * dt
        clean_data[t_arr >= 60.0] += 50.0 * np.exp(-0.1 * (t_arr[t_arr >= 60.0] - 60.0)) * np.sin(2.0 * np.pi * 1.0 * (t_arr[t_arr >= 60.0] - 60.0))
        spiky_data = rng.normal(0.0, 1.0, size=n)
        spiky_data[::80] += 40.0  # chronic periodic instrumental spikes every 4s
        flat_data = np.full(n, 5.0)
        nan_data = clean_data.copy()
        nan_data[100] = float("nan")

        rep = assess_trace_quality([
            Trace(data=clean_data, delta=dt, station="GRA1", comp="Z"),
            Trace(data=spiky_data, delta=dt, station="WET", comp="Z"),
            Trace(data=flat_data, delta=dt, station="DEAD", comp="Z"),
            Trace(data=nan_data, delta=dt, station="NAN1", comp="Z"),
        ])
        by_sta = {r["station"]: r for r in rep}
        self.assertTrue(by_sta["GRA1"]["ok"])
        self.assertIn("SPIKING", by_sta["WET"]["flags"])
        self.assertIn("FLAT", by_sta["DEAD"]["flags"])
        self.assertIn("NON_FINITE", by_sta["NAN1"]["flags"])


if __name__ == "__main__":
    unittest.main()
