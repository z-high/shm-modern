"""Targeted unit tests for Paket-A refactored components.

Covers:
1. TestSHEngineCalcAndASTSafety: AST safety and CALC command execution in sh_engine.py.
2. TestConfigStationAndFlinnEngdahl: Flinn-Engdahl geographical lookups, station groups,
   station list parsing, and reference station resolution in config_station.py.
3. TestButterworthSplaneAndBilinearDesign: Continuous s-plane poles/zeros and discrete
   bilinear IIR transform frequency responses and stability in filters.py.
4. TestCommonGridResamplingAndStacking: Common time-grid interpolation, weighted sum,
   phase-weighted stacking, and beamforming in trace_ops.py and array_loc.py.
5. TestSFDGenerationEdgeCases: Directory scanning, stream sorting, endian detection,
   skipping of invalid/hidden/truncated files, and SFD formatting/parsing in mseed.py.
6. TestDialogReExportsAndMapHelpers: Facade re-exports in dialogs.py and antimeridian
   polyline/great-circle/distance-ring geometry and basemap rendering in map_helpers.py.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/shm_mpl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shm_app.core.config_station import (
    DEFAULT_REF_LAT,
    DEFAULT_REF_LON,
    DEFAULT_REF_STATION,
    GERESS_STATIONS_DEFAULT,
    GRF_STATIONS,
    GRSN_STATIONS,
    FlinnEngdahlLookup,
    SHMConfig,
    geress_stations,
    get_config,
    parse_station_list,
    resolve_reference_station,
)
from shm_app.core.models import Trace
from shm_app.gui.map_helpers import (
    distance_ring_segments,
    draw_country_basemap,
    get_country_polygons,
    great_circle_segments,
    load_world_countries,
    split_antimeridian_polyline,
)
from shm_app.interpreter.sh_engine import SHCommandEngine, _safe_eval_math_expr
from shm_app.io.mseed import (
    SFDRecord,
    compress_steim1_record,
    generate_sfd_file,
    parse_seed_header,
    parse_sfd_file,
)
from shm_app.processing.filters import (
    _bilinear_stage_from_butter,
    _butterworth_splane,
    design_butterworth_stages,
    evaluate_stage_freq,
)
from shm_app.processing.trace_ops import (
    phase_weighted_stack,
    resample_traces_to_common_grid,
    sum_traces,
)
from shm_app.seismology.array_loc import compute_beam


class TestSHEngineCalcAndASTSafety(unittest.TestCase):
    """Rigorous tests for AST safety and CALC command execution."""

    def test_safe_eval_math_expr_valid_arithmetic_and_precedence(self) -> None:
        # Basic binary operators
        self.assertEqual(_safe_eval_math_expr("1 + 2 * 3"), 7.0)
        self.assertEqual(_safe_eval_math_expr("(10 - 4) / 2"), 3.0)
        self.assertEqual(_safe_eval_math_expr("2 ** 3"), 8.0)
        self.assertEqual(_safe_eval_math_expr("2 ** 3 * 2"), 16.0)
        self.assertEqual(_safe_eval_math_expr("10 // 3"), 3.0)
        self.assertEqual(_safe_eval_math_expr("10 % 3"), 1.0)

        # Unary operators and sign chaining
        self.assertEqual(_safe_eval_math_expr("-5 + +3"), -2.0)
        self.assertEqual(_safe_eval_math_expr("-(4 * 2)"), -8.0)
        self.assertEqual(_safe_eval_math_expr("+(-(+10))"), -10.0)

        # Float literals and scientific notation
        self.assertAlmostEqual(_safe_eval_math_expr("1.5 * 2.0"), 3.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("1e3 / 10"), 100.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("2.5e-1 * 4"), 1.0, places=9)

    def test_safe_eval_math_expr_constants_and_functions(self) -> None:
        # Constants
        self.assertAlmostEqual(_safe_eval_math_expr("pi"), math.pi, places=12)
        self.assertAlmostEqual(_safe_eval_math_expr("e"), math.e, places=12)
        self.assertAlmostEqual(_safe_eval_math_expr("2 * pi"), 2.0 * math.pi, places=12)

        # Trigonometric functions
        self.assertAlmostEqual(_safe_eval_math_expr("sin(0)"), 0.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("sin(pi / 2)"), 1.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("cos(0)"), 1.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("cos(pi)"), -1.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("tan(0)"), 0.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("tan(pi / 4)"), 1.0, places=9)

        # Math helpers: sqrt, log (base 10), abs
        self.assertAlmostEqual(_safe_eval_math_expr("sqrt(16)"), 4.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("sqrt(2) ** 2"), 2.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("log(10)"), 1.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("log(100)"), 2.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("log(1000)"), 3.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("abs(-42.5)"), 42.5, places=9)

        # Nested calls
        self.assertAlmostEqual(_safe_eval_math_expr("sqrt(abs(-25))"), 5.0, places=9)
        self.assertAlmostEqual(_safe_eval_math_expr("abs(sin(-pi / 2))"), 1.0, places=9)

    def test_safe_eval_math_expr_rejects_unsafe_and_forbidden_ast(self) -> None:
        unsafe_expressions = [
            "__import__('os').system('ls')",
            "eval('1 + 1')",
            "open('/etc/passwd')",
            "exec('x = 1')",
            "print('hack')",
            "globals()",
            "locals()",
            "math.sin(1)",
            "os.path.exists('.')",
            "sin(1, foo=2)",
            "x = 1",
            "[x for x in (1, 2)]",
            "{x: x for x in (1, 2)}",
            "(x for x in (1, 2))",
            "lambda x: x",
            "True",
            "False",
            "None",
            "'hello'",
            "\"string_literal\"",
            "a[0]",
            "(1, 2)[0]",
            "unknown_var",
            "cmd",
        ]
        for expr in unsafe_expressions:
            with self.subTest(expr=expr):
                with self.assertRaises((ValueError, SyntaxError)):
                    _safe_eval_math_expr(expr)

        # Syntax errors
        for malformed in ["1 + ", "* 5", "((1 + 2)", "sin()", "sin(1, 2)"]:
            with self.subTest(malformed=malformed):
                with self.assertRaises((ValueError, SyntaxError, TypeError)):
                    _safe_eval_math_expr(malformed)

    def test_cmd_calc_variable_assignment_and_formatting(self) -> None:
        engine = SHCommandEngine()

        # Assignment with '='
        res1 = engine.execute_line("CALC A = 10 * 2.5")
        self.assertEqual(res1, "A = 25")
        self.assertEqual(engine.symbols["A"], "25")

        # Assignment with '$' prefix in variable name
        res2 = engine.execute_line("CALC $B = sqrt(25) + 3")
        self.assertEqual(res2, "B = 8")
        self.assertEqual(engine.symbols["B"], "8")

        # Assignment without '=' (space-separated)
        res3 = engine.execute_line("CALC C 10 + 20")
        self.assertEqual(res3, "C = 30")
        self.assertEqual(engine.symbols["C"], "30")

        # 6g precision formatting for floating-point results
        res_frac = engine.execute_line("CALC F = 1.0 / 3.0")
        self.assertEqual(res_frac, "F = 0.333333")
        self.assertEqual(engine.symbols["F"], "0.333333")

        res_big = engine.execute_line("CALC BIG = 1000000.0")
        self.assertEqual(res_big, "BIG = 1e+06")
        self.assertEqual(engine.symbols["BIG"], "1e+06")

        # Insufficient arguments return empty string without error
        self.assertEqual(engine.execute_line("CALC"), "")
        self.assertEqual(engine.execute_line("CALC VAR"), "")

        # Unsafe expression caught and formatted gracefully
        res_err = engine.execute_line("CALC BAD = __import__('os')")
        self.assertTrue(res_err.startswith("SH Error in CALC:"))
        self.assertNotIn("BAD", engine.symbols)


class TestConfigStationAndFlinnEngdahl(unittest.TestCase):
    """Tests for station database, Flinn-Engdahl lookup, and station resolution."""

    def test_flinn_engdahl_lookup_quadrants_and_wraparound(self) -> None:
        fer = FlinnEngdahlLookup()

        # Quadrant 1 (lat >= 0, lon >= 0): Central Europe / Germany
        igreg1, grname1, isreg1, srname1 = fer.lookup(49.69, 11.22)
        self.assertEqual(igreg1, 543)
        self.assertEqual(grname1, "Germany")
        self.assertEqual(isreg1, 36)
        self.assertEqual(srname1, "NORTHWESTERN EUROPE")

        # Quadrant 2 (lat >= 0, lon < 0): North America (California)
        igreg2, grname2, _, _ = fer.lookup(37.5, -122.0)
        self.assertIn("California", grname2)

        # Quadrant 3 (lat < 0, lon < 0): South America (Chile)
        igreg3, grname3, _, _ = fer.lookup(-33.0, -71.0)
        self.assertIn("Chile", grname3)

        # Quadrant 4 (lat < 0, lon >= 0): Australia
        igreg4, grname4, _, _ = fer.lookup(-25.0, 135.0)
        self.assertIn("Australia", grname4)

        # Longitude wraparound: 190.0 wraps to -170.0; -190.0 wraps to 170.0; 550.0 wraps to -170.0
        igreg_wrap, _, _, _ = fer.lookup(52.0, 190.0)
        igreg_norm, _, _, _ = fer.lookup(52.0, -170.0)
        self.assertEqual(igreg_wrap, igreg_norm)

        igreg_wrap_neg, _, _, _ = fer.lookup(52.0, -190.0)
        igreg_norm_pos, _, _, _ = fer.lookup(52.0, 170.0)
        self.assertEqual(igreg_wrap_neg, igreg_norm_pos)

        igreg_multi_wrap, _, _, _ = fer.lookup(52.0, 550.0)
        self.assertEqual(igreg_multi_wrap, igreg_norm)

        # region_by_id
        grn, isr, srn = fer.region_by_id(543)
        self.assertEqual(grn, "Germany")
        self.assertEqual(isr, 36)
        self.assertEqual(srn, "NORTHWESTERN EUROPE")

        # Unknown region id safely returns fallback
        grn_unk, isr_unk, srn_unk = fer.region_by_id(99999)
        self.assertEqual(grn_unk, "")
        self.assertEqual(isr_unk, 0)
        self.assertEqual(srn_unk, "")

    def test_config_root_and_fer_binding(self) -> None:
        cfg = get_config()
        self.assertEqual(cfg.root, cfg.shm_root)
        self.assertEqual(cfg.fer.shm_root, cfg.root)
        fer_custom = FlinnEngdahlLookup(shm_root=cfg.root)
        self.assertEqual(fer_custom.shm_root, cfg.root)

    def test_station_groups_and_constants(self) -> None:
        self.assertEqual(len(GRF_STATIONS), 13)
        self.assertTrue(all(s.startswith("GR") for s in GRF_STATIONS))
        self.assertIn("GRA1", GRF_STATIONS)
        self.assertIn("GRC4", GRF_STATIONS)

        self.assertEqual(len(GRSN_STATIONS), 23)
        self.assertIn("BFO", GRSN_STATIONS)
        self.assertIn("GRFO", GRSN_STATIONS)
        self.assertIn("WET", GRSN_STATIONS)

        self.assertEqual(len(GERESS_STATIONS_DEFAULT), 25)
        self.assertTrue(all(s.startswith("GE") for s in GERESS_STATIONS_DEFAULT))
        self.assertEqual(len(geress_stations()), 25)

        # Override via custom config station_groups
        custom_cfg = SHMConfig(cfg_root := get_config().root)
        custom_cfg.station_groups["GERESS"] = ["GEA1", "GEA2", "GEA3"]
        self.assertEqual(geress_stations(custom_cfg), ["GEA1", "GEA2", "GEA3"])

    def test_parse_station_list(self) -> None:
        # Delimiters: comma, semicolon, spaces, tabs; case insensitivity and deduplication
        raw_str = "GRA1,  gra2; GRA3\tgrb1, gra1"
        self.assertEqual(parse_station_list(raw_str), ["GRA1", "GRA2", "GRA3", "GRB1"])

        # Sequence of tokens
        tokens = ["GRA1, GRA2", "gra1; GRC1", "grc2  grb5"]
        self.assertEqual(parse_station_list(tokens), ["GRA1", "GRA2", "GRC1", "GRC2", "GRB5"])

        # Empty / whitespace-only inputs
        self.assertEqual(parse_station_list(""), [])
        self.assertEqual(parse_station_list("   \t  \n  "), [])
        self.assertEqual(parse_station_list([]), [])
        self.assertEqual(parse_station_list(["", "   "]), [])

    def test_resolve_reference_station(self) -> None:
        cfg = get_config()

        # Known station in DB
        code, lat, lon = resolve_reference_station(cfg, "GRFO")
        self.assertEqual(code, "GRFO")
        self.assertAlmostEqual(lat, 49.6909, places=3)
        self.assertAlmostEqual(lon, 11.2203, places=3)

        # Unknown station with explicit fallback coordinates
        code_fb, lat_fb, lon_fb = resolve_reference_station(
            cfg, "UNKNOWN_STA", fallback_lat=12.345, fallback_lon=67.890
        )
        self.assertEqual(code_fb, "UNKNOWN_STA")
        self.assertEqual((lat_fb, lon_fb), (12.345, 67.890))

        # Unknown station without fallback coordinates defaults to DEFAULT_REF_*
        code_def, lat_def, lon_def = resolve_reference_station(cfg, "NON_EXISTENT")
        self.assertEqual(code_def, "NON_EXISTENT")
        self.assertAlmostEqual(lat_def, DEFAULT_REF_LAT, places=3)
        self.assertAlmostEqual(lon_def, DEFAULT_REF_LON, places=3)

        # None station code resolves to DEFAULT_REF_STATION
        code_none, lat_none, lon_none = resolve_reference_station(cfg, None)
        self.assertEqual(code_none, DEFAULT_REF_STATION)
        self.assertAlmostEqual(lat_none, 49.6909, places=3)
        self.assertAlmostEqual(lon_none, 11.2203, places=3)


class TestButterworthSplaneAndBilinearDesign(unittest.TestCase):
    """Tests for continuous s-plane Butterworth design and discrete bilinear IIR mapping."""

    def test_butterworth_splane_pole_zero_counts_and_norm(self) -> None:
        w1 = 2.0 * math.pi * 1.0
        w2 = 2.0 * math.pi * 4.0

        for order in (2, 3, 4):
            # Lowpass
            z_lp, p_lp, k_lp = _butterworth_splane("LP", order, w1, w2)
            self.assertEqual(len(z_lp), 0)
            self.assertEqual(len(p_lp), order)
            self.assertAlmostEqual(k_lp, float(w1 ** order), places=4)

            # Highpass
            z_hp, p_hp, k_hp = _butterworth_splane("HP", order, w1, w2)
            self.assertEqual(len(z_hp), order)
            self.assertTrue(all(z == 0j for z in z_hp))
            self.assertEqual(len(p_hp), order)
            self.assertEqual(k_hp, 1.0)

            # Bandpass
            z_bp, p_bp, k_bp = _butterworth_splane("BP", order, w1, w2)
            self.assertEqual(len(z_bp), order)
            self.assertTrue(all(z == 0j for z in z_bp))
            self.assertEqual(len(p_bp), 2 * order)
            self.assertAlmostEqual(k_bp, float((w2 - w1) ** order), places=4)

            # Bandstop
            z_bs, p_bs, k_bs = _butterworth_splane("BS", order, w1, w2)
            self.assertEqual(len(z_bs), 2 * order)
            self.assertEqual(len(p_bs), 2 * order)
            self.assertEqual(k_bs, 1.0)

        # Unknown filter type raises ValueError
        with self.assertRaises(ValueError):
            _butterworth_splane("NOT_A_FILTER", 3, w1, w2)

    def test_bilinear_frequency_responses_and_prewarping(self) -> None:
        dt = 0.01  # 100 Hz sampling rate, Nyquist = 50 Hz

        # 1. Lowpass (order 4, corner 2.0 Hz)
        stg_lp = _bilinear_stage_from_butter("LP", 4, 2.0, 2.0, dt)
        self.assertEqual(stg_lp.store_id, 3)
        h_lp = evaluate_stage_freq(stg_lp, 2.0 * math.pi * np.array([0.001, 2.0, 10.0]))
        self.assertAlmostEqual(abs(h_lp[0]), 1.0, places=3)
        self.assertAlmostEqual(abs(h_lp[1]), 1.0 / math.sqrt(2.0), places=3)  # -3 dB corner
        self.assertLess(abs(h_lp[2]), 0.01)  # Stopband attenuation

        # 2. Highpass (order 4, corner 2.0 Hz)
        stg_hp = _bilinear_stage_from_butter("HP", 4, 2.0, 2.0, dt)
        self.assertEqual(stg_hp.store_id, 3)
        h_hp = evaluate_stage_freq(stg_hp, 2.0 * math.pi * np.array([0.001, 2.0, 25.0]))
        self.assertLess(abs(h_hp[0]), 0.001)  # DC suppression
        self.assertAlmostEqual(abs(h_hp[1]), 1.0 / math.sqrt(2.0), places=3)  # -3 dB corner
        self.assertAlmostEqual(abs(h_hp[2]), 1.0, places=2)  # Passband gain

        # 3. Bandpass (order 3, corners 1.0 Hz - 4.0 Hz, center = 2.0 Hz)
        stg_bp = _bilinear_stage_from_butter("BP", 3, 1.0, 4.0, dt)
        self.assertEqual(stg_bp.store_id, 3)
        h_bp = evaluate_stage_freq(stg_bp, 2.0 * math.pi * np.array([0.01, 1.0, 2.0, 4.0, 25.0]))
        self.assertLess(abs(h_bp[0]), 0.01)  # Low stopband
        self.assertAlmostEqual(abs(h_bp[1]), 1.0 / math.sqrt(2.0), places=2)  # Low cutoff
        self.assertAlmostEqual(abs(h_bp[2]), 1.0, places=2)  # Passband peak
        self.assertAlmostEqual(abs(h_bp[3]), 1.0 / math.sqrt(2.0), places=2)  # High cutoff
        self.assertLess(abs(h_bp[4]), 0.01)  # High stopband

        # 4. Bandstop (order 3, corners 1.0 Hz - 4.0 Hz, notch = 2.0 Hz)
        stg_bs = _bilinear_stage_from_butter("BS", 3, 1.0, 4.0, dt)
        self.assertEqual(stg_bs.store_id, 3)
        h_bs = evaluate_stage_freq(stg_bs, 2.0 * math.pi * np.array([0.01, 1.0, 2.0, 4.0, 25.0]))
        self.assertAlmostEqual(abs(h_bs[0]), 1.0, places=2)  # DC passband
        self.assertAlmostEqual(abs(h_bs[1]), 1.0 / math.sqrt(2.0), places=2)  # Corner
        self.assertLess(abs(h_bs[2]), 1e-4)  # Deep notch at center
        self.assertAlmostEqual(abs(h_bs[3]), 1.0 / math.sqrt(2.0), places=2)  # Corner
        self.assertAlmostEqual(abs(h_bs[4]), 1.0, places=2)  # High passband

    def test_filter_stability_discrete_poles_inside_unit_circle(self) -> None:
        for ftype in ("LP", "HP", "BP", "BS"):
            for order in (2, 3, 4):
                for dt in (0.05, 0.01, 0.005):
                    with self.subTest(ftype=ftype, order=order, dt=dt):
                        stg = _bilinear_stage_from_butter(ftype, order, 1.0, 5.0, dt)
                        roots = np.roots(stg.den_b)
                        self.assertTrue(
                            all(abs(r) < 1.0 for r in roots),
                            f"Unstable discrete pole found in {ftype} order={order} dt={dt}: {roots}",
                        )

            # Also verify mapped poles z = (c + s) / (c - s) for higher orders
            for order in (6, 8):
                c = 2.0 / 0.01
                w1 = c * math.tan(math.pi * 1.0 * 0.01)
                w2 = c * math.tan(math.pi * 5.0 * 0.01)
                _, s_poles, _ = _butterworth_splane(ftype, order, w1, w2)
                z_poles = [(c + p) / (c - p) for p in s_poles]
                self.assertTrue(
                    all(abs(z) < 1.0 for z in z_poles),
                    f"Mapped pole outside unit circle in {ftype} order={order}",
                )

    def test_design_butterworth_stages_api(self) -> None:
        # Recursive design returns store_id=3
        stages_rec = design_butterworth_stages("LP", order=3, freq_lo=1.0, dt=0.05, recursive=True)
        self.assertEqual(len(stages_rec), 1)
        self.assertEqual(stages_rec[0].store_id, 3)

        # Continuous design returns store_id=1
        stages_cont = design_butterworth_stages("BP", order=3, freq_lo=1.0, freq_hi=4.0, recursive=False)
        self.assertEqual(len(stages_cont), 1)
        self.assertEqual(stages_cont[0].store_id, 1)

        # Invalid type raises ValueError
        with self.assertRaises(ValueError):
            design_butterworth_stages("UNKNOWN", order=2)


class TestCommonGridResamplingAndStacking(unittest.TestCase):
    """Tests for common grid interpolation, weighted sum, phase-weighted stacking, and beamforming."""

    def test_resample_traces_to_common_grid_starts_rates_and_gaps(self) -> None:
        t0 = 1000.0
        # Trace 1: 0..10s at dt=0.05 (200 samples)
        tr1 = Trace(data=np.ones(200), station="GRA1", start_epoch=t0, delta=0.05)
        # Trace 2: 5..15s at dt=0.02 (500 samples)
        tr2 = Trace(data=np.full(500, 2.0), station="GRA2", start_epoch=t0 + 5.0, delta=0.02)
        # Trace 3: 20..25s at dt=0.05 (no temporal overlap with tr1)
        tr3 = Trace(data=np.full(100, 3.0), station="GRA3", start_epoch=t0 + 20.0, delta=0.05)

        # 1. Resample traces with different start times and sampling rates
        g_t0, g_dt, grid_t, aligned = resample_traces_to_common_grid([tr1, tr2])
        self.assertEqual(g_t0, t0)
        self.assertEqual(g_dt, 0.05)
        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0].size, grid_t.size)
        self.assertEqual(aligned[1].size, grid_t.size)

        # Before 5s: tr1 has 1.0, tr2 is zero-padded to 0.0
        self.assertEqual(aligned[0][0], 1.0)
        self.assertEqual(aligned[1][0], 0.0)

        # At 12.0s (after tr1 ended at 10.0s, while tr2 is active until 15.0s):
        idx_mid = int(round(12.0 / 0.05))
        self.assertEqual(aligned[0][idx_mid], 0.0)
        self.assertEqual(aligned[1][idx_mid], 2.0)

        # 2. Resample non-overlapping traces with a temporal gap
        g_t0_gap, _, grid_gap, aligned_gap = resample_traces_to_common_grid([tr1, tr3])
        self.assertEqual(g_t0_gap, t0)
        # In the gap between 10s and 20s (e.g. at 15s), both traces must be zero-padded
        idx_gap = int(round(15.0 / 0.05))
        self.assertEqual(aligned_gap[0][idx_gap], 0.0)
        self.assertEqual(aligned_gap[1][idx_gap], 0.0)

        # 3. Empty list raises ValueError
        with self.assertRaises(ValueError):
            resample_traces_to_common_grid([])

    def test_sum_traces_weights_and_empty(self) -> None:
        self.assertIsNone(sum_traces([]))

        t0 = 1000.0
        tr1 = Trace(data=np.array([2.0, 4.0, 6.0]), station="GRA1", start_epoch=t0, delta=0.05, weight=1.0)
        tr2 = Trace(data=np.array([4.0, 8.0, 12.0]), station="GRA2", start_epoch=t0, delta=0.05, weight=2.0)

        # Weighted mean: (1*2 + 2*4) / 3 = 10/3; (1*4 + 2*8) / 3 = 20/3; (1*6 + 2*12) / 3 = 30/3 = 10.0
        s_tr = sum_traces([tr1, tr2], normalize_weights=True)
        self.assertIsNotNone(s_tr)
        assert s_tr is not None
        self.assertEqual(s_tr.station, "SUM")
        self.assertEqual(s_tr.display_offset, 0.0)
        np.testing.assert_allclose(s_tr.data, np.array([10.0 / 3.0, 20.0 / 3.0, 10.0]))

        # Unnormalized sum: 1*tr1 + 2*tr2
        s_tr_unnorm = sum_traces([tr1, tr2], normalize_weights=False)
        self.assertIsNotNone(s_tr_unnorm)
        assert s_tr_unnorm is not None
        np.testing.assert_allclose(s_tr_unnorm.data, np.array([10.0, 20.0, 30.0]))

    def test_phase_weighted_stack_coherence_and_cancellation(self) -> None:
        self.assertIsNone(phase_weighted_stack([]))

        t = np.linspace(0, 2.0 * math.pi, 200)
        sin_data = np.sin(t)
        tr1 = Trace(data=sin_data, station="GRA1", delta=0.05)
        tr2 = Trace(data=sin_data, station="GRA2", delta=0.05)

        # Perfectly coherent signals retain amplitude
        pws_coh = phase_weighted_stack([tr1, tr2], power=2.0)
        self.assertIsNotNone(pws_coh)
        assert pws_coh is not None
        self.assertEqual(pws_coh.station, "PWS")
        np.testing.assert_allclose(pws_coh.data, sin_data, atol=1e-3)

        # Out-of-phase signals cancel out
        tr_inv = Trace(data=-sin_data, station="GRA3", delta=0.05)
        pws_incoh = phase_weighted_stack([tr1, tr_inv], power=2.0)
        self.assertIsNotNone(pws_incoh)
        assert pws_incoh is not None
        np.testing.assert_allclose(pws_incoh.data, np.zeros_like(sin_data), atol=1e-12)

    def test_compute_beam_linear_and_nth_root(self) -> None:
        # Empty trace list raises ValueError
        with self.assertRaises(ValueError):
            compute_beam([], 45.0, 6.5)

        tr1 = Trace(data=np.ones(100), station="GRA1", delta=0.05)
        tr2 = Trace(data=np.ones(100), station="GRA2", delta=0.05)

        # Linear beam (nth_root=1)
        b_lin, delays_lin = compute_beam([tr1, tr2], azimuth_deg=45.0, slowness_s_deg=6.5, ref_station="CENTRE", nth_root=1)
        self.assertEqual(b_lin.station, "BEAM")
        self.assertEqual(b_lin.azimuth, 45.0)
        self.assertEqual(b_lin.slowness, 6.5)
        self.assertEqual(len(delays_lin), 2)
        self.assertTrue(all(isinstance(d, float) for d in delays_lin))
        # Edge check: before wavefront arrives at all stations, beam is partial
        self.assertLessEqual(b_lin.data[0], 1.0)
        # In the overlap region, linear beam sums to 1.0
        np.testing.assert_allclose(b_lin.data[10:50], 1.0)

        # N-th root beam (nth_root=2)
        b_nth, _ = compute_beam([tr1, tr2], azimuth_deg=45.0, slowness_s_deg=6.5, ref_station="CENTRE", nth_root=2)
        self.assertEqual(b_nth.station, "BEAM")
        np.testing.assert_allclose(b_nth.data[10:50], 1.0)


class TestSFDGenerationEdgeCases(unittest.TestCase):
    """Tests for Mini-SEED directory scanning, sorting, endianness, and edge-case handling in SFD generation."""

    def test_generate_sfd_sorting_swap_hdr_skipping_and_none_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.arange(50, dtype=np.int32)

            # 1. Big-Endian record: station BFO, stream bfo-bh-z, start_epoch = 2000.0
            raw_be_late, _ = compress_steim1_record(data, "BFO", "BHZ", start_epoch=2000.0, reclth=256)
            with open(os.path.join(tmpdir, "bfo_late.mseed"), "wb") as f:
                f.write(raw_be_late)

            # 2. Big-Endian record: station BFO, stream bfo-bh-z, start_epoch = 1000.0 (earlier)
            raw_be_early, _ = compress_steim1_record(data, "BFO", "BHZ", start_epoch=1000.0, reclth=256)
            with open(os.path.join(tmpdir, "bfo_early.mseed"), "wb") as f:
                f.write(raw_be_early)

            # 3. Little-Endian record: station CLZ, stream clz-bh-z, swap_hdr should be 1
            # Build valid Little-Endian Mini-SEED header
            reclth = 256
            hdr_fixed_le = struct.pack(
                "<6scB5s2s3s2sHHBBBBHHhhBBBBiHH",
                b"000001", b"D", 8, b"CLZ  ", b"  ", b"BHZ", b"GR",
                2003, 306, 2, 25, 0, 0, 0,
                50, 20, 1, 0, 0, 0, 1, 0, 64, 48,
            )
            blk1000_le = struct.pack("<HHBBBB", 1000, 0, 1, 0, 8, 0)  # fmt=1 (int16), LE (word_order=0)
            pad_le = b"\x00" * (64 - len(hdr_fixed_le) - len(blk1000_le))
            payload_le = np.arange(50, dtype="<i2").tobytes()
            rec_le = hdr_fixed_le + blk1000_le + pad_le + payload_le + b"\x00" * (reclth - 64 - len(payload_le))

            with open(os.path.join(tmpdir, "clz_le.mseed"), "wb") as f:
                f.write(rec_le)

            # 4. Files that MUST be skipped:
            # a) .sfd file
            with open(os.path.join(tmpdir, "existing.sfd"), "w") as f:
                f.write("dummy sfd line\n")
            # b) hidden file
            with open(os.path.join(tmpdir, ".hidden.mseed"), "wb") as f:
                f.write(raw_be_early)
            # c) undersized file (< 64 bytes)
            with open(os.path.join(tmpdir, "tiny.bin"), "wb") as f:
                f.write(b"SHORT_FILE_LESS_THAN_64_BYTES")
            # d) corrupt file (non-SEED header)
            with open(os.path.join(tmpdir, "corrupt.mseed"), "wb") as f:
                f.write(b"\xFF" * 256)
            # e) file smaller than record length (so nrecs <= 0)
            with open(os.path.join(tmpdir, "too_small_for_rec.mseed"), "wb") as f:
                f.write(raw_be_early[:100])

            # Verify LE header structure before writing
            hdr_check_le = parse_seed_header(rec_le)
            self.assertFalse(hdr_check_le.big_endian_hdr)
            self.assertEqual(hdr_check_le.station, "CLZ")

            # A. Test output_sfd=None: returns list of SFDRecord
            records = generate_sfd_file(tmpdir, output_sfd=None)
            self.assertEqual(len(records), 3)
            self.assertTrue(all(isinstance(r, SFDRecord) for r in records))

            # Check sorting strictly by (stream, t_start)
            self.assertEqual(records[0].stream, "bfo-bh-z")
            self.assertEqual(records[0].t_start, 1000.0)
            self.assertEqual(records[1].stream, "bfo-bh-z")
            self.assertEqual(records[1].t_start, 2000.0)
            self.assertEqual(records[2].stream, "clz-bh-z")

            # Check swap_hdr endianness detection
            self.assertEqual(records[0].swap_hdr, 0)  # BE
            self.assertEqual(records[1].swap_hdr, 0)  # BE
            self.assertEqual(records[2].swap_hdr, 1)  # LE

            # B. Test output_sfd writing to file and parsing back
            sfd_file = os.path.join(tmpdir, "sfdlist.sfd")
            records_written = generate_sfd_file(tmpdir, output_sfd=sfd_file)
            self.assertEqual(len(records_written), 3)
            self.assertTrue(os.path.isfile(sfd_file))

            # Verify parse_sfd_file matches the generated records
            parsed_records = parse_sfd_file(sfd_file)
            self.assertEqual(len(parsed_records), 3)
            self.assertEqual(parsed_records[0].stream, "bfo-bh-z")
            self.assertEqual(parsed_records[0].swap_hdr, 0)
            self.assertEqual(parsed_records[1].stream, "bfo-bh-z")
            self.assertEqual(parsed_records[2].stream, "clz-bh-z")
            self.assertEqual(parsed_records[2].swap_hdr, 1)


class TestDialogReExportsAndMapHelpers(unittest.TestCase):
    """Tests for dialog facade re-exports and cartographic geometry helpers."""

    def test_dialogs_facade_reexports(self) -> None:
        import shm_app.gui.dialogs as dlg

        expected_symbols = [
            "EventSummaryDialog",
            "FKAnalysisDialog",
            "FilterDialog",
            "NetworkLocationDialog",
            "ParameterBoxPanel",
            "ParticleMotionDialog",
            "PhaseDialogPanel",
            "ReadDataDialog",
            "SpectrumDialog",
            "VespagramDialog",
        ]
        self.assertEqual(sorted(dlg.__all__), sorted(expected_symbols))

        for symbol in expected_symbols:
            cls = getattr(dlg, symbol, None)
            self.assertIsNotNone(cls, f"Missing dialog re-export: {symbol}")
            self.assertTrue(isinstance(cls, type), f"{symbol} is not a class")

    def test_map_helpers_split_antimeridian_polyline(self) -> None:
        # Edge cases: empty or single-point polylines
        self.assertEqual(split_antimeridian_polyline([], []), [])
        self.assertEqual(split_antimeridian_polyline([10.0], [20.0]), [])

        # Non-wrapping continuous line
        lons = [10.0, 20.0, 30.0]
        lats = [0.0, 5.0, 10.0]
        segs_nowrap = split_antimeridian_polyline(lons, lats)
        self.assertEqual(len(segs_nowrap), 1)
        self.assertEqual(segs_nowrap[0].shape, (3, 2))

        # Antimeridian wrapping line (jumps from 179 to -179)
        lons_wrap = [170.0, 179.0, -179.0, -170.0]
        lats_wrap = [10.0, 10.0, 10.0, 10.0]
        segs_wrap = split_antimeridian_polyline(lons_wrap, lats_wrap)
        self.assertEqual(len(segs_wrap), 2)
        self.assertEqual(segs_wrap[0].shape, (2, 2))
        self.assertEqual(segs_wrap[1].shape, (2, 2))

    def test_map_helpers_great_circle_segments(self) -> None:
        # Zero distance returns empty list
        self.assertEqual(great_circle_segments(10.0, 20.0, 10.0, 20.0), [])

        # Trans-Pacific path crossing the antimeridian
        segs_cross = great_circle_segments(35.0, 140.0, 37.0, -122.0, n_pts=64)
        self.assertGreaterEqual(len(segs_cross), 1)
        for s in segs_cross:
            self.assertEqual(s.shape[1], 2)

        # Near-antipodal points
        segs_anti = great_circle_segments(0.0, 0.0, 0.0, 180.0, n_pts=30)
        self.assertGreaterEqual(len(segs_anti), 1)

    def test_map_helpers_distance_ring_segments(self) -> None:
        # Distance ring around (0, 0)
        rings = distance_ring_segments(0.0, 0.0, 30.0, n_pts=72)
        self.assertGreaterEqual(len(rings), 1)
        for r in rings:
            self.assertEqual(r.shape[1], 2)

        # Distance ring crossing date line
        rings_wrap = distance_ring_segments(10.0, 175.0, 20.0, n_pts=72)
        self.assertGreaterEqual(len(rings_wrap), 1)

    def test_map_helpers_load_world_countries_and_polygons(self) -> None:
        countries = load_world_countries()
        self.assertIsInstance(countries, list)
        self.assertGreater(len(countries), 100)

        polys = get_country_polygons()
        self.assertIsInstance(polys, list)
        self.assertGreater(len(polys), 500)
        self.assertEqual(polys[0].ndim, 2)
        self.assertEqual(polys[0].shape[1], 2)

    def test_draw_country_basemap_rendering(self) -> None:
        fig, ax = plt.subplots()
        try:
            draw_country_basemap(
                ax,
                xticks=np.array([-180, -90, 0, 90, 180]),
                yticks=np.array([-90, -45, 0, 45, 90]),
            )
            # PolyCollection added
            self.assertGreaterEqual(len(ax.collections), 1)
            # Zero axes (hline and vline) added
            self.assertGreaterEqual(len(ax.lines), 2)
            # Ticks correctly configured
            np.testing.assert_array_equal(ax.get_xticks(), np.array([-180, -90, 0, 90, 180]))
            np.testing.assert_array_equal(ax.get_yticks(), np.array([-90, -45, 0, 45, 90]))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
