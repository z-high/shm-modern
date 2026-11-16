"""Intensive end-to-end tests for the SH Command Interpreter (shc) and PyQt6 GUI (shm)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.models import PhasePick
from shm_app.gui.dialogs import (
    EventSummaryDialog,
    FKAnalysisDialog,
    FilterDialog,
    NetworkLocationDialog,
    ParticleMotionDialog,
    ReadDataDialog,
    SpectrumDialog,
    VespagramDialog,
)
from shm_app.gui.main_window import SHMMainWindow
from shm_app.interpreter.sh_engine import SHCommandEngine


class TestSHCommandInterpreter(unittest.TestCase):
    def test_sh_command_pipeline_and_scripting(self) -> None:
        engine = SHCommandEngine()
        gse_file = os.path.join(SHM_DEFAULT_ROOT, "data-examples", "gse", "GRF_031102_0225.GSE")

        script = f"""
        ! Load GRF array GSE waveforms, filter, pick, locate, beam, and compute magnitudes
        SDEF PHASE P
        READGSE {gse_file}
        DEMEAN ALL
        TREND ALL
        BUTTER BP 0.7 2.5 3
        PICK ALL
        CORRPICK $PHASE
        LOCATE $PHASE
        BEAM 45.0 6.5 CENTRE
        ALIGN $PHASE
        MAG $PHASE
        THEO 65.0 33.0
        SPECTRUM /FFT 1
        SPECTRUM /PSD 1
        SPECTRUM /MEM 1
        CALC X = 2 + 3 * 4
        IF $X EQ 14 ECHO CALC_OK
        LIST
        """
        logs = engine.execute_script(script)
        joined = "\n".join(logs)
        self.assertIn("CALC_OK", joined)
        # 19 original GRF traces + 1 BEAM + 1 ALIGN = 21 traces
        self.assertEqual(len(engine.traces), 21)
        self.assertGreater(engine.current_params.slowness, 0.0)
        self.assertGreater(engine.current_params.mag_mb, 0.0)

    def test_sh_engine_io_commands(self) -> None:
        engine = SHCommandEngine()
        with tempfile.TemporaryDirectory() as td:
            # 1. CREATE trace and verify initial state
            create_msg = engine.execute_line("CREATE KUEPPER 20.0 0.05 5.0 100.0 1.0")
            self.assertIn("Created synthetic KUEPPER trace", create_msg)
            self.assertEqual(len(engine.traces), 1)
            engine.traces[0].station = "TEST"
            engine.traces[0].comp = "Z"
            orig_data = engine.traces[0].data.copy()
            self.assertGreater(float(np.max(np.abs(orig_data))), 0.0)

            # 2. WRITE & READ Q-file (.QHD + .QBN)
            qpath = os.path.join(td, "test_qfile")
            w_res = engine.execute_line(f"WRITE {qpath} ALL")
            self.assertIn("Wrote 1 trace(s)", w_res)
            self.assertTrue(os.path.isfile(qpath + ".QHD"))
            self.assertTrue(os.path.isfile(qpath + ".QBN"))
            engine.execute_line("DEL ALL")
            self.assertEqual(len(engine.traces), 0)
            r_res = engine.execute_line(f"READ {qpath}")
            self.assertIn("Read 1 trace(s)", r_res)
            self.assertEqual(len(engine.traces), 1)
            self.assertEqual(engine.traces[0].station, "TEST")
            self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)
            np.testing.assert_allclose(engine.traces[0].data, orig_data, atol=1e-3)

            # Test READ with specific record index
            engine.execute_line("DEL ALL")
            r_rec = engine.execute_line(f"READ {qpath} 1")
            self.assertIn("Read 1 trace(s)", r_rec)
            self.assertEqual(len(engine.traces), 1)
            self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)
            np.testing.assert_allclose(engine.traces[0].data, orig_data, atol=1e-3)

            # 3. WRITEA & READA ASCII
            apath = os.path.join(td, "test.asc")
            wa_res = engine.execute_line(f"WRITEA {apath} ALL")
            self.assertIn("Wrote 1 trace(s)", wa_res)
            self.assertTrue(os.path.isfile(apath))
            engine.execute_line("DEL ALL")
            ra_res = engine.execute_line(f"READA {apath}")
            self.assertIn("Read 1 ASCII trace(s)", ra_res)
            self.assertEqual(len(engine.traces), 1)
            self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)
            np.testing.assert_allclose(engine.traces[0].data, orig_data, atol=1e-3)

            # 4. WRITESEED & READS / READM
            mpath = os.path.join(td, "test.mseed")
            ws_res = engine.execute_line(f"WRITESEED {mpath} ALL")
            self.assertIn("Wrote 1 trace(s)", ws_res)
            self.assertTrue(os.path.isfile(mpath))
            engine.execute_line("DEL ALL")
            rs_res = engine.execute_line(f"READS {mpath}")
            self.assertIn("Read 1 Mini-SEED trace(s)", rs_res)
            self.assertEqual(len(engine.traces), 1)
            self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)
            np.testing.assert_allclose(engine.traces[0].data, orig_data, atol=0.5)
            # Test READM alias
            engine.execute_line("DEL ALL")
            rm_res = engine.execute_line(f"READM {mpath}")
            self.assertIn("Read 1 Mini-SEED trace(s)", rm_res)
            self.assertEqual(len(engine.traces), 1)
            self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)
            np.testing.assert_allclose(engine.traces[0].data, orig_data, atol=0.5)

            # 5. WRITEGSE & READGSE
            gpath = os.path.join(td, "test.gse")
            wg_res = engine.execute_line(f"WRITEGSE {gpath} ALL")
            self.assertIn("Wrote 1 trace(s)", wg_res)
            self.assertTrue(os.path.isfile(gpath))
            engine.execute_line("DEL ALL")
            rg_res = engine.execute_line(f"READGSE {gpath}")
            self.assertIn("Read 1 GSE trace(s)", rg_res)
            self.assertEqual(len(engine.traces), 1)
            self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)
            np.testing.assert_allclose(engine.traces[0].data, orig_data, atol=0.5)

            # 6. WRITEEVT & READEVT
            epath = os.path.join(td, "test.evt")
            engine.current_params.lat = 50.1
            engine.current_params.lon = 12.3
            engine.current_params.depth = 15.0
            engine.traces[0].add_or_replace_pick(
                PhasePick(station="TEST", comp="Z", onset="01-JAN-2020_12:00:00.000", name="P")
            )
            we_res = engine.execute_line(f"WRITEEVT {epath}")
            self.assertIn("Wrote event parameters", we_res)
            self.assertTrue(os.path.isfile(epath))
            engine.current_params.lat = 0.0
            re_res = engine.execute_line(f"READEVT {epath}")
            self.assertIn("SEISMIC HANDLER EVENT SUMMARY", re_res)
            self.assertAlmostEqual(engine.current_params.lat, 50.1)

            # 7. Error handling for missing arguments & missing files
            for cmd in (
                "READ",
                "WRITE",
                "READS",
                "READM",
                "WRITESEED",
                "READGSE",
                "WRITEGSE",
                "READA",
                "WRITEA",
                "READEVT",
                "WRITEEVT",
            ):
                err_msg = engine.execute_line(cmd)
                self.assertIn(f"SH Error in {cmd}", err_msg)

            non_existent = os.path.join(td, "does_not_exist_file")
            self.assertIn("SH Error in READ", engine.execute_line(f"READ {non_existent}"))
            self.assertIn("SH Error in READA", engine.execute_line(f"READA {non_existent}"))
            self.assertIn("SH Error in READEVT", engine.execute_line(f"READEVT {non_existent}"))

    def test_sh_engine_signal_processing_commands(self) -> None:
        engine = SHCommandEngine()

        # 1. CREATE command variations
        c_res = engine.execute_line("CREATE KUEPPER 40.0 0.05 10.0 100.0 2.0")
        self.assertIn("Created synthetic KUEPPER trace", c_res)
        self.assertEqual(len(engine.traces), 1)
        self.assertEqual(engine.traces[0].length, 800)

        engine.execute_line("CREATE SPIKE 10.0 0.05 5.0 50.0")
        self.assertEqual(len(engine.traces), 2)
        engine.execute_line("CREATE SINE 10.0 0.05 1.0 10.0")
        self.assertEqual(len(engine.traces), 3)
        engine.execute_line("DEL 2,3")
        self.assertEqual(len(engine.traces), 1)

        # 2. FILTER with composite chain -GRSN+WWSSN_SP
        pre_filter = engine.traces[0].data.copy()
        self.assertGreater(float(np.max(np.abs(pre_filter))), 0.0)
        f_res = engine.execute_line("FILTER -GRSN+WWSSN_SP")
        self.assertIn("Applied filter '-GRSN+WWSSN_SP'", f_res)
        self.assertEqual(engine.traces[0].filter_name, "-GRSN+WWSSN_SP")
        self.assertFalse(np.allclose(engine.traces[0].data, pre_filter))
        self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)

        # FILTER qualifiers and defaults
        f_qual = engine.execute_line("FILTER WWSSN_SP /LIST=1 /FFT")
        self.assertIn("Applied filter 'WWSSN_SP'", f_qual)
        engine.loaded_filters = ["WWSSN_SP"]
        f_def = engine.execute_line("FILTER")
        self.assertIn("Applied filter 'WWSSN_SP'", f_def)

        # 3. FLIP and MULT
        pre_flip = engine.traces[0].data.copy()
        self.assertGreater(float(np.max(np.abs(pre_flip))), 0.0)
        orig_val = float(engine.traces[0].data[200])
        engine.execute_line("FLIP 1")
        np.testing.assert_allclose(engine.traces[0].data, -pre_flip)
        self.assertAlmostEqual(engine.traces[0].data[200], -orig_val)

        pre_mult = engine.traces[0].data.copy()
        engine.execute_line("MULT 1 2.0")
        np.testing.assert_allclose(engine.traces[0].data, 2.0 * pre_mult)
        self.assertAlmostEqual(engine.traces[0].data[200], -2.0 * orig_val)
        engine.execute_line("MULT 0.5")
        np.testing.assert_allclose(engine.traces[0].data, pre_mult)
        self.assertAlmostEqual(engine.traces[0].data[200], -orig_val)

        # 4. DESPIKE
        engine.traces[0].data[50] += 5000.0
        d_res = engine.execute_line("DESPIKE ALL 3.0")
        self.assertIn("spikes removed", d_res)
        self.assertLess(abs(engine.traces[0].data[50]), 2000.0)

        # 5. RESAMPLE & DECIMATE
        engine.execute_line("RESAMPLE 0.1")
        self.assertEqual(engine.traces[0].delta, 0.1)
        self.assertEqual(engine.traces[0].length, 400)
        engine.execute_line("DECIMATE 2")
        self.assertEqual(engine.traces[0].delta, 0.2)
        self.assertEqual(engine.traces[0].length, 200)

        # 6. CUT
        cut_res = engine.execute_line("CUT 5.0 15.0")
        self.assertIn("Cut 1 trace(s)", cut_res)
        self.assertAlmostEqual(engine.traces[0].duration, 10.0, delta=0.5)

        # 7. INT & DERIVE
        pre_int = engine.traces[0].data.copy()
        int_res = engine.execute_line("INT ALL")
        self.assertIn("Integrated 1 trace(s)", int_res)
        self.assertFalse(np.allclose(engine.traces[0].data, pre_int))
        self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)

        pre_der1 = engine.traces[0].data.copy()
        der_res1 = engine.execute_line("DERIVE ALL 1")
        self.assertIn("Differentiated 1 trace(s)", der_res1)
        self.assertFalse(np.allclose(engine.traces[0].data, pre_der1))
        self.assertGreater(float(np.max(np.abs(engine.traces[0].data))), 0.0)

        pre_der2 = engine.traces[0].data.copy()
        der_res2 = engine.execute_line("DIFF ALL 2")
        self.assertIn("Differentiated 1 trace(s)", der_res2)
        self.assertFalse(np.allclose(engine.traces[0].data, pre_der2))

        # 8. HILBERT, ENVELOPE, SPHASE
        h_res = engine.execute_line("HILBERT 1")
        self.assertIn("Computed Hilbert transform", h_res)
        self.assertEqual(engine.traces[0].comp, "H")

        env_res = engine.execute_line("ENVELOPE 1")
        self.assertIn("Computed analytic envelope", env_res)
        self.assertEqual(engine.traces[0].comp, "E")
        self.assertTrue(np.all(engine.traces[0].data >= 0.0))

        sph_res = engine.execute_line("SPHASE 1")
        self.assertIn("Computed instantaneous phase", sph_res)
        self.assertTrue(np.all((engine.traces[0].data >= -180.0) & (engine.traces[0].data <= 180.0)))

        # 9. MEND
        engine.traces[0].data[10:15] = 999.0
        mend_res = engine.execute_line("MEND 10 15 1")
        self.assertIn("Mended samples [10..15]", mend_res)
        self.assertNotEqual(engine.traces[0].data[12], 999.0)

        # 10. ATTENUATE
        att_res = engine.execute_line("ATTENUATE 0.5 ALL")
        self.assertIn("Applied attenuation t*=0.5", att_res)

        # 11. SPIKING, SUM, and DECONV
        engine.execute_line("CREATE KUEPPER 20.0 0.05")
        spik_res = engine.execute_line("SPIKING 40 0.01")
        self.assertIn("Computed spiking deconvolution trace", spik_res)
        self.assertEqual(engine.traces[-1].station, "SPIK")

        sum_res = engine.execute_line("SUM ALL")
        self.assertIn("Appended SUM trace", sum_res)
        self.assertEqual(engine.traces[-1].station, "SUM")

        sum_nonorm = engine.execute_line("SUM ALL /NONORM")
        self.assertIn("Appended SUM trace", sum_nonorm)

        engine.execute_line("CREATE KUEPPER 10.0 0.2")
        decv_res = engine.execute_line("DECONV 0.02")
        self.assertIn("Appended water-level deconvolved trace DECV", decv_res)
        self.assertEqual(engine.traces[-1].station, "DECV")

        # 12. Error handling & edge cases
        self.assertIn("SH Error in RESAMPLE", engine.execute_line("RESAMPLE"))
        self.assertIn("SH Error in DECIMATE", engine.execute_line("DECIMATE"))
        self.assertIn("SH Error in CUT", engine.execute_line("CUT 5.0"))
        self.assertIn("SH Error in MEND", engine.execute_line("MEND 10"))

        temp_traces = engine.traces
        engine.traces = []
        self.assertIn("No trace available", engine.execute_line("SPIKING"))
        self.assertIn("SH Error in DECONV", engine.execute_line("DECONV"))
        engine.traces = temp_traces

    def test_sh_engine_3c_and_array_commands(self) -> None:
        engine = SHCommandEngine()

        # Setup Z, N, E triplet for station 'TRI'
        for c in ("Z", "N", "E"):
            engine.execute_line("CREATE KUEPPER 20.0 0.05 5.0 100.0 1.0")
            engine.traces[-1].station = "TRI"
            engine.traces[-1].comp = c
        self.assertEqual(len(engine.traces), 3)

        # 1. ROT / ROT2 / ROTAZ
        n_data = engine.traces[1].data.copy()
        e_data = engine.traces[2].data.copy()
        rad = np.radians(45.0)
        expected_r = n_data * np.cos(rad) + e_data * np.sin(rad)
        expected_t = -n_data * np.sin(rad) + e_data * np.cos(rad)

        rot_res = engine.execute_line("ROT 45.0")
        self.assertIn("Rotated (TRI N,E) by 45.0 deg -> appended R, T traces", rot_res)
        self.assertEqual(len(engine.traces), 5)
        self.assertEqual(engine.traces[-2].comp, "R")
        self.assertEqual(engine.traces[-1].comp, "T")
        self.assertGreater(float(np.max(np.abs(engine.traces[-2].data))), 0.0)
        self.assertGreater(float(np.max(np.abs(engine.traces[-1].data))), 0.0)
        np.testing.assert_allclose(engine.traces[-2].data, expected_r, atol=1e-5)
        np.testing.assert_allclose(engine.traces[-1].data, expected_t, atol=1e-5)

        rot2_res = engine.execute_line("ROT2 30.0")
        self.assertIn("Rotated (TRI N,E) by 30.0 deg", rot2_res)

        rotaz_res = engine.execute_line("ROTAZ 60.0")
        self.assertIn("Rotated (TRI N,E) by 60.0 deg", rotaz_res)

        # 2. ROT3 / ROTINCI
        orig_energy = float(np.sum(engine.traces[0].data**2 + engine.traces[1].data**2 + engine.traces[2].data**2))
        rot3_res = engine.execute_line("ROT3 45.0 20.0")
        self.assertIn("Rotated 3D (TRI Z,N,E) by az=45.0, inci=20.0 -> appended L, Q, T traces", rot3_res)
        self.assertEqual(engine.traces[-3].comp, "L")
        self.assertEqual(engine.traces[-2].comp, "Q")
        self.assertEqual(engine.traces[-1].comp, "T")
        self.assertGreater(float(np.max(np.abs(engine.traces[-3].data))), 0.0)
        self.assertGreater(float(np.max(np.abs(engine.traces[-2].data))), 0.0)
        self.assertGreater(float(np.max(np.abs(engine.traces[-1].data))), 0.0)
        rot_energy = float(np.sum(engine.traces[-3].data**2 + engine.traces[-2].data**2 + engine.traces[-1].data**2))
        self.assertAlmostEqual(orig_energy, rot_energy, places=3)

        rotinci_res = engine.execute_line("ROTINCI 45.0 25.0")
        self.assertIn("Rotated 3D (TRI Z,N,E) by az=45.0, inci=25.0", rotinci_res)

        # 3. UVW transform
        uvw_res = engine.execute_line("UVW")
        self.assertIn("Transformed STS-2 ZNE->UVW", uvw_res)
        uvw_inv = engine.execute_line("UVW /INV")
        self.assertIn("Transformed STS-2 UVW->ZNE", uvw_inv)

        # 4. MDIR (Covariance polarization)
        mdir_res = engine.execute_line("MDIR 3.0 7.0")
        self.assertIn("MDIR Polarization (TRI)", mdir_res)
        self.assertGreaterEqual(engine.current_params.incidence, 0.0)
        self.assertGreaterEqual(engine.current_params.azimuth, 0.0)

        # 5. POLFIL (Sliding window polarization filter)
        pol_res = engine.execute_line("POLFIL 1.0 2.0 1")
        self.assertIn("Applied polarization filter (mode=1, win=1.0s, power=2.0) -> appended 3 traces", pol_res)
        pol_res2 = engine.execute_line("POLFIL 1.0 2.0 2")
        self.assertIn("Applied polarization filter (mode=2", pol_res2)

        # 6. CORR (Cross-correlation)
        corr_res = engine.execute_line("CORR 1 2 0.0 5.0")
        self.assertIn("Cross-correlation peak =", corr_res)
        self.assertEqual(engine.traces[-1].station, "CORR")

        # 7. Error handling when traces are insufficient
        engine.execute_line("DEL ALL")
        self.assertIn("SH Error in ROT", engine.execute_line("ROT 45.0"))
        self.assertIn("SH Error in ROT2", engine.execute_line("ROT2 45.0"))
        self.assertIn("SH Error in ROT3", engine.execute_line("ROT3 45.0 20.0"))
        self.assertIn("SH Error in MDIR", engine.execute_line("MDIR"))
        self.assertIn("SH Error in POLFIL", engine.execute_line("POLFIL"))
        self.assertIn("SH Error in CORR", engine.execute_line("CORR 1 2"))
        self.assertIn("SH Error in UVW", engine.execute_line("UVW"))

    def test_sh_engine_workspace_management_commands(self) -> None:
        engine = SHCommandEngine()

        for _ in range(4):
            engine.execute_line("CREATE KUEPPER 20.0 0.05")
        self.assertEqual(len(engine.traces), 4)

        # 1. HIDE & SHOW / DISPLAY / UNHIDE
        h_res = engine.execute_line("HIDE 2")
        self.assertIn("Hidden 1 trace(s)", h_res)
        self.assertTrue(engine.traces[1].hidden)
        self.assertEqual(len(engine.visible_traces()), 3)

        s_res = engine.execute_line("SHOW 2")
        self.assertIn("Displayed 1 trace(s)", s_res)
        self.assertFalse(engine.traces[1].hidden)
        self.assertEqual(len(engine.visible_traces()), 4)

        engine.execute_line("HIDE 2")
        d_res = engine.execute_line("DISPLAY 2")
        self.assertIn("Displayed 1 trace(s)", d_res)
        self.assertFalse(engine.traces[1].hidden)

        engine.execute_line("HIDE 2")
        u_res = engine.execute_line("UNHIDE 2")
        self.assertIn("Displayed 1 trace(s)", u_res)
        self.assertFalse(engine.traces[1].hidden)

        # 2. SEL / SELECT
        sel_res = engine.execute_line("SEL 1,3")
        self.assertIn("Selected 2 trace(s)", sel_res)
        self.assertTrue(engine.traces[0].selected)
        self.assertFalse(engine.traces[1].selected)
        self.assertTrue(engine.traces[2].selected)
        self.assertFalse(engine.traces[3].selected)

        sel_none = engine.execute_line("SEL NONE")
        self.assertIn("Cleared trace selection", sel_none)
        self.assertTrue(all(not t.selected for t in engine.traces))

        sel_alias = engine.execute_line("SELECT 2")
        self.assertIn("Selected 1 trace(s)", sel_alias)
        self.assertTrue(engine.traces[1].selected)
        engine.execute_line("SEL")
        self.assertTrue(all(not t.selected for t in engine.traces))

        # 3. ZOOM
        z_res = engine.execute_line("ZOOM 2.5")
        self.assertIn("Global zoom = 2.5", z_res)
        self.assertEqual(engine.global_zoom, 2.5)

        z_rel = engine.execute_line("ZOOM 2.0 /REL")
        self.assertAlmostEqual(engine.global_zoom, 5.0)

        z_tr = engine.execute_line("ZOOM 1 3.0")
        self.assertIn("Updated zoom on 1 trace(s) to 3", z_tr)
        self.assertEqual(engine.traces[0].norm, 3.0)

        engine.execute_line("ZOOM 1 2.0 /REL")
        self.assertEqual(engine.traces[0].norm, 6.0)

        # 4. NORM
        pre_norm = engine.traces[0].data.copy()
        self.assertGreater(float(np.max(np.abs(pre_norm))), 0.0)
        for mode in ("A", "C", "AW", "SW"):
            n_res = engine.execute_line(f"NORM {mode}")
            self.assertIn(f"Normalization mode: {mode}", n_res)
            self.assertEqual(engine.norm_mode, mode)
            np.testing.assert_array_equal(engine.traces[0].data, pre_norm)
            self.assertFalse(np.any(np.isnan(engine.traces[0].data)))
        self.assertIn("SH Error in NORM", engine.execute_line("NORM INVALID"))

        # 5. STW & DTW (including reversed bounds auto-swap)
        stw_res = engine.execute_line("STW 2.0 8.0")
        self.assertIn("Time window set to [2.00, 8.00] s", stw_res)
        self.assertEqual(engine.time_window, (2.0, 8.0))
        self.assertEqual(engine.view_window, (2.0, 8.0))

        # Test reversed bounds auto-swap
        stw_rev = engine.execute_line("STW 9.0 3.0")
        self.assertEqual(engine.time_window, (3.0, 9.0))
        self.assertEqual(engine.view_window, (3.0, 9.0))

        stw_reset = engine.execute_line("STW")
        self.assertIn("Time window reset to full trace duration", stw_reset)
        self.assertIsNone(engine.time_window)
        self.assertIsNone(engine.view_window)

        engine.execute_line("STW 1.0 4.0")
        dtw_res = engine.execute_line("DTW")
        self.assertIn("Time window reset", dtw_res)
        self.assertIsNone(engine.time_window)
        self.assertIsNone(engine.view_window)

        # 6. TRESET
        engine.execute_line("ZOOM 3.0")
        engine.execute_line("NORM A")
        engine.execute_line("STW 1.0 4.0")
        engine.execute_line("HIDE 1")
        engine.traces[2].display_offset = 2.5
        engine.traces[2].norm = 4.0

        tr_res = engine.execute_line("TRESET")
        self.assertIn("Reset trace zooms, shifts, and time windows", tr_res)
        self.assertEqual(engine.global_zoom, 1.0)
        self.assertEqual(engine.norm_mode, "C")
        self.assertIsNone(engine.time_window)
        self.assertIsNone(engine.view_window)
        self.assertFalse(engine.traces[0].hidden)
        self.assertEqual(engine.traces[2].display_offset, 0.0)
        self.assertEqual(engine.traces[2].norm, 1.0)

        # 7. DEL / DELETE
        del_res = engine.execute_line("DEL 1")
        self.assertIn("Deleted 1 trace(s)", del_res)
        self.assertEqual(len(engine.traces), 3)

        del_alias = engine.execute_line("DELETE 1")
        self.assertIn("Deleted 1 trace(s)", del_alias)
        self.assertEqual(len(engine.traces), 2)

        del_all = engine.execute_line("DEL ALL")
        self.assertIn("Deleted 2 trace(s)", del_all)
        self.assertEqual(len(engine.traces), 0)

        # 8. Extra workspace commands: SHIFT, TITLE, FILI
        engine.execute_line("CREATE KUEPPER 10.0 0.05")
        sh_res = engine.execute_line("SHIFT 1 1.5")
        self.assertIn("Shifted 1 trace(s) by +1.500 s", sh_res)
        self.assertEqual(engine.traces[0].display_offset, 1.5)

        t_res = engine.execute_line("TITLE My Custom Title")
        self.assertEqual(engine.title, "My Custom Title")

        fili_res = engine.execute_line("FILI WWSSN_SP GRSN")
        self.assertEqual(engine.loaded_filters, ["WWSSN_SP", "GRSN"])


class TestSHMDesktopGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_menus_canvases_dialogs_and_shortcuts(self) -> None:
        win = SHMMainWindow(autoload_example=True)
        win.show()
        self.app.processEvents()

        self.assertEqual(len(win.engine.traces), 19)

        # Render MainTraceCanvas and MagnifierCanvas
        win.main_canvas.repaint()
        win.mag_canvas.repaint()
        self.app.processEvents()

        # Simulate single-key shortcuts ('+', 'i', '6')
        for ch in ("+", "i", "6"):
            ev = QKeyEvent(QKeyEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, ch)
            win.keyPressEvent(ev)
        self.assertEqual(win.phase_panel.active_sign, 1)
        self.assertEqual(win.phase_panel.active_spec, "i")
        self.assertEqual(win.phase_panel.active_quality, 6)

        # Simulate clicking on trace 0 to pick P phase
        ref_ep = win.main_canvas.reference_epoch()
        win._on_main_trace_clicked(0, ref_ep + 25.0, 25.0)
        self.assertIsNotNone(win.engine.traces[0].get_pick("P"))

        # Test Magnifier amplitude measurement callback
        win._on_magnifier_amplitude_measured(0, 20.0, 30.0)
        self.assertGreater(win.engine.current_params.mag_mb, 0.0)

        # Exercise Work / Array / Locate menu actions
        win.action_demean()
        win.action_detrend()
        win.action_auto_pick()
        win.action_corr_pick()
        win.action_locate_plane_wave()
        win.action_form_beam()
        win.action_align_phase()
        win.action_theoretical_phases()
        win.action_compute_magnitudes()
        win.action_unfilter()
        self.app.processEvents()

        # Instantiate and test every interactive dialog in offscreen mode
        read_dlg = ReadDataDialog(config=win.config, parent=win)
        self.assertGreater(len(read_dlg.station_checks), 20)
        read_dlg.close()

        flt_dlg = FilterDialog(dt=0.05, config=win.config, parent=win)
        flt_dlg.but_type.setCurrentIndex(0)
        flt_dlg._update_preview()
        flt_dlg.close()

        fk_dlg = FKAnalysisDialog(win.engine.visible_traces()[:6], t_lo=10.0, t_hi=30.0, config=win.config, parent=win)
        self.assertIsNotNone(fk_dlg.last_result)
        fk_dlg.close()

        vespa_dlg = VespagramDialog(win.engine.visible_traces()[:6], azimuth_deg=45.0, config=win.config, parent=win)
        self.assertGreater(len(vespa_dlg.vespa_traces), 0)
        vespa_dlg.close()

        spec_dlg = SpectrumDialog(win.engine.traces[0], parent=win)
        for m_i in range(5):
            spec_dlg.method_combo.setCurrentIndex(m_i)
        spec_dlg.close()

        pm_dlg = ParticleMotionDialog(win.engine.traces[0], win.engine.traces[1], win.engine.traces[2], parent=win)
        pm_dlg.chk_rotated.setChecked(True)
        pm_dlg.close()

        all_picks = [pk for tr in win.engine.traces for pk in tr.picks]
        net_dlg = NetworkLocationDialog(all_picks, config=win.config, parent=win)
        net_dlg.close()

        ev_dlg = EventSummaryDialog(win.engine.current_params, all_picks, config=win.config, parent=win)
        ev_dlg.close()

        # Execute command via bottom GUI console
        win.cmd_input.setText("BUTTER BP 0.8 3.0 3")
        win._on_run_command_line()
        self.assertIn("BUTTER", win.engine.traces[0].filter_name)

        win.close()

    def test_canvas_selection_and_frame_rate_performance(self):
        import time
        from PyQt6.QtCore import QPointF

        win = SHMMainWindow(autoload_example=True)
        win.resize(1600, 900)
        win.show()
        self.app.processEvents()

        # Load 36-trace Mini-SEED dataset
        win.load_example_dataset("mseed")
        self.app.processEvents()
        self.assertEqual(len(win.engine.traces), 36)

        # 1. Interactive hover repaint must be well under 5 ms/frame (typically ~0.35 ms)
        t0 = time.perf_counter()
        n_hover = 40
        for i in range(n_hover):
            win.main_canvas._hover_pos = QPointF(250.0 + (i % 300), 150.0 + (i % 200))
            win.main_canvas.repaint()
        avg_hover_ms = (time.perf_counter() - t0) * 1000.0 / n_hover
        self.assertLess(avg_hover_ms, 5.0)

        # 2. Single-trace selection toggle via invalidate_slot must be well under 8 ms/frame (typically ~1.6 ms)
        t0 = time.perf_counter()
        n_sel = 24
        for i in range(n_sel):
            win.engine.traces[i].selected = not win.engine.traces[i].selected
            win.main_canvas.invalidate_slot(i)
            win.main_canvas.repaint()
        avg_slot_ms = (time.perf_counter() - t0) * 1000.0 / n_sel
        self.assertLess(avg_slot_ms, 8.0)

        # 3. Select all & deselect all must invalidate cache cleanly
        win.action_select_all()
        win.main_canvas.repaint()
        self.assertTrue(all(t.selected for t in win.engine.traces))
        win.action_deselect_all()
        win.main_canvas.repaint()
        self.assertFalse(any(t.selected for t in win.engine.traces))

        win.close()


if __name__ == "__main__":
    unittest.main()
