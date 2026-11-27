"""Comprehensive tests for M3: FitType.FOCAL_MECHANISM, SHCommandEngine, shc, and shm launchers."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from shm_app.__version__ import __version__
from shm_app.core.fit_models import FitType
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.main_window import SHMMainWindow
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.io.gse import write_gse2_file
from shm_app.seismology.focal_mechanism import estimate_takeoff_angle, p_wave_radiation_pattern
from shm_app.seismology.traveltime import get_ttt_manager, sphere_add


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHC_PATH = os.path.join(REPO_ROOT, "shc")
SHM_PATH = os.path.join(REPO_ROOT, "shm")


class TestM3SHCommandEngine(unittest.TestCase):
    def test_tokenizer_quotes_chained_qualifiers_and_paths(self) -> None:
        # Quoted tokens preserve internal spaces and semicolons
        toks, quals = SHCommandEngine._tokenize_command('ECHO "hello world" \'foo  bar\'')
        self.assertEqual(toks, ["ECHO", "hello world", "foo  bar"])
        self.assertEqual(quals, {})

        toks_qpath, quals_qpath = SHCommandEngine._tokenize_command('READGSE "/tmp/my file.gse"')
        self.assertEqual(toks_qpath, ["READGSE", "/tmp/my file.gse"])
        self.assertEqual(quals_qpath, {})

        # Quoted value inside qualifier=/value
        toks_qval, quals_qval = SHCommandEngine._tokenize_command('REPORT /SVG /OUT="/tmp/my dir/out.svg"')
        self.assertEqual(toks_qval, ["REPORT"])
        self.assertEqual(quals_qval, {"SVG": "TRUE", "OUT": "/tmp/my dir/out.svg"})

        # Chained qualifiers
        toks_ch, quals_ch = SHCommandEngine._tokenize_command("FILTER /REC/AUTOCUT WWSSN_SP")
        self.assertEqual(toks_ch, ["FILTER", "WWSSN_SP"])
        self.assertEqual(quals_ch, {"REC": "TRUE", "AUTOCUT": "TRUE"})

        toks_gen, quals_gen = SHCommandEngine._tokenize_command("CUSTOMCMD /QUAL1/QUAL2 arg1")
        self.assertEqual(toks_gen, ["CUSTOMCMD", "arg1"])
        self.assertEqual(quals_gen, {"QUAL1": "TRUE", "QUAL2": "TRUE"})

        # Nonexistent paths must NOT be misclassified as qualifiers
        toks_np1, quals_np1 = SHCommandEngine._tokenize_command("READGSE /nonexistent")
        self.assertEqual(toks_np1, ["READGSE", "/nonexistent"])
        self.assertEqual(quals_np1, {})

        toks_np2, quals_np2 = SHCommandEngine._tokenize_command("READGSE /nonexistent/sub/trace.gse")
        self.assertEqual(toks_np2, ["READGSE", "/nonexistent/sub/trace.gse"])
        self.assertEqual(quals_np2, {})

        toks_np3, quals_np3 = SHCommandEngine._tokenize_command("ECHO /nonexistent")
        self.assertEqual(toks_np3, ["ECHO", "/nonexistent"])
        self.assertEqual(quals_np3, {})

    def test_semicolon_in_quotes_and_boundary_safe_substitutions(self) -> None:
        emitted: list[str] = []
        engine = SHCommandEngine(output_callback=emitted.append)

        # Semicolon inside quotes must not split command
        out = engine.execute_line('ECHO "a;b"')
        self.assertEqual(out, "a;b")
        self.assertEqual(emitted, ["a;b"])

        # SDEF + $SYM boundary safety ($X vs $X1, quoted "X" literal)
        emitted.clear()
        engine.execute_line("SDEF X alpha; SDEF X1 beta; ECHO $X $X1 $UNKNOWN")
        self.assertEqual(emitted[-1], "alpha beta $UNKNOWN")

        emitted.clear()
        self.assertEqual(engine.execute_line('ECHO "X" "PI"'), "X PI")

        # #1 vs #10 positional script args
        emitted.clear()
        args_10 = [f"ARG_{i}" for i in range(1, 12)]
        logs = engine.execute_script("ECHO #1 #10 #11 #1_tail", args=args_10)
        self.assertEqual(logs, ["ARG_1 ARG_10 ARG_11 ARG_1_tail"])
        self.assertEqual(emitted, ["ARG_1 ARG_10 ARG_11 ARG_1_tail"])

    def test_all_informative_commands_emit_without_duplication(self) -> None:
        emitted: list[str] = []
        engine = SHCommandEngine(output_callback=emitted.append)

        cmds_single = [
            "SDEF MYVAR 42",
            "CALC Y = 2 + 3 * 4",
            "NORM A",
            "STW 1.0 8.0",
            "DTW",
            "TITLE Test Title",
            "FILI WWSSN_SP",
            "CREATE KUEPPER 20.0 0.05 10.0 100.0 2.0",
            "CREATE KUEPPER 20.0 0.05 10.0 100.0 2.0",
            "HIDE 2",
            "DISPLAY 2",
            "SEL 1",
            "SEL NONE",
            "ZOOM 2.0",
            "TRESET",
            "DEMEAN ALL",
            "TREND ALL",
            "DESPIKE ALL 4.0",
            "RESAMPLE ALL 0.1",
            "DECIMATE ALL 2",
            "CUT ALL 1.0 15.0",
            "SHIFT ALL 0.5",
            "INT ALL",
            "DERIVE ALL 1",
            "HILBERT ALL",
            "ENVELOPE ALL",
            "SPHASE ALL",
            "FLIP ALL",
            "MULT ALL 2.0",
            "SUM ALL",
            "PWS ALL",
            "MEND 2 4 ALL",
            "ATTENUATE 0.2 ALL",
            "SPIKING 20 0.01",
            "DECONV 0.05",
            "CORR 1 2 0.0 5.0",
        ]
        for cmd in cmds_single:
            emitted.clear()
            ret = engine.execute_line(cmd)
            self.assertTrue(ret, f"Expected non-empty output from {cmd!r}")
            self.assertEqual(len(emitted), 1, f"Expected single emit for {cmd!r}, got {emitted}")
            self.assertEqual(emitted[0], ret)

        # 3C and array commands emitting
        engine.execute_line("DEL ALL")
        for c in ("Z", "N", "E"):
            engine.execute_line("CREATE KUEPPER 20.0 0.05 8.0 100.0 1.5")
            engine.traces[-1].station = "GRA1"
            engine.traces[-1].comp = c
            engine.traces[-1].add_or_replace_pick(
                PhasePick(name="P", onset=format_sh_time(engine.traces[-1].start_epoch + 8.0), station="GRA1")
            )

        for cmd in ("ROT2 45.0", "ROT3 45.0 20.0", "UVW", "POLFIL 1.0 2.0 1", "ALIGN P", "VESPA 45.0 4.0 8.0 2.0 2"):
            emitted.clear()
            ret = engine.execute_line(cmd)
            self.assertTrue(ret)
            self.assertEqual(len(emitted), 1, f"Expected single emit for {cmd!r}, got {emitted}")
            self.assertEqual(emitted[0], ret)

        emitted.clear()
        q_ret = engine.execute_line("QUIT")
        self.assertEqual(q_ret, "Exiting SH.")
        self.assertEqual(emitted, ["Exiting SH."])

    def test_silent_arg_count_bugs_and_r_mode_heuristic(self) -> None:
        engine = SHCommandEngine()
        engine.execute_line("CREATE KUEPPER 20.0 0.05")
        engine.execute_line("CREATE KUEPPER 20.0 0.05")
        for comp in ("Z", "N", "E"):
            engine.execute_line("CREATE KUEPPER 20.0 0.05")
            engine.traces[-1].station = "TRI"
            engine.traces[-1].comp = comp

        self.assertIn("SH Error in DEMEAN", engine.execute_line("DEMEAN ALL 2.0"))
        self.assertIn("SH Error in MDIR", engine.execute_line("MDIR 3.0"))
        self.assertIn("SH Error in CORR", engine.execute_line("CORR 1 2 5.0"))

        # FILTER / ADDFIL must NOT force mode="REC" just because filter name starts with 'R'
        engine.execute_line("SEL 1")
        with patch("shm_app.interpreter.sh_engine.filter_trace", side_effect=lambda tr, *a, **kw: tr) as mock_ft:
            engine.execute_line("FILTER RESTITUTE_FOO")
            self.assertEqual(mock_ft.call_args.kwargs["mode"], "AUTO")

            mock_ft.reset_mock()
            engine.execute_line("FILTER /REC WWSSN_SP")
            self.assertEqual(mock_ft.call_args.kwargs["mode"], "REC")

            mock_ft.reset_mock()
            engine.execute_line("ADDFIL RESTITUTE_FOO blue")
            self.assertEqual(mock_ft.call_args.kwargs["mode"], "AUTO")

            mock_ft.reset_mock()
            engine.execute_line("ADDFIL /REC WWSSN_SP blue")
            self.assertEqual(mock_ft.call_args.kwargs["mode"], "REC")

    def test_surfmag_does_not_clobber_p_pick(self) -> None:
        engine = SHCommandEngine()
        dt = 0.2
        t = np.arange(1500) * dt
        env = np.exp(-0.5 * ((t - 150.0) / 35.0) ** 2)
        rayleigh = 4000.0 * env * np.sin(2.0 * math.pi * (1.0 / 20.0) * t)

        tr_z = Trace(data=rayleigh, delta=dt, station="GRA1", comp="Z", start_epoch=1700000000.0, distance=40.0)
        p_pick = PhasePick(
            name="P",
            onset=format_sh_time(1700000020.0),
            station="GRA1",
            ampl=123.4,
            period=1.25,
            mag_mb=4.9,
            mag_ms=0.0,
        )
        tr_z.add_or_replace_pick(p_pick)
        engine.traces = [tr_z]

        msg = engine.execute_line("SURFMAG")
        self.assertIn("Surface Wave Magnitude", msg)

        kept_p = tr_z.get_pick("P")
        self.assertIsNotNone(kept_p)
        self.assertAlmostEqual(kept_p.ampl, 123.4, places=3)
        self.assertAlmostEqual(kept_p.period, 1.25, places=3)
        self.assertAlmostEqual(kept_p.mag_mb, 4.9, places=3)
        self.assertEqual(kept_p.mag_ms, 0.0)

        lr_pick = tr_z.get_pick("LR")
        self.assertIsNotNone(lr_pick)
        self.assertGreater(lr_pick.ampl, 3000.0)
        self.assertGreater(lr_pick.mag_ms, 4.0)

    def test_find_zne_triplet_rejects_three_z_traces_and_fit_focal_type(self) -> None:
        engine = SHCommandEngine()
        for sta in ("GRA1", "GRA2", "GRB1"):
            engine.execute_line("CREATE KUEPPER 20.0 0.05")
            engine.traces[-1].station = sta
            engine.traces[-1].comp = "Z"

        self.assertIn("SH Error in ROT3", engine.execute_line("ROT3 45.0 20.0"))
        self.assertIn("SH Error in MDIR", engine.execute_line("MDIR 1.0 5.0"))
        self.assertIn("SH Error in POLFIL", engine.execute_line("POLFIL 1.0 2.0 1"))
        self.assertIn("SH Error in FIT", engine.execute_line("FIT /POL 1.0 5.0"))

        # FitType.FOCAL_MECHANISM contract
        self.assertEqual(FitType.FOCAL_MECHANISM.value, "focal_mechanism")
        self.assertEqual(FitType.FOCAL_MECHANISM, "focal_mechanism")

        engine.traces = []
        ev_lat, ev_lon, depth_km = 45.0, 10.0, 15.0
        engine.params.lat = ev_lat
        engine.params.lon = ev_lon
        engine.params.depth = depth_km
        ttt = get_ttt_manager(engine.config.inputs_dir)
        for idx, az_deg in enumerate(range(0, 360, 30)):
            dist_deg = 2.5 + (idx % 3) * 3.0
            st_lat, st_lon = sphere_add(ev_lat, ev_lon, dist_deg, float(az_deg))
            ih = estimate_takeoff_angle(dist_deg, depth_km=depth_km, phase="P", ttt=ttt)
            rad = p_wave_radiation_pattern(30.0, 85.0, 10.0, ih, float(az_deg))
            if abs(rad) < 0.05:
                continue
            tr = Trace(
                data=np.zeros(20),
                delta=0.05,
                station=f"FM{idx:02d}",
                comp="Z",
                lat=st_lat,
                lon=st_lon,
                start_epoch=1700000000.0,
            )
            tr.extra_headers["station_lat"] = st_lat
            tr.extra_headers["station_lon"] = st_lon
            tr.add_or_replace_pick(
                PhasePick(name="P", onset=format_sh_time(1700000010.0), sign=(1 if rad > 0 else -1))
            )
            engine.traces.append(tr)

        msg = engine.execute_line("FIT /FOCAL")
        self.assertIn("Focal Mechanism Fit", msg)
        self.assertIsNotNone(engine.last_fit_diagnostic)
        self.assertEqual(engine.last_fit_diagnostic.fit_type, FitType.FOCAL_MECHANISM)

    def test_help_topic_lookup_calc_if_and_aliases(self) -> None:
        emitted: list[str] = []
        engine = SHCommandEngine(output_callback=emitted.append)

        overview = engine.execute_line("HELP")
        self.assertIn("READM", overview)
        self.assertIn("EQI", overview)
        self.assertIn("sqrt", overview)

        calc_help = engine.execute_line("HELP CALC")
        self.assertIn("sin(x)", calc_help)
        self.assertIn("sqrt(x)", calc_help)

        if_help = engine.execute_line("HELP IF")
        self.assertIn("EQI", if_help)
        self.assertIn("EQS", if_help)

        alias_help = engine.execute_line("? READM")
        self.assertIn("READM -> READS", alias_help)

        missing_help = engine.execute_line("HELP NONEXISTENT_TOPIC")
        self.assertIn("No help topic found", missing_help)


class TestM3SHCLauncherSubprocess(unittest.TestCase):
    def _run_shc(self, args: list[str], stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, SHC_PATH, *args],
            input=stdin_text,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=30,
        )

    def test_shc_version_and_silent_commands_output(self) -> None:
        ver = self._run_shc(["--version"])
        self.assertEqual(ver.returncode, 0)
        self.assertIn(__version__, ver.stdout)

        ok = self._run_shc(["-c", "CREATE KUEPPER 10 0.05; DEMEAN ALL; CALC X = 2 + 2; ZOOM 2.0"])
        self.assertEqual(ok.returncode, 0)
        self.assertIn("Created synthetic KUEPPER trace", ok.stdout)
        self.assertIn("Demeaned 1 trace(s).", ok.stdout)
        self.assertIn("X = 4", ok.stdout)
        self.assertIn("Global zoom = 2", ok.stdout)
        self.assertEqual(ok.stderr, "")

    def test_shc_error_exit_codes_and_conflicting_args(self) -> None:
        err = self._run_shc(["-c", "UNKNOWN_CMD"])
        self.assertEqual(err.returncode, 1)
        self.assertIn("SH Error in UNKNOWN_CMD", err.stderr)

        partial_err = self._run_shc(["-c", "CREATE KUEPPER 10 0.05; UNKNOWN_CMD"])
        self.assertEqual(partial_err.returncode, 1)
        self.assertIn("Created synthetic KUEPPER trace", partial_err.stdout)
        self.assertIn("SH Error in UNKNOWN_CMD", partial_err.stderr)

        with tempfile.TemporaryDirectory() as td:
            script_ok = os.path.join(td, "ok.shc")
            with open(script_ok, "w", encoding="utf-8") as fh:
                fh.write("ECHO #1 #10\nCALC Z = 3 + 4\n")
            s_ok = self._run_shc([script_ok, *[f"P{i}" for i in range(1, 11)]])
            self.assertEqual(s_ok.returncode, 0)
            self.assertIn("P1 P10", s_ok.stdout)
            self.assertIn("Z = 7", s_ok.stdout)

            # Reject combining -c and script explicitly
            conflict = self._run_shc(["-c", "ECHO hi", script_ok])
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("Cannot specify both -c/--command and a script file", conflict.stderr)

            script_bad = os.path.join(td, "bad.shc")
            with open(script_bad, "w", encoding="utf-8") as fh:
                fh.write("ECHO before\nBAD_VERB\n")
            s_bad = self._run_shc([script_bad])
            self.assertEqual(s_bad.returncode, 1)
            self.assertIn("before", s_bad.stdout)
            self.assertIn("SH Error in BAD_VERB", s_bad.stderr)

    def test_shc_non_tty_stdin_suppresses_prompt(self) -> None:
        piped_ok = self._run_shc([], stdin_text='CREATE KUEPPER 5 0.05\nECHO "a;b"\nQUIT\n')
        self.assertEqual(piped_ok.returncode, 0)
        self.assertNotIn("SH>", piped_ok.stdout)
        self.assertNotIn("Type HELP", piped_ok.stdout)
        self.assertIn("a;b", piped_ok.stdout)

        piped_err = self._run_shc([], stdin_text="UNKNOWN_PIPED_CMD\n")
        self.assertEqual(piped_err.returncode, 1)
        self.assertNotIn("SH>", piped_err.stdout)
        self.assertIn("SH Error in UNKNOWN_PIPED_CMD", piped_err.stderr)


class TestM3SHMLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _run_shm(self, args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, SHM_PATH, *args],
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=30,
        )

    def test_shm_version_and_missing_file_nonzero_exit(self) -> None:
        ver = self._run_shm(["--version"])
        self.assertEqual(ver.returncode, 0)
        self.assertIn(__version__, ver.stdout)

        with tempfile.TemporaryDirectory() as td:
            sess_path = os.path.join(td, "session.json")
            missing = self._run_shm(
                ["-platform", "offscreen", os.path.join(td, "does not exist.gse")],
                extra_env={"SHM_SESSION_PATH": sess_path},
            )
            self.assertEqual(missing.returncode, 1)
            self.assertIn("Input path not found", missing.stderr)

    def test_shm_parse_known_args_and_load_spaces_qml_sxml_and_unfilter(self) -> None:
        import importlib.machinery
        import importlib.util

        loader = importlib.machinery.SourceFileLoader("shm_launcher_mod", SHM_PATH)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.assertIsNotNone(spec)
        shm_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(shm_mod)

        args, qt_argv = shm_mod.parse_cli_args(["-platform", "offscreen", "--example", "/tmp/my file.gse"])
        self.assertTrue(args.example)
        self.assertEqual(args.files, ["/tmp/my file.gse"])
        self.assertIn("-platform", qt_argv)
        self.assertIn("offscreen", qt_argv)

        with tempfile.TemporaryDirectory() as td:
            sess_path = os.path.join(td, "session.json")
            gse_path = os.path.join(td, "my trace file.gse")
            sxml_path = os.path.join(td, "my stations.xml")
            qml_path = os.path.join(td, "my event.xml")

            tr = Trace(
                data=np.sin(2.0 * math.pi * 1.0 * (np.arange(400) * 0.05)) * 200.0,
                delta=0.05,
                station="SYNT",
                comp="Z",
                start_epoch=1700000000.0,
            )
            write_gse2_file(gse_path, [tr])

            with open(sxml_path, "w", encoding="utf-8") as fh:
                fh.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.1">
  <Source>Test</Source><Created>2024-01-01T00:00:00Z</Created>
  <Network code="GR">
    <Station code="SYNT" startDate="2020-01-01T00:00:00Z">
      <Latitude>49.6919</Latitude><Longitude>11.2217</Longitude><Elevation>500.0</Elevation>
      <Site><Name>Synt</Name></Site>
      <Channel code="BHZ" locationCode="" startDate="2020-01-01T00:00:00Z">
        <Latitude>49.6919</Latitude><Longitude>11.2217</Longitude><Elevation>500.0</Elevation>
        <Depth>0.0</Depth><SampleRate>20.0</SampleRate>
      </Channel>
    </Station>
  </Network>
</FDSNStationXML>"""
                )

            with open(qml_path, "w", encoding="utf-8") as fh:
                fh.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2">
  <eventParameters publicID="smi:local/ep">
    <event publicID="smi:local/ev">
      <preferredOriginID>smi:local/orig</preferredOriginID>
      <origin publicID="smi:local/orig">
        <time><value>2023-11-14T22:13:20.000000Z</value></time>
        <latitude><value>41.25</value></latitude>
        <longitude><value>19.75</value></longitude>
        <depth><value>12000.0</value></depth>
      </origin>
    </event>
  </eventParameters>
</q:quakeml>"""
                )

            win = SHMMainWindow(session_path=sess_path, prompt_session_recovery=False)
            errors = shm_mod.load_startup_files(win, [gse_path, sxml_path, qml_path])
            self.assertEqual(errors, [])
            self.assertEqual(len(win.engine.traces), 1)
            self.assertIn("SYNT", win.config.stations)
            self.assertAlmostEqual(win.engine.params.lat, 41.25, places=2)
            self.assertGreater(len(win._unfiltered_by_uid), 0)

            # Filter and verify Ctrl+U (action_unfilter) restores raw trace loaded from CLI
            raw_before = win.engine.traces[0].data.copy()
            win.engine.execute_line("BUTTER BP 0.5 2.0 3")
            self.assertFalse(np.allclose(win.engine.traces[0].data, raw_before))
            win.action_unfilter()
            np.testing.assert_allclose(win.engine.traces[0].data, raw_before, atol=1e-6)
            win.close()

    def test_shm_main_subprocess_with_platform_offscreen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sess_path = os.path.join(td, "session.json")
            gse_path = os.path.join(td, "cli space.gse")
            tr = Trace(data=np.ones(200), delta=0.05, station="GRA1", comp="Z", start_epoch=1700000000.0)
            write_gse2_file(gse_path, [tr])

            driver = (
                "import importlib.machinery, importlib.util, sys\n"
                "from PyQt6.QtCore import QTimer\n"
                f"loader = importlib.machinery.SourceFileLoader('shm_mod', {SHM_PATH!r})\n"
                "spec = importlib.util.spec_from_loader(loader.name, loader)\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(mod)\n"
                "orig_show = mod.SHMMainWindow.show\n"
                "def _auto_close(self_win):\n"
                "    orig_show(self_win)\n"
                "    assert len(self_win.engine.traces) == 1\n"
                "    assert len(self_win._unfiltered_by_uid) == 1\n"
                "    QTimer.singleShot(100, self_win.close)\n"
                "mod.SHMMainWindow.show = _auto_close\n"
                f"sys.exit(mod.main(['-platform', 'offscreen', {gse_path!r}]))\n"
            )
            env = dict(os.environ)
            env.pop("QT_QPA_PLATFORM", None)
            env["SHM_SESSION_PATH"] = sess_path
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=env,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
