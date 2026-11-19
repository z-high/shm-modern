"""Headless end-to-end integration tests for MainWindow Phase 3 & Phase 4 features (M-12..M-25)."""

from __future__ import annotations

import ast
import json
import math
import os
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QMessageBox, QPushButton

app = QApplication.instance()
if app is None:
    app = QApplication(["shm_p3_p4_main_test"])

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import FitModificationRequest, FitType
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.session import has_unclean_session
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.dialogs_analysis import FilterDialog, FKAnalysisDialog, ParticleMotionDialog, SpectrumDialog
from shm_app.gui.dialogs_io import SeedLinkStreamDialog
from shm_app.gui.main_window import SHMMainWindow
from shm_app.io.mseed import compress_steim1_record, read_mseed_bytes
from shm_app.processing.spectrum import source_spectrum_model
from shm_app.seismology.focal_mechanism import estimate_takeoff_angle, p_wave_radiation_pattern
from shm_app.seismology.traveltime import get_ttt_manager, sphere_add, sphere_diff


SAMPLE_STATIONXML = """<?xml version="1.0" encoding="UTF-8"?>
<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.1">
  <Source>SHM-Test</Source>
  <Created>2024-01-01T00:00:00Z</Created>
  <Network code="XX">
    <Station code="TST1">
      <Latitude>48.1234</Latitude>
      <Longitude>11.5678</Longitude>
      <Elevation>525.0</Elevation>
      <Site><Name>Test Station 1</Name></Site>
      <Channel code="BHN" locationCode="00" startDate="2020-01-01T00:00:00Z">
        <Latitude>48.1235</Latitude>
        <Longitude>11.5679</Longitude>
        <Elevation>525.0</Elevation>
        <Depth>12.0</Depth>
        <Azimuth>3.5</Azimuth>
        <Dip>0.0</Dip>
        <SampleRate>20.0</SampleRate>
        <Sensor><Description>STS-2/Test</Description></Sensor>
        <Response>
          <InstrumentSensitivity>
            <Value>5.0e8</Value>
            <Frequency>1.0</Frequency>
            <InputUnits><Name>M/S</Name></InputUnits>
            <OutputUnits><Name>COUNTS</Name></OutputUnits>
          </InstrumentSensitivity>
        </Response>
      </Channel>
    </Station>
  </Network>
</FDSNStationXML>
"""


class _MinimalLoopbackSeedLinkServer:
    """Tiny SeedLink v3.1 TCP loopback server emitting one 512-byte Steim-1 Mini-SEED packet."""

    def __init__(self, packet_mseed: bytes) -> None:
        self.packet_mseed = packet_mseed
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(2)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(0.2)
                buf = b""
                while not self._stop.is_set():
                    try:
                        chunk = conn.recv(1024)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        cmd = line.decode("ascii", errors="ignore").strip().upper()
                        if cmd == "HELLO":
                            conn.sendall(b"SeedLink v3.1 (test)\r\nSHM-Test\r\n")
                        elif cmd.startswith(("STATION", "SELECT", "DATA", "TIME")):
                            conn.sendall(b"OK\r\n")
                        elif cmd == "END":
                            conn.sendall(b"SL000001" + self.packet_mseed)
                            conn.sendall(b"END")
                            return

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


def _find_action_by_text(win: SHMMainWindow, needle: str):
    low = needle.lower()
    for act, _title, _key, _sc in win._help_actions:
        if low in act.text().replace("&", "").lower():
            return act
    return None


class TestMainWindowPhase3And4Integration(unittest.TestCase):
    """End-to-end headless tests for Wave 2B MainWindow integrations."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.session_file = os.path.join(self.tmpdir.name, ".shm_session.json")
        self.win = SHMMainWindow(
            autoload_example=False,
            session_path=self.session_file,
            prompt_session_recovery=False,
        )
        self.win.resize(1380, 860)

    def tearDown(self) -> None:
        for d in list(self.win._open_dialogs):
            d.close()
        self.win.close()
        app.processEvents()
        self.tmpdir.cleanup()

    def test_01_menus_actions_and_seedlink_dialog_quality(self) -> None:
        expected_labels = [
            "Import StationXML...",
            "Import QuakeML 1.2 (.xml)...",
            "Restore Last Session",
            "QuakeML 1.2 (.xml)...",
            "GeoJSON RFC 7946 (.geojson)...",
            "Observatory Bulletin (PDF/SVG)...",
            "Show Pick Uncertainty Error Bars",
            "Standard Crosshair",
            "Hilbert 90° Phase Shift Cursor",
            "Reference Waveform Cursor",
            "Capture Reference Window",
            "Trace Overlay Filter (colour_addfil)...",
            "Clear Overlay Filters on Selected",
            "Identify Phases (IASP91)...",
            "Solve Focal Mechanism (Beachball)",
            "Scan Surface Waves (18-22s Ms)",
            "SeedLink Stream...",
            "Stop SeedLink Stream",
        ]
        for label in expected_labels:
            self.assertIsNotNone(_find_action_by_text(self.win, label), f"Missing menu action: {label}")

        # SeedLinkStreamDialog usability guardrails
        sl_dlg = SeedLinkStreamDialog(self.win.live_coordinator, parent=self.win)
        sl_dlg.show()
        app.processEvents()
        try:
            auto_btns = [b.text() for b in sl_dlg.findChildren(QPushButton) if b.autoDefault()]
            self.assertEqual(auto_btns, [])
            for sb in sl_dlg.findChildren(QDoubleSpinBox):
                self.assertIn(".", sb.textFromValue(1.5))
        finally:
            sl_dlg.close()

    def test_02_m12_stationxml_import_and_mseed_enrichment(self) -> None:
        sxml_path = os.path.join(self.tmpdir.name, "test_station.xml")
        with open(sxml_path, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE_STATIONXML)

        # Build a 512-byte Mini-SEED record for XX.TST1.00.BHN
        t0 = 1700000000.0
        counts = np.round(np.sin(np.linspace(0, 6 * np.pi, 100)) * 500.0).astype(np.int32)
        rec_bytes, n_packed = compress_steim1_record(
            counts,
            station="TST1",
            channel="BHN",
            network="XX",
            locid="00",
            start_epoch=t0,
            dt=0.05,
            seqno=1,
            reclth=512,
        )
        self.assertGreater(n_packed, 0)

        cfg = SHMConfig(SHM_DEFAULT_ROOT)
        cfg.load_stationxml(sxml_path)
        traces = read_mseed_bytes(rec_bytes, config=cfg)
        self.assertEqual(len(traces), 1)
        tr = traces[0]
        self.assertAlmostEqual(tr.lat, 48.1235, places=4)
        self.assertAlmostEqual(tr.lon, 11.5679, places=4)
        self.assertAlmostEqual(tr.calib, 1e9 / 5.0e8, places=4)
        self.assertAlmostEqual(tr.extra_headers["azimuth"], 3.5, places=3)
        self.assertAlmostEqual(tr.extra_headers["dip"], 0.0, places=3)
        self.assertAlmostEqual(tr.extra_headers["station_elev"], 525.0, places=2)

        # Also test MainWindow.action_import_stationxml updating existing workspace trace
        tr_plain = Trace(data=np.ones(50), delta=0.05, station="TST1", comp="N", chan1="B", chan2="H", network="XX", locid="00", start_epoch=t0)
        self.win.engine.traces = [tr_plain]
        self.win.action_import_stationxml(sxml_path)
        self.assertAlmostEqual(tr_plain.lat, 48.1235, places=4)
        self.assertAlmostEqual(tr_plain.lon, 11.5679, places=4)
        self.assertAlmostEqual(tr_plain.extra_headers["azimuth"], 3.5, places=3)

    def test_03_m13_m14_m24_canvas_toggles_autocut_and_overlay_filter(self) -> None:
        dt = 0.05
        t = np.arange(800) * dt
        t0 = 1000000.0
        sig0 = 200.0 * np.sin(2.0 * np.pi * 1.2 * t)
        sig1 = 200.0 * np.cos(2.0 * np.pi * 1.2 * t)
        tr0 = Trace(data=sig0.copy(), delta=dt, station="GRA1", comp="Z", start_epoch=t0, filter_name="GLOBAL_KEEP")
        tr1 = Trace(data=sig1.copy(), delta=dt, station="GRA2", comp="Z", start_epoch=t0, filter_name="GLOBAL_KEEP")
        pk = PhasePick(name="P", onset=format_sh_time(t0 + 10.0), station="GRA1", quality=4)
        tr0.add_or_replace_pick(pk)

        self.win.engine.traces = [tr0, tr1]
        self.win._register_unfiltered_traces(self.win.engine.traces, replace=True, overwrite_existing=True)
        self.win.refresh_views(reset_time=True)

        # M-13 pick error bar toggle & pickChanged signal wiring
        self.win.act_pick_error_bars.setChecked(False)
        self.assertFalse(self.win.main_canvas.show_pick_error_bars)
        self.assertFalse(self.win.mag_canvas.show_pick_error_bars)
        self.win.act_pick_error_bars.setChecked(True)
        self.assertTrue(self.win.main_canvas.show_pick_error_bars)

        pk.set_quality(2)
        self.win.main_canvas.pickChanged.emit(0, pk)
        self.assertIn("Q=2", self.win.status_bar.currentMessage())

        # M-14 cursor mode toggles + capture reference window
        self.win.action_set_mag_cursor_mode("HILBERT")
        self.assertEqual(self.win.mag_canvas.cursor_overlay_mode, "HILBERT")
        self.assertTrue(self.win.act_mag_cursor_hilbert.isChecked())
        self.win.action_capture_reference_window()
        self.assertIsNotNone(self.win.mag_canvas._reference_waveform)

        # M-24 trace-specific overlay filter (colour_addfil) on selected trace 0 only
        tr0.selected = True
        tr1.selected = False
        n_mod = self.win.apply_overlay_filter("BUTTER BP 0.8 2.5 3", color="blue")
        self.assertEqual(n_mod, 1)
        self.assertEqual(self.win.engine.traces[0].filter_name, "GLOBAL_KEEP")
        self.assertEqual(self.win.engine.traces[0].overlay_filter_name, "BUTTER BP 0.8 2.5 3")
        self.assertEqual(self.win.engine.traces[0].overlay_filter_color, "blue")
        self.assertEqual(self.win.engine.traces[1].overlay_filter_name, "NONE")
        np.testing.assert_allclose(self.win.engine.traces[1].data, sig1)

        # Undo restores overlay filter state
        self.win.action_undo()
        self.assertEqual(self.win.engine.traces[0].overlay_filter_name, "NONE")

        # Clear overlay filter action
        self.win.apply_overlay_filter("BUTTER BP 0.8 2.5 3", color="red", trace_indices=[0])
        self.win.action_clear_overlay_filters()
        self.assertEqual(self.win.engine.traces[0].overlay_filter_name, "NONE")
        self.assertEqual(self.win.engine.traces[0].overlay_filter_color, "")

        # FilterDialog autocut & zero_phase wiring
        with patch.object(FilterDialog, "exec", return_value=QDialog.DialogCode.Accepted), patch.object(
            FilterDialog,
            "get_filter_options",
            return_value={
                "filter_expr": "BUTTER BP 0.7 2.5 4",
                "mode": "REC",
                "use_lpc": False,
                "demean_first": True,
                "autocut": True,
                "zero_phase": True,
            },
        ):
            len_before = self.win.engine.traces[0].length
            self.win.action_open_filter_dialog()
            self.assertLess(self.win.engine.traces[0].length, len_before)
            self.assertIn("BUTTER", self.win.engine.traces[0].filter_name)

    def test_04_bridge_routing_highlight_and_recompute_dispatch(self) -> None:
        # Prepare a synthetic Brune waveform + 3C Z/N/E traces + array traces
        dt = 0.05
        n_sig = 512
        freqs = np.fft.rfftfreq(n_sig, d=dt)
        amp = np.zeros_like(freqs)
        pos = freqs > 0.0
        amp[pos] = source_spectrum_model(freqs[pos], omega0=800.0, fc=1.5, t_star=0.03, model="brune")
        pulse = np.fft.irfft(amp * np.exp(-1j * 2.0 * np.pi * freqs * 5.0), n=n_sig) / dt
        t0 = 1700000000.0

        tr_z = Trace(data=pulse.copy(), delta=dt, station="GRA1", comp="Z", start_epoch=t0, distance=1.0)
        tr_n = Trace(data=-0.5 * pulse.copy(), delta=dt, station="GRA1", comp="N", start_epoch=t0, distance=1.0)
        tr_e = Trace(data=0.4 * pulse.copy(), delta=dt, station="GRA1", comp="E", start_epoch=t0, distance=1.0)
        tr_other = Trace(data=pulse.copy(), delta=dt, station="GRA2", comp="Z", start_epoch=t0, distance=1.0)
        self.win.engine.traces = [tr_z, tr_n, tr_e, tr_other]
        self.win.refresh_views(reset_time=True)

        # 1. Highlight trace sets tr.selected = True on matching station
        self.win._on_highlight_fit_trace("GRA2")
        self.assertFalse(self.win.engine.traces[0].selected)
        self.assertTrue(self.win.engine.traces[3].selected)
        self.assertEqual(self.win.engine.selected_trace_indices, {3})

        # 2. SpectrumDialog.fitDiagnosticReady -> FitDiagnosticWindow
        self.win.main_canvas.mag_trace_idx = 0
        self.win.main_canvas.mag_t_lo = 0.5
        self.win.main_canvas.mag_t_hi = 25.0
        self.win.action_open_spectrum_dialog()
        spec_dlg = self.win._find_open_dialog(SpectrumDialog)
        self.assertIsNotNone(spec_dlg)
        spec_dlg.run_brune_inversion(sync=True)
        app.processEvents()
        self.assertIsNotNone(self.win.fit_dock.current_result)
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.BRUNE_SPECTRUM)
        spec_dlg.close()

        # 3. ParticleMotionDialog.fitDiagnosticReady -> FitDiagnosticWindow
        self.win.action_open_pmotion_dialog()
        pm_dlg = self.win._find_open_dialog(ParticleMotionDialog)
        self.assertIsNotNone(pm_dlg)
        app.processEvents()
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.POLARIZATION_3C)
        pm_dlg.close()

        # 4. FKAnalysisDialog.fitDiagnosticReady -> FitDiagnosticWindow
        self.win.action_open_fk_dialog()
        fk_dlg = self.win._find_open_dialog(FKAnalysisDialog)
        self.assertIsNotNone(fk_dlg)
        fk_dlg.run_fk(sync=True)
        app.processEvents()
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.FK_SEMBLANCE)
        fk_dlg.close()

        # 5. CLI callback -> FitDiagnosticWindow
        self.win.engine.execute_line("FIT /BRUNE 1 0.5 25.0")
        app.processEvents()
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.BRUNE_SPECTRUM)

        # 6. _on_recompute_fit_requested domain dispatch (async QThread for Brune & FK/ARF)
        self.win.fit_dock.current_result = None
        self.win._on_recompute_fit_requested(FitModificationRequest(fit_type=FitType.BRUNE_SPECTRUM))
        self.assertGreaterEqual(len(self.win._recompute_workers), 1)
        self.assertTrue(self.win._recompute_workers[-1].wait(15000))
        app.processEvents()
        self.assertIsNotNone(self.win.fit_dock.current_result)
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.BRUNE_SPECTRUM)

        self.win.fit_dock.current_result = None
        self.win._on_recompute_fit_requested(FitModificationRequest(fit_type=FitType.FK_SEMBLANCE))
        self.assertGreaterEqual(len(self.win._recompute_workers), 1)
        self.assertTrue(self.win._recompute_workers[-1].wait(15000))
        app.processEvents()
        self.assertIsNotNone(self.win.fit_dock.current_result)
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.FK_SEMBLANCE)
        self.assertEqual(self.win.fit_dock.current_result.overlay_curves["arf_grid"].shape, (51, 51))

        self.win._on_recompute_fit_requested(FitModificationRequest(fit_type=FitType.POLARIZATION_3C))
        self.assertEqual(self.win.fit_dock.current_result.fit_type, FitType.POLARIZATION_3C)
        self.win._on_recompute_fit_requested(FitModificationRequest(fit_type=FitType.RESTITUTION_FIT))

    def test_05_m18_m22_m25_and_m19_m20_exports(self) -> None:
        cfg = get_config()
        ttt = get_ttt_manager(cfg.inputs_dir)
        ev_lat, ev_lon, depth_km = 45.0, 10.0, 15.0
        t0 = 1700000000.0
        true_s, true_d, true_r = 35.0, 80.0, 15.0

        self.win.engine.params = AnalysisParamSet(
            lat=ev_lat,
            lon=ev_lon,
            depth=depth_km,
            origin=format_sh_time(t0),
            distance=5.0,
            mag_mb=5.4,
            region_name="NORTHERN ITALY",
        )

        dt = 0.05
        t = np.arange(2000) * dt
        sw_wave = 8000.0 * np.sin(2.0 * np.pi / 20.0 * t) * np.exp(-0.5 * ((t - 50.0) / 15.0) ** 2)

        traces = []
        for idx, az_deg in enumerate(range(0, 360, 30)):
            dist_deg = 3.0 + (idx % 3) * 3.0
            st_lat, st_lon = sphere_add(ev_lat, ev_lon, dist_deg, float(az_deg))
            ih = estimate_takeoff_angle(dist_deg, depth_km=depth_km, phase="P", ttt=ttt)
            rad = p_wave_radiation_pattern(true_s, true_d, true_r, ih, float(az_deg))
            pol = 1 if rad >= 0.0 else -1
            tt_s = ttt.travel_time("S", dist_deg, depth_km) or 80.0

            sta_code = f"FM{idx:02d}"
            tr = Trace(
                data=sw_wave.copy(),
                delta=dt,
                station=sta_code,
                comp="Z",
                network="XX",
                start_epoch=t0,
                distance=dist_deg,
                lat=st_lat,
                lon=st_lon,
                extra_headers={"station_lat": st_lat, "station_lon": st_lon},
            )
            tr.add_or_replace_pick(
                PhasePick(
                    name="UNK",
                    onset=format_sh_time(t0 + tt_s),
                    sign=pol,
                    weight=2,
                    station=sta_code,
                    comp="Z",
                )
            )
            traces.append(tr)

        self.win.engine.traces = traces
        self.win.refresh_views(reset_time=True)

        # M-18: Identify Phases (IASP91)
        self.win.action_identify_phases()
        self.assertTrue(any(pk.name == "S" for tr in self.win.engine.traces for pk in tr.picks))

        # M-22: Scan Surface Waves (18-22s Ms)
        self.win.action_scan_surface_waves()
        self.assertGreater(self.win.engine.current_params.mag_ms, 0.0)

        # M-25: Solve Focal Mechanism (Beachball) -> Tab 5
        fm = self.win.action_solve_focal_mechanism()
        self.assertIsNotNone(fm)
        self.assertEqual(self.win.fit_dock.tabs.currentIndex(), 4)
        self.assertIsNotNone(self.win.fit_dock._last_fm_result)

        # M-19: QuakeML 1.2 export + import & GeoJSON export
        qml_path = os.path.join(self.tmpdir.name, "event.quakeml.xml")
        geojson_path = os.path.join(self.tmpdir.name, "event.geojson")
        self.win.action_export_quakeml(qml_path)
        self.win.action_export_geojson(geojson_path)
        self.assertTrue(os.path.isfile(qml_path))
        self.assertTrue(os.path.isfile(geojson_path))

        with open(geojson_path, "r", encoding="utf-8") as gjf:
            gj = json.load(gjf)
        self.assertEqual(gj["type"], "FeatureCollection")
        self.assertGreaterEqual(len(gj["features"]), 2)

        # Reset params and import QuakeML back
        self.win.engine.param_sets[self.win.engine.active_set_index] = AnalysisParamSet()
        self.win.action_import_quakeml(qml_path)
        self.assertAlmostEqual(self.win.engine.current_params.lat, ev_lat, places=3)
        self.assertAlmostEqual(self.win.engine.current_params.lon, ev_lon, places=3)

        # M-20: Vector PDF & SVG Observatory Bulletin via BulletinWorker QThread
        pdf_path = os.path.join(self.tmpdir.name, "bulletin.pdf")
        svg_path = os.path.join(self.tmpdir.name, "bulletin.svg")
        w_pdf = self.win.action_export_bulletin(pdf_path, fmt="pdf")
        self.assertIsNotNone(w_pdf)
        self.assertTrue(w_pdf.wait(15000))
        app.processEvents()
        with open(pdf_path, "rb") as pf:
            self.assertTrue(pf.read(4) == b"%PDF")

        w_svg = self.win.action_export_bulletin(svg_path, fmt="svg")
        self.assertIsNotNone(w_svg)
        self.assertTrue(w_svg.wait(15000))
        app.processEvents()
        with open(svg_path, "r", encoding="utf-8") as sf:
            self.assertIn("<svg", sf.read(1024))

    def test_06_m21_seedlink_streaming_and_clean_shutdown(self) -> None:
        t0 = 1700000000.0
        counts = np.arange(-50, 50, dtype=np.int32) * 10
        pkt_mseed, n_packed = compress_steim1_record(
            counts,
            station="GRA1",
            channel="BHZ",
            network="GR",
            locid="00",
            start_epoch=t0,
            dt=0.05,
            seqno=1,
            reclth=512,
        )
        self.assertEqual(n_packed, counts.size)
        srv = _MinimalLoopbackSeedLinkServer(pkt_mseed)
        try:
            self.win.start_seedlink_stream(
                host="127.0.0.1",
                port=srv.port,
                streams=["GR_GRA1:00BHZ"],
                timeout=1.5,
                auto_reconnect=False,
            )
            # Wait briefly for SeedLinkStreamWorker to receive and emit tracesUpdated
            for _ in range(40):
                app.processEvents()
                if any(tr.extra_headers.get("seedlink_stream") for tr in self.win.engine.traces):
                    break
                threading.Event().wait(0.05)

            sl_traces = [tr for tr in self.win.engine.traces if tr.extra_headers.get("seedlink_stream")]
            self.assertGreaterEqual(len(sl_traces), 1)
            self.assertEqual(sl_traces[0].station, "GRA1")
            self.assertEqual(sl_traces[0].length, counts.size)

            self.win.close()
            self.assertFalse(self.win.live_coordinator.is_seedlink_running)
        finally:
            srv.close()

    def test_07_m23_session_autosave_crash_recovery_and_clean_exit(self) -> None:
        t0 = 1700000000.0
        tr = Trace(data=np.linspace(0.0, 10.0, 120), delta=0.05, station="BFO", comp="Z", start_epoch=t0)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0 + 2.5), station="BFO", quality=2))
        self.win.engine.traces = [tr]
        self.win.engine.current_params.lat = 48.33
        self.win.engine.current_params.lon = 8.33

        # Simulate periodic autosave (clean_exit=False -> unclean crash state)
        saved_path = self.win.autosave_session(clean_exit=False)
        self.assertEqual(saved_path, self.session_file)
        self.assertTrue(has_unclean_session(self.session_file))

        # Startup recovery prompt restores state when user chooses Yes
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes) as mock_q:
            win2 = SHMMainWindow(autoload_example=False, session_path=self.session_file)
            try:
                mock_q.assert_called_once()
                self.assertEqual(len(win2.engine.traces), 1)
                self.assertEqual(win2.engine.traces[0].station, "BFO")
                self.assertEqual(len(win2.engine.traces[0].picks), 1)
                self.assertAlmostEqual(win2.engine.current_params.lat, 48.33, places=2)
            finally:
                win2.close()

        # Clean exit marks clean_exit=True so next startup does NOT prompt
        self.assertFalse(has_unclean_session(self.session_file))
        with patch.object(QMessageBox, "question") as mock_q2:
            win3 = SHMMainWindow(autoload_example=False, session_path=self.session_file)
            try:
                mock_q2.assert_not_called()
            finally:
                win3.close()

        # Empty / schema-invalid session payloads must NOT claim unclean shutdown
        for bad_payload in ("{}", '{"clean_exit": false}', '{"version": 1, "traces": "not_a_list"}'):
            with open(self.session_file, "w", encoding="utf-8") as fh:
                fh.write(bad_payload)
            self.assertFalse(has_unclean_session(self.session_file))

    def test_08_architectural_import_invariants(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mseed_path = os.path.join(repo_root, "shm_app", "io", "mseed.py")
        session_path = os.path.join(repo_root, "shm_app", "core", "session.py")
        gui_files = [
            os.path.join(repo_root, "shm_app", "gui", "main_window.py"),
            os.path.join(repo_root, "shm_app", "gui", "main_window_menus.py"),
            os.path.join(repo_root, "shm_app", "gui", "dialogs_io.py"),
            os.path.join(repo_root, "shm_app", "gui", "help_texts.py"),
        ]

        def _imports_in(path: str) -> set[str]:
            with open(path, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            mods: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mods.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods.add(node.module.split(".")[0])
            return mods

        mseed_mods = _imports_in(mseed_path)
        self.assertTrue(mseed_mods.isdisjoint({"PyQt6", "scipy", "obspy", "reportlab", "lxml"}))
        session_mods = _imports_in(session_path)
        self.assertTrue(session_mods.isdisjoint({"PyQt6", "scipy", "obspy", "reportlab", "lxml"}))
        for gf in gui_files:
            gmods = _imports_in(gf)
            self.assertTrue(gmods.isdisjoint({"scipy", "obspy", "reportlab", "lxml"}), f"Forbidden import in {gf}: {gmods}")


if __name__ == "__main__":
    unittest.main()
