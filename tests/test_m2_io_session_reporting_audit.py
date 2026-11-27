"""Comprehensive regression and audit tests for Milestone M2 (I/O, Session, Reporting, Live Plugins)."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/shm_mpl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtWidgets import QApplication

from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.session import (
    SessionManager,
    atomic_write_json,
    deserialize_trace,
    resolve_session_path,
    serialize_trace,
)
from shm_app.core.time_utils import parse_sh_time
from shm_app.gui.reporting import (
    render_beachball,
    render_header_and_metadata,
    render_waveform_panels,
)
from shm_app.io.evt import read_evt_file, write_evt_file
from shm_app.io.geojson import export_geojson, write_geojson
from shm_app.io.gse import read_gse_file
from shm_app.io.mseed import (
    compress_steim1_record,
    decompress_steim,
    read_mseed_bytes,
    read_mseed_file,
    write_mseed_file,
)
from shm_app.io.qfile import read_qfile, write_qfile
from shm_app.io.quakeml import BED_NS, export_quakeml, parse_quakeml
from shm_app.io.seedlink import (
    SeedLinkClient,
    SeedLinkStreamSpec,
    WaveformRingBuffer,
)
from shm_app.io.stationxml import (
    StationXMLChannel,
    StationXMLCoefficients,
    StationXMLPolesZeros,
    StationXMLResponseStage,
    StationXMLSensitivity,
    eval_stationxml_response,
)
from shm_app.plugins.bgr_fetch import StationFetchTarget
from shm_app.plugins.live_events import EventDeduplicator, LiveEvent, parse_usgs_geojson
from shm_app.plugins.live_planner import LiveConfig
from shm_app.plugins.live_service import LiveEventCoordinator


class TestMseedAuditFixes(unittest.TestCase):
    def test_read_mseed_bytes_splits_on_large_gap_ge_3600s(self) -> None:
        t0 = parse_sh_time("2024-06-01T10:00:00")
        s1 = np.arange(100, dtype=np.int32)
        s2 = np.arange(200, 300, dtype=np.int32)
        r1, n1 = compress_steim1_record(s1, "GRA1", "BHZ", network="GR", start_epoch=t0, dt=0.05, seqno=1, reclth=512)
        r2, n2 = compress_steim1_record(s2, "GRA1", "BHZ", network="GR", start_epoch=t0 + 4000.0, dt=0.05, seqno=2, reclth=512)
        self.assertEqual(n1, 100)
        self.assertEqual(n2, 100)

        trs = read_mseed_bytes(r1 + r2, calib_override=1.0)
        self.assertEqual(len(trs), 2)
        self.assertEqual(trs[0].length, 100)
        self.assertEqual(trs[1].length, 100)
        np.testing.assert_allclose(trs[0].data, s1.astype(np.float64))
        np.testing.assert_allclose(trs[1].data, s2.astype(np.float64))
        self.assertAlmostEqual(trs[1].start_epoch - trs[0].start_epoch, 4000.0, places=3)

    def test_read_mseed_bytes_splits_on_sample_rate_change(self) -> None:
        t0 = parse_sh_time("2024-06-01T10:00:00")
        s1 = np.ones(50, dtype=np.int32) * 10
        s2 = np.ones(50, dtype=np.int32) * 20
        r1, _ = compress_steim1_record(s1, "GRA1", "BHZ", network="GR", start_epoch=t0, dt=0.05, seqno=1, reclth=512)
        r2, _ = compress_steim1_record(s2, "GRA1", "BHZ", network="GR", start_epoch=t0 + 2.5, dt=0.01, seqno=2, reclth=512)

        trs = read_mseed_bytes(r1 + r2, calib_override=1.0)
        self.assertEqual(len(trs), 2)
        self.assertAlmostEqual(trs[0].delta, 0.05, places=6)
        self.assertAlmostEqual(trs[1].delta, 0.01, places=6)

    def test_read_mseed_bytes_logs_warning_on_corrupt_record(self) -> None:
        t0 = parse_sh_time("2024-06-01T10:00:00")
        r1, _ = compress_steim1_record(np.arange(40, dtype=np.int32), "GRA1", "BHZ", start_epoch=t0, reclth=512)
        corrupt = b"\xff" * 512
        with self.assertLogs("shm_app.io.mseed", level="WARNING") as cm:
            trs = read_mseed_bytes(r1 + corrupt, calib_override=1.0)
        self.assertEqual(len(trs), 1)
        self.assertEqual(trs[0].length, 40)
        self.assertTrue(any("Stopping Mini-SEED parse" in msg for msg in cm.output))

    def test_compress_steim1_large_step_int64_no_overflow(self) -> None:
        samples = np.array([2000000000, -2000000000, 2000000000], dtype=np.int32)
        rec, n_packed = compress_steim1_record(
            samples,
            "GRA1",
            "BHZ",
            prev_sample=-2000000000,
            reclth=512,
        )
        self.assertEqual(n_packed, 3)
        dec = decompress_steim(rec[64:], 3, steim_level=1, big_endian=True)
        np.testing.assert_array_equal(dec, samples)

    def test_compress_steim1_reclth_power_of_two_and_non_ascii_metadata(self) -> None:
        with self.assertRaises(ValueError):
            compress_steim1_record(np.array([1, 2], dtype=np.int32), "GRA1", "BHZ", reclth=500)
        with self.assertRaises(ValueError):
            compress_steim1_record(np.array([1, 2], dtype=np.int32), "GRA1", "BHZ", reclth=64)

        rec, n_packed = compress_steim1_record(
            np.array([1, 2, 3], dtype=np.int32),
            station="GRÄ1",
            channel="BHZ",
            network="DÜ",
            locid="Ö0",
            reclth=512,
        )
        self.assertEqual(n_packed, 3)
        self.assertEqual(len(rec), 512)

    def test_write_mseed_file_sanitizes_nan_inf_with_warning(self) -> None:
        tr = Trace(
            data=np.array([10.0, float("nan"), float("inf"), float("-inf"), -5.0]),
            delta=0.05,
            station="GRA1",
            calib=1.0,
            start_epoch=1700000000.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "nan_test.mseed")
            with self.assertLogs("shm_app.io.mseed", level="WARNING") as cm:
                write_mseed_file(out_path, [tr], reclth=512)
            self.assertTrue(any("non-finite" in m for m in cm.output))
            rb = read_mseed_file(out_path, calib_override=1.0)
            self.assertEqual(len(rb), 1)
            np.testing.assert_allclose(rb[0].data, [10.0, 0.0, 0.0, 0.0, -5.0])


class TestEvtAuditFixes(unittest.TestCase):
    def test_read_evt_pick_type_and_trailing_pick_without_eof_marker(self) -> None:
        content = (
            "Station code               : GRA1\n"
            "Onset time                 : 02-NOV-2003_02:27:25.120\n"
            "Phase name                 : P\n"
            "PickType                   : a\n"
            "--- End of Phase ---\n"
            "Station code               : GRB1\n"
            "Onset time                 : 02-NOV-2003_02:27:26.500\n"
            "Phase name                 : S\n"
            "PickType                   : t\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "trailing.evt")
            Path(fpath).write_text(content, encoding="latin-1")
            _param, picks = read_evt_file(fpath)
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0].station, "GRA1")
        self.assertEqual(picks[0].source, "a")
        self.assertEqual(picks[1].station, "GRB1")
        self.assertEqual(picks[1].source, "t")

    def test_write_evt_serializes_valid_zero_fields(self) -> None:
        param = AnalysisParamSet(
            origin="02-NOV-2003_02:15:10.500",
            lat=0.0,
            lon=0.0,
            azimuth=0.0,
            distance=10.0,
            mag_ml=0.0,
            mags_count=1,
        )
        pk = PhasePick(
            name="P",
            onset="02-NOV-2003_02:17:00.000",
            station="GRA1",
            ampl=25.0,
            mag_ml=0.0,
            resid=0.0,
            slowness=5.0,
            beam_azim=0.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "zeros.evt")
            write_evt_file(fpath, param, [pk])
            txt = Path(fpath).read_text(encoding="latin-1")
            self.assertIn("Latitude", txt)
            self.assertIn("Longitude", txt)
            self.assertIn("Back-azimuth (deg)", txt)
            self.assertIn("Phase beam azimuth", txt)
            self.assertIn("Residual time", txt)
            self.assertIn("Magnitude ml", txt)
            self.assertIn("Mean magnitude ml", txt)
            p_back, pks_back = read_evt_file(fpath)
            self.assertEqual(len(pks_back), 1)
            self.assertEqual(p_back.lat, 0.0)
            self.assertEqual(p_back.lon, 0.0)
            self.assertEqual(p_back.azimuth, 0.0)


class TestGseAndQfileAuditFixes(unittest.TestCase):
    def test_gse_malformed_wid1_and_wid2_bounds_check_no_index_error(self) -> None:
        gse_text = (
            "WID1 2003306\n"
            "line2\n"
            "WID2 2003/11/02 02:25:00.000 GRA1\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "malformed.gse")
            Path(fpath).write_text(gse_text, encoding="ascii")
            with self.assertLogs("shm_app.io.gse", level="WARNING") as cm:
                trs = read_gse_file(fpath)
            self.assertEqual(trs, [])
            self.assertGreaterEqual(len(cm.output), 2)

    def test_gse_negative_calib_and_checksum_mismatch_warn_and_track(self) -> None:
        gse_text = (
            "WID2 2003/11/02 02:25:00.000 GRA1  BHZ      INT        4   20.000000 -1.50000000e+00   1.000 STS-2  -1.0  0.0\n"
            "DAT2\n"
            "10 20 30 40\n"
            "CHK2 999999\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "bad_chk_calib.gse")
            Path(fpath).write_text(gse_text, encoding="ascii")
            with self.assertLogs("shm_app.io.gse", level="WARNING") as cm:
                trs = read_gse_file(fpath)
            self.assertEqual(len(trs), 1)
            self.assertFalse(trs[0].extra_headers["checksum_ok"])
            self.assertEqual(trs[0].calib, 1.0)
            self.assertTrue(any("checksum mismatch" in m for m in cm.output))
            self.assertTrue(any("invalid calibration" in m for m in cm.output))

    def test_qfile_guards_trclines_bounds_and_newline_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. trclines <= 0
            qhd_bad_trc = Path(tmpdir) / "bad_trc.QHD"
            qbn_bad_trc = Path(tmpdir) / "bad_trc.QBN"
            qhd_bad_trc.write_text("43981 1 0\n01|L001:000004~ L800:00000000~\n", encoding="latin-1")
            np.zeros(4, dtype=np.float32).tofile(qbn_bad_trc)
            with self.assertRaises(ValueError):
                read_qfile(str(qhd_bad_trc))

            # 2. Out-of-bounds .QBN offset + length
            qhd_oob = Path(tmpdir) / "oob.QHD"
            qbn_oob = Path(tmpdir) / "oob.QBN"
            qhd_oob.write_text("43981 1 2\n01|L001:000100~ L800:00000000~\n01|S000:GRA1~\n", encoding="latin-1")
            np.zeros(4, dtype=np.float32).tofile(qbn_oob)
            with self.assertRaises(ValueError):
                read_qfile(str(qhd_oob))

            # 3. Multiline comment & tilde sanitization in write_qfile
            tr1 = Trace(
                data=np.arange(10, dtype=np.float64),
                station="GRA1",
                comment="line1\nline2~extra",
                opinfo="op\r\ninfo",
            )
            tr2 = Trace(data=np.arange(10, 20, dtype=np.float64), station="GRB1")
            good_base = os.path.join(tmpdir, "clean_q")
            write_qfile(good_base, [tr1, tr2])
            rb = read_qfile(good_base)
            self.assertEqual(len(rb), 2)
            self.assertEqual(rb[0].comment, "line1 line2 extra")
            self.assertEqual(rb[0].opinfo, "op  info")


class TestGeoJsonQuakeMLStationXMLAuditFixes(unittest.TestCase):
    def test_geojson_sanitizes_nan_inf_and_dumps_strict_rfc8259(self) -> None:
        params = AnalysisParamSet(
            event_id=42,
            lat=float("nan"),
            lon=11.2,
            mag_ml=float("inf"),
            rmse=float("-inf"),
            error_ellipse=(float("nan"), 5.0, 10.0),
        )
        tr = Trace(data=np.zeros(10), station="GRA1", lat=49.69, lon=11.22)
        gj = export_geojson([tr], params)
        dumped = json.dumps(gj, allow_nan=False)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)
        with tempfile.TemporaryDirectory() as tmpdir:
            gj_path = os.path.join(tmpdir, "strict.geojson")
            write_geojson(gj_path, [tr], params)
            parsed = json.loads(Path(gj_path).read_text(encoding="utf-8"))
            self.assertIsNone(parsed["features"][0]["properties"]["mag_ml"])

    def test_quakeml_azimuth_zero_in_az_gap_and_confidence_ellipsoid_fallback(self) -> None:
        params = AnalysisParamSet(event_id=7, lat=49.0, lon=11.0, depth=10.0)
        tr_north = Trace(
            data=np.zeros(10),
            station="GRA1",
            distance=1.0,
            azimuth=0.0,
            picks=[PhasePick(name="P", onset="02-NOV-2003_02:27:25.120", station="GRA1")],
        )
        tr_south = Trace(
            data=np.zeros(10),
            station="GRC1",
            distance=1.0,
            azimuth=180.0,
            picks=[PhasePick(name="P", onset="02-NOV-2003_02:27:25.500", station="GRC1")],
        )
        xml_str = export_quakeml([tr_north, tr_south], params)
        self.assertNotIn("if False", xml_str)
        parsed = parse_quakeml(xml_str)
        self.assertAlmostEqual(parsed["origin"]["quality"]["azimuthalGap"], 180.0, places=2)

        # Now test confidenceEllipsoid fallback when maxHorizontalUncertainty/minHorizontalUncertainty are absent
        qml_ce_only = f"""<?xml version="1.0" encoding="UTF-8"?>
<q:quakeml xmlns="{BED_NS}" xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">
  <eventParameters publicID="smi:test/ep/1">
    <event publicID="smi:test/ev/1">
      <preferredOriginID>smi:test/org/1</preferredOriginID>
      <origin publicID="smi:test/org/1">
        <time><value>2024-01-01T00:00:00.000000Z</value></time>
        <latitude><value>49.5</value></latitude>
        <longitude><value>11.2</value></longitude>
        <depth><value>10000.0</value></depth>
        <originUncertainty>
          <confidenceEllipsoid>
            <semiMajorAxisLength>15000.0</semiMajorAxisLength>
            <semiMinorAxisLength>6500.0</semiMinorAxisLength>
            <semiIntermediateAxisLength>6500.0</semiIntermediateAxisLength>
            <majorAxisPlunge>0.0</majorAxisPlunge>
            <majorAxisAzimuth>38.0</majorAxisAzimuth>
            <majorAxisRotation>0.0</majorAxisRotation>
          </confidenceEllipsoid>
        </originUncertainty>
      </origin>
    </event>
  </eventParameters>
</q:quakeml>
"""
        parsed_ce = parse_quakeml(qml_ce_only)
        self.assertIsNotNone(parsed_ce["params"].error_ellipse)
        a_km, b_km, az = parsed_ce["params"].error_ellipse
        self.assertAlmostEqual(a_km, 15.0, places=3)
        self.assertAlmostEqual(b_km, 6.5, places=3)
        self.assertAlmostEqual(az, 38.0, places=3)

    def test_stationxml_evaluates_direct_poles_zeros_when_stages_empty_and_horner(self) -> None:
        p0 = -2.0 * math.pi * 0.01
        w1 = 2.0 * math.pi * 1.0
        a0 = abs((1j * w1 - p0) / (1j * w1))
        pz = StationXMLPolesZeros(
            pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
            normalization_factor=a0,
            normalization_frequency=1.0,
            zeros=[0.0 + 0.0j],
            poles=[complex(p0, 0.0)],
        )
        ch_direct = StationXMLChannel(
            network="GR",
            station="GRA1",
            location="00",
            code="BHZ",
            sample_rate=20.0,
            poles_zeros=pz,
            stages=[],
            instrument_sensitivity=StationXMLSensitivity(value=1000.0, frequency=1.0),
        )
        r_dc_and_1hz = eval_stationxml_response(ch_direct, np.array([0.0, 1.0]))
        self.assertAlmostEqual(float(np.abs(r_dc_and_1hz[0])), 0.0, places=9)
        self.assertAlmostEqual(float(np.abs(r_dc_and_1hz[1])), 1000.0, delta=1.0)

        # Coefficients stage using Horner's method (e.g. IIR rational filter)
        cf = StationXMLCoefficients(numerator=[0.5, 0.5], denominator=[1.0, -0.25])
        ch_cf = StationXMLChannel(
            network="GR",
            station="GRA1",
            location="00",
            code="BHZ",
            sample_rate=20.0,
            stages=[StationXMLResponseStage(number=1, coefficients=cf)],
        )
        freqs = np.array([0.0, 2.5, 10.0])
        resp = eval_stationxml_response(ch_cf, freqs, include_sensitivity=False)
        z_inv = np.exp(-1j * 2.0 * math.pi * freqs / 20.0)
        expected = (0.5 + 0.5 * z_inv) / (1.0 - 0.25 * z_inv)
        np.testing.assert_allclose(resp, expected, rtol=1e-12, atol=1e-12)


class TestSeedLinkAuditFixes(unittest.TestCase):
    def test_ring_buffer_trims_one_sample_overlap_and_preserves_picks(self) -> None:
        buf = WaveformRingBuffer(max_samples_per_stream=1000)
        t0 = 1700000000.0
        dt = 0.05
        tr1 = Trace(data=np.arange(10, dtype=np.float64), delta=dt, station="GRA1", network="GR", start_epoch=t0)
        # 1-sample overlap: starts at t0 + 9*dt instead of t0 + 10*dt (gap = -dt)
        pk_ov1 = PhasePick(name="P", onset="02-NOV-2003_02:25:00.000", station="GRA1")
        tr2 = Trace(
            data=np.arange(9, 20, dtype=np.float64),
            delta=dt,
            station="GRA1",
            network="GR",
            start_epoch=t0 + 9 * dt,
            picks=[pk_ov1],
        )
        # Larger overlap (-3*dt < -1.5*dt) also carries a pick
        pk_ov2 = PhasePick(name="S", onset="02-NOV-2003_02:25:05.000", station="GRA1")
        tr3 = Trace(
            data=np.arange(17, 25, dtype=np.float64),
            delta=dt,
            station="GRA1",
            network="GR",
            start_epoch=t0 + 17 * dt,
            picks=[pk_ov2],
        )
        buf.append_traces([tr1, tr2, tr3])
        snaps = buf.snapshot_traces()
        self.assertEqual(len(snaps), 1)
        np.testing.assert_array_equal(snaps[0].data, np.arange(25, dtype=np.float64))
        self.assertEqual([p.name for p in snaps[0].picks], ["P", "S"])

    def test_seedlink_out_of_order_seq_no_16m_packets_lost(self) -> None:
        t0 = parse_sh_time("2024-06-01T10:00:00")
        rec10, _ = compress_steim1_record(np.arange(20, dtype=np.int32), "GRA1", "BHZ", network="GR", start_epoch=t0, reclth=512)
        rec9, _ = compress_steim1_record(np.arange(20, dtype=np.int32), "GRA1", "BHZ", network="GR", start_epoch=t0 - 1.0, reclth=512)
        rec11, _ = compress_steim1_record(np.arange(20, dtype=np.int32), "GRA1", "BHZ", network="GR", start_epoch=t0 + 1.0, reclth=512)

        def _pkt(seq: int, rec: bytes) -> bytes:
            return f"SL{seq:06X}".encode("ascii") + rec

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(2)
        host, port = srv.getsockname()

        def serve() -> None:
            conn, _ = srv.accept()
            try:
                buf = b""
                while not buf.endswith(b"END\r\n") and not buf.endswith(b"END\n"):
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        up = line.decode("ascii").strip().upper()
                        if up == "HELLO":
                            conn.sendall(b"SL v3.1\r\nTest\r\n")
                        elif up == "END":
                            conn.sendall(_pkt(10, rec10) + _pkt(9, rec9) + _pkt(11, rec11) + b"END")
                            return
                        else:
                            conn.sendall(b"OK\r\n")
            finally:
                conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            client = SeedLinkClient(host=host, port=port, streams=[("GR", "GRA1", "BHZ.D")], timeout=2.0)
            pkts = client.stream(auto_reconnect=False)
            self.assertEqual(len(pkts), 3)
            self.assertEqual(client.packets_lost, 0)
            self.assertEqual(client.sequence_gaps, 0)
            self.assertEqual(client.last_seq, 11)
        finally:
            srv.close()
            t.join(timeout=2.0)

    def test_seedlink_multi_station_per_station_resume_seq(self) -> None:
        t0 = parse_sh_time("2024-06-01T10:00:00")
        rec_a, _ = compress_steim1_record(np.ones(20, dtype=np.int32), "GRA1", "BHZ", network="GR", start_epoch=t0, reclth=512)
        rec_b, _ = compress_steim1_record(np.ones(20, dtype=np.int32), "BFO", "BHZ", network="GR", start_epoch=t0, reclth=512)

        def _pkt(seq: int, rec: bytes) -> bytes:
            return f"SL{seq:06X}".encode("ascii") + rec

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        host, port = srv.getsockname()
        cmds_by_conn: list[list[str]] = []

        def serve() -> None:
            for conn_idx in range(2):
                conn, _ = srv.accept()
                cmds: list[str] = []
                cmds_by_conn.append(cmds)
                try:
                    buf = b""
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            break
                        buf += chunk
                        done = False
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            txt = line.decode("ascii").strip()
                            cmds.append(txt)
                            up = txt.upper()
                            if up == "HELLO":
                                conn.sendall(b"SL v3.1\r\nTest\r\n")
                            elif up == "END":
                                if conn_idx == 0:
                                    conn.sendall(_pkt(3, rec_a) + _pkt(7, rec_b))
                                else:
                                    conn.sendall(_pkt(4, rec_a) + b"END")
                                done = True
                                break
                            else:
                                conn.sendall(b"OK\r\n")
                        if done:
                            break
                finally:
                    conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            client = SeedLinkClient(
                host=host,
                port=port,
                streams=[
                    SeedLinkStreamSpec("GR", "GRA1", "BHZ.D"),
                    SeedLinkStreamSpec("GR", "BFO", "BHZ.D"),
                ],
                timeout=2.0,
            )
            pkts = client.stream(auto_reconnect=True, reconnect_delay=0.05)
            self.assertEqual(len(pkts), 3)
            conn1_cmds = cmds_by_conn[1]
            self.assertIn("DATA 000004", conn1_cmds)
            self.assertIn("DATA 000008", conn1_cmds)
        finally:
            srv.close()
            t.join(timeout=2.0)


class TestSessionAuditFixes(unittest.TestCase):
    def test_session_deserialize_validates_delta_gt_zero(self) -> None:
        tr_zero = deserialize_trace({"station": "GRA1", "delta": 0.0, "data": [1.0, 2.0]})
        self.assertGreater(tr_zero.delta, 0.0)
        self.assertAlmostEqual(tr_zero.delta, 0.05)

        tr_neg = deserialize_trace({"station": "GRA1", "delta": -0.25, "data": [1.0, 2.0]})
        self.assertGreater(tr_neg.delta, 0.0)
        self.assertAlmostEqual(tr_neg.delta, 0.05)

    def test_session_stores_inline_for_filtered_traces_and_uses_compact_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_file = Path(tmpdir) / "dummy.mseed"
            ext_file.write_bytes(b"MSEED")

            tr_filtered = Trace(
                data=np.linspace(-5.0, 5.0, 200),
                delta=0.05,
                station="GRA1",
                filter_name="WWSSN_SP",
                source_path=str(ext_file),
            )
            entry = serialize_trace(tr_filtered)
            self.assertTrue(entry["has_inline_data"])
            self.assertIsInstance(entry["data"], list)
            self.assertEqual(len(entry["data"]), 200)

            tr_forced = Trace(data=np.ones(10), station="GRB1", source_path=str(ext_file))
            self.assertFalse(serialize_trace(tr_forced)["has_inline_data"])
            self.assertTrue(serialize_trace(tr_forced, store_inline=True)["has_inline_data"])

            session_file = Path(tmpdir) / "session.json"
            mgr = SessionManager(session_file)
            mgr.save([tr_filtered])
            raw_text = session_file.read_text(encoding="utf-8")
            self.assertNotIn("\n  ", raw_text)
            restored = mgr.load()
            self.assertTrue(restored["ok"])
            np.testing.assert_allclose(restored["traces"][0].data, tr_filtered.data)

    def test_session_resolve_session_path_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = os.path.join(tmpdir, "custom_session.json")
            explicit_file = os.path.join(tmpdir, "explicit.json")
            with mock.patch.dict(os.environ, {"SHM_SESSION_PATH": env_file}):
                self.assertEqual(resolve_session_path(None), Path(env_file))
                self.assertEqual(resolve_session_path(explicit_file), Path(explicit_file))

    def test_session_nan_inf_sanitized_with_warning(self) -> None:
        tr = Trace(data=np.array([1.0, float("nan"), float("inf")]), station="NAN1")
        with self.assertLogs("shm_app.core.session", level="WARNING") as cm:
            entry = serialize_trace(tr)
        self.assertTrue(any("non-finite" in m for m in cm.output))
        self.assertEqual(entry["data"], [1.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "s.json"
            atomic_write_json(out_path, {"version": 1, "traces": [entry], "param_sets": []})
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["traces"][0]["data"], [1.0, 0.0, 0.0])


class TestReportingAuditFixes(unittest.TestCase):
    def test_reporting_header_labels_d_lat_d_lon_in_km(self) -> None:
        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        params = AnalysisParamSet(lat=49.6919, d_lat=2.45, lon=11.2217, d_lon=3.15)
        render_header_and_metadata(ax, params, traces=[])
        texts = [c.get_text().get_text() for c in ax.tables[0].get_celld().values()]
        lat_cells = [t for t in texts if "+49.6919" in t]
        lon_cells = [t for t in texts if "+11.2217" in t]
        self.assertEqual(len(lat_cells), 1)
        self.assertEqual(len(lon_cells), 1)
        self.assertIn("2.45 km", lat_cells[0])
        self.assertIn("3.15 km", lon_cells[0])
        self.assertNotIn("2.450°", lat_cells[0])
        self.assertNotIn("3.150°", lon_cells[0])

    def test_reporting_beachball_antipodally_projects_upgoing_polarities(self) -> None:
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        # Upgoing ray at az=45 deg, ih=150 deg -> antipodal lower-hemisphere az=225 deg, ih=30 deg -> px<0, py<0
        render_beachball(ax, strike_deg=0.0, dip_deg=45.0, rake_deg=90.0, polarities=[(45.0, 150.0, 1)])
        self.assertGreaterEqual(len(ax.collections), 3)
        offsets = ax.collections[-1].get_offsets()
        self.assertEqual(len(offsets), 1)
        px, py = float(offsets[0][0]), float(offsets[0][1])
        expected_r = math.sqrt(2.0) * math.sin(math.radians(30.0) / 2.0)
        self.assertAlmostEqual(math.hypot(px, py), expected_r, places=5)
        self.assertLess(px, 0.0)
        self.assertLess(py, 0.0)

    def test_reporting_waveform_axvspan_aligned_to_ylim_span(self) -> None:
        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        t0 = 1700000000.0
        traces = [
            Trace(
                data=np.sin(np.linspace(0, 10, 100)),
                delta=0.05,
                station="GRA1",
                start_epoch=t0,
                picks=[PhasePick(name="P", quality=2, station="GRA1")],
            ),
            Trace(
                data=np.cos(np.linspace(0, 10, 100)),
                delta=0.05,
                station="GRB1",
                start_epoch=t0,
                picks=[PhasePick(name="P", quality=2, station="GRB1")],
            ),
        ]
        # Set onset strings cleanly via PhasePick.onset_epoch setter
        for idx, tr in enumerate(traces):
            tr.picks[0].onset_epoch = t0 + 1.0 + idx

        render_waveform_panels(ax, traces)
        self.assertEqual(ax.get_ylim(), (-0.6, 1.6))
        self.assertEqual(len(ax.patches), 2)
        # First trace at y_base = 1.0, second at y_base = 0.0
        for expected_center, patch in zip((1.0, 0.0), ax.patches):
            y0_frac = float(patch.get_y())
            h_frac = float(patch.get_height())
            y_frac_mid = y0_frac + 0.5 * h_frac
            data_mid = -0.6 + y_frac_mid * (1.6 - (-0.6))
            self.assertAlmostEqual(data_mid, expected_center, places=5)


class TestLiveEventsAndServiceAuditFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_parse_usgs_geojson_guards_bad_upd_ms(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "us_test_1",
                    "geometry": {"type": "Point", "coordinates": [11.2, 49.7, 10.0]},
                    "properties": {
                        "time": 1700000000000,
                        "updated": "not_a_number",
                        "mag": 3.5,
                        "place": "Bavaria",
                    },
                },
                {
                    "id": "us_test_2",
                    "geometry": {"type": "Point", "coordinates": [12.0, 50.0, 8.0]},
                    "properties": {
                        "time": 1700000100000,
                        "updated": float("nan"),
                        "mag": 4.0,
                        "place": "Saxony",
                    },
                },
            ],
        }
        # Test JSON with string bad updated + raw NaN/empty updated
        events = parse_usgs_geojson(json.dumps(payload))
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].updated_epoch, 1700000000.0, places=3)
        self.assertAlmostEqual(events[1].updated_epoch, 1700000100.0, places=3)

    def test_coordinator_prunes_entries_and_ordered_ids_on_dedup_eviction(self) -> None:
        coord = LiveEventCoordinator(allow_network_discovery=False)
        coord._dedup = EventDeduplicator(max_entries=50)
        try:
            t0 = 1700000000.0
            for i in range(65):
                ev = LiveEvent(
                    event_id=f"ev_{i:03d}",
                    origin_epoch=t0 + i * 100.0,
                    origin_iso="2023-11-14T22:13:20",
                    lat=49.7 + (i % 5) * 0.1,
                    lon=11.2 + (i % 5) * 0.1,
                    depth_km=10.0,
                    magnitude=2.5,
                    region="TEST",
                )
                coord.ingest_event(ev)
            self.assertLessEqual(len(coord.entries), 50)
            self.assertEqual(len(coord.ordered_ids), len(coord.entries))
            self.assertIn("ev_064", coord.entries)
        finally:
            coord.shutdown()

    def test_coordinator_shutdown_cancels_slow_fetch_cleanly(self) -> None:
        calls = []
        started = threading.Event()

        def slow_opener(url: str, timeout: float):
            calls.append(url)
            started.set()
            time.sleep(0.2)
            return 204, b""

        with tempfile.TemporaryDirectory() as tmpdir:
            lcfg = LiveConfig(
                station_strategy="observatory",
                station_preset="CUSTOM",
                custom_stations=("GRA1", "GRB1", "GRC1", "BFO"),
                archive_grace_sec=0.0,
                output_base_dir=tmpdir,
            )
            coord = LiveEventCoordinator(
                live_config=lcfg,
                opener=slow_opener,
                time_fn=lambda: 1700000500.0,
                allow_network_discovery=False,
            )
            ev = LiveEvent(
                event_id="slow_ev",
                origin_epoch=1700000000.0,
                origin_iso="2023-11-14T22:13:20",
                lat=49.7,
                lon=11.2,
                depth_km=10.0,
                magnitude=4.5,
                region="BAVARIA",
            )
            coord.ingest_event(ev)
            coord.force_fetch_event("slow_ev")
            self.assertTrue(started.wait(timeout=2.0))
            t_start = time.monotonic()
            coord.shutdown(wait_ms=2000)
            elapsed = time.monotonic() - t_start
            self.assertLess(elapsed, 1.0)
            # Subsequent stations were cancelled before even issuing HTTP requests
            self.assertLess(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
