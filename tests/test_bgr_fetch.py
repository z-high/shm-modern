"""Tests for the BGR EIDA FDSN-WS Mini-SEED downloader plugin, SH command, and GUI dialog."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
from unittest import mock
import urllib.parse

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from shm_app.core.config_station import get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import month_day_to_julian, parse_sh_time
from shm_app.gui.main_window import SHMMainWindow
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.io.mseed import compress_steim1_record, read_mseed_bytes, write_mseed_file
from shm_app.plugins.bgr_fetch import (
    BGR_FDSN_DATASELECT_URL,
    GRF_STATIONS,
    GRSN_STATIONS,
    build_dataselect_url,
    expand_channels,
    fetch_bgr_waveforms,
    format_fdsn_time,
    geress_stations,
    parse_station_list,
)


def _make_steim1_mseed_bytes(station: str, channel: str, start_epoch: float, n_samples: int = 1200, dt: float = 0.05) -> bytes:
    """Create valid multi-record Steim-1 Mini-SEED bytes matching BGR output."""
    t = np.arange(n_samples, dtype=np.float64) * dt
    counts = np.round(500.0 * np.sin(2.0 * np.pi * 1.2 * t)).astype(np.int32)
    chunks = []
    idx = 0
    seq = 1
    prev = 0
    while idx < counts.size:
        rec, packed = compress_steim1_record(
            counts[idx:],
            station=station,
            channel=channel,
            network="GR",
            locid="",
            start_epoch=start_epoch + idx * dt,
            dt=dt,
            seqno=seq,
            prev_sample=prev,
            reclth=512,
        )
        chunks.append(rec)
        prev = int(counts[idx + packed - 1])
        idx += packed
        seq += 1
    return b"".join(chunks)


def _make_float32_mseed_bytes(station: str, channel: str, start_epoch: float, values: np.ndarray, dt: float = 0.05) -> bytes:
    """Build a 512-byte Mini-SEED record with IEEE float32 (encoding=4) payload."""
    import datetime

    dt_obj = datetime.datetime.fromtimestamp(start_epoch, tz=datetime.timezone.utc)
    jday = month_day_to_julian(dt_obj.year, dt_obj.month, dt_obj.day)
    rate = int(round(1.0 / dt))
    nsmp = int(values.size)
    hdr = struct.pack(
        ">6scB5s2s3s2sHHBBBBHHhhBBBBiHH",
        b"000001",
        b"D",
        9,
        f"{station[:5]:<5s}".encode("ascii"),
        b"00",
        f"{channel[:3]:<3s}".encode("ascii"),
        b"GR",
        dt_obj.year,
        jday,
        dt_obj.hour,
        dt_obj.minute,
        dt_obj.second,
        0,
        0,
        nsmp,
        rate,
        1,
        0,
        0,
        0,
        1,
        0,
        64,
        48,
    )
    blk1000 = struct.pack(">HHBBBB", 1000, 0, 4, 1, 9, 0) + (b"\x00" * 8)
    payload = values.astype(">f4").tobytes()
    pad = b"\x00" * (512 - 64 - len(payload))
    return hdr + blk1000 + payload + pad


class TestBGRFetchCore(unittest.TestCase):
    def test_url_builder_channels_and_time_formatting(self) -> None:
        t0 = parse_sh_time("2024-05-01T12:00:00")
        t1 = t0 + 300.0
        self.assertEqual(format_fdsn_time(t0), "2024-05-01T12:00:00")
        self.assertEqual(expand_channels("BH", ("Z", "N")), ["BHZ", "BHN"])
        self.assertEqual(expand_channels("HHZ,BH?", ("Z",)), ["HHZ", "BH?"])
        self.assertEqual(parse_station_list("gra1, fur; BFO gra1"), ["GRA1", "FUR", "BFO"])

        url = build_dataselect_url(
            BGR_FDSN_DATASELECT_URL,
            network="GR",
            station="GRA1",
            channels=["BHZ", "BHN", "BHE"],
            start_epoch=t0,
            end_epoch=t1,
            location="*",
        )
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(parsed.netloc, "eida.bgr.de")
        self.assertEqual(qs["network"], ["GR"])
        self.assertEqual(qs["station"], ["GRA1"])
        self.assertEqual(qs["channel"], ["BHZ,BHN,BHE"])
        self.assertEqual(qs["starttime"], ["2024-05-01T12:00:00"])
        self.assertEqual(qs["endtime"], ["2024-05-01T12:05:00"])
        self.assertNotIn("location", qs)

        url_loc = build_dataselect_url(
            BGR_FDSN_DATASELECT_URL,
            network="GR",
            station="GRA1",
            channels="BHZ",
            start_epoch=t0,
            end_epoch=t1,
            location="00",
        )
        qs_loc = urllib.parse.parse_qs(urllib.parse.urlparse(url_loc).query)
        self.assertEqual(qs_loc["location"], ["00"])

        with self.assertRaises(ValueError):
            build_dataselect_url(BGR_FDSN_DATASELECT_URL, "GR", "GRA1", "BHZ", t0, t0)

    def test_read_mseed_bytes_supports_float32_and_steim1(self) -> None:
        t0 = parse_sh_time("2024-06-10T08:30:00")
        vals = np.linspace(-10.0, 25.0, 50, dtype=np.float64)
        raw_f32 = _make_float32_mseed_bytes("FUR", "BHZ", t0, vals, dt=0.05)
        trs = read_mseed_bytes(raw_f32, config=get_config())
        self.assertEqual(len(trs), 1)
        self.assertEqual(trs[0].station, "FUR")
        self.assertEqual(trs[0].locid, "00")
        self.assertAlmostEqual(trs[0].data[0] / trs[0].calib, -10.0, places=4)
        self.assertAlmostEqual(trs[0].data[-1] / trs[0].calib, 25.0, places=4)

    def test_fetch_bgr_waveforms_saves_mseed_sfd_and_handles_nodata_error(self) -> None:
        t0 = parse_sh_time("2024-05-01T12:00:00")
        cfg = get_config()
        payload_gra1 = (
            _make_steim1_mseed_bytes("GRA1", "BHZ", t0 - 5.0, n_samples=1600)
            + _make_steim1_mseed_bytes("GRA1", "BHN", t0 - 5.0, n_samples=1600)
        )

        def fake_open(url: str, _timeout: float):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            sta = qs["station"][0]
            if sta == "GRA1":
                return 200, payload_gra1
            if sta == "GRB1":
                return 204, b""
            if sta == "BAD":
                return 200, b"<html>502 Bad Gateway</html>"
            raise TimeoutError("timed out")

        progress_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            summary = fetch_bgr_waveforms(
                stations=["GRA1", "GRB1", "BAD", "FUR"],
                start_time="2024-05-01T12:00:00",
                duration=40.0,
                output_dir=tmp,
                network="GR",
                channels="BHZ,BHN",
                config=cfg,
                opener=fake_open,
                progress_cb=lambda i, n, r: progress_calls.append((i, n, r.station, r.status)),
            )

            self.assertFalse(summary.cancelled)
            self.assertEqual(len(progress_calls), 4)
            by_st = {r.station: r for r in summary.results}
            self.assertEqual(by_st["GRA1"].status, "ok")
            self.assertEqual(by_st["GRA1"].n_traces, 2)
            self.assertTrue(os.path.isfile(by_st["GRA1"].filepath))
            self.assertEqual(by_st["GRB1"].status, "nodata")
            self.assertEqual(by_st["BAD"].status, "error")
            self.assertEqual(by_st["FUR"].status, "error")
            self.assertFalse(any("BAD" in fn for fn in os.listdir(tmp)))
            self.assertIsNotNone(summary.sfd_path)
            self.assertTrue(os.path.isfile(summary.sfd_path))

            # Traces should be trimmed close to requested 40 s duration
            self.assertEqual(len(summary.traces), 2)
            for tr in summary.traces:
                self.assertAlmostEqual(tr.start_epoch, t0, delta=0.06)
                self.assertAlmostEqual(tr.duration, 40.0, delta=0.2)

    def test_fetch_cancel_stops_remaining_stations(self) -> None:
        raw = _make_steim1_mseed_bytes("GRA1", "BHZ", parse_sh_time("2024-05-01T12:00:00"), n_samples=400)
        count = 0

        def fake_open(_url: str, _timeout: float):
            nonlocal count
            count += 1
            return 200, raw

        with tempfile.TemporaryDirectory() as tmp:
            summary = fetch_bgr_waveforms(
                stations=["GRA1", "GRA2", "GRA3"],
                start_time="2024-05-01T12:00:00",
                duration=15.0,
                output_dir=tmp,
                opener=fake_open,
                cancel_check=lambda: count >= 1,
            )
            self.assertTrue(summary.cancelled)
            self.assertEqual(len(summary.results), 1)


class TestBGRFetchEngineAndGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_sh_engine_fetch_command(self) -> None:
        t0 = parse_sh_time("2024-05-01T12:00:00")
        raw = _make_steim1_mseed_bytes("BFO", "BHZ", t0, n_samples=600)
        eng = SHCommandEngine(config=get_config())

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("shm_app.plugins.bgr_fetch._default_http_open", return_value=(200, raw)):
                out = eng.execute_line(f"FETCH /BGR GR BFO BHZ 2024-05-01T12:00:00 20 /OUT={tmp}")
                self.assertIn("Fetched 1 Mini-SEED trace(s)", out)
                self.assertEqual(len(eng.traces), 1)
                self.assertEqual(eng.traces[0].station, "BFO")

                # /APPEND preserves existing traces
                eng.execute_line(f"FETCH /BGR GR BFO BHZ 2024-05-01T12:00:00 20 /OUT={tmp} /APPEND")
                self.assertEqual(len(eng.traces), 2)

    def test_gui_dialog_download_load_undo_and_close(self) -> None:
        from shm_app.plugins.bgr_dialog import BGRFetchDialog

        win = SHMMainWindow(autoload_example=True)
        n_initial = len(win.engine.traces)
        win.show()
        self.app.processEvents()

        try:
            win.action_open_bgr_fetch_dialog()
            self.app.processEvents()
            dlgs = [d for d in win._open_dialogs if isinstance(d, BGRFetchDialog)]
            self.assertEqual(len(dlgs), 1)
            dlg = dlgs[0]
            self.assertTrue(dlg.isVisible())
            self.assertGreaterEqual(dlg.network_combo.minimumWidth(), 130)
            self.assertGreaterEqual(dlg.location_combo.minimumWidth(), 120)
            self.assertGreaterEqual(dlg.chan_combo.minimumWidth(), 170)
            self.assertGreaterEqual(dlg.chan_combo.view().minimumWidth(), 210)

            # Check no size grip and no auto-default button artifacts
            from PyQt6.QtCore import QDate, QTime
            from PyQt6.QtWidgets import QPushButton
            self.assertFalse(dlg.isSizeGripEnabled())
            for btn in dlg.findChildren(QPushButton):
                self.assertFalse(btn.autoDefault())
                self.assertFalse(btn.isDefault())

            # Calling again raises existing dialog instead of spawning duplicates
            win.action_open_bgr_fetch_dialog()
            self.assertEqual(len([d for d in win._open_dialogs if isinstance(d, BGRFetchDialog)]), 1)

            # Configure preset & custom window
            dlg._apply_station_preset(["GRA1", "BFO"])
            self.assertEqual(dlg.selected_stations(), ["GRA1", "BFO"])
            self.assertTrue(set(geress_stations()).issuperset({"GEA0", "GEC2"}))
            self.assertIn("GRA1", GRF_STATIONS)
            self.assertIn("BFO", GRSN_STATIONS)

            t0 = parse_sh_time("2024-05-01T12:00:00")
            raw_gra1 = _make_steim1_mseed_bytes("GRA1", "BHZ", t0, n_samples=500)
            raw_bfo = _make_steim1_mseed_bytes("BFO", "BHZ", t0, n_samples=500)

            def fake_http(url: str, _timeout: float):
                sta = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["station"][0]
                return 200, (raw_gra1 if sta == "GRA1" else raw_bfo)

            with tempfile.TemporaryDirectory() as tmp:
                dlg.out_dir_edit.setText(tmp)
                dlg.set_start_time("2024-05-01T12:00:00")
                dlg.dur_spin.setValue(600.0)
                self.assertEqual(dlg.start_date_edit.date(), QDate(2024, 5, 1))
                self.assertEqual(dlg.start_time_edit.time(), QTime(12, 0, 0))
                self.assertEqual(dlg.end_time_edit.time(), QTime(12, 10, 0))
                self.assertIn("10 min", dlg.dur_human_lbl.text())

                # Adjusting End time updates Duration cleanly
                dlg.end_time_edit.setTime(QTime(12, 25, 0))
                self.assertAlmostEqual(dlg.dur_spin.value(), 1500.0, places=1)

                # Shifting End earlier than Start shifts Start back while preserving Duration
                dlg.dur_spin.setValue(120.0)
                dlg.end_time_edit.setTime(QTime(11, 30, 0))
                self.assertEqual(dlg.start_time_edit.time(), QTime(11, 28, 0))

                dlg.set_time_window(t0, 20.0)
                dlg.chk_z.setChecked(True)
                dlg.chk_n.setChecked(False)
                dlg.chk_e.setChecked(False)
                dlg.chk_append.setChecked(False)

                with mock.patch("shm_app.plugins.bgr_fetch._default_http_open", side_effect=fake_http):
                    dlg._start_download()
                    worker = dlg._worker
                    self.assertIsNotNone(worker)
                    self.assertTrue(worker.wait(5000))
                    self.app.processEvents()

            self.assertEqual(len(win.engine.traces), 2)
            self.assertEqual({t.station for t in win.engine.traces}, {"GRA1", "BFO"})

            # Undo restores previous workspace traces cleanly
            win.action_undo()
            self.assertEqual(len(win.engine.traces), n_initial)

            # Closing with corner X removes from _open_dialogs
            QApplication.sendEvent(dlg, QCloseEvent())
            self.assertEqual([d for d in win._open_dialogs if isinstance(d, BGRFetchDialog)], [])
        finally:
            for d in list(win._open_dialogs):
                d.close()
            win.close()


if __name__ == "__main__":
    unittest.main()
