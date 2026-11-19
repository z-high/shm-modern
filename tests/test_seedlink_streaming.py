"""Tests for M-21 Native SeedLink v3.1 client, WaveformRingBuffer, and QThread worker."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import unittest
from typing import Callable, List, Optional, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from shm_app.core.config_station import get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.mseed import compress_steim1_record
from shm_app.io.seedlink import (
    SeedLinkClient,
    SeedLinkCommandError,
    SeedLinkPacket,
    SeedLinkProtocolError,
    SeedLinkStreamSpec,
    SeedLinkTimeoutError,
    WaveformRingBuffer,
    stream_key_for_trace,
)
from shm_app.plugins.live_service import LiveEventCoordinator, SeedLinkStreamWorker


def _make_512_mseed_packets(
    station: str,
    channel: str,
    network: str,
    start_epoch: float,
    total_samples: int,
    dt: float = 0.05,
    locid: str = "",
) -> Tuple[List[bytes], np.ndarray]:
    """Generate real 512-byte Steim-1 Mini-SEED records and return (records, counts)."""
    t = np.arange(total_samples, dtype=np.float64) * dt
    counts = np.round(400.0 * np.sin(2.0 * np.pi * 1.1 * t) + 120.0 * np.cos(2.0 * np.pi * 0.4 * t)).astype(np.int32)
    records: List[bytes] = []
    idx = 0
    seq = 1
    prev = 0
    while idx < counts.size:
        rec, packed = compress_steim1_record(
            counts[idx:],
            station=station,
            channel=channel,
            network=network,
            locid=locid,
            start_epoch=start_epoch + idx * dt,
            dt=dt,
            seqno=seq,
            prev_sample=prev,
            reclth=512,
        )
        assert len(rec) == 512
        records.append(rec)
        prev = int(counts[idx + packed - 1])
        idx += packed
        seq += 1
    return records, counts


def _frame_sl_packet(seq_num: int, record_512: bytes) -> bytes:
    """Wrap a 512-byte Mini-SEED record in an 8-byte SLxxxxxx SeedLink frame."""
    assert len(record_512) == 512
    hdr = f"SL{seq_num & 0xFFFFFF:06X}".encode("ascii")
    return hdr + record_512


class _LoopbackSeedLinkServer:
    """In-process TCP SeedLink v3.1 loopback server on 127.0.0.1:0 for deterministic tests."""

    def __init__(self, session_handler: Callable[[socket.socket, int, List[str]], None]) -> None:
        self._handler = session_handler
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self._srv.settimeout(0.2)
        self.host, self.port = self._srv.getsockname()
        self._stop = threading.Event()
        self.commands_by_conn: List[List[str]] = []
        self.conn_count = 0
        self._threads: List[threading.Thread] = []
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn_idx = self.conn_count
            self.conn_count += 1
            cmds: List[str] = []
            self.commands_by_conn.append(cmds)
            t = threading.Thread(target=self._run_conn, args=(conn, conn_idx, cmds), daemon=True)
            self._threads.append(t)
            t.start()

    def _run_conn(self, conn: socket.socket, conn_idx: int, cmds: List[str]) -> None:
        try:
            conn.settimeout(5.0)
            self._handler(conn, conn_idx, cmds)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def read_command_line(conn: socket.socket) -> Optional[str]:
        buf = bytearray()
        while True:
            ch = conn.recv(1)
            if not ch:
                return None
            if ch == b"\n":
                return buf.decode("ascii", errors="replace").strip()
            if ch != b"\r":
                buf.extend(ch)

    def handle_standard_handshake(
        self,
        conn: socket.socket,
        cmds: List[str],
        reject_station: Optional[str] = None,
    ) -> bool:
        while True:
            line = self.read_command_line(conn)
            if line is None:
                return False
            cmds.append(line)
            upper = line.upper()
            if upper == "HELLO":
                conn.sendall(b"SeedLink v3.1 (SHM Test Server)\r\nSHM Observatory Loopback\r\n")
            elif upper.startswith("STATION "):
                if reject_station and reject_station.upper() in upper:
                    conn.sendall(b"ERROR\r\n")
                else:
                    conn.sendall(b"OK\r\n")
            elif upper.startswith("SELECT ") or upper.startswith("DATA") or upper.startswith("TIME"):
                conn.sendall(b"OK\r\n")
            elif upper == "END":
                return True
            else:
                conn.sendall(b"ERROR\r\n")

    def close(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        self._accept_thread.join(timeout=2.0)
        for t in self._threads:
            t.join(timeout=1.0)


class TestSeedLinkProtocolAndBuffer(unittest.TestCase):
    def test_zero_forbidden_imports_in_seedlink_module(self) -> None:
        # Ensure shm_app.io.seedlink has zero PyQt6, obspy, or scipy dependencies
        import shm_app.io.seedlink as sl_mod

        with open(sl_mod.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("PyQt6", src)
        self.assertNotIn("obspy", src)
        self.assertNotIn("scipy", src)
        self.assertNotIn("PyQt6", sys.modules.get("shm_app.io.seedlink", object).__dict__)

    def test_handshake_streaming_decoding_and_ring_buffer_assembly(self) -> None:
        cfg = get_config()
        t0 = parse_sh_time("2024-06-01T10:00:00")
        records, expected_counts = _make_512_mseed_packets(
            station="GRA1",
            channel="BHZ",
            network="GR",
            start_epoch=t0,
            total_samples=900,
            dt=0.05,
        )
        self.assertGreaterEqual(len(records), 3)

        def handler(conn: socket.socket, _idx: int, cmds: List[str]) -> None:
            if not server.handle_standard_handshake(conn, cmds):
                return
            for seq_i, rec in enumerate(records, start=1):
                conn.sendall(_frame_sl_packet(seq_i, rec))
            conn.sendall(b"END")

        server = _LoopbackSeedLinkServer(handler)
        try:
            ring = WaveformRingBuffer(max_samples_per_stream=5000, max_traces=16)
            client = SeedLinkClient(
                host=server.host,
                port=server.port,
                streams=[SeedLinkStreamSpec("GR", "GRA1", "BHZ.D")],
                timeout=3.0,
                config=cfg,
                ring_buffer=ring,
            )
            packets = client.stream(auto_reconnect=False)

            self.assertEqual(len(packets), len(records))
            self.assertTrue(all(p.is_valid for p in packets))
            self.assertEqual(client.last_seq, len(records))
            self.assertEqual(client.malformed_packets, 0)
            self.assertEqual(client.sequence_gaps, 0)

            # Verify SeedLink v3.1 handshake commands sent
            cmds = server.commands_by_conn[0]
            self.assertEqual(cmds, ["HELLO", "STATION GRA1 GR", "SELECT BHZ.D", "DATA", "END"])

            # Verify assembled continuous trace in ring buffer
            traces = ring.snapshot_traces()
            self.assertEqual(len(traces), 1)
            tr = traces[0]
            self.assertEqual(stream_key_for_trace(tr), "GR.GRA1..BHZ")
            self.assertEqual(tr.length, expected_counts.size)
            calib = cfg.get_sensitivity("gra1-bh-z", t0)
            np.testing.assert_allclose(tr.data, expected_counts.astype(np.float64) * calib, rtol=1e-6, atol=1e-6)
        finally:
            server.close()

    def test_handshake_error_response_raises_command_error(self) -> None:
        def handler(conn: socket.socket, _idx: int, cmds: List[str]) -> None:
            server.handle_standard_handshake(conn, cmds, reject_station="BADSTA")

        server = _LoopbackSeedLinkServer(handler)
        try:
            client = SeedLinkClient(
                host=server.host,
                port=server.port,
                streams=[("GR", "BADSTA", "BHZ.D")],
                timeout=2.0,
            )
            client.connect()
            try:
                with self.assertRaises(SeedLinkCommandError):
                    client.negotiate()
            finally:
                client.close()
        finally:
            server.close()

    def test_socket_disconnect_recovery_and_sequence_resume(self) -> None:
        cfg = get_config()
        t0 = parse_sh_time("2024-06-01T11:00:00")
        records, expected_counts = _make_512_mseed_packets("WET", "BHZ", "GR", t0, total_samples=1200)
        self.assertGreaterEqual(len(records), 4)
        split_at = len(records) // 2

        def handler(conn: socket.socket, conn_idx: int, cmds: List[str]) -> None:
            if not server.handle_standard_handshake(conn, cmds):
                return
            if conn_idx == 0:
                # Send first half then abruptly close TCP socket without END
                for seq_i in range(1, split_at + 1):
                    conn.sendall(_frame_sl_packet(seq_i, records[seq_i - 1]))
                return
            # Second connection: send remaining records and clean END
            for seq_i in range(split_at + 1, len(records) + 1):
                conn.sendall(_frame_sl_packet(seq_i, records[seq_i - 1]))
            conn.sendall(b"END")

        server = _LoopbackSeedLinkServer(handler)
        try:
            ring = WaveformRingBuffer(max_samples_per_stream=10000, max_traces=16)
            errors: List[Exception] = []
            client = SeedLinkClient(
                host=server.host,
                port=server.port,
                streams=[("GR", "WET", "BHZ.D")],
                timeout=2.0,
                config=cfg,
                ring_buffer=ring,
            )
            packets = client.stream(
                auto_reconnect=True,
                reconnect_delay=0.05,
                on_error=errors.append,
            )

            self.assertEqual(len(packets), len(records))
            self.assertGreaterEqual(server.conn_count, 2)
            self.assertGreaterEqual(len(errors), 1)
            # Reconnect handshake must request DATA <next_seq_hex>
            expected_resume = f"DATA {split_at + 1:06X}"
            self.assertIn(expected_resume, server.commands_by_conn[1])

            traces = ring.snapshot_traces()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].length, expected_counts.size)
        finally:
            server.close()

    def test_timeout_recovery_on_stalled_server(self) -> None:
        t0 = parse_sh_time("2024-06-01T12:00:00")
        records, _ = _make_512_mseed_packets("GRA1", "BHZ", "GR", t0, total_samples=400)

        def handler(conn: socket.socket, conn_idx: int, cmds: List[str]) -> None:
            if not server.handle_standard_handshake(conn, cmds):
                return
            if conn_idx == 0:
                # Stall longer than client timeout=0.15s
                time.sleep(0.45)
                return
            for seq_i, rec in enumerate(records, start=1):
                conn.sendall(_frame_sl_packet(seq_i, rec))
            conn.sendall(b"END")

        server = _LoopbackSeedLinkServer(handler)
        try:
            # First verify explicit SeedLinkTimeoutError when auto_reconnect=False
            ring = WaveformRingBuffer()
            client = SeedLinkClient(
                host=server.host,
                port=server.port,
                streams=[("GR", "GRA1", "BHZ.D")],
                timeout=0.15,
                ring_buffer=ring,
            )
            client.connect()
            client.negotiate()
            with self.assertRaises(SeedLinkTimeoutError):
                client.read_packet()
            client.close()

            # Second, verify auto_reconnect recovers from timeout seamlessly
            # Spawn fresh client; conn_idx=1 from previous client already passed, so reset server handler behavior via new server
        finally:
            server.close()

        # Test auto-reconnect timeout recovery on a dedicated server
        def handler_recover(conn: socket.socket, conn_idx: int, cmds: List[str]) -> None:
            if not server2.handle_standard_handshake(conn, cmds):
                return
            if conn_idx == 0:
                time.sleep(0.45)
                return
            for seq_i, rec in enumerate(records, start=1):
                conn.sendall(_frame_sl_packet(seq_i, rec))
            conn.sendall(b"END")

        server2 = _LoopbackSeedLinkServer(handler_recover)
        try:
            ring2 = WaveformRingBuffer()
            errs: List[Exception] = []
            client2 = SeedLinkClient(
                host=server2.host,
                port=server2.port,
                streams=[("GR", "GRA1", "BHZ.D")],
                timeout=0.15,
                ring_buffer=ring2,
            )
            pkts = client2.stream(auto_reconnect=True, reconnect_delay=0.05, on_error=errs.append)
            self.assertEqual(len(pkts), len(records))
            self.assertTrue(any(isinstance(e, SeedLinkTimeoutError) for e in errs))
            self.assertGreater(ring2.sample_count(), 0)
        finally:
            server2.close()

    def test_malformed_packet_and_packet_loss_handling(self) -> None:
        t0 = parse_sh_time("2024-06-01T13:00:00")
        records, _ = _make_512_mseed_packets("GRA1", "BHZ", "GR", t0, total_samples=1200)
        self.assertGreaterEqual(len(records), 3)

        def handler(conn: socket.socket, conn_idx: int, cmds: List[str]) -> None:
            if not server.handle_standard_handshake(conn, cmds):
                return
            if conn_idx == 0:
                # Valid seq 1, malformed Mini-SEED at seq 2, sequence jump to seq 5 skipping records[1]
                conn.sendall(_frame_sl_packet(1, records[0]))
                conn.sendall(_frame_sl_packet(2, b"\xff" * 512))
                conn.sendall(_frame_sl_packet(5, records[2]))
                conn.sendall(b"END")
            else:
                # Bad framing header (not SLxxxxxx and not END)
                conn.sendall(b"BADFRAME" + b"\x00" * 512)

        server = _LoopbackSeedLinkServer(handler)
        try:
            ring = WaveformRingBuffer()
            client = SeedLinkClient(
                host=server.host,
                port=server.port,
                streams=[("GR", "GRA1", "BHZ.D")],
                timeout=2.0,
                ring_buffer=ring,
            )
            pkts = client.stream(auto_reconnect=False)
            self.assertEqual(len(pkts), 3)
            self.assertTrue(pkts[0].is_valid)
            self.assertFalse(pkts[1].is_valid)
            self.assertTrue(pkts[1].error)
            self.assertTrue(pkts[2].is_valid)
            self.assertEqual(client.malformed_packets, 1)
            self.assertEqual(client.sequence_gaps, 1)
            self.assertEqual(client.packets_lost, 2)  # seq 3 and 4 missed between 2 and 5
            self.assertEqual(len(ring.snapshot_traces()), 2)  # seq 1 and seq 5 separated by time gap

            # Bad framing header raises SeedLinkProtocolError
            client2 = SeedLinkClient(host=server.host, port=server.port, timeout=2.0)
            client2.connect()
            client2.negotiate()
            with self.assertRaises(SeedLinkProtocolError):
                client2.read_packet()
            client2.close()
        finally:
            server.close()

    def test_ring_buffer_bounded_capacity_and_thread_safety(self) -> None:
        buf = WaveformRingBuffer(max_samples_per_stream=250, max_traces=3)
        t0 = 1717200000.0
        dt = 0.05

        # Append 10 contiguous 100-sample chunks (1000 samples total > max 250)
        for i in range(10):
            data = np.arange(i * 100, (i + 1) * 100, dtype=np.float64)
            tr = Trace(data=data, delta=dt, station="GRA1", network="GR", locid="00", chan1="B", chan2="H", comp="Z", start_epoch=t0 + i * 100 * dt)
            buf.append_trace(tr)

        self.assertEqual(buf.sample_count("GR.GRA1.00.BHZ"), 250)
        snaps = buf.snapshot_traces("GR.GRA1.00.BHZ")
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].length, 250)
        # Must retain the newest 250 samples: 750..999
        np.testing.assert_array_equal(snaps[0].data, np.arange(750, 1000, dtype=np.float64))
        self.assertAlmostEqual(snaps[0].start_epoch, t0 + 750 * dt, places=6)
        # Returned snapshot must be decoupled (no internal buffer mutation or shared base array)
        self.assertTrue(snaps[0].data.flags["OWNDATA"])
        snaps[0].data[:] = -999.0
        self.assertEqual(buf.snapshot_traces("GR.GRA1.00.BHZ")[0].data[0], 750.0)

        # LRU eviction when adding > max_traces (3) distinct stream keys
        for sta in ("GRA2", "GRA3", "GRA4"):
            tr = Trace(data=np.ones(50), delta=dt, station=sta, network="GR", chan1="B", chan2="H", comp="Z", start_epoch=t0)
            buf.append_trace(tr)
        keys = buf.stream_keys()
        self.assertEqual(len(keys), 3)
        self.assertNotIn("GR.GRA1.00.BHZ", keys)
        self.assertEqual(keys, ["GR.GRA2..BHZ", "GR.GRA3..BHZ", "GR.GRA4..BHZ"])

        # Concurrent writer + snapshot reader stress test
        errors: List[Exception] = []

        def writer() -> None:
            try:
                for k in range(100):
                    buf.append_trace(
                        Trace(
                            data=np.full(40, float(k)),
                            delta=dt,
                            station="GRA4",
                            network="GR",
                            chan1="B",
                            chan2="H",
                            comp="Z",
                            start_epoch=t0 + (k + 1) * 40 * dt,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(100):
                    s = buf.snapshot_traces()
                    for item in s:
                        _ = item.length
            except Exception as exc:
                errors.append(exc)

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start()
        rt.start()
        wt.join(timeout=5.0)
        rt.join(timeout=5.0)
        self.assertEqual(errors, [])
        self.assertLessEqual(buf.sample_count("GR.GRA4..BHZ"), 250)


class TestSeedLinkWorkerAndCoordinator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _run_events_until(self, cond_fn: Callable[[], bool], timeout_ms: int = 3000) -> bool:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            self.app.processEvents()
            if cond_fn():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(cond_fn())

    def test_seedlink_worker_signals_and_clean_stop(self) -> None:
        t0 = parse_sh_time("2024-06-01T14:00:00")
        records, expected_counts = _make_512_mseed_packets("GRA1", "BHZ", "GR", t0, total_samples=700)
        hold_open = threading.Event()

        def handler(conn: socket.socket, _idx: int, cmds: List[str]) -> None:
            if not server.handle_standard_handshake(conn, cmds):
                return
            for seq_i, rec in enumerate(records, start=1):
                conn.sendall(_frame_sl_packet(seq_i, rec))
            # Keep socket open until client calls stop()
            hold_open.wait(timeout=3.0)

        server = _LoopbackSeedLinkServer(handler)
        try:
            ring = WaveformRingBuffer(max_samples_per_stream=5000)
            worker = SeedLinkStreamWorker(
                host=server.host,
                port=server.port,
                streams=[("GR", "GRA1", "BHZ.D")],
                ring_buffer=ring,
                timeout=2.0,
                auto_reconnect=False,
            )
            statuses: List[str] = []
            packets: List[SeedLinkPacket] = []
            trace_updates: List[List[Trace]] = []
            errors: List[str] = []

            worker.statusChanged.connect(statuses.append)
            worker.packetReceived.connect(packets.append)
            worker.tracesUpdated.connect(trace_updates.append)
            worker.errorOccurred.connect(errors.append)

            worker.start()
            ok = self._run_events_until(lambda: len(packets) == len(records))
            self.assertTrue(ok, f"Timed out waiting for packets; got {len(packets)}/{len(records)}, statuses={statuses}, errors={errors}")
            self.assertIn("connecting", statuses)
            self.assertIn("streaming", statuses)
            self.assertGreaterEqual(len(trace_updates), 1)
            self.assertEqual(trace_updates[-1][0].length, expected_counts.size)

            # Request clean stop while server is still holding socket open
            t_stop0 = time.monotonic()
            worker.stop(wait_ms=2000)
            elapsed = time.monotonic() - t_stop0
            self.app.processEvents()
            self.assertFalse(worker.isRunning())
            self.assertLess(elapsed, 1.5)
            self.assertEqual(statuses[-1], "stopped")
        finally:
            hold_open.set()
            server.close()

    def test_live_event_coordinator_seedlink_lifecycle_and_shutdown(self) -> None:
        t0 = parse_sh_time("2024-06-01T15:00:00")
        records, expected_counts = _make_512_mseed_packets("BFO", "BHZ", "GR", t0, total_samples=700)
        hold_open = threading.Event()

        def handler(conn: socket.socket, _idx: int, cmds: List[str]) -> None:
            if not server.handle_standard_handshake(conn, cmds):
                return
            for seq_i, rec in enumerate(records, start=1):
                conn.sendall(_frame_sl_packet(seq_i, rec))
            hold_open.wait(timeout=3.0)

        server = _LoopbackSeedLinkServer(handler)
        coord = LiveEventCoordinator(allow_network_discovery=False)
        try:
            snaps_seen: List[int] = []
            coord.seedlinkTracesUpdated.connect(lambda trs: snaps_seen.append(sum(t.length for t in trs)))

            coord.start_seedlink(
                host=server.host,
                port=server.port,
                streams=[("GR", "BFO", "BHZ.D")],
                max_samples_per_stream=2000,
                timeout=2.0,
            )
            self.assertTrue(coord.is_seedlink_running)
            ok = self._run_events_until(
                lambda: (
                    bool(coord.get_seedlink_traces())
                    and coord.get_seedlink_traces()[0].length == expected_counts.size
                    and len(snaps_seen) > 0
                )
            )
            self.assertTrue(ok)
            self.assertGreater(len(snaps_seen), 0)

            traces = coord.get_seedlink_traces()
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].station, "BFO")
            self.assertEqual(traces[0].length, expected_counts.size)

            # shutdown() must cleanly stop SeedLink worker as well as coordinator
            coord.shutdown()
            self.app.processEvents()
            self.assertFalse(coord.is_seedlink_running)
            self.assertEqual(coord.seedlink_status, "stopped")
        finally:
            hold_open.set()
            coord.shutdown()
            server.close()


if __name__ == "__main__":
    unittest.main()
