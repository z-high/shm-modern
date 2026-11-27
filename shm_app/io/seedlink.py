"""Native pure-Python SeedLink v3.1 TCP streaming client and thread-safe waveform ring buffer."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import select
import socket
import struct
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.io.mseed import SeedRecordHeader, parse_seed_header, read_mseed_bytes

SEEDLINK_DEFAULT_PORT = 18000
SEEDLINK_HEADER_LEN = 8
SEEDLINK_RECORD_LEN = 512
SEEDLINK_PACKET_LEN = SEEDLINK_HEADER_LEN + SEEDLINK_RECORD_LEN
SEEDLINK_END_MARKER = b"END"
_VALID_SEED_INDICATORS = frozenset(("D", "R", "Q", "M", " "))


class SeedLinkError(Exception):
    """Base exception for SeedLink client failures."""


class SeedLinkConnectionError(SeedLinkError):
    """Raised when TCP connection to a SeedLink server fails."""


class SeedLinkDisconnectError(SeedLinkError):
    """Raised when an established SeedLink TCP stream disconnects unexpectedly."""


class SeedLinkTimeoutError(SeedLinkError):
    """Raised when a socket operation exceeds its configured timeout."""


class SeedLinkCommandError(SeedLinkError):
    """Raised when a SeedLink handshake command receives ERROR or invalid response."""


class SeedLinkProtocolError(SeedLinkError):
    """Raised when SeedLink packet framing is corrupted or unexpected."""


class SeedLinkStoppedError(SeedLinkError):
    """Raised internally when the client is cancelled via stop_event or close()."""


@dataclass(frozen=True)
class SeedLinkStreamSpec:
    """Subscription specification for a SeedLink station stream."""

    network: str
    station: str
    selectors: str = "BHZ.D"
    seq: Optional[int] = None
    start_time: Optional[str] = None  # YYYY,MM,DD,HH,MM,SS


@dataclass
class SeedLinkPacket:
    """Decoded SeedLink v3.1 packet (8-byte SL header + 512-byte Mini-SEED record)."""

    seq_num: int
    seq_hex: str
    raw_header: bytes
    payload: bytes
    header: Optional[SeedRecordHeader]
    traces: List[Trace]
    stream_key: str
    received_at: float
    is_info: bool = False
    is_valid: bool = True
    error: str = ""


def stream_key_for_trace(tr: Trace) -> str:
    """Return canonical FDSN stream key 'NET.STA.LOC.CHA' for a Trace."""
    net = (tr.network or "").strip().upper()
    sta = (tr.station or "").strip().upper()
    loc = (tr.locid or "").strip().upper()
    cha = tr.channel3.strip().upper()
    return f"{net}.{sta}.{loc}.{cha}"


def _normalize_streams(
    streams: Optional[Sequence[Union[SeedLinkStreamSpec, Tuple[str, ...]]]],
) -> List[SeedLinkStreamSpec]:
    if not streams:
        return []
    out: List[SeedLinkStreamSpec] = []
    for item in streams:
        if isinstance(item, SeedLinkStreamSpec):
            out.append(item)
        elif isinstance(item, (tuple, list)):
            if len(item) == 2:
                out.append(SeedLinkStreamSpec(network=str(item[0]), station=str(item[1])))
            elif len(item) == 3:
                out.append(SeedLinkStreamSpec(network=str(item[0]), station=str(item[1]), selectors=str(item[2])))
            elif len(item) >= 4:
                out.append(
                    SeedLinkStreamSpec(
                        network=str(item[0]),
                        station=str(item[1]),
                        selectors=str(item[2]),
                        seq=item[3],
                        start_time=item[4] if len(item) > 4 else None,
                    )
                )
            else:
                raise ValueError(f"Invalid stream tuple: {item!r}")
        else:
            raise TypeError(f"Unsupported stream specification type: {type(item)!r}")
    return out


def parse_mseed_packet(
    payload: bytes,
    config: Optional[SHMConfig] = None,
) -> Tuple[SeedRecordHeader, List[Trace]]:
    """Validate and decode a single 512-byte Mini-SEED packet using shm_app.io.mseed."""
    if len(payload) < 64:
        raise ValueError(f"Mini-SEED packet too short ({len(payload)} bytes)")

    # Validate raw BTIME year before parse_seed_header clamps it
    year_be = struct.unpack(">H", payload[20:22])[0]
    year_le = struct.unpack("<H", payload[20:22])[0]
    if not (1950 <= year_be <= 2099 or 1950 <= year_le <= 2099):
        raise ValueError(f"Invalid SEED BTIME year (BE={year_be}, LE={year_le})")

    hdr = parse_seed_header(payload)
    if hdr.indicator not in _VALID_SEED_INDICATORS:
        raise ValueError(f"Invalid SEED data header indicator: {hdr.indicator!r}")
    if not hdr.station.strip() or not hdr.channel.strip():
        raise ValueError("Missing SEED station or channel code")
    if hdr.no_of_samples <= 0:
        raise ValueError(f"Invalid SEED sample count: {hdr.no_of_samples}")
    if hdr.dt <= 0 or not np.isfinite(hdr.dt):
        raise ValueError(f"Invalid SEED sample interval dt={hdr.dt}")
    if hdr.databegin < 48 or hdr.databegin >= len(payload):
        raise ValueError(f"Invalid SEED databegin offset: {hdr.databegin}")

    traces = read_mseed_bytes(payload, config=config)
    if not traces or all(tr.length <= 0 for tr in traces):
        raise ValueError("Mini-SEED packet decoded 0 samples")
    for tr in traces:
        if not np.all(np.isfinite(tr.data)):
            raise ValueError("Mini-SEED packet contains non-finite samples")
    return hdr, traces


class WaveformRingBuffer:
    """Thread-safe bounded ring buffer assembling Trace chunks per NET.STA.LOC.CHA key."""

    def __init__(
        self,
        max_samples_per_stream: int = 36000,
        max_traces: int = 128,
        max_chunks_per_stream: Optional[int] = None,
    ) -> None:
        if max_samples_per_stream <= 0:
            raise ValueError("max_samples_per_stream must be > 0")
        if max_traces <= 0:
            raise ValueError("max_traces must be > 0")
        if max_chunks_per_stream is not None and max_chunks_per_stream <= 0:
            raise ValueError("max_chunks_per_stream must be > 0 when specified")

        self.max_samples_per_stream = int(max_samples_per_stream)
        self.max_traces = int(max_traces)
        self.max_chunks_per_stream = int(max_chunks_per_stream) if max_chunks_per_stream is not None else None

        self._lock = threading.Lock()
        self._streams: "OrderedDict[str, Deque[Trace]]" = OrderedDict()
        self._total_appended_samples = 0
        self._dropped_samples = 0
        self._evicted_streams = 0

    def append_trace(self, tr: Trace) -> None:
        """Append a Trace chunk into the ring buffer, merging contiguous samples and trimming overflow."""
        if tr.length <= 0 or tr.delta <= 0:
            return

        key = stream_key_for_trace(tr)
        incoming = tr.copy(copy_picks=True)
        # Ensure incoming owns a compact contiguous float64 buffer
        incoming.data = np.array(incoming.data, dtype=np.float64, copy=True)

        with self._lock:
            self._total_appended_samples += incoming.length
            segments = self._streams.get(key)
            if segments is None:
                segments = deque()
                self._streams[key] = segments
            else:
                # Refresh LRU ordering
                self._streams.move_to_end(key, last=True)

            self._append_or_merge(segments, incoming)
            self._trim_stream_locked(segments)
            self._evict_overflow_streams_locked()

    def append_traces(self, traces: Sequence[Trace]) -> None:
        """Append multiple Trace chunks."""
        for tr in traces:
            self.append_trace(tr)

    def _append_or_merge(self, segments: Deque[Trace], incoming: Trace) -> None:
        if not segments:
            segments.append(incoming)
            return

        tail = segments[-1]
        dt = tail.delta
        if dt <= 0 or abs(incoming.delta - dt) > max(1e-6, 0.01 * dt):
            segments.append(incoming)
            return

        tail_end = tail.start_epoch + tail.length * dt
        gap = incoming.start_epoch - tail_end

        if abs(gap) <= 1.5 * dt:
            # Contiguous packet (or minor overlap within 1.5*dt) -> merge directly into tail Trace
            if gap < 0.0:
                skip = int(round(-gap / dt))
                if skip <= 0:
                    tail.data = np.concatenate((tail.data, incoming.data))
                elif skip < incoming.length:
                    tail.data = np.concatenate(
                        (tail.data, np.array(incoming.data[skip:], dtype=np.float64, copy=True))
                    )
            else:
                tail.data = np.concatenate((tail.data, incoming.data))
            if incoming.picks:
                for p in incoming.picks:
                    tail.add_or_replace_pick(p)
        elif gap < -1.5 * dt and incoming.start_epoch >= tail.start_epoch:
            # Partial or full overlap within the tail segment
            skip = int(round(-gap / dt))
            if skip < incoming.length:
                tail.data = np.concatenate((tail.data, np.array(incoming.data[skip:], dtype=np.float64, copy=True)))
            if incoming.picks:
                for p in incoming.picks:
                    tail.add_or_replace_pick(p)
        else:
            # Non-contiguous gap or out-of-order jump -> keep as separate segment
            segments.append(incoming)

    def _trim_stream_locked(self, segments: Deque[Trace]) -> None:
        if self.max_chunks_per_stream is not None:
            while len(segments) > self.max_chunks_per_stream:
                dropped_seg = segments.popleft()
                self._dropped_samples += dropped_seg.length

        total_samples = sum(seg.length for seg in segments)
        excess = total_samples - self.max_samples_per_stream
        while excess > 0 and segments:
            head = segments[0]
            if head.length <= excess:
                segments.popleft()
                self._dropped_samples += head.length
                excess -= head.length
            else:
                # Copy slice so old large NumPy allocation is immediately released (no memory leak)
                head.data = np.array(head.data[excess:], dtype=np.float64, copy=True)
                head.start_epoch += excess * head.delta
                self._dropped_samples += excess
                excess = 0

    def _evict_overflow_streams_locked(self) -> None:
        while len(self._streams) > self.max_traces:
            _old_key, old_segs = self._streams.popitem(last=False)
            self._evicted_streams += 1
            self._dropped_samples += sum(seg.length for seg in old_segs)

    def snapshot_traces(self, stream_key: Optional[str] = None) -> List[Trace]:
        """Return thread-safe deep copies of currently buffered Traces."""
        with self._lock:
            if stream_key is not None:
                segs = self._streams.get(stream_key)
                if not segs:
                    return []
                return [seg.copy(copy_picks=True) for seg in segs]

            out: List[Trace] = []
            for segs in self._streams.values():
                for seg in segs:
                    out.append(seg.copy(copy_picks=True))
            return out

    def stream_keys(self) -> List[str]:
        """Return currently buffered stream keys in LRU order."""
        with self._lock:
            return list(self._streams.keys())

    def sample_count(self, stream_key: Optional[str] = None) -> int:
        """Return total number of buffered samples (overall or for a single stream key)."""
        with self._lock:
            if stream_key is not None:
                segs = self._streams.get(stream_key)
                return sum(seg.length for seg in segs) if segs else 0
            return sum(seg.length for segs in self._streams.values() for seg in segs)

    def clear(self, stream_key: Optional[str] = None) -> None:
        """Clear buffered waveforms for one stream key or all streams."""
        with self._lock:
            if stream_key is not None:
                self._streams.pop(stream_key, None)
            else:
                self._streams.clear()

    def stats(self) -> Dict[str, Any]:
        """Return buffer occupancy and trimming diagnostics."""
        with self._lock:
            per_stream = {k: sum(s.length for s in segs) for k, segs in self._streams.items()}
            return {
                "stream_count": len(self._streams),
                "total_samples": sum(per_stream.values()),
                "per_stream_samples": per_stream,
                "total_appended_samples": self._total_appended_samples,
                "dropped_samples": self._dropped_samples,
                "evicted_streams": self._evicted_streams,
            }


class SeedLinkClient:
    """Pure-Python TCP SeedLink v3.1 client with handshake, 520-byte framing, and reconnect support."""

    def __init__(
        self,
        host: str,
        port: int = SEEDLINK_DEFAULT_PORT,
        streams: Optional[Sequence[Union[SeedLinkStreamSpec, Tuple[str, ...]]]] = None,
        timeout: float = 10.0,
        config: Optional[SHMConfig] = None,
        ring_buffer: Optional[WaveformRingBuffer] = None,
        resume_from_last_seq: bool = True,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.streams = _normalize_streams(streams)
        self.timeout = max(0.05, float(timeout))
        self.config = config or get_config()
        self.ring_buffer = ring_buffer
        self.resume_from_last_seq = bool(resume_from_last_seq)
        self.stop_event = stop_event or threading.Event()

        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self.server_info: Tuple[str, str] = ("", "")
        self.last_seq: Optional[int] = None
        self.last_seq_by_station: Dict[Tuple[str, str], int] = {}
        self.packets_received: int = 0
        self.malformed_packets: int = 0
        self.sequence_gaps: int = 0
        self.packets_lost: int = 0
        self._connect_count: int = 0

    @property
    def is_connected(self) -> bool:
        with self._sock_lock:
            return self._sock is not None

    def request_stop(self) -> None:
        """Signal any blocking wait/read loop to abort and close active socket."""
        self.stop_event.set()
        self.close()

    def connect(self) -> None:
        """Open a TCP connection to the SeedLink server."""
        self.close()
        if self.stop_event.is_set():
            raise SeedLinkStoppedError("SeedLink client stopped before connect")
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            sock.settimeout(self.timeout)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        except socket.timeout as exc:
            raise SeedLinkTimeoutError(f"Connection timed out ({self.host}:{self.port}): {exc}") from exc
        except OSError as exc:
            raise SeedLinkConnectionError(f"Failed to connect to {self.host}:{self.port}: {exc}") from exc

        with self._sock_lock:
            self._sock = sock
            self._connect_count += 1

    def close(self) -> None:
        """Close the underlying TCP socket cleanly."""
        with self._sock_lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _get_socket(self) -> socket.socket:
        with self._sock_lock:
            if self._sock is None:
                raise SeedLinkDisconnectError("Not connected to SeedLink server")
            return self._sock

    def _wait_readable(self, sock: socket.socket, deadline: float) -> None:
        """Wait until socket is readable, polling stop_event at <=0.1s intervals."""
        while True:
            if self.stop_event.is_set():
                raise SeedLinkStoppedError("SeedLink client stopped")
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise SeedLinkTimeoutError(f"Timed out after {self.timeout:.2f}s waiting for SeedLink data")
            step = min(0.1, rem)
            try:
                fd = sock.fileno()
                if fd < 0:
                    raise SeedLinkDisconnectError("Socket closed")
                readable, _, _ = select.select([sock], [], [], step)
                if readable:
                    return
            except (ValueError, OSError) as exc:
                if self.stop_event.is_set():
                    raise SeedLinkStoppedError("SeedLink client stopped") from exc
                raise SeedLinkDisconnectError(f"Socket error while waiting for data: {exc}") from exc

    def _recv_exact(self, nbytes: int) -> bytes:
        """Receive exact number of bytes without over-reading past frame boundaries."""
        sock = self._get_socket()
        buf = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(buf) < nbytes:
            self._wait_readable(sock, deadline)
            try:
                chunk = sock.recv(nbytes - len(buf))
            except socket.timeout as exc:
                raise SeedLinkTimeoutError(f"Socket recv timed out: {exc}") from exc
            except OSError as exc:
                if self.stop_event.is_set():
                    raise SeedLinkStoppedError("SeedLink client stopped") from exc
                raise SeedLinkDisconnectError(f"Socket recv failed: {exc}") from exc

            if not chunk:
                raise SeedLinkDisconnectError("SeedLink server closed TCP connection")
            buf.extend(chunk)
            # Reset idle deadline once progress is made so slow trickle doesn't prematurely fail
            deadline = time.monotonic() + self.timeout
        return bytes(buf)

    def _send_command(self, cmd: str) -> None:
        sock = self._get_socket()
        payload = (cmd.rstrip("\r\n") + "\r\n").encode("ascii")
        try:
            sock.sendall(payload)
        except socket.timeout as exc:
            raise SeedLinkTimeoutError(f"Timed out sending command {cmd!r}: {exc}") from exc
        except OSError as exc:
            raise SeedLinkDisconnectError(f"Failed sending command {cmd!r}: {exc}") from exc

    def _read_line(self, max_len: int = 4096) -> str:
        """Read a single CRLF/LF-terminated handshake line without buffering past newline."""
        sock = self._get_socket()
        line = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(line) < max_len:
            self._wait_readable(sock, deadline)
            try:
                ch = sock.recv(1)
            except socket.timeout as exc:
                raise SeedLinkTimeoutError(f"Timed out reading SeedLink response: {exc}") from exc
            except OSError as exc:
                if self.stop_event.is_set():
                    raise SeedLinkStoppedError("SeedLink client stopped") from exc
                raise SeedLinkDisconnectError(f"Socket closed reading SeedLink response: {exc}") from exc

            if not ch:
                raise SeedLinkDisconnectError("SeedLink server closed connection during handshake")
            if ch == b"\n":
                break
            if ch != b"\r":
                line.extend(ch)
        return line.decode("ascii", errors="replace").strip()

    def _expect_ok(self, context: str) -> None:
        resp = self._read_line()
        if resp.upper() != "OK":
            raise SeedLinkCommandError(f"{context} rejected by server: {resp!r}")

    def send_hello(self) -> Tuple[str, str]:
        """Send SeedLink HELLO command and parse 2-line server identification."""
        self._send_command("HELLO")
        line1 = self._read_line()
        if line1.upper() == "ERROR":
            raise SeedLinkCommandError("HELLO command returned ERROR")
        line2 = self._read_line()
        self.server_info = (line1, line2)
        return self.server_info

    def select_station(
        self,
        station: str,
        network: str,
        selectors: str = "",
        seq: Optional[int] = None,
        start_time: Optional[str] = None,
    ) -> None:
        """Execute STATION / SELECT / DATA handshake sequence for one station."""
        sta_u = station.strip().upper()
        net_u = network.strip().upper()
        self._send_command(f"STATION {sta_u} {net_u}")
        self._expect_ok(f"STATION {sta_u} {net_u}")

        if selectors and selectors.strip():
            for sel in selectors.replace(",", " ").split():
                self._send_command(f"SELECT {sel.strip()}")
                self._expect_ok(f"SELECT {sel.strip()}")

        self.send_data_command(seq=seq, start_time=start_time)

    def send_data_command(
        self,
        seq: Optional[int] = None,
        start_time: Optional[str] = None,
    ) -> None:
        """Send DATA [seq [time]] (or TIME <start_time>) command and await OK."""
        if seq is not None and start_time:
            cmd = f"DATA {int(seq) & 0xFFFFFF:06X} {start_time.strip()}"
        elif seq is not None:
            cmd = f"DATA {int(seq) & 0xFFFFFF:06X}"
        elif start_time:
            cmd = f"TIME {start_time.strip()}"
        else:
            cmd = "DATA"
        self._send_command(cmd)
        self._expect_ok(cmd)

    def end_handshake(self) -> None:
        """Send END command to enter SeedLink binary packet streaming mode."""
        self._send_command("END")

    def negotiate(
        self,
        streams: Optional[Sequence[Union[SeedLinkStreamSpec, Tuple[str, ...]]]] = None,
        send_hello: bool = True,
    ) -> Tuple[str, str]:
        """Run full SeedLink v3.1 handshake (HELLO, STATION, SELECT, DATA, END)."""
        specs = _normalize_streams(streams) if streams is not None else self.streams
        if send_hello:
            self.send_hello()

        is_reconnect = self._connect_count > 1 and self.last_seq is not None and self.resume_from_last_seq
        global_resume_seq = ((self.last_seq + 1) & 0xFFFFFF) if is_reconnect else None

        if specs:
            for spec in specs:
                if is_reconnect:
                    net_u = spec.network.strip().upper()
                    sta_u = spec.station.strip().upper()
                    sta_last = self.last_seq_by_station.get((net_u, sta_u))
                    if sta_last is None and net_u in ("*", ""):
                        for (k_net, k_sta), k_seq in self.last_seq_by_station.items():
                            if k_sta == sta_u:
                                sta_last = k_seq
                                break
                    if sta_last is not None:
                        seq: Optional[int] = (sta_last + 1) & 0xFFFFFF
                    else:
                        seq = global_resume_seq if global_resume_seq is not None else spec.seq
                else:
                    seq = spec.seq
                self.select_station(
                    station=spec.station,
                    network=spec.network,
                    selectors=spec.selectors,
                    seq=seq,
                    start_time=spec.start_time,
                )
        else:
            self.send_data_command(seq=global_resume_seq)

        self.end_handshake()
        return self.server_info

    def reconnect(self, streams: Optional[Sequence[Union[SeedLinkStreamSpec, Tuple[str, ...]]]] = None) -> None:
        """Close current connection, reconnect TCP socket, and renegotiate handshake."""
        self.close()
        self.connect()
        self.negotiate(streams=streams, send_hello=True)

    def read_packet(self) -> Optional[SeedLinkPacket]:
        """Read and decode the next 520-byte SeedLink packet, or return None on END terminator."""
        hdr3 = self._recv_exact(3)
        if hdr3 == SEEDLINK_END_MARKER:
            return None
        if hdr3 == b"ERR":
            rest = self._read_line()
            raise SeedLinkProtocolError(f"SeedLink server returned error during stream: ERR{rest}")

        hdr5 = self._recv_exact(5)
        raw_hdr = hdr3 + hdr5
        if raw_hdr[:2] != b"SL":
            raise SeedLinkProtocolError(f"Malformed SeedLink framing header: {raw_hdr!r}")

        is_info = raw_hdr.startswith(b"SLINFO")
        if is_info:
            payload = self._recv_exact(SEEDLINK_RECORD_LEN)
            return SeedLinkPacket(
                seq_num=-1,
                seq_hex="INFO",
                raw_header=raw_hdr,
                payload=payload,
                header=None,
                traces=[],
                stream_key="",
                received_at=time.time(),
                is_info=True,
                is_valid=True,
            )

        try:
            seq_hex = raw_hdr[2:8].decode("ascii")
            seq_num = int(seq_hex, 16)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SeedLinkProtocolError(f"Invalid SeedLink hex sequence header: {raw_hdr!r}") from exc

        payload = self._recv_exact(SEEDLINK_RECORD_LEN)
        now_ep = time.time()

        if self.last_seq is not None:
            expected = (self.last_seq + 1) & 0xFFFFFF
            if seq_num != expected:
                gap = (seq_num - expected) & 0xFFFFFF
                if gap < 0x800000:
                    self.sequence_gaps += 1
                    self.packets_lost += gap
                    self.last_seq = seq_num
                # Else: out-of-order or duplicate packet behind last_seq -> do not wrap into ~16M packets_lost
            else:
                self.last_seq = seq_num
        else:
            self.last_seq = seq_num
        self.packets_received += 1

        try:
            hdr, traces = parse_mseed_packet(payload, config=self.config)
        except Exception as exc:
            self.malformed_packets += 1
            return SeedLinkPacket(
                seq_num=seq_num,
                seq_hex=seq_hex.upper(),
                raw_header=raw_hdr,
                payload=payload,
                header=None,
                traces=[],
                stream_key="",
                received_at=now_ep,
                is_info=False,
                is_valid=False,
                error=str(exc),
            )

        sta_key = (hdr.network.strip().upper(), hdr.station.strip().upper())
        prev_sta_seq = self.last_seq_by_station.get(sta_key)
        if prev_sta_seq is None or ((seq_num - prev_sta_seq) & 0xFFFFFF) < 0x800000:
            self.last_seq_by_station[sta_key] = seq_num

        skey = stream_key_for_trace(traces[0]) if traces else ""
        if self.ring_buffer is not None:
            self.ring_buffer.append_traces(traces)

        return SeedLinkPacket(
            seq_num=seq_num,
            seq_hex=seq_hex.upper(),
            raw_header=raw_hdr,
            payload=payload,
            header=hdr,
            traces=traces,
            stream_key=skey,
            received_at=now_ep,
            is_info=False,
            is_valid=True,
        )

    def stream(
        self,
        max_packets: Optional[int] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 0.5,
        max_reconnect_delay: float = 30.0,
        on_packet: Optional[Callable[[SeedLinkPacket], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> List[SeedLinkPacket]:
        """Connect, negotiate, and stream packets with optional automatic reconnect."""
        collected: List[SeedLinkPacket] = []
        delay = max(0.05, float(reconnect_delay))

        try:
            while not self.stop_event.is_set():
                try:
                    self.connect()
                    self.negotiate(send_hello=True)
                    delay = max(0.05, float(reconnect_delay))

                    while not self.stop_event.is_set():
                        pkt = self.read_packet()
                        if pkt is None:
                            return collected
                        collected.append(pkt)
                        if on_packet is not None:
                            on_packet(pkt)
                        if max_packets is not None and len(collected) >= max_packets:
                            return collected

                except SeedLinkStoppedError:
                    break
                except (SeedLinkError, OSError) as exc:
                    if self.stop_event.is_set():
                        break
                    if on_error is not None:
                        on_error(exc)
                    if not auto_reconnect:
                        raise
                    self.close()
                    if self.stop_event.wait(delay):
                        break
                    delay = min(max_reconnect_delay, delay * 2.0)
        finally:
            self.close()

        return collected
