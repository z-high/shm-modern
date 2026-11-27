"""Background QThread worker for SeedLink v3.1 live Mini-SEED streaming."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional, Sequence

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.io.seedlink import (
    SEEDLINK_DEFAULT_PORT,
    SeedLinkClient,
    SeedLinkError,
    SeedLinkPacket,
    SeedLinkStoppedError,
    WaveformRingBuffer,
)


class SeedLinkStreamWorker(QThread):
    """Background QThread streaming Mini-SEED packets over SeedLink v3.1 into a WaveformRingBuffer."""

    packetReceived = pyqtSignal(object)  # SeedLinkPacket
    tracesUpdated = pyqtSignal(list)  # List[Trace] snapshot
    statusChanged = pyqtSignal(str)  # status state string
    errorOccurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        host: str,
        port: int = SEEDLINK_DEFAULT_PORT,
        streams: Optional[Sequence[Any]] = None,
        ring_buffer: Optional[WaveformRingBuffer] = None,
        shm_config: Optional[SHMConfig] = None,
        timeout: float = 5.0,
        reconnect_delay: float = 0.5,
        max_reconnect_delay: float = 30.0,
        auto_reconnect: bool = True,
        emit_interval_sec: float = 0.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.host = str(host)
        self.port = int(port)
        self.streams = list(streams) if streams else []
        self.ring_buffer = ring_buffer if ring_buffer is not None else WaveformRingBuffer()
        self.shm_config = shm_config or get_config()
        self.timeout = float(timeout)
        self.reconnect_delay = max(0.05, float(reconnect_delay))
        self.max_reconnect_delay = max(self.reconnect_delay, float(max_reconnect_delay))
        self.auto_reconnect = bool(auto_reconnect)
        self.emit_interval_sec = max(0.0, float(emit_interval_sec))

        self._stop_event = threading.Event()
        self._client = SeedLinkClient(
            host=self.host,
            port=self.port,
            streams=self.streams,
            timeout=self.timeout,
            config=self.shm_config,
            ring_buffer=self.ring_buffer,
            stop_event=self._stop_event,
        )

    def stop(self, wait_ms: int = 3000) -> None:
        """Request clean cancellation, unblock socket I/O, and optionally wait for thread exit."""
        self._stop_event.set()
        self.requestInterruption()
        self._client.request_stop()
        if wait_ms > 0 and self.isRunning() and QThread.currentThread() is not self:
            self.wait(wait_ms)

    def _should_stop(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        self._stop_event.clear()
        self._client.stop_event = self._stop_event
        delay = self.reconnect_delay
        last_trace_emit = 0.0
        self.statusChanged.emit("connecting")

        try:
            while not self._should_stop():
                try:
                    self._client.connect()
                    self._client.negotiate(send_hello=True)
                    delay = self.reconnect_delay
                    self.statusChanged.emit("streaming")

                    while not self._should_stop():
                        pkt: Optional[SeedLinkPacket] = self._client.read_packet()
                        if pkt is None:
                            self.statusChanged.emit("completed")
                            return

                        if not pkt.is_valid and pkt.error:
                            self.errorOccurred.emit(pkt.error)
                            continue

                        if pkt.is_info:
                            self.packetReceived.emit(pkt)
                            continue

                        self.packetReceived.emit(pkt)
                        now_m = time.monotonic()
                        if self.emit_interval_sec <= 0.0 or (now_m - last_trace_emit) >= self.emit_interval_sec:
                            last_trace_emit = now_m
                            self.tracesUpdated.emit(self.ring_buffer.snapshot_traces())

                except SeedLinkStoppedError:
                    break
                except (SeedLinkError, OSError) as exc:
                    if self._should_stop():
                        break
                    self.errorOccurred.emit(str(exc))
                    if not self.auto_reconnect:
                        self.statusChanged.emit("error")
                        break
                    self.statusChanged.emit("reconnecting")
                    self._client.close()
                    if self._stop_event.wait(delay):
                        break
                    delay = min(self.max_reconnect_delay, delay * 2.0)
        finally:
            self._client.close()
            self.statusChanged.emit("stopped")
