"""Background orchestration service for Live Event Mode (streams, scheduling, retries, and fetch queue)."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import AnalysisParamSet, Trace, normalize_shm_event_type
from shm_app.io.seedlink import (
    SEEDLINK_DEFAULT_PORT,
    SeedLinkClient,
    SeedLinkError,
    SeedLinkPacket,
    SeedLinkStoppedError,
    WaveformRingBuffer,
)
from shm_app.plugins.bgr_fetch import (
    BGRFetchSummary,
    HttpOpener,
    fetch_fdsn_waveforms,
)
from shm_app.plugins.live_events import (
    EMSCEventStream,
    EventDeduplicator,
    LiveEvent,
    fetch_events_from_source,
)
from shm_app.plugins.live_planner import (
    EventEvaluation,
    EventFetchPlan,
    LiveConfig,
    evaluate_event_relevance,
    plan_event_fetch_window,
    resolve_reference_coords,
)
from shm_app.seismology.traveltime import TravelTimeManager, get_ttt_manager

# Canonical lifecycle states
STATE_DETECTED = "DETECTED"
STATE_WAITING = "WAITING_ARRIVAL"
STATE_FETCHING = "FETCHING"
STATE_RETRY_WAIT = "RETRY_WAIT"
STATE_READY = "READY"
STATE_LOADED = "LOADED"
STATE_IGNORED = "IGNORED"
STATE_ERROR = "ERROR"


@dataclass
class LiveEventEntry:
    """Tracked state for a single detected earthquake across discovery, fetch, and loading."""

    event: LiveEvent
    evaluation: EventEvaluation
    plan: Optional[EventFetchPlan] = None
    state: str = STATE_DETECTED
    status_detail: str = ""
    retry_count: int = 0
    next_action_epoch: float = 0.0
    fetched_traces: List[Trace] = field(default_factory=list)
    last_summary: Optional[BGRFetchSummary] = None
    notified: bool = False
    forced: bool = False


class CatalogFetchWorker(QThread):
    """Background HTTP catalog query worker (catch-up, polling, or 24h replay)."""

    eventsFetched = pyqtSignal(list, str)  # (List[LiveEvent], context_label)
    fetchFailed = pyqtSignal(str, str)  # (error_msg, context_label)

    def __init__(
        self,
        source_key: str,
        lookback_hours: float,
        min_mag: Optional[float],
        limit: int,
        context_label: str,
        now_epoch: float,
        opener: Optional[HttpOpener] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.source_key = source_key
        self.lookback_hours = float(lookback_hours)
        self.min_mag = min_mag
        self.limit = int(limit)
        self.context_label = context_label
        self.now_epoch = float(now_epoch)
        self.opener = opener

    def run(self) -> None:
        try:
            events = fetch_events_from_source(
                source_key=self.source_key,
                lookback_hours=self.lookback_hours,
                min_mag=self.min_mag,
                limit=self.limit,
                now_epoch=self.now_epoch,
                opener=self.opener,
            )
            if not self.isInterruptionRequested():
                self.eventsFetched.emit(events, self.context_label)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.fetchFailed.emit(str(exc), self.context_label)


class EventFetchWorker(QThread):
    """Background worker resolving dynamic FDSNWS station endpoints and fetching Mini-SEED waveforms."""

    fetchCompleted = pyqtSignal(str, object, object)  # (event_id, EventFetchPlan, BGRFetchSummary)
    fetchFailed = pyqtSignal(str, str)  # (event_id, error_message)

    def __init__(
        self,
        event_id: str,
        plan: EventFetchPlan,
        live_cfg: LiveConfig,
        shm_cfg: SHMConfig,
        ttt: TravelTimeManager,
        now_epoch: float,
        allow_network_discovery: bool = True,
        opener: Optional[HttpOpener] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.event_id = event_id
        self.plan = plan
        self.live_cfg = copy.deepcopy(live_cfg)
        self.shm_cfg = shm_cfg
        self.ttt = ttt
        self.now_epoch = float(now_epoch)
        self.allow_network_discovery = allow_network_discovery
        self.opener = opener

    def run(self) -> None:
        try:
            active_plan = self.plan
            if self.live_cfg.station_strategy == "epicenter" and self.allow_network_discovery:
                # Perform FDSNWS station / EIDA routing HTTP lookups off the GUI thread
                active_plan = plan_event_fetch_window(
                    event=self.plan.event,
                    evaluation=self.plan.evaluation,
                    live_cfg=self.live_cfg,
                    shm_cfg=self.shm_cfg,
                    ttt=self.ttt,
                    opener=self.opener,
                    allow_network=True,
                    now_epoch=self.now_epoch,
                )

            if self.isInterruptionRequested():
                return

            summary = fetch_fdsn_waveforms(
                targets=active_plan.targets,
                start_time=active_plan.start_epoch,
                duration=active_plan.duration_sec,
                output_dir=active_plan.output_dir,
                build_sfd=True,
                config=self.shm_cfg,
                opener=self.opener,
                cancel_check=self.isInterruptionRequested,
            )
            self.fetchCompleted.emit(self.event_id, active_plan, summary)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.fetchFailed.emit(self.event_id, str(exc))


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


def build_analysis_param_set(
    entry: LiveEventEntry,
    shm_cfg: Optional[SHMConfig] = None,
    live_cfg: Optional[LiveConfig] = None,
) -> AnalysisParamSet:
    """Populate a full SHM AnalysisParamSet from a LiveEventEntry."""
    cfg = shm_cfg or get_config()
    lcfg = live_cfg or LiveConfig()
    ev = entry.event
    eva = entry.evaluation
    ref_lat, ref_lon, _ref_code = resolve_reference_coords(lcfg, cfg)

    mt = (ev.mag_type or "").upper()
    pset = AnalysisParamSet(
        distance=round(eva.dist_deg, 3),
        azimuth=round(eva.baz_deg, 2),
        slowness=round(eva.slowness_s_deg, 3),
        depth=round(ev.depth_km, 1),
        depth_type=1,
        origin=ev.origin_sh,
        lat=round(ev.lat, 4),
        lon=round(ev.lon, 4),
        phase=eva.first_phase,
        region_id=eva.region_id,
        region_name=eva.region_name,
        sregion_id=eva.sregion_id,
        sregion_name=eva.sregion_name,
        ref_lat=round(ref_lat, 4),
        ref_lon=round(ref_lon, 4),
        source=ev.agency or ev.source_feed,
        event_type=normalize_shm_event_type(ev.event_type, tier=eva.tier, dist_deg=eva.dist_deg),
    )
    if "ML" in mt:
        pset.mag_ml = round(ev.magnitude, 2)
    elif "MS" in mt:
        pset.mag_ms = round(ev.magnitude, 2)
    elif mt == "MBB":
        pset.mag_mbb = round(ev.magnitude, 2)
    elif "MB" in mt:
        pset.mag_mb = round(ev.magnitude, 2)
    else:
        pset.mag_mw = round(ev.magnitude, 2)
    return pset


class LiveEventCoordinator(QObject):
    """Coordinates live streams, relevance filtering, arrival waiting, retries, and auto-loading."""

    enabledChanged = pyqtSignal(bool)
    streamStatusChanged = pyqtSignal(str, str)  # (state, detail)
    entriesUpdated = pyqtSignal()
    countdownTick = pyqtSignal()
    eventReadyToNotify = pyqtSignal(str)  # event_id
    loadEventRequested = pyqtSignal(str, bool)  # (event_id, auto_triggered)
    logMessage = pyqtSignal(str)
    seedlinkPacketReceived = pyqtSignal(object)
    seedlinkTracesUpdated = pyqtSignal(list)
    seedlinkStatusChanged = pyqtSignal(str)
    seedlinkErrorOccurred = pyqtSignal(str)

    def __init__(
        self,
        shm_config: Optional[SHMConfig] = None,
        ttt: Optional[TravelTimeManager] = None,
        live_config: Optional[LiveConfig] = None,
        is_workspace_idle_cb: Optional[Callable[[], bool]] = None,
        opener: Optional[HttpOpener] = None,
        ws_factory: Optional[Callable[[QObject], object]] = None,
        time_fn: Callable[[], float] = time.time,
        allow_network_discovery: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.shm_config = shm_config or get_config()
        self.ttt = ttt or get_ttt_manager(self.shm_config.inputs_dir)
        self.live_config = live_config or LiveConfig()
        self._is_workspace_idle_cb = is_workspace_idle_cb or (lambda: True)
        self._opener = opener
        self._time_fn = time_fn
        self._allow_network_discovery = allow_network_discovery

        self._enabled = False
        self._stream_state = "stopped"
        self._stream_detail = "Live Event Mode is OFF"
        self._dedup = EventDeduplicator()
        self.entries: Dict[str, LiveEventEntry] = {}
        self.ordered_ids: List[str] = []

        self._fetch_queue: List[str] = []
        self._active_fetch_id: Optional[str] = None
        self._fetch_worker: Optional[EventFetchWorker] = None
        self._catalog_worker: Optional[CatalogFetchWorker] = None
        self._seedlink_buffer = WaveformRingBuffer()
        self._seedlink_worker: Optional[SeedLinkStreamWorker] = None
        self._seedlink_status: str = "stopped"

        self.stream = EMSCEventStream(ws_factory=ws_factory, parent=self)
        self.stream.eventReceived.connect(self.ingest_event)
        self.stream.connectionStateChanged.connect(self._on_ws_state_changed)

        self._scheduler_timer = QTimer(self)
        self._scheduler_timer.setInterval(1000)
        self._scheduler_timer.timeout.connect(self._on_scheduler_tick)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._on_poll_tick)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def stream_state(self) -> str:
        return self._stream_state

    @property
    def stream_detail(self) -> str:
        return self._stream_detail

    def now(self) -> float:
        return float(self._time_fn())

    def set_enabled(self, enabled: bool, run_catchup: bool = True) -> None:
        """Turn Live Event Mode ON or OFF."""
        want = bool(enabled)
        if want == self._enabled:
            return
        self._enabled = want
        if want:
            self._scheduler_timer.start()
            self._start_active_feed(run_catchup=run_catchup)
            self._log("Live Event Mode ENABLED.")
        else:
            self._stop_active_feed()
            self._scheduler_timer.stop()
            self._poll_timer.stop()
            self._fetch_queue.clear()
            self._set_stream_status("stopped", "Live Event Mode is OFF")
            self._log("Live Event Mode DISABLED.")
        self.enabledChanged.emit(self._enabled)

    def update_config(self, new_cfg: LiveConfig, reevaluate_existing: bool = True) -> None:
        """Apply updated LiveConfig and restart transport if the feed source changed."""
        old_source = self.live_config.source_key
        self.live_config = copy.deepcopy(new_cfg)
        if self._enabled and old_source != self.live_config.source_key:
            self._stop_active_feed()
            self._start_active_feed(run_catchup=True)
        if reevaluate_existing:
            self._reevaluate_pending_entries()
        self.entriesUpdated.emit()

    def _set_stream_status(self, state: str, detail: str) -> None:
        self._stream_state = state
        self._stream_detail = detail
        self.streamStatusChanged.emit(state, detail)

    def _start_active_feed(self, run_catchup: bool = True) -> None:
        src = self.live_config.source_key
        if src == "emsc_ws":
            self.stream.start()
            self._poll_timer.setInterval(180000)  # Periodic HTTP safety catch-up every 3 min
            self._poll_timer.start()
        else:
            self.stream.stop()
            self._set_stream_status("polling", f"HTTP Polling ({src}) active")
            self._poll_timer.setInterval(45000)
            self._poll_timer.start()

        if run_catchup:
            self.fetch_recent_catalog(lookback_hours=2.0, context_label="Catch-Up (2h)")

    def _stop_active_feed(self) -> None:
        self.stream.stop()
        self._poll_timer.stop()

    def _on_ws_state_changed(self, state: str, detail: str) -> None:
        if not self._enabled or self.live_config.source_key != "emsc_ws":
            return
        self._set_stream_status(state, detail)
        if state == "connected":
            # Catch up any events missed while disconnected
            self.fetch_recent_catalog(lookback_hours=1.0, context_label="WS Reconnect Catch-Up")

    def _on_poll_tick(self) -> None:
        if not self._enabled:
            return
        self.fetch_recent_catalog(lookback_hours=2.0, context_label="Scheduled Poll")

    def fetch_recent_catalog(self, lookback_hours: float = 24.0, context_label: str = "Replay") -> None:
        """Query recent catalog events asynchronously without blocking the GUI thread."""
        if self._catalog_worker is not None and self._catalog_worker.isRunning():
            return
        min_mag = min(
            self.live_config.min_mag_local,
            self.live_config.min_mag_regional,
            self.live_config.min_mag_tele,
            self.live_config.flat_min_mag,
        )
        worker = CatalogFetchWorker(
            source_key=self.live_config.source_key,
            lookback_hours=lookback_hours,
            min_mag=max(1.0, min_mag - 0.2),
            limit=100,
            context_label=context_label,
            now_epoch=self.now(),
            opener=self._opener,
            parent=self,
        )
        worker.eventsFetched.connect(self._on_catalog_fetched)
        worker.fetchFailed.connect(self._on_catalog_failed)
        self._catalog_worker = worker
        worker.start()

    def _on_catalog_fetched(self, events: list, context_label: str) -> None:
        self._catalog_worker = None
        # Process oldest-to-newest so newest ends up on top of ordered_ids
        ordered = sorted(events, key=lambda e: e.origin_epoch)
        n_new = 0
        for ev in ordered:
            before = ev.event_id in self.entries
            self.ingest_event(ev)
            if not before:
                n_new += 1
        self._log(f"{context_label}: processed {len(ordered)} catalog event(s) ({n_new} new).")
        if self._enabled and self.live_config.source_key != "emsc_ws":
            self._set_stream_status("polling", f"Last poll OK ({len(ordered)} events)")
        self._dispatch_queue()

    def _on_catalog_failed(self, err: str, context_label: str) -> None:
        self._catalog_worker = None
        self._log(f"{context_label} failed: {err}")
        if self._enabled and self.live_config.source_key != "emsc_ws":
            self._set_stream_status("error", f"Poll failed: {err}")

    def ingest_event(self, ev: LiveEvent) -> Optional[LiveEventEntry]:
        """Deduplicate, evaluate relevance, plan fetch timing, and update state machine."""
        kind, canonical = self._dedup.ingest(ev)
        if kind == "duplicate":
            return self.entries.get(canonical.event_id)

        evaluation = evaluate_event_relevance(canonical, self.live_config, self.shm_config, self.ttt)
        now_ep = self.now()

        if kind == "new":
            entry = self._create_new_entry(canonical, evaluation, now_ep)
            self.entries[canonical.event_id] = entry
            self.ordered_ids.insert(0, canonical.event_id)
            if entry.state in (STATE_WAITING, STATE_FETCHING):
                self._log(
                    f"Detected {canonical.event_id}: M{canonical.magnitude:.1f} {evaluation.region_display} "
                    f"(Δ={evaluation.dist_deg:.1f}°, {evaluation.tier}) -> {entry.status_detail}"
                )
        else:
            entry = self._apply_event_update(canonical, evaluation, now_ep)

        self.entriesUpdated.emit()
        self._dispatch_queue()
        return entry

    def _create_new_entry(self, ev: LiveEvent, evaluation: EventEvaluation, now_ep: float) -> LiveEventEntry:
        if not evaluation.relevant:
            return LiveEventEntry(
                event=ev,
                evaluation=evaluation,
                plan=None,
                state=STATE_IGNORED,
                status_detail=evaluation.reason,
            )
        # Immediate non-blocking plan (for epicenter strategy, network station lookup runs in worker)
        plan = plan_event_fetch_window(
            event=ev,
            evaluation=evaluation,
            live_cfg=self.live_config,
            shm_cfg=self.shm_config,
            ttt=self.ttt,
            opener=self._opener,
            allow_network=False,
            now_epoch=now_ep,
        )
        detail = self._waiting_detail_text(plan, now_ep)
        return LiveEventEntry(
            event=ev,
            evaluation=evaluation,
            plan=plan,
            state=STATE_WAITING,
            status_detail=detail,
            next_action_epoch=plan.ready_at_epoch,
        )

    def _apply_event_update(self, ev: LiveEvent, evaluation: EventEvaluation, now_ep: float) -> LiveEventEntry:
        entry = self.entries[ev.event_id]
        entry.event = ev
        entry.evaluation = evaluation

        # Don't revert loaded/active/forced entries unless currently ignored & now qualifies!
        if entry.state in (STATE_LOADED, STATE_READY, STATE_FETCHING):
            return entry

        if not evaluation.relevant and not entry.forced:
            entry.state = STATE_IGNORED
            entry.status_detail = f"Updated -> {evaluation.reason}"
            return entry

        # Re-plan window with refined origin/coords/depth/magnitude
        plan = plan_event_fetch_window(
            event=ev,
            evaluation=evaluation,
            live_cfg=self.live_config,
            shm_cfg=self.shm_config,
            ttt=self.ttt,
            opener=self._opener,
            allow_network=False,
            now_epoch=now_ep,
        )
        entry.plan = plan
        if entry.state in (STATE_IGNORED, STATE_WAITING, STATE_RETRY_WAIT):
            entry.state = STATE_WAITING
            entry.next_action_epoch = plan.ready_at_epoch
            entry.status_detail = self._waiting_detail_text(plan, now_ep)
        return entry

    def _reevaluate_pending_entries(self) -> None:
        now_ep = self.now()
        for eid in self.ordered_ids:
            entry = self.entries[eid]
            if entry.state in (STATE_LOADED, STATE_READY, STATE_FETCHING):
                continue
            eva = evaluate_event_relevance(entry.event, self.live_config, self.shm_config, self.ttt)
            entry.evaluation = eva
            if not eva.relevant and not entry.forced:
                entry.state = STATE_IGNORED
                entry.status_detail = eva.reason
            else:
                plan = plan_event_fetch_window(
                    entry.event,
                    eva,
                    self.live_config,
                    self.shm_config,
                    self.ttt,
                    opener=self._opener,
                    allow_network=False,
                    now_epoch=now_ep,
                )
                entry.plan = plan
                entry.state = STATE_WAITING
                entry.next_action_epoch = plan.ready_at_epoch
                entry.status_detail = self._waiting_detail_text(plan, now_ep)

    def _waiting_detail_text(self, plan: EventFetchPlan, now_ep: float) -> str:
        rem_ready = plan.ready_at_epoch - now_ep
        n_stas = len(plan.targets)
        if rem_ready <= 0.5:
            return f"Queued ({n_stas} stas)"
        if plan.earliest_arrival_epoch - now_ep > 5.0:
            return f"Fetch in {self._fmt_mmss(rem_ready)} ({n_stas} stas)"
        return f"Buffering archive {self._fmt_mmss(rem_ready)} ({n_stas} stas)"

    @staticmethod
    def _fmt_mmss(secs: float) -> str:
        s = max(0, int(round(secs)))
        mins, sec = divmod(s, 60)
        return f"{mins:02d}m {sec:02d}s" if mins else f"{sec:02d}s"

    def force_fetch_event(self, event_id: str) -> None:
        """Manually queue or re-fetch any detected event immediately (even IGNORED or ERROR)."""
        entry = self.entries.get(event_id)
        if entry is None:
            return
        if entry.state == STATE_FETCHING:
            return
        now_ep = self.now()
        entry.forced = True
        entry.retry_count = 0
        if entry.plan is None:
            entry.plan = plan_event_fetch_window(
                entry.event,
                entry.evaluation,
                self.live_config,
                self.shm_config,
                self.ttt,
                opener=self._opener,
                allow_network=False,
                now_epoch=now_ep,
            )
        entry.state = STATE_WAITING
        entry.next_action_epoch = now_ep
        entry.status_detail = "Manual fetch queued"
        if event_id not in self._fetch_queue:
            self._fetch_queue.insert(0, event_id)
        self.entriesUpdated.emit()
        self._dispatch_queue()

    def ignore_event(self, event_id: str) -> None:
        """Mark event IGNORED and remove from pending queue."""
        entry = self.entries.get(event_id)
        if entry is None:
            return
        if event_id in self._fetch_queue:
            self._fetch_queue.remove(event_id)
        entry.forced = False
        entry.state = STATE_IGNORED
        entry.status_detail = "Ignored by user"
        self.entriesUpdated.emit()

    def request_load_event(self, event_id: str, auto_triggered: bool = False) -> None:
        """Emit workspace load signal and mark entry LOADED."""
        entry = self.entries.get(event_id)
        if entry is None or not entry.fetched_traces:
            return
        entry.state = STATE_LOADED
        entry.status_detail = f"Loaded {len(entry.fetched_traces)} traces"
        self.entriesUpdated.emit()
        self.loadEventRequested.emit(event_id, auto_triggered)

    def _on_scheduler_tick(self) -> None:
        self._dispatch_queue()
        self.countdownTick.emit()

    def process_due_fetches(self) -> None:
        """Synchronously advance scheduler queue (handy for tests as well as timer ticks)."""
        self._dispatch_queue()

    def _dispatch_queue(self) -> None:
        now_ep = self.now()
        if self._enabled:
            for eid in self.ordered_ids:
                entry = self.entries[eid]
                if entry.state in (STATE_WAITING, STATE_RETRY_WAIT) and entry.next_action_epoch <= now_ep:
                    if eid != self._active_fetch_id and eid not in self._fetch_queue:
                        self._fetch_queue.append(eid)

        if self._fetch_worker is not None and self._fetch_worker.isRunning():
            return
        if not self._fetch_queue:
            return

        next_id = self._fetch_queue.pop(0)
        entry = self.entries.get(next_id)
        if entry is None or entry.state in (STATE_IGNORED, STATE_LOADED, STATE_READY):
            self._dispatch_queue()
            return
        if entry.plan is None:
            entry.plan = plan_event_fetch_window(
                entry.event,
                entry.evaluation,
                self.live_config,
                self.shm_config,
                self.ttt,
                opener=self._opener,
                allow_network=False,
                now_epoch=now_ep,
            )

        self._active_fetch_id = next_id
        entry.state = STATE_FETCHING
        entry.status_detail = f"Fetching {len(entry.plan.targets)} stations..."
        self.entriesUpdated.emit()

        worker = EventFetchWorker(
            event_id=next_id,
            plan=entry.plan,
            live_cfg=self.live_config,
            shm_cfg=self.shm_config,
            ttt=self.ttt,
            now_epoch=now_ep,
            allow_network_discovery=self._allow_network_discovery,
            opener=self._opener,
            parent=self,
        )
        worker.fetchCompleted.connect(self._on_event_fetch_completed)
        worker.fetchFailed.connect(self._on_event_fetch_failed)
        self._fetch_worker = worker
        worker.start()

    def _on_event_fetch_completed(self, event_id: str, plan: EventFetchPlan, summary: BGRFetchSummary) -> None:
        self._fetch_worker = None
        self._active_fetch_id = None
        entry = self.entries.get(event_id)
        if entry is None:
            self._dispatch_queue()
            return
        entry.plan = plan
        entry.last_summary = summary

        if entry.state == STATE_IGNORED:
            self._dispatch_queue()
            return

        if summary.traces:
            entry.fetched_traces = list(summary.traces)
            ok_stas = sum(1 for r in summary.results if r.status == "ok")
            entry.status_detail = f"{len(summary.traces)} traces ({ok_stas}/{len(summary.results)} stas)"
            self._log(
                f"Live Event {entry.event.event_id} READY: {entry.status_detail} saved in {summary.output_dir}"
            )
            policy = self.live_config.load_policy
            if policy == "auto_always":
                self.request_load_event(event_id, auto_triggered=True)
            elif policy == "auto_idle" and self._is_workspace_idle_cb():
                self.request_load_event(event_id, auto_triggered=True)
            else:
                entry.state = STATE_READY
                if not entry.notified:
                    entry.notified = True
                    self.eventReadyToNotify.emit(event_id)
        else:
            self._schedule_retry_or_error(entry, reason="No archive data yet (204/nodata)")

        self.entriesUpdated.emit()
        self._dispatch_queue()

    def _on_event_fetch_failed(self, event_id: str, err: str) -> None:
        self._fetch_worker = None
        self._active_fetch_id = None
        entry = self.entries.get(event_id)
        if entry is not None and entry.state != STATE_IGNORED:
            self._schedule_retry_or_error(entry, reason=err)
            self.entriesUpdated.emit()
        self._dispatch_queue()

    def _schedule_retry_or_error(self, entry: LiveEventEntry, reason: str) -> None:
        delays = tuple(self.live_config.retry_delays_sec) or (90, 180, 360)
        if entry.retry_count < len(delays):
            delay = int(delays[entry.retry_count])
            entry.retry_count += 1
            entry.state = STATE_RETRY_WAIT
            entry.next_action_epoch = self.now() + delay
            entry.status_detail = f"Retry {entry.retry_count}/{len(delays)} in {delay}s ({reason})"
            self._log(f"Event {entry.event.event_id}: {entry.status_detail}")
        else:
            entry.state = STATE_ERROR
            entry.status_detail = f"No waveforms after {len(delays)} retries ({reason})"
            self._log(f"Event {entry.event.event_id}: {entry.status_detail}")

    def countdown_text(self, event_id: str) -> str:
        """Human-friendly ETA / status cell text updated cleanly on each timer tick."""
        entry = self.entries.get(event_id)
        if entry is None:
            return ""
        now_ep = self.now()
        if entry.state == STATE_WAITING:
            rem = entry.next_action_epoch - now_ep
            if rem <= 0.5:
                return "Fetch due..."
            p_rem = entry.evaluation.first_arrival_epoch - now_ep
            if p_rem > 5.0:
                return f"{entry.evaluation.first_phase} in {self._fmt_mmss(p_rem)}"
            return f"Ready in {self._fmt_mmss(rem)}"
        if entry.state == STATE_RETRY_WAIT:
            rem = max(0.0, entry.next_action_epoch - now_ep)
            return f"Retry #{entry.retry_count} in {self._fmt_mmss(rem)}"
        if entry.state == STATE_FETCHING:
            return "Fetching..."
        return entry.status_detail

    @property
    def seedlink_status(self) -> str:
        return self._seedlink_status

    @property
    def is_seedlink_running(self) -> bool:
        return self._seedlink_worker is not None and self._seedlink_worker.isRunning()

    def start_seedlink(
        self,
        host: str,
        port: int = SEEDLINK_DEFAULT_PORT,
        streams: Optional[Sequence[Any]] = None,
        max_samples_per_stream: int = 36000,
        max_traces: int = 128,
        timeout: float = 5.0,
        auto_reconnect: bool = True,
        reconnect_delay: float = 0.5,
        emit_interval_sec: float = 0.0,
        clear_buffer: bool = False,
        ring_buffer: Optional[WaveformRingBuffer] = None,
    ) -> SeedLinkStreamWorker:
        """Start asynchronous SeedLink v3.1 streaming into the coordinator ring buffer."""
        self.stop_seedlink()
        if ring_buffer is not None:
            self._seedlink_buffer = ring_buffer
        elif (
            self._seedlink_buffer.max_samples_per_stream != int(max_samples_per_stream)
            or self._seedlink_buffer.max_traces != int(max_traces)
        ):
            self._seedlink_buffer = WaveformRingBuffer(
                max_samples_per_stream=max_samples_per_stream,
                max_traces=max_traces,
            )
        elif clear_buffer:
            self._seedlink_buffer.clear()

        worker = SeedLinkStreamWorker(
            host=host,
            port=port,
            streams=streams,
            ring_buffer=self._seedlink_buffer,
            shm_config=self.shm_config,
            timeout=timeout,
            reconnect_delay=reconnect_delay,
            auto_reconnect=auto_reconnect,
            emit_interval_sec=emit_interval_sec,
            parent=self,
        )
        worker.packetReceived.connect(self.seedlinkPacketReceived)
        worker.tracesUpdated.connect(self.seedlinkTracesUpdated)
        worker.statusChanged.connect(self._on_seedlink_status_changed)
        worker.errorOccurred.connect(self._on_seedlink_error)
        self._seedlink_worker = worker
        worker.start()
        self._log(f"SeedLink streaming started ({host}:{port}).")
        return worker

    def stop_seedlink(self) -> None:
        """Stop any active SeedLink streaming worker cleanly."""
        worker = self._seedlink_worker
        self._seedlink_worker = None
        if worker is not None:
            worker.stop(wait_ms=3000)
        if self._seedlink_status != "stopped":
            self._seedlink_status = "stopped"
            self.seedlinkStatusChanged.emit("stopped")

    def get_seedlink_traces(self, stream_key: Optional[str] = None) -> List[Trace]:
        """Return a thread-safe snapshot of buffered SeedLink Trace waveforms."""
        return self._seedlink_buffer.snapshot_traces(stream_key=stream_key)

    def _on_seedlink_status_changed(self, state: str) -> None:
        self._seedlink_status = state
        self.seedlinkStatusChanged.emit(state)

    def _on_seedlink_error(self, err: str) -> None:
        self._log(f"SeedLink warning: {err}")
        self.seedlinkErrorOccurred.emit(err)

    def _log(self, msg: str) -> None:
        self.logMessage.emit(f"[Live] {msg}")

    def shutdown(self) -> None:
        """Stop timers, WebSocket stream, SeedLink stream, and background workers cleanly before exit."""
        self.set_enabled(False)
        self.stop_seedlink()
        for attr in ("_catalog_worker", "_fetch_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(3000)
            setattr(self, attr, None)
