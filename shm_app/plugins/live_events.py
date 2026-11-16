"""Real-time and catch-up seismic event catalog feeds for SHM Live Event Mode.

Supports:
- EMSC SeismicPortal real-time WebSocket stream (`wss://www.seismicportal.eu/standing_order/websocket`)
- FDSNWS Event (`format=text`) feeds (GFZ GEOFON, EMSC) for catch-up and HTTP polling
- USGS FDSNWS / GeoJSON event feed
- Cross-feed spatial/temporal deduplication and update detection
"""

from __future__ import annotations

import datetime
import json
import math
import time
import urllib.parse
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtWebSockets import QWebSocket

from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.plugins.bgr_fetch import HttpOpener, _default_http_open, format_fdsn_time
from shm_app.seismology.traveltime import sphere_diff

EMSC_WS_URL = "wss://www.seismicportal.eu/standing_order/websocket"
EMSC_FDSN_EVENT_URL = "https://www.seismicportal.eu/fdsnws/event/1/query"
GFZ_FDSN_EVENT_URL = "https://geofon.gfz.de/fdsnws/event/1/query"
USGS_FDSN_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

FEED_SOURCES: Dict[str, str] = {
    "emsc_ws": "EMSC WebSocket (Real-Time) + Catch-Up",
    "gfz_fdsn": "GFZ GEOFON FDSNWS Poll",
    "emsc_fdsn": "EMSC FDSNWS Poll",
    "usgs_fdsn": "USGS GeoJSON Poll",
}


@dataclass(frozen=True)
class LiveEvent:
    """Normalized earthquake catalog entry from WebSocket or FDSNWS feeds."""

    event_id: str
    origin_epoch: float
    origin_iso: str
    lat: float
    lon: float
    depth_km: float
    magnitude: float
    mag_type: str = "M"
    region: str = ""
    agency: str = ""
    source_feed: str = ""
    action: str = "create"
    updated_epoch: float = 0.0
    event_type: str = "earthquake"
    url: str = ""

    @property
    def origin_sh(self) -> str:
        """Return origin time formatted as SH timestamp DD-MMM-YYYY_HH:MM:SS.SSS."""
        return format_sh_time(self.origin_epoch)


def parse_iso_utc_epoch(value: str) -> float:
    """Parse an ISO-8601 UTC timestamp string (with optional Z/offset) into epoch seconds."""
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp string")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).timestamp()
    except ValueError:
        return parse_sh_time(value)


def _safe_float(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        out = float(val)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def parse_emsc_ws_message(text: str) -> Optional[LiveEvent]:
    """Parse a JSON push message from EMSC SeismicPortal WebSocket."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    action = str(payload.get("action") or "create").lower()
    if action not in ("create", "update", "delete"):
        action = "create"

    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    props = data.get("properties")
    if not isinstance(props, dict):
        return None

    coords = (data.get("geometry") or {}).get("coordinates") or []
    lon = _safe_float(props.get("lon"), default=_safe_float(coords[0] if len(coords) > 0 else None, math.nan))
    lat = _safe_float(props.get("lat"), default=_safe_float(coords[1] if len(coords) > 1 else None, math.nan))
    depth_fallback = abs(_safe_float(coords[2] if len(coords) > 2 else None, 10.0))
    depth = _safe_float(props.get("depth"), default=depth_fallback)
    if depth < 0.0:
        depth = abs(depth)

    time_str = str(props.get("time") or "").strip()
    if not time_str or not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    try:
        origin_ep = parse_iso_utc_epoch(time_str)
    except ValueError:
        return None

    event_id = str(props.get("unid") or data.get("id") or "").strip()
    if not event_id:
        return None

    mag = _safe_float(props.get("mag"), default=0.0)
    mag_type = str(props.get("magtype") or "M").strip() or "M"
    region = str(props.get("flynn_region") or "").strip()
    agency = str(props.get("auth") or "EMSC").strip()
    lastupdate_str = str(props.get("lastupdate") or "").strip()
    updated_ep = origin_ep
    if lastupdate_str:
        try:
            updated_ep = parse_iso_utc_epoch(lastupdate_str)
        except ValueError:
            updated_ep = origin_ep

    return LiveEvent(
        event_id=event_id,
        origin_epoch=origin_ep,
        origin_iso=format_fdsn_time(origin_ep),
        lat=lat,
        lon=lon,
        depth_km=depth,
        magnitude=mag,
        mag_type=mag_type,
        region=region,
        agency=agency,
        source_feed="EMSC-WS",
        action=action,
        updated_epoch=updated_ep,
        event_type=str(props.get("evtype") or "earthquake").strip(),
    )


def parse_fdsn_event_text(text: str, source_feed: str = "GFZ-GEOFON") -> List[LiveEvent]:
    """Parse FDSNWS event/1/query `format=text` pipe-delimited catalog output."""
    events: List[LiveEvent] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 11:
            continue
        event_id = cols[0]
        time_str = cols[1]
        lat = _safe_float(cols[2], default=math.nan)
        lon = _safe_float(cols[3], default=math.nan)
        if not event_id or not time_str or not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        try:
            origin_ep = parse_iso_utc_epoch(time_str)
        except ValueError:
            continue
        depth = max(0.0, _safe_float(cols[4], default=10.0))
        agency = cols[5] or cols[7] or cols[6] or source_feed
        mag_type = cols[9] or "M"
        mag = _safe_float(cols[10], default=0.0)
        region = cols[12] if len(cols) > 12 else ""
        ev_type = (cols[13] if len(cols) > 13 and cols[13] else "earthquake")
        events.append(
            LiveEvent(
                event_id=event_id,
                origin_epoch=origin_ep,
                origin_iso=format_fdsn_time(origin_ep),
                lat=lat,
                lon=lon,
                depth_km=depth,
                magnitude=mag,
                mag_type=mag_type,
                region=region,
                agency=agency,
                source_feed=source_feed,
                action="create",
                updated_epoch=origin_ep,
                event_type=ev_type,
            )
        )
    return events


def parse_usgs_geojson(text: str, source_feed: str = "USGS") -> List[LiveEvent]:
    """Parse USGS GeoJSON feature collection response into LiveEvent records."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        return []

    out: List[LiveEvent] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon = _safe_float(coords[0], default=math.nan)
        lat = _safe_float(coords[1], default=math.nan)
        depth = max(0.0, _safe_float(coords[2] if len(coords) > 2 else None, default=10.0))
        t_ms = props.get("time")
        if t_ms is None or not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        try:
            origin_ep = float(t_ms) / 1000.0
        except (TypeError, ValueError):
            continue
        upd_ms = props.get("updated")
        updated_ep = (float(upd_ms) / 1000.0) if upd_ms is not None else origin_ep
        ev_id = str(feat.get("id") or f"{props.get('net', 'us')}{props.get('code', '')}").strip()
        if not ev_id:
            continue
        out.append(
            LiveEvent(
                event_id=ev_id,
                origin_epoch=origin_ep,
                origin_iso=format_fdsn_time(origin_ep),
                lat=lat,
                lon=lon,
                depth_km=depth,
                magnitude=_safe_float(props.get("mag"), default=0.0),
                mag_type=str(props.get("magType") or "M").strip() or "M",
                region=str(props.get("place") or "").strip(),
                agency=str(props.get("net") or "USGS").upper(),
                source_feed=source_feed,
                action="create",
                updated_epoch=updated_ep,
                event_type=str(props.get("type") or "earthquake").strip(),
                url=str(props.get("url") or "").strip(),
            )
        )
    return out


class EventDeduplicator:
    """Tracks seen earthquake IDs and detects cross-feed duplicates or revisions."""

    def __init__(
        self,
        max_entries: int = 2000,
        time_match_s: float = 20.0,
        dist_match_deg: float = 1.2,
    ) -> None:
        self.max_entries = max(50, int(max_entries))
        self.time_match_s = float(time_match_s)
        self.dist_match_deg = float(dist_match_deg)
        self._events_by_id: Dict[str, LiveEvent] = {}

    def clear(self) -> None:
        self._events_by_id.clear()

    def _find_matching_id(self, ev: LiveEvent) -> Optional[str]:
        if ev.event_id in self._events_by_id:
            return ev.event_id
        for key, prev in self._events_by_id.items():
            if abs(prev.origin_epoch - ev.origin_epoch) > self.time_match_s:
                continue
            dist_deg, _, _ = sphere_diff(prev.lat, prev.lon, ev.lat, ev.lon)
            if dist_deg <= self.dist_match_deg and abs(prev.magnitude - ev.magnitude) <= 1.2:
                return key
        return None

    def _is_material_update(self, prev: LiveEvent, new: LiveEvent) -> bool:
        if new.action == "update" and new.updated_epoch > prev.updated_epoch + 0.5:
            return True
        if abs(new.magnitude - prev.magnitude) >= 0.05:
            return True
        if abs(new.lat - prev.lat) >= 0.02 or abs(new.lon - prev.lon) >= 0.02:
            return True
        if abs(new.depth_km - prev.depth_km) >= 1.5:
            return True
        if not prev.region and new.region:
            return True
        return False

    def ingest(self, ev: LiveEvent) -> Tuple[str, LiveEvent]:
        """Register event and return (status, canonical_event).

        Status is 'new', 'updated', or 'duplicate'.
        Canonical event retains the primary tracked event_id when matched cross-feed.
        """
        matched_id = self._find_matching_id(ev)
        if matched_id is None:
            if len(self._events_by_id) >= self.max_entries:
                oldest = sorted(self._events_by_id.items(), key=lambda kv: kv[1].origin_epoch)
                for old_id, _ in oldest[: max(1, len(oldest) // 4)]:
                    self._events_by_id.pop(old_id, None)
            self._events_by_id[ev.event_id] = ev
            return "new", ev

        prev = self._events_by_id[matched_id]
        canonical = ev if ev.event_id == matched_id else replace(ev, event_id=matched_id)
        if self._is_material_update(prev, canonical):
            merged = replace(
                canonical,
                region=canonical.region or prev.region,
                agency=canonical.agency or prev.agency,
            )
            self._events_by_id[matched_id] = merged
            return "updated", merged
        return "duplicate", prev


def build_fdsn_event_query_url(
    base_url: str,
    start_epoch: float,
    end_epoch: Optional[float] = None,
    min_mag: Optional[float] = 1.5,
    limit: int = 100,
    fmt: str = "text",
) -> str:
    """Build an FDSNWS event/1/query URL for text or GeoJSON catalog retrieval."""
    params: Dict[str, str] = {
        "starttime": format_fdsn_time(start_epoch),
        "format": fmt,
        "orderby": "time",
        "limit": str(max(1, int(limit))),
    }
    if end_epoch is not None and end_epoch > start_epoch:
        params["endtime"] = format_fdsn_time(end_epoch)
    if min_mag is not None:
        params["minmagnitude"] = f"{float(min_mag):.2f}"
    sep = "&" if "?" in base_url else "?"
    return f"{base_url.rstrip('?&')}{sep}{urllib.parse.urlencode(params)}"


def fetch_events_from_source(
    source_key: str,
    lookback_hours: float = 2.0,
    min_mag: Optional[float] = 1.5,
    limit: int = 100,
    now_epoch: Optional[float] = None,
    opener: Optional[HttpOpener] = None,
    timeout: float = 15.0,
) -> List[LiveEvent]:
    """Query recent events from a configured catalog source synchronously."""
    now_ep = float(now_epoch) if now_epoch is not None else time.time()
    start_ep = now_ep - max(0.05, float(lookback_hours)) * 3600.0
    http_open = opener or _default_http_open

    if source_key == "usgs_fdsn":
        url = build_fdsn_event_query_url(
            USGS_FDSN_EVENT_URL,
            start_epoch=start_ep,
            end_epoch=now_ep,
            min_mag=min_mag,
            limit=limit,
            fmt="geojson",
        )
        status, raw = http_open(url, timeout)
        if status == 204 or not raw:
            return []
        if status != 200:
            raise RuntimeError(f"USGS HTTP {status}")
        return parse_usgs_geojson(raw.decode("utf-8", errors="replace"), source_feed="USGS")

    # Try EMSC / GFZ FDSNWS format=text; for emsc_ws catch-up fall back to GFZ if EMSC fails
    if source_key == "gfz_fdsn":
        candidates = [(GFZ_FDSN_EVENT_URL, "GFZ-GEOFON")]
    elif source_key == "emsc_fdsn":
        candidates = [(EMSC_FDSN_EVENT_URL, "EMSC-FDSN")]
    else:
        candidates = [(EMSC_FDSN_EVENT_URL, "EMSC-FDSN"), (GFZ_FDSN_EVENT_URL, "GFZ-GEOFON")]

    last_err: Optional[Exception] = None
    for base_url, feed_label in candidates:
        url = build_fdsn_event_query_url(
            base_url,
            start_epoch=start_ep,
            end_epoch=now_ep,
            min_mag=min_mag,
            limit=limit,
            fmt="text",
        )
        try:
            status, raw = http_open(url, timeout)
        except Exception as exc:
            last_err = exc
            continue
        if status == 204 or not raw:
            return []
        if status != 200:
            last_err = RuntimeError(f"{feed_label} HTTP {status}")
            continue
        return parse_fdsn_event_text(raw.decode("utf-8", errors="replace"), source_feed=feed_label)

    if last_err is not None:
        raise RuntimeError(f"Catalog query failed: {last_err}") from last_err
    return []


class EMSCEventStream(QObject):
    """Qt WebSocket client subscribing to EMSC SeismicPortal real-time earthquakes."""

    eventReceived = pyqtSignal(object)  # LiveEvent
    connectionStateChanged = pyqtSignal(str, str)  # (state, detail)

    def __init__(
        self,
        ws_url: str = EMSC_WS_URL,
        ws_factory: Optional[Callable[[QObject], object]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.ws_url = ws_url
        self._ws_factory = ws_factory or (lambda p: QWebSocket("", parent=p))
        self._ws: Optional[object] = None
        self._running = False
        self._backoff_s = 5
        self._state = "disconnected"

        self._ping_timer = QTimer(self)
        self._ping_timer.setInterval(45000)
        self._ping_timer.timeout.connect(self._send_ping)

        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._open_socket)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self.connectionStateChanged.emit(state, detail)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._backoff_s = 5
        self._open_socket()

    def stop(self) -> None:
        self._running = False
        self._ping_timer.stop()
        self._reconnect_timer.stop()
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                ws.close()
                ws.deleteLater()
            except Exception:
                pass
        self._set_state("disconnected", "Live stream stopped")

    def _open_socket(self) -> None:
        if not self._running:
            return
        if self._ws is not None:
            old = self._ws
            self._ws = None
            try:
                old.abort()
                old.deleteLater()
            except Exception:
                pass

        ws = self._ws_factory(self)
        self._ws = ws
        ws.connected.connect(self._on_connected)
        ws.disconnected.connect(self._on_disconnected)
        ws.textMessageReceived.connect(self._on_text_message)
        if hasattr(ws, "errorOccurred"):
            ws.errorOccurred.connect(self._on_error)
        self._set_state("connecting", f"Connecting to {self.ws_url}...")
        ws.open(QUrl(self.ws_url))

    def _send_ping(self) -> None:
        if self._running and self._ws is not None:
            try:
                self._ws.ping(b"shm-live")
            except Exception:
                pass

    def _on_connected(self) -> None:
        self._backoff_s = 5
        self._ping_timer.start()
        self._set_state("connected", "EMSC WebSocket connected")

    def _on_disconnected(self) -> None:
        self._ping_timer.stop()
        if not self._running:
            return
        delay = self._backoff_s
        self._backoff_s = min(60, self._backoff_s * 2)
        self._set_state("reconnecting", f"EMSC disconnected — retrying in {delay}s")
        self._reconnect_timer.start(delay * 1000)

    def _on_error(self, _err) -> None:
        detail = ""
        if self._ws is not None and hasattr(self._ws, "errorString"):
            detail = self._ws.errorString()
        self._ping_timer.stop()
        if not self._running:
            return
        delay = self._backoff_s
        self._backoff_s = min(60, self._backoff_s * 2)
        msg = f"WebSocket error ({detail or 'network'}) — retrying in {delay}s"
        self._set_state("error", msg)
        if not self._reconnect_timer.isActive():
            self._reconnect_timer.start(delay * 1000)

    def _on_text_message(self, text: str) -> None:
        ev = parse_emsc_ws_message(text)
        if ev is not None:
            self.eventReceived.emit(ev)
