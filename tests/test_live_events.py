"""Tests for SHM Live Event Mode (feeds, deduplication, planning, scheduler, policies, GUI)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
import urllib.parse
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QPushButton

from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick
from shm_app.core.time_utils import parse_sh_time
from shm_app.gui.main_window import SHMMainWindow
from shm_app.plugins.bgr_dialog import BGRFetchDialog
from shm_app.plugins.live_dialog import LiveEventDialog
from shm_app.plugins.live_events import (
    EMSC_FDSN_EVENT_URL,
    EMSC_WS_URL,
    GFZ_FDSN_EVENT_URL,
    USGS_FDSN_EVENT_URL,
    EMSCEventStream,
    EventDeduplicator,
    LiveEvent,
    build_fdsn_event_query_url,
    fetch_events_from_source,
    format_fdsn_time,
    parse_emsc_ws_message,
    parse_fdsn_event_text,
    parse_usgs_geojson,
)
from shm_app.plugins.live_planner import (
    LiveConfig,
    evaluate_event_relevance,
    plan_event_fetch_window,
    select_epicenter_targets,
    select_observatory_targets,
)
from shm_app.plugins.live_service import (
    STATE_ERROR,
    STATE_IGNORED,
    STATE_LOADED,
    STATE_READY,
    STATE_RETRY_WAIT,
    STATE_WAITING,
    LiveEventCoordinator,
)
from tests.test_bgr_fetch import _make_steim1_mseed_bytes


class _FakeWebSocket(QObject):
    """Minimal QtWebSocket mock for deterministic offline WebSocket stream testing."""

    connected = pyqtSignal()
    disconnected = pyqtSignal()
    textMessageReceived = pyqtSignal(str)
    errorOccurred = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.opened_url = ""
        self.closed = False

    def open(self, url: QUrl) -> None:
        self.opened_url = url.toString()
        self.connected.emit()

    def close(self) -> None:
        self.closed = True
        self.disconnected.emit()

    def abort(self) -> None:
        self.closed = True

    def ping(self, _payload: bytes) -> None:
        pass

    def errorString(self) -> str:
        return "mock error"


class _MockWebSocket(QObject):
    """Detailed mock for testing WebSocket lifecycle, manual connects, pings, and backoff."""

    connected = pyqtSignal()
    disconnected = pyqtSignal()
    textMessageReceived = pyqtSignal(str)
    errorOccurred = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.opened_url: str = ""
        self.closed: bool = False
        self.aborted: bool = False
        self.pings: List[bytes] = []
        self._err_string: str = "mock network error"

    def open(self, url: QUrl) -> None:
        self.opened_url = url.toString()

    def close(self) -> None:
        self.closed = True
        self.disconnected.emit()

    def abort(self) -> None:
        self.aborted = True

    def ping(self, payload: bytes) -> None:
        self.pings.append(payload)

    def errorString(self) -> str:
        return self._err_string


class TestLiveCatalogParsingAndDedup(unittest.TestCase):
    def test_emsc_ws_fdsn_text_and_usgs_geojson_parsers(self) -> None:
        ws_msg = json.dumps(
            {
                "action": "create",
                "data": {
                    "id": "20240501_0000123",
                    "geometry": {"type": "Point", "coordinates": [25.2, 35.1, -12.0]},
                    "properties": {
                        "time": "2024-05-01T12:00:00.0Z",
                        "lat": 35.1,
                        "lon": 25.2,
                        "depth": 12.0,
                        "mag": 5.4,
                        "magtype": "mw",
                        "auth": "EMSC",
                        "unid": "20240501_0000123",
                        "flynn_region": "CRETE, GREECE",
                        "lastupdate": "2024-05-01T12:03:00.0Z",
                    },
                },
            }
        )
        ev_ws = parse_emsc_ws_message(ws_msg)
        self.assertIsNotNone(ev_ws)
        self.assertEqual(ev_ws.event_id, "20240501_0000123")
        self.assertAlmostEqual(ev_ws.magnitude, 5.4)
        self.assertEqual(ev_ws.region, "CRETE, GREECE")

        fdsn_txt = (
            "#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType\n"
            "gfz2024aaaa|2024-05-01T12:00:02.0|35.12|25.22|14.0|GFZ|GEOFON|GFZ|gfz2024aaaa|mb|5.6|GFZ|CRETE, GREECE|earthquake\n"
            "malformed|line\n"
        )
        evs_gfz = parse_fdsn_event_text(fdsn_txt, source_feed="GFZ-GEOFON")
        self.assertEqual(len(evs_gfz), 1)
        self.assertEqual(evs_gfz[0].event_id, "gfz2024aaaa")
        self.assertAlmostEqual(evs_gfz[0].depth_km, 14.0)

        t_ms = int(parse_sh_time("2024-05-01T12:00:00") * 1000)
        usgs_json = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": "us7000abcd",
                        "geometry": {"coordinates": [139.77, 35.68, 35.0]},
                        "properties": {
                            "time": t_ms,
                            "updated": t_ms + 30000,
                            "mag": 6.2,
                            "magType": "mww",
                            "place": "Near Tokyo, Japan",
                            "net": "us",
                            "type": "earthquake",
                        },
                    }
                ],
            }
        )
        evs_usgs = parse_usgs_geojson(usgs_json)
        self.assertEqual(len(evs_usgs), 1)
        self.assertEqual(evs_usgs[0].event_id, "us7000abcd")
        self.assertAlmostEqual(evs_usgs[0].magnitude, 6.2)

    def test_deduplicator_handles_duplicates_updates_and_cross_feed(self) -> None:
        t0 = parse_sh_time("2024-05-01T12:00:00")
        ev1 = LiveEvent("unid1", t0, "2024-05-01T12:00:00", 35.1, 25.2, 10.0, 5.2, "mw", "CRETE", "EMSC", "EMSC-WS")
        dedup = EventDeduplicator()

        st1, c1 = dedup.ingest(ev1)
        self.assertEqual(st1, "new")
        self.assertEqual(c1.event_id, "unid1")

        # Exact replay -> duplicate
        st2, _ = dedup.ingest(ev1)
        self.assertEqual(st2, "duplicate")

        # Cross-feed event from GFZ with nearly identical origin & coords -> merged update (mag +0.4)
        ev_gfz = LiveEvent("gfz999", t0 + 3.0, "2024-05-01T12:00:03", 35.15, 25.21, 12.0, 5.6, "mb", "CRETE", "GFZ", "GFZ-GEOFON")
        st3, c3 = dedup.ingest(ev_gfz)
        self.assertEqual(st3, "updated")
        self.assertEqual(c3.event_id, "unid1")
        self.assertAlmostEqual(c3.magnitude, 5.6)


class TestFDSNEventQueryUrl(unittest.TestCase):
    """Unit tests for build_fdsn_event_query_url query string formatting and validation."""

    def test_default_parameters(self) -> None:
        base = "https://geofon.gfz.de/fdsnws/event/1/query"
        t0 = 1714564800.0  # 2024-05-01T12:00:00
        url = build_fdsn_event_query_url(base, start_epoch=t0)
        parsed = urllib.parse.urlparse(url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "geofon.gfz.de")
        self.assertEqual(parsed.path, "/fdsnws/event/1/query")

        qs = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(qs.get("starttime"), [format_fdsn_time(t0)])
        self.assertEqual(qs.get("format"), ["text"])
        self.assertEqual(qs.get("orderby"), ["time"])
        self.assertEqual(qs.get("limit"), ["100"])
        self.assertEqual(qs.get("minmagnitude"), ["1.50"])
        self.assertNotIn("endtime", qs)

    def test_endtime_logic(self) -> None:
        base = "https://geofon.gfz.de/fdsnws/event/1/query"
        t0 = 1714564800.0
        t1 = 1714568400.0

        # end_epoch > start_epoch -> endtime is included
        url_gt = build_fdsn_event_query_url(base, start_epoch=t0, end_epoch=t1)
        qs_gt = urllib.parse.parse_qs(urllib.parse.urlparse(url_gt).query)
        self.assertEqual(qs_gt.get("endtime"), [format_fdsn_time(t1)])

        # end_epoch == start_epoch -> endtime omitted
        url_eq = build_fdsn_event_query_url(base, start_epoch=t0, end_epoch=t0)
        qs_eq = urllib.parse.parse_qs(urllib.parse.urlparse(url_eq).query)
        self.assertNotIn("endtime", qs_eq)

        # end_epoch < start_epoch -> endtime omitted
        url_lt = build_fdsn_event_query_url(base, start_epoch=t0, end_epoch=t0 - 3600.0)
        qs_lt = urllib.parse.parse_qs(urllib.parse.urlparse(url_lt).query)
        self.assertNotIn("endtime", qs_lt)

        # end_epoch is None -> endtime omitted
        url_none = build_fdsn_event_query_url(base, start_epoch=t0, end_epoch=None)
        qs_none = urllib.parse.parse_qs(urllib.parse.urlparse(url_none).query)
        self.assertNotIn("endtime", qs_none)

    def test_min_magnitude_formatting_and_omission(self) -> None:
        base = "https://geofon.gfz.de/fdsnws/event/1/query"
        t0 = 1714564800.0

        # min_mag is None -> minmagnitude omitted
        url_none = build_fdsn_event_query_url(base, start_epoch=t0, min_mag=None)
        qs_none = urllib.parse.parse_qs(urllib.parse.urlparse(url_none).query)
        self.assertNotIn("minmagnitude", qs_none)

        # min_mag = 0.0 -> formatted as "0.00"
        url_zero = build_fdsn_event_query_url(base, start_epoch=t0, min_mag=0.0)
        qs_zero = urllib.parse.parse_qs(urllib.parse.urlparse(url_zero).query)
        self.assertEqual(qs_zero.get("minmagnitude"), ["0.00"])

        # min_mag = 2.4567 -> 2-decimal rounded "2.46"
        url_rnd = build_fdsn_event_query_url(base, start_epoch=t0, min_mag=2.4567)
        qs_rnd = urllib.parse.parse_qs(urllib.parse.urlparse(url_rnd).query)
        self.assertEqual(qs_rnd.get("minmagnitude"), ["2.46"])

        # negative min_mag = -0.5
        url_neg = build_fdsn_event_query_url(base, start_epoch=t0, min_mag=-0.5)
        qs_neg = urllib.parse.parse_qs(urllib.parse.urlparse(url_neg).query)
        self.assertEqual(qs_neg.get("minmagnitude"), ["-0.50"])

    def test_limit_clamping(self) -> None:
        base = "https://geofon.gfz.de/fdsnws/event/1/query"
        t0 = 1714564800.0

        # limit <= 0 clamped to 1
        url_zero = build_fdsn_event_query_url(base, start_epoch=t0, limit=0)
        qs_zero = urllib.parse.parse_qs(urllib.parse.urlparse(url_zero).query)
        self.assertEqual(qs_zero.get("limit"), ["1"])

        url_neg = build_fdsn_event_query_url(base, start_epoch=t0, limit=-20)
        qs_neg = urllib.parse.parse_qs(urllib.parse.urlparse(url_neg).query)
        self.assertEqual(qs_neg.get("limit"), ["1"])

        # normal limits
        url_pos = build_fdsn_event_query_url(base, start_epoch=t0, limit=250)
        qs_pos = urllib.parse.parse_qs(urllib.parse.urlparse(url_pos).query)
        self.assertEqual(qs_pos.get("limit"), ["250"])

    def test_format_parameter(self) -> None:
        base = "https://geofon.gfz.de/fdsnws/event/1/query"
        t0 = 1714564800.0

        url_geo = build_fdsn_event_query_url(base, start_epoch=t0, fmt="geojson")
        qs_geo = urllib.parse.parse_qs(urllib.parse.urlparse(url_geo).query)
        self.assertEqual(qs_geo.get("format"), ["geojson"])

        url_txt = build_fdsn_event_query_url(base, start_epoch=t0, fmt="text")
        qs_txt = urllib.parse.parse_qs(urllib.parse.urlparse(url_txt).query)
        self.assertEqual(qs_txt.get("format"), ["text"])

    def test_base_url_with_existing_query_string_and_delimiters(self) -> None:
        t0 = 1714564800.0

        # Base URL without query -> joined with '?'
        url_clean = build_fdsn_event_query_url("https://service.org/event/query", start_epoch=t0)
        self.assertIn("?", url_clean)
        self.assertEqual(url_clean.count("?"), 1)

        # Base URL with existing parameter (?token=...) -> joined with '&'
        url_param = build_fdsn_event_query_url("https://service.org/event/query?token=secret123", start_epoch=t0)
        self.assertEqual(url_param.count("?"), 1)
        qs_param = urllib.parse.parse_qs(urllib.parse.urlparse(url_param).query)
        self.assertEqual(qs_param.get("token"), ["secret123"])
        self.assertEqual(qs_param.get("format"), ["text"])
        self.assertIn("https://service.org/event/query?token=secret123&", url_param)

        # Base URL with existing parameter and trailing '&' -> trailing '&' stripped and joined with '&'
        url_trailing_amp = build_fdsn_event_query_url("https://service.org/event/query?token=secret123&", start_epoch=t0)
        self.assertEqual(url_trailing_amp.count("?"), 1)
        self.assertNotIn("&&", url_trailing_amp)
        self.assertIn("https://service.org/event/query?token=secret123&", url_trailing_amp)


class TestFetchEventsFromSourceMocked(unittest.TestCase):
    """Offline unit tests for fetch_events_from_source covering USGS, GFZ, EMSC and fallback logic."""

    def test_usgs_fdsn_success_and_parsing(self) -> None:
        calls: List[Tuple[str, float]] = []
        t_ms = 1714564800000
        sample_geojson = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": "us1000abc",
                        "geometry": {"coordinates": [-122.4, 37.8, 10.5]},
                        "properties": {
                            "time": t_ms,
                            "updated": t_ms + 60000,
                            "mag": 4.5,
                            "magType": "mw",
                            "place": "Northern California",
                            "net": "nc",
                            "type": "earthquake",
                            "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us1000abc",
                        },
                    }
                ],
            }
        ).encode("utf-8")

        def fake_opener(url: str, timeout: float) -> Tuple[int, bytes]:
            calls.append((url, timeout))
            return 200, sample_geojson

        events = fetch_events_from_source(
            source_key="usgs_fdsn",
            lookback_hours=3.0,
            min_mag=2.0,
            limit=50,
            now_epoch=1714568400.0,
            opener=fake_opener,
            timeout=10.0,
        )

        self.assertEqual(len(calls), 1)
        url, timeout = calls[0]
        self.assertEqual(timeout, 10.0)
        self.assertTrue(url.startswith(USGS_FDSN_EVENT_URL))
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(qs.get("format"), ["geojson"])
        self.assertEqual(qs.get("minmagnitude"), ["2.00"])
        self.assertEqual(qs.get("limit"), ["50"])

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.event_id, "us1000abc")
        self.assertAlmostEqual(ev.lat, 37.8)
        self.assertAlmostEqual(ev.lon, -122.4)
        self.assertAlmostEqual(ev.depth_km, 10.5)
        self.assertAlmostEqual(ev.magnitude, 4.5)
        self.assertEqual(ev.agency, "NC")
        self.assertEqual(ev.source_feed, "USGS")

    def test_usgs_fdsn_204_and_empty_returns_empty_list(self) -> None:
        def opener_204(url: str, _t: float) -> Tuple[int, bytes]:
            return 204, b""

        events_204 = fetch_events_from_source(
            source_key="usgs_fdsn",
            opener=opener_204,
            now_epoch=1714568400.0,
        )
        self.assertEqual(events_204, [])

        def opener_empty_200(url: str, _t: float) -> Tuple[int, bytes]:
            return 200, b""

        events_empty = fetch_events_from_source(
            source_key="usgs_fdsn",
            opener=opener_empty_200,
            now_epoch=1714568400.0,
        )
        self.assertEqual(events_empty, [])

    def test_usgs_fdsn_http_error_raises_runtime_error(self) -> None:
        def opener_500(url: str, _t: float) -> Tuple[int, bytes]:
            return 500, b"Internal Server Error"

        with self.assertRaises(RuntimeError) as ctx:
            fetch_events_from_source(
                source_key="usgs_fdsn",
                opener=opener_500,
                now_epoch=1714568400.0,
            )
        self.assertIn("USGS HTTP 500", str(ctx.exception))

        def opener_404(url: str, _t: float) -> Tuple[int, bytes]:
            return 404, b"Not Found"

        with self.assertRaises(RuntimeError) as ctx404:
            fetch_events_from_source(
                source_key="usgs_fdsn",
                opener=opener_404,
                now_epoch=1714568400.0,
            )
        self.assertIn("USGS HTTP 404", str(ctx404.exception))

    def test_gfz_and_emsc_fdsn_text_feed_parsing(self) -> None:
        sample_gfz = (
            "#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType\n"
            "gfz2024test|2024-05-01T12:00:00.0|40.5|-3.7|12.0|GFZ|GEOFON|GFZ|gfz2024test|Mw|5.1|GFZ|SPAIN|earthquake\n"
        ).encode("utf-8")

        sample_emsc = (
            "#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType\n"
            "emsc2024test|2024-05-01T12:05:00.0|38.2|23.5|8.0|EMSC|EMSC|EMSC|emsc2024test|ML|4.2|EMSC|GREECE|earthquake\n"
        ).encode("utf-8")

        # Test gfz_fdsn
        gfz_calls: List[str] = []
        def opener_gfz(url: str, _t: float) -> Tuple[int, bytes]:
            gfz_calls.append(url)
            return 200, sample_gfz

        evs_gfz = fetch_events_from_source(
            source_key="gfz_fdsn",
            opener=opener_gfz,
            now_epoch=1714568400.0,
        )
        self.assertEqual(len(evs_gfz), 1)
        self.assertEqual(evs_gfz[0].event_id, "gfz2024test")
        self.assertEqual(evs_gfz[0].source_feed, "GFZ-GEOFON")
        self.assertTrue(gfz_calls[0].startswith(GFZ_FDSN_EVENT_URL))

        # Test emsc_fdsn
        emsc_calls: List[str] = []
        def opener_emsc(url: str, _t: float) -> Tuple[int, bytes]:
            emsc_calls.append(url)
            return 200, sample_emsc

        evs_emsc = fetch_events_from_source(
            source_key="emsc_fdsn",
            opener=opener_emsc,
            now_epoch=1714568400.0,
        )
        self.assertEqual(len(evs_emsc), 1)
        self.assertEqual(evs_emsc[0].event_id, "emsc2024test")
        self.assertEqual(evs_emsc[0].source_feed, "EMSC-FDSN")
        self.assertTrue(emsc_calls[0].startswith(EMSC_FDSN_EVENT_URL))

    def test_emsc_ws_catch_up_fallback_chain(self) -> None:
        sample_gfz = (
            "#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType\n"
            "gfz_fallback|2024-05-01T12:00:00.0|45.0|10.0|10.0|GFZ|GEOFON|GFZ|gfz_fallback|mb|4.9|GFZ|NORTHERN ITALY|earthquake\n"
        ).encode("utf-8")

        # Scenario 1: EMSC fails (HTTP 502) -> GFZ succeeds (HTTP 200)
        called_urls: List[str] = []
        def opener_emsc_fails_gfz_ok(url: str, _t: float) -> Tuple[int, bytes]:
            called_urls.append(url)
            if "seismicportal.eu" in url:
                return 502, b"Bad Gateway"
            if "geofon.gfz.de" in url:
                return 200, sample_gfz
            return 404, b""

        events = fetch_events_from_source(
            source_key="emsc_ws",
            opener=opener_emsc_fails_gfz_ok,
            now_epoch=1714568400.0,
        )
        self.assertEqual(len(called_urls), 2)
        self.assertIn("seismicportal.eu", called_urls[0])
        self.assertIn("geofon.gfz.de", called_urls[1])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "gfz_fallback")
        self.assertEqual(events[0].source_feed, "GFZ-GEOFON")

        # Scenario 2: EMSC raises network exception -> GFZ succeeds
        called_urls.clear()
        def opener_emsc_exc_gfz_ok(url: str, _t: float) -> Tuple[int, bytes]:
            called_urls.append(url)
            if "seismicportal.eu" in url:
                raise ConnectionResetError("Connection reset by peer")
            if "geofon.gfz.de" in url:
                return 200, sample_gfz
            return 404, b""

        events_exc = fetch_events_from_source(
            source_key="emsc_ws",
            opener=opener_emsc_exc_gfz_ok,
            now_epoch=1714568400.0,
        )
        self.assertEqual(len(called_urls), 2)
        self.assertEqual(len(events_exc), 1)
        self.assertEqual(events_exc[0].event_id, "gfz_fallback")

        # Scenario 3: Both EMSC and GFZ fail -> raises RuntimeError("Catalog query failed...")
        called_urls.clear()
        def opener_both_fail(url: str, _t: float) -> Tuple[int, bytes]:
            called_urls.append(url)
            if "seismicportal.eu" in url:
                return 500, b"EMSC Server Error"
            if "geofon.gfz.de" in url:
                return 503, b"GFZ Service Unavailable"
            return 404, b""

        with self.assertRaises(RuntimeError) as ctx:
            fetch_events_from_source(
                source_key="emsc_ws",
                opener=opener_both_fail,
                now_epoch=1714568400.0,
            )
        self.assertIn("Catalog query failed", str(ctx.exception))
        self.assertIn("GFZ-GEOFON HTTP 503", str(ctx.exception))

    def test_lookback_hours_clamping_and_time_window(self) -> None:
        calls: List[str] = []
        def opener_record(url: str, _t: float) -> Tuple[int, bytes]:
            calls.append(url)
            return 204, b""

        t_now = 1714568400.0  # 2024-05-01T13:00:00

        # Sub-minimum lookback (0.01 hours -> clamped to 0.05 hours = 180s)
        fetch_events_from_source(
            source_key="usgs_fdsn",
            lookback_hours=0.01,
            now_epoch=t_now,
            opener=opener_record,
        )
        qs1 = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
        self.assertEqual(qs1.get("endtime"), [format_fdsn_time(t_now)])
        expected_t_start1 = t_now - 0.05 * 3600.0
        self.assertEqual(qs1.get("starttime"), [format_fdsn_time(expected_t_start1)])

        # Normal lookback (6.0 hours = 21600s)
        calls.clear()
        fetch_events_from_source(
            source_key="usgs_fdsn",
            lookback_hours=6.0,
            now_epoch=t_now,
            opener=opener_record,
        )
        qs2 = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
        self.assertEqual(qs2.get("endtime"), [format_fdsn_time(t_now)])
        expected_t_start2 = t_now - 6.0 * 3600.0
        self.assertEqual(qs2.get("starttime"), [format_fdsn_time(expected_t_start2)])


class TestEMSCEventStreamLifecycle(unittest.TestCase):
    """Lifecycle and state machine tests for EMSCEventStream using injected _MockWebSocket."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_state_changes_connecting_and_connected(self) -> None:
        ws_instances: List[_MockWebSocket] = []
        def mock_factory(parent) -> _MockWebSocket:
            sock = _MockWebSocket(parent)
            ws_instances.append(sock)
            return sock

        stream = EMSCEventStream(ws_factory=mock_factory)
        try:
            self.assertEqual(stream.state, "disconnected")
            self.assertFalse(stream.is_running)

            state_changes: List[Tuple[str, str]] = []
            stream.connectionStateChanged.connect(lambda s, d: state_changes.append((s, d)))

            stream.start()
            self.assertTrue(stream.is_running)
            self.assertEqual(stream.state, "connecting")
            self.assertEqual(len(ws_instances), 1)
            ws = ws_instances[0]
            self.assertEqual(ws.opened_url, EMSC_WS_URL)
            self.assertEqual(state_changes[-1][0], "connecting")

            # Simulate WebSocket connection established
            ws.connected.emit()
            self.assertEqual(stream.state, "connected")
            self.assertEqual(state_changes[-1][0], "connected")
            self.assertEqual(state_changes[-1][1], "EMSC WebSocket connected")
            self.assertTrue(stream._ping_timer.isActive())
            self.assertEqual(stream._backoff_s, 5)
        finally:
            stream.stop()

    def test_heartbeat_ping_emission(self) -> None:
        ws_instances: List[_MockWebSocket] = []
        stream = EMSCEventStream(ws_factory=lambda p: ws_instances.append(_MockWebSocket(p)) or ws_instances[-1])
        try:
            stream.start()
            ws = ws_instances[0]
            ws.connected.emit()
            self.assertEqual(ws.pings, [])

            # Trigger heartbeat timeout
            stream._ping_timer.timeout.emit()
            self.assertEqual(ws.pings, [b"shm-live"])

            # Trigger another ping
            stream._ping_timer.timeout.emit()
            self.assertEqual(ws.pings, [b"shm-live", b"shm-live"])
        finally:
            stream.stop()

    def test_push_event_parsing_and_emission(self) -> None:
        ws_instances: List[_MockWebSocket] = []
        stream = EMSCEventStream(ws_factory=lambda p: ws_instances.append(_MockWebSocket(p)) or ws_instances[-1])
        try:
            stream.start()
            ws = ws_instances[0]
            ws.connected.emit()

            received: List[LiveEvent] = []
            stream.eventReceived.connect(lambda ev: received.append(ev))

            # Send valid EMSC push message
            valid_payload = json.dumps(
                {
                    "action": "create",
                    "data": {
                        "id": "20240501_9999999",
                        "geometry": {"type": "Point", "coordinates": [12.5, 42.0, 10.0]},
                        "properties": {
                            "time": "2024-05-01T12:00:00.0Z",
                            "lat": 42.0,
                            "lon": 12.5,
                            "depth": 10.0,
                            "mag": 4.1,
                            "magtype": "ML",
                            "auth": "EMSC",
                            "unid": "20240501_9999999",
                            "flynn_region": "CENTRAL ITALY",
                            "lastupdate": "2024-05-01T12:01:00.0Z",
                        },
                    },
                }
            )
            ws.textMessageReceived.emit(valid_payload)
            self.assertEqual(len(received), 1)
            ev = received[0]
            self.assertEqual(ev.event_id, "20240501_9999999")
            self.assertAlmostEqual(ev.magnitude, 4.1)
            self.assertEqual(ev.region, "CENTRAL ITALY")

            # Send malformed message -> ignored, no exception or duplicate event
            ws.textMessageReceived.emit("MALFORMED JSON {{{")
            self.assertEqual(len(received), 1)
        finally:
            stream.stop()

    def test_disconnect_exponential_backoff_and_reset_on_reconnect(self) -> None:
        ws_instances: List[_MockWebSocket] = []
        def ws_factory(p):
            sock = _MockWebSocket(p)
            ws_instances.append(sock)
            return sock

        stream = EMSCEventStream(ws_factory=ws_factory)
        try:
            stream.start()
            self.assertEqual(len(ws_instances), 1)
            ws1 = ws_instances[0]
            ws1.connected.emit()
            self.assertEqual(stream.state, "connected")
            self.assertEqual(stream._backoff_s, 5)

            # 1st disconnect: delay 5s -> backoff grows to 10s
            ws1.disconnected.emit()
            self.assertEqual(stream.state, "reconnecting")
            self.assertFalse(stream._ping_timer.isActive())
            self.assertTrue(stream._reconnect_timer.isActive())
            self.assertEqual(stream._reconnect_timer.interval(), 5000)
            self.assertEqual(stream._backoff_s, 10)

            # Reconnect timer fires -> creates ws2
            stream._reconnect_timer.timeout.emit()
            self.assertEqual(len(ws_instances), 2)
            self.assertEqual(stream.state, "connecting")
            ws2 = ws_instances[1]

            # 2nd disconnect: delay 10s -> backoff grows to 20s
            ws2.disconnected.emit()
            self.assertEqual(stream._reconnect_timer.interval(), 10000)
            self.assertEqual(stream._backoff_s, 20)

            # 3rd cycle: delay 20s -> backoff grows to 40s
            stream._reconnect_timer.timeout.emit()
            self.assertEqual(len(ws_instances), 3)
            ws3 = ws_instances[2]
            ws3.disconnected.emit()
            self.assertEqual(stream._reconnect_timer.interval(), 20000)
            self.assertEqual(stream._backoff_s, 40)

            # 4th cycle: delay 40s -> backoff grows to 60s (capped)
            stream._reconnect_timer.timeout.emit()
            self.assertEqual(len(ws_instances), 4)
            ws4 = ws_instances[3]
            ws4.disconnected.emit()
            self.assertEqual(stream._reconnect_timer.interval(), 40000)
            self.assertEqual(stream._backoff_s, 60)

            # 5th cycle: delay 60s -> backoff stays 60s (cap reached)
            stream._reconnect_timer.timeout.emit()
            self.assertEqual(len(ws_instances), 5)
            ws5 = ws_instances[4]
            ws5.disconnected.emit()
            self.assertEqual(stream._reconnect_timer.interval(), 60000)
            self.assertEqual(stream._backoff_s, 60)

            # 6th cycle: delay stays 60s
            stream._reconnect_timer.timeout.emit()
            self.assertEqual(len(ws_instances), 6)
            ws6 = ws_instances[5]
            ws6.disconnected.emit()
            self.assertEqual(stream._reconnect_timer.interval(), 60000)
            self.assertEqual(stream._backoff_s, 60)

            # Now reconnect succeeds on 7th socket -> backoff must reset to 5s!
            stream._reconnect_timer.timeout.emit()
            self.assertEqual(len(ws_instances), 7)
            ws7 = ws_instances[6]
            ws7.connected.emit()
            self.assertEqual(stream.state, "connected")
            self.assertEqual(stream._backoff_s, 5)

            # Subsequent disconnect starts fresh backoff from 5s again
            ws7.disconnected.emit()
            self.assertEqual(stream._reconnect_timer.interval(), 5000)
            self.assertEqual(stream._backoff_s, 10)
        finally:
            stream.stop()

    def test_error_occurred_triggers_reconnect(self) -> None:
        ws_instances: List[_MockWebSocket] = []
        stream = EMSCEventStream(ws_factory=lambda p: ws_instances.append(_MockWebSocket(p)) or ws_instances[-1])
        try:
            state_events: List[Tuple[str, str]] = []
            stream.connectionStateChanged.connect(lambda s, d: state_events.append((s, d)))
            stream.start()
            ws = ws_instances[0]
            ws.connected.emit()
            self.assertEqual(stream.state, "connected")

            # Error occurred
            ws.errorOccurred.emit(1)
            self.assertEqual(stream.state, "error")
            self.assertEqual(state_events[-1][0], "error")
            self.assertIn("mock network error", state_events[-1][1])
            self.assertFalse(stream._ping_timer.isActive())
            self.assertTrue(stream._reconnect_timer.isActive())
            self.assertEqual(stream._reconnect_timer.interval(), 5000)
            self.assertEqual(stream._backoff_s, 10)
        finally:
            stream.stop()

    def test_clean_stop_shuts_down_timers_and_sockets(self) -> None:
        ws_instances: List[_MockWebSocket] = []
        stream = EMSCEventStream(ws_factory=lambda p: ws_instances.append(_MockWebSocket(p)) or ws_instances[-1])
        stream.start()
        ws = ws_instances[0]
        ws.connected.emit()
        self.assertTrue(stream.is_running)
        self.assertTrue(stream._ping_timer.isActive())

        stream.stop()
        self.assertFalse(stream.is_running)
        self.assertEqual(stream.state, "disconnected")
        self.assertFalse(stream._ping_timer.isActive())
        self.assertFalse(stream._reconnect_timer.isActive())
        self.assertTrue(ws.closed)
        self.assertIsNone(stream._ws)

        # Calling _on_disconnected when not running does not start reconnect timer
        stream._on_disconnected()
        self.assertFalse(stream._reconnect_timer.isActive())


class TestLivePlannerAndStationSelection(unittest.TestCase):
    def test_tiered_distance_magnitude_filter_and_quarry_exclusion(self) -> None:
        cfg = get_config()
        live_cfg = LiveConfig()
        t0 = parse_sh_time("2024-05-01T12:00:00")

        # Local Bavaria M2.3 (near GRFO 49.69, 11.22) -> accepted
        ev_loc = LiveEvent("loc1", t0, "2024-05-01T12:00:00", 50.1, 12.1, 8.0, 2.3, "ML", "BAVARIA", "BGR")
        eva_loc = evaluate_event_relevance(ev_loc, live_cfg, cfg)
        self.assertTrue(eva_loc.relevant)
        self.assertEqual(eva_loc.tier, "local")
        self.assertLess(eva_loc.dist_deg, 3.0)

        # Local quarry blast excluded by default
        ev_qb = LiveEvent("qb1", t0, "2024-05-01T12:00:00", 50.1, 12.1, 0.0, 2.6, "ML", "BAVARIA", "BGR", event_type="quarry blast")
        self.assertFalse(evaluate_event_relevance(ev_qb, live_cfg, cfg).relevant)

        # Teleseismic Tokyo M4.9 ignored (< 5.5), M6.1 accepted
        ev_tele_small = LiveEvent("tele_s", t0, "2024-05-01T12:00:00", 35.68, 139.77, 30.0, 4.9, "mb", "JAPAN")
        eva_ts = evaluate_event_relevance(ev_tele_small, live_cfg, cfg)
        self.assertFalse(eva_ts.relevant)
        self.assertEqual(eva_ts.tier, "tele")
        self.assertGreater(eva_ts.dist_deg, 80.0)

        ev_tele_big = LiveEvent("tele_b", t0, "2024-05-01T12:00:00", 35.68, 139.77, 30.0, 6.1, "mw", "JAPAN")
        eva_tb = evaluate_event_relevance(ev_tele_big, live_cfg, cfg)
        self.assertTrue(eva_tb.relevant)
        self.assertGreater(eva_tb.first_arrival_sec, 650.0)  # P travel time ~750s

    def test_observatory_and_epicenter_azimuthal_station_selection(self) -> None:
        cfg = get_config()
        t0 = parse_sh_time("2024-05-01T12:00:00")
        ev_crete = LiveEvent("crete1", t0, "2024-05-01T12:00:00", 35.1, 25.2, 15.0, 5.4, "mw", "CRETE")

        # Strategy A: Observatory Focus
        lcfg_obs = LiveConfig(station_strategy="observatory", max_stations=6)
        eva = evaluate_event_relevance(ev_crete, lcfg_obs, cfg)
        targets_obs = select_observatory_targets(ev_crete, eva, lcfg_obs, cfg)
        self.assertEqual(len(targets_obs), 6)
        self.assertIn("GRA1", [t.station for t in targets_obs])
        self.assertTrue(all(t.network == "GR" for t in targets_obs))

        # Strategy B: Epicenter-Near with mocked FDSNWS Station + EIDA routing -> azimuthal sector balance
        station_txt = "\n".join(
            [
                "#Network|Station|Latitude|Longitude|Elevation",
                "HL|SIVA|35.02|25.02|100",
                "HL|IDI|35.29|24.89|700",
                "HL|ZKR|35.11|26.21|200",
                "HT|ARG|36.22|28.12|100",
                "GE|SANT|36.37|25.46|150",
                # Several clustered stations due North that should NOT crowd out South/East/West sectors
                "HL|MRSA|36.60|25.20|50",
                "HL|MRSB|36.80|25.22|50",
                "HL|MRSC|37.10|25.21|50",
                "KO|KARP|35.50|27.15|80",
                "MN|TIR|34.30|23.90|50",
            ]
        )
        routing_post = (
            "https://geofon.gfz.de/fdsnws/dataselect/1/query\n"
            "GE SANT * * 2024-05-01T00:00:00 2024-05-02T00:00:00\n"
            "https://eida.gein.noa.gr/fdsnws/dataselect/1/query\n"
            "HL SIVA * * 2024-05-01T00:00:00 2024-05-02T00:00:00\n"
            "HL ZKR * * 2024-05-01T00:00:00 2024-05-02T00:00:00\n"
        )

        def fake_open(url: str, _timeout: float) -> Tuple[int, bytes]:
            if "station/1/query" in url and "geofon.gfz.de" in url:
                return 200, station_txt.encode("utf-8")
            if "eidaws/routing" in url:
                return 200, routing_post.encode("utf-8")
            return 204, b""

        lcfg_epi = LiveConfig(station_strategy="epicenter", max_stations=5, epicenter_max_radius_deg=5.0)
        targets_epi, _note = select_epicenter_targets(ev_crete, lcfg_epi, cfg, opener=fake_open, allow_network=True)
        self.assertEqual(len(targets_epi), 5)
        by_sta = {t.station: t for t in targets_epi}
        # Clustered MRSB/MRSC in the exact same North sector should leave room for ZKR/KARP/TIR
        self.assertIn("ZKR", by_sta)
        self.assertEqual(by_sta["SANT"].base_url, "https://geofon.gfz.de/fdsnws/dataselect/1/query")
        self.assertEqual(by_sta["SIVA"].base_url, "https://eida.gein.noa.gr/fdsnws/dataselect/1/query")

        # Offline fallback with allow_network=False still finds local SHMConfig stations
        targets_offline, off_note = select_epicenter_targets(ev_crete, lcfg_epi, cfg, allow_network=False)
        self.assertTrue(len(targets_offline) > 0)
        self.assertIn("Offline station DB fallback", off_note)

    def test_travel_time_window_schedule_waits_for_live_tele_arrival(self) -> None:
        cfg = get_config()
        t0 = parse_sh_time("2024-05-01T12:00:00")
        ev_tokyo = LiveEvent("tok1", t0, "2024-05-01T12:00:00", 35.68, 139.77, 35.0, 6.4, "mw", "TOKYO, JAPAN")
        lcfg = LiveConfig(station_strategy="observatory", max_stations=4, archive_grace_sec=75.0)
        eva = evaluate_event_relevance(ev_tokyo, lcfg, cfg)
        plan = plan_event_fetch_window(ev_tokyo, eva, lcfg, cfg, allow_network=False, now_epoch=t0 + 15.0)

        self.assertGreater(plan.earliest_arrival_epoch, t0 + 600.0)
        self.assertGreater(plan.ready_at_epoch, plan.end_epoch + 60.0)
        self.assertGreater(plan.duration_sec, 120.0)


class TestLiveServiceStateMachineAndPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _run_events_until(self, cond_fn, timeout_ms: int = 3000) -> bool:
        import time as _time

        deadline = _time.monotonic() + timeout_ms / 1000.0
        while _time.monotonic() < deadline:
            self.app.processEvents()
            if cond_fn():
                return True
            _time.sleep(0.01)
        self.app.processEvents()
        return bool(cond_fn())

    def test_arrival_wait_retry_backoff_and_loading_policies(self) -> None:
        cfg = get_config()
        t0 = parse_sh_time("2024-05-01T12:00:00")
        clock = [t0 + 5.0]
        raw_gra1 = _make_steim1_mseed_bytes("GRA1", "BHZ", t0 - 30.0, n_samples=6000)
        fetch_attempts = 0
        ws_instances = []

        def ws_factory(parent):
            sock = _FakeWebSocket(parent)
            ws_instances.append(sock)
            return sock

        def fake_http(url: str, _timeout: float) -> Tuple[int, bytes]:
            nonlocal fetch_attempts
            if "dataselect" in url:
                sta = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["station"][0]
                if sta == "GRA1":
                    fetch_attempts += 1
                    # First attempt -> 204 nodata; second attempt -> Mini-SEED!
                    return (204, b"") if fetch_attempts == 1 else (200, raw_gra1)
                return 204, b""
            return 204, b""

        with tempfile.TemporaryDirectory() as tmp:
            lcfg = LiveConfig(
                station_strategy="observatory",
                station_preset="CUSTOM",
                custom_stations=("GRA1",),
                channel_band="BH",
                components=("Z",),
                max_window_sec=180.0,
                archive_grace_sec=30.0,
                retry_delays_sec=(90, 180),
                load_policy="notify",
                output_base_dir=tmp,
            )
            idle_flag = [True]
            notified_ids = []
            loaded_calls = []

            coord = LiveEventCoordinator(
                shm_config=cfg,
                live_config=lcfg,
                is_workspace_idle_cb=lambda: idle_flag[0],
                opener=fake_http,
                ws_factory=ws_factory,
                time_fn=lambda: clock[0],
                allow_network_discovery=False,
            )
            try:
                coord.eventReadyToNotify.connect(lambda eid: notified_ids.append(eid))
                coord.loadEventRequested.connect(lambda eid, auto: loaded_calls.append((eid, auto)))
                coord.set_enabled(True, run_catchup=False)
                self.assertEqual(len(ws_instances), 1)
                self.assertEqual(coord.stream_state, "connected")

                # Push local Bavaria event over fake WebSocket
                ws_instances[0].textMessageReceived.emit(
                    json.dumps(
                        {
                            "action": "create",
                            "data": {
                                "id": "bav01",
                                "properties": {
                                    "time": "2024-05-01T12:00:00.0Z",
                                    "lat": 50.1,
                                    "lon": 12.1,
                                    "depth": 8.0,
                                    "mag": 2.8,
                                    "magtype": "ML",
                                    "unid": "bav01",
                                    "flynn_region": "BAVARIA",
                                },
                            },
                        }
                    )
                )
                self.app.processEvents()
                entry = coord.entries["bav01"]
                self.assertEqual(entry.state, STATE_WAITING)
                self.assertGreater(entry.next_action_epoch, clock[0])

                # Advance clock past archive ready time -> first fetch hits 204 -> RETRY_WAIT
                clock[0] = entry.next_action_epoch + 1.0
                coord.process_due_fetches()
                ok_retry = self._run_events_until(lambda: coord.entries["bav01"].state == STATE_RETRY_WAIT)
                self.assertTrue(ok_retry)
                self.assertEqual(entry.retry_count, 1)

                # Advance clock +90s -> second fetch succeeds -> policy='notify' -> READY
                clock[0] = entry.next_action_epoch + 1.0
                coord.process_due_fetches()
                ok_ready = self._run_events_until(lambda: coord.entries["bav01"].state == STATE_READY)
                self.assertTrue(ok_ready)
                self.assertEqual(notified_ids, ["bav01"])
                self.assertEqual(loaded_calls, [])
                self.assertGreater(len(entry.fetched_traces), 0)

                # Now test auto_idle when workspace is BUSY (no load) vs IDLE (auto-loads)
                lcfg_idle = LiveConfig(
                    station_strategy="observatory",
                    station_preset="CUSTOM",
                    custom_stations=("GRA1",),
                    components=("Z",),
                    max_window_sec=180.0,
                    archive_grace_sec=0.0,
                    load_policy="auto_idle",
                    output_base_dir=tmp,
                )
                coord.update_config(lcfg_idle, reevaluate_existing=False)
                idle_flag[0] = False  # Busy picking!
                clock[0] = t0 + 1000.0
                coord.ingest_event(LiveEvent("ev_busy", t0 + 50.0, "2024-05-01T12:00:50", 50.1, 12.1, 8.0, 2.9, "ML", "BAVARIA"))
                self._run_events_until(lambda: coord.entries["ev_busy"].state == STATE_READY)
                self.assertEqual(coord.entries["ev_busy"].state, STATE_READY)
                self.assertEqual(loaded_calls, [])

                idle_flag[0] = True  # Workspace idle -> auto_idle loads automatically!
                coord.ingest_event(LiveEvent("ev_idle", t0 + 200.0, "2024-05-01T12:03:20", 50.1, 12.1, 8.0, 3.0, "ML", "BAVARIA"))
                ok_loaded = self._run_events_until(lambda: coord.entries["ev_idle"].state == STATE_LOADED)
                self.assertTrue(ok_loaded)
                self.assertEqual(loaded_calls, [("ev_idle", True)])
            finally:
                coord.shutdown()


class TestLiveEventMainWindowAndDialogGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._sess_dir = tempfile.TemporaryDirectory()
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "SHM_SESSION_PATH": os.path.join(self._sess_dir.name, "session.json"),
                "SHM_NO_SESSION_PROMPT": "1",
            },
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self.app.processEvents()
        self._env_patch.stop()
        self._sess_dir.cleanup()

    def test_opt_in_default_dialog_widgets_load_params_theo_undo_and_bgr_handoff(self) -> None:
        win = SHMMainWindow(autoload_example=True)
        n_initial = len(win.engine.traces)
        win.resize(1400, 860)
        win.show()
        self.app.processEvents()

        try:
            # Must be strictly OFF on startup!
            self.assertFalse(win.live_coordinator.is_enabled)
            self.assertFalse(win.act_live_toggle.isChecked())
            self.assertEqual(win.act_live_toggle.text(), "Live: OFF")
            self.assertFalse(win.live_notify_btn.isVisible())

            # Busy idle check should detect manual pick vs clean workspace
            self.assertTrue(win._is_workspace_idle_for_live())
            win.engine.traces[0].picks.append(PhasePick(name="P", onset=win.engine.traces[0].start_time_str, source="m"))
            self.assertFalse(win._is_workspace_idle_for_live())

            # Open non-modal dialog once; second call raises single instance
            win.action_open_live_event_dialog()
            self.app.processEvents()
            dlgs = [d for d in win._open_dialogs if isinstance(d, LiveEventDialog)]
            self.assertEqual(len(dlgs), 1)
            dlg = dlgs[0]
            self.assertFalse(dlg.isSizeGripEnabled())
            for btn in dlg.findChildren(QPushButton):
                self.assertFalse(btn.autoDefault())
                self.assertFalse(btn.isDefault())
            self.assertGreaterEqual(dlg.combo_source.minimumWidth(), 220)
            self.assertGreaterEqual(dlg.combo_policy.minimumWidth(), 300)

            win.action_open_live_event_dialog()
            self.assertEqual(len([d for d in win._open_dialogs if isinstance(d, LiveEventDialog)]), 1)

            # Ingest an event with synthetic downloaded Mini-SEED traces and test READY -> Load -> THEO -> Undo
            t0 = parse_sh_time("2024-05-01T12:00:00")
            raw_gra1 = _make_steim1_mseed_bytes("GRA1", "BHZ", t0 - 60.0, n_samples=4800)
            ev = LiveEvent("gui_ev1", t0, "2024-05-01T12:00:00", 35.1, 25.2, 14.0, 5.8, "mw", "CRETE, GREECE", "EMSC")
            entry = win.live_coordinator.ingest_event(ev)
            self.assertIsNotNone(entry)

            from shm_app.io.mseed import read_mseed_bytes

            trs = read_mseed_bytes(raw_gra1, config=win.config)
            entry.fetched_traces = trs
            entry.state = STATE_READY
            entry.status_detail = f"{len(trs)} traces ready"
            win.live_coordinator.entriesUpdated.emit()
            win.live_coordinator.eventReadyToNotify.emit(ev.event_id)
            self.app.processEvents()

            self.assertEqual(dlg.events_table.rowCount(), 1)
            self.assertTrue(win.live_notify_btn.isVisible())
            self.assertIn("5.8", win.live_notify_btn.text())

            # Click non-intrusive Load button -> replaces workspace traces, populates AnalysisParamSet + THEO
            win.live_notify_btn.click()
            self.app.processEvents()
            self.assertFalse(win.live_notify_btn.isVisible())
            self.assertEqual(entry.state, STATE_LOADED)
            self.assertEqual(len(win.engine.traces), len(trs))
            self.assertEqual(win.engine.traces[0].station, "GRA1")
            p = win.engine.current_params
            self.assertGreater(p.distance, 10.0)
            self.assertAlmostEqual(p.depth, 14.0)
            self.assertAlmostEqual(p.mag_mw, 5.8)
            self.assertEqual(p.region_name or entry.evaluation.region_display, p.region_name)
            self.assertTrue(len(win.engine.traces[0].theo_picks) > 0)

            # Undo restores previous workspace cleanly!
            win.action_undo()
            self.app.processEvents()
            self.assertEqual(len(win.engine.traces), n_initial)

            # Handoff to BGR Downloader pre-fills event window & stations
            dlg.events_table.selectRow(0)
            dlg._on_open_in_bgr()
            self.app.processEvents()
            bgr_dlgs = [d for d in win._open_dialogs if isinstance(d, BGRFetchDialog)]
            self.assertEqual(len(bgr_dlgs), 1)
            self.assertTrue(len(bgr_dlgs[0].selected_stations()) > 0)

            # Flat Min Mag & Custom Stations must always remain editable (never locked!)
            dlg.tabs.setCurrentIndex(1)
            self.assertTrue(dlg.spin_flat_mag.isEnabled())
            self.assertTrue(dlg.edit_custom_stations.isEnabled())
            self.assertTrue(dlg.spin_epi_radius.isEnabled())

            dlg.spin_flat_mag.setValue(3.4)
            self.app.processEvents()
            self.assertFalse(dlg.chk_tiered.isChecked())
            cfg_flat = dlg.build_config_from_widgets()
            self.assertFalse(cfg_flat.use_tiered_filter)
            self.assertAlmostEqual(cfg_flat.flat_min_mag, 3.4, places=2)

            dlg.spin_local_mag.setValue(2.2)
            self.app.processEvents()
            self.assertTrue(dlg.chk_tiered.isChecked())

            dlg.edit_custom_stations.setFocus()
            dlg.edit_custom_stations.setText("GRA1, BFO, WET")
            dlg.edit_custom_stations.textEdited.emit("GRA1, BFO, WET")
            self.app.processEvents()
            self.assertEqual(dlg.combo_preset.currentData(), "CUSTOM")
            cfg_cust = dlg.build_config_from_widgets()
            self.assertEqual(cfg_cust.custom_stations, ("GRA1", "BFO", "WET"))
            dlg.tabs.setCurrentIndex(0)

            # Main window horizontal slider + horizontal scrollbar navigation (start from full view)
            win.action_reset_windows()
            self.app.processEvents()
            f0, f1 = win.main_canvas.full_time_span()
            full_dur = f1 - f0
            self.assertFalse(win.h_scroll.isEnabled())
            self.assertEqual(win.h_zoom_slider.value(), 0)
            self.assertEqual(win.h_zoom_lbl.text(), "Full")

            win.h_zoom_slider.setValue(50)  # ~7.07x horizontal zoom
            self.app.processEvents()
            lo_z, hi_z = win.main_canvas.active_time_span()
            self.assertLess(hi_z - lo_z, full_dur * 0.25)
            self.assertTrue(win.h_scroll.isEnabled())
            self.assertGreater(win.h_scroll.maximum(), 0)

            win.h_scroll.setValue(win.h_scroll.maximum())
            self.app.processEvents()
            lo_s, hi_s = win.main_canvas.active_time_span()
            self.assertAlmostEqual(hi_s - lo_s, hi_z - lo_z, places=1)
            self.assertGreater(lo_s, lo_z)

            win.h_zoom_slider.setValue(0)
            self.app.processEvents()
            self.assertAlmostEqual(win.main_canvas.active_time_span()[0], f0, places=2)
            self.assertAlmostEqual(win.main_canvas.active_time_span()[1], f1, places=2)
            self.assertFalse(win.h_scroll.isEnabled())

            # Corner close does NOT break clean teardown
            QApplication.sendEvent(dlg, QCloseEvent())
            self.assertEqual([d for d in win._open_dialogs if isinstance(d, LiveEventDialog)], [])
        finally:
            for d in list(win._open_dialogs):
                d.close()
            win.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
