"""Interactive Seismic World Map for stations, active events, ray paths, and Live Event Monitor."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from shm_app.core.config_station import (
    GRF_STATIONS,
    GRSN_STATIONS,
    SHMConfig,
    geress_stations,
    get_config,
    resolve_reference_station,
)
from shm_app.core.models import AnalysisParamSet
from shm_app.core.time_utils import parse_sh_time
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.map_helpers import (
    distance_ring_segments,
    draw_country_basemap,
    get_country_polygons,
    great_circle_segments,
    load_world_countries,
    split_antimeridian_polyline,
)
from shm_app.gui.ui_helpers import apply_c_locale, cleanup_mpl_canvases, disable_button_autodefault
from shm_app.plugins.live_service import (
    STATE_ERROR,
    STATE_FETCHING,
    STATE_IGNORED,
    STATE_LOADED,
    STATE_READY,
    STATE_RETRY_WAIT,
    STATE_WAITING,
    LiveEventCoordinator,
)
from shm_app.seismology.traveltime import sphere_add, sphere_diff

_LIVE_STATE_COLORS: Dict[str, str] = {
    STATE_LOADED: "#00e5ff",
    STATE_READY: "#3fb950",
    STATE_FETCHING: "#58a6ff",
    STATE_RETRY_WAIT: "#f0883e",
    STATE_WAITING: "#d29922",
    STATE_ERROR: "#f85149",
    STATE_IGNORED: "#6e7681",
}


@dataclass(frozen=True)
class MapStationItem:
    code: str
    lat: float
    lon: float
    elevation: float
    name: str
    category: str  # "loaded", "ref", "grf", "grsn", "other"
    channels: Tuple[str, ...]
    pick_count: int = 0


@dataclass(frozen=True)
class MapEventItem:
    event_id: str
    title: str
    source: str
    lat: float
    lon: float
    depth_km: float
    magnitude: float
    mag_type: str
    origin_sh: str
    origin_epoch: float
    region: str
    dist_deg: float
    baz_deg: float
    status: str  # "ACTIVE", "READY", "LOADED", "WAITING", ...
    color: str
    planned_stations: Tuple[str, ...] = ()
    is_live: bool = False
    is_active_param: bool = False


def load_quarry_coordinates(inputs_dir: str) -> List[Tuple[float, float]]:
    """Load quarry coordinates (lat, lon) from german_quarries.dat if available."""
    fpath = os.path.join(inputs_dir, "german_quarries.dat")
    if not os.path.isfile(fpath):
        return []
    pts: List[Tuple[float, float]] = []
    with open(fpath, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return pts


def _norm_lon(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _viewport_for_preset(
    preset: str,
    event_item: Optional[MapEventItem],
    stations: Sequence[MapStationItem],
    ref_station_pos: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if preset == "europe":
        return (-12.0, 32.0), (34.0, 62.0)
    if preset == "germany":
        return (4.5, 16.0), (46.5, 55.5)
    if preset == "grf":
        return (10.3, 12.6), (48.55, 50.15)
    if preset == "focus_event":
        pts: List[Tuple[float, float]] = []
        if event_item is not None:
            pts.append((event_item.lon, event_item.lat))
        loaded_pts = [(s.lon, s.lat) for s in stations if s.category in ("loaded", "ref")]
        if loaded_pts:
            pts.extend(loaded_pts)
        else:
            pts.append((ref_station_pos[1], ref_station_pos[0]))
        if len(pts) >= 2:
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            lon_span = max(lons) - min(lons)
            if lon_span > 220.0:
                return (-180.0, 180.0), (-78.0, 86.0)
            pad_x = max(8.0, lon_span * 0.22)
            pad_y = max(6.0, (max(lats) - min(lats)) * 0.25)
            x0 = max(-180.0, min(lons) - pad_x)
            x1 = min(180.0, max(lons) + pad_x)
            y0 = max(-86.0, min(lats) - pad_y)
            y1 = min(86.0, max(lats) + pad_y)
            return (x0, x1), (y0, y1)
    return (-180.0, 180.0), (-78.0, 86.0)


class WorldMapDialog(QDialog):
    """Non-modal interactive Seismic World Map synchronized with SHM workspace & Live Event Mode."""

    selectStationTracesRequested = pyqtSignal(str)
    openBGRFetchRequested = pyqtSignal(object)  # {"stations": [...], "start_epoch": ..., "duration_sec": ...}
    adoptEventParamsRequested = pyqtSignal(object)  # dict of event parameters for AnalysisParamSet
    loadLiveEventRequested = pyqtSignal(str)  # event_id
    openLiveMonitorRequested = pyqtSignal()

    def __init__(
        self,
        engine: Any,
        config: Optional[SHMConfig] = None,
        live_coordinator: Optional[LiveEventCoordinator] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.config = config or get_config()
        self.live_coordinator = live_coordinator

        self.setWindowTitle("SHM World Map — Stations, Events & Ray Paths")
        self.setSizeGripEnabled(False)
        self.resize(1180, 740)

        self._xlim: Tuple[float, float] = (-180.0, 180.0)
        self._ylim: Tuple[float, float] = (-74.0, 82.0)
        self._drag_start_px: Optional[Tuple[float, float]] = None
        self._drag_start_lim: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._dragging = False
        self._click_moved = False

        self._station_items: List[MapStationItem] = []
        self._event_items: List[MapEventItem] = []
        self._quarry_pts: List[Tuple[float, float]] = load_quarry_coordinates(self.config.inputs_dir)
        self._selected_item: Optional[Dict[str, Any]] = None
        self._hovered_item: Optional[Dict[str, Any]] = None
        self._pickable_targets: List[Dict[str, Any]] = []
        self._country_cache_polys: Optional[List[np.ndarray]] = None
        self._ax: Any = None
        self._hover_ring: Any = None
        self._select_ring: Any = None

        self._build_ui()
        self.refresh_data(preserve_view=False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addLayout(self._build_top_toolbar())
        root.addLayout(self._build_layer_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        map_holder = QWidget()
        map_l = QVBoxLayout(map_holder)
        map_l.setContentsMargins(0, 0, 0, 0)
        map_l.setSpacing(4)

        self.fig = Figure(figsize=(8.6, 5.4), dpi=100, facecolor="#0d1117")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        map_l.addWidget(self.canvas, stretch=1)

        self.lbl_coords = QLabel("Move mouse over map for coordinates, Δ and backazimuth; scroll to zoom, drag to pan, click markers to inspect.")
        self.lbl_coords.setStyleSheet("color: #79c0ff; font-family: monospace; font-size: 11px; padding: 2px 4px;")
        map_l.addWidget(self.lbl_coords)
        splitter.addWidget(map_holder)

        splitter.addWidget(self._build_inspector_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        bot = QHBoxLayout()
        bot.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        bot.addWidget(btn_close)
        root.addLayout(bot)

        disable_button_autodefault(self)
        apply_c_locale(self)

        self._mpl_cids = [
            self.canvas.mpl_connect("scroll_event", self._on_scroll),
            self.canvas.mpl_connect("button_press_event", self._on_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("button_release_event", self._on_release),
        ]

        self._update_inspector_ui()

    def closeEvent(self, event: QCloseEvent) -> None:
        cleanup_mpl_canvases(self)
        super().closeEvent(event)

    def _build_top_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(6)

        preset_specs = [
            ("World", "world", "Reset map to full global view"),
            ("Europe", "europe", "Zoom to European-Mediterranean region"),
            ("Germany / GRSN", "germany", "Zoom to Germany and GRSN stations"),
            ("GRF Array", "grf", "Zoom closely to Graefenberg Array"),
            ("Focus Event", "focus_event", "Frame the active event epicenter and loaded stations"),
        ]
        for label, key, tip in preset_specs:
            b = QPushButton(label)
            b.clicked.connect(lambda _c=False, k=key: self.set_view_preset(k))
            register_tutorial_help(b, f"Map Extent: {label}", "wmap_view_preset", "", tip)
            bar.addWidget(b)

        bar.addSpacing(10)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find station (e.g. GRA1, BFO) or event region...")
        self.search_edit.setMinimumWidth(235)
        self.search_edit.returnPressed.connect(self._on_search_triggered)
        register_tutorial_help(
            self.search_edit,
            "Find Station or Event",
            "wmap_inspector",
            "",
            "Type station code/name or event region and press Enter to center",
        )
        btn_find = QPushButton("Find")
        btn_find.clicked.connect(self._on_search_triggered)
        bar.addWidget(self.search_edit)
        bar.addWidget(btn_find)
        bar.addStretch(1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(lambda: self.refresh_data(preserve_view=True))
        bar.addWidget(btn_refresh)
        return bar

    def _build_layer_bar(self) -> QGridLayout:
        layer_bar = QGridLayout()
        layer_bar.setHorizontalSpacing(12)
        layer_bar.setVerticalSpacing(4)

        self.chk_db_stations = QCheckBox("All DB Stations (~1400)")
        self.chk_db_stations.setChecked(True)
        self.chk_loaded_stations = QCheckBox("Loaded / Ref Stations")
        self.chk_loaded_stations.setChecked(True)
        self.chk_events = QCheckBox("Events (Active && Live)")
        self.chk_events.setChecked(True)
        self.chk_live_all = QCheckBox("Include Below-Threshold Live")
        self.chk_live_all.setChecked(True)
        self.chk_paths = QCheckBox("Great-Circle Paths")
        self.chk_paths.setChecked(True)
        self.chk_rings = QCheckBox("Distance Rings (30°)")
        self.chk_rings.setChecked(True)
        self.chk_quarries = QCheckBox(f"German Quarries ({len(self._quarry_pts)})")
        self.chk_quarries.setChecked(False)

        specs = [
            (self.chk_db_stations, "Show All Database Stations", "wmap_layer_stations", "Show all stations from STATINF/STATLOC"),
            (self.chk_loaded_stations, "Show Loaded & Reference Stations", "wmap_layer_stations", "Highlight loaded traces and reference station"),
            (self.chk_events, "Show Active & Live Events", "wmap_layer_events", "Show Parameter Box event and Live Event Monitor feeds"),
            (self.chk_live_all, "Show All Live Catalog Events", "wmap_layer_events", "Show gray markers for below-threshold/ignored live events"),
            (self.chk_paths, "Great-Circle Ray Paths", "wmap_layer_paths", "Draw geodesic ray paths from epicenter to stations"),
            (self.chk_rings, "Concentric Distance Rings", "wmap_layer_rings", "Draw 30/60/90/120 deg circles around active event or reference station"),
            (self.chk_quarries, "German Quarry Sites", "wmap_layer_quarries", "Show quarry blasting sites from german_quarries.dat"),
        ]
        for idx, (cb, title, hkey, tip) in enumerate(specs):
            cb.toggled.connect(lambda _v: self.redraw_map())
            register_tutorial_help(cb, title, hkey, "", tip)
            layer_bar.addWidget(cb, idx // 4, idx % 4)
        layer_bar.setColumnStretch(4, 1)
        return layer_bar

    def _build_inspector_panel(self) -> QFrame:
        insp_frame = QFrame()
        insp_frame.setMinimumWidth(290)
        insp_frame.setMaximumWidth(370)
        insp_frame.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; } QLabel { border: none; }")
        insp_l = QVBoxLayout(insp_frame)
        insp_l.setContentsMargins(10, 10, 10, 10)
        insp_l.setSpacing(8)

        hdr_insp = QLabel("Map Inspector")
        hdr_insp.setStyleSheet("font-size: 14px; font-weight: bold; color: #58a6ff;")
        insp_l.addWidget(hdr_insp)

        self.lbl_summary_counts = QLabel("")
        self.lbl_summary_counts.setWordWrap(True)
        self.lbl_summary_counts.setStyleSheet("color: #9ba7b4; font-size: 11px;")
        insp_l.addWidget(self.lbl_summary_counts)

        self.lbl_selected_title = QLabel("Nothing Selected")
        self.lbl_selected_title.setWordWrap(True)
        self.lbl_selected_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #f0f6fc; padding-top: 4px;")
        insp_l.addWidget(self.lbl_selected_title)

        self.lbl_selected_details = QLabel("Click any station, event star, or live earthquake circle on the map to inspect details and launch SHM actions.")
        self.lbl_selected_details.setWordWrap(True)
        self.lbl_selected_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_selected_details.setStyleSheet("color: #c9d1d9; font-size: 12px;")
        register_tutorial_help(
            self.lbl_selected_details,
            "Selection Inspector",
            "wmap_inspector",
            "",
            "Details for the clicked station or earthquake",
        )
        insp_l.addWidget(self.lbl_selected_details)

        self.btn_select_traces = QPushButton("Select Station Traces in Workspace")
        self.btn_select_traces.clicked.connect(self._on_select_station_traces)
        register_tutorial_help(
            self.btn_select_traces,
            "Select Station Traces",
            "wmap_select_traces",
            "",
            "Highlight loaded waveforms of this station in the main window",
        )
        insp_l.addWidget(self.btn_select_traces)

        self.btn_adopt_event = QPushButton("Set as Active Event (Parameter Box)")
        self.btn_adopt_event.clicked.connect(self._on_adopt_selected_event)
        register_tutorial_help(
            self.btn_adopt_event,
            "Adopt Event to Parameter Box",
            "wmap_adopt_event",
            "",
            "Populate SHM's active Analysis Parameter Box from selected event",
        )
        insp_l.addWidget(self.btn_adopt_event)

        self.btn_load_live = QPushButton("Load Prefetched Live Waveforms")
        self.btn_load_live.clicked.connect(self._on_load_selected_live)
        register_tutorial_help(
            self.btn_load_live,
            "Load Live Event Waveforms",
            "dlg_live_actions",
            "",
            "Load prefetched Mini-SEED traces of this READY live event into SHM",
        )
        insp_l.addWidget(self.btn_load_live)

        self.btn_open_bgr = QPushButton("Open in BGR / FDSN Downloader...")
        self.btn_open_bgr.clicked.connect(self._on_open_selected_in_bgr)
        register_tutorial_help(
            self.btn_open_bgr,
            "Open Selection in BGR Downloader",
            "wmap_bgr_fetch",
            "",
            "Prefill BGR FDSN Downloader with selected station or event window",
        )
        insp_l.addWidget(self.btn_open_bgr)

        self.btn_zoom_item = QPushButton("Zoom to Selection")
        self.btn_zoom_item.clicked.connect(self._on_zoom_to_selection)
        insp_l.addWidget(self.btn_zoom_item)

        self.btn_open_live = QPushButton("Open Live Monitor...")
        self.btn_open_live.clicked.connect(self.openLiveMonitorRequested.emit)
        register_tutorial_help(
            self.btn_open_live,
            "Open Live Event Monitor",
            "dlg_live_actions",
            "",
            "Open or focus the Live Event Monitor dashboard",
        )
        insp_l.addWidget(self.btn_open_live)

        insp_l.addStretch(1)

        legend = QLabel(
            "<b>Legend</b><br>"
            "<span style='color:#ff9800'>▲</span> Loaded Workspace Station &nbsp; "
            "<span style='color:#ffd54f'>◆</span> Reference Station<br>"
            "<span style='color:#4fc3f7'>▲</span> GRF Array &nbsp; "
            "<span style='color:#66bb6a'>▲</span> GRSN &nbsp; "
            "<span style='color:#7e8fa4'>▲</span> Other DB Stations<br>"
            "<span style='color:#ff5252'>★</span> Active Event (Param Box) &nbsp; "
            "<span style='color:#3fb950'>●</span> Live Ready &nbsp; "
            "<span style='color:#00e5ff'>●</span> Live Loaded<br>"
            "<span style='color:#d29922'>●</span> Live Waiting / Fetching &nbsp; "
            "<span style='color:#ab47bc'>♦</span> German Quarry"
        )
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #8b949e; font-size: 11px; border-top: 1px solid #30363d; padding-top: 8px;")
        insp_l.addWidget(legend)
        return insp_frame

    def _ref_station_code_and_pos(self) -> Tuple[str, float, float]:
        code = ""
        if self.live_coordinator is not None:
            code = (self.live_coordinator.live_config.ref_station or "").strip().upper()
        pset = self._current_param_set()
        fb_lat = pset.ref_lat if pset is not None else None
        fb_lon = pset.ref_lon if pset is not None else None
        return resolve_reference_station(
            self.config,
            ref_station=code or None,
            fallback_lat=fb_lat,
            fallback_lon=fb_lon,
        )

    def _current_param_set(self) -> Optional[AnalysisParamSet]:
        psets = getattr(self.engine, "param_sets", None)
        idx = int(getattr(self.engine, "active_set_index", 0) or 0)
        if psets and 0 <= idx < len(psets):
            return psets[idx]
        return None

    def refresh_data(self, preserve_view: bool = True) -> None:
        """Rebuild station/event snapshots from engine, config, and live coordinator."""
        ref_code, ref_lat, ref_lon = self._ref_station_code_and_pos()
        grf_set = set(GRF_STATIONS) | set(self.config.station_groups.get("GRF", []))
        grsn_set = set(GRSN_STATIONS) | set(self.config.station_groups.get("GRSN", []))
        geress_set = set(geress_stations(self.config)) | set(self.config.station_groups.get("GERESS", []))

        loaded_channels: Dict[str, List[str]] = {}
        loaded_picks: Dict[str, int] = {}
        for tr in getattr(self.engine, "traces", []):
            code = (tr.station or "").strip().upper()
            if not code or code in ("SUM", "BEAM"):
                continue
            ch = f"{tr.chan1}{tr.chan2}{tr.comp}".strip()
            loaded_channels.setdefault(code, [])
            if ch and ch not in loaded_channels[code]:
                loaded_channels[code].append(ch)
            loaded_picks[code] = loaded_picks.get(code, 0) + len(tr.picks)

        stations: List[MapStationItem] = []
        for code, info in self.config.stations.items():
            cu = code.strip().upper()
            if info.lat == 0.0 and info.lon == 0.0:
                continue
            if cu in loaded_channels:
                cat = "loaded"
            elif cu == ref_code:
                cat = "ref"
            elif cu in grf_set or (cu.startswith(("GRA", "GRB", "GRC")) and len(cu) == 4):
                cat = "grf"
            elif cu in grsn_set or cu in geress_set:
                cat = "grsn"
            else:
                cat = "other"
            stations.append(
                MapStationItem(
                    code=cu,
                    lat=float(info.lat),
                    lon=float(info.lon),
                    elevation=float(info.elevation),
                    name=info.name or cu,
                    category=cat,
                    channels=tuple(loaded_channels.get(cu, ())),
                    pick_count=loaded_picks.get(cu, 0),
                )
            )
        self._station_items = stations

        # Build event items: active Parameter Box event + Live Event Monitor entries
        events: List[MapEventItem] = []
        pset = self._current_param_set()
        if pset is not None:
            elat, elon = float(pset.lat), float(pset.lon)
            if elat == 0.0 and elon == 0.0 and pset.distance > 0.0:
                elat, elon = sphere_add(ref_lat, ref_lon, pset.distance, pset.azimuth)
            if abs(elat) > 1e-4 or abs(elon) > 1e-4:
                dist_deg, baz_deg, _az_ev_to_ref = sphere_diff(ref_lat, ref_lon, elat, elon)
                best_mag = max(pset.mag_mw, pset.mag_mb, pset.mag_ms, pset.mag_ml, pset.mag_mbb, 0.0)
                reg = pset.region_name or self.config.fer.lookup(elat, elon)[1]
                origin_ep = 0.0
                if pset.origin and pset.origin.strip():
                    try:
                        origin_ep = float(parse_sh_time(pset.origin.strip()))
                    except Exception:
                        origin_ep = 0.0
                events.append(
                    MapEventItem(
                        event_id="ACTIVE_PARAM_SET",
                        title="Active Event (Parameter Box)",
                        source=pset.source or "Parameter Box",
                        lat=elat,
                        lon=elon,
                        depth_km=float(pset.depth),
                        magnitude=float(best_mag),
                        mag_type="M",
                        origin_sh=pset.origin or "",
                        origin_epoch=origin_ep,
                        region=reg,
                        dist_deg=dist_deg,
                        baz_deg=baz_deg,
                        status="ACTIVE",
                        color="#ff5252",
                        is_live=False,
                        is_active_param=True,
                    )
                )

        if self.live_coordinator is not None:
            for eid, entry in self.live_coordinator.entries.items():
                ev = entry.event
                planned = tuple(t.station for t in entry.plan.targets) if entry.plan is not None else ()
                color = _LIVE_STATE_COLORS.get(entry.state, "#9e9e9e")
                events.append(
                    MapEventItem(
                        event_id=eid,
                        title=f"Live M{ev.magnitude:.1f} {ev.region or eid}",
                        source=ev.source_feed or "Live Feed",
                        lat=float(ev.lat),
                        lon=float(ev.lon),
                        depth_km=float(ev.depth_km),
                        magnitude=float(ev.magnitude),
                        mag_type=ev.mag_type or "M",
                        origin_sh=ev.origin_sh,
                        origin_epoch=float(ev.origin_epoch),
                        region=ev.region or self.config.fer.lookup(ev.lat, ev.lon)[1],
                        dist_deg=float(entry.evaluation.dist_deg),
                        baz_deg=float(entry.evaluation.baz_deg),
                        status=entry.state,
                        color=color,
                        planned_stations=planned,
                        is_live=True,
                        is_active_param=False,
                    )
                )

        self._event_items = events

        n_loaded = len(loaded_channels)
        n_live = sum(1 for e in events if e.is_live)
        has_active = any(e.is_active_param for e in events)
        self.lbl_summary_counts.setText(
            f"DB Stations: {len(stations)}  |  Loaded Stations: {n_loaded}  |  "
            f"Active Event: {'Yes' if has_active else 'None'}  |  Live Events: {n_live}"
        )

        if not preserve_view:
            active_ev = next((e for e in events if e.is_active_param), None)
            self._xlim, self._ylim = _viewport_for_preset("world", active_ev, stations, (ref_lat, ref_lon))

        self.redraw_map()
        self._update_inspector_ui()

    def set_view_preset(self, preset: str) -> None:
        _, ref_lat, ref_lon = self._ref_station_code_and_pos()
        active_ev = next((e for e in self._event_items if e.is_active_param), None)
        if active_ev is None and self._selected_item and self._selected_item.get("kind") == "event":
            active_ev = self._selected_item.get("item")
        self._xlim, self._ylim = _viewport_for_preset(preset, active_ev, self._station_items, (ref_lat, ref_lon))
        self.redraw_map()

    def _get_country_polys(self) -> List[np.ndarray]:
        if self._country_cache_polys is None:
            self._country_cache_polys = get_country_polygons()
        return self._country_cache_polys

    def _grid_step_for_span(self, span_lon: float) -> float:
        if span_lon > 150.0:
            return 30.0
        if span_lon > 45.0:
            return 10.0
        if span_lon > 12.0:
            return 2.0
        if span_lon > 3.0:
            return 0.5
        return 0.25

    def redraw_map(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor="#09111c")
        self._ax = ax
        self._pickable_targets = []

        span_lon = max(0.2, self._xlim[1] - self._xlim[0])
        span_lat = max(0.2, self._ylim[1] - self._ylim[0])
        self._draw_basemap(ax, span_lon)
        self._draw_quarries(ax)
        _, ref_lat, ref_lon = self._ref_station_code_and_pos()
        self._draw_rings_and_paths(ax, ref_lat, ref_lon)
        self._draw_stations(ax, span_lon, span_lat)
        self._draw_events(ax, span_lon)

        (self._select_ring,) = ax.plot(
            [], [], marker="o", ms=14, mfc="none", mec="#ffffff", mew=1.8, linestyle="none", zorder=11, visible=False
        )
        (self._hover_ring,) = ax.plot(
            [], [], marker="o", ms=12, mfc="none", mec="#ffd54f", mew=1.3, linestyle="none", zorder=11, visible=False
        )
        if self._selected_item is not None:
            self._select_ring.set_data([self._selected_item["lon"]], [self._selected_item["lat"]])
            self._select_ring.set_visible(True)

        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)
        ax.set_xlabel("Longitude (°E)", color="#9ba7b4", fontsize=9)
        ax.set_ylabel("Latitude (°N)", color="#9ba7b4", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        self.fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.095)
        self.canvas.draw_idle()

    def _draw_basemap(self, ax: Any, span_lon: float) -> None:
        gstep = self._grid_step_for_span(span_lon)
        xticks = np.arange(math.floor(self._xlim[0] / gstep) * gstep, self._xlim[1] + gstep * 0.5, gstep)
        yticks = np.arange(math.floor(self._ylim[0] / gstep) * gstep, self._ylim[1] + gstep * 0.5, gstep)
        draw_country_basemap(
            ax,
            self._get_country_polys(),
            edge_linewidth=0.65,
            grid_color="#1b2b3f",
            axis_color="#293d57",
            axis_lw=0.8,
            axis_alpha=0.7,
            xticks=xticks,
            yticks=yticks,
        )

    def _draw_quarries(self, ax: Any) -> None:
        if self.chk_quarries.isChecked() and self._quarry_pts:
            qlats = [p[0] for p in self._quarry_pts]
            qlons = [p[1] for p in self._quarry_pts]
            ax.scatter(qlons, qlats, marker="d", s=22, color="#ab47bc", alpha=0.85, zorder=3, label="Quarries")

    def _draw_rings_and_paths(self, ax: Any, ref_lat: float, ref_lon: float) -> None:
        focus_ev: Optional[MapEventItem] = None
        if self._selected_item and self._selected_item.get("kind") == "event":
            focus_ev = self._selected_item["item"]
        if focus_ev is None:
            focus_ev = next((e for e in self._event_items if e.is_active_param), None)
        if focus_ev is None and self._event_items:
            focus_ev = next((e for e in self._event_items if e.status in (STATE_READY, STATE_LOADED)), None)

        if self.chk_rings.isChecked():
            ring_lat = focus_ev.lat if focus_ev is not None else ref_lat
            ring_lon = focus_ev.lon if focus_ev is not None else ref_lon
            ring_segs: List[np.ndarray] = []
            for d_deg in (30.0, 60.0, 90.0, 120.0):
                ring_segs.extend(distance_ring_segments(ring_lat, ring_lon, d_deg))
                lbl_lat, lbl_lon = sphere_add(ring_lat, ring_lon, d_deg, 0.0)
                if self._xlim[0] < lbl_lon < self._xlim[1] and self._ylim[0] < lbl_lat < self._ylim[1]:
                    ax.text(
                        lbl_lon,
                        lbl_lat,
                        f"{int(d_deg)}°",
                        color="#5c7696",
                        fontsize=7.5,
                        ha="center",
                        va="bottom",
                        zorder=2,
                    )
            if ring_segs:
                ax.add_collection(LineCollection(ring_segs, colors="#2e4869", linewidths=0.75, linestyles="--", alpha=0.8, zorder=2))

        if self.chk_paths.isChecked() and focus_ev is not None:
            path_stations = [s for s in self._station_items if s.category in ("loaded", "ref")]
            if focus_ev.planned_stations:
                p_set = set(focus_ev.planned_stations[:16])
                extra = [s for s in self._station_items if s.code in p_set and s.category not in ("loaded", "ref")]
                path_stations.extend(extra)
            if not path_stations:
                path_stations = [MapStationItem("REF", ref_lat, ref_lon, 0.0, "Reference", "ref", ())]
            gc_segs: List[np.ndarray] = []
            for st in path_stations:
                gc_segs.extend(great_circle_segments(focus_ev.lat, focus_ev.lon, st.lat, st.lon))
            if gc_segs:
                line_col = "#38bdf8" if focus_ev.is_active_param else focus_ev.color
                ax.add_collection(LineCollection(gc_segs, colors=line_col, linewidths=1.05, alpha=0.62, zorder=3))

    def _draw_stations(self, ax: Any, span_lon: float, span_lat: float) -> None:
        vis_stations_for_labels: List[MapStationItem] = []
        cat_groups: Dict[str, List[MapStationItem]] = {
            "other": [],
            "grsn": [],
            "grf": [],
            "ref": [],
            "loaded": [],
        }
        for st in self._station_items:
            cat_groups[st.category].append(st)
            if self._xlim[0] <= st.lon <= self._xlim[1] and self._ylim[0] <= st.lat <= self._ylim[1]:
                if st.category in ("loaded", "ref") or self.chk_db_stations.isChecked():
                    vis_stations_for_labels.append(st)

        if self.chk_db_stations.isChecked():
            for cat, color, size, alpha, zo in [
                ("other", "#7e8fa4", 16, 0.68, 4),
                ("grsn", "#66bb6a", 34, 0.90, 5),
                ("grf", "#4fc3f7", 36, 0.92, 5),
            ]:
                items = cat_groups[cat]
                if items:
                    ax.scatter(
                        [s.lon for s in items],
                        [s.lat for s in items],
                        marker="^",
                        s=size,
                        color=color,
                        alpha=alpha,
                        edgecolors="#09111c",
                        linewidths=0.4,
                        zorder=zo,
                    )
                    for s in items:
                        self._pickable_targets.append({"kind": "station", "id": s.code, "lat": s.lat, "lon": s.lon, "item": s, "prio": 1})

        if self.chk_loaded_stations.isChecked():
            if cat_groups["ref"]:
                items = cat_groups["ref"]
                ax.scatter(
                    [s.lon for s in items],
                    [s.lat for s in items],
                    marker="D",
                    s=64,
                    color="#ffd54f",
                    edgecolors="#09111c",
                    linewidths=0.8,
                    zorder=6,
                )
                for s in items:
                    self._pickable_targets.append({"kind": "station", "id": s.code, "lat": s.lat, "lon": s.lon, "item": s, "prio": 2})

            if cat_groups["loaded"]:
                items = cat_groups["loaded"]
                ax.scatter(
                    [s.lon for s in items],
                    [s.lat for s in items],
                    marker="^",
                    s=78,
                    color="#ff9800",
                    edgecolors="#ffffff",
                    linewidths=0.8,
                    zorder=7,
                )
                for s in items:
                    self._pickable_targets.append({"kind": "station", "id": s.code, "lat": s.lat, "lon": s.lon, "item": s, "prio": 3})

        prio_order = {"ref": 0, "loaded": 1, "grf": 2, "grsn": 3, "other": 4}
        if span_lon <= 24.0:
            raw_candidates = sorted(
                vis_stations_for_labels,
                key=lambda st: (prio_order.get(st.category, 9), st.code),
            )
        else:
            raw_candidates = sorted(
                [s for s in vis_stations_for_labels if s.category in ("ref", "loaded")],
                key=lambda st: (prio_order.get(st.category, 9), st.code),
            )
        min_dx = span_lon * 0.055
        min_dy = span_lat * 0.038
        placed_lbls: List[Tuple[float, float]] = []
        for s in raw_candidates:
            if len(placed_lbls) >= 65:
                break
            if any(abs(s.lon - px) < min_dx and abs(s.lat - py) < min_dy for px, py in placed_lbls):
                continue
            placed_lbls.append((s.lon, s.lat))
            lbl_col = "#ffd180" if s.category == "loaded" else ("#ffe57f" if s.category == "ref" else "#b0c4de")
            ax.text(
                s.lon,
                s.lat + span_lat * 0.008,
                f" {s.code}",
                color=lbl_col,
                fontsize=7.5 if s.category in ("loaded", "ref") else 6.8,
                fontweight="bold" if s.category in ("loaded", "ref") else "normal",
                ha="left",
                va="bottom",
                zorder=8,
            )

    def _draw_events(self, ax: Any, span_lon: float) -> None:
        if not self.chk_events.isChecked():
            return
        for ev in self._event_items:
            if ev.is_live:
                if not self.chk_live_all.isChecked() and ev.status not in (
                    STATE_READY,
                    STATE_LOADED,
                    STATE_FETCHING,
                    STATE_WAITING,
                    STATE_RETRY_WAIT,
                ):
                    continue
                msize = max(36.0, min(220.0, (max(1.5, ev.magnitude) - 1.0) * 32.0))
                ax.scatter(
                    [ev.lon],
                    [ev.lat],
                    marker="o",
                    s=msize,
                    color=ev.color,
                    edgecolors="#ffffff" if ev.status in (STATE_READY, STATE_LOADED) else "#09111c",
                    linewidths=1.0,
                    alpha=0.92,
                    zorder=9,
                )
                if span_lon <= 120.0 or ev.status in (STATE_READY, STATE_LOADED):
                    ax.text(
                        ev.lon,
                        ev.lat - span_lon * 0.005,
                        f" M{ev.magnitude:.1f}",
                        color=ev.color,
                        fontsize=7.5,
                        fontweight="bold",
                        va="top",
                        zorder=9,
                    )
                self._pickable_targets.append({"kind": "event", "id": ev.event_id, "lat": ev.lat, "lon": ev.lon, "item": ev, "prio": 4})

        for ev in self._event_items:
            if ev.is_active_param:
                ax.scatter(
                    [ev.lon],
                    [ev.lat],
                    marker="*",
                    s=240,
                    color="#ff5252",
                    edgecolors="#ffffff",
                    linewidths=0.9,
                    zorder=10,
                )
                lbl = f" ★ M{ev.magnitude:.1f}" if ev.magnitude > 0 else " ★ Active Event"
                ax.text(ev.lon, ev.lat, lbl, color="#ff8a80", fontsize=8.5, fontweight="bold", va="bottom", zorder=10)
                self._pickable_targets.append({"kind": "event", "id": ev.event_id, "lat": ev.lat, "lon": ev.lon, "item": ev, "prio": 5})

    def hit_test(self, x_disp: float, y_disp: float, max_px: float = 14.0) -> Optional[Dict[str, Any]]:
        """Return highest-priority station or event within max_px screen distance of display coords."""
        if self._ax is None or not self._pickable_targets:
            return None
        best: Optional[Dict[str, Any]] = None
        best_score: Tuple[int, float] = (-1, float("inf"))
        pts = np.array([[t["lon"], t["lat"]] for t in self._pickable_targets], dtype=np.float64)
        disp = self._ax.transData.transform(pts)
        dx = disp[:, 0] - float(x_disp)
        dy = disp[:, 1] - float(y_disp)
        d_px = np.hypot(dx, dy)
        xlim0, xlim1 = self._ax.get_xlim()
        ylim0, ylim1 = self._ax.get_ylim()
        for idx, dist_px in enumerate(d_px):
            if dist_px > max_px:
                continue
            tgt = self._pickable_targets[idx]
            if not (xlim0 <= tgt["lon"] <= xlim1 and ylim0 <= tgt["lat"] <= ylim1):
                continue
            prio = int(tgt.get("prio", 1))
            if prio > best_score[0] or (prio == best_score[0] and dist_px < best_score[1]):
                best_score = (prio, float(dist_px))
                best = tgt
        return best

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.78 if event.button == "up" else 1.28
        cx, cy = float(event.xdata), float(event.ydata)
        x0, x1 = self._xlim
        y0, y1 = self._ylim
        new_w = max(0.35, min(360.0, (x1 - x0) * scale))
        new_h = max(0.25, min(176.0, (y1 - y0) * scale))
        rx = (cx - x0) / max(1e-9, x1 - x0)
        ry = (cy - y0) / max(1e-9, y1 - y0)
        nx0 = cx - rx * new_w
        nx1 = nx0 + new_w
        ny0 = cy - ry * new_h
        ny1 = ny0 + new_h
        if nx1 - nx0 >= 360.0:
            nx0, nx1 = -180.0, 180.0
        else:
            if nx0 < -180.0:
                nx1 += -180.0 - nx0
                nx0 = -180.0
            if nx1 > 180.0:
                nx0 -= nx1 - 180.0
                nx1 = 180.0
        ny0 = max(-88.0, min(88.0 - new_h, ny0))
        ny1 = min(88.0, ny0 + new_h)
        self._xlim = (nx0, nx1)
        self._ylim = (ny0, ny1)
        self.redraw_map()

    def _on_press(self, event: Any) -> None:
        if event.inaxes is not self._ax or event.button != 1:
            return
        if event.dblclick and event.xdata is not None and event.ydata is not None:
            # Zoom in centered on double click
            cx, cy = float(event.xdata), float(event.ydata)
            w = max(0.5, (self._xlim[1] - self._xlim[0]) * 0.45)
            h = max(0.4, (self._ylim[1] - self._ylim[0]) * 0.45)
            self._xlim = (max(-180.0, cx - w / 2.0), min(180.0, cx + w / 2.0))
            self._ylim = (max(-88.0, cy - h / 2.0), min(88.0, cy + h / 2.0))
            self.redraw_map()
            return
        self._drag_start_px = (float(event.x), float(event.y))
        self._drag_start_lim = (self._xlim, self._ylim)
        self._dragging = True
        self._click_moved = False

    def _on_motion(self, event: Any) -> None:
        if self._ax is None or event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        if self._dragging and self._drag_start_px is not None and self._drag_start_lim is not None:
            dx_px = float(event.x) - self._drag_start_px[0]
            dy_px = float(event.y) - self._drag_start_px[1]
            if math.hypot(dx_px, dy_px) > 4.0:
                self._click_moved = True
                bbox = self._ax.bbox
                if bbox.width > 1 and bbox.height > 1:
                    (sx0, sx1), (sy0, sy1) = self._drag_start_lim
                    w, h = sx1 - sx0, sy1 - sy0
                    dlon = -dx_px * w / bbox.width
                    dlat = -dy_px * h / bbox.height
                    nx0 = max(-180.0, min(180.0 - w, sx0 + dlon))
                    ny0 = max(-88.0, min(88.0 - h, sy0 + dlat))
                    self._xlim = (nx0, nx0 + w)
                    self._ylim = (ny0, ny0 + h)
                    self._ax.set_xlim(self._xlim)
                    self._ax.set_ylim(self._ylim)
                    self.canvas.draw_idle()
                    return

        lon = _norm_lon(event.xdata)
        lat = max(-90.0, min(90.0, float(event.ydata)))
        ref_code, ref_lat, ref_lon = self._ref_station_code_and_pos()
        d_ref, baz, _az_pt_to_ref = sphere_diff(ref_lat, ref_lon, lat, lon)
        hit = self.hit_test(float(event.x), float(event.y))
        hover_txt = ""
        if hit is not None:
            if hit["kind"] == "station":
                st: MapStationItem = hit["item"]
                ch_info = f"  [{','.join(st.channels)}]" if st.channels else ""
                hover_txt = f"  |  Station {st.code} ({st.category.upper()}){ch_info}"
            else:
                ev: MapEventItem = hit["item"]
                hover_txt = f"  |  {ev.title} [{ev.status}] Δ={ev.dist_deg:.1f}°"
            if self._hover_ring is not None:
                self._hover_ring.set_data([hit["lon"]], [hit["lat"]])
                self._hover_ring.set_visible(True)
                self.canvas.draw_idle()
        elif self._hover_ring is not None and self._hover_ring.get_visible():
            self._hover_ring.set_visible(False)
            self.canvas.draw_idle()

        self.lbl_coords.setText(
            f"Lat: {lat:+.3f}°   Lon: {lon:+.3f}°   |   From {ref_code}: Δ={d_ref:.2f}°  BAZ={baz:.1f}°{hover_txt}"
        )

    def _on_release(self, event: Any) -> None:
        was_pan = self._dragging and self._click_moved
        self._dragging = False
        self._drag_start_px = None
        self._drag_start_lim = None
        if was_pan:
            # Final crisp redraw so labels & graticule update cleanly for the new viewport
            self.redraw_map()
            return
        if event.inaxes is not self._ax or event.button != 1:
            return
        hit = self.hit_test(float(event.x), float(event.y))
        self.select_item(hit)

    def select_item(self, hit: Optional[Dict[str, Any]]) -> None:
        """Programmatically select a station or event hit target and update Inspector & paths."""
        self._selected_item = hit
        if hit is None:
            if self._select_ring is not None:
                self._select_ring.set_visible(False)
                self.canvas.draw_idle()
            self._update_inspector_ui()
            return
        self.redraw_map()
        self._update_inspector_ui()

    def _update_inspector_ui(self) -> None:
        self.btn_select_traces.setEnabled(False)
        self.btn_adopt_event.setEnabled(False)
        self.btn_load_live.setEnabled(False)
        self.btn_open_bgr.setEnabled(False)
        self.btn_zoom_item.setEnabled(False)

        if self._selected_item is None:
            self.lbl_selected_title.setText("Nothing Selected")
            self.lbl_selected_details.setText(
                "Click any station, event star, or live earthquake marker on the map to inspect details and launch SHM actions."
            )
            return

        kind = self._selected_item["kind"]
        if kind == "station":
            st: MapStationItem = self._selected_item["item"]
            self.lbl_selected_title.setText(f"Station {st.code} — {st.name}")
            lines = [
                f"<b>Category:</b> {st.category.upper()}",
                f"<b>Coordinates:</b> {st.lat:+.4f}° N, {st.lon:+.4f}° E",
                f"<b>Elevation:</b> {st.elevation:.1f} m",
                f"<b>Loaded Channels:</b> {', '.join(st.channels) if st.channels else 'None in workspace'}",
            ]
            if st.pick_count > 0:
                lines.append(f"<b>Workspace Picks:</b> {st.pick_count}")
            active_ev = next((e for e in self._event_items if e.is_active_param), None)
            if active_ev is not None:
                d_ev, baz_se, _az_ev_to_st = sphere_diff(st.lat, st.lon, active_ev.lat, active_ev.lon)
                lines.append(f"<b>From Active Event:</b> Δ = {d_ev:.2f}°, BAZ = {baz_se:.1f}°")
            self.lbl_selected_details.setText("<br>".join(lines))
            self.btn_select_traces.setEnabled(bool(st.channels))
            self.btn_open_bgr.setEnabled(True)
            self.btn_zoom_item.setEnabled(True)
        else:
            ev: MapEventItem = self._selected_item["item"]
            self.lbl_selected_title.setText(ev.title)
            lines = [
                f"<b>Source:</b> {ev.source}  ({ev.status})",
                f"<b>Origin UTC:</b> {ev.origin_sh or 'Not set'}",
                f"<b>Epicenter:</b> {ev.lat:+.3f}° N, {ev.lon:+.3f}° E  |  Depth: {ev.depth_km:.1f} km",
                f"<b>Magnitude:</b> {ev.mag_type} {ev.magnitude:.1f}" if ev.magnitude > 0 else "<b>Magnitude:</b> —",
                f"<b>F-E Region:</b> {ev.region or 'Unknown'}",
                f"<b>From Ref Station:</b> Δ = {ev.dist_deg:.2f}°, BAZ = {ev.baz_deg:.1f}°",
            ]
            if ev.planned_stations:
                lines.append(f"<b>Planned Stations:</b> {', '.join(ev.planned_stations[:12])}")
            self.lbl_selected_details.setText("<br>".join(lines))
            self.btn_adopt_event.setEnabled(True)
            self.btn_load_live.setEnabled(ev.is_live and ev.status == STATE_READY)
            self.btn_open_bgr.setEnabled(True)
            self.btn_zoom_item.setEnabled(True)

    def focus_event_by_id(self, event_id: str) -> bool:
        """Select and zoom onto a specific live or active event by event_id."""
        target = (event_id or "").strip()
        if not target:
            return False
        ev_match = next((e for e in self._event_items if e.event_id == target), None)
        if ev_match is None:
            return False
        self._xlim = (max(-180.0, ev_match.lon - 18.0), min(180.0, ev_match.lon + 18.0))
        self._ylim = (max(-88.0, ev_match.lat - 12.0), min(88.0, ev_match.lat + 12.0))
        self.select_item(
            {"kind": "event", "id": ev_match.event_id, "lat": ev_match.lat, "lon": ev_match.lon, "item": ev_match, "prio": 4}
        )
        return True

    def focus_search(self, query: str) -> bool:
        """Search stations or events by code/region and center on the best match."""
        q = (query or "").strip()
        if not q:
            return False
        qu = q.upper()
        # Exact station code match first, then substring match in station or event
        st_match = next((s for s in self._station_items if s.code.upper() == qu), None)
        if st_match is None:
            st_match = next(
                (s for s in self._station_items if qu in s.code.upper() or qu in s.name.upper()),
                None,
            )
        if st_match is not None:
            self._xlim = (max(-180.0, st_match.lon - 3.5), min(180.0, st_match.lon + 3.5))
            self._ylim = (max(-88.0, st_match.lat - 2.5), min(88.0, st_match.lat + 2.5))
            self.select_item({"kind": "station", "id": st_match.code, "lat": st_match.lat, "lon": st_match.lon, "item": st_match, "prio": 2})
            return True

        ev_match = next(
            (e for e in self._event_items if qu in e.event_id.upper() or qu in e.region.upper() or qu in e.title.upper()),
            None,
        )
        if ev_match is not None:
            self._xlim = (max(-180.0, ev_match.lon - 18.0), min(180.0, ev_match.lon + 18.0))
            self._ylim = (max(-88.0, ev_match.lat - 12.0), min(88.0, ev_match.lat + 12.0))
            self.select_item({"kind": "event", "id": ev_match.event_id, "lat": ev_match.lat, "lon": ev_match.lon, "item": ev_match, "prio": 4})
            return True
        return False

    def _on_search_triggered(self) -> None:
        if not self.focus_search(self.search_edit.text()):
            self.lbl_coords.setText(f"No station or event matching '{self.search_edit.text().strip()}'.")

    def _on_select_station_traces(self) -> None:
        if self._selected_item and self._selected_item.get("kind") == "station":
            st: MapStationItem = self._selected_item["item"]
            self.selectStationTracesRequested.emit(st.code)

    def _on_adopt_selected_event(self) -> None:
        if self._selected_item and self._selected_item.get("kind") == "event":
            ev: MapEventItem = self._selected_item["item"]
            self.adoptEventParamsRequested.emit(
                {
                    "lat": ev.lat,
                    "lon": ev.lon,
                    "depth": ev.depth_km,
                    "origin": ev.origin_sh,
                    "distance": ev.dist_deg,
                    "azimuth": ev.baz_deg,
                    "region_name": ev.region,
                    "magnitude": ev.magnitude,
                    "source": ev.source,
                }
            )

    def _on_load_selected_live(self) -> None:
        if self._selected_item and self._selected_item.get("kind") == "event":
            ev: MapEventItem = self._selected_item["item"]
            if ev.is_live and ev.status == STATE_READY:
                self.loadLiveEventRequested.emit(ev.event_id)

    def _on_open_selected_in_bgr(self) -> None:
        if self._selected_item is None:
            return
        if self._selected_item["kind"] == "station":
            st: MapStationItem = self._selected_item["item"]
            self.openBGRFetchRequested.emit({"stations": [st.code], "start_epoch": None, "duration_sec": None})
        else:
            ev: MapEventItem = self._selected_item["item"]
            stations = list(ev.planned_stations) if ev.planned_stations else [
                s.code for s in self._station_items if s.category == "loaded"
            ]
            self.openBGRFetchRequested.emit(
                {
                    "stations": stations,
                    "start_epoch": (ev.origin_epoch - 30.0) if ev.origin_epoch > 0 else None,
                    "duration_sec": 600.0 if ev.origin_epoch > 0 else None,
                }
            )

    def _on_zoom_to_selection(self) -> None:
        if self._selected_item is None:
            return
        lon = float(self._selected_item["lon"])
        lat = float(self._selected_item["lat"])
        if self._selected_item["kind"] == "station":
            dx, dy = 2.5, 1.8
        else:
            dx, dy = 18.0, 12.0
        self._xlim = (max(-180.0, lon - dx), min(180.0, lon + dx))
        self._ylim = (max(-88.0, lat - dy), min(88.0, lat + dy))
        self.redraw_map()
