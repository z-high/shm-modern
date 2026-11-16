"""Network location and event bulletin / epicenter map dialogs for modernized SHM."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from shm_app.core.config_station import SHMConfig, get_config, resolve_reference_station
from shm_app.core.models import AnalysisParamSet, PhasePick
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.map_helpers import draw_country_basemap, great_circle_segments
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault
from shm_app.io.evt import format_ev2view_summary, write_evt_file
from shm_app.seismology.array_loc import locate_network_geiger
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


class NetworkLocationDialog(QDialog):
    """Multi-Station Network / Geiger / LocSAT Location Dialog."""

    adoptLocation = pyqtSignal(object)  # AnalysisParamSet
    diagnosticAvailable = pyqtSignal(object)  # FitDiagnosticResult

    def __init__(
        self,
        picks: List[PhasePick],
        initial_depth: float = 33.0,
        config: Optional[SHMConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multi-Station Network Location (Geiger / LocSAT)")
        self.resize(740, 460)
        self.picks = picks
        self.config = config or get_config()
        self.result_param: Optional[AnalysisParamSet] = None

        layout = QVBoxLayout(self)

        ctrl_h = QHBoxLayout()
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 700.0)
        self.depth_spin.setValue(initial_depth)
        self.depth_spin.setSuffix(" km")
        self.chk_free_depth = QCheckBox("Solve for Free Depth")
        btn_solve = QPushButton("Run Iterative Location")
        btn_solve.clicked.connect(self.run_location)

        ctrl_h.addWidget(QLabel("Starting / Fixed Depth:"))
        ctrl_h.addWidget(self.depth_spin)
        ctrl_h.addWidget(self.chk_free_depth)
        ctrl_h.addWidget(btn_solve)
        ctrl_h.addStretch(1)
        layout.addLayout(ctrl_h)

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color: #3fb950; font-weight: bold;")
        layout.addWidget(self.summary_lbl)

        self.table = QTableWidget(0, 7)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setHorizontalHeaderLabels(
            ["Station", "Comp", "Phase", "Onset Time", "Weight", "Dist (deg)", "Residual (s)"]
        )
        hdr = self.table.horizontalHeader()
        for col in range(6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        bot_h = QHBoxLayout()
        btn_adopt = QPushButton("Adopt Solution to Parameter Box")
        btn_adopt.clicked.connect(self._on_adopt)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_h.addStretch(1)
        bot_h.addWidget(btn_adopt)
        bot_h.addWidget(btn_close)
        layout.addLayout(bot_h)

        register_tutorial_help(self.depth_spin, "Initial / Fixed Focal Depth", "dlg_netloc_depth", "", "Trial focal depth in km")
        register_tutorial_help(self.chk_free_depth, "Free-Depth Inversion", "dlg_netloc_depth", "", "Invert jointly for depth as well as epicenter and origin time")
        disable_button_autodefault(self)
        apply_c_locale(self)

        if self.picks:
            self.run_location()

    def run_location(self) -> None:
        if not self.picks:
            self.summary_lbl.setText("No phase picks available. Pick phases on traces first.")
            return
        try:
            param, station_resids, diag = locate_network_geiger(
                self.picks,
                depth_km=self.depth_spin.value(),
                free_depth=self.chk_free_depth.isChecked(),
                config=self.config,
                return_diagnostics=True,
            )
            self.result_param = param
            self.diagnosticAvailable.emit(diag)
            self.summary_lbl.setText(
                f"Epicenter: {param.lat:+.3f} N, {param.lon:+.3f} E   |   Depth: {param.depth:.1f} km   |   "
                f"Origin: {param.origin}\nRegion: [{param.region_id}] {param.region_name}"
            )
            self.table.setRowCount(len(self.picks))
            for r_i, pk in enumerate(self.picks):
                st = self.config.get_station(pk.station)
                dist, _, _ = sphere_diff(st.lat, st.lon, param.lat, param.lon)
                vals = [
                    pk.station,
                    pk.comp or "-",
                    pk.name,
                    pk.onset,
                    str(pk.weight),
                    f"{dist:.2f}",
                    f"{pk.resid:+.3f}",
                ]
                for c_i, v in enumerate(vals):
                    self.table.setItem(r_i, c_i, QTableWidgetItem(v))
        except Exception as exc:
            self.summary_lbl.setText(f"Location Error: {exc}")

    def _on_adopt(self) -> None:
        if self.result_param is not None:
            self.adoptLocation.emit(self.result_param)
            self.accept()


class EventSummaryDialog(QDialog):
    """Event Bulletin Viewer (ev2view), .evt File Saver, and Epicenter Map Plugin (epimap)."""

    openWorldMapRequested = pyqtSignal()

    def __init__(
        self,
        param: AnalysisParamSet,
        picks: List[PhasePick],
        config: Optional[SHMConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Event Summary Bulletin (ev2view) & Epicenter Map (epimap)")
        self.resize(820, 620)
        self.param = param
        self.picks = picks
        self.config = config or get_config()

        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # Tab 1: ev2view Bulletin Text
        self.text_edit = QTextEdit()
        self.text_edit.setFontFamily("Monospace")
        self.text_edit.setPlainText(format_ev2view_summary(param, picks))
        tabs.addTab(self.text_edit, "Event Summary (ev2view)")

        # Tab 2: Epicenter & Station Map (epimap plugin)
        map_w = QWidget()
        map_l = QVBoxLayout(map_w)
        self.fig = Figure(figsize=(7.5, 4.2), dpi=95, facecolor="#10141d")
        self.canvas = FigureCanvas(self.fig)
        map_l.addWidget(self.canvas)
        tabs.addTab(map_w, "Epicenter && Station Map (epimap)")

        # Tab 3: Travel-Time Curves (ttplot plugin)
        tt_w = QWidget()
        tt_l = QVBoxLayout(tt_w)
        self.fig_tt = Figure(figsize=(7.5, 4.2), dpi=95, facecolor="#10141d")
        self.canvas_tt = FigureCanvas(self.fig_tt)
        tt_l.addWidget(self.canvas_tt)
        tabs.addTab(tt_w, "Travel-Time Curves (ttplot)")

        layout.addWidget(tabs)

        btn_h = QHBoxLayout()
        btn_world_map = QPushButton("Open in World Map...")
        btn_world_map.clicked.connect(self.openWorldMapRequested.emit)
        btn_save_evt = QPushButton("Save .evt File...")
        btn_save_evt.clicked.connect(self._save_evt)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_h.addWidget(btn_world_map)
        btn_h.addStretch(1)
        btn_h.addWidget(btn_save_evt)
        btn_h.addWidget(btn_close)
        layout.addLayout(btn_h)

        register_tutorial_help(btn_world_map, "Open Event in World Map", "world_map", "", "Inspect current event, picks, and global stations on the interactive World Map")
        register_tutorial_help(btn_save_evt, "Save .evt Event File", "dlg_evt_save", "", "Export event parameters and picks to SeismicHandler .evt format")

        disable_button_autodefault(self)
        apply_c_locale(self)
        self._draw_map()
        self._draw_tt_curves()

    def _draw_map(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor="#0b1320")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-85, 85)
        draw_country_basemap(
            ax,
            edge_linewidth=0.6,
            grid_color="#1f2937",
            axis_color="#30363d",
            axis_lw=0.7,
            axis_alpha=1.0,
        )

        # Plot station locations and great-circle ray paths
        seen_sta: Dict[str, Tuple[float, float]] = {}
        for pk in self.picks:
            st = self.config.get_station(pk.station)
            if (st.lat != 0 or st.lon != 0) and pk.station not in seen_sta:
                seen_sta[pk.station] = (st.lat, st.lon)
        if not seen_sta:
            ref_code, ref_lat, ref_lon = resolve_reference_station(
                self.config, ref_station="GRA1", default_code="GRA1"
            )
            seen_sta[ref_code] = (ref_lat, ref_lon)

        has_epi = (self.param.lat != 0.0 or self.param.lon != 0.0)
        st_lons: List[float] = []
        st_lats: List[float] = []
        gc_segs = []
        for s_lat, s_lon in seen_sta.values():
            st_lats.append(s_lat)
            st_lons.append(s_lon)
            if has_epi:
                gc_segs.extend(great_circle_segments(s_lat, s_lon, self.param.lat, self.param.lon, n_pts=48))
        if gc_segs:
            ax.add_collection(
                LineCollection(gc_segs, colors="#58a6ff", linewidths=1.1, linestyles="--", alpha=0.6, zorder=3)
            )

        ax.scatter(st_lons, st_lats, marker="^", s=65, color="#58a6ff", edgecolors="#0b1320", label="Seismic Stations", zorder=4)
        ax.scatter([self.param.lon], [self.param.lat], marker="*", s=200, color="#ff7b72", edgecolors="#ffffff", linewidths=0.5, label="Epicenter", zorder=5)
        ax.set_xlabel("Longitude (deg)", color="#e6edf3")
        ax.set_ylabel("Latitude (deg)", color="#e6edf3")
        ax.set_title(
            f"Epicenter: ({self.param.lat:+.2f}N, {self.param.lon:+.2f}E) - {self.param.region_name or 'Global Map'}",
            color="#e6edf3",
        )
        ax.tick_params(colors="#8b949e")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_tt_curves(self) -> None:
        self.fig_tt.clear()
        ax = self.fig_tt.add_subplot(111, facecolor="#0d1117")
        ttt = get_ttt_manager(self.config.inputs_dir)
        dists = np.linspace(1.0, 150.0, 150)
        colors = ["#58a6ff", "#ff7b72", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff"]
        for idx, ph in enumerate(["P", "pP", "PP", "PcP", "S", "SKS"]):
            ts = [ttt.travel_time(ph, float(d), self.param.depth) for d in dists]
            valid_d = [d for d, t in zip(dists, ts) if t is not None]
            valid_t = [t / 60.0 for t in ts if t is not None]
            if valid_d:
                ax.plot(valid_d, valid_t, lw=1.5, color=colors[idx % len(colors)], label=ph)
        if self.param.distance > 0:
            ax.axvline(self.param.distance, color="#ffd700", ls="--", lw=1.2, label=f"Event Dist ({self.param.distance:.1f} deg)")
        ax.set_xlabel("Epicentral Distance (deg)", color="#e6edf3")
        ax.set_ylabel("Travel Time (minutes)", color="#e6edf3")
        ax.set_title(f"IASPEI Travel-Time Curves (Depth = {self.param.depth:.0f} km)", color="#e6edf3")
        ax.grid(True, color="#21262d", ls="--", lw=0.6)
        ax.tick_params(colors="#8b949e")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        self.fig_tt.tight_layout()
        self.canvas_tt.draw_idle()

    def _save_evt(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Save .evt Event File", "shm_event.evt", "Event Files (*.evt)")
        if fpath:
            write_evt_file(fpath, self.param, self.picks)
