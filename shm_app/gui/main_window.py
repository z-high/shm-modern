"""Main Application Window for modernized SeismicHandler Motif (SHM)."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid
import weakref

import functools
import numpy as np

from PyQt6 import sip
from PyQt6.QtCore import QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSlider,
    QSplitter,
    QStatusBar,
    QStyleFactory,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import get_config
from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.core.fit_models import FitModificationRequest, FitType
from shm_app.core.models import AnalysisParamSet, Trace
from shm_app.core.session import SessionManager, resolve_session_path
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.gui.fit_dialog import FitDiagnosticWindow
from shm_app.seismology.traveltime import sphere_diff
from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    FilterDialog,
    ParticleMotionDialog,
    SpectrumDialog,
    VespagramDialog,
)
from shm_app.gui.dialogs_io import ReadDataDialog, SeedLinkStreamDialog
from shm_app.gui.dialogs_location import EventSummaryDialog, NetworkLocationDialog
from shm_app.gui.help_texts import apply_tutorial_tooltips, register_tutorial_help, render_tooltip
from shm_app.gui.main_window_menus import build_menus_and_toolbar
from shm_app.gui.panels import ParameterBoxPanel, PhaseDialogPanel
from shm_app.gui.reporting import BulletinWorker
from shm_app.gui.ui_helpers import apply_c_locale, get_dark_stylesheet
from shm_app.gui.waveform_canvases import MagnifierCanvas, MainTraceCanvas
from shm_app.gui.world_map import WorldMapDialog
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.io.geojson import write_geojson
from shm_app.io.gse import read_gse_file
from shm_app.io.mseed import read_mseed_file
from shm_app.io.quakeml import parse_quakeml, write_quakeml
from shm_app.plugins.bgr_dialog import BGRFetchDialog
from shm_app.plugins.live_dialog import LiveEventDialog
from shm_app.plugins.live_service import LiveEventCoordinator, build_analysis_param_set
from shm_app.processing.filters import filter_trace
from shm_app.processing.trace_ops import cut_trace
from shm_app.seismology.focal_mechanism import extract_polarities_from_traces, solve_focal_mechanism
from shm_app.seismology.traveltime import get_ttt_manager


def apply_dark_theme(app: Optional[QApplication]) -> None:
    """Force consistent Fusion style and dark QPalette across desktop environments."""
    apply_c_locale()
    if app is None:
        return
    if app.property("_shm_dark_theme_applied"):
        return
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#10141d"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#0d1117"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#21262d"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#58a6ff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#1f6feb"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Light, QColor("#3b434f"))
    pal.setColor(QPalette.ColorRole.Midlight, QColor("#2c333d"))
    pal.setColor(QPalette.ColorRole.Mid, QColor("#30363d"))
    pal.setColor(QPalette.ColorRole.Dark, QColor("#484f58"))
    pal.setColor(QPalette.ColorRole.Shadow, QColor("#090c10"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8b949e"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#484f58"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#484f58"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6e7681"))
    app.setPalette(pal)
    app.setProperty("_shm_dark_theme_applied", True)


class _RecomputeFitWorker(QThread):
    """Background worker for >50 ms FitDiagnosticWindow re-fits (FK/plane-wave and Brune spectrum)."""

    finishedDiagnostic = pyqtSignal(int, object, str)  # (seq_id, FitDiagnosticResult, log_msg)
    failed = pyqtSignal(int, str)  # (seq_id, error_msg)

    def __init__(
        self,
        seq_id: int,
        fit_type: FitType,
        *,
        traces: Optional[List[Trace]] = None,
        trace: Optional[Trace] = None,
        t_lo: float = 0.0,
        t_hi: float = 0.0,
        dist_deg: float = 1.0,
        config: Any = None,
        calib_table: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.seq_id = int(seq_id)
        self.fit_type = fit_type
        self.traces = traces or []
        self.trace = trace
        self.t_lo = float(t_lo)
        self.t_hi = float(t_hi)
        self.dist_deg = float(dist_deg)
        self.config = config
        self.calib_table = calib_table
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            if self._is_cancelled:
                return
            if self.fit_type in (FitType.FK_SEMBLANCE, FitType.PLANE_WAVE):
                from shm_app.seismology.array_loc import (
                    build_fk_fit_diagnostic,
                    compute_array_response_function,
                    compute_beam_diagnostics,
                    compute_fk_analysis,
                )

                cfg = self.config or get_config()
                fk_res = compute_fk_analysis(
                    self.traces,
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    f_lo=0.5,
                    f_hi=3.0,
                    max_slowness=15.0,
                    grid_points=51,
                    config=cfg,
                    progress_callback=lambda _cur, _tot: not self._is_cancelled,
                    calib_table=self.calib_table,
                    apply_calibration=self.calib_table is not None,
                )
                if self._is_cancelled or not fk_res:
                    return
                beam_diag = compute_beam_diagnostics(
                    self.traces,
                    azimuth_deg=float(fk_res["azimuth"]),
                    slowness_s_deg=float(fk_res["slowness"]),
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    config=cfg,
                )
                if self._is_cancelled:
                    return
                rel_coords = cfg.relative_xy_km(
                    [t.station for t in self.traces],
                    ref_station="CENTRE",
                )
                x_km = [c[0] for c in rel_coords]
                y_km = [c[1] for c in rel_coords]
                arf = compute_array_response_function(
                    x_km,
                    y_km,
                    slowness_max=15.0,
                    n_grid=51,
                    freq_hz=1.5,
                )
                if self._is_cancelled:
                    return
                diag = build_fk_fit_diagnostic(
                    fk_res,
                    beam_diag,
                    arf_result=arf,
                    calib_info=fk_res,
                )
                msg = (
                    f"Recomputed {self.fit_type.value}: BAZ={fk_res['azimuth']:.1f} deg, "
                    f"Slow={fk_res['slowness']:.2f} s/deg"
                )
                if not self._is_cancelled:
                    self.finishedDiagnostic.emit(self.seq_id, diag, msg)
                return

            if self.fit_type == FitType.BRUNE_SPECTRUM:
                if self.trace is None:
                    raise ValueError("No trace provided for Brune spectrum recomputation.")
                from shm_app.processing.spectrum import build_brune_fit_diagnostic, invert_brune_spectrum

                inv_res = invert_brune_spectrum(
                    self.trace,
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    distance_m=float(self.dist_deg) * 111194.9266,
                )
                if self._is_cancelled:
                    return
                diag = build_brune_fit_diagnostic(inv_res, station=self.trace.station)
                if not self._is_cancelled:
                    self.finishedDiagnostic.emit(self.seq_id, diag, "")
                return
        except Exception as exc:
            if not self._is_cancelled:
                prefix = (
                    "Brune re-fit error"
                    if self.fit_type == FitType.BRUNE_SPECTRUM
                    else f"Re-fit error ({self.fit_type.value})"
                )
                self.failed.emit(self.seq_id, f"{prefix}: {exc}")


class SHMMainWindow(QMainWindow):
    """Complete modernized SeismicHandler Motif (SHM) desktop application window."""

    def __init__(
        self,
        shm_root: str = SHM_DEFAULT_ROOT,
        autoload_example: bool = False,
        parent: Optional[QWidget] = None,
        session_path: Optional[Union[str, os.PathLike]] = None,
        prompt_session_recovery: Optional[bool] = None,
    ) -> None:
        super().__init__(parent)
        apply_dark_theme(QApplication.instance())
        self.config = get_config(shm_root)
        self.ttt = get_ttt_manager(self.config.inputs_dir)
        self.engine = SHCommandEngine(config=self.config, output_callback=self._append_console_log)

        # Keep unfiltered backup of loaded traces keyed by stable UID for Ctrl+U (Unfilter)
        self._unfiltered_by_uid: Dict[str, Trace] = {}
        self._unfiltered_backup: List[Trace] = []

        # Undo / redo, non-modal dialogs, workers, session and ALIGN toggle state
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._open_dialogs: List[QDialog] = []
        self._bulletin_workers: List[Any] = []
        self._recompute_workers: List[_RecomputeFitWorker] = []
        self._recompute_seq: int = 0
        self._align_saved_offsets: Dict[Tuple[str, str, float], float] = {}
        self._pending_ready_event_id: Optional[str] = None
        self._last_pmotion_station: Optional[str] = None
        self._last_fit_diagnostic: Any = None
        self._last_focal_mechanism: Optional[Dict[str, Any]] = None

        self._explicit_session_path: bool = session_path is not None
        resolved_sess = resolve_session_path(
            session_path if session_path is not None else (os.environ.get("SHM_SESSION_PATH") or None)
        )
        self.session_manager = SessionManager(resolved_sess)
        self._has_autosaved: bool = False

        self.live_coordinator = LiveEventCoordinator(
            shm_config=self.config,
            ttt=self.ttt,
            is_workspace_idle_cb=self._is_workspace_idle_for_live,
            parent=self,
        )

        self.setWindowTitle("SeismicHandler Motif (SHM 3.0 Modernized)")
        self.resize(1380, 860)
        self._apply_dark_stylesheet()

        # Central vertical splitter: Main Multi-Trace Canvas (top), Magnifier Canvas (middle), Command Console (bottom)
        self.central_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_canvas = MainTraceCanvas()
        self.mag_canvas = MagnifierCanvas()
        self.engine.norm_mode = self.main_canvas.norm_mode
        self._synced_view_window_version: int = self.engine._view_window_version

        # Command Console Widget
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(4, 2, 4, 2)
        console_layout.setSpacing(2)
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setMaximumHeight(110)
        self.console_log.setFontFamily("Monospace")

        cmd_row = QHBoxLayout()
        cmd_row.addWidget(QLabel("SH>"))
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText(
            "Enter any SH command (e.g., FILTER WWSSN_SP, BUTTER BP 0.7 2.5 3, BEAM 45 6.5, FK, LOCATE, LIST, HELP)..."
        )
        self.cmd_input.returnPressed.connect(self._on_run_command_line)
        btn_run_cmd = QPushButton("Execute")
        btn_run_cmd.clicked.connect(self._on_run_command_line)
        register_tutorial_help(
            self.cmd_input,
            "SHM Command Prompt",
            "cmd_input",
            "",
            "Type SH commands (e.g. FILTER SH_BP, BEAM, THEO 65 30, FETCH /BGR) and press Enter",
        )
        register_tutorial_help(
            btn_run_cmd,
            "Execute SHM Command",
            "cmd_execute",
            "",
            "Run the typed SH command",
        )
        cmd_row.addWidget(self.cmd_input)
        cmd_row.addWidget(btn_run_cmd)

        console_layout.addWidget(self.console_log)
        console_layout.addLayout(cmd_row)

        # Main canvas with vertical & horizontal scrollbars plus horizontal zoom slider
        canvas_container = QWidget()
        canvas_grid = QGridLayout(canvas_container)
        canvas_grid.setContentsMargins(0, 0, 0, 0)
        canvas_grid.setSpacing(0)
        canvas_grid.setRowStretch(0, 1)
        canvas_grid.setColumnStretch(0, 1)

        self.v_scroll = QScrollBar(Qt.Orientation.Vertical)
        self.v_scroll.setVisible(False)

        h_bar = QWidget()
        h_bar_l = QHBoxLayout(h_bar)
        h_bar_l.setContentsMargins(0, 2, 4, 2)
        h_bar_l.setSpacing(8)
        self.h_scroll = QScrollBar(Qt.Orientation.Horizontal)
        register_tutorial_help(
            self.h_scroll,
            "Horizontal Time Scrollbar",
            "h_scroll",
            "",
            "Horizontal time scroll bar (active when zoomed into the time window)",
        )
        self.h_scroll.setRange(0, 0)
        self.h_scroll.setEnabled(False)
        h_bar_l.addWidget(self.h_scroll, stretch=1)

        h_zoom_title = QLabel("Time Zoom:")
        h_zoom_title.setStyleSheet("color: #8b949e;")
        self.h_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_zoom_slider.setRange(0, 100)
        self.h_zoom_slider.setValue(0)
        self.h_zoom_slider.setFixedWidth(150)
        register_tutorial_help(
            self.h_zoom_slider,
            "Horizontal Time Zoom Slider",
            "h_zoom_slider",
            "",
            "Horizontal time zoom (Full span .. 50x)",
        )
        self.h_zoom_lbl = QLabel("Full")
        self.h_zoom_lbl.setMinimumWidth(42)
        self.h_zoom_lbl.setStyleSheet("color: #e6edf3; font-family: Monospace;")
        h_bar_l.addWidget(h_zoom_title)
        h_bar_l.addWidget(self.h_zoom_slider)
        h_bar_l.addWidget(self.h_zoom_lbl)

        h_bar_l.addSpacing(10)
        row_title = QLabel("Row Height:")
        row_title.setStyleSheet("color: #8b949e;")
        self.row_slider = QSlider(Qt.Orientation.Horizontal)
        self.row_slider.setRange(0, self.ROW_SLIDER_MAX)
        self.row_slider.setFixedWidth(120)
        register_tutorial_help(
            self.row_slider,
            "Trace Row Height Slider",
            "row_slider",
            "",
            "Trace row height: far left = fit all traces in the window, otherwise a fixed height with scrolling",
        )
        self.row_slider.valueChanged.connect(self._on_row_slider)
        self.row_label = QLabel("Fit")
        self.row_label.setMinimumWidth(48)
        self.row_label.setStyleSheet("color: #e6edf3; font-family: Monospace;")
        h_bar_l.addWidget(row_title)
        h_bar_l.addWidget(self.row_slider)
        h_bar_l.addWidget(self.row_label)

        canvas_grid.addWidget(self.main_canvas, 0, 0)
        canvas_grid.addWidget(self.v_scroll, 0, 1)
        canvas_grid.addWidget(h_bar, 1, 0)

        self.central_splitter.addWidget(canvas_container)
        self.central_splitter.addWidget(self.mag_canvas)
        self.central_splitter.addWidget(console_widget)
        self.central_splitter.setStretchFactor(0, 6)
        self.central_splitter.setStretchFactor(1, 2)
        self.central_splitter.setStretchFactor(2, 1)
        self.setCentralWidget(self.central_splitter)

        # Right Dock: Phase Dialog Panel (wrapped in QScrollArea so height can shrink <= 800 px)
        self.phase_dock = QDockWidget("Phase Picking Box", self)
        self.phase_panel = PhaseDialogPanel(self)
        self.phase_dock.setWidget(self._wrap_dock_scroll(self.phase_panel))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.phase_dock)

        # Right Dock: Parameter Box Panel (wrapped in QScrollArea so height can shrink <= 800 px)
        self.param_dock = QDockWidget("Analysis Parameter Box", self)
        self.param_panel = ParameterBoxPanel(self.engine.param_sets, self, config=self.config)
        self.param_dock.setWidget(self._wrap_dock_scroll(self.param_panel))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.param_dock)

        # Bottom Dock: Fit Diagnostics & Inversion Residuals
        self.fit_bridge = FitDiagnosticBridge(self)
        self.fit_dock = FitDiagnosticWindow(self, bridge=self.fit_bridge)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.fit_dock)
        self.fit_dock.hide()
        self.engine.fit_diagnostic_callback = self.fit_bridge.fitDiagnosticAvailable.emit
        self.fit_bridge.fitDiagnosticAvailable.connect(self._on_fit_diagnostic_received)
        self.fit_bridge.recomputeFitRequested.connect(self._on_recompute_fit_requested)
        self.fit_bridge.adoptParametersRequested.connect(self._on_adopt_network_loc)
        self.fit_bridge.highlightTraceRequested.connect(self._on_highlight_fit_trace)
        self.fit_bridge.excludeStationRequested.connect(self._on_exclude_fit_station)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.cursor_info_lbl = QLabel("Ready")
        self.cursor_info_lbl.setMinimumWidth(180)
        self.cursor_info_lbl.setStyleSheet("font-family: Monospace;")
        self.state_lbl = QLabel("")
        self.state_lbl.setStyleSheet("color: #8b949e;")
        self.state_lbl.setMinimumWidth(200)
        self.live_notify_btn = QPushButton("")
        self.live_notify_btn.setObjectName("live_notify_btn")
        self.live_notify_btn.setVisible(False)
        self.live_notify_btn.clicked.connect(self._on_live_notify_clicked)
        register_tutorial_help(
            self.live_notify_btn,
            "Live Event Ready Notification",
            "live_notify",
            "",
            "Click to load the prefetched Live Event waveforms into SHM",
        )
        self.status_bar.addPermanentWidget(self.live_notify_btn)
        self.status_bar.addPermanentWidget(self.state_lbl)
        self.status_bar.addPermanentWidget(self.cursor_info_lbl)

        self.live_coordinator.enabledChanged.connect(self._on_live_enabled_changed)
        self.live_coordinator.entriesUpdated.connect(self._sync_open_world_map)
        self.live_coordinator.eventReadyToNotify.connect(self._on_live_event_ready)
        self.live_coordinator.loadEventRequested.connect(self._on_live_load_requested)
        self.live_coordinator.logMessage.connect(self._append_console_log)
        self.live_coordinator.seedlinkTracesUpdated.connect(self._on_seedlink_traces_updated)
        self.live_coordinator.seedlinkStatusChanged.connect(self._on_seedlink_status_changed)
        self.live_coordinator.seedlinkErrorOccurred.connect(self._on_seedlink_error)

        # Keep status-bar state summary current (cheap, text only updated on change)
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(250)
        self._state_timer.timeout.connect(self._update_state_label)
        self._state_timer.start()

        # Periodic 60s atomic workspace autosave (M-23)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60000)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._autosave_timer.start()

        # Vertical & horizontal scrollbars <-> canvas
        self.main_canvas.verticalRangeChanged.connect(self._on_vertical_range_changed)
        self.main_canvas.verticalOffsetChanged.connect(self._on_vertical_offset_changed)
        self.v_scroll.valueChanged.connect(lambda v: self.main_canvas.set_v_offset(float(v)))

        self.main_canvas.horizontalRangeChanged.connect(self._on_horizontal_range_changed)
        self.main_canvas.horizontalValueChanged.connect(self._on_horizontal_value_changed)
        self.main_canvas.horizontalZoomChanged.connect(self._on_horizontal_zoom_changed)
        self.main_canvas.amplitudeZoomChanged.connect(self._on_canvas_amp_zoom_changed)
        self.main_canvas.viewChanged.connect(self._on_canvas_view_changed)
        self.h_scroll.valueChanged.connect(self.main_canvas.set_time_offset_from_scroll)
        self.h_zoom_slider.valueChanged.connect(self._on_h_zoom_slider)

        # Pick editing, overlay filters & context menu
        self.main_canvas.pickDeleteRequested.connect(self._on_pick_delete_requested)
        self.main_canvas.pickMoved.connect(self._on_pick_moved)
        self.main_canvas.pickChanged.connect(self._on_pick_changed)
        self.main_canvas.traceInvertRequested.connect(self._on_trace_invert_requested)
        self.main_canvas.overlayFilterRequested.connect(self._on_overlay_filter_requested)
        self.main_canvas.clearOverlayFilterRequested.connect(self._on_clear_overlay_filter_requested)
        self.main_canvas.contextMenuRequested.connect(self._on_trace_context_menu)

        # Connect canvas & dock signals
        self.main_canvas.traceClicked.connect(self._on_main_trace_clicked)
        self.main_canvas.magnifierWindowChanged.connect(self._on_magnifier_window_changed)
        self.main_canvas.cursorMoved.connect(self._on_cursor_moved)
        self.main_canvas.cursorModeCycled.connect(self._on_cursor_mode_cycled)

        self.mag_canvas.pickPlaced.connect(self._on_magnifier_pick_placed)
        self.mag_canvas.pickChanged.connect(self._on_pick_changed)
        self.mag_canvas.amplitudeMeasured.connect(self._on_magnifier_amplitude_measured)
        self.mag_canvas.cursorModeCycled.connect(self._on_cursor_mode_cycled)
        self.mag_canvas.cursorOverlayModeChanged.connect(self._on_mag_cursor_overlay_mode_changed)

        self.phase_panel.deleteActivePickRequested.connect(self.action_delete_active_pick)
        self.phase_panel.depthPhaseCheckToggled.connect(self._on_depth_phase_check_toggled)

        self.param_panel.parametersUpdated.connect(self._on_param_panel_updated)
        self.param_panel.normChanged.connect(self._on_norm_changed)
        self.param_panel.zoomRequested.connect(self._on_zoom_requested)
        self.param_panel.timeStepRequested.connect(self._on_time_step_requested)

        # Build all 12 SHM Menus and Toolbar
        self._build_menus_and_toolbar()
        self.param_panel.set_norm_mode(self.engine.norm_mode)
        apply_c_locale(self)

        if autoload_example:
            self.load_example_dataset("GRF_031102_0225.GSE")
            self._undo_stack.clear()
            self._update_undo_actions()

        no_prompt = os.environ.get("SHM_NO_SESSION_PROMPT", "").strip().lower() in ("1", "true", "yes")
        if (prompt_session_recovery is not False) and (not no_prompt) and self.session_manager.has_unclean_session():
            ans = QMessageBox.question(
                self,
                "Restore Previous Session",
                "Previous session terminated unexpectedly. Restore workspace state?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.Yes:
                self.restore_session()

    @staticmethod
    def _wrap_dock_scroll(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        min_w = max(widget.minimumSizeHint().width(), widget.sizeHint().width()) + 16
        scroll.setMinimumWidth(min_w)
        scroll.setMinimumHeight(110)
        return scroll

    def _apply_dark_stylesheet(self) -> None:
        self.setStyleSheet(get_dark_stylesheet())

    def _build_menus_and_toolbar(self) -> None:
        build_menus_and_toolbar(self)

    # ------------------------------------------------------------------
    # Tutorial Mode
    # ------------------------------------------------------------------

    def _refresh_tooltips(self) -> None:
        for act, title, key, sc in self._help_actions:
            if self.tutorial_mode:
                act.setToolTip(render_tooltip(title, key, sc))
            else:
                act.setToolTip(f"{title} ({sc})" if sc else title)
        for m in self._menus:
            m.setToolTipsVisible(self.tutorial_mode)
        apply_tutorial_tooltips(self, self.tutorial_mode)
        for dlg in list(self._open_dialogs):
            if dlg is not None and not sip.isdeleted(dlg):
                apply_tutorial_tooltips(dlg, self.tutorial_mode)

    def _set_tutorial_mode(self, on: bool) -> None:
        self.tutorial_mode = bool(on)
        self._refresh_tooltips()
        self.status_bar.showMessage(
            "Tutorial Mode ON: hover over any menu entry or toolbar button for an explanation."
            if on
            else "Tutorial Mode off.",
            6000,
        )

    # ------------------------------------------------------------------
    # Undo / Redo (snapshots of traces and parameter sets)
    # ------------------------------------------------------------------

    UNDO_LIMIT = 20

    def _snapshot(self, label: str):
        return (
            label,
            [t.copy() for t in self.engine.traces],
            copy.deepcopy(self.engine.param_sets),
            self.engine.active_set_index,
            dict(self._unfiltered_by_uid),
        )

    def _push_undo(self, label: str) -> None:
        self._undo_stack.append(self._snapshot(label))
        del self._undo_stack[: -self.UNDO_LIMIT]
        self._redo_stack.clear()
        self._update_undo_actions()

    def _restore_snapshot(self, snap) -> None:
        if len(snap) >= 5:
            _label, traces, params, active, unfiltered_map = snap[:5]
            self._unfiltered_by_uid = dict(unfiltered_map)
        else:
            _label, traces, params, active = snap[:4]
        self.engine.traces = traces
        self.engine.param_sets[:] = params
        self.engine.active_set_index = active
        self._sync_unfiltered_backup_view()
        self.refresh_views()

    def _update_undo_actions(self) -> None:
        if hasattr(self, "act_undo"):
            self.act_undo.setEnabled(bool(self._undo_stack))
            self.act_redo.setEnabled(bool(self._redo_stack))
            nxt = self._undo_stack[-1][0] if self._undo_stack else ""
            self.act_undo.setText(f"&Undo {nxt}".strip())

    def action_undo(self) -> None:
        if not self._undo_stack:
            self.status_bar.showMessage("Nothing to undo.", 3000)
            return
        snap = self._undo_stack.pop()
        self._redo_stack.append(self._snapshot(snap[0]))
        self._restore_snapshot(snap)
        self._update_undo_actions()
        self.status_bar.showMessage(f"Undid: {snap[0]}", 4000)

    def action_redo(self) -> None:
        if not self._redo_stack:
            self.status_bar.showMessage("Nothing to redo.", 3000)
            return
        snap = self._redo_stack.pop()
        self._undo_stack.append(self._snapshot(snap[0]))
        self._restore_snapshot(snap)
        self._update_undo_actions()
        self.status_bar.showMessage(f"Redid: {snap[0]}", 4000)

    # ------------------------------------------------------------------
    # Pick editing, trace context menu, view history
    # ------------------------------------------------------------------

    def _on_pick_delete_requested(self, trace_idx: int, phase: str) -> None:
        if not (0 <= trace_idx < len(self.engine.traces)):
            return
        tr = self.engine.traces[trace_idx]
        if not any(p.name == phase for p in tr.picks):
            return
        self._push_undo(f"Delete {phase} pick")
        tr.picks = [p for p in tr.picks if p.name != phase]
        self.main_canvas.invalidate_slot(trace_idx)
        self.mag_canvas.invalidate_cache()
        self.status_bar.showMessage(f"Deleted {phase} pick on {tr.station} (Ctrl+Z to undo).", 4000)

    def _on_pick_moved(self, trace_idx: int, phase: str, epoch: float) -> None:
        if not (0 <= trace_idx < len(self.engine.traces)):
            return
        tr = self.engine.traces[trace_idx]
        pk = tr.get_pick(phase)
        if pk is None:
            return
        self._push_undo(f"Move {phase} pick")
        pk.onset_epoch = epoch
        self.main_canvas.invalidate_slot(trace_idx)
        self.mag_canvas.invalidate_cache()
        self.status_bar.showMessage(f"Moved {phase} pick on {tr.station} to {pk.onset}.", 4000)

    def _on_trace_context_menu(self, trace_idx: int, rel_t: float, gx: int, gy: int) -> None:
        if not (0 <= trace_idx < len(self.engine.traces)):
            return
        tr = self.engine.traces[trace_idx]
        ref_ep = self.main_canvas.reference_epoch()
        epoch = self.main_canvas.snap_time_to_feature(tr, ref_ep + rel_t - tr.display_offset)
        phase = self.phase_panel.active_phase

        menu = QMenu(self)
        header = menu.addAction(f"#{trace_idx + 1}  {tr.station} {tr.channel3}")
        header.setEnabled(False)
        menu.addSeparator()
        menu.addAction(f"Pick {phase} here", lambda: self._on_main_trace_clicked(trace_idx, epoch, rel_t))
        if tr.picks:
            sub = menu.addMenu("Delete pick")
            for pk in list(tr.picks):
                sub.addAction(
                    f"{pk.spec}{pk.sign_char}{pk.name}",
                    lambda name=pk.name: self._on_pick_delete_requested(trace_idx, name),
                )
        menu.addSeparator()
        menu.addAction("Focus Magnifier here", lambda: self._focus_magnifier(trace_idx, rel_t))
        menu.addAction(
            "Deselect trace" if tr.selected else "Select trace",
            lambda: self._toggle_trace_selection(trace_idx),
        )
        menu.addAction("Trace Overlay Filter...", lambda: self._on_overlay_filter_requested(trace_idx))
        if getattr(tr, "overlay_filter_name", "NONE") not in ("", "NONE"):
            menu.addAction(
                f"Clear Overlay Filter ({tr.overlay_filter_name})",
                lambda: self._on_clear_overlay_filter_requested(trace_idx),
            )
        menu.addAction("Invert sign", lambda: self._invert_trace_sign(trace_idx))
        menu.addAction("Hide trace", lambda: self._hide_trace(trace_idx))
        menu.addAction("Delete trace", lambda: self._delete_trace(trace_idx))
        menu.exec(QPoint(gx, gy))

    def _focus_magnifier(self, trace_idx: int, rel_t: float) -> None:
        half_w = 0.5 * max(2.0, self.main_canvas.mag_t_hi - self.main_canvas.mag_t_lo)
        self.main_canvas.mag_trace_idx = trace_idx
        self.main_canvas.mag_t_lo = rel_t - half_w
        self.main_canvas.mag_t_hi = rel_t + half_w
        self._on_magnifier_window_changed(trace_idx, self.main_canvas.mag_t_lo, self.main_canvas.mag_t_hi)
        self.main_canvas.update()

    def _toggle_trace_selection(self, trace_idx: int) -> None:
        tr = self.engine.traces[trace_idx]
        tr.selected = not tr.selected
        self.main_canvas.invalidate_slot(trace_idx)

    def _invert_trace_sign(self, trace_idx: int) -> None:
        self._push_undo("Invert sign")
        tr = self.engine.traces[trace_idx]
        tr.data = -tr.data
        self.main_canvas.invalidate_slot(trace_idx)
        self.mag_canvas.invalidate_cache()

    def _hide_trace(self, trace_idx: int) -> None:
        self._push_undo("Hide trace")
        self.engine.traces[trace_idx].hidden = True
        self.refresh_views()
        self.status_bar.showMessage("Trace hidden. Use Traces > Show All Hidden Traces to restore.", 5000)

    def _delete_trace(self, trace_idx: int) -> None:
        self._push_undo("Delete trace")
        del self.engine.traces[trace_idx]
        self.refresh_views()

    def action_show_hidden(self) -> None:
        self._push_undo("Show hidden traces")
        for tr in self.engine.traces:
            tr.hidden = False
        self.refresh_views()

    def action_view_back(self) -> None:
        if not self.main_canvas.view_back():
            self.status_bar.showMessage("No earlier view.", 2500)

    def action_view_forward(self) -> None:
        if not self.main_canvas.view_forward():
            self.status_bar.showMessage("No later view.", 2500)

    # ------------------------------------------------------------------
    # Row height slider and component grouping
    # ------------------------------------------------------------------

    ROW_SLIDER_MAX = 184  # slider value v maps to 16 + v px (0 = fit all traces)

    @staticmethod
    def _slider_to_px(v: int) -> int:
        return 0 if v <= 0 else 16 + v

    @staticmethod
    def _px_to_slider(px: float) -> int:
        return 0 if px <= 0 else max(1, min(SHMMainWindow.ROW_SLIDER_MAX, int(round(px)) - 16))

    def _set_row_height(self, px: int) -> None:
        """Set the minimum trace row height (0 = fit) and keep slider, label and presets in sync."""
        self.main_canvas.set_min_slot_height(px)
        self.row_slider.blockSignals(True)
        self.row_slider.setValue(self._px_to_slider(px))
        self.row_slider.blockSignals(False)
        self.row_label.setText("Fit" if px <= 0 else f"{int(px)} px")
        for preset_px, act in self._row_actions.items():
            act.setChecked(preset_px == px)

    def _on_row_slider(self, value: int) -> None:
        px = self._slider_to_px(value)
        self.main_canvas.set_min_slot_height(px)
        self.row_label.setText("Fit" if px <= 0 else f"{px} px")
        for preset_px, act in self._row_actions.items():
            act.setChecked(preset_px == px)

    def action_toggle_grouping(self) -> None:
        on = self.act_group.isChecked()
        self.main_canvas.set_group_by_station(on)
        self.status_bar.showMessage(
            "Components grouped by station (Z, N, E)." if on else "Original trace order restored.", 4000
        )

    def action_toggle_pick_error_bars(self, checked: Optional[bool] = None) -> None:
        if checked is not None:
            on = bool(checked)
        else:
            on = self.act_pick_error_bars.isChecked() if hasattr(self, "act_pick_error_bars") else True
        self.main_canvas.set_show_pick_error_bars(on)
        self.mag_canvas.set_show_pick_error_bars(on)
        self.status_bar.showMessage(
            f"Pick uncertainty error bars {'shown' if on else 'hidden'}.", 3000
        )

    def action_set_mag_cursor_mode(self, mode: str) -> None:
        self.mag_canvas.set_cursor_overlay_mode(mode)
        self._on_mag_cursor_overlay_mode_changed(self.mag_canvas.cursor_overlay_mode)
        self.status_bar.showMessage(
            f"Magnifier cursor mode: {self.mag_canvas.cursor_overlay_mode}", 3000
        )

    def _on_mag_cursor_overlay_mode_changed(self, mode: str) -> None:
        m = (mode or "STANDARD").upper()
        mapping = {
            "STANDARD": getattr(self, "act_mag_cursor_standard", None),
            "HILBERT": getattr(self, "act_mag_cursor_hilbert", None),
            "REFERENCE": getattr(self, "act_mag_cursor_reference", None),
        }
        act = mapping.get(m)
        if act is not None and not act.isChecked():
            act.blockSignals(True)
            act.setChecked(True)
            act.blockSignals(False)

    def action_capture_reference_window(self) -> None:
        if self.mag_canvas.capture_reference_window():
            if self.mag_canvas.cursor_overlay_mode == "STANDARD":
                self.mag_canvas.set_cursor_overlay_mode("REFERENCE")
            self.status_bar.showMessage(
                "Captured Magnifier window as Reference / Hilbert cursor template.", 4000
            )
            self._append_console_log("Captured reference waveform window from Magnifier.")
        else:
            self.status_bar.showMessage("No active Magnifier waveform window to capture.", 4000)

    def _on_pick_changed(self, trace_idx: int, pk: Any) -> None:
        if 0 <= trace_idx < len(self.engine.traces):
            self.main_canvas.invalidate_slot(trace_idx)
        else:
            self.main_canvas.invalidate_cache()
        self.mag_canvas.invalidate_cache()
        if pk is not None and 0 <= trace_idx < len(self.engine.traces):
            tr = self.engine.traces[trace_idx]
            self.status_bar.showMessage(
                f"Updated {pk.name} pick on {tr.station}: Q={pk.quality} (±{pk.sigma_t(tr.delta):.2f} s)",
                4000,
            )

    # ------------------------------------------------------------------
    # Spectrum: curves from the yellow Magnifier window, accumulated in one graph
    # ------------------------------------------------------------------

    def _spectrum_window_for(self, tr: Trace) -> Tuple[float, float]:
        """Magnifier window converted to trace-relative seconds (clamped to the trace)."""
        cv = self.main_canvas
        rel_start = tr.start_epoch + tr.display_offset - cv.reference_epoch()
        lo = max(0.0, cv.mag_t_lo - rel_start)
        hi = min(tr.duration, cv.mag_t_hi - rel_start)
        return lo, hi

    def _add_spectra(self, traces: List[Trace]) -> None:
        cv = self.main_canvas
        items = []
        skipped = []
        for tr in traces:
            lo, hi = self._spectrum_window_for(tr)
            if hi - lo < 8 * tr.delta:
                skipped.append(f"{tr.station} {tr.channel3}")
                continue
            label = f"{tr.station} {tr.channel3}  [{cv.mag_t_lo:.1f}..{cv.mag_t_hi:.1f} s]"
            items.append((tr, lo, hi, label))
        if not items:
            self.status_bar.showMessage(
                "The yellow window contains no data of the chosen trace(s): move it over the signal.", 6000
            )
            return
        dlg = self._find_open_dialog(SpectrumDialog)
        is_new = dlg is None
        if is_new:
            dlg = SpectrumDialog(None, parent=self)
            dlg.fitDiagnosticReady.connect(self.fit_bridge.fitDiagnosticAvailable.emit)
        added = sum(1 for tr, lo, hi, label in items if dlg.add_curve(tr, lo, hi, label))
        if is_new:
            self._show_nonmodal(dlg)
        else:
            self._raise_nonmodal(dlg)
        msg = f"Spectrum: {added} curve(s) added ({len(dlg.curves)} total)."
        if added == 0:
            msg = "Spectrum for this window and trace is already in the graph."
        if skipped:
            msg += " No data in the window for: " + ", ".join(skipped)
        self.status_bar.showMessage(msg, 6000)

    def action_open_spectrum_dialog(self) -> None:
        if not self.engine.visible_traces():
            return
        idx = max(0, min(self.main_canvas.mag_trace_idx, len(self.engine.traces) - 1))
        self._add_spectra([self.engine.traces[idx]])

    def action_open_spectrum_selected(self) -> None:
        sel = [t for t in self.engine.traces if t.selected and not t.hidden]
        if not sel:
            self.status_bar.showMessage("No traces selected: click the left margin of a trace first.", 5000)
            return
        self._add_spectra(sel)

    def _prune_open_dialogs(self) -> None:
        """Drop any destroyed C++ dialog wrappers from ``_open_dialogs``."""
        self._open_dialogs = [d for d in self._open_dialogs if not sip.isdeleted(d)]

    def _find_open_dialog(self, dlg_cls):
        self._prune_open_dialogs()
        return next(
            (
                d for d in self._open_dialogs
                if not sip.isdeleted(d)
                and isinstance(d, dlg_cls)
                and d.isVisible()
                and not getattr(d, "_closing", False)
            ),
            None,
        )

    @staticmethod
    def _raise_nonmodal(dlg: QDialog) -> None:
        if dlg.isMinimized():
            dlg.showNormal()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _remove_open_dialog(self, ref: "weakref.ref[QDialog]") -> None:
        dlg = ref()
        if dlg is None:
            self._prune_open_dialogs()
            return
        self._open_dialogs = [d for d in self._open_dialogs if not sip.isdeleted(d) and d is not dlg]

    def _show_nonmodal(self, dlg: QDialog) -> None:
        """Show an analysis dialog without blocking the main window."""
        self._prune_open_dialogs()
        dlg.setModal(False)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        apply_tutorial_tooltips(dlg, self.tutorial_mode)
        if dlg not in self._open_dialogs:
            self._open_dialogs.append(dlg)
        ref = weakref.ref(dlg)
        cleanup = lambda *_args, r=ref: self._remove_open_dialog(r)
        dlg.finished.connect(cleanup)
        dlg.destroyed.connect(cleanup)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _update_state_label(self) -> None:
        trs = self.engine.traces
        vis = [t for t in trs if not t.hidden]
        sel = sum(1 for t in vis if t.selected)
        lo, hi = self.main_canvas.active_time_span()
        mode = "fit" if self.main_canvas.min_slot_h <= 0 else f"{int(self.main_canvas.min_slot_h)}px"
        txt = (
            f"Phase {self.phase_panel.active_phase}  |  Traces {len(vis)} ({sel} sel)  |  "
            f"View {lo:.1f}..{hi:.1f} s  |  Rows {mode}  |  Norm {self.main_canvas.norm_mode}"
        )
        if txt != self.state_lbl.text():
            self.state_lbl.setText(txt)

    # ------------------------------------------------------------------
    # Single-Key Shortcuts matching SHM Motif XmDrawingArea translations
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        text = event.text()

        # Single-key shortcuts when command input does not have focus
        if not self.cmd_input.hasFocus():
            if text == "+":
                self.phase_panel.sign_combo.setCurrentIndex(1)  # Compression 'c'
                self.status_bar.showMessage("Phase Sign: + (Compression 'c')", 2000)
                return
            if text == "-":
                self.phase_panel.sign_combo.setCurrentIndex(2)  # Dilatation 'd'
                self.status_bar.showMessage("Phase Sign: - (Dilatation 'd')", 2000)
                return
            if text == "0":
                self.phase_panel.sign_combo.setCurrentIndex(0)  # None
                self.status_bar.showMessage("Phase Sign: 0 (None)", 2000)
                return
            if text == "i":
                self.phase_panel.spec_combo.setCurrentIndex(1)  # Impulsive
                self.status_bar.showMessage("Onset Type: i (Impulsive)", 2000)
                return
            if text == "e":
                self.phase_panel.spec_combo.setCurrentIndex(0)  # Emergent
                self.status_bar.showMessage("Onset Type: e (Emergent)", 2000)
                return
            if text in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
                self.phase_panel.qual_spin.setValue(int(text))
                self.status_bar.showMessage(f"Pick Quality set to {text}", 2000)
                return
            if key == Qt.Key.Key_Left:
                self._on_time_step_requested(-0.15)
                return
            if key == Qt.Key.Key_Right:
                self._on_time_step_requested(0.15)
                return
            if key == Qt.Key.Key_Up:
                self._on_zoom_requested(1.25)
                return
            if key == Qt.Key.Key_Down:
                self._on_zoom_requested(0.8)
                return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Workspace & Canvas Synchronization
    # ------------------------------------------------------------------

    def _sync_unfiltered_backup_view(self) -> None:
        self._unfiltered_backup = [
            self._unfiltered_by_uid[tr.extra_headers["_raw_uid"]]
            for tr in self.engine.traces
            if tr.extra_headers.get("_raw_uid") in self._unfiltered_by_uid
        ]

    def _register_unfiltered_traces(
        self,
        traces: Sequence[Trace],
        replace: bool = False,
        overwrite_existing: bool = False,
    ) -> None:
        if replace:
            self._unfiltered_by_uid = {}
        for tr in traces:
            uid = tr.extra_headers.get("_raw_uid")
            if not uid:
                uid = uuid.uuid4().hex
                tr.extra_headers["_raw_uid"] = uid
            if overwrite_existing or uid not in self._unfiltered_by_uid:
                if tr.filter_name == "NONE" or overwrite_existing:
                    raw_copy = tr.copy(copy_picks=False)
                    raw_copy.extra_headers["_raw_uid"] = uid
                    self._unfiltered_by_uid[uid] = raw_copy
        self._sync_unfiltered_backup_view()

    def _invalidate_stale_unfiltered(self) -> None:
        for tr in self.engine.traces:
            uid = tr.extra_headers.get("_raw_uid")
            if not uid or uid not in self._unfiltered_by_uid:
                continue
            raw = self._unfiltered_by_uid[uid]
            if (
                raw.length != tr.length
                or abs(raw.delta - tr.delta) > 1e-6
                or raw.station != tr.station
                or raw.channel3 != tr.channel3
            ):
                if tr.filter_name == "NONE":
                    raw_copy = tr.copy(copy_picks=False)
                    raw_copy.extra_headers["_raw_uid"] = uid
                    self._unfiltered_by_uid[uid] = raw_copy
                else:
                    del self._unfiltered_by_uid[uid]
        self._sync_unfiltered_backup_view()

    def _sync_controls_from_engine(self) -> None:
        mode = self.engine.norm_mode if self.engine.norm_mode in ("A", "C", "AW", "SW") else "A"
        if self.main_canvas.norm_mode != mode:
            self.main_canvas.norm_mode = mode
        self.param_panel.set_norm_mode(mode)
        eng_zoom = max(0.05, min(200.0, float(self.engine.global_zoom)))
        if abs(self.main_canvas.global_zoom - eng_zoom) > 1e-9:
            self.main_canvas.global_zoom = eng_zoom
        if self._synced_view_window_version != self.engine._view_window_version:
            self._synced_view_window_version = self.engine._view_window_version
            if self.engine.view_window is not None:
                self.main_canvas.view_t_lo, self.main_canvas.view_t_hi = self.engine.view_window
            else:
                self.main_canvas.view_t_lo = None
                self.main_canvas.view_t_hi = None

    def refresh_views(self, reset_time: bool = False) -> None:
        self._sync_unfiltered_backup_view()
        self._sync_controls_from_engine()
        if reset_time:
            self.engine.view_window = None
            self._synced_view_window_version = self.engine._view_window_version
        self.main_canvas.set_traces(self.engine.traces, reset_view=reset_time)
        vis = self.engine.visible_traces()
        if vis:
            idx = min(self.main_canvas.mag_trace_idx, len(self.engine.traces) - 1)
            idx = max(0, idx)
            self.main_canvas.mag_trace_idx = idx
            self.mag_canvas.set_magnifier_view(
                self.engine.traces[idx],
                idx,
                self.main_canvas.reference_epoch(),
                self.main_canvas.mag_t_lo,
                self.main_canvas.mag_t_hi,
            )
        else:
            self.mag_canvas.set_magnifier_view(None, 0, 0.0, 0.0, 30.0)
        self.param_panel.refresh_from_model()
        self._sync_open_world_map()

    def _on_vertical_range_changed(self, max_offset: int, page: int) -> None:
        self.v_scroll.setRange(0, max_offset)
        self.v_scroll.setPageStep(max(1, page))
        self.v_scroll.setVisible(max_offset > 0)

    def _on_vertical_offset_changed(self, offset: int) -> None:
        if self.v_scroll.value() != offset:
            self.v_scroll.blockSignals(True)
            self.v_scroll.setValue(offset)
            self.v_scroll.blockSignals(False)

    def _on_horizontal_range_changed(self, min_tick: int, max_tick: int, page: int) -> None:
        self.h_scroll.blockSignals(True)
        self.h_scroll.setRange(min_tick, max_tick)
        self.h_scroll.setPageStep(max(1, page))
        self.h_scroll.setSingleStep(max(1, page // 10))
        self.h_scroll.setEnabled(max_tick > min_tick)
        self.h_scroll.blockSignals(False)

    def _on_horizontal_value_changed(self, val: int) -> None:
        if self.h_scroll.value() != val:
            self.h_scroll.blockSignals(True)
            self.h_scroll.setValue(val)
            self.h_scroll.blockSignals(False)

    def _on_horizontal_zoom_changed(self, zoom_factor: float) -> None:
        clamped = max(1.0, min(50.0, float(zoom_factor)))
        slider_val = 0 if clamped <= 1.01 else int(round(100.0 * math.log(clamped) / math.log(50.0)))
        self.h_zoom_slider.blockSignals(True)
        self.h_zoom_slider.setValue(max(0, min(100, slider_val)))
        self.h_zoom_slider.blockSignals(False)
        self.h_zoom_lbl.setText("Full" if zoom_factor <= 1.01 else f"{zoom_factor:.1f}x")

    def _on_h_zoom_slider(self, val: int) -> None:
        factor = 1.0 if val <= 0 else (50.0 ** (float(val) / 100.0))
        self.h_zoom_lbl.setText("Full" if factor <= 1.01 else f"{factor:.1f}x")
        self.main_canvas.set_horizontal_zoom(factor)

    def _append_console_log(self, text: str) -> None:
        if text:
            self.console_log.append(text)

    def _on_run_command_line(self) -> None:
        cmd = self.cmd_input.text().strip()
        if not cmd:
            return
        self.cmd_input.clear()
        self._append_console_log(f"SH> {cmd}")
        tokens = cmd.split()
        verb = tokens[0].upper() if tokens else "CMD"
        self._push_undo(f"Command {verb}")
        self.engine.execute_line(cmd)
        if self.engine.quit_requested:
            self.close()
            return
        if verb in ("READ", "READS", "READM", "READSEED", "READGSE", "READA", "READEVT", "CREATE", "FETCH", "BGRFETCH"):
            self._register_unfiltered_traces(self.engine.traces)
        self._invalidate_stale_unfiltered()
        self.refresh_views()

    def _on_main_trace_clicked(self, trace_idx: int, epoch_time: float, rel_t: float) -> None:
        if not (0 <= trace_idx < len(self.engine.traces)):
            return
        tr = self.engine.traces[trace_idx]
        pk = self.phase_panel.create_pick_for_trace(tr, epoch_time)
        self._push_undo(f"Pick {pk.name}")
        tr.add_or_replace_pick(pk)
        self.status_bar.showMessage(
            f"Picked {pk.spec}{pk.sign_char}{pk.name} on {tr.station} at {pk.onset} (Q={pk.quality}/W={pk.weight})",
            5000,
        )
        self.main_canvas.invalidate_slot(trace_idx)
        self.mag_canvas.invalidate_cache()

    def _on_magnifier_window_changed(self, trace_idx: int, t_lo: float, t_hi: float) -> None:
        if 0 <= trace_idx < len(self.engine.traces):
            self.mag_canvas.set_magnifier_view(
                self.engine.traces[trace_idx],
                trace_idx,
                self.main_canvas.reference_epoch(),
                t_lo,
                t_hi,
            )
            self.engine.time_window = (t_lo, t_hi)

    def _on_magnifier_pick_placed(self, trace_idx: int, epoch_time: float) -> None:
        self._on_main_trace_clicked(trace_idx, epoch_time, 0.0)

    def _on_trace_invert_requested(self, trace_idx: int) -> None:
        if self.engine.invert_trace(trace_idx):
            self.main_canvas.invalidate_slot(trace_idx)
            self.mag_canvas.invalidate_cache()

    def _on_magnifier_amplitude_measured(self, trace_idx: int, rel_lo: float, rel_hi: float) -> None:
        res = self.engine.measure_amplitude_magnitudes(
            trace_idx,
            rel_lo,
            rel_hi,
            prefer_phase=self.phase_panel.active_phase,
        )
        if res is None:
            return
        self.param_panel.refresh_from_model()
        self._sync_open_world_map()
        msg = (
            f"Amplitude ({res['station']}): A={res['ampl_nm']:.1f} nm, T={res['period']:.2f} s  ->  "
            f"mb={res['mb']:.2f}, ML={res['ml']:.2f}, MS={res['ms']:.2f}, mB={res['mbb']:.2f}"
        )
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 6000)

    def _on_cursor_moved(self, trace_idx: int, epoch_time: float, amp: float) -> None:
        if 0 <= trace_idx < len(self.engine.traces):
            tr = self.engine.traces[trace_idx]
            self.cursor_info_lbl.setText(
                f"#{trace_idx + 1} {tr.station} {tr.channel3}  |  {format_sh_time(epoch_time)}  |  Amp: {amp:+.2f}"
            )

    def _on_depth_phase_check_toggled(self, checked: bool) -> None:
        if not checked:
            self.mag_canvas.depth_phase_cursors = []
        else:
            p = self.engine.current_params
            dist = p.distance if p.distance > 0 else 60.0
            depth = p.depth if p.depth > 0 else 33.0
            t_p = self.ttt.travel_time("P", dist, depth) or 600.0
            t_pp = self.ttt.travel_time("pP", dist, depth) or (t_p + 10.0)
            t_sp = self.ttt.travel_time("sP", dist, depth) or (t_p + 15.0)
            self.mag_canvas.depth_phase_cursors = [("pP", t_pp - t_p), ("sP", t_sp - t_p)]
        self.mag_canvas.update()

    def _on_param_panel_updated(self) -> None:
        self.engine.active_set_index = self.param_panel.active_set_idx
        self._sync_open_world_map()

    def _on_norm_changed(self, mode: str) -> None:
        self.engine.norm_mode = mode
        self.main_canvas.norm_mode = mode
        self.main_canvas.invalidate_cache()

    def _on_cursor_mode_cycled(self, mode: str) -> None:
        if self.main_canvas.cursor_lock_mode != mode:
            self.main_canvas.cursor_lock_mode = mode
            self.main_canvas.invalidate_cache()
        self.mag_canvas.set_cursor_lock_mode(mode)
        self.status_bar.showMessage(f"Cursor Lock Mode: {mode}", 3000)

    def _on_canvas_amp_zoom_changed(self, zoom: float) -> None:
        self.engine.global_zoom = max(0.05, min(200.0, float(zoom)))

    def _on_canvas_view_changed(self) -> None:
        if self.main_canvas.view_t_lo is not None and self.main_canvas.view_t_hi is not None:
            self.engine.view_window = (self.main_canvas.view_t_lo, self.main_canvas.view_t_hi)
        else:
            self.engine.view_window = None

    def _on_zoom_requested(self, factor: float) -> None:
        if factor == 0.0:
            self.main_canvas.global_zoom = 1.0
            self.engine.global_zoom = 1.0
            for t in self.engine.traces:
                t.norm = 1.0
        else:
            new_zoom = max(0.05, min(200.0, self.main_canvas.global_zoom * factor))
            self.main_canvas.global_zoom = new_zoom
            self.engine.global_zoom = new_zoom
        self.main_canvas.invalidate_cache()

    def _on_time_step_requested(self, frac: float) -> None:
        self.main_canvas.push_view_history(coalesce_ms=400)
        lo, hi = self.main_canvas.active_time_span()
        shift = (hi - lo) * frac
        self.main_canvas.view_t_lo = lo + shift
        self.main_canvas.view_t_hi = hi + shift
        self.main_canvas.invalidate_cache()

    # ------------------------------------------------------------------
    # Menu Action Implementations
    # ------------------------------------------------------------------

    def load_example_dataset(self, name: str) -> None:
        if name == "mseed":
            dpath = os.path.join(self.config.data_examples_dir, "mseed")
            sfd_path = os.path.join(dpath, "sfdlist.sfd")
            if os.path.isfile(sfd_path):
                from shm_app.io.mseed import read_from_sfd

                trs = read_from_sfd(sfd_path, config=self.config)
            else:
                trs = []
                for fn in sorted(os.listdir(dpath)):
                    if fn.endswith(".bhz"):
                        trs.extend(read_mseed_file(os.path.join(dpath, fn), config=self.config))
        else:
            fpath = os.path.join(self.config.data_examples_dir, "gse", name)
            trs = read_gse_file(fpath, config=self.config)

        self.engine.traces = trs
        self._register_unfiltered_traces(trs, replace=True, overwrite_existing=True)
        self.refresh_views(reset_time=True)
        self._append_console_log(f"Loaded {len(trs)} traces from example dataset '{name}'.")

    def _load_waveforms(self, traces: List[Trace], append_mode: bool, undo_label: str, log_msg: str) -> None:
        self._push_undo(undo_label)
        if append_mode:
            self.engine.traces.extend(traces)
            self._register_unfiltered_traces(traces, replace=False, overwrite_existing=True)
        else:
            self.engine.traces = list(traces)
            self._register_unfiltered_traces(traces, replace=True, overwrite_existing=True)
        self.refresh_views(reset_time=not append_mode)
        self._append_console_log(log_msg)
        self.status_bar.showMessage(log_msg, 6000)

    def action_open_read_dialog(self) -> None:
        dlg = ReadDataDialog(config=self.config, parent=self)
        if dlg.exec():
            self._load_waveforms(
                dlg.loaded_traces,
                dlg.append_mode,
                "Read waveforms",
                f"Loaded {len(dlg.loaded_traces)} waveform trace(s).",
            )

    def _workspace_stations(self) -> List[str]:
        selected = [tr for tr in self.engine.traces if tr.selected]
        source = selected if selected else self.engine.traces
        out: List[str] = []
        seen = set()
        for tr in source:
            st = (tr.station or "").strip().upper()
            if st and st not in seen:
                seen.add(st)
                out.append(st)
        return out

    def _open_bgr_fetch_prefilled(
        self,
        stations: Optional[Sequence[str]] = None,
        start_epoch: Optional[float] = None,
        duration_sec: Optional[float] = None,
    ) -> None:
        dlg = self._find_open_dialog(BGRFetchDialog)
        if dlg is None:
            view_win = None
            if self.engine.traces:
                lo, hi = self.main_canvas.active_time_span()
                if hi > lo:
                    view_win = (self.main_canvas.reference_epoch() + lo, hi - lo)
            dlg = BGRFetchDialog(
                config=self.config,
                view_window=view_win,
                origin_str=self.engine.current_params.origin,
                workspace_stations=self._workspace_stations(),
                parent=self,
            )
            dlg.waveformsDownloaded.connect(self._on_bgr_waveforms_downloaded)
            self._show_nonmodal(dlg)
        else:
            self._raise_nonmodal(dlg)

        if stations:
            dlg._apply_station_preset([str(s) for s in stations])
        if start_epoch is not None and duration_sec is not None:
            dlg.set_time_window(float(start_epoch), float(duration_sec))

    def action_open_bgr_fetch_dialog(self) -> None:
        self._open_bgr_fetch_prefilled()

    def _on_bgr_waveforms_downloaded(self, traces: list, append_mode: bool, msg: str) -> None:
        if not traces:
            return
        self._load_waveforms(traces, append_mode, "BGR Download", msg)

    # ------------------------------------------------------------------
    # Live Event Mode
    # ------------------------------------------------------------------

    def _is_workspace_idle_for_live(self) -> bool:
        if not self.engine.traces:
            return True
        return not any(pk.source != "t" for tr in self.engine.traces for pk in tr.picks)

    def action_toggle_live_mode(self) -> None:
        self.live_coordinator.set_enabled(not self.live_coordinator.is_enabled)

    def _on_live_enabled_changed(self, enabled: bool) -> None:
        if hasattr(self, "act_live_toggle"):
            self.act_live_toggle.blockSignals(True)
            self.act_live_toggle.setChecked(enabled)
            self.act_live_toggle.setText("LIVE: ON" if enabled else "Live: OFF")
            self.act_live_toggle.blockSignals(False)
        self.status_bar.showMessage(
            "Live Event Mode enabled (monitoring real-time seismic feeds)." if enabled else "Live Event Mode paused.",
            5000,
        )

    def action_open_live_event_dialog(self) -> None:
        existing = self._find_open_dialog(LiveEventDialog)
        if existing is not None:
            self._raise_nonmodal(existing)
            return
        dlg = LiveEventDialog(self.live_coordinator, parent=self)
        dlg.openInBGRRequested.connect(self._on_live_open_in_bgr)
        dlg.showOnMapRequested.connect(self._on_live_show_on_map)
        self._show_nonmodal(dlg)

    def _on_live_open_in_bgr(self, entry) -> None:
        if entry.plan is not None:
            self._open_bgr_fetch_prefilled(
                stations=[t.station for t in entry.plan.targets],
                start_epoch=entry.plan.start_epoch,
                duration_sec=entry.plan.duration_sec,
            )
        else:
            self._open_bgr_fetch_prefilled(start_epoch=entry.event.origin_epoch - 30.0, duration_sec=600.0)

    def _on_live_show_on_map(self, event_id: str) -> None:
        self.action_open_world_map()
        wmap = self._find_open_dialog(WorldMapDialog)
        if wmap is not None:
            wmap.focus_event_by_id(event_id)

    def action_open_world_map(self) -> None:
        existing = self._find_open_dialog(WorldMapDialog)
        if existing is not None:
            existing.refresh_data(preserve_view=True)
            self._raise_nonmodal(existing)
            return
        dlg = WorldMapDialog(
            self.engine,
            config=self.config,
            live_coordinator=self.live_coordinator,
            parent=self,
        )
        dlg.selectStationTracesRequested.connect(self._on_world_map_select_traces)
        dlg.openBGRFetchRequested.connect(self._on_world_map_open_bgr)
        dlg.adoptEventParamsRequested.connect(self._on_world_map_adopt_event)
        dlg.loadLiveEventRequested.connect(lambda eid: self.live_coordinator.request_load_event(eid, auto_triggered=False))
        dlg.openLiveMonitorRequested.connect(self.action_open_live_event_dialog)
        self._show_nonmodal(dlg)

    def _sync_open_world_map(self) -> None:
        dlg = self._find_open_dialog(WorldMapDialog)
        if dlg is not None:
            dlg.refresh_data(preserve_view=True)

    def _on_world_map_select_traces(self, station_code: str) -> None:
        target = (station_code or "").strip().upper()
        matched = 0
        for tr in self.engine.traces:
            hit = (tr.station or "").strip().upper() == target
            tr.selected = hit
            if hit:
                matched += 1
        self.main_canvas.invalidate_cache()
        self.status_bar.showMessage(f"Selected {matched} trace(s) for station {target}.", 4000)

    def _on_world_map_open_bgr(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._open_bgr_fetch_prefilled(
            stations=payload.get("stations") or [],
            start_epoch=payload.get("start_epoch"),
            duration_sec=payload.get("duration_sec"),
        )

    def _on_world_map_adopt_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        pset = self.engine.param_sets[self.engine.active_set_index]
        pset.lat = float(payload.get("lat", pset.lat))
        pset.lon = float(payload.get("lon", pset.lon))
        pset.depth = float(payload.get("depth", pset.depth))
        if payload.get("origin"):
            pset.origin = str(payload["origin"])
        pset.distance = float(payload.get("distance", pset.distance))
        pset.azimuth = float(payload.get("azimuth", pset.azimuth))
        if payload.get("region_name"):
            pset.region_name = str(payload["region_name"])
        mag = float(payload.get("magnitude", 0.0) or 0.0)
        if mag > 0.0:
            pset.mag_mb = mag
        if payload.get("source"):
            pset.source = str(payload["source"])
        self.param_panel.refresh_from_model()
        self._sync_open_world_map()
        self.status_bar.showMessage(
            f"Adopted event into Parameter Box: {pset.region_name or 'Unknown'} (Δ={pset.distance:.1f}°, BAZ={pset.azimuth:.1f}°)",
            5000,
        )

    def _on_live_event_ready(self, event_id: str) -> None:
        entry = self.live_coordinator.entries.get(event_id)
        if entry is None:
            return
        self._pending_ready_event_id = event_id
        n_tr = len(entry.fetched_traces)
        label = f"Load Live M{entry.event.magnitude:.1f} {entry.evaluation.region_display[:28]} ({n_tr} tr)"
        self.live_notify_btn.setText(label)
        register_tutorial_help(
            self.live_notify_btn,
            f"Load Live {entry.event.event_id}",
            "live_notify",
            "",
            f"Click to load {entry.event.event_id} waveforms into SHM",
        )
        apply_tutorial_tooltips(self.live_notify_btn, self.tutorial_mode)
        self.live_notify_btn.setVisible(True)

    def _on_live_notify_clicked(self) -> None:
        eid = self._pending_ready_event_id
        self._pending_ready_event_id = None
        self.live_notify_btn.setVisible(False)
        if eid:
            self.live_coordinator.request_load_event(eid, auto_triggered=False)

    def _on_live_load_requested(self, event_id: str, auto_triggered: bool) -> None:
        entry = self.live_coordinator.entries.get(event_id)
        if entry is None or not entry.fetched_traces:
            return
        if self._pending_ready_event_id == event_id:
            self._pending_ready_event_id = None
            self.live_notify_btn.setVisible(False)

        traces = [tr.copy() for tr in entry.fetched_traces]
        mode_tag = "auto-loaded" if auto_triggered else "loaded"
        msg = (
            f"Live Event {entry.event.event_id} {mode_tag}: M{entry.event.magnitude:.1f} "
            f"{entry.evaluation.region_display} ({len(traces)} traces, Δ={entry.evaluation.dist_deg:.1f}°)"
        )
        self._load_waveforms(traces, append_mode=False, undo_label=f"Live M{entry.event.magnitude:.1f}", log_msg=msg)

        # Populate Analysis Parameter Set cleanly inside active slot
        pset = build_analysis_param_set(entry, self.config, self.live_coordinator.live_config)
        self.engine.param_sets[self.engine.active_set_index] = pset

        lcfg = self.live_coordinator.live_config
        filt_expr = (lcfg.auto_filter_expr or "").strip()
        if filt_expr:
            if filt_expr.upper().startswith(("BUTTER", "FILTER", "SIMULATE")):
                self.engine.execute_line(filt_expr)
            else:
                self.engine.execute_line(f"FILTER {filt_expr}")

        if lcfg.auto_theo_phases and pset.distance > 0.0:
            self.engine.execute_line(f"THEO {pset.distance} {pset.depth}")

        self.refresh_views(reset_time=True)

    def action_read_evt(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(self, "Read .evt Event File", self.config.shm_root, "Event Files (*.evt);;All Files (*)")
        if fpath:
            self.engine.execute_line(f"READEVT {fpath}")
            self._register_unfiltered_traces(self.engine.traces)
            self.refresh_views()

    def action_open_filter_dialog(self) -> None:
        dt = self.engine.traces[0].delta if self.engine.traces else 0.05
        dlg = FilterDialog(dt=dt, config=self.config, parent=self)
        if dlg.exec():
            opts = dlg.get_filter_options()
            if opts["filter_expr"] == "NONE":
                self.action_unfilter()
                return
            self._push_undo("Filter")
            self.engine.apply_filter_to_traces(
                opts["filter_expr"],
                mode=opts["mode"],
                use_lpc=opts["use_lpc"],
                demean_first=opts["demean_first"],
                visible_only=True,
                autocut=opts["autocut"],
                zero_phase=opts["zero_phase"],
            )
            self.refresh_views()
            ac_note = " + autocut" if opts["autocut"] else ""
            zp_note = " + zero-phase" if opts["zero_phase"] else ""
            self._append_console_log(
                f"Applied filter '{opts['filter_expr']}' ({opts['mode']}{ac_note}{zp_note})."
            )

    def _restore_pre_overlay_waveform(self, tr: Trace) -> None:
        """Restore raw or pre-overlay globally filtered waveform when clearing an overlay filter."""
        uid = tr.extra_headers.get("_raw_uid")
        raw_tr = self._unfiltered_by_uid.get(uid) if uid else None
        if (
            raw_tr is None
            or raw_tr.length != tr.length
            or abs(raw_tr.delta - tr.delta) > 1e-6
            or raw_tr.station != tr.station
            or raw_tr.channel3 != tr.channel3
        ):
            return
        if tr.filter_name == "NONE":
            tr.data = raw_tr.data.copy()
        else:
            try:
                rebuilt = filter_trace(
                    raw_tr.copy(),
                    tr.filter_name,
                    config=self.config,
                )
                if rebuilt.length == tr.length:
                    tr.data = rebuilt.data.copy()
            except Exception:
                pass

    def apply_overlay_filter(
        self,
        filter_expr: str,
        color: str = "red",
        trace_indices: Optional[Sequence[int]] = None,
        mode: str = "AUTO",
        autocut: bool = False,
        zero_phase: bool = False,
        clear: bool = False,
    ) -> int:
        """Apply or clear a trace-specific secondary overlay filter (M-24) without modifying global filter_name."""
        if trace_indices is None:
            idxs = [i for i, t in enumerate(self.engine.traces) if t.selected and not t.hidden]
        else:
            idxs = [i for i in trace_indices if 0 <= i < len(self.engine.traces) and not self.engine.traces[i].hidden]
        if not idxs:
            return 0

        self._push_undo("Clear overlay filter" if clear else "Overlay filter")
        if clear or filter_expr.strip().upper() in ("NONE", "CLEAR", "OFF", ""):
            for i in idxs:
                tr = self.engine.traces[i]
                tr.overlay_filter_name = "NONE"
                tr.overlay_filter_color = ""
                self._restore_pre_overlay_waveform(tr)
            self.refresh_views()
            self._append_console_log(f"Cleared overlay filter on {len(idxs)} trace(s).")
            return len(idxs)

        eff_color = (color or "red").strip() or "red"
        for i in idxs:
            tr = self.engine.traces[i]
            prev_global = tr.filter_name
            filtered = filter_trace(
                tr,
                filter_expr,
                mode=mode,
                config=self.config,
                autocut=autocut,
                zero_phase=zero_phase,
            )
            filtered.filter_name = prev_global
            filtered.overlay_filter_name = filter_expr
            filtered.overlay_filter_color = eff_color
            filtered.filter_history = list(tr.filter_history) + [filter_expr]
            self.engine.traces[i] = filtered

        self.refresh_views()
        self._append_console_log(
            f"Applied overlay filter '{filter_expr}' ({eff_color}) to {len(idxs)} trace(s)."
        )
        return len(idxs)

    def action_overlay_filter_dialog(self, target_indices: Optional[Sequence[int]] = None) -> None:
        if target_indices is None:
            idxs = [i for i, t in enumerate(self.engine.traces) if t.selected and not t.hidden]
        else:
            idxs = [i for i in target_indices if 0 <= i < len(self.engine.traces)]
        if not idxs:
            self.status_bar.showMessage("Select at least one trace first (click left margin).", 5000)
            return

        dt = self.engine.traces[idxs[0]].delta if self.engine.traces[idxs[0]].delta > 0 else 0.05
        dlg = FilterDialog(dt=dt, config=self.config, parent=self)
        dlg.setWindowTitle("Trace Overlay Filter (colour_addfil)")
        if not dlg.exec():
            return
        opts = dlg.get_filter_options()
        if opts["filter_expr"] == "NONE":
            self.apply_overlay_filter("NONE", trace_indices=idxs, clear=True)
            return

        color, ok = QInputDialog.getItem(
            self,
            "Overlay Filter Color",
            "Badge && Overlay Color:",
            ["red", "green", "blue"],
            0,
            False,
        )
        if not ok:
            return
        self.apply_overlay_filter(
            opts["filter_expr"],
            color=color,
            trace_indices=idxs,
            mode=opts["mode"],
            autocut=opts["autocut"],
            zero_phase=opts["zero_phase"],
        )

    def action_clear_overlay_filters(self) -> None:
        idxs = [i for i, t in enumerate(self.engine.traces) if t.selected and not t.hidden]
        if not idxs:
            idxs = [
                i
                for i, t in enumerate(self.engine.traces)
                if not t.hidden and getattr(t, "overlay_filter_name", "NONE") not in ("", "NONE")
            ]
        if not idxs:
            self.status_bar.showMessage("No trace overlay filters to clear.", 3000)
            return
        for i in idxs:
            tr = self.engine.traces[i]
            tr.overlay_filter_name = "NONE"
            tr.overlay_filter_color = ""
            self._restore_pre_overlay_waveform(tr)
        self.refresh_views()
        self._append_console_log(f"Cleared overlay filter on {len(idxs)} trace(s).")

    def _on_overlay_filter_requested(self, trace_idx: int) -> None:
        if 0 <= trace_idx < len(self.engine.traces):
            self.action_overlay_filter_dialog(target_indices=[trace_idx])

    def _on_clear_overlay_filter_requested(self, trace_idx: int) -> None:
        if 0 <= trace_idx < len(self.engine.traces):
            tr = self.engine.traces[trace_idx]
            tr.overlay_filter_name = "NONE"
            tr.overlay_filter_color = ""
            self._restore_pre_overlay_waveform(tr)
            self.refresh_views()
            self._append_console_log(f"Cleared overlay filter on {tr.station} {tr.channel3}.")

    def action_unfilter(self) -> None:
        restored = 0
        for cur_tr in self.engine.traces:
            uid = cur_tr.extra_headers.get("_raw_uid")
            raw_tr = self._unfiltered_by_uid.get(uid) if uid else None
            if (
                raw_tr is not None
                and raw_tr.length == cur_tr.length
                and abs(raw_tr.delta - cur_tr.delta) <= 1e-6
                and raw_tr.station == cur_tr.station
                and raw_tr.channel3 == cur_tr.channel3
            ):
                cur_tr.data = raw_tr.data.copy()
                cur_tr.delta = raw_tr.delta
                restored += 1
            cur_tr.filter_name = "NONE"
        self.refresh_views()
        if restored:
            self._append_console_log(f"Restored unfiltered raw waveforms on {restored} trace(s).")
        else:
            self._append_console_log("Cleared filter tags on workspace traces.")

    def action_export_plot(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export Waveform Canvas Screenshot", "shm_waveforms.png", "PNG Images (*.png)")
        if fpath:
            pix = self.main_canvas.grab()
            pix.save(fpath)
            self._append_console_log(f"Saved waveform plot image to {fpath}")

    def action_auto_pick(self) -> None:
        self.engine.execute_line("PICK ALL")
        self.refresh_views()

    def action_corr_pick(self) -> None:
        self.engine.execute_line(f"CORRPICK {self.phase_panel.active_phase}")
        self.refresh_views()

    def action_theoretical_phases(self) -> None:
        p = self.engine.current_params
        dist = p.distance if p.distance > 0 else 65.0
        depth = p.depth if p.depth > 0 else 33.0
        self.engine.execute_line(f"THEO {dist} {depth}")
        self.refresh_views()

    def action_delete_active_pick(self) -> None:
        ph = self.phase_panel.active_phase
        count = 0
        for tr in self.engine.traces:
            before = len(tr.picks)
            tr.picks = [p for p in tr.picks if p.name != ph]
            count += before - len(tr.picks)
        self.refresh_views()
        self._append_console_log(f"Deleted {count} pick(s) for phase '{ph}'.")

    def action_demean(self) -> None:
        self.engine.execute_line("DEMEAN ALL")
        self.refresh_views()

    def action_detrend(self) -> None:
        self.engine.execute_line("TREND ALL")
        self.refresh_views()

    def action_despike(self) -> None:
        self.engine.execute_line("DESPIKE ALL 5.0")
        self.refresh_views()

    def action_integrate(self) -> None:
        self.engine.execute_line("INT ALL")
        self.refresh_views()

    def action_differentiate(self) -> None:
        self.engine.execute_line("DERIVE ALL 1")
        self.refresh_views()

    def action_hilbert(self) -> None:
        self.engine.execute_line("HILBERT ALL")
        self.refresh_views()

    def action_envelope(self) -> None:
        self.engine.execute_line("ENVELOPE ALL")
        self.refresh_views()

    def action_resample(self) -> None:
        val, ok = QInputDialog.getDouble(self, "Resample Traces", "New Sample Interval dt (seconds):", 0.025, 0.001, 10.0, 4)
        if ok:
            self._push_undo("Resample")
            self.engine.execute_line(f"RESAMPLE {val}")
            self._invalidate_stale_unfiltered()
            self.refresh_views()

    def action_decimate(self) -> None:
        val, ok = QInputDialog.getInt(self, "Decimate Traces", "Decimation Factor (integer):", 2, 1, 100)
        if ok:
            self._push_undo("Decimate")
            self.engine.execute_line(f"DECIMATE {val}")
            self._invalidate_stale_unfiltered()
            self.refresh_views()

    def action_flip_traces(self) -> None:
        self.engine.execute_line("FLIP SEL")
        self.refresh_views()

    def action_rotate_2d(self) -> None:
        az = self.engine.current_params.azimuth
        self.engine.execute_line(f"ROT2 {az}")
        self.refresh_views()

    def action_rotate_3d(self) -> None:
        p = self.engine.current_params
        if p.incidence > 0:
            self.engine.execute_line(f"ROT3 {p.azimuth} {p.incidence}")
        else:
            self.engine.execute_line(f"ROT3 {p.azimuth}")
        self.refresh_views()

    def action_zoom_to_magnifier(self) -> None:
        self.main_canvas.push_view_history()
        self.main_canvas.view_t_lo = self.main_canvas.mag_t_lo
        self.main_canvas.view_t_hi = self.main_canvas.mag_t_hi
        self.engine.view_window = (self.main_canvas.mag_t_lo, self.main_canvas.mag_t_hi)
        self.main_canvas.invalidate_cache()

    def action_cut_to_magnifier(self) -> None:
        lo, hi = self.main_canvas.mag_t_lo, self.main_canvas.mag_t_hi
        ref_ep = self.main_canvas.reference_epoch()
        for i, tr in enumerate(self.engine.traces):
            self.engine.traces[i] = cut_trace(tr, lo, hi, relative_to_epoch=ref_ep)
        self._invalidate_stale_unfiltered()
        self.refresh_views(reset_time=True)

    def action_reset_windows(self) -> None:
        self.main_canvas.push_view_history()
        self.engine.execute_line("TRESET")
        self.main_canvas.view_t_lo = None
        self.main_canvas.view_t_hi = None
        self.refresh_views(reset_time=True)

    def action_form_beam(self) -> None:
        """Toggle: add a beam trace; run again to remove it."""
        if any(t.station == "BEAM" for t in self.engine.traces):
            self.engine.traces = [t for t in self.engine.traces if t.station != "BEAM"]
            self.refresh_views()
            self._append_console_log("BEAM trace removed.")
            return
        p = self.engine.current_params
        az = p.azimuth if p.azimuth != 0 else 45.0
        slo = p.slowness if p.slowness > 0 else 6.5
        self.engine.execute_line(f"BEAM {az} {slo} CENTRE")
        self.refresh_views()

    @staticmethod
    def _offset_key(tr: Trace) -> Tuple[str, str, float]:
        return (tr.station, tr.channel3, tr.start_epoch)

    def action_align_phase(self) -> None:
        """Toggle: align on the active phase and add a sum trace; run again to restore and remove it."""
        if any(t.station == "ALIGN" for t in self.engine.traces):
            self.engine.traces = [t for t in self.engine.traces if t.station != "ALIGN"]
            for tr in self.engine.traces:
                tr.display_offset = self._align_saved_offsets.get(self._offset_key(tr), 0.0)
            self._align_saved_offsets = {}
            self.refresh_views(reset_time=True)
            self._append_console_log("ALIGN removed: original trace positions restored.")
            return
        self._align_saved_offsets = {self._offset_key(t): t.display_offset for t in self.engine.traces}
        self.engine.execute_line(f"ALIGN {self.phase_panel.active_phase}")
        if not any(t.station == "ALIGN" for t in self.engine.traces):
            self._align_saved_offsets = {}
        self.refresh_views(reset_time=True)

    def action_sum_traces(self) -> None:
        self.engine.execute_line("SUM ALL")
        self.refresh_views()

    def action_pws_traces(self) -> None:
        self.engine.execute_line("PWS ALL 2.0")
        self.refresh_views()

    def action_open_fk_dialog(self) -> None:
        vis = self.engine.visible_traces()
        dlg = FKAnalysisDialog(
            vis,
            t_lo=self.main_canvas.mag_t_lo,
            t_hi=self.main_canvas.mag_t_hi,
            config=self.config,
            parent=self,
        )
        dlg.adoptParameters.connect(self._on_adopt_az_slo)
        dlg.adoptAndLocate.connect(self._on_adopt_az_slo_and_locate)
        dlg.fitDiagnosticReady.connect(self.fit_bridge.fitDiagnosticAvailable.emit)
        self._show_nonmodal(dlg)

    def _on_adopt_az_slo(self, az: float, slo: float) -> None:
        p = self.engine.current_params
        if az is not None:
            p.azimuth = az
        if slo is not None:
            p.slowness = slo
        az_fmt = f"{az:.1f}" if az is not None else "N/A"
        slo_fmt = f"{slo:.2f}" if slo is not None else "N/A"
        had_beam = any(t.station == "BEAM" for t in self.engine.traces)
        if had_beam:
            source_traces = [t for t in self.engine.traces if t.station != "BEAM"]
            self.engine.traces = source_traces
            if source_traces and az is not None and slo is not None:
                self.engine.execute_line(f"BEAM {az} {slo} CENTRE")
                self._append_console_log(f"Re-formed BEAM with updated slowness: BAZ={az_fmt} deg, Slowness={slo_fmt} s/deg.")
        self.refresh_views()
        self._append_console_log(f"Adopted FK parameters: BAZ={az_fmt} deg, Slowness={slo_fmt} s/deg.")
        self.status_bar.showMessage(f"Adopted FK BAZ={az_fmt}°, Slow={slo_fmt} s/deg", 5000)

    def _on_adopt_az_slo_and_locate(self, az: float, slo: float) -> None:
        p = self.engine.current_params
        if az is not None:
            p.azimuth = az
        if slo is not None:
            p.slowness = slo
        az_fmt = f"{az:.1f}" if az is not None else "N/A"
        slo_fmt = f"{slo:.2f}" if slo is not None else "N/A"
        had_beam = any(t.station == "BEAM" for t in self.engine.traces)
        if had_beam:
            source_traces = [t for t in self.engine.traces if t.station != "BEAM"]
            self.engine.traces = source_traces
            if source_traces and az is not None and slo is not None:
                self.engine.execute_line(f"BEAM {az} {slo} CENTRE")
                self._append_console_log(f"Re-formed BEAM with updated slowness: BAZ={az_fmt} deg, Slowness={slo_fmt} s/deg.")
        self.engine.execute_line(f"LOCATE /SLOWNESS {self.phase_panel.active_phase}")
        self.refresh_views()

    def action_open_vespa_dialog(self) -> None:
        vis = self.engine.visible_traces()
        dlg = VespagramDialog(
            vis,
            azimuth_deg=self.engine.current_params.azimuth or 45.0,
            config=self.config,
            parent=self,
        )
        dlg.appendVespaTraces.connect(self._on_append_vespa_traces)
        self._show_nonmodal(dlg)

    def _on_append_vespa_traces(self, v_trs: List[Trace]) -> None:
        self.engine.traces.extend(v_trs)
        self.refresh_views()

    def action_locate_plane_wave(self) -> None:
        ph = self.phase_panel.active_phase
        # If fewer than 3 picks exist, auto-pick first so user gets immediate result
        vis = self.engine.visible_traces()
        picked = [t for t in vis if t.get_pick(ph) is not None]
        if len(picked) < 3 and len(vis) >= 3:
            msg = f"Fewer than 3 '{ph}' picks found: running the automatic picker first."
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 6000)
            self.engine.execute_line("PICK ALL")
        self.engine.execute_line(f"LOCATE {ph}")
        self.refresh_views()

    def action_locate_by_slowness(self) -> None:
        self.engine.execute_line(f"LOCATE /SLOWNESS {self.phase_panel.active_phase}")
        self.refresh_views()

    def action_locate_by_sp_diff(self) -> None:
        res = self.engine.locate_by_sp_diff()
        if res is not None:
            self.refresh_views()
            self._append_console_log(
                f"S-P Differential Location ({res['station']}): dt={res['dt_sp']:.2f}s -> Dist={res['distance']:.2f} deg"
            )
            return
        self._append_console_log("Pick both P (or Pg/Pn) and S (or Sg/Sn) on a trace to compute S-P distance.")

    def action_fit_depth_phase(self) -> None:
        res = self.engine.fit_depth_phase()
        if res is not None:
            self.refresh_views()
            self._append_console_log(
                f"Depth Phase Fit ({res['phase']}-P = {res['dt_dp']:.2f} s at {res['distance']:.1f} deg): "
                f"Depth = {res['depth']:.1f} +/- {res['d_depth']:.1f} km"
            )
            return
        self._append_console_log("Pick both 'P' and 'pP' (or 'sP') on a trace to invert for focal depth.")

    def action_open_network_loc_dialog(self) -> None:
        all_picks = [pk for tr in self.engine.visible_traces() for pk in tr.picks]
        dlg = NetworkLocationDialog(
            all_picks,
            initial_depth=self.engine.current_params.depth,
            config=self.config,
            parent=self,
        )
        dlg.adoptLocation.connect(self._on_adopt_network_loc)
        if hasattr(dlg, "diagnosticAvailable"):
            dlg.diagnosticAvailable.connect(self.fit_bridge.fitDiagnosticAvailable)
        self._show_nonmodal(dlg)

    def action_open_fit_diagnostic_window(self) -> None:
        """Open or show the dockable Fit Diagnostics Window and compute initial diagnostics."""
        self.fit_dock.show()
        self.fit_dock.raise_()
        all_picks = [pk for tr in self.engine.visible_traces() for pk in tr.picks if pk.onset and pk.station]
        if not all_picks:
            all_picks = [pk for tr in self.engine.traces for pk in tr.picks if pk.onset and pk.station]
        if len(all_picks) >= 3:
            req = FitModificationRequest(fit_type=FitType.HYPOCENTER_GEIGER)
            self._on_recompute_fit_requested(req)

    def _on_fit_diagnostic_received(self, diag: Any) -> None:
        self._last_fit_diagnostic = diag
        self.engine.last_fit_diagnostic = diag

    def _cancel_active_recompute_workers(self) -> int:
        alive: List[_RecomputeFitWorker] = []
        for w in getattr(self, "_recompute_workers", []):
            if w is not None and not sip.isdeleted(w) and w.isRunning():
                w.cancel()
                alive.append(w)
        self._recompute_workers = alive
        self._recompute_seq = getattr(self, "_recompute_seq", 0) + 1
        return self._recompute_seq

    def _launch_recompute_worker(self, worker: _RecomputeFitWorker) -> _RecomputeFitWorker:
        worker.finishedDiagnostic.connect(self._on_recompute_worker_finished)
        worker.failed.connect(self._on_recompute_worker_failed)
        self._recompute_workers.append(worker)
        worker.start()
        return worker

    def _on_recompute_worker_finished(self, seq_id: int, diag: Any, log_msg: str) -> None:
        if seq_id != self._recompute_seq:
            return
        self.fit_bridge.fitDiagnosticAvailable.emit(diag)
        if log_msg:
            self._append_console_log(log_msg)

    def _on_recompute_worker_failed(self, seq_id: int, err_msg: str) -> None:
        if seq_id != self._recompute_seq:
            return
        if err_msg:
            self._append_console_log(err_msg)

    def _on_recompute_fit_requested(self, req: Any) -> None:
        """Re-compute diagnostics / re-invert according to req.fit_type with exclusions and weights."""
        fit_type = getattr(req, "fit_type", FitType.HYPOCENTER_GEIGER)
        excluded = list(getattr(req, "excluded_identifiers", None) or [])
        excluded_upper = {str(x).split(".")[0].strip().upper() for x in excluded if str(x).strip()}
        weight_overrides = getattr(req, "weight_overrides", None)
        seq_id = self._cancel_active_recompute_workers()

        if fit_type in (
            FitType.HYPOCENTER_GEIGER,
            FitType.ORIGIN_TIME,
            FitType.TTFIT_SIMPLEX,
            FitType.SP_DIFF_DEPTH,
        ):
            from shm_app.seismology.array_loc import locate_network_geiger

            all_picks = [pk for tr in self.engine.visible_traces() for pk in tr.picks if pk.onset and pk.station]
            if not all_picks:
                all_picks = [pk for tr in self.engine.traces for pk in tr.picks if pk.onset and pk.station]
            if len(all_picks) < 3:
                self._append_console_log("Cannot re-invert: need at least 3 phase picks across network.")
                return

            try:
                param, _station_residuals, diag = locate_network_geiger(
                    all_picks,
                    depth_km=self.engine.current_params.depth,
                    free_depth=(self.engine.current_params.depth_type == 3),
                    config=self.config,
                    ttt=self.engine.ttt,
                    excluded_stations=excluded or None,
                    weight_overrides=weight_overrides,
                    return_diagnostics=True,
                )
                self._on_adopt_network_loc(param)
                self.fit_bridge.fitDiagnosticAvailable.emit(diag)
                self._append_console_log(
                    f"Re-inverted Geiger hypocenter: Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, "
                    f"Depth={param.depth:.1f} km, RMSE={diag.rmse:.3f} s"
                )
            except Exception as exc:
                self._append_console_log(f"Re-fit error: {exc}")
            return

        if fit_type in (FitType.FK_SEMBLANCE, FitType.PLANE_WAVE):
            vis = [
                tr.copy()
                for tr in self.engine.visible_traces()
                if tr.station.strip().upper() not in excluded_upper
            ]
            if len(vis) < 2:
                self._append_console_log("Cannot recompute FK/Array fit: need at least 2 active traces.")
                return
            worker = _RecomputeFitWorker(
                seq_id,
                fit_type,
                traces=vis,
                t_lo=self.main_canvas.mag_t_lo,
                t_hi=self.main_canvas.mag_t_hi,
                config=self.config,
                calib_table=getattr(self.engine, "calib_table", None),
                parent=self,
            )
            self._launch_recompute_worker(worker)
            return

        if fit_type == FitType.BRUNE_SPECTRUM:
            vis = self.engine.visible_traces()
            if not vis:
                self._append_console_log("Cannot recompute Brune spectrum: no traces loaded.")
                return
            idx = max(0, min(self.main_canvas.mag_trace_idx, len(self.engine.traces) - 1))
            tr = self.engine.traces[idx].copy()
            lo, hi = self._spectrum_window_for(tr)
            if hi - lo < 8 * tr.delta:
                lo, hi = 0.0, tr.duration
            dist_deg = tr.distance if tr.distance > 0 else (self.engine.current_params.distance or 1.0)
            worker = _RecomputeFitWorker(
                seq_id,
                fit_type,
                trace=tr,
                t_lo=lo,
                t_hi=hi,
                dist_deg=float(dist_deg),
                parent=self,
            )
            self._launch_recompute_worker(worker)
            return

        if fit_type == FitType.POLARIZATION_3C:
            try:
                from shm_app.processing.polarization import analyze_polarization_diagnostic

                prefer_sta = None
                if 0 <= self.main_canvas.mag_trace_idx < len(self.engine.traces):
                    prefer_sta = self.engine.traces[self.main_canvas.mag_trace_idx].station
                tz, tn, te = self.engine._find_zne_triplet(prefer_station=prefer_sta)
                lo, hi = self._spectrum_window_for(tz)
                if hi - lo <= 2.0 * max(1e-6, tz.delta):
                    lo, hi = 0.0, tz.duration
                diag = analyze_polarization_diagnostic(tz, tn, te, t_lo=lo, t_hi=hi)
                self.fit_bridge.fitDiagnosticAvailable.emit(diag)
            except Exception as exc:
                self._append_console_log(f"Polarization re-fit error: {exc}")
            return

        # Other domains (RESTITUTION_FIT, RECEIVER_FUNC, SURFACE_DISPERSION): re-emit cached result if matching
        last_diag = self._last_fit_diagnostic
        if last_diag is not None and getattr(last_diag, "fit_type", None) == fit_type:
            self.fit_bridge.fitDiagnosticAvailable.emit(last_diag)
        self._append_console_log(f"Re-fit handled for domain {getattr(fit_type, 'value', fit_type)}.")

    def _on_highlight_fit_trace(self, station: str) -> None:
        """Highlight and select matching station trace(s) in main canvas when clicked in diagnostic table."""
        st_upper = (station or "").strip().upper()
        if not st_upper:
            return
        selected_indices = set()
        first_idx: Optional[int] = None
        for idx, tr in enumerate(self.engine.traces):
            if tr.hidden:
                continue
            matched = tr.station.strip().upper() == st_upper
            tr.selected = matched
            if matched:
                selected_indices.add(idx)
                if first_idx is None:
                    first_idx = idx
        self.engine.selected_trace_indices = selected_indices
        if first_idx is not None:
            self.main_canvas.mag_trace_idx = first_idx
        self.main_canvas.invalidate_cache()
        self.main_canvas.update()
        self.refresh_views()

    def _on_exclude_fit_station(self, station: str, exclude: bool) -> None:
        """Handle explicit station exclusion toggle from external plugins."""
        req = FitModificationRequest(
            fit_type=FitType.HYPOCENTER_GEIGER,
            excluded_identifiers=[station] if exclude else [],
        )
        self._on_recompute_fit_requested(req)

    def _on_adopt_network_loc(self, param: AnalysisParamSet) -> None:
        self.engine.param_sets[self.engine.active_set_index] = param
        orig_ep = parse_sh_time(param.origin) if param.origin else 0.0

        for tr in self.engine.traces:
            st = self.config.get_station(tr.station)
            if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                dist_i, _, _ = sphere_diff(st.lat, st.lon, param.lat, param.lon)
            else:
                dist_i = param.distance
            tr.distance = dist_i

            if orig_ep > 0:
                arrs = self.engine.ttt.theoretical_arrivals(dist_i, param.depth, orig_ep)
                tr.theo_picks = arrs
                for pk in tr.picks:
                    if param.residuals and f"{tr.station}:{pk.name}" in param.residuals:
                        pk.resid = param.residuals[f"{tr.station}:{pk.name}"]
                    elif pk.name in arrs:
                        pk.resid = pk.onset_epoch - arrs[pk.name]

        self.refresh_views()
        self._append_console_log(
            f"Adopted location: {param.lat:+.3f} N, {param.lon:+.3f} E, Depth {param.depth:.1f} km, "
            f"Origin {param.origin}. Updated station distances, theoretical arrivals, and residuals."
        )


    def _copy_param_set(self, src: int, dst: int) -> None:
        self.engine.param_sets[dst] = copy.deepcopy(self.engine.param_sets[src])
        self.refresh_views()

    def action_clear_params(self) -> None:
        self.engine.param_sets[self.engine.active_set_index] = AnalysisParamSet()
        for tr in self.engine.traces:
            tr.picks.clear()
            tr.theo_picks.clear()
        self.refresh_views()

    def action_open_event_summary(self) -> None:
        all_picks = [pk for tr in self.engine.traces for pk in tr.picks]
        dlg = EventSummaryDialog(self.engine.current_params, all_picks, config=self.config, parent=self)
        dlg.openWorldMapRequested.connect(self._on_event_summary_open_world_map)
        self._show_nonmodal(dlg)

    def _on_event_summary_open_world_map(self) -> None:
        self.action_open_world_map()
        wmap = self._find_open_dialog(WorldMapDialog)
        if wmap is not None:
            wmap.set_view_preset("focus_event")

    def action_compute_magnitudes(self, prefer: str = "ALL") -> None:
        self.engine.execute_line(f"MAG {self.phase_panel.active_phase}")
        self.refresh_views()

    def action_select_all(self) -> None:
        for tr in self.engine.traces:
            if not tr.hidden:
                tr.selected = True
        self.main_canvas.invalidate_cache()

    def action_deselect_all(self) -> None:
        for tr in self.engine.traces:
            tr.selected = False
        self.main_canvas.invalidate_cache()

    def action_delete_selected(self) -> None:
        sel = [t for t in self.engine.traces if t.selected]
        if sel:
            self.engine.traces = [t for t in self.engine.traces if not t.selected]
            self.refresh_views()

    def action_keep_selected(self) -> None:
        sel = [t for t in self.engine.traces if t.selected]
        if sel:
            for t in sel:
                t.selected = False
            self.engine.traces = sel
            self.refresh_views()

    def action_sort_by_distance(self) -> None:
        self.engine.sort_traces(by="distance")
        self.refresh_views()

    def action_sort_by_station(self) -> None:
        self.engine.sort_traces(by="station")
        self.refresh_views()

    def action_save_evt(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Save Event Parameters (.evt)", "shm_event.evt", "Event Files (*.evt)")
        if fpath:
            self.engine.execute_line(f"WRITEEVT {fpath}")

    def action_save_evt_and_clear(self) -> None:
        self.action_save_evt()
        self.action_clear_params()

    def action_export_gse2(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export GSE2 File", "export.GSE", "GSE Files (*.GSE *.gse)")
        if fpath:
            self.engine.execute_line(f"WRITEGSE {fpath}")

    def action_export_mseed(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export Mini-SEED File", "export.mseed", "Mini-SEED Files (*.mseed *.bhz)")
        if fpath:
            self.engine.execute_line(f"WRITESEED {fpath}")

    def action_export_qfile(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export SH Q-File", "export.QHD", "Q-Files (*.QHD)")
        if fpath:
            self.engine.execute_line(f"WRITE {fpath}")

    def action_export_ascii(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export ASCII Waveforms", "export.asc", "ASCII Files (*.asc *.txt)")
        if fpath:
            self.engine.execute_line(f"WRITEA {fpath}")

    def action_open_pmotion_dialog(self) -> None:
        prefer_sta = None
        if 0 <= self.main_canvas.mag_trace_idx < len(self.engine.traces):
            prefer_sta = self.engine.traces[self.main_canvas.mag_trace_idx].station
        try:
            tz, tn, te = self.engine._find_zne_triplet(prefer_station=prefer_sta)
        except ValueError:
            vis = self.engine.visible_traces()
            if len(vis) >= 3:
                tz, tn, te = vis[0], vis[1], vis[2]
            else:
                QMessageBox.information(self, "Particle Motion", "Need at least 3 traces (Z, N, E) for 3D particle motion.")
                return
        self._last_pmotion_station = tz.station
        lo, hi = self._spectrum_window_for(tz)
        if hi - lo <= 2.0 * max(1e-6, tz.delta):
            lo, hi = 0.0, tz.duration
        dlg = ParticleMotionDialog(tz, tn, te, t_lo=lo, t_hi=hi, parent=self)
        dlg.adoptAngles.connect(self._on_adopt_pmotion_angles)
        dlg.fitDiagnosticReady.connect(self.fit_bridge.fitDiagnosticAvailable.emit)
        dlg.emit_diagnostic()
        self._show_nonmodal(dlg)

    def _on_adopt_pmotion_angles(self, az: float, inc: float) -> None:
        self.engine.current_params.azimuth = az
        self.engine.current_params.incidence = inc
        target_sta = (self._last_pmotion_station or "").strip().upper()
        for tr in self.engine.traces:
            if target_sta:
                if tr.station.strip().upper() == target_sta and tr.comp in ("Z", "N", "E"):
                    tr.inci = inc
            elif tr.comp in ("Z", "N", "E"):
                tr.inci = inc
        self.refresh_views()
        self._append_console_log(f"Adopted Particle Motion angles: BAZ={az:.1f} deg, Incidence={inc:.1f} deg.")
        self.status_bar.showMessage(f"Adopted BAZ={az:.1f}°, Inc={inc:.1f}°", 5000)

    def action_apply_polfil_interactive(self) -> None:
        try:
            self.engine.execute_line("POLFIL 1.0 2.0 1")
            self.refresh_views()
        except Exception as exc:
            QMessageBox.warning(self, "Polarization Filter", str(exc))

    # ------------------------------------------------------------------
    # Phase 4: StationXML, QuakeML, GeoJSON, Vector Bulletin, Phase ID,
    # Surface Wave Scan, Focal Mechanism, SeedLink, and Session Recovery
    # ------------------------------------------------------------------

    def action_import_stationxml(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = QFileDialog.getOpenFileName(
                self,
                "Import FDSN StationXML",
                self.config.shm_root,
                "StationXML Files (*.xml *.stationxml);;All Files (*)",
            )
        if not fpath:
            return
        try:
            inv = self.config.load_stationxml(fpath)
            self.engine.last_stationxml_inventory = inv
            for tr in self.engine.traces:
                ch_meta = self.config.get_channel_metadata(
                    tr.network, tr.station, tr.locid, tr.channel3, tr.start_epoch
                )
                if ch_meta is not None:
                    if ch_meta.latitude != 0.0 or ch_meta.longitude != 0.0:
                        tr.lat = float(ch_meta.latitude)
                        tr.lon = float(ch_meta.longitude)
                    if ch_meta.sensor_description:
                        tr.instype = ch_meta.sensor_description
                    tr.extra_headers["azimuth"] = float(ch_meta.azimuth)
                    tr.extra_headers["dip"] = float(ch_meta.dip)
                    tr.extra_headers["station_lat"] = float(tr.lat)
                    tr.extra_headers["station_lon"] = float(tr.lon)
                    tr.extra_headers["station_elev"] = float(ch_meta.elevation)
                    sxml_calib = self.config._stationxml_calib_nm_per_count(ch_meta)
                    if sxml_calib is not None and sxml_calib > 0:
                        tr.calib = float(sxml_calib)
                else:
                    st = self.config.get_station(tr.station)
                    if st is not None and (st.lat != 0.0 or st.lon != 0.0) and tr.lat == 0.0 and tr.lon == 0.0:
                        tr.lat = st.lat
                        tr.lon = st.lon
            self.refresh_views()
            n_sta = len(inv.stations)
            n_cha = sum(len(s.channels) for s in inv.stations)
            msg = f"Imported StationXML ({n_sta} station(s), {n_cha} channel(s)) from {fpath}."
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"StationXML import error: {exc}")
            QMessageBox.warning(self, "StationXML Import Error", str(exc))

    def action_import_quakeml(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = QFileDialog.getOpenFileName(
                self,
                "Import QuakeML Event",
                self.config.shm_root,
                "QuakeML Files (*.xml *.qml);;All Files (*)",
            )
        if not fpath:
            return
        try:
            self._push_undo("Import QuakeML")
            qml = parse_quakeml(fpath)
            param = qml["params"]
            picks = qml.get("phase_picks", [])
            fms = qml.get("focal_mechanisms", [])
            if fms:
                fm0 = fms[0]
                np1 = fm0.get("nodalPlane1")
                if isinstance(np1, dict):
                    fm_dict: Dict[str, Any] = {
                        "strike": float(np1.get("strike", 0.0)),
                        "dip": float(np1.get("dip", 0.0)),
                        "rake": float(np1.get("rake", 0.0)),
                    }
                    np2 = fm0.get("nodalPlane2")
                    if isinstance(np2, dict):
                        fm_dict["strike2"] = float(np2.get("strike", 0.0))
                        fm_dict["dip2"] = float(np2.get("dip", 0.0))
                        fm_dict["rake2"] = float(np2.get("rake", 0.0))
                    self.engine.last_focal_mechanism = fm_dict
                    self._last_focal_mechanism = fm_dict
                    setattr(param, "focal_mechanism", fm_dict)
                    self.fit_dock.display_focal_mechanism(fm_dict)

            self.engine.param_sets[self.engine.active_set_index] = param
            for p in picks:
                for tr in self.engine.traces:
                    if tr.station.upper() == p.station.upper() and (not p.comp or tr.comp.upper() == p.comp.upper()):
                        tr.add_or_replace_pick(p)
                        break
                else:
                    for tr in self.engine.traces:
                        if tr.station.upper() == p.station.upper():
                            tr.add_or_replace_pick(p)
                            break
            self.refresh_views()
            msg = (
                f"Imported QuakeML {qml.get('schema_version', '1.2')} from {fpath}: "
                f"Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, {len(picks)} pick(s)."
            )
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"QuakeML import error: {exc}")
            QMessageBox.warning(self, "QuakeML Import Error", str(exc))

    def action_export_quakeml(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = QFileDialog.getSaveFileName(
                self,
                "Export QuakeML 1.2",
                "event.quakeml.xml",
                "QuakeML XML (*.xml *.qml)",
            )
        if not fpath:
            return
        try:
            fm = self._last_focal_mechanism or getattr(self.engine, "last_focal_mechanism", None)
            write_quakeml(
                fpath,
                self.engine.traces,
                self.engine.current_params,
                focal_mechanism=fm,
            )
            msg = f"Exported QuakeML 1.2 to {fpath}"
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"QuakeML export error: {exc}")
            QMessageBox.warning(self, "QuakeML Export Error", str(exc))

    def action_export_geojson(self, fpath: Optional[str] = None) -> None:
        if fpath is None:
            fpath, _ = QFileDialog.getSaveFileName(
                self,
                "Export GeoJSON (RFC 7946)",
                "event.geojson",
                "GeoJSON Files (*.geojson *.json)",
            )
        if not fpath:
            return
        try:
            fm = self._last_focal_mechanism or getattr(self.engine, "last_focal_mechanism", None)
            write_geojson(
                fpath,
                self.engine.traces,
                self.engine.current_params,
                focal_mechanism=fm,
            )
            msg = f"Exported RFC 7946 GeoJSON to {fpath}"
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 5000)
        except Exception as exc:
            self._append_console_log(f"GeoJSON export error: {exc}")
            QMessageBox.warning(self, "GeoJSON Export Error", str(exc))

    def action_export_bulletin(
        self,
        fpath: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> Optional[BulletinWorker]:
        if fpath is None:
            fpath, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export Observatory Bulletin (PDF / SVG)",
                "observatory_bulletin.pdf",
                "Vector PDF (*.pdf);;Vector SVG (*.svg)",
            )
            if fpath and fmt is None and "SVG" in selected_filter and not fpath.lower().endswith(".svg"):
                fpath = fpath + ".svg"
        if not fpath:
            return None

        self._bulletin_workers = [
            w for w in self._bulletin_workers if w is not None and not sip.isdeleted(w) and w.isRunning()
        ]
        fm = self._last_focal_mechanism or getattr(self.engine, "last_focal_mechanism", None)
        fit_res = self._last_fit_diagnostic or getattr(self.engine, "last_fit_diagnostic", None)
        worker = BulletinWorker(
            output_path=fpath,
            traces=[t.copy() for t in self.engine.traces],
            params=copy.deepcopy(self.engine.current_params),
            fit_result=fit_res,
            focal_mech=fm,
            format=fmt,
            config=self.config,
            parent=self,
        )
        worker.finishedReport.connect(self._on_bulletin_finished)
        worker.failed.connect(self._on_bulletin_failed)
        self._bulletin_workers.append(worker)
        worker.start()
        self.status_bar.showMessage(f"Generating observatory bulletin: {fpath}...", 4000)
        return worker

    def _on_bulletin_finished(self, out_path: str) -> None:
        msg = f"Generated vector observatory bulletin: {out_path}"
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)

    def _on_bulletin_failed(self, err: str) -> None:
        msg = f"Observatory bulletin export failed: {err}"
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)

    def action_identify_phases(self, dist_deg: Optional[float] = None) -> None:
        picked_traces = [tr for tr in self.engine.traces if tr.picks]
        if not picked_traces:
            self.status_bar.showMessage("Pick at least one phase before identifying phases.", 4000)
            self._append_console_log("IDENTIFY: no picked traces in workspace.")
            return

        p = self.engine.current_params
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        if dist_deg is None and not has_hypo and p.distance <= 0.0 and not any(tr.distance > 0.0 for tr in picked_traces):
            val, ok = QInputDialog.getDouble(
                self,
                "Identify Phases (IASP91)",
                "Epicentral Distance (degrees):",
                65.0,
                0.1,
                180.0,
                2,
            )
            if not ok:
                return
            dist_deg = float(val)

        try:
            self._push_undo("Identify phases")
            cmd = f"IDENTIFY /DIST={float(dist_deg):.4f}" if dist_deg is not None else "IDENTIFY"
            msg = self.engine.execute_line(cmd)
            self.refresh_views()
            if msg:
                self.status_bar.showMessage(msg.splitlines()[0], 5000)
        except Exception as exc:
            self._append_console_log(f"Phase identification error: {exc}")
            self.status_bar.showMessage(f"Identify error: {exc}", 5000)

    def action_scan_surface_waves(self) -> None:
        if not self.engine.traces:
            self.status_bar.showMessage("Load traces before scanning surface waves.", 4000)
            return
        try:
            self._push_undo("Scan surface waves (18-22s Ms)")
            msg = self.engine.execute_line("SURFMAG")
            self.refresh_views()
            if msg:
                self.status_bar.showMessage(msg, 6000)
        except Exception as exc:
            self._append_console_log(f"Surface wave scan error: {exc}")
            self.status_bar.showMessage(f"Surface wave scan error: {exc}", 5000)

    def action_solve_focal_mechanism(self) -> Optional[Dict[str, Any]]:
        p = self.engine.current_params
        vis = self.engine.visible_traces()
        try:
            pols = extract_polarities_from_traces(
                vis,
                p.lat,
                p.lon,
                depth_km=p.depth,
                config=self.config,
                ttt=self.ttt,
            )
            if not pols:
                msg = "Need at least one pick with +/- polarity ('c'/'d') to solve focal mechanism."
                self._append_console_log(msg)
                self.status_bar.showMessage(msg, 5000)
                return None
            fm_res = solve_focal_mechanism(pols)
            pol_tuples = [
                (float(m["azimuth"]), float(m["takeoff"]), int(m["observed"]))
                for m in fm_res["station_matches"]
            ]
            fm_payload: Dict[str, Any] = {
                **fm_res,
                "misfit": float(fm_res["misfit_ratio"]),
                "polarities": pol_tuples,
            }
            self._last_focal_mechanism = fm_payload
            self.engine.last_focal_mechanism = fm_payload
            setattr(p, "focal_mechanism", fm_payload)
            self.fit_dock.show()
            self.fit_dock.raise_()
            self.fit_dock.display_focal_mechanism(fm_payload)
            msg = (
                f"Focal Mechanism ({fm_res['n_obs']} polarities): "
                f"Strike={fm_res['strike']:.1f}°, Dip={fm_res['dip']:.1f}°, Rake={fm_res['rake']:.1f}° "
                f"(misfit={fm_res['misfit_ratio']:.2f})"
            )
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 6000)
            return fm_payload
        except Exception as exc:
            self._append_console_log(f"Focal mechanism solver error: {exc}")
            self.status_bar.showMessage(f"Focal mechanism error: {exc}", 5000)
            return None

    def action_open_seedlink_dialog(self) -> None:
        existing = self._find_open_dialog(SeedLinkStreamDialog)
        if existing is not None:
            self._raise_nonmodal(existing)
            return
        dlg = SeedLinkStreamDialog(self.live_coordinator, parent=self)
        self._show_nonmodal(dlg)

    def start_seedlink_stream(
        self,
        host: str,
        port: int = 18000,
        streams: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ):
        norm_streams: Optional[List[Any]] = None
        if streams is not None:
            norm_streams = []
            for item in streams:
                if isinstance(item, str):
                    tok = item.strip()
                    if ":" in tok:
                        left, sel = tok.split(":", 1)
                        parts = [p for p in left.replace(".", "_").split("_") if p]
                        if len(parts) >= 2:
                            sel_norm = sel.strip().upper()
                            if "." not in sel_norm:
                                sel_norm = f"{sel_norm}.D"
                            norm_streams.append((parts[0].upper(), parts[1].upper(), sel_norm))
                            continue
                    if "." in tok:
                        parts = tok.split(".")
                        if len(parts) >= 4:
                            loc = parts[2].strip().upper()
                            cha = parts[3].strip().upper() or "BHZ"
                            sel = f"{loc}{cha}.D" if loc else f"{cha}.D"
                            norm_streams.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                            continue
                        if len(parts) >= 2:
                            sel = f"{parts[2].strip().upper()}.D" if len(parts) == 3 and parts[2].strip() else "BHZ.D"
                            norm_streams.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                            continue
                    if "_" in tok:
                        parts = [p for p in tok.split("_") if p]
                        if len(parts) >= 2:
                            sel = f"{parts[2].upper()}.D" if len(parts) >= 3 else "BHZ.D"
                            norm_streams.append((parts[0].upper(), parts[1].upper(), sel))
                            continue
                norm_streams.append(item)

        worker = self.live_coordinator.start_seedlink(
            host=host,
            port=port,
            streams=norm_streams,
            **kwargs,
        )
        if hasattr(self, "act_seedlink_stop"):
            self.act_seedlink_stop.setEnabled(True)
        return worker

    def action_stop_seedlink(self) -> None:
        self.live_coordinator.stop_seedlink()
        if hasattr(self, "act_seedlink_stop"):
            self.act_seedlink_stop.setEnabled(False)
        self.status_bar.showMessage("SeedLink stream stopped.", 4000)

    def _on_seedlink_traces_updated(self, stream_traces: list) -> None:
        if not stream_traces:
            return
        was_empty = len(self.engine.traces) == 0
        existing_picks: Dict[Tuple[str, str, str, str], List[Any]] = {}
        kept_traces: List[Trace] = []
        for tr in self.engine.traces:
            key = (tr.network.upper(), tr.station.upper(), tr.locid.upper(), tr.channel3.upper())
            if tr.picks:
                existing_picks[key] = list(tr.picks)
            if not tr.extra_headers.get("seedlink_stream"):
                kept_traces.append(tr)

        sl_traces: List[Trace] = []
        for tr in stream_traces:
            tr_copy = tr.copy()
            tr_copy.extra_headers["seedlink_stream"] = True
            key = (tr_copy.network.upper(), tr_copy.station.upper(), tr_copy.locid.upper(), tr_copy.channel3.upper())
            if not tr_copy.picks and key in existing_picks:
                tr_copy.picks = copy.deepcopy(existing_picks[key])
            sl_traces.append(tr_copy)

        self.engine.traces = kept_traces + sl_traces
        self._register_unfiltered_traces(sl_traces, replace=False, overwrite_existing=False)
        self.refresh_views(reset_time=was_empty)

    def _on_seedlink_status_changed(self, status: str) -> None:
        running = bool(self.live_coordinator.is_seedlink_running)
        if hasattr(self, "act_seedlink_stop"):
            self.act_seedlink_stop.setEnabled(running)
        self.status_bar.showMessage(f"SeedLink: {status}", 3000)

    def _on_seedlink_error(self, err: str) -> None:
        self.status_bar.showMessage(f"SeedLink warning: {err}", 4000)

    def autosave_session(self, clean_exit: bool = False) -> Optional[str]:
        disp = {
            "norm_mode": self.main_canvas.norm_mode,
            "global_zoom": self.main_canvas.global_zoom,
            "view_t_lo": self.main_canvas.view_t_lo,
            "view_t_hi": self.main_canvas.view_t_hi,
            "mag_t_lo": self.main_canvas.mag_t_lo,
            "mag_t_hi": self.main_canvas.mag_t_hi,
            "min_slot_h": self.main_canvas.min_slot_h,
            "group_by_station": self.main_canvas.group_by_station,
            "show_pick_error_bars": self.main_canvas.show_pick_error_bars,
            "cursor_overlay_mode": self.mag_canvas.cursor_overlay_mode,
        }
        sel_idxs = [i for i, t in enumerate(self.engine.traces) if t.selected]
        saved = self.session_manager.save(
            traces=self.engine.traces,
            param_sets=self.engine.param_sets,
            active_set_index=self.engine.active_set_index,
            selected_trace_indices=sel_idxs,
            active_trace_index=self.main_canvas.mag_trace_idx,
            display_settings=disp,
            clean_exit=clean_exit,
        )
        self._has_autosaved = True
        return saved

    def _on_autosave_timer(self) -> None:
        if self.engine.traces or self._explicit_session_path or self._has_autosaved:
            try:
                self.autosave_session(clean_exit=False)
            except Exception:
                pass

    def _load_trace_data_for_session(self, raw_tr: Dict[str, Any]) -> Optional[np.ndarray]:
        src = str(raw_tr.get("source_path") or "")
        if not src or not os.path.isfile(src):
            return None
        sta = str(raw_tr.get("station") or "").upper()
        ch1 = str(raw_tr.get("chan1") or "B").upper()
        ch2 = str(raw_tr.get("chan2") or "H").upper()
        cp = str(raw_tr.get("comp") or "Z").upper()
        ch3 = f"{ch1}{ch2}{cp}"
        low = src.lower()
        try:
            if low.endswith(".gse"):
                trs = read_gse_file(src, config=self.config)
            elif low.endswith((".qhd", ".qbn")):
                from shm_app.io.qfile import read_qfile

                trs = read_qfile(src, config=self.config)
            elif low.endswith((".asc", ".txt")):
                from shm_app.io.qfile import read_ascii_file

                trs = read_ascii_file(src, config=self.config)
            else:
                trs = read_mseed_file(src, config=self.config)
        except Exception:
            return None
        for cand in trs:
            if cand.station.upper() == sta and cand.channel3.upper() == ch3:
                return cand.data
        for cand in trs:
            if cand.station.upper() == sta:
                return cand.data
        return trs[0].data if trs else None

    def restore_session(self, session_path: Optional[Union[str, os.PathLike]] = None) -> bool:
        mgr = SessionManager(session_path) if session_path is not None else self.session_manager
        res = mgr.load(trace_loader=self._load_trace_data_for_session, raise_on_error=False)
        if not res.get("ok", False):
            err = res.get("error") or "Unable to restore session."
            self._append_console_log(f"Session restore skipped: {err}")
            return False

        self.engine.traces = list(res.get("traces") or [])
        ps_list = list(res.get("param_sets") or [])
        for idx, ps in enumerate(ps_list):
            if idx < len(self.engine.param_sets):
                self.engine.param_sets[idx] = ps
            else:
                self.engine.param_sets.append(ps)
        if self.engine.param_sets:
            self.engine.active_set_index = max(
                0, min(len(self.engine.param_sets) - 1, int(res.get("active_set_index", 0)))
            )

        disp = res.get("display_settings") or {}
        if disp.get("norm_mode") in ("A", "C", "AW", "SW"):
            self.engine.norm_mode = disp["norm_mode"]
            self.main_canvas.norm_mode = disp["norm_mode"]
        if disp.get("global_zoom") is not None:
            gz = max(0.05, min(200.0, float(disp["global_zoom"])))
            self.engine.global_zoom = gz
            self.main_canvas.global_zoom = gz
        v_lo = disp.get("view_t_lo")
        v_hi = disp.get("view_t_hi")
        if v_lo is not None and v_hi is not None:
            self.engine.view_window = (float(v_lo), float(v_hi))
            self.main_canvas.view_t_lo = float(v_lo)
            self.main_canvas.view_t_hi = float(v_hi)
            self._synced_view_window_version = self.engine._view_window_version
        if disp.get("mag_t_lo") is not None and disp.get("mag_t_hi") is not None:
            self.main_canvas.mag_t_lo = float(disp["mag_t_lo"])
            self.main_canvas.mag_t_hi = float(disp["mag_t_hi"])
        if disp.get("min_slot_h") is not None:
            self._set_row_height(int(disp["min_slot_h"]))
        if "group_by_station" in disp:
            grp_on = bool(disp["group_by_station"])
            if hasattr(self, "act_group"):
                self.act_group.setChecked(grp_on)
            self.main_canvas.set_group_by_station(grp_on)
        if "show_pick_error_bars" in disp:
            eb_on = bool(disp["show_pick_error_bars"])
            if hasattr(self, "act_pick_error_bars"):
                self.act_pick_error_bars.setChecked(eb_on)
            self.main_canvas.set_show_pick_error_bars(eb_on)
            self.mag_canvas.set_show_pick_error_bars(eb_on)
        if disp.get("cursor_overlay_mode"):
            self.mag_canvas.set_cursor_overlay_mode(str(disp["cursor_overlay_mode"]))
        self.main_canvas.mag_trace_idx = int(res.get("active_trace_index", 0))

        self._register_unfiltered_traces(self.engine.traces, replace=True, overwrite_existing=False)
        self.refresh_views(reset_time=False)
        msg = f"Restored workspace session ({len(self.engine.traces)} trace(s)) from {mgr.session_path}"
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)
        return True

    def action_restore_session(self) -> None:
        if not self.restore_session():
            self.status_bar.showMessage("No valid saved session found to restore.", 4000)

    def action_create_synthetics(self) -> None:
        for kind, pos in [("KUEPPER", 12.0), ("GAUSS", 25.0), ("SHARP", 38.0)]:
            self.engine.execute_line(f"CREATE {kind} 60.0 0.05 {pos} 150.0 2.0")
        self._register_unfiltered_traces(self.engine.traces)
        self.refresh_views(reset_time=True)

    def action_generate_sfd(self) -> None:
        dpath = QFileDialog.getExistingDirectory(self, "Select Directory of Mini-SEED Files", self.config.data_examples_dir)
        if dpath:
            self.engine.execute_line(f"SFD {dpath}")

    def action_show_shortcuts_help(self) -> None:
        QMessageBox.information(
            self,
            "SHM Keyboard & Mouse Shortcuts Reference",
            "Mouse Controls (Main Trace Canvas):\n"
            "  • Left-Click on Waveform: Pick active phase onset & focus Magnifier\n"
            "  • Drag a pick marker: move it  |  Drag error-bar edge: adjust pick quality Q\n"
            "  • Double-click or Del on a marker: delete it\n"
            "  • Shift + Left-Drag: pan time window  |  Alt+Left / Alt+Right: previous / next view\n"
            "  • Right-Click (no drag): trace context menu\n"
            "  • Shift + Mouse Wheel: scroll rows vertically (View > Trace Row Height)\n"
            "  • Left-Click in Left Margin: Select / Deselect trace (*)\n"
            "  • Middle-Click on Waveform: Cycle Cursor Lock (Crosshair / Peak / Zero)\n"
            "  • Right-Drag: Move / Resize Magnifier Time Window\n"
            "  • Mouse Wheel: Zoom Time Axis (Ctrl+Wheel: Zoom Amplitude)\n\n"
            "Mouse Controls (Magnifier Window):\n"
            "  • Left-Click: High-precision phase onset pick\n"
            "  • Right-Click & Drag: Measure peak-to-peak Amplitude & Period -> Magnitudes\n"
            "  • View > Magnifier Cursor Mode: Standard, Hilbert 90°, Reference Waveform\n\n"
            "Single-Key Accelerators:\n"
            "  • + / - / 0 : Set first-motion polarity (+ compression 'c', - dilatation 'd', 0 none)\n"
            "  • i / e     : Set onset type (i impulsive, e emergent)\n"
            "  • 1 .. 9    : Set pick quality (1..9) and automatic SHM weight\n"
            "  • Left/Right: Scroll time window    • Up/Down: Zoom amplitude\n\n"
            "Ctrl Accelerators:\n"
            "  • Ctrl+R: Read Dialog    • Ctrl+G: BGR Download   • Ctrl+F: Filter Dialog    • Ctrl+U: Unfilter\n"
            "  • Ctrl+P: AutoPick       • Ctrl+T: Theo Phases      • Ctrl+Shift+Del: Delete Phase Picks   • Ctrl+Z / Ctrl+Shift+Z: Undo / Redo\n"
            "  • Ctrl+B: Beam           • Ctrl+N: Align            • Ctrl+K: FK Analysis\n"
            "  • Ctrl+V: Vespagram      • Ctrl+L: Plane Wave Loc   • Ctrl+O: 2D Rotation\n"
            "  • Ctrl+Shift+D: Fit Diagnostics Window\n"
            "  • Ctrl+A: Select All     • Ctrl+Y: Deselect All     • Ctrl+D: Delete Selected",
        )

    def action_show_about(self) -> None:
        QMessageBox.about(
            self,
            "About SeismicHandler Motif (Modernized)",
            "SeismicHandler Motif (SHM) & Command Interpreter (SHC) v3.0\n\n"
            "Reverse-engineered and modernized from Klaus Stammler's BGR SeismicHandler\n"
            "using Python 3, PyQt6, NumPy, and Matplotlib.\n\n"
            "Supports Mini-SEED (Steim-1/2, SFD), SeedLink v3.1 streaming, StationXML, QuakeML 1.2, GeoJSON,\n"
            "GSE1/GSE2, SH Q-Files, ASCII traces, .evt files, SOS Butterworth & recursive IIR filters,\n"
            "trace-specific overlay filters, Brune/Boatwright source spectra, 3C polarization & beachball solver,\n"
            "array beamforming with slowness calibration, IASP91 phase ID, vector PDF/SVG bulletins, and session recovery.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self, "_state_timer") and self._state_timer is not None:
            self._state_timer.stop()
        if hasattr(self, "_autosave_timer") and self._autosave_timer is not None:
            self._autosave_timer.stop()
        self.live_coordinator.shutdown()
        for w in list(getattr(self, "_recompute_workers", [])):
            if w is not None and not sip.isdeleted(w):
                w.cancel()
                if w.isRunning():
                    w.wait(3000)
        if hasattr(self, "_recompute_workers"):
            self._recompute_workers.clear()
        for w in list(getattr(self, "_bulletin_workers", [])):
            if w is not None and not sip.isdeleted(w) and w.isRunning():
                w.wait(3000)
        self._bulletin_workers.clear()
        if hasattr(self, "session_manager") and self.session_manager is not None:
            is_offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            should_save = (
                self._explicit_session_path
                or self._has_autosaved
                or self.session_manager.session_path.is_file()
                or (not is_offscreen and bool(self.engine.traces))
            )
            if should_save:
                try:
                    self.autosave_session(clean_exit=True)
                except Exception:
                    pass
        for dlg in list(self._open_dialogs):
            if dlg is not None and not sip.isdeleted(dlg):
                dlg.close()
        super().closeEvent(event)


def _undoable(label: str):
    """Decorator: snapshot traces and parameters before running a state-changing action."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            self._push_undo(label)
            return fn(self, *args, **kwargs)

        return wrapper

    return deco


_UNDOABLE_ACTIONS = {
    "load_example_dataset": "Load sample data",
    "action_read_evt": "Read event file",
    "action_unfilter": "Unfilter",
    "action_clear_overlay_filters": "Clear overlay filters",
    "_on_clear_overlay_filter_requested": "Clear overlay filter",
    "action_auto_pick": "Auto pick",
    "action_corr_pick": "Correlation pick",
    "action_theoretical_phases": "Theoretical phases",
    "action_delete_active_pick": "Delete picks",
    "action_demean": "Demean",
    "action_detrend": "Detrend",
    "action_despike": "Despike",
    "action_integrate": "Integrate",
    "action_differentiate": "Differentiate",
    "action_hilbert": "Hilbert",
    "action_envelope": "Envelope",
    "action_flip_traces": "Invert sign",
    "_on_trace_invert_requested": "Invert sign",
    "action_rotate_2d": "Rotate 2D",
    "action_rotate_3d": "Rotate 3D",
    "action_cut_to_magnifier": "Cut traces",
    "action_form_beam": "Beam",
    "action_align_phase": "Align",
    "action_sum_traces": "Sum",
    "action_pws_traces": "Phase-weighted stack",
    "action_locate_plane_wave": "Plane wave location",
    "action_locate_by_slowness": "Locate by slowness",
    "action_locate_by_sp_diff": "S-P location",
    "action_fit_depth_phase": "Depth phase fit",
    "action_compute_magnitudes": "Magnitudes",
    "action_clear_params": "Clear picks and parameters",
    "action_delete_selected": "Delete selected traces",
    "action_keep_selected": "Keep selected traces",
    "action_sort_by_distance": "Sort by distance",
    "action_sort_by_station": "Sort by station",
    "action_apply_polfil_interactive": "Polarization filter",
    "action_create_synthetics": "Create synthetics",
    "_on_append_vespa_traces": "Vespagram traces",
    "_on_adopt_az_slo": "Adopt FK azimuth/slowness",
    "_on_adopt_az_slo_and_locate": "Adopt FK & locate",
    "_on_adopt_network_loc": "Adopt network location",
    "_on_adopt_pmotion_angles": "Adopt particle-motion angles",
    "_copy_param_set": "Copy parameter set",
}

for _name, _label in _UNDOABLE_ACTIONS.items():
    setattr(SHMMainWindow, _name, _undoable(_label)(getattr(SHMMainWindow, _name)))

MainWindow = SHMMainWindow
