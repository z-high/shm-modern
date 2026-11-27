"""Main Application Window for modernized SeismicHandler Motif (SHM)."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import os
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import uuid
import weakref
import numpy as np

from PyQt6 import sip
from PyQt6.QtCore import QPoint, QTimer, Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import get_config
from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.core.models import Trace
from shm_app.core.session import SessionManager, resolve_session_path
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.dialogs_analysis import FilterDialog
from shm_app.gui.dialogs_io import ReadDataDialog
from shm_app.gui.fit_dialog import FitDiagnosticWindow
from shm_app.gui.help_texts import apply_tutorial_tooltips, register_tutorial_help, render_tooltip
from shm_app.gui.main_window_analysis import _MainWindowAnalysisMixin
from shm_app.gui.main_window_help import _MainWindowHelpMixin
from shm_app.gui.main_window_io import _MainWindowIOMixin
from shm_app.gui.main_window_live import _MainWindowLiveMixin
from shm_app.gui.main_window_map import _MainWindowMapMixin
from shm_app.gui.main_window_menus import build_menus_and_toolbar
from shm_app.gui.main_window_seedlink import _MainWindowSeedLinkMixin
from shm_app.gui.main_window_session import _MainWindowSessionMixin
from shm_app.gui.main_window_utils import (
    _install_qt_excepthook,
    _quote_sh_arg,
    _undoable,
    apply_dark_theme,
)
from shm_app.gui.main_window_workers import _RecomputeFitWorker
from shm_app.gui.panels import ParameterBoxPanel, PhaseDialogPanel
from shm_app.gui.ui_helpers import apply_c_locale, get_dark_stylesheet
from shm_app.gui.waveform_canvases import MagnifierCanvas, MainTraceCanvas
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.io.gse import read_gse_file
from shm_app.io.mseed import read_mseed_file
from shm_app.plugins.live_service import LiveEventCoordinator
from shm_app.processing.filters import filter_trace
from shm_app.processing.trace_ops import cut_trace
from shm_app.seismology.traveltime import get_ttt_manager


class SHMMainWindow(
    _MainWindowAnalysisMixin,
    _MainWindowIOMixin,
    _MainWindowSessionMixin,
    _MainWindowSeedLinkMixin,
    _MainWindowLiveMixin,
    _MainWindowMapMixin,
    _MainWindowHelpMixin,
    QMainWindow,
):
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
        _install_qt_excepthook()
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
        self._busy_depth: int = 0
        self._align_saved_offsets: Dict[Tuple[str, str, float], float] = {}
        self._pending_ready_event_id: Optional[str] = None
        self._last_pmotion_station: Optional[str] = None
        self._last_fit_diagnostic: Any = None
        self._last_focal_mechanism: Optional[Dict[str, Any]] = None
        self._last_session_restore_error: Optional[str] = None
        self._temp_session_dir: Optional[tempfile.TemporaryDirectory] = None

        env_session_path = os.environ.get("SHM_SESSION_PATH") or None
        is_offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        self._explicit_session_path: bool = (session_path is not None) or (env_session_path is not None)
        if session_path is not None:
            resolved_sess = resolve_session_path(session_path)
        elif env_session_path is not None:
            resolved_sess = resolve_session_path(env_session_path)
        elif is_offscreen:
            self._temp_session_dir = tempfile.TemporaryDirectory(prefix="shm_offscreen_session_")
            resolved_sess = resolve_session_path(os.path.join(self._temp_session_dir.name, ".shm_session.json"))
        else:
            resolved_sess = resolve_session_path(None)
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
        self.fit_bridge.adoptFocalMechanismRequested.connect(self._on_adopt_focal_mechanism)
        self.fit_bridge.highlightTraceRequested.connect(self._on_highlight_fit_trace)
        self.fit_bridge.zoomToPhaseRequested.connect(self._on_zoom_to_fit_phase)
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
        self.main_canvas.pickEditStarted.connect(self._on_pick_edit_started)
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
        self.mag_canvas.pickDeleteRequested.connect(self._on_pick_delete_requested)
        self.mag_canvas.pickMoved.connect(self._on_pick_moved)
        self.mag_canvas.pickChanged.connect(self._on_pick_changed)
        self.mag_canvas.pickEditStarted.connect(self._on_pick_edit_started)
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
            self.load_example_dataset("GRF_031102_0225.GSE", interactive=False)
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

    def _begin_busy(self, message: str = "") -> None:
        if message and hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(message, 0)
        if QApplication.instance() is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._busy_depth += 1

    def _end_busy(self) -> None:
        if self._busy_depth > 0:
            self._busy_depth -= 1
            if QApplication.instance() is not None:
                QApplication.restoreOverrideCursor()

    @contextmanager
    def _busy_cursor(self, message: str = "") -> Iterator[None]:
        self._begin_busy(message)
        try:
            yield
        finally:
            self._end_busy()

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

    def _on_pick_edit_started(self, _trace_idx: int, label: str) -> None:
        self._push_undo(label or "Edit pick")

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

    def _resolve_visible_mag_trace_idx(self) -> Optional[int]:
        """Resolve main_canvas.mag_trace_idx to a non-hidden (visible) trace index."""
        vis_indexed = [(i, t) for i, t in enumerate(self.engine.traces) if not t.hidden and t.length > 0]
        if not vis_indexed:
            vis_indexed = [(i, t) for i, t in enumerate(self.engine.traces) if not t.hidden]
        if not vis_indexed:
            return None
        cur = int(self.main_canvas.mag_trace_idx)
        if any(i == cur for i, _ in vis_indexed):
            return cur
        return vis_indexed[0][0]

    # ------------------------------------------------------------------
    # Spectrum: curves from the yellow Magnifier window, accumulated in one graph
    # ------------------------------------------------------------------

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

        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit)):
            super().keyPressEvent(event)
            return

        # Single-key shortcuts when no text/input widget has focus
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

    def _exec_engine_checked(self, cmd: str, title: str = "SHM Command Error") -> Tuple[bool, str]:
        """Run an engine command and surface any 'SH Error' return via status bar and QMessageBox."""
        res = self.engine.execute_line(cmd)
        if res and res.lstrip().startswith("SH Error"):
            self.status_bar.showMessage(res, 6000)
            QMessageBox.warning(self, title, res)
            return False, res
        return True, res or ""

    def refresh_views(self, reset_time: bool = False) -> None:
        self._sync_unfiltered_backup_view()
        self._sync_controls_from_engine()
        if reset_time:
            self.engine.view_window = None
            self._synced_view_window_version = self.engine._view_window_version
        self.main_canvas.set_traces(self.engine.traces, reset_view=reset_time)
        idx = self._resolve_visible_mag_trace_idx()
        if idx is not None:
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
        if hasattr(self, "phase_panel") and self.phase_panel.chk_depth_rays.isChecked():
            self._on_depth_phase_check_toggled(True)
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

    def load_example_dataset(self, name: str, *, interactive: bool = True) -> bool:
        ex_dir = self.config.data_examples_dir
        if name == "mseed":
            dpath = os.path.join(ex_dir, "mseed")
            target_missing = not os.path.isdir(dpath)
            missing_desc = dpath
        else:
            fpath = os.path.join(ex_dir, "gse", name)
            target_missing = not os.path.isfile(fpath)
            missing_desc = fpath

        if target_missing:
            msg = (
                f"Sample dataset '{name}' is not available (not found: {missing_desc}). "
                "Example datasets are external to shm-modern; set SHM_ROOT to an SHM directory "
                "containing data-examples/, or use Tools -> Create Synthetic Test Waveforms..."
            )
            self._append_console_log(msg)
            self.status_bar.showMessage(f"Sample dataset '{name}' not found (data-examples/ missing).", 7000)
            if interactive:
                QMessageBox.warning(self, "Sample Data Not Available", msg)
            return False

        try:
            with self._busy_cursor(f"Loading sample dataset '{name}'..."):
                if name == "mseed":
                    sfd_path = os.path.join(dpath, "sfdlist.sfd")
                    if os.path.isfile(sfd_path):
                        from shm_app.io.sfd import read_from_sfd

                        trs = read_from_sfd(sfd_path, config=self.config)
                    else:
                        trs = []
                        for fn in sorted(os.listdir(dpath)):
                            if fn.endswith(".bhz"):
                                trs.extend(read_mseed_file(os.path.join(dpath, fn), config=self.config))
                else:
                    trs = read_gse_file(fpath, config=self.config)
        except Exception as exc:
            msg = f"Failed to load sample dataset '{name}': {exc}"
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 7000)
            if interactive:
                QMessageBox.warning(self, "Sample Data Load Error", msg)
            return False

        if not trs:
            msg = f"Sample dataset '{name}' contained no readable traces."
            self._append_console_log(msg)
            self.status_bar.showMessage(msg, 6000)
            if interactive:
                QMessageBox.warning(self, "Sample Data Not Available", msg)
            return False

        if interactive:
            self._push_undo("Load sample data")
        self.engine.traces = trs
        self._register_unfiltered_traces(trs, replace=True, overwrite_existing=True)
        self.refresh_views(reset_time=True)
        msg = f"Loaded {len(trs)} traces from example dataset '{name}'."
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)
        return True

    def action_create_synthetics(self) -> None:
        for kind, pos in [("KUEPPER", 12.0), ("GAUSS", 25.0), ("SHARP", 38.0)]:
            self.engine.execute_line(f"CREATE {kind} 60.0 0.05 {pos} 150.0 2.0")
        self._register_unfiltered_traces(self.engine.traces)
        self.refresh_views(reset_time=True)

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
        try:
            dlg = ReadDataDialog(config=self.config, parent=self)
            if dlg.exec():
                if not dlg.loaded_traces:
                    QMessageBox.warning(self, "Read Waveforms", "No waveform traces were loaded.")
                    return
                self._load_waveforms(
                    dlg.loaded_traces,
                    dlg.append_mode,
                    "Read waveforms",
                    f"Loaded {len(dlg.loaded_traces)} waveform trace(s).",
                )
        except Exception as exc:
            self._append_console_log(f"Open Waveforms error: {exc}")
            QMessageBox.warning(self, "Open Waveforms Error", str(exc))

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

    def action_auto_pick(self) -> None:
        with self._busy_cursor("Auto-picking phases..."):
            self._exec_engine_checked("PICK ALL", title="Auto Pick")
        self.refresh_views()

    def action_corr_pick(self) -> None:
        with self._busy_cursor("Correlation picking..."):
            self._exec_engine_checked(f"CORRPICK {self.phase_panel.active_phase}", title="Correlation Pick")
        self.refresh_views()

    def action_theoretical_phases(self) -> None:
        p = self.engine.current_params
        dist = p.distance if p.distance > 0 else 65.0
        depth = p.depth if p.depth > 0 else 33.0
        self._exec_engine_checked(f"THEO {dist} {depth}", title="Theoretical Phases")
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
        self._exec_engine_checked("DEMEAN ALL", title="Demean")
        self.refresh_views()

    def action_detrend(self) -> None:
        self._exec_engine_checked("TREND ALL", title="Detrend")
        self.refresh_views()

    def action_despike(self) -> None:
        self._exec_engine_checked("DESPIKE ALL 5.0", title="Despike")
        self.refresh_views()

    def action_integrate(self) -> None:
        self._exec_engine_checked("INT ALL", title="Integrate")
        self.refresh_views()

    def action_differentiate(self) -> None:
        self._exec_engine_checked("DERIVE ALL 1", title="Differentiate")
        self.refresh_views()

    def action_hilbert(self) -> None:
        self._exec_engine_checked("HILBERT ALL", title="Hilbert")
        self.refresh_views()

    def action_envelope(self) -> None:
        self._exec_engine_checked("ENVELOPE ALL", title="Envelope")
        self.refresh_views()

    def action_resample(self) -> None:
        val, ok = QInputDialog.getDouble(self, "Resample Traces", "New Sample Interval dt (seconds):", 0.025, 0.001, 10.0, 4)
        if ok:
            self._push_undo("Resample")
            self._exec_engine_checked(f"RESAMPLE {val}", title="Resample")
            self._invalidate_stale_unfiltered()
            self.refresh_views()

    def action_decimate(self) -> None:
        val, ok = QInputDialog.getInt(self, "Decimate Traces", "Decimation Factor (integer):", 2, 1, 100)
        if ok:
            self._push_undo("Decimate")
            self._exec_engine_checked(f"DECIMATE {val}", title="Decimate")
            self._invalidate_stale_unfiltered()
            self.refresh_views()

    def action_flip_traces(self) -> None:
        self._exec_engine_checked("FLIP SEL", title="Invert Sign")
        self.refresh_views()

    def action_rotate_2d(self) -> None:
        az = self.engine.current_params.azimuth
        self._exec_engine_checked(f"ROT2 {az}", title="2D Rotation")
        self.refresh_views()

    def action_rotate_3d(self) -> None:
        p = self.engine.current_params
        if p.incidence > 0:
            self._exec_engine_checked(f"ROT3 {p.azimuth} {p.incidence}", title="3D Rotation")
        else:
            self._exec_engine_checked(f"ROT3 {p.azimuth}", title="3D Rotation")
        self.refresh_views()

    def action_zoom_to_magnifier(self) -> None:
        self.main_canvas.push_view_history()
        self.main_canvas.view_t_lo = self.main_canvas.mag_t_lo
        self.main_canvas.view_t_hi = self.main_canvas.mag_t_hi
        self.engine.view_window = (self.main_canvas.mag_t_lo, self.main_canvas.mag_t_hi)
        self.main_canvas.invalidate_cache()

    def action_cut_to_magnifier(self) -> None:
        if not self.engine.traces:
            self.status_bar.showMessage("No traces loaded to cut.", 4000)
            return
        lo, hi = self.main_canvas.mag_t_lo, self.main_canvas.mag_t_hi
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            QMessageBox.warning(self, "Cut to Magnifier", "Invalid magnifier time window selected.")
            return
        ref_ep = self.main_canvas.reference_epoch()
        snap = self._snapshot("Cut traces")
        try:
            cut_list = [cut_trace(tr, lo, hi, relative_to_epoch=ref_ep) for tr in self.engine.traces]
        except Exception as exc:
            QMessageBox.warning(self, "Cut to Magnifier", str(exc))
            self.status_bar.showMessage(f"Cut failed: {exc}", 5000)
            return
        self._undo_stack.append(snap)
        del self._undo_stack[: -self.UNDO_LIMIT]
        self._redo_stack.clear()
        self._update_undo_actions()
        self.engine.traces[:] = cut_list
        self._invalidate_stale_unfiltered()
        self.refresh_views(reset_time=True)

    def action_reset_windows(self) -> None:
        self.main_canvas.push_view_history()
        self.engine.execute_line("TRESET")
        self.main_canvas.view_t_lo = None
        self.main_canvas.view_t_hi = None
        self.refresh_views(reset_time=True)

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

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self, "_state_timer") and self._state_timer is not None:
            self._state_timer.stop()
        if hasattr(self, "_autosave_timer") and self._autosave_timer is not None:
            self._autosave_timer.stop()
        while getattr(self, "_busy_depth", 0) > 0:
            self._end_busy()
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
                or (not is_offscreen and (self.session_manager.session_path.is_file() or bool(self.engine.traces)))
            )
            if should_save:
                try:
                    self.autosave_session(clean_exit=True)
                except Exception:
                    pass
        if hasattr(self, "_temp_session_dir") and self._temp_session_dir is not None:
            try:
                self._temp_session_dir.cleanup()
            except Exception:
                pass
            self._temp_session_dir = None
        if hasattr(self, "fit_dock") and self.fit_dock is not None and not sip.isdeleted(self.fit_dock):
            try:
                self.fit_dock.cleanup_canvases()
            except Exception:
                pass
            try:
                self.fit_dock.close()
            except Exception:
                pass
        for dlg in list(self._open_dialogs):
            if dlg is not None and not sip.isdeleted(dlg):
                dlg.close()
        super().closeEvent(event)


_UNDOABLE_ACTIONS = {
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
    "_on_world_map_adopt_event": "Adopt event parameters",
    "_copy_param_set": "Copy parameter set",
}

for _name, _label in _UNDOABLE_ACTIONS.items():
    setattr(SHMMainWindow, _name, _undoable(_label)(getattr(SHMMainWindow, _name)))

MainWindow = SHMMainWindow
