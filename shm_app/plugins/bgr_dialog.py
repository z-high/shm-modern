"""Interactive PyQt6 dialog and background worker for the BGR EIDA Mini-SEED downloader plugin."""

from __future__ import annotations

import datetime
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QDate, QTime, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QTextCharFormat
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault, widen_combo
from shm_app.plugins.bgr_fetch import (
    BGRFetchSummary,
    BGR_FDSN_DATASELECT_URL,
    DEFAULT_DOWNLOAD_DIR,
    GRF_STATIONS,
    GRSN_STATIONS,
    StationDownloadResult,
    expand_channels,
    fetch_bgr_waveforms,
    format_fdsn_time,
    geress_stations,
    parse_station_list,
    parse_time_input,
)


class BGRDownloadWorker(QThread):
    """Background worker thread running FDSNWS Mini-SEED downloads without blocking the GUI."""

    progress = pyqtSignal(int, int, object)  # (index, total, StationDownloadResult)
    finishedSummary = pyqtSignal(object)  # BGRFetchSummary
    failed = pyqtSignal(str)

    def __init__(
        self,
        stations: List[str],
        start_epoch: float,
        duration: float,
        output_dir: str,
        network: str,
        channels: List[str],
        location: str,
        base_url: str,
        build_sfd: bool,
        config: SHMConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.stations = list(stations)
        self.start_epoch = float(start_epoch)
        self.duration = float(duration)
        self.output_dir = output_dir
        self.network = network
        self.channels = list(channels)
        self.location = location
        self.base_url = base_url
        self.build_sfd = build_sfd
        self.config = config

    def run(self) -> None:
        try:
            summary = fetch_bgr_waveforms(
                stations=self.stations,
                start_time=self.start_epoch,
                duration=self.duration,
                output_dir=self.output_dir,
                network=self.network,
                channels=self.channels,
                location=self.location,
                base_url=self.base_url,
                build_sfd=self.build_sfd,
                config=self.config,
                progress_cb=lambda idx, total, res: self.progress.emit(idx, total, res),
                cancel_check=self.isInterruptionRequested,
            )
            self.finishedSummary.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class BGRFetchDialog(QDialog):
    """Plugin dialog to download Mini-SEED waveforms from BGR EIDA (FDSNWS) and load them into SHM."""

    waveformsDownloaded = pyqtSignal(list, bool, str)  # (traces, append_mode, status_message)

    def __init__(
        self,
        config: Optional[SHMConfig] = None,
        view_window: Optional[tuple] = None,
        origin_str: str = "",
        workspace_stations: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("BGR EIDA Mini-SEED Downloader (FDSN-WS)")
        self.setSizeGripEnabled(False)
        self.resize(840, 620)
        self.config = config or get_config()
        self._view_window = view_window  # (start_epoch, duration) of current workspace view if available
        self._origin_str = origin_str.strip()
        self._workspace_stations = parse_station_list(workspace_stations or [])
        self._worker: Optional[BGRDownloadWorker] = None
        self.last_summary: Optional[BGRFetchSummary] = None
        self._syncing_time = False
        self._start_subsec = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        default_start, default_dur = self._initial_time_window()

        self._build_stations_group(layout)
        self._build_time_window_group(layout, default_dur)
        self._build_output_group(layout)
        self._build_progress_and_actions(layout)
        self._register_tutorial_help()
        disable_button_autodefault(self)
        apply_c_locale(self)

        self.start_date_edit.dateChanged.connect(self._on_start_widgets_changed)
        self.start_time_edit.timeChanged.connect(self._on_start_widgets_changed)
        self.end_date_edit.dateChanged.connect(self._on_end_widgets_changed)
        self.end_time_edit.timeChanged.connect(self._on_end_widgets_changed)
        self.dur_spin.valueChanged.connect(self._on_dur_changed)

        self.set_time_window(default_start, default_dur)

    def _build_stations_group(self, layout: QVBoxLayout) -> None:
        grp_st = QGroupBox("Stations && Network (BGR EIDA / GRSN / GRF / GERES)")
        st_v = QVBoxLayout(grp_st)
        st_v.setContentsMargins(8, 6, 8, 6)
        st_v.setSpacing(5)

        grid = QGridLayout()
        grid.setSpacing(4)
        all_listed = GRF_STATIONS + [s for s in GRSN_STATIONS if s not in GRF_STATIONS]
        self.station_checks: Dict[str, QCheckBox] = {}
        grf_set = set(GRF_STATIONS)
        for idx, sta in enumerate(all_listed):
            cb = QCheckBox(sta)
            cb.setChecked(sta in grf_set)
            self.station_checks[sta] = cb
            grid.addWidget(cb, idx // 9, idx % 9)
        st_v.addLayout(grid)

        preset_h = QHBoxLayout()
        btn_grf = QPushButton("GRF Array")
        btn_grf.clicked.connect(lambda: self._apply_station_preset(GRF_STATIONS))
        btn_grsn = QPushButton("GRSN Regional")
        btn_grsn.clicked.connect(lambda: self._apply_station_preset(GRSN_STATIONS))
        btn_geres = QPushButton("GERES Array")
        btn_geres.clicked.connect(self._select_geress)
        preset_h.addWidget(btn_grf)
        preset_h.addWidget(btn_grsn)
        preset_h.addWidget(btn_geres)
        if self._workspace_stations:
            btn_ws = QPushButton(f"Workspace ({len(self._workspace_stations)})")
            btn_ws.clicked.connect(lambda: self._apply_station_preset(self._workspace_stations))
            preset_h.addWidget(btn_ws)
        btn_all = QPushButton("All Listed")
        btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self.station_checks.values()])
        btn_none = QPushButton("Clear")
        btn_none.clicked.connect(self._clear_all_stations)
        preset_h.addWidget(btn_all)
        preset_h.addWidget(btn_none)
        preset_h.addStretch(1)
        st_v.addLayout(preset_h)

        net_h = QHBoxLayout()
        self.network_combo = QComboBox()
        self.network_combo.setEditable(True)
        self.network_combo.addItems(["GR", "GE", "*", "BW", "SX", "TH"])
        self.network_combo.setCurrentText("GR")
        widen_combo(self.network_combo, min_width=130, popup_width=160)

        self.location_combo = QComboBox()
        self.location_combo.setEditable(True)
        self.location_combo.addItems(["*", "--", "00", "01"])
        self.location_combo.setCurrentText("*")
        widen_combo(self.location_combo, min_width=120, popup_width=150)

        self.extra_stations_edit = QLineEdit()
        self.extra_stations_edit.setPlaceholderText("Optional extra station codes (e.g. GEA0,GEA1,ROT)")

        net_h.addWidget(QLabel("Network:"))
        net_h.addWidget(self.network_combo)
        net_h.addWidget(QLabel("Location:"))
        net_h.addWidget(self.location_combo)
        net_h.addWidget(QLabel("Extra Stations:"))
        net_h.addWidget(self.extra_stations_edit, stretch=1)
        st_v.addLayout(net_h)
        layout.addWidget(grp_st)

    def _build_time_window_group(self, layout: QVBoxLayout, default_dur: float) -> None:
        grp_win = QGroupBox("Time Window (UTC) && Channels")
        form = QFormLayout(grp_win)
        form.setContentsMargins(8, 6, 8, 6)
        form.setVerticalSpacing(5)

        time_grid = QGridLayout()
        time_grid.setHorizontalSpacing(6)
        time_grid.setVerticalSpacing(4)
        self.start_date_edit = self._create_date_edit()
        self.start_time_edit = self._create_time_edit()
        self.end_date_edit = self._create_date_edit()
        self.end_time_edit = self._create_time_edit()

        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(5.0, 86400.0)
        self.dur_spin.setDecimals(1)
        self.dur_spin.setSingleStep(60.0)
        self.dur_spin.setKeyboardTracking(False)
        self.dur_spin.setMinimumWidth(96)
        self.dur_spin.setValue(default_dur)
        self.dur_spin.setSuffix(" s")

        self.dur_human_lbl = QLabel("")
        self.dur_human_lbl.setStyleSheet("color: #79c0ff; font-family: monospace; font-size: 11px;")

        time_grid.addWidget(QLabel("Start:"), 0, 0)
        time_grid.addWidget(self.start_date_edit, 0, 1)
        time_grid.addWidget(self.start_time_edit, 0, 2)
        arrow_lbl = QLabel("→")
        arrow_lbl.setStyleSheet("color: #8b949e; font-weight: bold; padding: 0 2px;")
        time_grid.addWidget(arrow_lbl, 0, 3)
        time_grid.addWidget(QLabel("End:"), 0, 4)
        time_grid.addWidget(self.end_date_edit, 0, 5)
        time_grid.addWidget(self.end_time_edit, 0, 6)
        time_grid.setColumnStretch(7, 1)

        len_h = QHBoxLayout()
        len_h.setSpacing(6)
        len_h.addWidget(self.dur_spin)
        len_h.addWidget(self.dur_human_lbl)
        len_h.addStretch(1)
        time_grid.addWidget(QLabel("Length:"), 1, 0)
        time_grid.addLayout(len_h, 1, 1, 1, 7)
        form.addRow("UTC Window:", time_grid)

        quick_grid = QGridLayout()
        quick_grid.setHorizontalSpacing(6)
        quick_grid.setVerticalSpacing(4)

        row0_h = QHBoxLayout()
        row0_h.setSpacing(6)
        btn_recent = QPushButton("Last 30 min")
        btn_recent.clicked.connect(self._fill_last_30_min)
        row0_h.addWidget(btn_recent)
        btn_recent_2h = QPushButton("Last 2 h")
        btn_recent_2h.clicked.connect(self._fill_last_2_h)
        row0_h.addWidget(btn_recent_2h)
        if self._view_window is not None:
            btn_view = QPushButton("From Current View")
            btn_view.clicked.connect(self._fill_from_view)
            row0_h.addWidget(btn_view)
        if self._origin_str:
            btn_org = QPushButton("From Origin Time")
            btn_org.clicked.connect(self._fill_from_origin)
            row0_h.addWidget(btn_org)
        self.btn_paste_time = QPushButton("Paste SH / ISO...")
        self.btn_paste_time.clicked.connect(self._prompt_paste_time)
        row0_h.addWidget(self.btn_paste_time)
        row0_h.addStretch(1)
        quick_grid.addLayout(row0_h, 0, 0)

        row1_h = QHBoxLayout()
        row1_h.setSpacing(6)
        row1_h.addWidget(QLabel("Duration:"))
        for label, secs in [("2 min", 120.0), ("10 min", 600.0), ("30 min", 1800.0), ("1 h", 3600.0)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=secs: self.dur_spin.setValue(s))
            row1_h.addWidget(b)
        row1_h.addStretch(1)
        quick_grid.addLayout(row1_h, 1, 0)
        form.addRow("Quick Fill:", quick_grid)

        chan_h = QHBoxLayout()
        chan_h.setSpacing(8)
        self.chan_combo = QComboBox()
        self.chan_combo.setEditable(True)
        self.chan_combo.addItems(["BH", "HH", "LH", "SH", "EH", "HH?,BH?"])
        self.chan_combo.setCurrentText("BH")
        widen_combo(self.chan_combo, min_width=170, popup_width=210)
        self.chk_z = QCheckBox("Z")
        self.chk_z.setChecked(True)
        self.chk_n = QCheckBox("N")
        self.chk_n.setChecked(True)
        self.chk_e = QCheckBox("E")
        self.chk_e.setChecked(True)
        chan_h.addWidget(QLabel("Band / Pattern:"))
        chan_h.addWidget(self.chan_combo)
        chan_h.addSpacing(8)
        chan_h.addWidget(QLabel("Components:"))
        chan_h.addWidget(self.chk_z)
        chan_h.addWidget(self.chk_n)
        chan_h.addWidget(self.chk_e)
        chan_h.addStretch(1)
        form.addRow("Channels:", chan_h)
        layout.addWidget(grp_win)

    def _build_output_group(self, layout: QVBoxLayout) -> None:
        grp_io = QGroupBox("Output && Loading")
        io_form = QFormLayout(grp_io)
        io_form.setContentsMargins(8, 6, 8, 6)
        io_form.setVerticalSpacing(5)

        out_h = QHBoxLayout()
        self.out_dir_edit = QLineEdit(DEFAULT_DOWNLOAD_DIR)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dir)
        out_h.addWidget(self.out_dir_edit, stretch=1)
        out_h.addWidget(btn_browse)
        io_form.addRow("Save Mini-SEED to:", out_h)

        self.url_edit = QLineEdit(BGR_FDSN_DATASELECT_URL)
        io_form.addRow("FDSNWS Endpoint:", self.url_edit)

        opt_h = QHBoxLayout()
        opt_h.setSpacing(14)
        self.chk_autoload = QCheckBox("Automatically load into SHM")
        self.chk_autoload.setChecked(True)
        self.chk_append = QCheckBox("Append to existing traces")
        self.chk_sfd = QCheckBox("Generate sfdlist.sfd index")
        self.chk_sfd.setChecked(True)
        opt_h.addWidget(self.chk_autoload)
        opt_h.addWidget(self.chk_append)
        opt_h.addWidget(self.chk_sfd)
        opt_h.addStretch(1)
        io_form.addRow("Options:", opt_h)
        layout.addWidget(grp_io)

    def _build_progress_and_actions(self, layout: QVBoxLayout) -> None:
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        stat_h = QHBoxLayout()
        self.status_label = QLabel("Ready to query BGR EIDA FDSNWS.")
        self.status_label.setStyleSheet("color: #8b949e;")
        stat_h.addWidget(self.status_label, stretch=1)
        self.btn_toggle_log = QPushButton("Activity Log")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setChecked(False)
        self.btn_toggle_log.toggled.connect(self._on_toggle_log)
        stat_h.addWidget(self.btn_toggle_log)
        layout.addLayout(stat_h)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.log_box.setVisible(False)
        layout.addWidget(self.log_box)

        btn_h = QHBoxLayout()
        self.btn_download = QPushButton("Download && Load")
        self.btn_download.setStyleSheet(
            "QPushButton { background-color: #1f6feb; border-color: #388bfd; color: #ffffff; font-weight: bold; outline: none; }"
            "QPushButton:hover { background-color: #388bfd; }"
            "QPushButton:disabled { background-color: #16335c; color: #8b949e; border-color: #1f4b8f; }"
        )
        self.btn_download.clicked.connect(self._start_download)
        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_download)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_h.addStretch(1)
        btn_h.addWidget(self.btn_download)
        btn_h.addWidget(self.btn_cancel)
        btn_h.addWidget(btn_close)
        layout.addLayout(btn_h)

    def _on_toggle_log(self, visible: bool) -> None:
        self.log_box.setVisible(visible)

    def _register_tutorial_help(self) -> None:
        register_tutorial_help(self.network_combo, "FDSN Network Code", "dlg_bgr_provider", "", "FDSN network code (e.g. GR, GE, *)")
        register_tutorial_help(self.location_combo, "Location Code", "dlg_bgr_provider", "", "Seed location identifier (*, 00, --)")
        register_tutorial_help(self.extra_stations_edit, "Additional Station Codes", "dlg_bgr_stations", "", "Comma-separated extra FDSN station codes")
        register_tutorial_help(self.start_date_edit, "Start Date (UTC)", "dlg_bgr_time", "", "UTC start date (YYYY-MM-DD)")
        register_tutorial_help(self.start_time_edit, "Start Time (UTC)", "dlg_bgr_time", "", "UTC start time (HH:MM:SS)")
        register_tutorial_help(self.end_date_edit, "End Date (UTC)", "dlg_bgr_time", "", "UTC end date (YYYY-MM-DD)")
        register_tutorial_help(self.end_time_edit, "End Time (UTC)", "dlg_bgr_time", "", "UTC end time (HH:MM:SS)")
        register_tutorial_help(self.dur_spin, "Time Window Length (s)", "dlg_bgr_time", "", "Request window length in seconds")
        register_tutorial_help(
            self.btn_paste_time,
            "Paste Timestamp",
            "dlg_bgr_time",
            "",
            "Paste a timestamp in SH format (DD-MMM-YYYY_HH:MM:SS) or ISO UTC",
        )
        register_tutorial_help(self.chan_combo, "Channel / Stream Code", "dlg_bgr_stations", "", "Band/stream prefix (BH, HH, LH, ...)")
        register_tutorial_help(self.url_edit, "FDSN Dataselect Endpoint", "dlg_bgr_provider", "", "FDSN dataselect endpoint URL")
        register_tutorial_help(self.chk_autoload, "Auto-Load Waveforms", "dlg_bgr_provider", "", "Automatically open downloaded waveforms in SHM")
        register_tutorial_help(self.chk_append, "Append to Existing Traces", "dlg_bgr_provider", "", "Append instead of replacing workspace traces")
        register_tutorial_help(self.chk_sfd, "Generate SFD Index", "dlg_bgr_provider", "", "Write sfdlist.sfd index in download directory")
        register_tutorial_help(self.btn_download, "Download && Load Waveforms", "dlg_bgr_provider", "", "Fetch Mini-SEED data from BGR EIDA / FDSN")

    @staticmethod
    def _create_date_edit() -> QDateEdit:
        w = QDateEdit()
        w.setDisplayFormat("yyyy-MM-dd")
        w.setCalendarPopup(True)
        w.setKeyboardTracking(False)
        w.setMinimumWidth(134)
        w.setToolTip("UTC date (YYYY-MM-DD) — click arrow to open calendar")
        cal = w.calendarWidget()
        if cal is not None:
            cal.setFirstDayOfWeek(Qt.DayOfWeek.Monday)
            cal.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
            weekend_fmt = QTextCharFormat()
            weekend_fmt.setForeground(QColor("#79c0ff"))
            cal.setWeekdayTextFormat(Qt.DayOfWeek.Saturday, weekend_fmt)
            cal.setWeekdayTextFormat(Qt.DayOfWeek.Sunday, weekend_fmt)
        return w

    @staticmethod
    def _create_time_edit() -> QTimeEdit:
        w = QTimeEdit()
        w.setDisplayFormat("HH:mm:ss")
        w.setKeyboardTracking(False)
        w.setMinimumWidth(98)
        w.setToolTip("UTC time (HH:MM:SS)")
        return w

    def _initial_time_window(self) -> Tuple[float, float]:
        if self._view_window is not None:
            t0, dur = self._view_window
            return float(t0), max(10.0, min(86400.0, float(dur)))
        now = time.time() - 3600.0
        return float(now - (now % 60.0)), 600.0

    @staticmethod
    def _widgets_to_epoch(date_edit: QDateEdit, time_edit: QTimeEdit, subsec: float = 0.0) -> float:
        d = date_edit.date()
        t = time_edit.time()
        dt = datetime.datetime(
            d.year(), d.month(), d.day(), t.hour(), t.minute(), t.second(),
            tzinfo=datetime.timezone.utc,
        )
        return dt.timestamp() + float(subsec)

    @staticmethod
    def _set_widgets_from_epoch(date_edit: QDateEdit, time_edit: QTimeEdit, epoch: float) -> float:
        sec = math.floor(float(epoch))
        sub = float(epoch) - sec
        if sub >= 0.9995:
            sec += 1
            sub = 0.0
        elif sub < 0.0005:
            sub = 0.0
        dt = datetime.datetime.fromtimestamp(sec, tz=datetime.timezone.utc)
        date_edit.setDate(QDate(dt.year, dt.month, dt.day))
        time_edit.setTime(QTime(dt.hour, dt.minute, dt.second))
        return sub

    @staticmethod
    def _format_duration_label(secs: float) -> str:
        if secs < 59.5:
            return f"({int(secs)} s)" if abs(secs - round(secs)) < 0.05 else f"({secs:.1f} s)"
        if secs < 3599.5:
            m = int(secs // 60.0)
            s = int(round(secs - m * 60.0))
            if s == 60:
                m += 1
                s = 0
            return f"({m} min)" if s == 0 else f"({m}m {s:02d}s)"
        h = int(secs // 3600.0)
        m = int(round((secs % 3600.0) / 60.0))
        if m == 60:
            h += 1
            m = 0
        return f"({h} h)" if m == 0 else f"({h}h {m:02d}m)"

    def start_epoch(self) -> float:
        """Return currently selected UTC start time as POSIX epoch seconds."""
        return self._widgets_to_epoch(self.start_date_edit, self.start_time_edit, self._start_subsec)

    def end_epoch(self) -> float:
        """Return currently selected UTC end time as POSIX epoch seconds."""
        return self.start_epoch() + float(self.dur_spin.value())

    def set_time_window(self, start_epoch: float, duration: Optional[float] = None) -> None:
        """Programmatically set the UTC start time and optional duration in seconds."""
        dur = float(self.dur_spin.value() if duration is None else duration)
        dur = max(5.0, min(86400.0, dur))
        self._syncing_time = True
        try:
            self._start_subsec = self._set_widgets_from_epoch(self.start_date_edit, self.start_time_edit, start_epoch)
            self.dur_spin.setValue(dur)
            self._set_widgets_from_epoch(self.end_date_edit, self.end_time_edit, float(start_epoch) + dur)
            self.dur_human_lbl.setText(self._format_duration_label(dur))
        finally:
            self._syncing_time = False

    def set_start_time(self, start_time: object) -> None:
        """Set UTC start time from a POSIX epoch float or SH/ISO timestamp string."""
        ep = parse_time_input(start_time) if isinstance(start_time, str) else float(start_time)
        self.set_time_window(ep)

    def _apply_station_preset(self, names: List[str]) -> None:
        target = set(names)
        for sta, cb in self.station_checks.items():
            cb.setChecked(sta in target)
        self.extra_stations_edit.clear()

    def _select_geress(self) -> None:
        for cb in self.station_checks.values():
            cb.setChecked(False)
        ger = geress_stations(self.config)
        rest = []
        for s in ger:
            if s in self.station_checks:
                self.station_checks[s].setChecked(True)
            else:
                rest.append(s)
        self.extra_stations_edit.setText(",".join(rest))

    def _clear_all_stations(self) -> None:
        for cb in self.station_checks.values():
            cb.setChecked(False)
        self.extra_stations_edit.clear()

    def selected_stations(self) -> List[str]:
        """Return ordered list of checked stations plus any extra stations."""
        listed = [sta for sta, cb in self.station_checks.items() if cb.isChecked()]
        extra = parse_station_list(self.extra_stations_edit.text())
        return parse_station_list(listed + extra)

    def selected_channels(self) -> List[str]:
        """Return resolved FDSN channel codes from band/pattern and Z/N/E checkboxes."""
        comps = []
        if self.chk_z.isChecked():
            comps.append("Z")
        if self.chk_n.isChecked():
            comps.append("N")
        if self.chk_e.isChecked():
            comps.append("E")
        return expand_channels(self.chan_combo.currentText(), components=comps)

    def _fill_last_30_min(self) -> None:
        end_ep = time.time() - 120.0
        end_ep -= end_ep % 60.0
        self.set_time_window(end_ep - 1800.0, 1800.0)

    def _fill_last_2_h(self) -> None:
        end_ep = time.time() - 120.0
        end_ep -= end_ep % 60.0
        self.set_time_window(end_ep - 7200.0, 7200.0)

    def _fill_from_view(self) -> None:
        if self._view_window is None:
            return
        t0, dur = self._view_window
        self.set_time_window(float(t0), max(10.0, min(86400.0, float(dur))))

    def _fill_from_origin(self) -> None:
        if not self._origin_str:
            return
        try:
            self.set_start_time(self._origin_str)
        except Exception:
            pass

    def _prompt_paste_time(self) -> None:
        current = format_sh_time(self.start_epoch())
        text, ok = QInputDialog.getText(
            self,
            "Enter Start Time (UTC)",
            "Timestamp in SH format (DD-MMM-YYYY_HH:MM:SS) or ISO (YYYY-MM-DDTHH:MM:SS):",
            text=current,
        )
        if ok and text.strip():
            try:
                self.set_start_time(text.strip())
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Timestamp", f"Could not parse start time: {exc}")

    def _on_start_widgets_changed(self) -> None:
        if self._syncing_time:
            return
        self._syncing_time = True
        try:
            self._start_subsec = 0.0
            start_ep = self._widgets_to_epoch(self.start_date_edit, self.start_time_edit)
            dur = float(self.dur_spin.value())
            self._set_widgets_from_epoch(self.end_date_edit, self.end_time_edit, start_ep + dur)
            self.dur_human_lbl.setText(self._format_duration_label(dur))
        finally:
            self._syncing_time = False

    def _on_dur_changed(self, dur: float) -> None:
        if self._syncing_time:
            return
        self._syncing_time = True
        try:
            start_ep = self.start_epoch()
            self._set_widgets_from_epoch(self.end_date_edit, self.end_time_edit, start_ep + float(dur))
            self.dur_human_lbl.setText(self._format_duration_label(float(dur)))
        finally:
            self._syncing_time = False

    def _on_end_widgets_changed(self) -> None:
        if self._syncing_time:
            return
        self._syncing_time = True
        try:
            end_ep = self._widgets_to_epoch(self.end_date_edit, self.end_time_edit)
            start_ep = self.start_epoch()
            delta = end_ep - start_ep
            if delta < 5.0:
                # Shifting End before Start preserves current duration cleanly
                dur = float(self.dur_spin.value())
                new_start = end_ep - dur
                self._start_subsec = self._set_widgets_from_epoch(self.start_date_edit, self.start_time_edit, new_start)
            elif delta > 86400.0:
                dur = 86400.0
                self.dur_spin.setValue(dur)
                new_start = end_ep - dur
                self._start_subsec = self._set_widgets_from_epoch(self.start_date_edit, self.start_time_edit, new_start)
            else:
                dur = round(delta, 1)
                self.dur_spin.setValue(dur)
            self.dur_human_lbl.setText(self._format_duration_label(float(self.dur_spin.value())))
        finally:
            self._syncing_time = False

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory for Mini-SEED", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def _start_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        stations = self.selected_stations()
        if not stations:
            QMessageBox.warning(self, "BGR Download", "Please select at least one station.")
            return
        channels = self.selected_channels()
        if not channels:
            QMessageBox.warning(self, "BGR Download", "Please select a channel band and at least one component.")
            return
        start_epoch = self.start_epoch()

        duration = float(self.dur_spin.value())
        out_dir = self.out_dir_edit.text().strip() or DEFAULT_DOWNLOAD_DIR
        network = self.network_combo.currentText().strip().upper() or "GR"
        location = self.location_combo.currentText().strip() or "*"
        base_url = self.url_edit.text().strip() or BGR_FDSN_DATASELECT_URL

        self.btn_toggle_log.setChecked(True)
        self.log_box.setVisible(True)
        self.log_box.clear()
        self.log_box.append(
            f"Querying {base_url} | Net={network} | {len(stations)} station(s) | "
            f"Cha={','.join(channels)} | {format_fdsn_time(start_epoch)} + {duration:.0f}s"
        )
        self.progress_bar.setRange(0, len(stations))
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Downloading 0 / {len(stations)} stations from BGR...")
        self.btn_download.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        worker = BGRDownloadWorker(
            stations=stations,
            start_epoch=start_epoch,
            duration=duration,
            output_dir=out_dir,
            network=network,
            channels=channels,
            location=location,
            base_url=base_url,
            build_sfd=self.chk_sfd.isChecked(),
            config=self.config,
            parent=self,
        )
        worker.progress.connect(self._on_worker_progress)
        worker.finishedSummary.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _cancel_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self.status_label.setText("Stopping after current request...")
            self.btn_cancel.setEnabled(False)

    def _on_worker_progress(self, idx: int, total: int, res: StationDownloadResult) -> None:
        self.progress_bar.setValue(idx)
        self.status_label.setText(f"Downloading {idx} / {total}: {res.network}.{res.station} ({res.status})")
        tag = res.status.upper()
        extra = f" -> {os.path.basename(res.filepath)}" if res.filepath else ""
        self.log_box.append(f"[{tag}] {res.network}.{res.station}: {res.message}{extra}")

    def _on_worker_finished(self, summary: BGRFetchSummary) -> None:
        self._worker = None
        self.last_summary = summary
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        ok_cnt = sum(1 for r in summary.results if r.status == "ok")
        nd_cnt = sum(1 for r in summary.results if r.status == "nodata")
        err_cnt = sum(1 for r in summary.results if r.status == "error")
        msg = (
            f"BGR Fetch {'cancelled' if summary.cancelled else 'complete'}: "
            f"{len(summary.traces)} trace(s) from {ok_cnt} station(s) "
            f"({nd_cnt} no data, {err_cnt} errors). Saved in {summary.output_dir}"
        )
        self.status_label.setText(msg)
        self.log_box.append(msg)
        if summary.sfd_path:
            self.log_box.append(f"Generated SFD index: {summary.sfd_path}")

        if summary.traces and self.chk_autoload.isChecked():
            self.waveformsDownloaded.emit(summary.traces, self.chk_append.isChecked(), msg)

    def _on_worker_failed(self, err: str) -> None:
        self._worker = None
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText(f"Download failed: {err}")
        self.btn_toggle_log.setChecked(True)
        self.log_box.setVisible(True)
        self.log_box.append(f"[ERROR] {err}")
        QMessageBox.warning(self, "BGR Download Error", err)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(3000)
        super().closeEvent(event)
