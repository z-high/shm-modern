"""Interactive PyQt6 Live Event Monitor dialog for SHM Live Event Mode."""

from __future__ import annotations

import datetime
import os
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault, make_push_button, widen_combo
from shm_app.plugins.live_events import FEED_SOURCES
from shm_app.plugins.live_planner import DEFAULT_LIVE_CACHE_DIR, LiveConfig
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

_STATE_COLORS: Dict[str, str] = {
    STATE_WAITING: "#d29922",
    STATE_FETCHING: "#58a6ff",
    STATE_RETRY_WAIT: "#f0883e",
    STATE_READY: "#3fb950",
    STATE_LOADED: "#56d364",
    STATE_IGNORED: "#8b949e",
    STATE_ERROR: "#f85149",
}


class LiveEventDialog(QDialog):
    """Non-modal dashboard to control, filter, monitor, and load live earthquake events."""

    openInBGRRequested = pyqtSignal(object)  # LiveEventEntry
    showOnMapRequested = pyqtSignal(str)     # event_id

    def __init__(
        self,
        coordinator: LiveEventCoordinator,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.coordinator = coordinator
        self.setWindowTitle("Live Event Monitor — Automated Earthquake Acquisition")
        self.resize(1120, 740)
        self.setSizeGripEnabled(False)

        self._row_event_ids: List[str] = []
        self._suppress_ui_signals = False
        self._log_count = 0
        self._build_ui()
        self._load_config_to_widgets(self.coordinator.live_config)
        disable_button_autodefault(self)
        apply_c_locale(self)
        self._connect_coordinator()
        self._sync_toggle_button(self.coordinator.is_enabled)
        self._on_stream_status(self.coordinator.stream_state, self.coordinator.stream_detail)
        self._rebuild_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._build_top_bar())

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)
        self.tabs.addTab(self._build_events_tab(), "Detected Events")
        self.tabs.addTab(self._build_rules_tab(), "Rules, Stations && Loading")
        self.tabs.addTab(self._build_log_tab(), "Activity Log")

        footer = QHBoxLayout()
        self.lbl_footer_hint = QLabel("Tip: Closing this window keeps Live Mode running in the background when enabled.")
        self.lbl_footer_hint.setStyleSheet("color: #8b949e;")
        footer.addWidget(self.lbl_footer_hint)
        footer.addStretch(1)
        self.btn_close = make_push_button("Close", slot=self.close)
        footer.addWidget(self.btn_close)
        root.addLayout(footer)

    def _build_top_bar(self) -> QGroupBox:
        top_box = QGroupBox("Live Stream Control (Opt-In)")
        top_l = QHBoxLayout(top_box)
        top_l.setSpacing(10)

        self.btn_toggle_live = make_push_button(
            "Live Mode: OFF",
            slot=self._on_live_toggled,
            checkable=True,
            tooltip="Enable or pause automatic earthquake monitoring & background waveform prefetching.",
        )
        self.btn_toggle_live.setMinimumWidth(195)
        self.btn_toggle_live.setMinimumHeight(30)
        top_l.addWidget(self.btn_toggle_live)

        self.lbl_stream_status = QLabel("● Idle")
        self.lbl_stream_status.setMinimumWidth(200)
        top_l.addWidget(self.lbl_stream_status)

        top_l.addWidget(QLabel("Feed:"))
        self.combo_source = QComboBox()
        for key, label in FEED_SOURCES.items():
            self.combo_source.addItem(label, key)
        widen_combo(self.combo_source, min_width=235, popup_width=285)
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        top_l.addWidget(self.combo_source)

        top_l.addStretch(1)

        self.btn_replay_24h = make_push_button(
            "Fetch Last 24h",
            slot=self._on_replay_24h,
            tooltip="Query the last 24 hours from the active catalog feed (ideal for testing rules & fetches).",
        )
        self.btn_open_cache = make_push_button(
            "Cache Folder",
            slot=self._on_open_cache_dir,
            tooltip="Browse or inspect ~/shm_live_events downloaded Mini-SEED archive directory.",
        )
        top_l.addWidget(self.btn_replay_24h)
        top_l.addWidget(self.btn_open_cache)

        register_tutorial_help(
            self.btn_toggle_live,
            "Toggle Live Event Mode",
            "live_toggle",
            "",
            "Enable or pause automatic earthquake monitoring & background waveform prefetching.",
        )
        register_tutorial_help(self.combo_source, "Live Earthquake Feed", "dlg_live_rules", "", "Select real-time catalog source (EMSC, GFZ, USGS)")
        register_tutorial_help(
            self.btn_replay_24h,
            "Fetch Last 24h Catalog",
            "dlg_live_actions",
            "",
            "Query the last 24 hours from the active catalog feed (ideal for testing rules & fetches).",
        )
        register_tutorial_help(
            self.btn_open_cache,
            "Open Live Cache Folder",
            "dlg_live_actions",
            "",
            "Browse or inspect ~/shm_live_events downloaded Mini-SEED archive directory.",
        )
        return top_box

    def _build_events_tab(self) -> QWidget:
        tab_events = QWidget()
        ev_l = QVBoxLayout(tab_events)
        ev_l.setContentsMargins(6, 6, 6, 6)
        ev_l.setSpacing(6)

        self.events_table = QTableWidget(0, 7)
        self.events_table.setHorizontalHeaderLabels(
            ["Origin UTC", "Mag", "Depth", "Region", "Δ / BAZ", "Phase / ETA", "Status / Waveforms"]
        )
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.events_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.events_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events_table.setAlternatingRowColors(True)
        self.events_table.verticalHeader().setVisible(False)
        hdr = self.events_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.events_table.itemSelectionChanged.connect(self._update_selected_controls)
        self.events_table.itemDoubleClicked.connect(lambda _item: self._on_load_selected())
        ev_l.addWidget(self.events_table, stretch=1)

        self.lbl_event_preview = QLabel("Select an event row to inspect station plan, UTC window, and actions.")
        self.lbl_event_preview.setWordWrap(True)
        self.lbl_event_preview.setStyleSheet("color: #9ba7b4; padding: 3px 4px;")
        ev_l.addWidget(self.lbl_event_preview)

        act_row = QHBoxLayout()
        self.btn_load_event = make_push_button("Load into SHM", slot=self._on_load_selected)
        self.btn_fetch_now = make_push_button("Fetch / Retry Now", slot=self._on_fetch_selected)
        self.btn_show_map = make_push_button("Show on Map", slot=self._on_show_on_map)
        self.btn_open_bgr = make_push_button("Open in BGR Downloader...", slot=self._on_open_in_bgr)
        self.btn_ignore_event = make_push_button("Ignore Event", slot=self._on_ignore_selected)
        act_row.addWidget(self.btn_load_event)
        act_row.addWidget(self.btn_fetch_now)
        act_row.addWidget(self.btn_show_map)
        act_row.addWidget(self.btn_open_bgr)
        act_row.addWidget(self.btn_ignore_event)
        act_row.addStretch(1)
        self.lbl_counts = QLabel("0 events")
        self.lbl_counts.setStyleSheet("color: #8b949e;")
        act_row.addWidget(self.lbl_counts)
        ev_l.addLayout(act_row)

        register_tutorial_help(self.btn_load_event, "Load Selected Event", "dlg_live_actions", "", "Load prefetched Mini-SEED waveforms into SHM workspace")
        register_tutorial_help(self.btn_fetch_now, "Fetch / Retry Waveforms", "dlg_live_actions", "", "Trigger immediate Mini-SEED fetch for the selected event")
        register_tutorial_help(self.btn_show_map, "Show Selected Event on Map", "dlg_live_actions", "", "Open World Map centered/focused on the selected live event")
        register_tutorial_help(self.btn_open_bgr, "Open in BGR Downloader", "dlg_live_actions", "", "Transfer event UTC window and target stations to BGR Downloader")
        register_tutorial_help(self.btn_ignore_event, "Ignore Event", "dlg_live_actions", "", "Suppress notifications for the selected event")
        return tab_events

    def _build_rules_tab(self) -> QWidget:
        tab_cfg = QWidget()
        cfg_root = QVBoxLayout(tab_cfg)
        cfg_root.setContentsMargins(8, 8, 8, 8)
        cfg_root.setSpacing(8)

        cfg_grid = QGridLayout()
        cfg_grid.setSpacing(10)

        # Box A: Observatory & Tiered Distance/Magnitude Filter
        grp_filt = QGroupBox("Observatory && Relevance Filter")
        f_form = QFormLayout(grp_filt)

        ref_h = QHBoxLayout()
        self.edit_ref_station = QLineEdit()
        self.edit_ref_station.setMaximumWidth(90)
        self.lbl_ref_coords = QLabel("")
        self.lbl_ref_coords.setStyleSheet("color: #8b949e;")
        ref_h.addWidget(self.edit_ref_station)
        ref_h.addWidget(self.lbl_ref_coords, stretch=1)
        f_form.addRow("Reference Station:", ref_h)

        self.chk_tiered = QCheckBox("Use Tiered Distance vs. Magnitude Rules (uncheck for Flat Min Mag)")
        self.chk_tiered.toggled.connect(self._update_filter_mode_enabled)
        f_form.addRow(self.chk_tiered)

        loc_h = QHBoxLayout()
        self.spin_local_deg = QDoubleSpinBox()
        self.spin_local_deg.setRange(0.2, 15.0)
        self.spin_local_deg.setSuffix(" °")
        self.spin_local_mag = QDoubleSpinBox()
        self.spin_local_mag.setRange(-1.0, 9.5)
        self.spin_local_mag.setSingleStep(0.1)
        loc_h.addWidget(QLabel("Δ ≤"))
        loc_h.addWidget(self.spin_local_deg)
        loc_h.addWidget(QLabel("Min M:"))
        loc_h.addWidget(self.spin_local_mag)
        f_form.addRow("Local Tier:", loc_h)

        reg_h = QHBoxLayout()
        self.spin_reg_deg = QDoubleSpinBox()
        self.spin_reg_deg.setRange(2.0, 60.0)
        self.spin_reg_deg.setSuffix(" °")
        self.spin_reg_mag = QDoubleSpinBox()
        self.spin_reg_mag.setRange(0.0, 9.5)
        self.spin_reg_mag.setSingleStep(0.1)
        reg_h.addWidget(QLabel("Δ ≤"))
        reg_h.addWidget(self.spin_reg_deg)
        reg_h.addWidget(QLabel("Min M:"))
        reg_h.addWidget(self.spin_reg_mag)
        f_form.addRow("Regional Tier:", reg_h)

        tele_h = QHBoxLayout()
        self.spin_tele_mag = QDoubleSpinBox()
        self.spin_tele_mag.setRange(1.0, 9.5)
        self.spin_tele_mag.setSingleStep(0.1)
        self.spin_max_dist = QDoubleSpinBox()
        self.spin_max_dist.setRange(1.0, 180.0)
        self.spin_max_dist.setSuffix(" °")
        tele_h.addWidget(QLabel("Min M:"))
        tele_h.addWidget(self.spin_tele_mag)
        tele_h.addWidget(QLabel("Max Δ:"))
        tele_h.addWidget(self.spin_max_dist)
        f_form.addRow("Teleseismic Tier:", tele_h)

        self.spin_flat_mag = QDoubleSpinBox()
        self.spin_flat_mag.setRange(0.0, 9.5)
        self.spin_flat_mag.setSingleStep(0.1)
        register_tutorial_help(
            self.spin_flat_mag,
            "Flat Minimum Magnitude",
            "dlg_live_rules",
            "",
            "Global minimum magnitude when tiered rules are unchecked (editing activates Flat mode).",
        )
        self.spin_flat_mag.valueChanged.connect(self._on_flat_mag_edited)
        for _w in (self.spin_local_deg, self.spin_local_mag, self.spin_reg_deg, self.spin_reg_mag, self.spin_tele_mag):
            _w.valueChanged.connect(self._on_tiered_spin_edited)
        f_form.addRow("Flat Min Mag (all Δ):", self.spin_flat_mag)

        self.chk_quarry = QCheckBox("Include quarry blasts / explosions")
        f_form.addRow(self.chk_quarry)
        for _w, _t in [
            (self.edit_ref_station, "Reference Station Code"),
            (self.chk_tiered, "Tiered Distance-Magnitude Rules"),
            (self.spin_local_deg, "Local Distance Radius"),
            (self.spin_local_mag, "Local Minimum Magnitude"),
            (self.spin_reg_deg, "Regional Distance Radius"),
            (self.spin_reg_mag, "Regional Minimum Magnitude"),
            (self.spin_tele_mag, "Teleseismic Minimum Magnitude"),
            (self.spin_max_dist, "Maximum Epicentral Distance"),
            (self.chk_quarry, "Include Quarry Blasts"),
        ]:
            register_tutorial_help(_w, _t, "dlg_live_rules", "", _t)
        cfg_grid.addWidget(grp_filt, 0, 0)

        # Box B: Station Selection Strategy
        grp_sta = QGroupBox("Automatic Station Selection")
        s_form = QFormLayout(grp_sta)

        self.combo_strategy = QComboBox()
        self.combo_strategy.addItem("Observatory Focus (BGR / GRF / GRSN / GERESS)", "observatory")
        self.combo_strategy.addItem("Epicenter-Near (FDSNWS / EIDA Azimuthal)", "epicenter")
        widen_combo(self.combo_strategy, min_width=270, popup_width=320)
        self.combo_strategy.currentIndexChanged.connect(self._update_strategy_enabled)
        s_form.addRow("Strategy:", self.combo_strategy)
        register_tutorial_help(self.combo_strategy, "Live Station Selection Strategy", "dlg_live_stations", "", "Observatory Focus vs Epicenter-Near station planning")

        self.combo_preset = QComboBox()
        for p in ("AUTO", "GRF", "GRSN", "GRF+GRSN", "GERESS", "CUSTOM"):
            self.combo_preset.addItem(p, p)
        widen_combo(self.combo_preset, min_width=180, popup_width=220)
        self.combo_preset.currentIndexChanged.connect(self._update_strategy_enabled)
        s_form.addRow("Observatory Preset:", self.combo_preset)

        self.edit_custom_stations = QLineEdit()
        self.edit_custom_stations.setPlaceholderText("Comma-separated codes, e.g. GRA1, GRB1, BFO, CLZ")
        register_tutorial_help(
            self.edit_custom_stations,
            "Custom Station List",
            "dlg_live_stations",
            "",
            "Station list used for CUSTOM preset (typing automatically selects CUSTOM).",
        )
        self.edit_custom_stations.textEdited.connect(self._on_custom_stations_edited)
        s_form.addRow("Custom Stations:", self.edit_custom_stations)

        sta_lim_h = QHBoxLayout()
        self.spin_max_stations = QSpinBox()
        self.spin_max_stations.setRange(1, 30)
        self.spin_epi_radius = QDoubleSpinBox()
        self.spin_epi_radius.setRange(1.0, 60.0)
        self.spin_epi_radius.setSuffix(" °")
        self.spin_epi_radius.valueChanged.connect(self._on_epi_radius_edited)
        sta_lim_h.addWidget(self.spin_max_stations)
        sta_lim_h.addWidget(QLabel("Epicenter Radius:"))
        sta_lim_h.addWidget(self.spin_epi_radius)
        s_form.addRow("Max Stations:", sta_lim_h)

        ch_h = QHBoxLayout()
        self.combo_band = QComboBox()
        for band in ("BH", "HH", "LH", "EH"):
            self.combo_band.addItem(band, band)
        widen_combo(self.combo_band, min_width=100, popup_width=130)
        self.chk_z = QCheckBox("Z")
        self.chk_n = QCheckBox("N")
        self.chk_e = QCheckBox("E")
        ch_h.addWidget(self.combo_band)
        ch_h.addSpacing(8)
        ch_h.addWidget(self.chk_z)
        ch_h.addWidget(self.chk_n)
        ch_h.addWidget(self.chk_e)
        ch_h.addStretch(1)
        s_form.addRow("Channels:", ch_h)

        cfg_grid.addWidget(grp_sta, 0, 1)

        # Box C: Time Window, Grace Wait & Workspace Auto-Loading
        grp_win = QGroupBox("Wave Window, Archive Delay && Auto-Load Policy")
        w_form = QFormLayout(grp_win)

        win_h = QHBoxLayout()
        self.spin_pre_p = QSpinBox()
        self.spin_pre_p.setRange(10, 600)
        self.spin_pre_p.setSuffix(" s")
        self.spin_post_s = QSpinBox()
        self.spin_post_s.setRange(30, 1800)
        self.spin_post_s.setSuffix(" s")
        self.spin_max_win = QSpinBox()
        self.spin_max_win.setRange(120, 3600)
        self.spin_max_win.setSuffix(" s")
        self.spin_grace = QSpinBox()
        self.spin_grace.setRange(0, 600)
        self.spin_grace.setSuffix(" s")
        win_h.addWidget(QLabel("Pre-P:"))
        win_h.addWidget(self.spin_pre_p)
        win_h.addWidget(QLabel("Post-S:"))
        win_h.addWidget(self.spin_post_s)
        win_h.addWidget(QLabel("Max Window:"))
        win_h.addWidget(self.spin_max_win)
        win_h.addWidget(QLabel("Archive Grace:"))
        win_h.addWidget(self.spin_grace)
        win_h.addStretch(1)
        w_form.addRow("Timing Buffers:", win_h)

        pol_h = QHBoxLayout()
        self.combo_policy = QComboBox()
        self.combo_policy.addItem("Prefetch & Notify (Never interrupts active picking)", "notify")
        self.combo_policy.addItem("Auto-Load when Idle (No manual picks present)", "auto_idle")
        self.combo_policy.addItem("Always Auto-Load (Saves Undo snapshot)", "auto_always")
        widen_combo(self.combo_policy, min_width=310, popup_width=360)
        pol_h.addWidget(self.combo_policy)

        self.chk_auto_theo = QCheckBox("Compute THEO phases on load")
        pol_h.addWidget(self.chk_auto_theo)

        pol_h.addWidget(QLabel("Auto Filter:"))
        self.combo_auto_filter = QComboBox()
        self.combo_auto_filter.setEditable(True)
        for expr in ("", "BUTTER BP 0.7 2.5 3", "WWSSN_SP", "WWSSN_LP", "WOOD_ANDERSON"):
            self.combo_auto_filter.addItem(expr or "(None)", expr)
        widen_combo(self.combo_auto_filter, min_width=190, popup_width=230)
        pol_h.addWidget(self.combo_auto_filter)
        pol_h.addStretch(1)
        w_form.addRow("Workspace Action:", pol_h)

        dir_h = QHBoxLayout()
        self.edit_cache_dir = QLineEdit(DEFAULT_LIVE_CACHE_DIR)
        self.btn_browse_cache = make_push_button("Browse...", slot=self._browse_cache_dir)
        dir_h.addWidget(self.edit_cache_dir, stretch=1)
        dir_h.addWidget(self.btn_browse_cache)
        w_form.addRow("Live Cache Dir:", dir_h)

        cfg_grid.addWidget(grp_win, 1, 0, 1, 2)
        cfg_root.addLayout(cfg_grid)

        apply_h = QHBoxLayout()
        apply_h.addStretch(1)
        self.btn_reset_defaults = make_push_button("Restore Defaults", slot=self._restore_default_config)
        self.btn_apply_cfg = make_push_button("Apply && Re-evaluate Rules", slot=self._apply_config_from_widgets)
        self.btn_apply_cfg.setMinimumWidth(200)
        apply_h.addWidget(self.btn_reset_defaults)
        apply_h.addWidget(self.btn_apply_cfg)
        cfg_root.addLayout(apply_h)
        cfg_root.addStretch(1)
        return tab_cfg

    def _build_log_tab(self) -> QWidget:
        tab_log = QWidget()
        log_l = QVBoxLayout(tab_log)
        log_l.setContentsMargins(6, 6, 6, 6)
        log_l.setSpacing(6)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFontFamily("Monospace")
        self.log_box.setPlaceholderText("Live stream events, prefetch attempts, and diagnostic messages appear here.")
        log_l.addWidget(self.log_box, stretch=1)
        return tab_log

    def _connect_coordinator(self) -> None:
        self.coordinator.enabledChanged.connect(self._sync_toggle_button)
        self.coordinator.streamStatusChanged.connect(self._on_stream_status)
        self.coordinator.entriesUpdated.connect(self._rebuild_table)
        self.coordinator.countdownTick.connect(self._refresh_countdowns)
        self.coordinator.logMessage.connect(self._append_log)

    # ---- Config Sync ----

    def _load_config_to_widgets(self, cfg: LiveConfig) -> None:
        self._suppress_ui_signals = True
        try:
            idx_src = self.combo_source.findData(cfg.source_key)
            if idx_src >= 0:
                self.combo_source.setCurrentIndex(idx_src)

            self.edit_ref_station.setText(cfg.ref_station)
            self.lbl_ref_coords.setText(f"({cfg.ref_lat:.3f}°, {cfg.ref_lon:.3f}°)")
            self.chk_tiered.setChecked(cfg.use_tiered_filter)
            self.spin_local_deg.setValue(cfg.local_max_deg)
            self.spin_local_mag.setValue(cfg.min_mag_local)
            self.spin_reg_deg.setValue(cfg.regional_max_deg)
            self.spin_reg_mag.setValue(cfg.min_mag_regional)
            self.spin_tele_mag.setValue(cfg.min_mag_tele)
            self.spin_flat_mag.setValue(cfg.flat_min_mag)
            self.spin_max_dist.setValue(cfg.max_distance_deg)
            self.chk_quarry.setChecked(cfg.include_quarry_blasts)

            idx_st = self.combo_strategy.findData(cfg.station_strategy)
            if idx_st >= 0:
                self.combo_strategy.setCurrentIndex(idx_st)
            idx_pr = self.combo_preset.findData(cfg.station_preset)
            if idx_pr >= 0:
                self.combo_preset.setCurrentIndex(idx_pr)
            self.edit_custom_stations.setText(", ".join(cfg.custom_stations))
            self.spin_max_stations.setValue(cfg.max_stations)
            self.spin_epi_radius.setValue(cfg.epicenter_max_radius_deg)

            idx_b = self.combo_band.findData(cfg.channel_band)
            if idx_b >= 0:
                self.combo_band.setCurrentIndex(idx_b)
            comps = set(cfg.components)
            self.chk_z.setChecked("Z" in comps or not comps)
            self.chk_n.setChecked("N" in comps)
            self.chk_e.setChecked("E" in comps)

            self.spin_pre_p.setValue(int(cfg.pre_p_sec))
            self.spin_post_s.setValue(int(cfg.post_s_sec))
            self.spin_max_win.setValue(int(cfg.max_window_sec))
            self.spin_grace.setValue(int(cfg.archive_grace_sec))

            idx_pol = self.combo_policy.findData(cfg.load_policy)
            if idx_pol >= 0:
                self.combo_policy.setCurrentIndex(idx_pol)
            self.chk_auto_theo.setChecked(cfg.auto_theo_phases)
            if cfg.auto_filter_expr:
                self.combo_auto_filter.setCurrentText(cfg.auto_filter_expr)
            else:
                self.combo_auto_filter.setCurrentIndex(0)
            self.edit_cache_dir.setText(cfg.output_base_dir or DEFAULT_LIVE_CACHE_DIR)
        finally:
            self._suppress_ui_signals = False
            self._update_filter_mode_enabled()
            self._update_strategy_enabled()

    def build_config_from_widgets(self) -> LiveConfig:
        """Read current GUI controls into a validated LiveConfig instance."""
        comps = tuple(c for c, chk in (("Z", self.chk_z), ("N", self.chk_n), ("E", self.chk_e)) if chk.isChecked())
        if not comps:
            comps = ("Z",)
        custom_raw = [tok.strip().upper() for tok in self.edit_custom_stations.text().replace(";", ",").split(",") if tok.strip()]

        ref_code = self.edit_ref_station.text().strip().upper() or "GRFO"
        ref_lat, ref_lon = self.coordinator.live_config.ref_lat, self.coordinator.live_config.ref_lon
        st = self.coordinator.shm_config.get_station(ref_code)
        if st is not None and (abs(st.lat) > 1e-6 or abs(st.lon) > 1e-6):
            ref_lat, ref_lon = float(st.lat), float(st.lon)

        filt_txt = self.combo_auto_filter.currentText().strip()
        if filt_txt == "(None)":
            filt_txt = ""

        return LiveConfig(
            source_key=str(self.combo_source.currentData() or "emsc_ws"),
            ref_station=ref_code,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            use_tiered_filter=self.chk_tiered.isChecked(),
            local_max_deg=self.spin_local_deg.value(),
            min_mag_local=self.spin_local_mag.value(),
            regional_max_deg=max(self.spin_local_deg.value() + 0.5, self.spin_reg_deg.value()),
            min_mag_regional=self.spin_reg_mag.value(),
            min_mag_tele=self.spin_tele_mag.value(),
            flat_min_mag=self.spin_flat_mag.value(),
            max_distance_deg=self.spin_max_dist.value(),
            include_quarry_blasts=self.chk_quarry.isChecked(),
            station_strategy=str(self.combo_strategy.currentData() or "observatory"),
            station_preset=str(self.combo_preset.currentData() or "AUTO"),
            custom_stations=tuple(custom_raw),
            max_stations=self.spin_max_stations.value(),
            epicenter_max_radius_deg=self.spin_epi_radius.value(),
            channel_band=str(self.combo_band.currentData() or "BH"),
            components=comps,
            pre_p_sec=float(self.spin_pre_p.value()),
            post_s_sec=float(self.spin_post_s.value()),
            max_window_sec=float(self.spin_max_win.value()),
            archive_grace_sec=float(self.spin_grace.value()),
            load_policy=str(self.combo_policy.currentData() or "notify"),
            auto_theo_phases=self.chk_auto_theo.isChecked(),
            auto_filter_expr=filt_txt,
            output_base_dir=self.edit_cache_dir.text().strip() or DEFAULT_LIVE_CACHE_DIR,
        )

    def _on_flat_mag_edited(self) -> None:
        if not self._suppress_ui_signals and self.chk_tiered.isChecked():
            self.chk_tiered.setChecked(False)

    def _on_tiered_spin_edited(self) -> None:
        if not self._suppress_ui_signals and not self.chk_tiered.isChecked():
            self.chk_tiered.setChecked(True)

    def _on_custom_stations_edited(self, text: str) -> None:
        if self._suppress_ui_signals or not text.strip():
            return
        self._suppress_ui_signals = True
        try:
            idx_obs = self.combo_strategy.findData("observatory")
            if idx_obs >= 0:
                self.combo_strategy.setCurrentIndex(idx_obs)
            idx_cust = self.combo_preset.findData("CUSTOM")
            if idx_cust >= 0:
                self.combo_preset.setCurrentIndex(idx_cust)
        finally:
            self._suppress_ui_signals = False
            self._update_strategy_enabled()

    def _on_epi_radius_edited(self) -> None:
        if self._suppress_ui_signals:
            return
        idx_epi = self.combo_strategy.findData("epicenter")
        if idx_epi >= 0 and self.combo_strategy.currentIndex() != idx_epi:
            self.combo_strategy.setCurrentIndex(idx_epi)

    def _update_filter_mode_enabled(self) -> None:
        tiered = self.chk_tiered.isChecked()
        active_css = "border: 1px solid #3fb950;"
        idle_css = ""
        for w in (self.spin_local_deg, self.spin_local_mag, self.spin_reg_deg, self.spin_reg_mag, self.spin_tele_mag):
            w.setEnabled(True)
            w.setStyleSheet(active_css if tiered else idle_css)
        self.spin_flat_mag.setEnabled(True)
        self.spin_flat_mag.setStyleSheet(idle_css if tiered else active_css)

    def _update_strategy_enabled(self) -> None:
        is_obs = str(self.combo_strategy.currentData()) == "observatory"
        is_custom = is_obs and str(self.combo_preset.currentData()) == "CUSTOM"
        self.combo_preset.setEnabled(True)
        self.edit_custom_stations.setEnabled(True)
        self.spin_epi_radius.setEnabled(True)
        self.edit_custom_stations.setStyleSheet("border: 1px solid #3fb950;" if is_custom else "")
        self.spin_epi_radius.setStyleSheet("border: 1px solid #3fb950;" if not is_obs else "")

    # ---- Top Bar & Rule Handlers ----

    def _validate_and_warn(self, show_popup: bool = True) -> List[str]:
        warnings_list: List[str] = []
        if not any((self.chk_z.isChecked(), self.chk_n.isChecked(), self.chk_e.isChecked())):
            warnings_list.append("No waveform components (Z, N, E) were selected; defaulting to Z.")
            self.chk_z.setChecked(True)
        raw_ref = self.edit_ref_station.text().strip().upper()
        if raw_ref:
            st = self.coordinator.shm_config.stations.get(raw_ref)
            if st is None or (abs(st.lat) <= 1e-6 and abs(st.lon) <= 1e-6):
                warnings_list.append(
                    f"Unknown reference station '{raw_ref}'; keeping coordinates "
                    f"({self.coordinator.live_config.ref_lat:.3f}°, {self.coordinator.live_config.ref_lon:.3f}°)."
                )
        if warnings_list:
            msg = " ".join(warnings_list)
            self.lbl_footer_hint.setStyleSheet("color: #f85149; font-weight: bold;")
            self.lbl_footer_hint.setText(msg)
            self._append_log(f"[Live Warning] {msg}")
            if show_popup:
                QMessageBox.warning(self, "Live Event Configuration", "\n".join(warnings_list))
        else:
            self.lbl_footer_hint.setStyleSheet("")
        return warnings_list

    def _on_live_toggled(self, checked: bool) -> None:
        if self._suppress_ui_signals:
            return
        if checked:
            self._validate_and_warn(show_popup=True)
        self.coordinator.update_config(self.build_config_from_widgets(), reevaluate_existing=False)
        self.coordinator.set_enabled(checked)

    def _sync_toggle_button(self, enabled: bool) -> None:
        self._suppress_ui_signals = True
        try:
            self.btn_toggle_live.setChecked(enabled)
            if enabled:
                self.btn_toggle_live.setText("LIVE: Monitoring ON")
                self.btn_toggle_live.setStyleSheet(
                    "background-color: #1b4d2e; color: #e6ffed; border: 1px solid #3fb950; font-weight: bold;"
                )
            else:
                self.btn_toggle_live.setText("Live Mode: OFF (Start)")
                self.btn_toggle_live.setStyleSheet("")
        finally:
            self._suppress_ui_signals = False

    def _on_source_changed(self) -> None:
        if self._suppress_ui_signals:
            return
        self._apply_config_from_widgets()

    def _on_stream_status(self, state: str, detail: str) -> None:
        dot_color = {
            "connected": "#3fb950",
            "polling": "#58a6ff",
            "connecting": "#d29922",
            "reconnecting": "#f0883e",
            "error": "#f85149",
        }.get(state, "#8b949e")
        clean_detail = "Idle" if detail == "Live Event Mode is OFF" else detail
        self.lbl_stream_status.setText(f"<span style='color:{dot_color};'>●</span> {clean_detail}")

    def _on_replay_24h(self) -> None:
        self._validate_and_warn(show_popup=True)
        self.coordinator.update_config(self.build_config_from_widgets(), reevaluate_existing=False)
        self.coordinator.fetch_recent_catalog(lookback_hours=24.0, context_label="Replay (24h)")

    def _on_open_cache_dir(self) -> None:
        cache_dir = os.path.abspath(os.path.expanduser(self.edit_cache_dir.text().strip() or DEFAULT_LIVE_CACHE_DIR))
        os.makedirs(cache_dir, exist_ok=True)
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(cache_dir))
        if opened:
            self.lbl_footer_hint.setText(f"Opened archive folder: {cache_dir}")
        else:
            self.lbl_footer_hint.setText(f"Archive folder: {cache_dir}")
        self._append_log(f"Live event Mini-SEED archive folder: {cache_dir} (opened={opened})")

    def _browse_cache_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose Live Event Cache Folder",
            os.path.expanduser(self.edit_cache_dir.text() or DEFAULT_LIVE_CACHE_DIR),
        )
        if chosen:
            self.edit_cache_dir.setText(chosen)

    def _restore_default_config(self) -> None:
        self._load_config_to_widgets(LiveConfig())

    def _apply_config_from_widgets(self) -> None:
        self._validate_and_warn(show_popup=True)
        cfg = self.build_config_from_widgets()
        self.lbl_ref_coords.setText(f"({cfg.ref_lat:.3f}°, {cfg.ref_lon:.3f}°)")
        self.coordinator.update_config(cfg, reevaluate_existing=True)
        self._append_log("[Live] Configuration applied and pending events re-evaluated.")

    # ---- Events Table & Row Actions ----

    def _selected_event_id(self) -> Optional[str]:
        row = self.events_table.currentRow()
        if 0 <= row < len(self._row_event_ids):
            return self._row_event_ids[row]
        return None

    def _rebuild_table(self) -> None:
        prev_id = self._selected_event_id()
        eids = list(self.coordinator.ordered_ids)
        self._row_event_ids = eids
        self.events_table.setRowCount(len(eids))

        ready_cnt = 0
        waiting_cnt = 0
        for row, eid in enumerate(eids):
            entry = self.coordinator.entries[eid]
            ev = entry.event
            eva = entry.evaluation
            if entry.state in (STATE_READY, STATE_LOADED):
                ready_cnt += 1
            elif entry.state in (STATE_WAITING, STATE_RETRY_WAIT, STATE_FETCHING):
                waiting_cnt += 1

            dt = datetime.datetime.fromtimestamp(ev.origin_epoch, tz=datetime.timezone.utc)
            origin_txt = dt.strftime("%Y-%m-%d %H:%M:%S")
            mag_txt = f"M{ev.magnitude:.1f} {ev.mag_type}"
            depth_txt = f"{ev.depth_km:.0f} km"
            dist_txt = f"{eva.dist_deg:.1f}° / {eva.baz_deg:.0f}°"
            eta_txt = self.coordinator.countdown_text(eid)
            status_txt = f"[{entry.state}] {entry.status_detail}"

            vals = [origin_txt, mag_txt, depth_txt, eva.region_display, dist_txt, eta_txt, status_txt]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if col in (1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif col == 6:
                    item.setForeground(QColor(_STATE_COLORS.get(entry.state, "#e6edf3")))
                elif col == 3:
                    item.setToolTip(f"{ev.event_id} ({ev.source_feed}) @ {ev.lat:.2f}°, {ev.lon:.2f}°")
                self.events_table.setItem(row, col, item)

        self.lbl_counts.setText(f"{len(eids)} event(s) | {waiting_cnt} scheduled | {ready_cnt} ready/loaded")

        # Restore selection cleanly or highlight newest entry
        if prev_id and prev_id in eids:
            self.events_table.selectRow(eids.index(prev_id))
        elif eids and self.events_table.currentRow() < 0:
            self.events_table.selectRow(0)
        self._update_selected_controls()

    def _refresh_countdowns(self) -> None:
        for row, eid in enumerate(self._row_event_ids):
            entry = self.coordinator.entries.get(eid)
            if entry is None:
                continue
            item_eta = self.events_table.item(row, 5)
            item_st = self.events_table.item(row, 6)
            if item_eta is not None:
                item_eta.setText(self.coordinator.countdown_text(eid))
            if item_st is not None:
                item_st.setText(f"[{entry.state}] {entry.status_detail}")

    def _update_selected_controls(self) -> None:
        eid = self._selected_event_id()
        entry = self.coordinator.entries.get(eid) if eid else None
        if entry is None:
            self.btn_load_event.setEnabled(False)
            self.btn_fetch_now.setEnabled(False)
            self.btn_show_map.setEnabled(False)
            self.btn_open_bgr.setEnabled(False)
            self.btn_ignore_event.setEnabled(False)
            self.lbl_event_preview.setText("Select an event row to inspect station plan, UTC window, and actions.")
            return

        has_traces = bool(entry.fetched_traces)
        self.btn_load_event.setEnabled(has_traces)
        self.btn_fetch_now.setEnabled(entry.state != STATE_FETCHING)
        self.btn_show_map.setEnabled(True)
        self.btn_open_bgr.setEnabled(True)
        self.btn_ignore_event.setEnabled(entry.state not in (STATE_IGNORED, STATE_FETCHING))

        ev = entry.event
        eva = entry.evaluation
        if entry.plan is not None:
            stas = ", ".join(f"{t.network}.{t.station}" for t in entry.plan.targets)
            s_dt = datetime.datetime.fromtimestamp(entry.plan.start_epoch, tz=datetime.timezone.utc).strftime("%H:%M:%S")
            e_dt = datetime.datetime.fromtimestamp(entry.plan.end_epoch, tz=datetime.timezone.utc).strftime("%H:%M:%S")
            win_info = f"Window {s_dt}–{e_dt} UTC ({entry.plan.duration_sec:.0f}s) | Stations: {stas}"
        else:
            win_info = eva.reason
        self.lbl_event_preview.setText(
            f"<b>{ev.event_id}</b> ({ev.source_feed}) — {ev.lat:.2f}°, {ev.lon:.2f}°, z={ev.depth_km:.1f} km — {win_info}"
        )

    def _on_load_selected(self) -> None:
        eid = self._selected_event_id()
        if eid:
            self.coordinator.request_load_event(eid, auto_triggered=False)

    def _on_fetch_selected(self) -> None:
        eid = self._selected_event_id()
        if eid:
            self.coordinator.force_fetch_event(eid)

    def _on_show_on_map(self) -> None:
        eid = self._selected_event_id()
        if eid:
            self.showOnMapRequested.emit(eid)

    def _on_open_in_bgr(self) -> None:
        eid = self._selected_event_id()
        entry = self.coordinator.entries.get(eid) if eid else None
        if entry is not None:
            self.openInBGRRequested.emit(entry)

    def _on_ignore_selected(self) -> None:
        eid = self._selected_event_id()
        if eid:
            self.coordinator.ignore_event(eid)

    def _append_log(self, msg: str) -> None:
        if msg:
            self.log_box.append(msg)
            self._log_count += 1
            self.tabs.setTabText(2, f"Activity Log ({self._log_count})")
