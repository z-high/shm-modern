"""Waveform data import dialog for modernized SHM."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shm_app.core.config_station import (
    GRF_STATIONS,
    SHM_READ_REGIONAL_STATIONS,
    SHMConfig,
    get_config,
)
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault
from shm_app.io.gse import read_gse_file
from shm_app.io.mseed import read_from_sfd, read_mseed_file
from shm_app.io.qfile import read_ascii_file, read_qfile


class ReadDataDialog(QDialog):
    """SHM Read Sparse / GSE / Q-File / Event Dialog with Station Matrix and Example Dataset Quick-Loaders."""

    def __init__(self, config: Optional[SHMConfig] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Read Waveforms (Mini-SEED / SFD / GSE / Q-File)")
        self.resize(780, 560)
        self.config = config or get_config()
        self.loaded_traces: List[Trace] = []
        self.append_mode: bool = False

        layout = QVBoxLayout(self)

        # Quick-load bar for bundled SHM example datasets
        grp_ex = QGroupBox("Bundled SHM Example Datasets (One-Click Load)")
        ex_grid = QGridLayout(grp_ex)
        ex_grid.setSpacing(6)
        btn_ex_mseed = QPushButton("Load GRSN+GRF Mini-SEED (02-Nov-2003)")
        btn_ex_mseed.clicked.connect(self._load_example_mseed)
        btn_ex_grf_gse = QPushButton("Load GRF Array GSE (02-Nov-2003 02:25)")
        btn_ex_grf_gse.clicked.connect(lambda: self._load_example_gse("GRF_031102_0225.GSE"))
        btn_ex_grsn1 = QPushButton("Load GRSN GSE (02-Oct-2003 23:34)")
        btn_ex_grsn1.clicked.connect(lambda: self._load_example_gse("GRSN_031002_2334.GSE"))
        btn_ex_grsn2 = QPushButton("Load GRSN GSE (02-Nov-2003 02:24)")
        btn_ex_grsn2.clicked.connect(lambda: self._load_example_gse("GRSN_031102_0224.GSE"))
        ex_grid.addWidget(btn_ex_mseed, 0, 0)
        ex_grid.addWidget(btn_ex_grf_gse, 0, 1)
        ex_grid.addWidget(btn_ex_grsn1, 1, 0)
        ex_grid.addWidget(btn_ex_grsn2, 1, 1)
        layout.addWidget(grp_ex)

        # Station Matrix
        grp_st = QGroupBox("Station Selection Matrix (GRF Array && GRSN Regional Network)")
        st_v = QVBoxLayout(grp_st)
        st_grid = QGridLayout()
        st_grid.setSpacing(4)

        default_stations = [*GRF_STATIONS, *SHM_READ_REGIONAL_STATIONS]
        grf_set = set(GRF_STATIONS)
        self.station_checks: Dict[str, QCheckBox] = {}
        for idx, st in enumerate(default_stations):
            cb = QCheckBox(st.upper())
            cb.setChecked(True)
            self.station_checks[st.upper()] = cb
            st_grid.addWidget(cb, idx // 9, idx % 9)
        st_v.addLayout(st_grid)

        st_btns = QHBoxLayout()
        btn_all_on = QPushButton("All On")
        btn_all_on.clicked.connect(lambda: [c.setChecked(True) for c in self.station_checks.values()])
        btn_all_off = QPushButton("All Off")
        btn_all_off.clicked.connect(lambda: [c.setChecked(False) for c in self.station_checks.values()])
        btn_grf_only = QPushButton("GRF Array Only")
        btn_grf_only.clicked.connect(
            lambda: [c.setChecked(k in grf_set) for k, c in self.station_checks.items()]
        )
        btn_grsn_only = QPushButton("GRSN Only")
        btn_grsn_only.clicked.connect(
            lambda: [c.setChecked(k not in grf_set) for k, c in self.station_checks.items()]
        )
        st_btns.addWidget(btn_all_on)
        st_btns.addWidget(btn_all_off)
        st_btns.addWidget(btn_grf_only)
        st_btns.addWidget(btn_grsn_only)
        st_btns.addStretch(1)
        st_v.addLayout(st_btns)
        layout.addWidget(grp_st)

        # Channel, Component, Time Window, and File Path
        grp_opt = QGroupBox("Stream && Time Window Parameters")
        form = QFormLayout(grp_opt)

        path_h = QHBoxLayout()
        mseed_default = os.path.join(self.config.data_examples_dir, "mseed")
        self.path_edit = QLineEdit(mseed_default)
        btn_browse_file = QPushButton("Browse File...")
        btn_browse_file.clicked.connect(self._browse_file)
        btn_browse_dir = QPushButton("Browse Dir...")
        btn_browse_dir.clicked.connect(self._browse_dir)
        path_h.addWidget(self.path_edit)
        path_h.addWidget(btn_browse_file)
        path_h.addWidget(btn_browse_dir)
        form.addRow("Data Source:", path_h)

        chan_h = QHBoxLayout()
        self.chan_combo = QComboBox()
        self.chan_combo.addItems(["BH", "HH", "LH", "SH", "VH", "ALL"])
        chan_h.addWidget(QLabel("Channel:"))
        chan_h.addWidget(self.chan_combo)
        chan_h.addSpacing(20)
        chan_h.addWidget(QLabel("Components:"))
        self.chk_z = QCheckBox("Z")
        self.chk_z.setChecked(True)
        self.chk_n = QCheckBox("N")
        self.chk_e = QCheckBox("E")
        chan_h.addWidget(self.chk_z)
        chan_h.addWidget(self.chk_n)
        chan_h.addWidget(self.chk_e)
        chan_h.addStretch(1)
        form.addRow("Streams:", chan_h)

        time_grid = QGridLayout()
        time_grid.setSpacing(6)
        self.start_edit = QLineEdit("02-NOV-2003_02:23:54.000")
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(1.0, 86400.0)
        self.len_spin.setValue(600.0)
        self.len_spin.setSuffix(" s")
        self.len_spin.setMinimumHeight(30)
        self.chk_full_dur = QCheckBox("Read Full File Duration")
        self.chk_full_dur.setChecked(True)
        time_grid.addWidget(QLabel("Start:"), 0, 0)
        time_grid.addWidget(self.start_edit, 0, 1)
        time_grid.addWidget(QLabel("Length:"), 0, 2)
        time_grid.addWidget(self.len_spin, 0, 3)
        time_grid.addWidget(self.chk_full_dur, 1, 0, 1, 4)
        time_grid.setColumnStretch(1, 1)
        form.addRow("Time Window:", time_grid)
        layout.addWidget(grp_opt)

        # Dialog action buttons
        act_h = QHBoxLayout()
        self.chk_keep = QCheckBox("Append to Existing Traces (Read Another)")
        btn_sxml = QPushButton("Load StationXML...")
        btn_sxml.clicked.connect(self._load_stationxml_file)
        act_h.addWidget(self.chk_keep)
        act_h.addWidget(btn_sxml)
        act_h.addStretch(1)
        btn_read = QPushButton("Read New / Selected")
        btn_read.clicked.connect(self._execute_read)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        act_h.addWidget(btn_read)
        act_h.addWidget(btn_cancel)
        layout.addLayout(act_h)

        register_tutorial_help(self.path_edit, "Waveform Source Path", "dlg_read_stations", "", "Path to GSE, Q-file, Mini-SEED file, or SFD directory")
        register_tutorial_help(self.chan_combo, "Channel Prefix", "dlg_read_channels", "", "2-character seismic stream prefix (BH, HH, LH, ...)")
        for comp_cb in (self.chk_z, self.chk_n, self.chk_e):
            register_tutorial_help(comp_cb, f"Component {comp_cb.text()}", "dlg_read_channels", "", f"Include {comp_cb.text()} component traces")
        register_tutorial_help(self.start_edit, "Read Start Time (UTC)", "dlg_read_time", "", "Start time in SH UTC format")
        register_tutorial_help(self.len_spin, "Read Window Length", "dlg_read_time", "", "Window duration in seconds")
        register_tutorial_help(self.chk_full_dur, "Read Full File Duration", "dlg_read_time", "", "Ignore fixed length and load the whole file")
        register_tutorial_help(self.chk_keep, "Append Mode", "dlg_read_append", "", "Append read traces to current workspace")
        disable_button_autodefault(self)
        apply_c_locale(self)

    def _browse_file(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Waveform File",
            self.config.data_examples_dir,
            "All Seismic Files (*.bhz *.bhn *.bhe *.mseed *.sfd *.GSE *.gse *.QHD *.qhd *.asc *.txt);;All Files (*)",
        )
        if fpath:
            self.path_edit.setText(fpath)

    def _browse_dir(self) -> None:
        dpath = QFileDialog.getExistingDirectory(self, "Select Waveform Directory", self.config.data_examples_dir)
        if dpath:
            self.path_edit.setText(dpath)

    def _load_example_mseed(self) -> None:
        self.path_edit.setText(os.path.join(self.config.data_examples_dir, "mseed"))
        self._execute_read()

    def _load_example_gse(self, fname: str) -> None:
        fpath = os.path.join(self.config.data_examples_dir, "gse", fname)
        self.path_edit.setText(fpath)
        self._execute_read()

    def selected_components(self) -> List[str]:
        """Return checked 1-char component codes (subset of Z, N, E)."""
        return [c for c, cb in (("Z", self.chk_z), ("N", self.chk_n), ("E", self.chk_e)) if cb.isChecked()]

    def _sfd_stream_patterns(self, band: str, comps: List[str]) -> List[str]:
        b = "*" if band in ("", "*", "ALL") else band.lower()
        return [f"*-{b}-{c.lower()}" for c in comps]

    @staticmethod
    def _matches_stream_filter(tr: Trace, band: str, comps: List[str]) -> bool:
        if (tr.comp or "Z").upper() not in comps:
            return False
        if band not in ("", "*", "ALL"):
            tr_band = f"{tr.chan1}{tr.chan2}".upper()
            if tr_band != band:
                return False
        return True

    def _execute_read(self) -> None:
        self.append_mode = self.chk_keep.isChecked()
        src = self.path_edit.text().strip()
        if not src:
            return

        selected_comps = self.selected_components()
        if not selected_comps:
            QMessageBox.warning(self, "Read Waveforms", "Please select at least one component (Z, N, E).")
            return
        band = self.chan_combo.currentText().strip().upper()
        sfd_streams = self._sfd_stream_patterns(band, selected_comps)

        selected_stations = {st for st, cb in self.station_checks.items() if cb.isChecked()}
        t_start = None if self.chk_full_dur.isChecked() else parse_sh_time(self.start_edit.text())
        dur = None if self.chk_full_dur.isChecked() else self.len_spin.value()

        traces: List[Trace] = []
        if os.path.isdir(src):
            sfd_path = os.path.join(src, "sfdlist.sfd")
            if os.path.isfile(sfd_path):
                traces = read_from_sfd(sfd_path, streams=sfd_streams, t_start=t_start, duration=dur, config=self.config)
            else:
                for fn in sorted(os.listdir(src)):
                    fp = os.path.join(src, fn)
                    if os.path.isfile(fp) and not fn.endswith(".sfd"):
                        traces.extend(read_mseed_file(fp, t_start=t_start, duration=dur, config=self.config))
        elif src.lower().endswith(".sfd"):
            traces = read_from_sfd(src, streams=sfd_streams, t_start=t_start, duration=dur, config=self.config)
        elif src.lower().endswith(".gse"):
            traces = read_gse_file(src, config=self.config)
        elif src.lower().endswith(".qhd") or src.lower().endswith(".qbn"):
            traces = read_qfile(src, config=self.config)
        elif src.lower().endswith(".asc") or src.lower().endswith(".txt"):
            traces = read_ascii_file(src, config=self.config)
        else:
            traces = read_mseed_file(src, t_start=t_start, duration=dur, config=self.config)

        if len(selected_stations) < len(self.station_checks):
            traces = [t for t in traces if t.station.upper() in selected_stations]
        traces = [t for t in traces if self._matches_stream_filter(t, band, selected_comps)]

        self.loaded_traces = traces
        self.accept()

    def _load_stationxml_file(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Load FDSN StationXML",
            self.config.shm_root,
            "StationXML Files (*.xml *.stationxml);;All Files (*)",
        )
        if not fpath:
            return
        try:
            inv = self.config.load_stationxml(fpath)
            n_sta = len(inv.stations)
            n_cha = sum(len(s.channels) for s in inv.stations)
            QMessageBox.information(
                self,
                "StationXML Imported",
                f"Loaded StationXML metadata: {n_sta} station(s), {n_cha} channel(s).",
            )
        except Exception as exc:
            QMessageBox.warning(self, "StationXML Error", str(exc))


class SeedLinkStreamDialog(QDialog):
    """Configuration and control dialog for native SeedLink v3.1 TCP streaming (M-21)."""

    def __init__(self, live_coordinator, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.live_coordinator = live_coordinator
        self.setWindowTitle("SeedLink v3.1 Live Waveform Stream")
        self.resize(520, 260)

        layout = QVBoxLayout(self)

        grp = QGroupBox("SeedLink Server && Stream Selection")
        form = QFormLayout(grp)

        self.host_edit = QLineEdit("geofon.gfz-potsdam.de")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(18000)

        self.streams_edit = QLineEdit("GR_GRA1:BHZ,GR_GRB1:BHZ")
        self.streams_edit.setPlaceholderText("Comma-separated NET_STA:LOCCHA (e.g. GR_GRA1:BHZ)")

        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(200, 500000)
        self.max_samples_spin.setValue(36000)
        self.max_samples_spin.setSingleStep(1000)

        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("Streams:", self.streams_edit)
        form.addRow("Buffer Samples / Stream:", self.max_samples_spin)
        layout.addWidget(grp)

        self.status_lbl = QLabel(f"Status: {getattr(live_coordinator, 'seedlink_status', 'idle')}")
        self.status_lbl.setStyleSheet("color: #58a6ff; font-family: Monospace;")
        layout.addWidget(self.status_lbl)

        btn_h = QHBoxLayout()
        self.btn_start = QPushButton("Start Stream")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop Stream")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_h.addStretch(1)
        btn_h.addWidget(self.btn_start)
        btn_h.addWidget(self.btn_stop)
        btn_h.addWidget(btn_close)
        layout.addLayout(btn_h)

        if hasattr(live_coordinator, "seedlinkStatusChanged"):
            live_coordinator.seedlinkStatusChanged.connect(self._on_status_changed)

        self._sync_buttons()
        disable_button_autodefault(self)
        apply_c_locale(self)

    def parsed_streams(self) -> List[tuple]:
        out: List[tuple] = []
        for raw in self.streams_edit.text().split(","):
            tok = raw.strip()
            if not tok:
                continue
            if ":" in tok:
                left, sel = tok.split(":", 1)
                parts = [p for p in left.replace(".", "_").split("_") if p]
                if len(parts) >= 2:
                    sel_norm = sel.strip().upper()
                    if "." not in sel_norm:
                        sel_norm = f"{sel_norm}.D"
                    out.append((parts[0].upper(), parts[1].upper(), sel_norm))
                    continue
            if "." in tok:
                parts = tok.split(".")
                if len(parts) >= 4:
                    loc = parts[2].strip().upper()
                    cha = parts[3].strip().upper() or "BHZ"
                    sel = f"{loc}{cha}.D" if loc else f"{cha}.D"
                    out.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                    continue
                if len(parts) >= 2:
                    sel = f"{parts[2].strip().upper()}.D" if len(parts) == 3 and parts[2].strip() else "BHZ.D"
                    out.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                    continue
            if "_" in tok:
                parts = [p for p in tok.split("_") if p]
                if len(parts) >= 2:
                    sel = f"{parts[2].upper()}.D" if len(parts) >= 3 else "BHZ.D"
                    out.append((parts[0].upper(), parts[1].upper(), sel))
        return out

    def _on_start(self) -> None:
        host = self.host_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "SeedLink", "Please enter a SeedLink host.")
            return
        streams = self.parsed_streams()
        self.live_coordinator.start_seedlink(
            host=host,
            port=int(self.port_spin.value()),
            streams=streams or None,
            max_samples_per_stream=int(self.max_samples_spin.value()),
        )
        self._sync_buttons()

    def _on_stop(self) -> None:
        self.live_coordinator.stop_seedlink()
        self._sync_buttons()

    def _on_status_changed(self, status: str) -> None:
        self.status_lbl.setText(f"Status: {status}")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        running = bool(getattr(self.live_coordinator, "is_seedlink_running", False))
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

