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
from shm_app.gui.dialog_seedlink import SeedLinkStreamDialog
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault
from shm_app.io.ascii_io import read_ascii_file
from shm_app.io.gse import read_gse_file
from shm_app.io.mseed import read_mseed_file
from shm_app.io.qfile import read_qfile
from shm_app.io.sfd import read_from_sfd


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
        mseed_ex_dir = os.path.join(self.config.data_examples_dir, "mseed")
        grf_gse_path = os.path.join(self.config.data_examples_dir, "gse", "GRF_031102_0225.GSE")
        grsn1_gse_path = os.path.join(self.config.data_examples_dir, "gse", "GRSN_031002_2334.GSE")
        grsn2_gse_path = os.path.join(self.config.data_examples_dir, "gse", "GRSN_031102_0224.GSE")
        missing_ex_tip = "Requires external SHM_ROOT data-examples/ (not found)"

        self.btn_ex_mseed = QPushButton("Load GRSN+GRF Mini-SEED (02-Nov-2003)")
        self.btn_ex_mseed.clicked.connect(self._load_example_mseed)
        self.btn_ex_grf_gse = QPushButton("Load GRF Array GSE (02-Nov-2003 02:25)")
        self.btn_ex_grf_gse.clicked.connect(lambda: self._load_example_gse("GRF_031102_0225.GSE"))
        self.btn_ex_grsn1 = QPushButton("Load GRSN GSE (02-Oct-2003 23:34)")
        self.btn_ex_grsn1.clicked.connect(lambda: self._load_example_gse("GRSN_031002_2334.GSE"))
        self.btn_ex_grsn2 = QPushButton("Load GRSN GSE (02-Nov-2003 02:24)")
        self.btn_ex_grsn2.clicked.connect(lambda: self._load_example_gse("GRSN_031102_0224.GSE"))

        ex_targets = (
            (self.btn_ex_mseed, os.path.isdir(mseed_ex_dir)),
            (self.btn_ex_grf_gse, os.path.isfile(grf_gse_path)),
            (self.btn_ex_grsn1, os.path.isfile(grsn1_gse_path)),
            (self.btn_ex_grsn2, os.path.isfile(grsn2_gse_path)),
        )
        for btn, available in ex_targets:
            btn.setEnabled(available)
            if not available:
                btn.setToolTip(missing_ex_tip)
        if not any(avail for _, avail in ex_targets):
            grp_ex.setTitle("Bundled SHM Example Datasets (Requires external SHM_ROOT data-examples/ — not found)")

        ex_grid.addWidget(self.btn_ex_mseed, 0, 0)
        ex_grid.addWidget(self.btn_ex_grf_gse, 0, 1)
        ex_grid.addWidget(self.btn_ex_grsn1, 1, 0)
        ex_grid.addWidget(self.btn_ex_grsn2, 1, 1)
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
        default_src = mseed_ex_dir if os.path.isdir(mseed_ex_dir) else ""
        self.path_edit = QLineEdit(default_src)
        self.path_edit.setPlaceholderText("Select a waveform file (.GSE, .mseed, .sfd, .QHD, .asc) or directory...")
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
        self.chk_full_dur.toggled.connect(self._on_full_duration_toggled)
        self.chk_full_dur.setChecked(True)
        self._on_full_duration_toggled(True)
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

    def _on_full_duration_toggled(self, full_dur: bool) -> None:
        self.start_edit.setEnabled(not full_dur)
        self.len_spin.setEnabled(not full_dur)

    def _default_browse_dir(self) -> str:
        if os.path.isdir(self.config.data_examples_dir):
            return self.config.data_examples_dir
        cwd = os.getcwd()
        if os.path.isdir(cwd):
            return cwd
        return os.path.expanduser("~")

    def _browse_file(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Waveform File",
            self._default_browse_dir(),
            "All Seismic Files (*.bhz *.bhn *.bhe *.mseed *.sfd *.GSE *.gse *.QHD *.qhd *.asc *.txt);;All Files (*)",
        )
        if fpath:
            self.path_edit.setText(fpath)

    def _browse_dir(self) -> None:
        dpath = QFileDialog.getExistingDirectory(self, "Select Waveform Directory", self._default_browse_dir())
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

    def _read_single_file(
        self,
        filepath: str,
        t_start: Optional[float],
        dur: Optional[float],
        sfd_streams: List[str],
    ) -> List[Trace]:
        low = filepath.lower()
        if low.endswith(".sfd"):
            return read_from_sfd(filepath, streams=sfd_streams, t_start=t_start, duration=dur, config=self.config)
        if low.endswith(".gse"):
            return read_gse_file(filepath, config=self.config)
        if low.endswith(".qhd") or low.endswith(".qbn"):
            return read_qfile(filepath, config=self.config)
        if low.endswith(".asc") or low.endswith(".txt"):
            return read_ascii_file(filepath, config=self.config)
        return read_mseed_file(filepath, t_start=t_start, duration=dur, config=self.config)

    def _execute_read(self) -> None:
        self.append_mode = self.chk_keep.isChecked()
        src = self.path_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "Read Waveforms", "Please enter or browse to a waveform file or directory.")
            return
        if not os.path.exists(src):
            QMessageBox.warning(self, "Read Waveforms", f"Waveform source not found:\n{src}")
            return

        selected_comps = self.selected_components()
        if not selected_comps:
            QMessageBox.warning(self, "Read Waveforms", "Please select at least one component (Z, N, E).")
            return
        band = self.chan_combo.currentText().strip().upper()
        sfd_streams = self._sfd_stream_patterns(band, selected_comps)

        unchecked_stations = {st for st, cb in self.station_checks.items() if not cb.isChecked()}
        try:
            t_start = None if self.chk_full_dur.isChecked() else parse_sh_time(self.start_edit.text())
        except Exception as exc:
            QMessageBox.warning(self, "Read Waveforms", f"Invalid start time:\n{exc}")
            return
        dur = None if self.chk_full_dur.isChecked() else self.len_spin.value()

        traces: List[Trace] = []
        dir_errors: List[str] = []
        try:
            if os.path.isdir(src):
                sfd_path = os.path.join(src, "sfdlist.sfd")
                if os.path.isfile(sfd_path):
                    traces = read_from_sfd(
                        sfd_path, streams=sfd_streams, t_start=t_start, duration=dur, config=self.config
                    )
                else:
                    for fn in sorted(os.listdir(src)):
                        fp = os.path.join(src, fn)
                        low = fn.lower()
                        if not os.path.isfile(fp) or fn.startswith(".") or low.endswith((".sfd", ".qbn")):
                            continue
                        try:
                            traces.extend(self._read_single_file(fp, t_start, dur, sfd_streams))
                        except Exception as exc:
                            dir_errors.append(f"{fn}: {exc}")
            else:
                traces = self._read_single_file(src, t_start, dur, sfd_streams)
        except Exception as exc:
            QMessageBox.warning(self, "Read Waveforms", f"Failed to read waveform data:\n{exc}")
            return

        if not traces and dir_errors:
            QMessageBox.warning(
                self,
                "Read Waveforms",
                f"No waveforms could be read from directory:\n{src}\n\n{dir_errors[0]}",
            )
            return

        if unchecked_stations:
            traces = [t for t in traces if t.station.upper() not in unchecked_stations]
        traces = [t for t in traces if self._matches_stream_filter(t, band, selected_comps)]

        if not traces:
            QMessageBox.warning(
                self,
                "Read Waveforms",
                "No traces matched the selected station, channel, or component filters.",
            )
            return

        self.loaded_traces = traces
        self.accept()

    def _load_stationxml_file(self) -> None:
        start_dir = self.config.shm_root if os.path.isdir(self.config.shm_root) else self._default_browse_dir()
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Load FDSN StationXML",
            start_dir,
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

