"""Network location dialog for modernized SHM."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import AnalysisParamSet, PhasePick
from shm_app.gui.dialog_event_summary import EventSummaryDialog
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault
from shm_app.seismology.array_loc import locate_network_geiger
from shm_app.seismology.geodesy import sphere_diff


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
        self._excluded_stations: List[str] = []
        self._last_outlier_stations: List[str] = []

        layout = QVBoxLayout(self)

        ctrl_h = QHBoxLayout()
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 700.0)
        self.depth_spin.setValue(initial_depth)
        self.depth_spin.setSuffix(" km")
        self.chk_free_depth = QCheckBox("Solve for Free Depth")
        self.chk_robust = QCheckBox("Robust (Outlier-Resistant)")
        self.chk_robust.setChecked(True)
        btn_solve = QPushButton("Run Iterative Location")
        btn_solve.clicked.connect(self.run_location)

        ctrl_h.addWidget(QLabel("Starting / Fixed Depth:"))
        ctrl_h.addWidget(self.depth_spin)
        ctrl_h.addWidget(self.chk_free_depth)
        ctrl_h.addWidget(self.chk_robust)
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
            if col == 3:
                hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            else:
                hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self._min_onset_width = (
            max(
                self.table.fontMetrics().horizontalAdvance("02-NOV-2003_02:25:30.123"),
                self.table.fontMetrics().horizontalAdvance("Onset Time"),
            )
            + 24
        )
        self.table.setColumnWidth(3, self._min_onset_width)
        layout.addWidget(self.table)

        bot_h = QHBoxLayout()
        self.btn_exclude_outliers = QPushButton("Exclude Outliers")
        self.btn_exclude_outliers.setEnabled(False)
        self.btn_exclude_outliers.clicked.connect(self._on_exclude_outliers)
        self.btn_reset_exclusions = QPushButton("Reset Exclusions")
        self.btn_reset_exclusions.setEnabled(False)
        self.btn_reset_exclusions.clicked.connect(self._on_reset_exclusions)
        btn_adopt = QPushButton("Adopt Solution to Parameter Box")
        btn_adopt.clicked.connect(self._on_adopt)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_h.addWidget(self.btn_exclude_outliers)
        bot_h.addWidget(self.btn_reset_exclusions)
        bot_h.addStretch(1)
        bot_h.addWidget(btn_adopt)
        bot_h.addWidget(btn_close)
        layout.addLayout(bot_h)

        register_tutorial_help(self.depth_spin, "Initial / Fixed Focal Depth", "dlg_netloc_depth", "", "Trial focal depth in km")
        register_tutorial_help(self.chk_free_depth, "Free-Depth Inversion", "dlg_netloc_depth", "", "Invert jointly for depth as well as epicenter and origin time")
        register_tutorial_help(self.chk_robust, "Robust Huber IRLS", "dlg_netloc_robust", "", "Downweight gross timing outliers automatically during iterative inversion")
        disable_button_autodefault(self)
        apply_c_locale(self)

        if self.picks:
            self.run_location()

    def _on_exclude_outliers(self) -> None:
        added = False
        for st in self._last_outlier_stations:
            st_u = st.upper()
            if st_u not in self._excluded_stations:
                self._excluded_stations.append(st_u)
                added = True
        if added:
            self.run_location()

    def _on_reset_exclusions(self) -> None:
        if self._excluded_stations:
            self._excluded_stations.clear()
            self.run_location()

    def run_location(self) -> None:
        if not self.picks:
            self.result_param = None
            self._last_outlier_stations = []
            self.btn_exclude_outliers.setEnabled(False)
            self.btn_reset_exclusions.setEnabled(bool(self._excluded_stations))
            self.table.setRowCount(0)
            self.summary_lbl.setStyleSheet("color: #f85149; font-weight: bold;")
            self.summary_lbl.setText("No phase picks available. Pick phases on traces first.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            param, station_resids, diag = locate_network_geiger(
                self.picks,
                depth_km=self.depth_spin.value(),
                free_depth=self.chk_free_depth.isChecked(),
                config=self.config,
                excluded_stations=self._excluded_stations or None,
                return_diagnostics=True,
                robust=self.chk_robust.isChecked(),
            )
            err_msg = (diag.context_data or {}).get("error") if diag is not None else None
            if err_msg or (diag is not None and not diag.converged and param.lat == 0.0 and param.lon == 0.0):
                self.result_param = None
                self._last_outlier_stations = []
                self.btn_exclude_outliers.setEnabled(False)
                self.btn_reset_exclusions.setEnabled(bool(self._excluded_stations))
                self.table.setRowCount(0)
                self.summary_lbl.setStyleSheet("color: #f85149; font-weight: bold;")
                self.summary_lbl.setText(f"Location Error: {err_msg or 'Location did not converge.'}")
                return
            self.result_param = param
            self.diagnosticAvailable.emit(diag)
            excluded_set = set(self._excluded_stations)
            outlier_keys = {
                (e.station.upper(), e.phase)
                for e in (diag.residuals if diag is not None else [])
                if e.is_outlier and e.station.upper() not in excluded_set
            }
            self._last_outlier_stations = sorted({st for st, _ph in outlier_keys})
            self.btn_exclude_outliers.setEnabled(bool(self._last_outlier_stations))
            self.btn_reset_exclusions.setEnabled(bool(self._excluded_stations))

            rmse_txt = f"RMSE: {diag.rmse:.2f} s" if diag is not None else ""
            out_txt = f"   |   Outliers: {len(self._last_outlier_stations)}" if self._last_outlier_stations else ""
            excl_txt = f"   |   Excluded: {', '.join(self._excluded_stations)}" if self._excluded_stations else ""
            self.summary_lbl.setStyleSheet("color: #3fb950; font-weight: bold;")
            self.summary_lbl.setText(
                f"Epicenter: {param.lat:+.3f} N, {param.lon:+.3f} E   |   Depth: {param.depth:.1f} km   |   "
                f"Origin: {param.origin}\nRegion: [{param.region_id}] {param.region_name}   |   {rmse_txt}{out_txt}{excl_txt}"
            )
            self.table.setRowCount(len(self.picks))
            for r_i, pk in enumerate(self.picks):
                st = self.config.get_station(pk.station)
                dist, _, _ = sphere_diff(st.lat, st.lon, param.lat, param.lon)
                is_excl = pk.station.upper() in excluded_set
                is_out = (pk.station.upper(), pk.name) in outlier_keys
                w_txt = "0 (excl)" if is_excl else str(pk.weight)
                res_txt = f"{pk.resid:+.3f}*" if is_out else f"{pk.resid:+.3f}"
                vals = [
                    pk.station,
                    pk.comp or "-",
                    pk.name,
                    pk.onset,
                    w_txt,
                    f"{dist:.2f}",
                    res_txt,
                ]
                for c_i, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    if is_excl:
                        item.setForeground(QColor("#8b949e"))
                    elif is_out:
                        item.setForeground(QColor("#ff7b72"))
                    self.table.setItem(r_i, c_i, item)
            self.table.resizeColumnToContents(3)
            if self.table.columnWidth(3) < self._min_onset_width:
                self.table.setColumnWidth(3, self._min_onset_width)
        except Exception as exc:
            self.result_param = None
            self._last_outlier_stations = []
            self.btn_exclude_outliers.setEnabled(False)
            self.btn_reset_exclusions.setEnabled(bool(self._excluded_stations))
            self.table.setRowCount(0)
            self.summary_lbl.setStyleSheet("color: #f85149; font-weight: bold;")
            self.summary_lbl.setText(f"Location Error: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_adopt(self) -> None:
        if self.result_param is not None:
            self.adoptLocation.emit(self.result_param)
            self.accept()
