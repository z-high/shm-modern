"""Filter, array-analysis (FK, Vespagram), spectral, and polarization dialogs for modernized SHM."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import FitDiagnosticResult
from shm_app.core.models import Trace
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import add_dark_colorbar, apply_c_locale, disable_button_autodefault
from shm_app.processing.filters import (
    STANDARD_SIMULATION_FILTERS,
    evaluate_stage_freq,
    load_named_filter,
)
from shm_app.processing.polarization import (
    analyze_polarization_diagnostic,
    compute_mdir,
    rotate_3d_lqt,
)
from shm_app.processing.spectrum import (
    build_brune_fit_diagnostic,
    compute_fft_spectrum,
    compute_mem_spectrum,
    compute_spectrogram,
    compute_welch_power_spectrum,
    fit_log_log_spectrum,
    invert_brune_spectrum,
)
from shm_app.seismology.array_loc import (
    SlownessCalibrationTable,
    build_fk_fit_diagnostic,
    compute_array_response_function,
    compute_beam_diagnostics,
    compute_fk_analysis,
    compute_vespagram,
    load_calibration_for_array,
)


class FilterDialog(QDialog):
    """Interactive Filter Designer & Simulation Filter Selector with live Bode response plot."""

    def __init__(self, dt: float = 0.05, config: Optional[SHMConfig] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filter Waveforms (Simulation & Butterworth Designer)")
        self.resize(780, 560)
        self.dt = dt if dt > 0 else 0.05
        self.config = config or get_config()
        self.selected_filter_expr: str = "WWSSN_SP"
        self.selected_mode: str = "AUTO"
        self.use_lpc: bool = False
        self.demean_first: bool = True
        self.autocut: bool = False
        self.zero_phase: bool = False

        layout = QVBoxLayout(self)

        # Tabs: 1) Standard Simulation Filters  2) Butterworth Designer  3) Filter File Browser
        self.tabs = QTabWidget()

        # Tab 1: Simulation Filters
        tab_sim = QWidget()
        sim_l = QVBoxLayout(tab_sim)
        sim_grid = QGridLayout()
        self.sim_combo = QComboBox()
        self.sim_combo.addItems(["NONE"] + STANDARD_SIMULATION_FILTERS)
        self.sim_combo.setCurrentText("WWSSN_SP")
        self.sim_combo.currentTextChanged.connect(self._update_preview)
        sim_l.addWidget(QLabel("Select Standard Seismometer Simulation / Restitution Filter:"))
        sim_l.addWidget(self.sim_combo)

        for idx, flt_name in enumerate(STANDARD_SIMULATION_FILTERS):
            btn = QPushButton(flt_name)
            btn.clicked.connect(lambda _c=False, fn=flt_name: self._select_sim_and_preview(fn))
            sim_grid.addWidget(btn, idx // 3, idx % 3)
        sim_l.addLayout(sim_grid)
        sim_l.addStretch(1)
        self.tabs.addTab(tab_sim, "Simulation Filters")

        # Tab 2: Butterworth Designer
        tab_but = QWidget()
        but_form = QFormLayout(tab_but)
        self.but_type = QComboBox()
        self.but_type.addItems(["BP (Bandpass)", "LP (Lowpass)", "HP (Highpass)", "BS (Bandstop)"])
        self.but_type.currentIndexChanged.connect(self._on_butter_params_changed)
        but_form.addRow("Filter Type:", self.but_type)

        self.but_f1 = QDoubleSpinBox()
        self.but_f1.setRange(0.001, 100.0)
        self.but_f1.setDecimals(3)
        self.but_f1.setValue(0.700)
        self.but_f1.valueChanged.connect(self._on_butter_params_changed)
        but_form.addRow("Low / Corner Freq f1:", self.but_f1)

        self.but_f2 = QDoubleSpinBox()
        self.but_f2.setRange(0.001, 100.0)
        self.but_f2.setDecimals(3)
        self.but_f2.setValue(2.500)
        self.but_f2.setSuffix(" Hz")
        self.but_f2.valueChanged.connect(self._update_preview)
        but_form.addRow("High Corner Freq f2:", self.but_f2)

        self.but_order = QSpinBox()
        self.but_order.setRange(1, 16)
        self.but_order.setValue(3)
        self.but_order.valueChanged.connect(self._update_preview)
        but_form.addRow("Poles / Order:", self.but_order)
        self.tabs.addTab(tab_but, "Butterworth Designer")

        # Tab 3: .FLF / .FLR Filter File List
        tab_file = QWidget()
        file_l = QVBoxLayout(tab_file)
        file_l.addWidget(QLabel(f"Filter Bank ({self.config.filter_dir}):"))
        self.file_search = QLineEdit()
        self.file_search.setPlaceholderText("Search filter files (e.g. WWSSN, GRF, SRO, 40S)...")
        self.file_search.setClearButtonEnabled(True)
        self.file_search.textChanged.connect(self._on_filter_search_changed)
        file_l.addWidget(self.file_search)

        self.file_list = QListWidget()
        if os.path.isdir(self.config.filter_dir):
            for fn in sorted(os.listdir(self.config.filter_dir)):
                if fn.upper().endswith((".FLF", ".FLR", ".FLT")):
                    self.file_list.addItem(fn)
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
        self.file_list.currentItemChanged.connect(self._on_filter_list_changed)
        file_l.addWidget(self.file_list)
        self.tabs.addTab(tab_file, "Filter Bank (.FLF/.FLR)")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

        # Execution options (two compact rows so checkboxes never clip horizontally)
        opt_grid = QGridLayout()
        opt_grid.setSpacing(6)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["AUTO", "FFT (Pole-Zero)", "RECURSIVE (IIR)"])
        self.chk_demean = QCheckBox("Demean First")
        self.chk_demean.setChecked(True)
        self.chk_lpc = QCheckBox("Burg AR(20) LPC Startup")
        self.chk_autocut = QCheckBox("Autocut ringing transient")
        self.chk_autocut.setChecked(False)
        self.chk_zero_phase = QCheckBox("Zero-phase (sosfiltfilt)")
        self.chk_zero_phase.setChecked(False)
        self.chk_zero_phase.toggled.connect(self._update_preview)

        opt_grid.addWidget(QLabel("Implementation:"), 0, 0)
        opt_grid.addWidget(self.mode_combo, 0, 1)
        opt_grid.addWidget(self.chk_demean, 0, 2)
        opt_grid.addWidget(self.chk_lpc, 0, 3)
        opt_grid.addWidget(self.chk_autocut, 1, 1, 1, 2)
        opt_grid.addWidget(self.chk_zero_phase, 1, 3)
        opt_grid.setColumnStretch(4, 1)
        layout.addLayout(opt_grid)

        # Live Bode frequency response plot
        self.fig = Figure(figsize=(6, 2.5), dpi=90, facecolor="#10141d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Buttons
        btn_h = QHBoxLayout()
        self.expr_lbl = QLabel("Filter: WWSSN_SP")
        btn_h.addWidget(self.expr_lbl)
        btn_h.addStretch(1)
        btn_apply = QPushButton("Apply Filter")
        btn_apply.clicked.connect(self._on_apply)
        btn_remove = QPushButton("Unfilter (NONE)")
        btn_remove.clicked.connect(self._on_unfilter)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_h.addWidget(btn_apply)
        btn_h.addWidget(btn_remove)
        btn_h.addWidget(btn_cancel)
        layout.addLayout(btn_h)

        register_tutorial_help(self.sim_combo, "Instrument Simulation Filter", "dlg_filter_sim", "", "Standard instrument simulation/restitution filter")
        register_tutorial_help(self.but_type, "Butterworth Filter Type", "dlg_filter_butter", "", "Select BP, LP, HP, or BS response")
        register_tutorial_help(self.but_f1, "Lower / Corner Frequency", "dlg_filter_butter", "", "Cutoff frequency f1 in Hz")
        register_tutorial_help(self.but_f2, "Upper Corner Frequency", "dlg_filter_butter", "", "Upper cutoff frequency f2 in Hz")
        register_tutorial_help(self.but_order, "Filter Order (Poles)", "dlg_filter_butter", "", "Butterworth pole count / steepness (1..16 via SOS)")
        register_tutorial_help(self.file_search, "Filter Bank Search", "dlg_filter_presets", "", "Filter .FLF / .FLR bank entries by name")
        register_tutorial_help(self.file_list, "Filter Bank (.FLF / .FLR)", "dlg_filter_presets", "", "SHM standard filter files from filter/")
        register_tutorial_help(self.mode_combo, "Filter Implementation", "dlg_filter_options", "", "AUTO, FFT zero-phase, or Recursive IIR")
        register_tutorial_help(self.chk_demean, "Demean First", "dlg_filter_options", "", "Remove DC offset prior to filtering")
        register_tutorial_help(self.chk_lpc, "Burg AR(20) Startup", "dlg_filter_options", "", "Extrapolate trace start via Burg AR(20) to reduce transients")
        register_tutorial_help(self.chk_autocut, "Autocut Ringing Transient", "dlg_filter_options", "", "Automatically trim initial filter settling ringing transient")
        register_tutorial_help(self.chk_zero_phase, "Zero-Phase (sosfiltfilt)", "dlg_filter_options", "", "Apply forward-backward zero-phase SOS filtering")

        disable_button_autodefault(self)
        apply_c_locale(self)
        self._update_butter_controls()
        self._update_preview()

    def get_filter_options(self) -> Dict[str, Any]:
        """Return currently configured/applied filter parameters."""
        return {
            "filter_expr": self.selected_filter_expr,
            "mode": self.selected_mode,
            "use_lpc": self.use_lpc,
            "demean_first": self.demean_first,
            "autocut": self.autocut,
            "zero_phase": self.zero_phase,
        }

    def _update_butter_controls(self) -> None:
        ft = self.but_type.currentText().split()[0]
        self.but_f2.setEnabled(ft not in ("LP", "HP"))
        f1 = max(1e-6, float(self.but_f1.value()))
        period = 1.0 / f1
        if period >= 100.0:
            t_str = f"{period:.0f}"
        elif period >= 10.0:
            t_str = f"{period:.1f}"
        else:
            t_str = f"{period:.2f}"
        self.but_f1.blockSignals(True)
        self.but_f1.setSuffix(f" Hz (T = {t_str} s)")
        self.but_f1.blockSignals(False)

    def _on_butter_params_changed(self) -> None:
        self._update_butter_controls()
        self._update_preview()

    def _select_sim_and_preview(self, fn: str) -> None:
        if self.tabs.currentIndex() != 0:
            self.tabs.setCurrentIndex(0)
        self.sim_combo.setCurrentText(fn)
        self._update_preview()

    def _on_filter_search_changed(self, text: str) -> None:
        needle = text.strip().lower()
        first_visible: Optional[int] = None
        current_item = self.file_list.currentItem()
        current_still_visible = False
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            matches = not needle or needle in item.text().lower()
            item.setHidden(not matches)
            if matches:
                if first_visible is None:
                    first_visible = row
                if item is current_item:
                    current_still_visible = True
        if not current_still_visible:
            if first_visible is not None:
                self.file_list.setCurrentRow(first_visible)
            else:
                self.file_list.setCurrentItem(None)
                if self.tabs.currentIndex() == 2:
                    self._update_preview()

    def _on_filter_list_changed(self, _cur, _prev) -> None:
        if self.tabs.currentIndex() == 2:
            self._update_preview()

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 2:
            cur = self.file_list.currentItem()
            if cur is None or cur.isHidden():
                for row in range(self.file_list.count()):
                    if not self.file_list.item(row).isHidden():
                        self.file_list.setCurrentRow(row)
                        break
        self._update_preview()

    def _current_expr(self) -> str:
        t_idx = self.tabs.currentIndex()
        if t_idx == 0:
            return self.sim_combo.currentText()
        if t_idx == 1:
            ft = self.but_type.currentText().split()[0]
            f1 = self.but_f1.value()
            f2 = self.but_f2.value()
            order = self.but_order.value()
            if ft in ("LP", "HP"):
                return f"BUTTER {ft} {f1:.3f} {order}"
            return f"BUTTER {ft} {f1:.3f} {f2:.3f} {order}"
        item = self.file_list.currentItem()
        if item is not None and not item.isHidden():
            return item.text()
        return ""

    def _update_preview(self) -> None:
        expr = self._current_expr()
        self.selected_filter_expr = expr or "NONE"
        self.autocut = self.chk_autocut.isChecked() if hasattr(self, "chk_autocut") else False
        self.zero_phase = self.chk_zero_phase.isChecked() if hasattr(self, "chk_zero_phase") else False
        self.expr_lbl.setText(f"Filter: {expr if expr else '(none selected)'}")

        self.fig.clear()
        ax1 = self.fig.add_subplot(121, facecolor="#161b22")
        ax2 = self.fig.add_subplot(122, facecolor="#161b22")

        freqs = np.logspace(-2, math.log10(0.5 / self.dt), 300)
        omega = 2.0 * math.pi * freqs
        stages = load_named_filter(expr, dt=self.dt, config=self.config) if expr else []
        resp = np.ones_like(omega, dtype=np.complex128)
        for stg in stages:
            resp *= evaluate_stage_freq(stg, omega)

        if self.zero_phase and stages and all(s.store_id == 3 for s in stages):
            amp = np.maximum(1e-12, np.abs(resp) ** 2)
            ph_deg = np.zeros_like(amp)
        else:
            amp = np.maximum(1e-12, np.abs(resp))
            ph_deg = np.degrees(np.angle(resp))

        ax1.loglog(freqs, amp, color="#58a6ff", lw=1.6)
        ax1.set_title("Amplitude Response |H(f)|", color="#e6edf3", fontsize=9)
        ax1.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
        ax1.tick_params(colors="#8b949e", labelsize=7)
        ax1.grid(True, which="both", color="#21262d", ls="--", lw=0.6)

        ax2.semilogx(freqs, ph_deg, color="#ffa657", lw=1.4)
        ax2.set_title("Phase Response (deg)", color="#e6edf3", fontsize=9)
        ax2.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
        ax2.tick_params(colors="#8b949e", labelsize=7)
        ax2.grid(True, which="both", color="#21262d", ls="--", lw=0.6)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_apply(self) -> None:
        expr = self._current_expr()
        self.selected_filter_expr = expr if expr else "NONE"
        m_txt = self.mode_combo.currentText()
        self.selected_mode = "FFT" if m_txt.startswith("FFT") else ("REC" if m_txt.startswith("REC") else "AUTO")
        self.use_lpc = self.chk_lpc.isChecked()
        self.demean_first = self.chk_demean.isChecked()
        self.autocut = self.chk_autocut.isChecked()
        self.zero_phase = self.chk_zero_phase.isChecked()
        self.accept()

    def _on_unfilter(self) -> None:
        self.selected_filter_expr = "NONE"
        self.autocut = False
        self.zero_phase = False
        self.accept()



def _compute_fk_bundle(
    traces: List[Trace],
    t_lo: Optional[float],
    t_hi: Optional[float],
    f_lo: float,
    f_hi: float,
    max_slowness: float,
    grid_points: int,
    config: Optional[SHMConfig],
    calib_table: Optional[SlownessCalibrationTable] = None,
    apply_calibration: bool = False,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run FK slowness grid search plus beam diagnostics, 2D ARF, and FitDiagnosticResult."""
    cfg = config or get_config()
    res = compute_fk_analysis(
        traces,
        t_lo=t_lo,
        t_hi=t_hi,
        f_lo=f_lo,
        f_hi=f_hi,
        max_slowness=max_slowness,
        grid_points=grid_points,
        config=cfg,
        progress_callback=progress_callback,
        calib_table=calib_table if apply_calibration else None,
        apply_calibration=bool(apply_calibration),
    )
    if not res or (is_cancelled is not None and is_cancelled()):
        return {}

    # Always preserve raw peak fields so toggling calibration can switch cleanly without re-gridding
    if "raw_azimuth" not in res:
        res["raw_azimuth"] = float(res["azimuth"])
        res["raw_slowness"] = float(res["slowness"])
        res["raw_sx"] = float(res["sx"])
        res["raw_sy"] = float(res["sy"])

    if calib_table is not None and len(calib_table) > 0 and not res.get("calib_applied", False):
        p_slo, p_az, _, _ = calib_table.calibrate(res["raw_slowness"], res["raw_azimuth"])
        res["preview_cal_slowness"] = float(p_slo)
        res["preview_cal_azimuth"] = float(p_az)

    beam_diag: Optional[Dict[str, Any]] = None
    try:
        beam_diag = compute_beam_diagnostics(
            traces,
            azimuth_deg=float(res["azimuth"]),
            slowness_s_deg=float(res["slowness"]),
            t_lo=t_lo,
            t_hi=t_hi,
            config=cfg,
        )
    except Exception:
        beam_diag = None

    if is_cancelled is not None and is_cancelled():
        return {}

    arf_result: Optional[Dict[str, Any]] = None
    try:
        rel_coords = cfg.relative_xy_km([t.station for t in traces], ref_station="CENTRE")
        x_km = [c[0] for c in rel_coords]
        y_km = [c[1] for c in rel_coords]
        f_mid = max(0.05, 0.5 * (float(f_lo) + float(f_hi)))
        n_arf = len(res["s_axis"]) if "s_axis" in res else max(11, int(grid_points) | 1)
        arf_result = compute_array_response_function(
            x_km,
            y_km,
            slowness_max=float(max_slowness),
            n_grid=n_arf,
            freq_hz=f_mid,
        )
    except Exception:
        arf_result = None

    if is_cancelled is not None and is_cancelled():
        return {}

    diag: Optional[FitDiagnosticResult] = None
    try:
        diag = build_fk_fit_diagnostic(
            res,
            beam_diag=beam_diag,
            arf_result=arf_result,
            calib_info=res,
        )
    except Exception:
        diag = None

    res["beam_diagnostics"] = beam_diag
    res["arf_result"] = arf_result
    res["fit_diagnostic"] = diag
    return res


class FKWorker(QThread):
    progress = pyqtSignal(int)
    finishedResult = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        traces: List[Trace],
        t_lo: Optional[float],
        t_hi: Optional[float],
        f_lo: float,
        f_hi: float,
        max_slowness: float,
        grid_points: int,
        config: Optional[SHMConfig],
        parent: Optional[QWidget] = None,
        calib_table: Optional[SlownessCalibrationTable] = None,
        apply_calibration: bool = False,
    ) -> None:
        super().__init__(parent)
        self.traces = traces
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.f_lo = f_lo
        self.f_hi = f_hi
        self.max_slowness = max_slowness
        self.grid_points = grid_points
        self.config = config
        self.calib_table = calib_table
        self.apply_calibration = apply_calibration
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            def cb(cur: int, tot: int) -> bool:
                if self._is_cancelled:
                    return False
                pct = int(100.0 * cur / max(1, tot))
                self.progress.emit(pct)
                return True

            res = _compute_fk_bundle(
                self.traces,
                t_lo=self.t_lo,
                t_hi=self.t_hi,
                f_lo=self.f_lo,
                f_hi=self.f_hi,
                max_slowness=self.max_slowness,
                grid_points=self.grid_points,
                config=self.config,
                calib_table=self.calib_table,
                apply_calibration=self.apply_calibration,
                progress_callback=cb,
                is_cancelled=lambda: self._is_cancelled,
            )
            if not self._is_cancelled and res:
                self.finishedResult.emit(res)
        except Exception as exc:
            if not self._is_cancelled:
                self.error.emit(str(exc))


class FKAnalysisDialog(QDialog):
    """Wideband Coherent Frequency-Wavenumber (FK) Analysis Window matching fkwid.c."""

    adoptParameters = pyqtSignal(float, float)  # (azimuth_deg, slowness_s_deg)
    adoptAndLocate = pyqtSignal(float, float)  # (azimuth_deg, slowness_s_deg)
    fitDiagnosticReady = pyqtSignal(object)  # FitDiagnosticResult

    def __init__(
        self,
        traces: List[Trace],
        t_lo: Optional[float] = None,
        t_hi: Optional[float] = None,
        config: Optional[SHMConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Array Frequency-Wavenumber (FK) Analysis")
        self.resize(880, 650)
        self.traces = traces
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.config = config or get_config()
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_diagnostic: Optional[FitDiagnosticResult] = None
        self.worker: Optional[FKWorker] = None

        stations = [t.station for t in self.traces if getattr(t, "station", "")]
        self.calib_table: Optional[SlownessCalibrationTable] = (
            load_calibration_for_array(stations, config=self.config) if stations else None
        )

        layout = QVBoxLayout(self)

        # Controls bar
        ctrl_grid = QGridLayout()
        ctrl_grid.setSpacing(6)
        self.flo_spin = QDoubleSpinBox()
        self.flo_spin.setRange(0.01, 50.0)
        self.flo_spin.setValue(0.5)
        self.flo_spin.setSuffix(" Hz")
        self.fhi_spin = QDoubleSpinBox()
        self.fhi_spin.setRange(0.05, 50.0)
        self.fhi_spin.setValue(3.0)
        self.fhi_spin.setSuffix(" Hz")
        self.max_slo_spin = QDoubleSpinBox()
        self.max_slo_spin.setRange(1.0, 40.0)
        self.max_slo_spin.setValue(12.0)
        self.max_slo_spin.setSuffix(" s/deg")
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(15, 101)
        self.grid_spin.setSingleStep(2)
        self.grid_spin.setValue(41)

        self.btn_run = QPushButton("Compute FK")
        self.btn_run.clicked.connect(lambda: self.run_fk(sync=False))
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_fk)

        self.chk_apply_calib = QCheckBox("Apply Slowness Calibration")
        has_calib = self.calib_table is not None and len(self.calib_table) > 0
        self.chk_apply_calib.setEnabled(has_calib)
        self.chk_apply_calib.setChecked(has_calib)
        self.chk_apply_calib.toggled.connect(self._on_calib_toggled)
        self.calib_status_lbl = QLabel(
            f"Calibration table loaded ({len(self.calib_table)} entries)"
            if has_calib
            else "No array slowness calibration table found"
        )
        self.calib_status_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")

        ctrl_grid.addWidget(QLabel("Freq Lo:"), 0, 0)
        ctrl_grid.addWidget(self.flo_spin, 0, 1)
        ctrl_grid.addWidget(QLabel("Freq Hi:"), 0, 2)
        ctrl_grid.addWidget(self.fhi_spin, 0, 3)
        ctrl_grid.addWidget(QLabel("Max Slowness:"), 1, 0)
        ctrl_grid.addWidget(self.max_slo_spin, 1, 1)
        ctrl_grid.addWidget(QLabel("Grid:"), 1, 2)
        ctrl_grid.addWidget(self.grid_spin, 1, 3)
        ctrl_grid.addWidget(self.btn_run, 0, 4, 1, 1)
        ctrl_grid.addWidget(self.btn_cancel, 1, 4, 1, 1)
        ctrl_grid.addWidget(self.chk_apply_calib, 2, 0, 1, 3)
        ctrl_grid.addWidget(self.calib_status_lbl, 2, 3, 1, 3)
        ctrl_grid.setColumnStretch(5, 1)
        layout.addLayout(ctrl_grid)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.res_label = QLabel("Click 'Compute FK' to run wideband coherent slowness-power search.")
        self.res_label.setStyleSheet("font-weight: bold; color: #58a6ff;")
        self.res_label.setWordWrap(True)
        layout.addWidget(self.res_label)

        self.diag_label = QLabel("Beam Coherence / Poor Channels (CC < 0.70): —")
        self.diag_label.setStyleSheet("color: #8b949e;")
        self.diag_label.setWordWrap(True)
        layout.addWidget(self.diag_label)

        self.fig = Figure(figsize=(7.6, 4.6), dpi=95, facecolor="#10141d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        bot_h = QHBoxLayout()
        self.chk_auto_locate = QCheckBox("Also run slowness location")
        self.chk_auto_locate.setChecked(False)
        btn_adopt = QPushButton("Adopt Azimuth && Slowness to Parameter Box")
        btn_adopt.clicked.connect(self._on_adopt)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_h.addWidget(self.chk_auto_locate)
        bot_h.addStretch(1)
        bot_h.addWidget(btn_adopt)
        bot_h.addWidget(btn_close)
        layout.addLayout(bot_h)

        for w, t in [
            (self.flo_spin, "Low Frequency Cutoff"),
            (self.fhi_spin, "High Frequency Cutoff"),
            (self.max_slo_spin, "Max Slowness Search Bound"),
            (self.grid_spin, "FK Slowness Grid Size"),
            (self.chk_apply_calib, "Apply empirical slowness/backazimuth calibration (calib_grf.dat)"),
            (self.chk_auto_locate, "Compute epicenter from slowness when adopting"),
        ]:
            register_tutorial_help(w, t, "dlg_fk_params", "", t)

        disable_button_autodefault(self)
        apply_c_locale(self)

        if len(self.traces) >= 3:
            self.run_fk(sync=True)

    def _on_calib_toggled(self, checked: bool) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.last_result is None or len(self.traces) < 3:
            return

        res = dict(self.last_result)
        raw_az = float(res.get("raw_azimuth", res["azimuth"]))
        raw_slo = float(res.get("raw_slowness", res["slowness"]))
        raw_sx = float(res.get("raw_sx", raw_slo * math.sin(math.radians(raw_az))))
        raw_sy = float(res.get("raw_sy", raw_slo * math.cos(math.radians(raw_az))))

        deg_to_km = 111.19492664455873
        if checked and self.calib_table is not None and len(self.calib_table) > 0:
            cal_slo, cal_az, d_slo, d_az = self.calib_table.calibrate(raw_slo, raw_az)
            cal_az_r = math.radians(cal_az)
            cal_sx = float(cal_slo * math.sin(cal_az_r))
            cal_sy = float(cal_slo * math.cos(cal_az_r))
            res.update(
                {
                    "azimuth": float(cal_az),
                    "slowness": float(cal_slo),
                    "apparent_velocity_kms": (deg_to_km / cal_slo) if cal_slo > 1e-6 else 999.9,
                    "sx": cal_sx,
                    "sy": cal_sy,
                    "calib_applied": True,
                    "raw_azimuth": raw_az,
                    "raw_slowness": raw_slo,
                    "raw_sx": raw_sx,
                    "raw_sy": raw_sy,
                    "cal_azimuth": float(cal_az),
                    "cal_slowness": float(cal_slo),
                    "cal_sx": cal_sx,
                    "cal_sy": cal_sy,
                    "calib_d_slowness": float(d_slo),
                    "calib_d_azimuth": float(d_az),
                }
            )
        else:
            if self.calib_table is not None and len(self.calib_table) > 0:
                p_slo, p_az, _, _ = self.calib_table.calibrate(raw_slo, raw_az)
                res["preview_cal_slowness"] = float(p_slo)
                res["preview_cal_azimuth"] = float(p_az)
            res.update(
                {
                    "azimuth": raw_az,
                    "slowness": raw_slo,
                    "apparent_velocity_kms": (deg_to_km / raw_slo) if raw_slo > 1e-6 else 999.9,
                    "sx": raw_sx,
                    "sy": raw_sy,
                    "calib_applied": False,
                    "raw_azimuth": raw_az,
                    "raw_slowness": raw_slo,
                    "raw_sx": raw_sx,
                    "raw_sy": raw_sy,
                }
            )

        try:
            beam_diag = compute_beam_diagnostics(
                self.traces,
                azimuth_deg=float(res["azimuth"]),
                slowness_s_deg=float(res["slowness"]),
                t_lo=self.t_lo,
                t_hi=self.t_hi,
                config=self.config,
            )
        except Exception:
            beam_diag = None
        arf_res = res.get("arf_result")
        try:
            diag = build_fk_fit_diagnostic(
                res,
                beam_diag=beam_diag,
                arf_result=arf_res,
                calib_info=res,
            )
        except Exception:
            diag = None
        res["beam_diagnostics"] = beam_diag
        res["fit_diagnostic"] = diag
        self._on_fk_finished(res)

    def run_fk(self, sync: bool = True) -> None:
        if len(self.traces) < 3:
            self.res_label.setText("Need at least 3 array traces in workspace for FK analysis.")
            return

        apply_cal = bool(self.chk_apply_calib.isChecked() and self.calib_table is not None)
        if not sync:
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel()
                self.worker.wait(500)
            self.btn_run.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.res_label.setText("Computing FK grid search, ARF, and beam diagnostics in background...")

            self.worker = FKWorker(
                self.traces,
                t_lo=self.t_lo,
                t_hi=self.t_hi,
                f_lo=self.flo_spin.value(),
                f_hi=self.fhi_spin.value(),
                max_slowness=self.max_slo_spin.value(),
                grid_points=self.grid_spin.value(),
                config=self.config,
                parent=self,
                calib_table=self.calib_table,
                apply_calibration=apply_cal,
            )
            self.worker.progress.connect(self._on_fk_progress)
            self.worker.finishedResult.connect(self._on_fk_finished)
            self.worker.error.connect(self._on_fk_error)
            self.worker.start()
        else:
            res = _compute_fk_bundle(
                self.traces,
                t_lo=self.t_lo,
                t_hi=self.t_hi,
                f_lo=self.flo_spin.value(),
                f_hi=self.fhi_spin.value(),
                max_slowness=self.max_slo_spin.value(),
                grid_points=self.grid_spin.value(),
                config=self.config,
                calib_table=self.calib_table,
                apply_calibration=apply_cal,
            )
            self._on_fk_finished(res)

    def _on_fk_progress(self, pct: int) -> None:
        self.progress_bar.setValue(pct)

    def _on_fk_finished(self, res: Dict[str, Any]) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        if not res:
            return
        self.last_result = res
        self.last_diagnostic = res.get("fit_diagnostic")

        raw_az = float(res.get("raw_azimuth", res["azimuth"]))
        raw_slo = float(res.get("raw_slowness", res["slowness"]))
        calib_applied = bool(res.get("calib_applied", False))
        if calib_applied:
            cal_az = float(res.get("cal_azimuth", res["azimuth"]))
            cal_slo = float(res.get("cal_slowness", res["slowness"]))
            self.res_label.setText(
                f"Raw BAZ: {raw_az:.1f} deg | Raw Slow: {raw_slo:.2f} s/deg   ->   "
                f"Cal BAZ: {cal_az:.1f} deg | Cal Slow: {cal_slo:.2f} s/deg   |   "
                f"App Vel: {res['apparent_velocity_kms']:.2f} km/s   |   "
                f"Semblance: {res['semblance']:.3f} ({res['power_db']:.1f} dB)"
            )
        else:
            cal_note = ""
            if "preview_cal_azimuth" in res and "preview_cal_slowness" in res:
                cal_note = (
                    f"   |   Calibrated (off): {res['preview_cal_azimuth']:.1f} deg, "
                    f"{res['preview_cal_slowness']:.2f} s/deg"
                )
            self.res_label.setText(
                f"Raw BAZ: {raw_az:.1f} deg | Raw Slow: {raw_slo:.2f} s/deg{cal_note}   |   "
                f"App Vel: {res['apparent_velocity_kms']:.2f} km/s   |   "
                f"Semblance: {res['semblance']:.3f} ({res['power_db']:.1f} dB)"
            )

        beam_diag = res.get("beam_diagnostics")
        if beam_diag:
            mean_cc = float(beam_diag.get("mean_cc", 0.0))
            ch_names = beam_diag.get("channel_names", [])
            ch_cc = beam_diag.get("channel_cc", [])
            poor_pairs = [(n, c) for n, c in zip(ch_names, ch_cc) if c < 0.70]
            if poor_pairs:
                poor_str = ", ".join(f"{n} (CC={c:.2f})" for n, c in poor_pairs)
                self.diag_label.setText(
                    f"Beam Mean CC: {mean_cc:.3f}   |   Poor channels (CC < 0.70): {poor_str}"
                )
                self.diag_label.setStyleSheet("color: #f85149; font-weight: bold;")
            else:
                self.diag_label.setText(
                    f"Beam Mean CC: {mean_cc:.3f}   |   Poor channels (CC < 0.70): None ({len(ch_names)} coherent)"
                )
                self.diag_label.setStyleSheet("color: #3fb950;")
        else:
            self.diag_label.setText("Beam Coherence / Poor Channels (CC < 0.70): N/A")
            self.diag_label.setStyleSheet("color: #8b949e;")

        self.fig.clear()
        arf_res = res.get("arf_result")
        ax = self.fig.add_subplot(121 if arf_res is not None else 111, facecolor="#0d1117")
        s_ax = res["s_axis"]
        grid = res["grid"]
        im = ax.contourf(s_ax, s_ax, grid, levels=20, cmap="inferno")
        add_dark_colorbar(self.fig, im, ax, "Relative Semblance Power")

        # Draw slowness circles & cardinal axes
        for r_s in (3.0, 6.0, 9.0, 12.0):
            if r_s <= self.max_slo_spin.value():
                th = np.linspace(0, 2 * math.pi, 120)
                ax.plot(r_s * np.cos(th), r_s * np.sin(th), color="#8b949e", ls="--", lw=0.7)
        ax.axhline(0, color="#8b949e", lw=0.7)
        ax.axvline(0, color="#8b949e", lw=0.7)

        raw_sx = float(res.get("raw_sx", res["sx"]))
        raw_sy = float(res.get("raw_sy", res["sy"]))
        if calib_applied:
            ax.plot([0, raw_sx], [0, raw_sy], color="#8b949e", lw=1.2, ls=":")
            ax.plot(raw_sx, raw_sy, marker="o", markersize=8, color="#ffa657", label="Raw FK Peak")
            ax.annotate(
                "",
                xy=(res["sx"], res["sy"]),
                xytext=(raw_sx, raw_sy),
                arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.8),
            )
            ax.plot(res["sx"], res["sy"], marker="*", markersize=14, color="#3fb950", label="Calibrated Peak")
            ax.legend(loc="upper right", fontsize=7, facecolor="#161b22", labelcolor="#e6edf3")
        else:
            ax.plot([0, res["sx"]], [0, res["sy"]], color="#58a6ff", lw=1.8)
            ax.plot(res["sx"], res["sy"], marker="*", markersize=14, color="#3fb950", label="FK Peak")

        ax.set_xlabel("East Slowness Sx (s/deg)", color="#e6edf3")
        ax.set_ylabel("North Slowness Sy (s/deg)", color="#e6edf3")
        ax.set_title(
            f"Wideband FK Slowness Map ({self.flo_spin.value():.1f}-{self.fhi_spin.value():.1f} Hz)",
            color="#e6edf3",
            fontsize=9,
        )
        ax.tick_params(colors="#8b949e")
        ax.set_aspect("equal")

        if arf_res is not None:
            ax_arf = self.fig.add_subplot(122, facecolor="#0d1117")
            arf_s = arf_res["s_axis"]
            arf_db = np.clip(arf_res["arf_db"], -25.0, 0.0)
            im_arf = ax_arf.contourf(arf_s, arf_s, arf_db, levels=16, cmap="viridis")
            add_dark_colorbar(self.fig, im_arf, ax_arf, "ARF |A(k)|^2 (dB)")
            ax_arf.axhline(0, color="#8b949e", lw=0.7)
            ax_arf.axvline(0, color="#8b949e", lw=0.7)
            ax_arf.set_xlabel("East Slowness Sx (s/deg)", color="#e6edf3")
            ax_arf.set_ylabel("North Slowness Sy (s/deg)", color="#e6edf3")
            ax_arf.set_title("Array Response Function (ARF)", color="#e6edf3", fontsize=9)
            ax_arf.tick_params(colors="#8b949e")
            ax_arf.set_aspect("equal")

        self.fig.tight_layout()
        self.canvas.draw_idle()

        if self.last_diagnostic is not None:
            self.fitDiagnosticReady.emit(self.last_diagnostic)

    def _on_fk_error(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.res_label.setText(f"FK Analysis Error: {err}")

    def cancel_fk(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
            self.btn_cancel.setEnabled(False)
            self.btn_run.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.res_label.setText("FK analysis cancelled.")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().closeEvent(event)

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().reject()

    def _on_adopt(self) -> None:
        if self.last_result is not None:
            az = self.last_result["azimuth"]
            slo = self.last_result["slowness"]
            if self.chk_auto_locate.isChecked():
                self.adoptAndLocate.emit(az, slo)
            else:
                self.adoptParameters.emit(az, slo)
            self.accept()



class VespagramWorker(QThread):
    progress = pyqtSignal(int)
    finishedResult = pyqtSignal(object)  # (v_trs, times, slos, mat)
    error = pyqtSignal(str)

    def __init__(
        self,
        traces: List[Trace],
        azimuth_deg: float,
        slow_lo: float,
        slow_hi: float,
        slow_step: float,
        power: float,
        ref_station: str = "CENTRE",
        config: Optional[SHMConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.traces = traces
        self.azimuth_deg = azimuth_deg
        self.slow_lo = slow_lo
        self.slow_hi = slow_hi
        self.slow_step = slow_step
        self.power = power
        self.ref_station = ref_station
        self.config = config
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            def cb(cur: int, tot: int) -> bool:
                if self._is_cancelled:
                    return False
                pct = int(100.0 * cur / max(1, tot))
                self.progress.emit(pct)
                return True

            v_trs, times, slos, mat = compute_vespagram(
                self.traces,
                azimuth_deg=self.azimuth_deg,
                slow_lo=self.slow_lo,
                slow_hi=self.slow_hi,
                slow_step=self.slow_step,
                power=self.power,
                ref_station=self.ref_station,
                config=self.config,
                progress_callback=cb,
            )
            if not self._is_cancelled:
                self.finishedResult.emit((v_trs, times, slos, mat))
        except Exception as exc:
            if not self._is_cancelled:
                self.error.emit(str(exc))


class VespagramDialog(QDialog):
    """Interactive N-th Root Vespagram Slowness-Time Analysis Dialog."""

    appendVespaTraces = pyqtSignal(list)

    def __init__(
        self,
        traces: List[Trace],
        azimuth_deg: float = 45.0,
        config: Optional[SHMConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vespagram (Slowness-Time Slant Stack)")
        self.resize(780, 560)
        self.traces = traces
        self.config = config or get_config()
        self.vespa_traces: List[Trace] = []
        self.worker: Optional[VespagramWorker] = None

        layout = QVBoxLayout(self)

        ctrl_grid = QGridLayout()
        ctrl_grid.setSpacing(6)
        self.az_spin = QDoubleSpinBox()
        self.az_spin.setRange(0.0, 360.0)
        self.az_spin.setValue(azimuth_deg % 360.0)
        self.slo_lo_spin = QDoubleSpinBox()
        self.slo_lo_spin.setRange(0.0, 30.0)
        self.slo_lo_spin.setValue(2.0)
        self.slo_hi_spin = QDoubleSpinBox()
        self.slo_hi_spin.setRange(0.5, 40.0)
        self.slo_hi_spin.setValue(12.0)
        self.slo_step_spin = QDoubleSpinBox()
        self.slo_step_spin.setRange(0.1, 5.0)
        self.slo_step_spin.setValue(0.5)
        self.root_spin = QDoubleSpinBox()
        self.root_spin.setRange(1.0, 10.0)
        self.root_spin.setValue(2.0)

        self.btn_calc = QPushButton("Compute Vespagram")
        self.btn_calc.clicked.connect(lambda: self.run_vespa(sync=False))
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_vespa)

        ctrl_grid.addWidget(QLabel("Azimuth:"), 0, 0)
        ctrl_grid.addWidget(self.az_spin, 0, 1)
        ctrl_grid.addWidget(QLabel("Slowness Lo:"), 0, 2)
        ctrl_grid.addWidget(self.slo_lo_spin, 0, 3)
        ctrl_grid.addWidget(QLabel("Hi:"), 0, 4)
        ctrl_grid.addWidget(self.slo_hi_spin, 0, 5)
        ctrl_grid.addWidget(QLabel("Step:"), 1, 0)
        ctrl_grid.addWidget(self.slo_step_spin, 1, 1)
        ctrl_grid.addWidget(QLabel("N-th Root:"), 1, 2)
        ctrl_grid.addWidget(self.root_spin, 1, 3)
        ctrl_grid.addWidget(self.btn_calc, 1, 4, 1, 1)
        ctrl_grid.addWidget(self.btn_cancel, 1, 5, 1, 1)
        ctrl_grid.setColumnStretch(6, 1)
        layout.addLayout(ctrl_grid)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.fig = Figure(figsize=(7, 4.2), dpi=95, facecolor="#10141d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        bot_h = QHBoxLayout()
        btn_send = QPushButton("Append Vespa Traces to Main Workspace")
        btn_send.clicked.connect(self._on_send)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_h.addStretch(1)
        bot_h.addWidget(btn_send)
        bot_h.addWidget(btn_close)
        layout.addLayout(bot_h)

        for w, t in [
            (self.az_spin, "Vespagram Backazimuth"),
            (self.slo_lo_spin, "Slowness Sweep Minimum"),
            (self.slo_hi_spin, "Slowness Sweep Maximum"),
            (self.slo_step_spin, "Slowness Step Increment"),
            (self.root_spin, "N-th Root Stacking Power"),
        ]:
            register_tutorial_help(w, t, "dlg_vespa_params", "", t)

        disable_button_autodefault(self)
        apply_c_locale(self)

        if self.traces:
            self.run_vespa(sync=True)

    def run_vespa(self, sync: bool = True) -> None:
        if not self.traces:
            return

        if not sync:
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel()
                self.worker.wait(500)
            self.btn_calc.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            self.worker = VespagramWorker(
                self.traces,
                azimuth_deg=self.az_spin.value(),
                slow_lo=self.slo_lo_spin.value(),
                slow_hi=self.slo_hi_spin.value(),
                slow_step=self.slo_step_spin.value(),
                power=self.root_spin.value(),
                config=self.config,
                parent=self,
            )
            self.worker.progress.connect(self._on_vespa_progress)
            self.worker.finishedResult.connect(self._on_vespa_finished)
            self.worker.error.connect(self._on_vespa_error)
            self.worker.start()
        else:
            v_trs, times, slos, mat = compute_vespagram(
                self.traces,
                azimuth_deg=self.az_spin.value(),
                slow_lo=self.slo_lo_spin.value(),
                slow_hi=self.slo_hi_spin.value(),
                slow_step=self.slo_step_spin.value(),
                power=self.root_spin.value(),
                config=self.config,
            )
            self._on_vespa_finished((v_trs, times, slos, mat))

    def _on_vespa_progress(self, pct: int) -> None:
        self.progress_bar.setValue(pct)

    def _on_vespa_finished(self, result: Tuple[List[Trace], np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.btn_calc.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        v_trs, times, slos, mat = result
        self.vespa_traces = v_trs
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor="#0d1117")
        env_mat = np.abs(mat)
        im = ax.pcolormesh(times, slos, env_mat, shading="auto", cmap="magma")
        add_dark_colorbar(self.fig, im, ax, "Vespa Beam Amplitude")
        ax.set_xlabel("Time (s)", color="#e6edf3")
        ax.set_ylabel("Slowness (s/deg)", color="#e6edf3")
        ax.set_title(
            f"N={self.root_spin.value():.0f} Vespagram at Backazimuth {self.az_spin.value():.1f} deg",
            color="#e6edf3",
        )
        ax.tick_params(colors="#8b949e")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_vespa_error(self, _err: str) -> None:
        self.btn_calc.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

    def cancel_vespa(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
            self.btn_cancel.setEnabled(False)
            self.btn_calc.setEnabled(True)
            self.progress_bar.setVisible(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().closeEvent(event)

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().reject()

    def _on_send(self) -> None:
        if self.vespa_traces:
            self.appendVespaTraces.emit(self.vespa_traces)
            self.accept()


class BruneInversionWorker(QThread):
    finishedResult = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        trace: Trace,
        t_lo: Optional[float],
        t_hi: Optional[float],
        noise_lo: Optional[float],
        noise_hi: Optional[float],
        model: str = "brune",
        f_lo: float = 0.1,
        f_hi: Optional[float] = None,
        t_star_init: float = 0.02,
        fit_t_star: bool = True,
        distance_km: float = 100.0,
        vel_kms: float = 3.5,
        phase: str = "P",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.trace = trace
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.noise_lo = noise_lo
        self.noise_hi = noise_hi
        self.model = model
        self.f_lo = f_lo
        self.f_hi = f_hi
        self.t_star_init = t_star_init
        self.fit_t_star = fit_t_star
        self.distance_km = distance_km
        self.vel_kms = vel_kms
        self.phase = phase
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            if self._is_cancelled:
                return
            inv_res = invert_brune_spectrum(
                self.trace,
                t_lo=self.t_lo,
                t_hi=self.t_hi,
                noise_lo=self.noise_lo,
                noise_hi=self.noise_hi,
                model=self.model,
                f_lo=self.f_lo,
                f_hi=self.f_hi,
                t_star_init=self.t_star_init,
                fit_t_star=self.fit_t_star,
                distance_m=max(100.0, float(self.distance_km) * 1000.0),
                vel_m_s=max(100.0, float(self.vel_kms) * 1000.0),
            )
            if self._is_cancelled:
                return
            diag = build_brune_fit_diagnostic(inv_res, station=self.trace.station, phase=self.phase)
            if not self._is_cancelled:
                self.finishedResult.emit({"inversion": inv_res, "diagnostic": diag})
        except Exception as exc:
            if not self._is_cancelled:
                self.error.emit(str(exc))


class SpectrumDialog(QDialog):
    """Interactive spectral analysis (FFT, Welch PSD, Burg MEM), spectrogram, and Brune/Boatwright source inversion."""

    CURVE_COLORS = ["#58a6ff", "#ffa657", "#3fb950", "#d2a8ff", "#ff7b72", "#f2cc60", "#56d4dd", "#ff9bce"]

    fitDiagnosticReady = pyqtSignal(object)  # FitDiagnosticResult

    def __init__(
        self,
        trace: Optional[Trace] = None,
        t_lo: Optional[float] = None,
        t_hi: Optional[float] = None,
        parent: Optional[QWidget] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectrum & Spectrogram")
        self.resize(880, 660)
        # Each curve: dict(trace=<snapshot>, t_lo, t_hi (trace-relative s), label, color)
        self.curves: List[dict] = []
        self._ax = None
        self._spec_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._closing = False
        self.brune_worker: Optional[BruneInversionWorker] = None
        self.last_brune_result: Optional[Dict[str, Any]] = None
        self.last_brune_diagnostic: Optional[FitDiagnosticResult] = None

        layout = QVBoxLayout(self)

        ctrl_grid = QGridLayout()
        ctrl_grid.setSpacing(6)
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                "FFT Amplitude Spectrum",
                "FFT Squared Spectrum",
                "Bartlett/Welch Power Spectral Density (PSD)",
                "Burg Maximum Entropy (MEM AR) Spectrum",
                "Sliding-Window Hann Spectrogram",
            ]
        )
        self.method_combo.currentIndexChanged.connect(self.replot)

        self.fit_f1 = QDoubleSpinBox()
        self.fit_f1.setRange(0.01, 50.0)
        self.fit_f1.setValue(0.5)
        self.fit_f2 = QDoubleSpinBox()
        self.fit_f2.setRange(0.05, 50.0)
        self.fit_f2.setValue(5.0)
        self.fit_f1.valueChanged.connect(self.replot)
        self.fit_f2.valueChanged.connect(self.replot)

        self.chk_fit = QCheckBox("Show fit lines")
        self.chk_fit.setChecked(True)
        self.chk_fit.toggled.connect(self.replot)

        btn_clear = QPushButton("Clear Curves")
        btn_clear.clicked.connect(self.clear_curves)

        ctrl_grid.addWidget(QLabel("Method:"), 0, 0)
        ctrl_grid.addWidget(self.method_combo, 0, 1, 1, 3)
        ctrl_grid.addWidget(btn_clear, 0, 5)
        ctrl_grid.addWidget(QLabel("Fit Band [Hz]:"), 1, 0)
        ctrl_grid.addWidget(self.fit_f1, 1, 1)
        ctrl_grid.addWidget(QLabel("to"), 1, 2)
        ctrl_grid.addWidget(self.fit_f2, 1, 3)
        ctrl_grid.addWidget(self.chk_fit, 1, 4)
        ctrl_grid.setColumnStretch(6, 1)
        layout.addLayout(ctrl_grid)

        # Brune / Boatwright w^-2 Source Inversion Controls (M-09)
        brune_grp = QGroupBox("Brune / Boatwright w^-2 Source Inversion")
        brune_grid = QGridLayout(brune_grp)
        brune_grid.setSpacing(6)

        self.brune_model_combo = QComboBox()
        self.brune_model_combo.addItems(["brune", "boatwright"])

        self.brune_tstar_spin = QDoubleSpinBox()
        self.brune_tstar_spin.setRange(0.0, 2.0)
        self.brune_tstar_spin.setDecimals(3)
        self.brune_tstar_spin.setSingleStep(0.01)
        self.brune_tstar_spin.setValue(0.020)
        self.brune_tstar_spin.setSuffix(" s")

        self.chk_fit_tstar = QCheckBox("Fit t*")
        self.chk_fit_tstar.setChecked(True)

        self.dist_km_spin = QDoubleSpinBox()
        self.dist_km_spin.setRange(0.1, 20000.0)
        self.dist_km_spin.setDecimals(1)
        self.dist_km_spin.setValue(100.0)
        self.dist_km_spin.setSuffix(" km")

        self.vel_kms_spin = QDoubleSpinBox()
        self.vel_kms_spin.setRange(0.5, 15.0)
        self.vel_kms_spin.setDecimals(2)
        self.vel_kms_spin.setValue(3.50)
        self.vel_kms_spin.setSuffix(" km/s")

        self.chk_use_noise = QCheckBox("Noise Window [s]:")
        self.chk_use_noise.setChecked(False)
        self.noise_lo_spin = QDoubleSpinBox()
        self.noise_lo_spin.setRange(-10000.0, 100000.0)
        self.noise_lo_spin.setDecimals(2)
        self.noise_lo_spin.setValue(0.0)
        self.noise_hi_spin = QDoubleSpinBox()
        self.noise_hi_spin.setRange(-10000.0, 100000.0)
        self.noise_hi_spin.setDecimals(2)
        self.noise_hi_spin.setValue(5.0)

        self.btn_fit_brune = QPushButton("Fit Brune/Boatwright")
        self.btn_fit_brune.clicked.connect(lambda: self.run_brune_inversion(sync=False))

        brune_grid.addWidget(QLabel("Model:"), 0, 0)
        brune_grid.addWidget(self.brune_model_combo, 0, 1)
        brune_grid.addWidget(QLabel("t*:"), 0, 2)
        brune_grid.addWidget(self.brune_tstar_spin, 0, 3)
        brune_grid.addWidget(self.chk_fit_tstar, 0, 4)
        brune_grid.addWidget(QLabel("Distance:"), 0, 5)
        brune_grid.addWidget(self.dist_km_spin, 0, 6)
        brune_grid.addWidget(QLabel("Velocity:"), 0, 7)
        brune_grid.addWidget(self.vel_kms_spin, 0, 8)

        brune_grid.addWidget(self.chk_use_noise, 1, 0, 1, 2)
        brune_grid.addWidget(self.noise_lo_spin, 1, 2, 1, 2)
        brune_grid.addWidget(QLabel("to"), 1, 4)
        brune_grid.addWidget(self.noise_hi_spin, 1, 5, 1, 2)
        brune_grid.addWidget(self.btn_fit_brune, 1, 7, 1, 2)
        brune_grid.setColumnStretch(9, 1)
        layout.addWidget(brune_grp)

        self.fit_label = QLabel("")
        self.fit_label.setStyleSheet("color: #ffa657; font-weight: bold;")
        self.fit_label.setWordWrap(True)
        layout.addWidget(self.fit_label)

        self.brune_label = QLabel("Brune/Boatwright Source Fit: Click 'Fit Brune/Boatwright' to invert source parameters.")
        self.brune_label.setStyleSheet("color: #3fb950; font-weight: bold;")
        self.brune_label.setWordWrap(True)
        layout.addWidget(self.brune_label)

        self.fig = Figure(figsize=(7.2, 4.2), dpi=95, facecolor="#10141d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        bottom_h = QHBoxLayout()
        self.cursor_label = QLabel("Cursor: —")
        self.cursor_label.setStyleSheet("color: #79c0ff; font-family: monospace; font-size: 12px;")
        bottom_h.addWidget(self.cursor_label, stretch=1)

        btn_close = QPushButton("Close")
        btn_close.setAutoDefault(False)
        btn_close.setDefault(False)
        btn_close.clicked.connect(self.close)
        self.setSizeGripEnabled(False)
        bottom_h.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(bottom_h)

        register_tutorial_help(self.method_combo, "Spectral Estimation Method", "dlg_spec_method", "", "Choose FFT, Welch PSD, Burg MEM, or Spectrogram")
        register_tutorial_help(self.fit_f1, "Spectral Fit Low Freq (Hz)", "dlg_spec_method", "", "Lower frequency for log-log spectral fit")
        register_tutorial_help(self.fit_f2, "Spectral Fit High Freq (Hz)", "dlg_spec_method", "", "Upper frequency for log-log spectral fit")
        register_tutorial_help(self.chk_fit, "Show Log-Log Fit Lines", "dlg_spec_method", "", "Overlay corner-frequency fit model")
        register_tutorial_help(self.brune_model_combo, "Source Spectrum Model", "dlg_spec_method", "", "Select Brune (gamma=1) or Boatwright (gamma=2) w^-2 model")
        register_tutorial_help(self.brune_tstar_spin, "Attenuation t*", "dlg_spec_method", "", "Initial or fixed attenuation operator t* in seconds")
        register_tutorial_help(self.dist_km_spin, "Epicentral Distance (km)", "dlg_spec_method", "", "Hypocentral/epicentral distance for seismic moment M0")
        register_tutorial_help(self.vel_kms_spin, "Phase Velocity (km/s)", "dlg_spec_method", "", "Wave speed for M0, source radius r0, and stress drop")

        disable_button_autodefault(self)
        apply_c_locale(self)

        self._mpl_cids = [
            self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion),
            self.canvas.mpl_connect("axes_leave_event", self._on_canvas_leave),
            self.canvas.mpl_connect("figure_leave_event", self._on_canvas_leave),
        ]

        if trace is not None:
            self.add_curve(trace, t_lo, t_hi, label=label)
        else:
            self.replot()

    # Backwards-compatible accessor for the most recently added curve
    @property
    def trace(self) -> Optional[Trace]:
        return self.curves[-1]["trace"] if self.curves else None

    def add_curve(
        self,
        trace: Trace,
        t_lo: Optional[float] = None,
        t_hi: Optional[float] = None,
        label: Optional[str] = None,
    ) -> bool:
        """Add a spectrum curve for trace[t_lo:t_hi] (trace-relative seconds). Returns False if duplicate."""
        key = (trace.station, trace.channel3, trace.start_epoch, None if t_lo is None else round(t_lo, 3),
               None if t_hi is None else round(t_hi, 3))
        if any(c["key"] == key for c in self.curves):
            return False
        if label is None:
            label = f"{trace.station} {trace.channel3}"
            if t_lo is not None and t_hi is not None:
                label += f"  [{t_lo:.1f}..{t_hi:.1f} s]"
        color = self.CURVE_COLORS[len(self.curves) % len(self.CURVE_COLORS)]
        self.curves.append(
            {
                "key": key,
                "trace": trace.copy(copy_picks=False),
                "t_lo": t_lo,
                "t_hi": t_hi,
                "label": label,
                "color": color,
            }
        )
        if getattr(trace, "distance", 0.0) > 0.0 and len(self.curves) == 1:
            self.dist_km_spin.setValue(max(0.1, float(trace.distance) * 111.19492664455873))
        if t_lo is not None and t_lo > 1.0 and len(self.curves) == 1:
            n_lo = max(0.0, float(t_lo) - 10.0)
            n_hi = max(n_lo + 0.5, float(t_lo) - 0.5)
            if n_hi < t_lo:
                self.noise_lo_spin.setValue(n_lo)
                self.noise_hi_spin.setValue(n_hi)
        self.setWindowTitle(f"Spectrum & Spectrogram ({len(self.curves)} curve{'s' if len(self.curves) != 1 else ''})")
        self.replot()
        return True

    def clear_curves(self) -> None:
        self.curves.clear()
        self.last_brune_result = None
        self.last_brune_diagnostic = None
        self.brune_label.setText("Brune/Boatwright Source Fit: Click 'Fit Brune/Boatwright' to invert source parameters.")
        self.setWindowTitle("Spectrum & Spectrogram")
        self._on_canvas_leave()
        self.replot()

    def run_brune_inversion(self, sync: bool = False) -> None:
        """Run non-linear Brune/Boatwright w^-2 spectral source inversion on the latest curve."""
        if not self.curves:
            self.brune_label.setText("Brune/Boatwright Source Fit: Add a spectrum curve first.")
            return

        c = self.curves[-1]
        tr: Trace = c["trace"]
        t_lo: Optional[float] = c["t_lo"]
        t_hi: Optional[float] = c["t_hi"]

        noise_lo: Optional[float] = None
        noise_hi: Optional[float] = None
        if self.chk_use_noise.isChecked():
            lo_v = float(self.noise_lo_spin.value())
            hi_v = float(self.noise_hi_spin.value())
            if abs(hi_v - lo_v) > max(tr.delta, 1e-6):
                noise_lo, noise_hi = min(lo_v, hi_v), max(lo_v, hi_v)

        model = self.brune_model_combo.currentText().strip().lower() or "brune"
        f_lo = min(0.1, float(self.fit_f1.value()))
        t_star_init = float(self.brune_tstar_spin.value())
        fit_t_star = bool(self.chk_fit_tstar.isChecked())
        dist_km = float(self.dist_km_spin.value())
        vel_kms = float(self.vel_kms_spin.value())

        if not sync:
            if self.brune_worker is not None and self.brune_worker.isRunning():
                self.brune_worker.cancel()
                self.brune_worker.wait(500)
            self.btn_fit_brune.setEnabled(False)
            self.brune_label.setText(f"Inverting {model.capitalize()} w^-2 source spectrum in background...")

            self.brune_worker = BruneInversionWorker(
                tr,
                t_lo=t_lo,
                t_hi=t_hi,
                noise_lo=noise_lo,
                noise_hi=noise_hi,
                model=model,
                f_lo=f_lo,
                f_hi=None,
                t_star_init=t_star_init,
                fit_t_star=fit_t_star,
                distance_km=dist_km,
                vel_kms=vel_kms,
                phase="P",
                parent=self,
            )
            self.brune_worker.finishedResult.connect(self._on_brune_finished)
            self.brune_worker.error.connect(self._on_brune_error)
            self.brune_worker.start()
        else:
            inv_res = invert_brune_spectrum(
                tr,
                t_lo=t_lo,
                t_hi=t_hi,
                noise_lo=noise_lo,
                noise_hi=noise_hi,
                model=model,
                f_lo=f_lo,
                f_hi=None,
                t_star_init=t_star_init,
                fit_t_star=fit_t_star,
                distance_m=max(100.0, dist_km * 1000.0),
                vel_m_s=max(100.0, vel_kms * 1000.0),
            )
            diag = build_brune_fit_diagnostic(inv_res, station=tr.station, phase="P")
            self._on_brune_finished({"inversion": inv_res, "diagnostic": diag})

    def _on_brune_finished(self, payload: Dict[str, Any]) -> None:
        self.btn_fit_brune.setEnabled(True)
        if getattr(self, "_closing", False):
            return
        inv_res = payload.get("inversion")
        diag = payload.get("diagnostic")
        if not inv_res:
            return

        self.last_brune_result = inv_res
        self.last_brune_diagnostic = diag

        fc = float(inv_res["fc"])
        omega0 = float(inv_res["omega0"])
        t_star = float(inv_res["t_star"])
        m0 = float(inv_res["m0"])
        mw = float(inv_res["mw"])
        dsigma_bar = float(inv_res["stress_drop_bar"])
        m_type = str(inv_res.get("model_type", "brune")).capitalize()

        self.brune_label.setText(
            f"{m_type} w^-2 Fit:   fc = {fc:.3f} Hz   |   Omega0 = {omega0:.3e} nm*s   |   "
            f"t* = {t_star:.3f} s   |   M0 = {m0:.3e} N*m   |   Mw = {mw:.2f}   |   "
            f"Delta_sigma = {dsigma_bar:.2f} bar"
        )
        self.brune_label.setStyleSheet("color: #3fb950; font-weight: bold;")

        if self.method_combo.currentIndex() == 4:
            self.method_combo.blockSignals(True)
            self.method_combo.setCurrentIndex(0)
            self.method_combo.blockSignals(False)

        self.replot()
        if diag is not None:
            self.fitDiagnosticReady.emit(diag)

    def _on_brune_error(self, err: str) -> None:
        self.btn_fit_brune.setEnabled(True)
        self.brune_label.setText(f"Brune/Boatwright Fit Error: {err}")
        self.brune_label.setStyleSheet("color: #f85149; font-weight: bold;")

    def _on_canvas_leave(self, _event=None) -> None:
        if hasattr(self, "cursor_label"):
            self.cursor_label.setText("Cursor: —")

    def _on_canvas_motion(self, event) -> None:
        if (
            self._closing
            or self._ax is None
            or not self.curves
            or event.inaxes is not self._ax
            or event.xdata is None
            or event.ydata is None
        ):
            self._on_canvas_leave()
            return
        x, y = float(event.xdata), float(event.ydata)
        if not (math.isfinite(x) and math.isfinite(y)):
            self._on_canvas_leave()
            return
        if self.method_combo.currentIndex() == 4:
            txt = f"Cursor: (t = {x:.2f} s, f = {y:.4g} Hz)"
            if self._spec_grid is not None:
                times, freqs, spec_db = self._spec_grid
                if spec_db.shape == (len(freqs), len(times)) and len(times) > 0 and len(freqs) > 0:
                    ti = int(np.argmin(np.abs(times - x)))
                    fi = int(np.argmin(np.abs(freqs - y)))
                    txt += f"   |   Power = {float(spec_db[fi, ti]):.1f} dB"
            self.cursor_label.setText(txt)
        else:
            txt = f"Cursor: (f = {x:.4g} Hz, A = {y:.3e})"
            if x > 0.0:
                txt += f"   |   T = {1.0 / x:.3g} s"
            self.cursor_label.setText(txt)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        if self.brune_worker is not None and self.brune_worker.isRunning():
            self.brune_worker.cancel()
            self.brune_worker.wait(1000)
        self._ax = None
        self._spec_grid = None
        for cid in getattr(self, "_mpl_cids", ()):
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids = []
        self.canvas._draw_pending = False
        win = self.windowHandle()
        if win is not None:
            try:
                win.removeEventFilter(self.canvas)
            except Exception:
                pass
        super().closeEvent(event)

    def reject(self) -> None:
        if self.brune_worker is not None and self.brune_worker.isRunning():
            self.brune_worker.cancel()
            self.brune_worker.wait(1000)
        super().reject()

    def _spectrum_of(self, c: dict, m_idx: int):
        tr, lo, hi = c["trace"], c["t_lo"], c["t_hi"]
        if m_idx == 0:
            return compute_fft_spectrum(tr, lo, hi, squared=False)
        if m_idx == 1:
            return compute_fft_spectrum(tr, lo, hi, squared=True)
        if m_idx == 2:
            return compute_welch_power_spectrum(tr, lo, hi)
        return compute_mem_spectrum(tr, lo, hi)

    def replot(self) -> None:
        if getattr(self, "_closing", False):
            return
        self.fig.clear()
        self._ax = None
        self._spec_grid = None
        self._on_canvas_leave()
        m_idx = self.method_combo.currentIndex()
        ax = self.fig.add_subplot(111, facecolor="#0d1117")

        if not self.curves:
            ax.text(0.5, 0.5, "No curves. Use Analysis > Spectrum to add the yellow window selection.",
                    color="#8b949e", ha="center", va="center", transform=ax.transAxes)
            self.fit_label.setText("")
        elif m_idx == 4:
            c = self.curves[-1]
            times, freqs, spec_db = compute_spectrogram(c["trace"])
            im = ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="viridis")
            add_dark_colorbar(self.fig, im, ax, "Power (dB)")
            ax.set_xlabel("Time (s)", color="#e6edf3")
            ax.set_ylabel("Frequency (Hz)", color="#e6edf3")
            ax.set_title(f"Hann Spectrogram: {c['trace'].station} {c['trace'].channel3} (latest curve, full trace)",
                         color="#e6edf3")
            self.fit_label.setText("Spectrogram time-frequency representation")
            self._ax = ax
            self._spec_grid = (times, freqs, spec_db)
        else:
            fit_texts = []
            for c in self.curves:
                freqs, amps = self._spectrum_of(c, m_idx)
                ax.loglog(freqs, np.maximum(1e-15, amps), color=c["color"], lw=1.3, label=c["label"])
                fit = fit_log_log_spectrum(freqs, amps, self.fit_f1.value(), self.fit_f2.value())
                if self.chk_fit.isChecked():
                    f_line = np.logspace(math.log10(max(1e-3, fit["f_lo"])), math.log10(max(2e-3, fit["f_hi"])), 50)
                    a_line = (10.0 ** fit["intercept"]) * (f_line ** fit["slope"])
                    ax.loglog(f_line, a_line, color=c["color"], lw=1.6, ls="--", alpha=0.85)
                fit_texts.append(f"{c['label']}: slope b = {fit['slope']:.2f}, plateau = {fit['plateau']:.2e}")

            if self.last_brune_result is not None:
                inv = self.last_brune_result
                bf = np.asarray(inv["freqs"], dtype=np.float64)
                b_obs = np.asarray(inv["obs_spec"], dtype=np.float64)
                b_mod = np.asarray(inv["model_spec"], dtype=np.float64)
                b_noise = np.asarray(inv["noise_spec"], dtype=np.float64)
                pos = bf > 0.0
                if np.any(pos):
                    if m_idx != 0:
                        ax.loglog(bf[pos], np.maximum(1e-15, b_obs[pos]), color="#79c0ff", lw=1.1, alpha=0.75, label="Observed U(f)")
                    m_lbl = str(inv.get("model_type", "brune")).capitalize()
                    fc_val = float(inv["fc"])
                    ax.loglog(
                        bf[pos],
                        np.maximum(1e-15, b_mod[pos]),
                        color="#3fb950",
                        lw=2.1,
                        label=f"{m_lbl} Fit (fc={fc_val:.2f} Hz)",
                    )
                    if b_noise.size == bf.size and np.any(b_noise[pos] > 0.0):
                        ax.loglog(
                            bf[pos],
                            np.maximum(1e-15, b_noise[pos]),
                            color="#f85149",
                            lw=1.3,
                            ls=":",
                            label="Noise Floor",
                        )
                    ax.axvline(fc_val, color="#d2a8ff", lw=1.1, ls="--", label=f"fc = {fc_val:.2f} Hz")

            ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
            ax.set_xlabel("Frequency (Hz)", color="#e6edf3")
            ax.set_ylabel("Spectral Amplitude", color="#e6edf3")
            ax.set_title(self.method_combo.currentText(), color="#e6edf3")
            ax.grid(True, which="both", color="#21262d", ls="--", lw=0.6)
            self.fit_label.setText(
                f"Log-log regression over {self.fit_f1.value():.2f}..{self.fit_f2.value():.2f} Hz   |   "
                + "   |   ".join(fit_texts)
            )
            self._ax = ax

        ax.tick_params(colors="#8b949e")
        self.fig.tight_layout()
        self.canvas.draw_idle()


_HODOGRAM_CMAP = LinearSegmentedColormap.from_list(
    "gyr_hodogram",
    [(0.0, "#2ea043"), (0.5, "#f2cc60"), (1.0, "#f85149")],
)


class ParticleMotionDialog(QDialog):
    """3-Component Particle Motion Hodograms (E-N, N-Z, E-Z & Rotated L-Q-T) with live Azimuth/Incidence sliders."""

    adoptAngles = pyqtSignal(float, float)  # (azimuth_deg, incidence_deg)
    fitDiagnosticReady = pyqtSignal(object)  # FitDiagnosticResult

    def __init__(
        self,
        tr_z: Trace,
        tr_n: Trace,
        tr_e: Trace,
        t_lo: Optional[float] = None,
        t_hi: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"3-Component Particle Motion & MDIR: {tr_z.station}")
        self.resize(860, 580)
        self.tr_z = tr_z
        self.tr_n = tr_n
        self.tr_e = tr_e
        self.t_lo = t_lo
        self.t_hi = t_hi

        mdir = compute_mdir(tr_z, tr_n, tr_e, t_lo, t_hi)
        self.last_mdir: Dict[str, float] = mdir
        self.last_diagnostic: Optional[FitDiagnosticResult] = None

        layout = QVBoxLayout(self)

        self.info_lbl = QLabel(self._format_metrics_text(mdir))
        self.info_lbl.setStyleSheet("color: #3fb950; font-weight: bold;")
        self.info_lbl.setWordWrap(True)
        layout.addWidget(self.info_lbl)

        # Sliders for interactive 3D rotation
        sl_h = QHBoxLayout()
        self.az_spin = QDoubleSpinBox()
        self.az_spin.setRange(0.0, 360.0)
        self.az_spin.setValue(mdir["azimuth"])
        self.az_spin.valueChanged.connect(self.replot)

        self.inc_spin = QDoubleSpinBox()
        self.inc_spin.setRange(0.0, 90.0)
        self.inc_spin.setValue(mdir["incidence"])
        self.inc_spin.valueChanged.connect(self.replot)

        self.chk_rotated = QCheckBox("Plot Rotated (L-Q-T) Hodograms")
        self.chk_rotated.toggled.connect(self.replot)

        sl_h.addWidget(QLabel("Rotation Azimuth (deg):"))
        sl_h.addWidget(self.az_spin)
        sl_h.addWidget(QLabel("Incidence (deg):"))
        sl_h.addWidget(self.inc_spin)
        sl_h.addWidget(self.chk_rotated)
        sl_h.addStretch(1)
        layout.addLayout(sl_h)

        self.fig = Figure(figsize=(8.0, 3.8), dpi=95, facecolor="#10141d")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        bot_h = QHBoxLayout()
        legend_lbl = QLabel("Trajectory Time Gradient: Green (Start) -> Yellow -> Red (End)")
        legend_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        btn_adopt = QPushButton("Adopt Azimuth && Incidence")
        btn_adopt.clicked.connect(lambda: (self.adoptAngles.emit(self.az_spin.value(), self.inc_spin.value()), self.accept()))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_h.addWidget(legend_lbl)
        bot_h.addStretch(1)
        bot_h.addWidget(btn_adopt)
        bot_h.addWidget(btn_close)
        layout.addLayout(bot_h)

        register_tutorial_help(self.az_spin, "3C Rotation Azimuth", "dlg_pmotion_angles", "", "Backazimuth rotation angle (deg)")
        register_tutorial_help(self.inc_spin, "3C Incidence Angle", "dlg_pmotion_angles", "", "Ray incidence angle (deg)")
        register_tutorial_help(self.chk_rotated, "Plot L-Q-T Hodograms", "dlg_pmotion_angles", "", "Show rotated ray-frame L-Q-T particle motion")

        disable_button_autodefault(self)
        apply_c_locale(self)

        self.replot()

    @staticmethod
    def _format_metrics_text(mdir: Dict[str, float]) -> str:
        return (
            f"MDIR Eigensolver:   Backazimuth = {mdir['azimuth']:.1f} deg   |   "
            f"Incidence = {mdir['incidence']:.1f} deg   |   Rectilinearity = {mdir['rectilinearity']:.3f}   |   "
            f"Planarity = {mdir['planarity']:.3f}\n"
            f"Trajectory Misfit RMS_pol = {mdir.get('rms_pol', 0.0):.3g}   |   "
            f"Confidence Cones (sigma_theta, sigma_phi) = "
            f"({mdir.get('sigma_theta_deg', 0.0):.2f} deg, {mdir.get('sigma_phi_deg', 0.0):.2f} deg)"
        )

    def emit_diagnostic(self) -> FitDiagnosticResult:
        """Compute polarization FitDiagnosticResult and emit fitDiagnosticReady."""
        diag = analyze_polarization_diagnostic(
            self.tr_z,
            self.tr_n,
            self.tr_e,
            t_lo=self.t_lo,
            t_hi=self.t_hi,
        )
        self.last_diagnostic = diag
        self.fitDiagnosticReady.emit(diag)
        return diag

    def replot(self) -> None:
        dt = self.tr_z.delta if self.tr_z.delta > 0 else 0.05
        lth = min(self.tr_z.length, self.tr_n.length, self.tr_e.length)
        if self.t_lo is not None and self.t_hi is not None:
            i0 = max(0, int(round(min(self.t_lo, self.t_hi) / dt)))
            i1 = min(lth, int(round(max(self.t_lo, self.t_hi) / dt)) + 1)
        else:
            i0, i1 = 0, min(lth, 400)
        if i1 <= i0 + 2:
            i0, i1 = 0, min(lth, 400)

        mdir = compute_mdir(self.tr_z, self.tr_n, self.tr_e, self.t_lo, self.t_hi)
        self.last_mdir = mdir
        self.info_lbl.setText(self._format_metrics_text(mdir))

        if self.chk_rotated.isChecked():
            tl, tq, tt = rotate_3d_lqt(self.tr_z, self.tr_n, self.tr_e, self.az_spin.value(), self.inc_spin.value())
            c1, c2, c3 = tl.data[i0:i1], tq.data[i0:i1], tt.data[i0:i1]
            names = ("L", "Q", "T")
        else:
            c1 = self.tr_z.data[i0:i1] - np.mean(self.tr_z.data[i0:i1])
            c2 = self.tr_n.data[i0:i1] - np.mean(self.tr_n.data[i0:i1])
            c3 = self.tr_e.data[i0:i1] - np.mean(self.tr_e.data[i0:i1])
            names = ("Z", "N", "E")

        self.fig.clear()
        pairs = [
            (c3, c2, f"{names[2]} vs {names[1]}"),
            (c2, c1, f"{names[1]} vs {names[0]}"),
            (c3, c1, f"{names[2]} vs {names[0]}"),
        ]
        for idx, (x_arr, y_arr, title) in enumerate(pairs, start=1):
            ax = self.fig.add_subplot(1, 3, idx, facecolor="#0d1117")
            if x_arr.size >= 2:
                pts = np.column_stack([x_arr, y_arr]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=_HODOGRAM_CMAP, norm=Normalize(0.0, 1.0), linewidths=1.6)
                lc.set_array(np.linspace(0.0, 1.0, len(segs)))
                ax.add_collection(lc)
                ax.update_datalim(np.column_stack([x_arr, y_arr]))
                ax.autoscale_view()
                ax.plot(x_arr[0], y_arr[0], marker="o", markersize=6, color="#2ea043", label="Start")
                ax.plot(x_arr[-1], y_arr[-1], marker="s", markersize=5, color="#f85149", label="End")
            elif x_arr.size == 1:
                ax.plot(x_arr[0], y_arr[0], marker="o", markersize=7, color="#2ea043", label="Start")
            if idx == 1 and not self.chk_rotated.isChecked():
                # Draw azimuth arrow
                rmax = max(1e-6, float(np.max(np.hypot(x_arr, y_arr)))) if x_arr.size > 0 else 1.0
                az_r = math.radians(self.az_spin.value())
                ax.plot([0, rmax * math.sin(az_r)], [0, rmax * math.cos(az_r)], color="#58a6ff", lw=1.8, ls="--")
            ax.set_title(title, color="#e6edf3", fontsize=9)
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.grid(True, color="#21262d", ls="--", lw=0.6)
            ax.set_aspect("equal", adjustable="datalim")

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.emit_diagnostic()

