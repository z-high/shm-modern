"""Interactive filter designer and simulation filter selector dialog."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import (
    apply_c_locale,
    cleanup_mpl_canvases,
    disable_button_autodefault,
    safe_tight_layout,
)
from shm_app.processing.filters import (
    STANDARD_SIMULATION_FILTERS,
    evaluate_stage_freq,
    load_named_filter,
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

        safe_tight_layout(self.fig, self.canvas)
        self.canvas.draw_idle()

    def closeEvent(self, event: QCloseEvent) -> None:
        cleanup_mpl_canvases(self)
        super().closeEvent(event)

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

