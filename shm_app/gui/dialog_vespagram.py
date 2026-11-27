"""Vespagram (velocity spectrum analysis) dialog and background worker."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import (
    add_dark_colorbar,
    apply_c_locale,
    cleanup_mpl_canvases,
    disable_button_autodefault,
    safe_tight_layout,
)
from shm_app.seismology.array_processing import compute_vespagram


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

        self.status_lbl = QLabel("Click 'Compute Vespagram' to calculate slowness-time slant stack.")
        self.status_lbl.setStyleSheet("color: #8b949e; font-weight: bold;")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

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
            self.status_lbl.setStyleSheet("color: #f85149; font-weight: bold;")
            self.status_lbl.setText("Need at least one trace in workspace for Vespagram.")
            return

        slo_lo = float(self.slo_lo_spin.value())
        slo_hi = float(self.slo_hi_spin.value())
        if slo_lo >= slo_hi:
            self._on_vespa_error(f"Slowness Lo ({slo_lo:.2f}) must be less than Slowness Hi ({slo_hi:.2f}).")
            return

        if not sync:
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel()
                self.worker.wait(500)
            self.btn_calc.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.status_lbl.setStyleSheet("color: #58a6ff; font-weight: bold;")
            self.status_lbl.setText("Computing vespagram in background...")

            self.worker = VespagramWorker(
                self.traces,
                azimuth_deg=self.az_spin.value(),
                slow_lo=slo_lo,
                slow_hi=slo_hi,
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
            try:
                v_trs, times, slos, mat = compute_vespagram(
                    self.traces,
                    azimuth_deg=self.az_spin.value(),
                    slow_lo=slo_lo,
                    slow_hi=slo_hi,
                    slow_step=self.slo_step_spin.value(),
                    power=self.root_spin.value(),
                    config=self.config,
                )
            except Exception as exc:
                self._on_vespa_error(str(exc))
                return
            self._on_vespa_finished((v_trs, times, slos, mat))

    def _on_vespa_progress(self, pct: int) -> None:
        self.progress_bar.setValue(pct)

    def _on_vespa_finished(self, result: Tuple[List[Trace], np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.btn_calc.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        v_trs, times, slos, mat = result
        self.vespa_traces = v_trs
        self.status_lbl.setStyleSheet("color: #3fb950; font-weight: bold;")
        self.status_lbl.setText(
            f"Computed {len(v_trs)} vespagram trace(s) over {slos[0]:.2f}..{slos[-1]:.2f} s/deg."
            if len(slos) > 0
            else f"Computed {len(v_trs)} vespagram trace(s)."
        )
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
        safe_tight_layout(self.fig, self.canvas)
        self.canvas.draw_idle()

    def _on_vespa_error(self, err: str) -> None:
        self.btn_calc.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_lbl.setStyleSheet("color: #f85149; font-weight: bold;")
        self.status_lbl.setText(f"Vespagram Error: {err}")

    def cancel_vespa(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
            self.btn_cancel.setEnabled(False)
            self.btn_calc.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_lbl.setStyleSheet("color: #8b949e; font-weight: bold;")
            self.status_lbl.setText("Vespagram calculation cancelled.")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        cleanup_mpl_canvases(self)
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

