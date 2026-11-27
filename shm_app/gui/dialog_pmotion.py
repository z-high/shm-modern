"""3C Particle Motion / Hodogram polarization analysis dialog."""

from __future__ import annotations

import math
import sys
from typing import Dict, Optional
import numpy as np

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from shm_app.core.fit_models import FitDiagnosticResult
from shm_app.core.models import Trace
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import (
    apply_c_locale,
    cleanup_mpl_canvases,
    disable_button_autodefault,
    safe_tight_layout,
)
from shm_app.processing.polarization import (
    analyze_polarization_diagnostic,
    compute_mdir,
    rotate_3d_lqt,
)


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
        self.btn_emit_diag = QPushButton("Update Fit Diagnostics")
        self.btn_emit_diag.clicked.connect(self.emit_diagnostic)
        btn_adopt = QPushButton("Adopt Azimuth && Incidence")
        btn_adopt.clicked.connect(lambda: (self.adoptAngles.emit(self.az_spin.value(), self.inc_spin.value()), self.accept()))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_h.addWidget(legend_lbl)
        bot_h.addStretch(1)
        bot_h.addWidget(self.btn_emit_diag)
        bot_h.addWidget(btn_adopt)
        bot_h.addWidget(btn_close)
        layout.addLayout(bot_h)

        register_tutorial_help(self.az_spin, "3C Rotation Azimuth", "dlg_pmotion_angles", "", "Backazimuth rotation angle (deg)")
        register_tutorial_help(self.inc_spin, "3C Incidence Angle", "dlg_pmotion_angles", "", "Ray incidence angle (deg)")
        register_tutorial_help(self.chk_rotated, "Plot L-Q-T Hodograms", "dlg_pmotion_angles", "", "Show rotated ray-frame L-Q-T particle motion")
        register_tutorial_help(self.btn_emit_diag, "Send 3C Diagnostics", "dlg_pmotion_angles", "", "Compute 3C polarization uncertainty and display in Fit Diagnostics dock")

        disable_button_autodefault(self)
        apply_c_locale(self)

        self.replot()

    def closeEvent(self, event: QCloseEvent) -> None:
        cleanup_mpl_canvases(self)
        super().closeEvent(event)

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

        safe_tight_layout(self.fig, self.canvas)
        self.canvas.draw_idle()


