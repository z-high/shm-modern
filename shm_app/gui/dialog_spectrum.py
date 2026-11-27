"""Spectral analysis (FFT, Welch PSD, MEM, Spectrogram, Brune source inversion) dialog."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import FitDiagnosticResult
from shm_app.core.models import Trace
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import (
    add_dark_colorbar,
    apply_c_locale,
    cleanup_mpl_canvases,
    disable_button_autodefault,
    safe_tight_layout,
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

        self.btn_clear = QPushButton("Clear Curves")
        self.btn_clear.clicked.connect(self.clear_curves)

        ctrl_grid.addWidget(QLabel("Method:"), 0, 0)
        ctrl_grid.addWidget(self.method_combo, 0, 1, 1, 3)
        ctrl_grid.addWidget(self.btn_clear, 0, 5)
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
        register_tutorial_help(self.btn_clear, "Clear Spectrum Curves", "dlg_spec_method", "", "Remove all overlaid spectral curves and reset inversion state")
        register_tutorial_help(self.brune_model_combo, "Source Spectrum Model", "dlg_spec_method", "", "Select Brune (gamma=1) or Boatwright (gamma=2) w^-2 model")
        register_tutorial_help(self.brune_tstar_spin, "Attenuation t*", "dlg_spec_method", "", "Initial or fixed attenuation operator t* in seconds")
        register_tutorial_help(self.chk_fit_tstar, "Jointly Fit Attenuation t*", "dlg_spec_method", "", "Allow nonlinear inversion to optimize t* alongside Omega0 and fc")
        register_tutorial_help(self.dist_km_spin, "Epicentral Distance (km)", "dlg_spec_method", "", "Hypocentral/epicentral distance for seismic moment M0")
        register_tutorial_help(self.vel_kms_spin, "Phase Velocity (km/s)", "dlg_spec_method", "", "Wave speed for M0, source radius r0, and stress drop")
        register_tutorial_help(self.chk_use_noise, "Enable Noise Window", "dlg_spec_method", "", "Compute and plot pre-signal noise floor over the specified trace-relative window")
        register_tutorial_help(self.noise_lo_spin, "Noise Window Start (s)", "dlg_spec_method", "", "Start of trace-relative pre-signal noise window in seconds")
        register_tutorial_help(self.noise_hi_spin, "Noise Window End (s)", "dlg_spec_method", "", "End of trace-relative pre-signal noise window in seconds")
        register_tutorial_help(self.btn_fit_brune, "Run Brune/Boatwright Inversion", "dlg_spec_method", "", "Invert displacement spectrum for fc, Omega0, t*, M0, Mw, and stress drop")

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

        fit_f1 = float(self.fit_f1.value())
        fit_f2 = float(self.fit_f2.value())
        if fit_f1 >= fit_f2:
            self._on_brune_error(f"Fit Band Low ({fit_f1:.2f} Hz) must be less than High ({fit_f2:.2f} Hz).")
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
        f_lo = min(0.1, fit_f1)
        t_star_init = float(self.brune_tstar_spin.value())
        fit_t_star = bool(self.chk_fit_tstar.isChecked())
        dist_km = float(self.dist_km_spin.value())
        vel_kms = float(self.vel_kms_spin.value())

        if not sync:
            if self.brune_worker is not None and self.brune_worker.isRunning():
                self.brune_worker.cancel()
                self.brune_worker.wait(500)
            self.btn_fit_brune.setEnabled(False)
            self.brune_label.setStyleSheet("color: #58a6ff; font-weight: bold;")
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
            try:
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
            except Exception as exc:
                self._on_brune_error(str(exc))
                return
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
        cleanup_mpl_canvases(self)
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
            self.fit_label.setStyleSheet("color: #ffa657; font-weight: bold;")
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
            self.fit_label.setStyleSheet("color: #ffa657; font-weight: bold;")
            self.fit_label.setText("Spectrogram time-frequency representation")
            self._ax = ax
            self._spec_grid = (times, freqs, spec_db)
        else:
            f1 = float(self.fit_f1.value())
            f2 = float(self.fit_f2.value())
            valid_band = f1 < f2
            fit_texts = []
            for c in self.curves:
                freqs, amps = self._spectrum_of(c, m_idx)
                ax.loglog(freqs, np.maximum(1e-15, amps), color=c["color"], lw=1.3, label=c["label"])
                if valid_band:
                    fit = fit_log_log_spectrum(freqs, amps, f1, f2)
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
            if valid_band:
                self.fit_label.setStyleSheet("color: #ffa657; font-weight: bold;")
                self.fit_label.setText(
                    f"Log-log regression over {f1:.2f}..{f2:.2f} Hz   |   "
                    + "   |   ".join(fit_texts)
                )
            else:
                self.fit_label.setStyleSheet("color: #f85149; font-weight: bold;")
                self.fit_label.setText(
                    f"Fit Band Error: Low freq ({f1:.2f} Hz) must be less than High freq ({f2:.2f} Hz)."
                )
            self._ax = ax

        ax.tick_params(colors="#8b949e")
        safe_tight_layout(self.fig, self.canvas)
        self.canvas.draw_idle()

