"""Wideband Frequency-Wavenumber (FK) array analysis dialog and background worker."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
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
from shm_app.seismology.array_processing import (
    SlownessCalibrationTable,
    build_fk_fit_diagnostic,
    compute_array_response_function,
    compute_beam_diagnostics,
    compute_fk_analysis,
)
from shm_app.seismology.calibration import load_calibration_for_array


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
            self.res_label.setStyleSheet("font-weight: bold; color: #f85149;")
            self.res_label.setText("Need at least 3 array traces in workspace for FK analysis.")
            return

        f_lo = float(self.flo_spin.value())
        f_hi = float(self.fhi_spin.value())
        if f_lo >= f_hi:
            self._on_fk_error(f"Freq Lo ({f_lo:.2f} Hz) must be less than Freq Hi ({f_hi:.2f} Hz).")
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
            self.res_label.setStyleSheet("font-weight: bold; color: #58a6ff;")
            self.res_label.setText("Computing FK grid search, ARF, and beam diagnostics in background...")

            self.worker = FKWorker(
                self.traces,
                t_lo=self.t_lo,
                t_hi=self.t_hi,
                f_lo=f_lo,
                f_hi=f_hi,
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
            try:
                res = _compute_fk_bundle(
                    self.traces,
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    f_lo=f_lo,
                    f_hi=f_hi,
                    max_slowness=self.max_slo_spin.value(),
                    grid_points=self.grid_spin.value(),
                    config=self.config,
                    calib_table=self.calib_table,
                    apply_calibration=apply_cal,
                )
            except Exception as exc:
                self._on_fk_error(str(exc))
                return
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
        self.res_label.setStyleSheet("font-weight: bold; color: #58a6ff;")

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

        safe_tight_layout(self.fig, self.canvas)
        self.canvas.draw_idle()

        if self.last_diagnostic is not None:
            self.fitDiagnosticReady.emit(self.last_diagnostic)

    def _on_fk_error(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.res_label.setStyleSheet("font-weight: bold; color: #f85149;")
        self.res_label.setText(f"FK Analysis Error: {err}")

    def cancel_fk(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
            self.btn_cancel.setEnabled(False)
            self.btn_run.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.res_label.setStyleSheet("font-weight: bold; color: #8b949e;")
            self.res_label.setText("FK analysis cancelled.")

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

    def _on_adopt(self) -> None:
        if self.last_result is not None:
            az = self.last_result["azimuth"]
            slo = self.last_result["slowness"]
            if self.chk_auto_locate.isChecked():
                self.adoptAndLocate.emit(az, slo)
            else:
                self.adoptParameters.emit(az, slo)
            self.accept()

