"""Background Qt worker thread for FitDiagnosticWindow recomputation."""

from __future__ import annotations

import sys
from typing import Any, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget

from shm_app.core.config_station import get_config
from shm_app.core.fit_models import FitType
from shm_app.core.models import Trace
from shm_app.gui.main_window_utils import (
    _install_qt_excepthook,
    _quote_sh_arg,
    _undoable,
    apply_dark_theme,
)


class _RecomputeFitWorker(QThread):
    """Background worker for >50 ms FitDiagnosticWindow re-fits (FK/plane-wave and Brune spectrum)."""

    finishedDiagnostic = pyqtSignal(int, object, str)  # (seq_id, FitDiagnosticResult, log_msg)
    failed = pyqtSignal(int, str)  # (seq_id, error_msg)

    def __init__(
        self,
        seq_id: int,
        fit_type: FitType,
        *,
        traces: Optional[List[Trace]] = None,
        trace: Optional[Trace] = None,
        t_lo: float = 0.0,
        t_hi: float = 0.0,
        dist_deg: float = 1.0,
        config: Any = None,
        calib_table: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.seq_id = int(seq_id)
        self.fit_type = fit_type
        self.traces = traces or []
        self.trace = trace
        self.t_lo = float(t_lo)
        self.t_hi = float(t_hi)
        self.dist_deg = float(dist_deg)
        self.config = config
        self.calib_table = calib_table
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            if self._is_cancelled:
                return
            if self.fit_type in (FitType.FK_SEMBLANCE, FitType.PLANE_WAVE):
                from shm_app.seismology.array_loc import (
                    build_fk_fit_diagnostic,
                    compute_array_response_function,
                    compute_beam_diagnostics,
                    compute_fk_analysis,
                )

                cfg = self.config or getattr(sys.modules.get("shm_app.gui.main_window"), "get_config", get_config)()
                fk_res = compute_fk_analysis(
                    self.traces,
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    f_lo=0.5,
                    f_hi=3.0,
                    max_slowness=15.0,
                    grid_points=51,
                    config=cfg,
                    progress_callback=lambda _cur, _tot: not self._is_cancelled,
                    calib_table=self.calib_table,
                    apply_calibration=self.calib_table is not None,
                )
                if self._is_cancelled or not fk_res:
                    return
                beam_diag = compute_beam_diagnostics(
                    self.traces,
                    azimuth_deg=float(fk_res["azimuth"]),
                    slowness_s_deg=float(fk_res["slowness"]),
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    config=cfg,
                )
                if self._is_cancelled:
                    return
                rel_coords = cfg.relative_xy_km(
                    [t.station for t in self.traces],
                    ref_station="CENTRE",
                )
                x_km = [c[0] for c in rel_coords]
                y_km = [c[1] for c in rel_coords]
                arf = compute_array_response_function(
                    x_km,
                    y_km,
                    slowness_max=15.0,
                    n_grid=51,
                    freq_hz=1.5,
                )
                if self._is_cancelled:
                    return
                diag = build_fk_fit_diagnostic(
                    fk_res,
                    beam_diag,
                    arf_result=arf,
                    calib_info=fk_res,
                )
                msg = (
                    f"Recomputed {self.fit_type.value}: BAZ={fk_res['azimuth']:.1f} deg, "
                    f"Slow={fk_res['slowness']:.2f} s/deg"
                )
                if not self._is_cancelled:
                    self.finishedDiagnostic.emit(self.seq_id, diag, msg)
                return

            if self.fit_type == FitType.BRUNE_SPECTRUM:
                if self.trace is None:
                    raise ValueError("No trace provided for Brune spectrum recomputation.")
                from shm_app.processing.spectrum import build_brune_fit_diagnostic, invert_brune_spectrum

                inv_res = invert_brune_spectrum(
                    self.trace,
                    t_lo=self.t_lo,
                    t_hi=self.t_hi,
                    distance_m=float(self.dist_deg) * 111194.9266,
                )
                if self._is_cancelled:
                    return
                diag = build_brune_fit_diagnostic(inv_res, station=self.trace.station)
                if not self._is_cancelled:
                    self.finishedDiagnostic.emit(self.seq_id, diag, "")
                return
        except Exception as exc:
            if not self._is_cancelled:
                prefix = (
                    "Brune re-fit error"
                    if self.fit_type == FitType.BRUNE_SPECTRUM
                    else f"Re-fit error ({self.fit_type.value})"
                )
                self.failed.emit(self.seq_id, f"{prefix}: {exc}")


