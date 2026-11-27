"""Interactive Single-Trace Magnifier Canvas for high-precision SHM phase picking."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget

from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.canvas_helpers import (
    CURSOR_LOCK_MODES,
    MAGNIFIER_CURSOR_MODES,
    _alloc_pixmap_for_widget,
    _build_trace_polygon,
    _draw_pick_uncertainty_interval,
    _pixmap_matches_widget,
    snap_time_to_feature,
)
from shm_app.processing.trace_ops import hilbert_transform


class MagnifierCanvas(QWidget):
    """Single-trace high-resolution zoomed Magnifier panel matching SHM's Magnifier Window.

    Uses an offscreen QPixmap backing store for the zoomed waveform + picks so cursor hover
    and amplitude measurement drag run with sub-millisecond latency.
    """

    pickPlaced = pyqtSignal(int, float)  # (trace_idx, epoch_time)
    pickDeleteRequested = pyqtSignal(int, str)  # (trace_idx, phase_name)
    pickMoved = pyqtSignal(int, str, float)  # (trace_idx, phase_name, new_epoch)
    pickChanged = pyqtSignal(int, object)  # (trace_idx, PhasePick)
    pickEditStarted = pyqtSignal(int, str)  # (trace_idx, label)
    amplitudeMeasured = pyqtSignal(int, float, float)  # (trace_idx, rel_t_lo, rel_t_hi)
    cursorModeCycled = pyqtSignal(str)  # new cursor lock mode
    cursorOverlayModeChanged = pyqtSignal(str)  # new overlay mode ('STANDARD', 'HILBERT', 'REFERENCE')

    ERROR_BAR_HIT_TOLERANCE_PX = 7.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 145)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self.trace: Optional[Trace] = None
        self.trace_idx: int = 0
        self.ref_epoch: float = 0.0
        self.t_lo: float = 0.0
        self.t_hi: float = 30.0
        self.cursor_lock_mode: str = "CROSSHAIR"
        self.cursor_overlay_mode: str = "STANDARD"
        self.show_pick_error_bars: bool = True
        self.depth_phase_cursors: List[Tuple[str, float]] = []  # [(phase, offset_sec), ...]

        self._reference_waveform: Optional[np.ndarray] = None
        self._reference_delta: float = 0.05
        self._reference_center_rel_sec: float = 0.0
        self._ref_version: int = 0
        self._hilbert_cache_key: Optional[Tuple] = None
        self._hilbert_cache_data: Optional[np.ndarray] = None
        self._hilbert_cache_delta: float = 0.05
        self._hilbert_cache_center_rel_sec: float = 0.0

        self._drag_start_rel: Optional[float] = None
        self._drag_cur_rel: Optional[float] = None
        self._hover_x: Optional[float] = None
        self._error_bar_drag: Optional[PhasePick] = None
        self._hover_pick: Optional[PhasePick] = None
        self._pick_drag: Optional[PhasePick] = None
        self._pick_drag_epoch: Optional[float] = None
        self._pick_drag_press_x: float = 0.0

        self._static_pixmap: Optional[QPixmap] = None
        self._cache_dirty: bool = True
        self._cache_state_key: Optional[Tuple] = None
        self._font = QFont("Monospace", 8)
        self._font.setStyleHint(QFont.StyleHint.TypeWriter)

    def invalidate_cache(self) -> None:
        self._cache_dirty = True
        self.update()

    def set_show_pick_error_bars(self, show: bool) -> None:
        val = bool(show)
        if val != self.show_pick_error_bars:
            self.show_pick_error_bars = val
            self.invalidate_cache()

    def set_cursor_lock_mode(self, mode: str) -> None:
        if mode in CURSOR_LOCK_MODES and mode != self.cursor_lock_mode:
            self.cursor_lock_mode = mode
            self.invalidate_cache()

    def set_cursor_overlay_mode(self, mode: str) -> None:
        m = (mode or "STANDARD").strip().upper()
        if m == "CROSSHAIR":
            m = "STANDARD"
        if m in MAGNIFIER_CURSOR_MODES and m != self.cursor_overlay_mode:
            self.cursor_overlay_mode = m
            self._hilbert_cache_key = None
            self.cursorOverlayModeChanged.emit(m)
            self.invalidate_cache()

    def set_cursor_mode(self, mode: str) -> None:
        """Unified cursor mode setter supporting both CURSOR_LOCK_MODES and MAGNIFIER_CURSOR_MODES."""
        m = (mode or "").strip().upper()
        if m in CURSOR_LOCK_MODES:
            self.set_cursor_lock_mode(m)
        elif m in MAGNIFIER_CURSOR_MODES:
            self.set_cursor_overlay_mode(m)

    def set_reference_waveform(
        self,
        trace_or_array,
        delta: Optional[float] = None,
        center_rel_sec: float = 0.0,
    ) -> None:
        """Set reference waveform (Trace or 1D numpy array) for REFERENCE / HILBERT cursor overlays."""
        if trace_or_array is None:
            self._reference_waveform = None
            self._ref_version += 1
            self._hilbert_cache_key = None
            self.invalidate_cache()
            return
        if isinstance(trace_or_array, Trace):
            arr = np.asarray(trace_or_array.data, dtype=np.float64)
            dt = float(delta) if (delta is not None and delta > 0) else float(trace_or_array.delta)
        else:
            arr = np.asarray(trace_or_array, dtype=np.float64).ravel()
            default_dt = self.trace.delta if (self.trace is not None and self.trace.delta > 0) else 0.05
            dt = float(delta) if (delta is not None and delta > 0) else float(default_dt)
        self._reference_waveform = arr.copy()
        self._reference_delta = max(1e-6, dt)
        self._reference_center_rel_sec = float(center_rel_sec)
        self._ref_version += 1
        self._hilbert_cache_key = None
        self.invalidate_cache()

    def capture_reference_window(
        self,
        t_lo: Optional[float] = None,
        t_hi: Optional[float] = None,
        center_rel_sec: Optional[float] = None,
    ) -> bool:
        """Capture the current magnifier window (or [t_lo, t_hi]) as the reference waveform."""
        if self.trace is None or self.trace.length < 2 or self.trace.delta <= 0:
            return False
        lo = self.t_lo if t_lo is None else float(t_lo)
        hi = self.t_hi if t_hi is None else float(t_hi)
        if hi < lo:
            lo, hi = hi, lo
        tr_rel_start = self.trace.start_epoch + self.trace.display_offset - self.ref_epoch
        i0 = max(0, int(math.floor((lo - tr_rel_start) / max(1e-9, self.trace.delta))))
        i1 = min(self.trace.length, int(math.ceil((hi - tr_rel_start) / max(1e-9, self.trace.delta))) + 1)
        if i1 - i0 < 2:
            return False
        seg = self.trace.data[i0:i1].copy()
        c_sec = 0.5 * (seg.size - 1) * self.trace.delta if center_rel_sec is None else float(center_rel_sec)
        self.set_reference_waveform(seg, delta=self.trace.delta, center_rel_sec=c_sec)
        return True

    def set_magnifier_view(
        self,
        trace: Optional[Trace],
        trace_idx: int,
        ref_epoch: float,
        t_lo: float,
        t_hi: float,
    ) -> None:
        self.trace = trace
        self.trace_idx = trace_idx
        self.ref_epoch = ref_epoch
        self.t_lo = t_lo
        self.t_hi = max(t_lo + 0.5, t_hi)
        self._hover_pick = None
        self._pick_drag = None
        self._pick_drag_epoch = None
        self._error_bar_drag = None
        self._hilbert_cache_key = None
        self.invalidate_cache()

    def plot_rect(self) -> QRectF:
        return QRectF(70, 22, max(80, self.width() - 86), max(40, self.height() - 44))

    def time_to_x(self, rel_t: float) -> float:
        r = self.plot_rect()
        return r.left() + (rel_t - self.t_lo) / max(1e-9, self.t_hi - self.t_lo) * r.width()

    def x_to_time(self, x: float) -> float:
        r = self.plot_rect()
        return self.t_lo + (x - r.left()) / max(1.0, r.width()) * (self.t_hi - self.t_lo)

    def _current_state_key(self) -> Tuple:
        tr = self.trace
        return (
            self.width(),
            self.height(),
            id(tr),
            self.trace_idx,
            round(self.ref_epoch, 3),
            round(self.t_lo, 4),
            round(self.t_hi, 4),
            self.cursor_lock_mode,
            self.cursor_overlay_mode,
            self.show_pick_error_bars,
            self._ref_version,
            tr.filter_name if tr else "",
            getattr(tr, "overlay_filter_name", "NONE") if tr else "",
            getattr(tr, "overlay_filter_color", "") if tr else "",
            len(tr.picks) if tr else 0,
            tuple((p.name, p.onset, p.quality, p.sign, p.spec) for p in tr.picks) if tr else (),
            round(float(self.devicePixelRatioF()), 3),
        )

    def _get_cursor_overlay_waveform(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """Return (samples, dt, center_rel_sec) for HILBERT or REFERENCE cursor overlay."""
        mode = self.cursor_overlay_mode
        if mode == "REFERENCE":
            if self._reference_waveform is None or self._reference_waveform.size < 2:
                return None
            return (self._reference_waveform, self._reference_delta, self._reference_center_rel_sec)

        if mode != "HILBERT":
            return None

        if self._reference_waveform is not None and self._reference_waveform.size >= 2:
            hkey = ("REF", self._ref_version, self._reference_delta, self._reference_waveform.size)
            if self._hilbert_cache_key != hkey:
                seg = self._reference_waveform - float(np.mean(self._reference_waveform))
                self._hilbert_cache_data = hilbert_transform(seg)
                self._hilbert_cache_delta = self._reference_delta
                self._hilbert_cache_center_rel_sec = self._reference_center_rel_sec
                self._hilbert_cache_key = hkey
            if self._hilbert_cache_data is not None:
                return (self._hilbert_cache_data, self._hilbert_cache_delta, self._hilbert_cache_center_rel_sec)
            return None

        if self.trace is None or self.trace.length < 2 or self.trace.delta <= 0:
            return None

        tr = self.trace
        tr_rel_start = tr.start_epoch + tr.display_offset - self.ref_epoch
        # Prefer segment around an in-view pick (e.g. P onset), else current magnifier window
        ref_pk: Optional[PhasePick] = None
        for pk in tr.picks:
            if pk.onset:
                rel_pk = pk.onset_epoch + tr.display_offset - self.ref_epoch
                if self.t_lo <= rel_pk <= self.t_hi:
                    ref_pk = pk
                    break
        if ref_pk is None:
            for pk in tr.picks:
                if pk.onset:
                    ref_pk = pk
                    break

        if ref_pk is not None:
            rel_pk = ref_pk.onset_epoch + tr.display_offset - self.ref_epoch
            half_w = min(5.0, max(1.0, 0.25 * (self.t_hi - self.t_lo)))
            w_lo = rel_pk - half_w
            w_hi = rel_pk + half_w
            i0 = max(0, int(math.floor((w_lo - tr_rel_start) / tr.delta)))
            i1 = min(tr.length, int(math.ceil((w_hi - tr_rel_start) / tr.delta)) + 1)
            if i1 - i0 >= 2:
                center_rel = rel_pk - (tr_rel_start + i0 * tr.delta)
            else:
                i0 = max(0, int(math.floor((self.t_lo - tr_rel_start) / tr.delta)))
                i1 = min(tr.length, int(math.ceil((self.t_hi - tr_rel_start) / tr.delta)) + 1)
                center_rel = 0.5 * max(0, i1 - i0 - 1) * tr.delta
        else:
            i0 = max(0, int(math.floor((self.t_lo - tr_rel_start) / tr.delta)))
            i1 = min(tr.length, int(math.ceil((self.t_hi - tr_rel_start) / tr.delta)) + 1)
            center_rel = 0.5 * max(0, i1 - i0 - 1) * tr.delta

        if i1 - i0 < 2:
            return None

        hkey = ("TRACE", id(tr), tr.length, i0, i1, round(center_rel, 4))
        if self._hilbert_cache_key != hkey:
            seg = tr.data[i0:i1] - float(np.mean(tr.data[i0:i1]))
            self._hilbert_cache_data = hilbert_transform(seg)
            self._hilbert_cache_delta = tr.delta
            self._hilbert_cache_center_rel_sec = center_rel
            self._hilbert_cache_key = hkey
        if self._hilbert_cache_data is not None:
            return (self._hilbert_cache_data, self._hilbert_cache_delta, self._hilbert_cache_center_rel_sec)
        return None

    def _ensure_static_pixmap(self) -> None:
        key = self._current_state_key()
        if (
            not self._cache_dirty
            and _pixmap_matches_widget(self._static_pixmap, self)
            and self._cache_state_key == key
        ):
            return

        if not _pixmap_matches_widget(self._static_pixmap, self):
            self._static_pixmap = _alloc_pixmap_for_widget(self)

        painter = QPainter(self._static_pixmap)
        painter.fillRect(self.rect(), QColor("#0d1117"))

        r = self.plot_rect()
        painter.setPen(QPen(QColor("#30363d"), 1.2))
        painter.drawRect(r)
        painter.setFont(self._font)

        if self.trace is None or self.trace.length == 0:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "Magnifier Window (click or right-drag on a trace)")
            painter.end()
            self._cache_dirty = False
            self._cache_state_key = key
            return

        tr = self.trace
        tr_rel_start = tr.start_epoch + tr.display_offset - self.ref_epoch
        i0 = max(0, int(math.floor((self.t_lo - tr_rel_start) / max(1e-9, tr.delta))))
        i1 = min(tr.length, int(math.ceil((self.t_hi - tr_rel_start) / max(1e-9, tr.delta))) + 1)

        painter.setPen(QColor("#ffd700"))
        ov_tag = (
            f"+{tr.overlay_filter_name}"
            if getattr(tr, "overlay_filter_name", "NONE") not in ("", "NONE")
            else ""
        )
        mode_tag = f"  Cursor: {self.cursor_overlay_mode}" if self.cursor_overlay_mode != "STANDARD" else ""
        hdr = (
            f"MAGNIFIER: #{self.trace_idx + 1} {tr.station} {tr.channel3} [{tr.filter_name}{ov_tag}]  "
            f"[{self.t_lo:.2f}..{self.t_hi:.2f}s]  Lock: {self.cursor_lock_mode}{mode_tag}  "
            f"(L-click: Pick | Dbl-click: Del | M-click: Lock | R-drag: Amp)"
        )
        painter.drawText(QRectF(72, 2, r.width(), 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hdr)

        if i1 <= i0:
            painter.end()
            self._cache_dirty = False
            self._cache_state_key = key
            return

        seg = tr.data[i0:i1]
        finite_mask = np.isfinite(seg)
        if np.any(finite_mask):
            finite_seg = seg[finite_mask]
            mean_v = float(np.mean(finite_seg))
            max_amp = max(1e-9, float(np.max(np.abs(finite_seg - mean_v))))
        else:
            mean_v = 0.0
            max_amp = 1.0
        y_mid = r.center().y()
        scale_y = 0.44 * r.height() / max_amp

        # Subtle vertical time gridlines and bottom time-axis ticks/labels
        span_t = max(1e-6, self.t_hi - self.t_lo)
        target_ticks = max(4, min(12, int(r.width() // 95)))
        raw_step = span_t / float(target_ticks)
        nice_steps = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
        tick_step = next((s for s in nice_steps if s >= raw_step), raw_step)
        first_tick = math.ceil(self.t_lo / tick_step) * tick_step
        grid_pen = QPen(QColor("#1c2432"), 1.0, Qt.PenStyle.DotLine)
        tick_pen = QPen(QColor("#484f58"), 1.0)
        t_cur = first_tick
        tick_guard = 0
        while t_cur <= self.t_hi + 1e-6 and tick_guard < 40:
            tick_guard += 1
            tx = self.time_to_x(t_cur)
            if r.left() + 1.0 <= tx <= r.right() - 1.0:
                painter.setPen(grid_pen)
                painter.drawLine(QPointF(tx, r.top() + 1.0), QPointF(tx, r.bottom() - 1.0))
                painter.setPen(tick_pen)
                painter.drawLine(QPointF(tx, r.bottom()), QPointF(tx, r.bottom() + 4.0))
                lbl = f"{t_cur:.2f}s" if tick_step < 0.2 else (f"{t_cur:.1f}s" if tick_step < 2.0 else f"{t_cur:.0f}s")
                painter.setPen(QColor("#8b949e"))
                painter.drawText(
                    QRectF(tx - 34.0, r.bottom() + 3.0, 68.0, 16.0),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    lbl,
                )
            t_cur += tick_step

        # Zero line
        painter.setPen(QPen(QColor("#21262d"), 0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(r.left(), y_mid), QPointF(r.right(), y_mid))

        # Amplitude scale labels in left margin
        painter.setPen(QColor("#8b949e"))
        painter.drawText(QRectF(4, r.top(), 62, 16), Qt.AlignmentFlag.AlignRight, f"+{max_amp:.1f}")
        painter.drawText(QRectF(4, y_mid - 8, 62, 16), Qt.AlignmentFlag.AlignRight, "0.0")
        painter.drawText(QRectF(4, r.bottom() - 16, 62, 16), Qt.AlignmentFlag.AlignRight, f"-{max_amp:.1f}")

        poly = _build_trace_polygon(
            tr.data,
            i0,
            i1,
            tr_rel_start,
            tr.delta,
            self.t_lo,
            self.t_hi,
            r.left(),
            r.width(),
            y_mid,
            scale_y,
            r.top() + 1.0,
            r.bottom() - 1.0,
            mean_v,
        )

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#79c0ff"), 1.0))
        painter.drawPolyline(poly)

        # Draw sample dots when zoomed in tightly (< 180 samples)
        if (i1 - i0) <= 180:
            painter.setPen(QPen(QColor("#ffd700"), 3.5))
            painter.drawPoints(poly)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Draw picks and [±sigma_t] uncertainty error bars on this trace
        if tr.picks:
            pick_pen = QPen(QColor("#ff7b72"), 1.8)
            for pk in tr.picks:
                if not pk.onset:
                    continue
                rel_pk = pk.onset_epoch + tr.display_offset - self.ref_epoch
                sigma = pk.sigma_t(tr.delta)
                if self.show_pick_error_bars and (rel_pk + sigma >= self.t_lo and rel_pk - sigma <= self.t_hi):
                    _draw_pick_uncertainty_interval(
                        painter,
                        self.time_to_x(rel_pk - sigma),
                        self.time_to_x(rel_pk + sigma),
                        r.top(),
                        r.height(),
                        r.left(),
                        r.right(),
                        QColor("#ff7b72"),
                    )
                if self.t_lo <= rel_pk <= self.t_hi:
                    px = self.time_to_x(rel_pk)
                    painter.setPen(pick_pen)
                    painter.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
                    painter.drawText(
                        QPointF(px + 4, r.top() + 14),
                        f"{pk.spec}{pk.sign_char}{pk.name} (Q{pk.quality} ±{sigma:.2g}s)",
                    )

        painter.end()
        self._cache_dirty = False
        self._cache_state_key = key

    def paintEvent(self, _event) -> None:
        self._ensure_static_pixmap()

        painter = QPainter(self)
        if self._static_pixmap is not None:
            painter.drawPixmap(0, 0, self._static_pixmap)

        if self.trace is None or self.trace.length == 0:
            painter.end()
            return

        r = self.plot_rect()
        painter.setFont(self._font)

        # Draw right-drag amplitude measurement box
        if self._drag_start_rel is not None and self._drag_cur_rel is not None:
            x0 = self.time_to_x(min(self._drag_start_rel, self._drag_cur_rel))
            x1 = self.time_to_x(max(self._drag_start_rel, self._drag_cur_rel))
            painter.fillRect(QRectF(x0, r.top(), max(2.0, x1 - x0), r.height()), QColor(63, 185, 80, 55))
            painter.setPen(QPen(QColor("#3fb950"), 1.2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x0, r.top(), max(2.0, x1 - x0), r.height()))

        # Highlight hovered pick or live-drag preview
        if self._pick_drag is not None and self._pick_drag_epoch is not None:
            px = self.time_to_x(self._pick_drag_epoch + self.trace.display_offset - self.ref_epoch)
            if r.left() <= px <= r.right():
                painter.setPen(QPen(QColor("#ffd700"), 2.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
                painter.setPen(QColor("#ffd700"))
                painter.drawText(QPointF(px + 4, r.top() + 26), format_sh_time(self._pick_drag_epoch)[-12:])
        elif self._hover_pick is not None and self._hover_pick.onset:
            px = self.time_to_x(self._hover_pick.onset_epoch + self.trace.display_offset - self.ref_epoch)
            if r.left() <= px <= r.right():
                painter.setPen(QPen(QColor("#ffd700"), 2.5))
                painter.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))

        # Draw hover cursor, Hilbert/Reference overlay waveform, and depth-phase check cursors
        if self._hover_x is not None and r.left() <= self._hover_x <= r.right():
            hover_t = self.x_to_time(self._hover_x)
            if self.cursor_lock_mode != "CROSSHAIR":
                hover_ep = self.ref_epoch + hover_t - self.trace.display_offset
                snapped_ep = snap_time_to_feature(self.trace, hover_ep, self.cursor_lock_mode)
                hover_t = snapped_ep + self.trace.display_offset - self.ref_epoch
            draw_x = self.time_to_x(hover_t)

            ov = self._get_cursor_overlay_waveform()
            if ov is not None:
                ov_arr, ov_dt, ov_center = ov
                if ov_arr.size >= 2 and ov_dt > 0:
                    ov_fm = np.isfinite(ov_arr)
                    if np.any(ov_fm):
                        ov_mean = float(np.mean(ov_arr[ov_fm]))
                        ov_dm = np.where(ov_fm, ov_arr - ov_mean, 0.0)
                        ov_max = float(np.max(np.abs(ov_dm)))
                    else:
                        ov_dm = ov_arr
                        ov_max = 0.0
                    if ov_max > 1e-12:
                        y_mid = r.center().y()
                        ov_scale_y = 0.36 * r.height() / ov_max
                        ov_rel_start = hover_t - ov_center
                        ov_poly = _build_trace_polygon(
                            ov_dm,
                            0,
                            ov_dm.size,
                            ov_rel_start,
                            ov_dt,
                            self.t_lo,
                            self.t_hi,
                            r.left(),
                            r.width(),
                            y_mid,
                            ov_scale_y,
                            r.top() + 1.0,
                            r.bottom() - 1.0,
                            0.0,
                        )
                        if ov_poly.count() >= 2:
                            painter.save()
                            painter.setClipRect(r)
                            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                            if self.cursor_overlay_mode == "HILBERT":
                                ov_col = QColor(210, 168, 255, 205)
                                badge = "HILBERT 90°"
                            else:
                                ov_col = QColor(63, 185, 80, 195)
                                badge = "REF"
                            painter.setPen(QPen(ov_col, 1.6))
                            painter.drawPolyline(ov_poly)
                            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                            painter.setPen(ov_col)
                            painter.drawText(QPointF(draw_x + 4, r.bottom() - 18), badge)
                            painter.restore()

            painter.setPen(QPen(QColor("#e6edf3"), 0, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(draw_x, r.top()), QPointF(draw_x, r.bottom()))
            if self.depth_phase_cursors:
                dp_pen = QPen(QColor("#d2a8ff"), 1.2, Qt.PenStyle.DashDotLine)
                painter.setPen(dp_pen)
                for dp_name, dp_off in self.depth_phase_cursors:
                    dx = self.time_to_x(hover_t + dp_off)
                    if r.left() <= dx <= r.right():
                        painter.drawLine(QPointF(dx, r.top()), QPointF(dx, r.bottom()))
                        painter.drawText(QPointF(dx + 3, r.bottom() - 6), dp_name)

        painter.end()

    def resizeEvent(self, event) -> None:
        self._cache_dirty = True
        super().resizeEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover_x = None
        self._hover_pick = None
        self.unsetCursor()
        self.setToolTip("")
        self.update()
        super().leaveEvent(event)

    def pick_at(self, pos: QPointF) -> Optional[PhasePick]:
        """Return PhasePick on the magnified trace within tolerance of pos, if any."""
        if self.trace is None or not self.trace.picks:
            return None
        r = self.plot_rect()
        if not r.contains(pos):
            return None
        best: Optional[Tuple[float, PhasePick]] = None
        for pk in self.trace.picks:
            if not pk.onset:
                continue
            rel_pk = pk.onset_epoch + self.trace.display_offset - self.ref_epoch
            px = self.time_to_x(rel_pk)
            d = abs(px - pos.x())
            if d <= 12.0 and (best is None or d < best[0]):
                best = (d, pk)
        return best[1] if best else None

    def pick_error_bar_at(self, pos: QPointF) -> Optional[Tuple[PhasePick, str]]:
        """Return (PhasePick, 'left'|'right') if pos is near an error-bar ±sigma_t boundary."""
        if not self.show_pick_error_bars or self.trace is None or not self.trace.picks:
            return None
        r = self.plot_rect()
        if not r.contains(pos):
            return None
        best: Optional[Tuple[float, PhasePick, str]] = None
        for pk in self.trace.picks:
            if not pk.onset:
                continue
            rel_pk = pk.onset_epoch + self.trace.display_offset - self.ref_epoch
            px = self.time_to_x(rel_pk)
            sigma = pk.sigma_t(self.trace.delta)
            for side, edge_x in (
                ("left", self.time_to_x(rel_pk - sigma)),
                ("right", self.time_to_x(rel_pk + sigma)),
            ):
                if r.left() <= edge_x <= r.right() and abs(edge_x - px) >= 3.0:
                    d = abs(edge_x - pos.x())
                    if d <= self.ERROR_BAR_HIT_TOLERANCE_PX and d < abs(pos.x() - px):
                        if best is None or d < best[0]:
                            best = (d, pk, side)
        return (best[1], best[2]) if best else None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.trace is None:
            return
        r = self.plot_rect()
        pos = event.position()
        if not r.contains(pos):
            return
        self._hover_x = pos.x()
        rel_t = self.x_to_time(pos.x())
        if event.button() == Qt.MouseButton.LeftButton:
            if self.show_pick_error_bars:
                eb_hit = self.pick_error_bar_at(pos)
                if eb_hit is not None:
                    self._error_bar_drag = eb_hit[0]
                    self.pickEditStarted.emit(self.trace_idx, f"Adjust {eb_hit[0].name} quality")
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    self.update()
                    return
            hit = self.pick_at(pos)
            if hit is not None:
                self._pick_drag = hit
                self._pick_drag_epoch = None
                self._pick_drag_press_x = pos.x()
                self.update()
                return
            ep = self.ref_epoch + rel_t - self.trace.display_offset
            snapped_ep = snap_time_to_feature(self.trace, ep, self.cursor_lock_mode)
            self.pickPlaced.emit(self.trace_idx, snapped_ep)
            self.invalidate_cache()
        elif event.button() == Qt.MouseButton.MiddleButton:
            idx_mode = CURSOR_LOCK_MODES.index(self.cursor_lock_mode) if self.cursor_lock_mode in CURSOR_LOCK_MODES else 0
            nxt = CURSOR_LOCK_MODES[(idx_mode + 1) % len(CURSOR_LOCK_MODES)]
            self.cursor_lock_mode = nxt
            self.cursorModeCycled.emit(nxt)
            self.invalidate_cache()
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start_rel = rel_t
            self._drag_cur_rel = rel_t
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.trace is not None:
            hit = self.pick_at(event.position())
            if hit is not None:
                self._pick_drag = None
                self._pick_drag_epoch = None
                self._hover_pick = None
                self.unsetCursor()
                self.setToolTip("")
                self.pickDeleteRequested.emit(self.trace_idx, hit.name)
                self.invalidate_cache()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and self.trace is not None:
            r = self.plot_rect()
            if self._hover_x is not None and r.left() <= self._hover_x <= r.right():
                rel_t = self.x_to_time(self._hover_x)
            else:
                rel_t = 0.5 * (self.t_lo + self.t_hi)
            ep = self.ref_epoch + rel_t - self.trace.display_offset
            snapped_ep = snap_time_to_feature(self.trace, ep, self.cursor_lock_mode)
            self.pickPlaced.emit(self.trace_idx, snapped_ep)
            self.invalidate_cache()
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and self._hover_pick is not None
            and self.trace is not None
        ):
            pk = self._hover_pick
            self._hover_pick = None
            self._pick_drag = None
            self._pick_drag_epoch = None
            self.unsetCursor()
            self.setToolTip("")
            self.pickDeleteRequested.emit(self.trace_idx, pk.name)
            self.invalidate_cache()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        r = self.plot_rect()
        if not event.buttons():
            eb_hit = self.pick_error_bar_at(pos) if self.show_pick_error_bars else None
            hit = self.pick_at(pos) if eb_hit is None else None
            old_hover = self._hover_pick
            self._hover_pick = hit
            if eb_hit is not None:
                eb_pk, _side = eb_hit
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                dt = self.trace.delta if self.trace else 0.05
                self.setToolTip(
                    f"{eb_pk.name} uncertainty Q={eb_pk.quality} (±{eb_pk.sigma_t(dt):.2g}s)\n"
                    "Drag error-bar edge to adjust pick uncertainty"
                )
            elif hit is not None and self.trace is not None:
                tr = self.trace
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.setToolTip(
                    f"{hit.spec}{hit.sign_char}{hit.name} on {tr.station} {tr.channel3}\n"
                    f"Onset: {hit.onset}   Q={hit.quality}  W={hit.weight}\n"
                    "Drag: move pick   |   Double-click or Del: delete pick"
                )
            elif r.contains(pos):
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.setToolTip("")
            else:
                self.unsetCursor()
                self.setToolTip("")
            if self._hover_x is not None and abs(pos.x() - self._hover_x) < 0.1:
                if old_hover is not hit:
                    self.update()
                return

        self._hover_x = pos.x()

        if self._pick_drag is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            if abs(pos.x() - self._pick_drag_press_x) > 2.0 and self.trace is not None:
                ep = self.ref_epoch + self.x_to_time(pos.x()) - self.trace.display_offset
                self._pick_drag_epoch = snap_time_to_feature(self.trace, ep, self.cursor_lock_mode)
            self.update()
            return

        if self._error_bar_drag is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            if self.trace is not None:
                hover_t = self.x_to_time(pos.x())
                hover_ep = self.ref_epoch + hover_t - self.trace.display_offset
                sigma_sec = abs(hover_ep - self._error_bar_drag.onset_epoch)
                self._error_bar_drag.set_quality(PhasePick.quality_from_sigma_t(sigma_sec))
                self.invalidate_cache()
            return

        if (event.buttons() & Qt.MouseButton.RightButton) and self._drag_start_rel is not None:
            self._drag_cur_rel = self.x_to_time(pos.x())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pick_drag is not None:
            pk = self._pick_drag
            if self._pick_drag_epoch is not None and self.trace is not None:
                self.pickMoved.emit(self.trace_idx, pk.name, self._pick_drag_epoch)
            self._pick_drag = None
            self._pick_drag_epoch = None
            self.update()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._error_bar_drag is not None:
            pk = self._error_bar_drag
            if self.trace is not None:
                hover_t = self.x_to_time(event.position().x())
                hover_ep = self.ref_epoch + hover_t - self.trace.display_offset
                sigma_sec = abs(hover_ep - pk.onset_epoch)
                pk.set_quality(PhasePick.quality_from_sigma_t(sigma_sec))
                self.invalidate_cache()
                self.pickChanged.emit(self.trace_idx, pk)
            self._error_bar_drag = None
            self.unsetCursor()
            self.update()
            return
        if event.button() == Qt.MouseButton.RightButton and self._drag_start_rel is not None and self._drag_cur_rel is not None:
            if self.trace is not None:
                tr_rel_start = self.trace.start_epoch + self.trace.display_offset - self.ref_epoch
                lo = min(self._drag_start_rel, self._drag_cur_rel) - tr_rel_start
                hi = max(self._drag_start_rel, self._drag_cur_rel) - tr_rel_start
                if hi - lo > self.trace.delta:
                    self.amplitudeMeasured.emit(self.trace_idx, lo, hi)
            self._drag_start_rel = None
            self._drag_cur_rel = None
            self.update()
