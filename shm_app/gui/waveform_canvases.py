"""Interactive Main Multi-Trace Canvas and Magnifier Single-Trace Canvas for SHM.

Uses a two-layer offscreen QPixmap backing store + vectorized pixel-column min/max envelope
decimation + single-slot partial redraws for sub-millisecond interactive frame times.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.processing.trace_ops import hilbert_transform

CURSOR_LOCK_MODES: Tuple[str, ...] = ("CROSSHAIR", "PEAK_LOCK", "ZERO_LOCK")
MAGNIFIER_CURSOR_MODES: Tuple[str, ...] = ("STANDARD", "HILBERT", "REFERENCE")

_NAMED_OVERLAY_COLORS = {
    "red": "#f85149",
    "green": "#3fb950",
    "blue": "#58a6ff",
    "yellow": "#d29922",
    "amber": "#d29922",
    "orange": "#ffa657",
    "purple": "#bc8cff",
    "magenta": "#bc8cff",
    "cyan": "#39c5cf",
}
_FALLBACK_OVERLAY_PALETTE = ("#f85149", "#3fb950", "#58a6ff", "#ffa657", "#bc8cff", "#39c5cf")


def _resolve_overlay_filter_color(color_spec: str, filter_name: str = "") -> QColor:
    """Resolve overlay filter color name/hex or deterministic fallback palette color."""
    spec = (color_spec or "").strip()
    if spec:
        low = spec.lower()
        if low in _NAMED_OVERLAY_COLORS:
            return QColor(_NAMED_OVERLAY_COLORS[low])
        qc = QColor(spec)
        if qc.isValid():
            return qc
    h = sum(ord(ch) for ch in (filter_name or "FLT"))
    return QColor(_FALLBACK_OVERLAY_PALETTE[h % len(_FALLBACK_OVERLAY_PALETTE)])


def _draw_pick_uncertainty_interval(
    painter: QPainter,
    x_lo_raw: float,
    x_hi_raw: float,
    y_top: float,
    height: float,
    plot_left: float,
    plot_right: float,
    base_color: QColor,
) -> None:
    """Render shaded [tau - sigma_t, tau + sigma_t] interval and bounded vertical tick marks."""
    if x_hi_raw < plot_left or x_lo_raw > plot_right:
        return
    bx0 = max(plot_left, x_lo_raw)
    bx1 = min(plot_right, x_hi_raw)
    band_h = max(2.0, height - 4.0)
    fill_col = QColor(base_color.red(), base_color.green(), base_color.blue(), 36)
    painter.fillRect(QRectF(bx0, y_top + 2.0, max(1.0, bx1 - bx0), band_h), fill_col)

    y_mid = y_top + 0.5 * height
    tick_half = max(3.0, min(14.0, 0.28 * height))
    tick_top = max(y_top + 2.0, y_mid - tick_half)
    tick_bot = min(y_top + height - 2.0, y_mid + tick_half)

    bar_col = QColor(base_color.red(), base_color.green(), base_color.blue(), 190)
    painter.setPen(QPen(bar_col, 1.1))
    if bx1 > bx0 + 0.5:
        painter.drawLine(QPointF(bx0, y_mid), QPointF(bx1, y_mid))

    tick_pen = QPen(bar_col.lighter(115), 1.3)
    painter.setPen(tick_pen)
    cap = min(3.0, max(1.0, 0.25 * (bx1 - bx0)))
    if plot_left <= x_lo_raw <= plot_right:
        painter.drawLine(QPointF(x_lo_raw, tick_top), QPointF(x_lo_raw, tick_bot))
        if cap >= 1.5:
            painter.drawLine(QPointF(x_lo_raw, tick_top), QPointF(x_lo_raw + cap, tick_top))
            painter.drawLine(QPointF(x_lo_raw, tick_bot), QPointF(x_lo_raw + cap, tick_bot))
    if plot_left <= x_hi_raw <= plot_right:
        painter.drawLine(QPointF(x_hi_raw, tick_top), QPointF(x_hi_raw, tick_bot))
        if cap >= 1.5:
            painter.drawLine(QPointF(x_hi_raw, tick_top), QPointF(x_hi_raw - cap, tick_top))
            painter.drawLine(QPointF(x_hi_raw, tick_bot), QPointF(x_hi_raw - cap, tick_bot))


def _alloc_pixmap_for_widget(widget: QWidget) -> QPixmap:
    dpr = max(1.0, float(widget.devicePixelRatioF()))
    pw = max(1, int(round(widget.width() * dpr)))
    ph = max(1, int(round(widget.height() * dpr)))
    pm = QPixmap(pw, ph)
    pm.setDevicePixelRatio(dpr)
    return pm


def _pixmap_matches_widget(pm: Optional[QPixmap], widget: QWidget) -> bool:
    if pm is None:
        return False
    dpr = max(1.0, float(widget.devicePixelRatioF()))
    return (
        pm.deviceIndependentSize().toSize() == widget.size()
        and abs(pm.devicePixelRatio() - dpr) < 1e-3
    )


def snap_time_to_feature(tr: Trace, epoch_click: float, mode: str) -> float:
    """Apply cursor lock mode ('CROSSHAIR', 'PEAK_LOCK', 'ZERO_LOCK') around epoch_click."""
    if mode == "CROSSHAIR" or tr.delta <= 0 or tr.length < 3:
        return epoch_click
    rel_s = epoch_click - tr.start_epoch
    idx_c = int(round(rel_s / tr.delta))
    half_w = 8
    i0 = max(0, idx_c - half_w)
    i1 = min(tr.length, idx_c + half_w + 1)
    if i1 - i0 < 2:
        return epoch_click
    raw_seg = tr.data[i0:i1]
    seg = raw_seg - np.mean(raw_seg)
    if seg.size < 2:
        return epoch_click
    if mode == "PEAK_LOCK":
        best = i0 + int(np.argmax(np.abs(seg)))
        return tr.start_epoch + best * tr.delta
    if mode == "ZERO_LOCK":
        best = i0 + int(np.argmin(np.abs(seg)))
        return tr.start_epoch + best * tr.delta
    return epoch_click


def _build_trace_polygon(
    data: np.ndarray,
    i0: int,
    i1: int,
    tr_rel_start: float,
    dt: float,
    t_lo: float,
    t_hi: float,
    x_left: float,
    width_px: float,
    y_mid: float,
    scale_y: float,
    y_min: float,
    y_max: float,
    mean_v: float,
) -> QPolygonF:
    """Build a QPolygonF for data[i0:i1] at full sample resolution up to max_direct,
    and true-timestamp sub-pixel min/max extrema bins when sample counts are huge.
    """
    n_pts = i1 - i0
    if n_pts <= 0:
        return QPolygonF()

    inv_span = width_px / max(1e-9, t_hi - t_lo)
    max_direct = max(4000, min(10000, int(width_px * 6)))

    if n_pts <= max_direct:
        idxs = np.arange(i0, i1, dtype=np.float64)
        ts = tr_rel_start + idxs * dt
        xs = x_left + (ts - t_lo) * inv_span
        ys = np.clip(y_mid - (data[i0:i1] - mean_v) * scale_y, y_min, y_max)
        return QPolygonF(map(QPointF, xs.tolist(), ys.tolist()))

    # High-resolution vectorized min/max envelope using exact argmin/argmax timestamps
    n_bins = max(900, min(4000, int(width_px * 1.5)))
    bin_size = max(2, n_pts // n_bins)
    n_used = (n_pts // bin_size) * bin_size
    seg_2d = data[i0 : i0 + n_used].reshape(-1, bin_size)
    nb = seg_2d.shape[0]

    arg_min = np.argmin(seg_2d, axis=1)
    arg_max = np.argmax(seg_2d, axis=1)
    rows = np.arange(nb)
    mins = seg_2d[rows, arg_min]
    maxs = seg_2d[rows, arg_max]

    min_first = arg_min <= arg_max
    idx_first = np.where(min_first, arg_min, arg_max)
    idx_second = np.where(min_first, arg_max, arg_min)
    v_first = np.where(min_first, mins, maxs)
    v_second = np.where(min_first, maxs, mins)

    bin_starts = i0 + np.arange(nb, dtype=np.float64) * bin_size
    sample_idxs = np.empty(2 * nb, dtype=np.float64)
    sample_idxs[0::2] = bin_starts + idx_first
    sample_idxs[1::2] = bin_starts + idx_second

    vals = np.empty(2 * nb, dtype=np.float64)
    vals[0::2] = v_first
    vals[1::2] = v_second

    rem = n_pts - n_used
    if rem > 0:
        tail = data[i0 + n_used : i1]
        t_start = i0 + n_used
        a_min = int(np.argmin(tail))
        a_max = int(np.argmax(tail))
        if a_min == a_max:
            extra_i = [float(t_start + a_min)]
            extra_v = [float(tail[a_min])]
        elif a_min < a_max:
            extra_i = [float(t_start + a_min), float(t_start + a_max)]
            extra_v = [float(tail[a_min]), float(tail[a_max])]
        else:
            extra_i = [float(t_start + a_max), float(t_start + a_min)]
            extra_v = [float(tail[a_max]), float(tail[a_min])]
        sample_idxs = np.concatenate((sample_idxs, np.array(extra_i, dtype=np.float64)))
        vals = np.concatenate((vals, np.array(extra_v, dtype=np.float64)))

    ts = tr_rel_start + sample_idxs * dt
    xs = x_left + (ts - t_lo) * inv_span
    ys = np.clip(y_mid - (vals - mean_v) * scale_y, y_min, y_max)
    return QPolygonF(map(QPointF, xs.tolist(), ys.tolist()))


class MainTraceCanvas(QWidget):
    """Multi-trace seismic waveform display canvas matching SHM's main Motif drawing area.

    Supports:
    - Left margin station/channel/filter labels & click-to-select trace
    - Normalization modes ('C' Constant, 'A' Auto, 'AW' Auto Window, 'SW' Single Window)
    - Left-click phase onset picking / amplitude window selection / magnifier trace focus
    - Middle-click cursor lock mode toggle ('CROSSHAIR', 'PEAK_LOCK', 'ZERO_LOCK')
    - Right-click drag to position/resize the Magnifier time window
    - Mouse wheel zoom
    - Two-layer offscreen QPixmap backing store + single-slot partial updates for zero-lag interaction
    """

    traceClicked = pyqtSignal(int, float, float)  # (trace_idx, epoch_time, rel_time_sec)
    traceSelectionToggled = pyqtSignal(int)  # trace_idx
    traceInvertRequested = pyqtSignal(int)  # trace_idx
    magnifierWindowChanged = pyqtSignal(int, float, float)  # (trace_idx, rel_t_lo, rel_t_hi)
    cursorMoved = pyqtSignal(int, float, float)  # (trace_idx, epoch_time, amplitude)
    cursorModeCycled = pyqtSignal(str)  # new cursor lock mode
    pickDeleteRequested = pyqtSignal(int, str)  # (trace_idx, phase_name)
    pickMoved = pyqtSignal(int, str, float)  # (trace_idx, phase_name, new_epoch)
    pickChanged = pyqtSignal(int, object)  # (trace_idx, PhasePick)
    overlayFilterRequested = pyqtSignal(int)  # trace_idx
    clearOverlayFilterRequested = pyqtSignal(int)  # trace_idx
    contextMenuRequested = pyqtSignal(int, float, int, int)  # (trace_idx, rel_t, global_x, global_y)
    verticalRangeChanged = pyqtSignal(int, int)  # (max_offset_px, page_px)
    verticalOffsetChanged = pyqtSignal(int)  # current vertical offset in px
    horizontalRangeChanged = pyqtSignal(int, int, int)  # (min_tick, max_tick, page_step)
    horizontalValueChanged = pyqtSignal(int)  # current horizontal tick
    horizontalZoomChanged = pyqtSignal(float)  # current full_span / visible_span factor
    amplitudeZoomChanged = pyqtSignal(float)  # global amplitude zoom factor
    viewChanged = pyqtSignal()  # time window changed by pan / history navigation

    PICK_HIT_TOLERANCE_PX = 6.0
    ERROR_BAR_HIT_TOLERANCE_PX = 5.0
    MAG_HANDLE_H = 7.0
    LEFT_DRAG_THRESHOLD_PX = 5.0
    RIGHT_DRAG_THRESHOLD_PX = 5.0

    MARGIN_LEFT = 140
    MARGIN_RIGHT = 24
    MARGIN_TOP = 28
    MARGIN_BOTTOM = 36

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self.traces: List[Trace] = []
        self.norm_mode: str = "A"  # 'C', 'A', 'AW', 'SW'
        self.global_zoom: float = 1.0
        self.view_t_lo: Optional[float] = None  # relative to reference start_epoch
        self.view_t_hi: Optional[float] = None
        self.title: str = "SeismicHandler Motif (SHM)"
        self.show_pick_error_bars: bool = True

        # Magnifier window highlight state
        self.mag_trace_idx: int = 0
        self.mag_t_lo: float = 0.0
        self.mag_t_hi: float = 30.0

        # Cursor state
        self.cursor_lock_mode: str = "CROSSHAIR"  # 'CROSSHAIR', 'PEAK_LOCK', 'ZERO_LOCK'
        self._hover_pos: Optional[QPointF] = None
        self._right_drag_start_t: Optional[float] = None
        self._right_drag_trace_idx: int = 0
        self._left_drag_start_t: Optional[float] = None
        self._left_drag_cur_t: Optional[float] = None
        self._left_press_pos: Optional[QPointF] = None
        self._left_press_info: Optional[Tuple[int, float, float]] = None

        # Pick editing, right-click context menu and pan state
        self._hover_pick: Optional[Tuple[int, object]] = None  # (trace_idx, PhasePick)
        self._pick_drag: Optional[Tuple[int, object]] = None
        self._pick_drag_epoch: Optional[float] = None
        self._pick_drag_press_x: float = 0.0
        self._error_bar_drag: Optional[Tuple[int, object]] = None  # (trace_idx, PhasePick)
        self._right_press_pos: Optional[QPointF] = None
        self._right_moved: bool = False
        self._pan_state: Optional[Tuple[float, float, float]] = None  # (press_x, lo, hi)

        # Time-window history (undo / redo of zoom and pan)
        self._view_history: List[Tuple[float, float]] = []
        self._view_future: List[Tuple[float, float]] = []
        self._last_history_push: float = 0.0

        # Vertical layout: 0 = fit all traces into the window, otherwise minimum row height in px
        self.min_slot_h: float = 0.0
        self.v_offset: float = 0.0
        self._h_scroll_origin: float = 0.0
        self._h_scroll_scale: float = 100.0
        self._h_scroll_win_dur: float = 60.0

        # Optional grouping of the Z/N/E components of one station into adjacent rows
        self.group_by_station: bool = False
        self._group_ends: set = set()

        # Drag state for moving the yellow Magnifier window (length stays constant)
        self._mag_move: Optional[Tuple[float, float, float]] = None  # (press_rel_t, lo, hi)

        # Offscreen backing-store pixmap cache for static waveform layer
        self._static_pixmap: Optional[QPixmap] = None
        self._cache_dirty: bool = True
        self._cache_state_key: Optional[Tuple] = None
        self._cached_global_max: float = 1.0
        self._cached_vis: Optional[List[Tuple[int, Trace]]] = None
        self._cached_ref_ep: Optional[float] = None

        # Color theme (dark high-contrast observatory theme)
        self.bg_color = QColor("#10141d")
        self.margin_bg_color = QColor("#171d29")
        self.grid_color = QColor("#252f42")
        self.trace_color = QColor("#58a6ff")
        self.selected_trace_color = QColor("#ffa657")
        self.beam_trace_color = QColor("#3fb950")
        self.pick_color = QColor("#ff7b72")
        self.theo_color = QColor("#d2a8ff")
        self.mag_box_color = QColor(255, 215, 0, 50)
        self.mag_border_color = QColor("#ffd700")
        self.text_color = QColor("#e6edf3")

        self._font_main = QFont("Monospace", 9)
        self._font_main.setStyleHint(QFont.StyleHint.TypeWriter)

    def invalidate_cache(self, emit_h_range: bool = True, emit_h_zoom: bool = True) -> None:
        """Mark the static waveform backing store dirty so it is rebuilt on the next paint."""
        self._cache_dirty = True
        self._cached_vis = None
        self._cached_ref_ep = None
        self._emit_horizontal_range(emit_range=emit_h_range, emit_zoom=emit_h_zoom)
        self.update()

    # ------------------------------------------------------------------
    # Vertical layout (fit-all or fixed minimum row height with scrolling)
    # ------------------------------------------------------------------

    def slot_height(self, n_vis: Optional[int] = None) -> float:
        n = n_vis if n_vis is not None else len(self.visible_indexed_traces())
        if n <= 0:
            return 1.0
        return max(self.plot_rect().height() / n, float(self.min_slot_h))

    def slot_top(self, slot_i: int, slot_h: float) -> float:
        return self.plot_rect().top() + slot_i * slot_h - self.v_offset

    def max_v_offset(self) -> float:
        n = len(self.visible_indexed_traces())
        if n <= 0:
            return 0.0
        return max(0.0, n * self.slot_height(n) - self.plot_rect().height())

    def _emit_vertical_range(self) -> None:
        mx = self.max_v_offset()
        self.v_offset = max(0.0, min(mx, self.v_offset))
        self.verticalRangeChanged.emit(int(math.ceil(mx)), int(self.plot_rect().height()))
        self.verticalOffsetChanged.emit(int(self.v_offset))

    def set_min_slot_height(self, px: float) -> None:
        """Set minimum row height in px (0 = fit all traces into the window)."""
        old_h = self.slot_height()
        frac = self.v_offset / old_h if old_h > 0 else 0.0
        self.min_slot_h = max(0.0, float(px))
        self.v_offset = frac * self.slot_height()  # keep the same top trace in view
        self._emit_vertical_range()
        self.invalidate_cache()

    def set_group_by_station(self, on: bool) -> None:
        """Group the Z, N, E (and R, T, L, Q) components of each station into adjacent rows."""
        self.group_by_station = bool(on)
        self._cached_vis = None
        self.v_offset = 0.0
        self._emit_vertical_range()
        self.invalidate_cache()

    def set_v_offset(self, px: float) -> None:
        new = max(0.0, min(self.max_v_offset(), float(px)))
        if abs(new - self.v_offset) < 0.5:
            return
        self.v_offset = new
        self.verticalOffsetChanged.emit(int(self.v_offset))
        self.invalidate_cache()

    def _emit_horizontal_range(self, emit_range: bool = True, emit_zoom: bool = True) -> None:
        f0, f1 = self.full_time_span()
        lo, hi = self.active_time_span()
        win_dur = max(0.5, hi - lo)
        full_dur = max(win_dur, f1 - f0)
        if emit_range:
            u0 = min(f0, lo)
            u1 = max(f1, hi)
            total_dur = max(win_dur, u1 - u0)
            scale = 100.0 if total_dur <= 1_000_000.0 else (100_000_000.0 / total_dur)
            self._h_scroll_origin = u0
            self._h_scroll_scale = scale
            self._h_scroll_win_dur = win_dur
            page_step = max(1, int(round(win_dur * scale)))
            max_tick = max(0, int(round((total_dur - win_dur) * scale)))
            cur_tick = max(0, min(max_tick, int(round((lo - u0) * scale))))
            self.horizontalRangeChanged.emit(0, max_tick, page_step)
            self.horizontalValueChanged.emit(cur_tick)
        if emit_zoom:
            self.horizontalZoomChanged.emit(full_dur / win_dur)

    def set_time_offset_from_scroll(self, tick: int) -> None:
        """Scroll the visible horizontal time window using cached scrollbar geometry."""
        scale = max(1e-6, self._h_scroll_scale)
        new_lo = self._h_scroll_origin + float(tick) / scale
        new_hi = new_lo + self._h_scroll_win_dur
        cur_lo, cur_hi = self.active_time_span()
        if abs(new_lo - cur_lo) < (0.5 / scale) and abs(new_hi - cur_hi) < (0.5 / scale):
            return
        self.push_view_history(coalesce_ms=500)
        self.view_t_lo = new_lo
        self.view_t_hi = new_hi
        self.horizontalValueChanged.emit(int(tick))
        self.invalidate_cache(emit_h_range=False, emit_h_zoom=False)
        self.viewChanged.emit()

    def set_horizontal_zoom(self, zoom_factor: float) -> None:
        """Zoom horizontally centered on the current viewport (`1.0` = full trace span)."""
        f0, f1 = self.full_time_span()
        full_dur = max(1.0, f1 - f0)
        z = max(1.0, float(zoom_factor))
        self.push_view_history(coalesce_ms=400)
        if z <= 1.005:
            self.view_t_lo = f0
            self.view_t_hi = f1
        else:
            new_span = max(1.0, full_dur / z)
            lo, hi = self.active_time_span()
            center = 0.5 * (lo + hi)
            new_lo = max(f0, min(f1 - new_span, center - 0.5 * new_span))
            self.view_t_lo = new_lo
            self.view_t_hi = new_lo + new_span
        self.invalidate_cache(emit_h_range=True, emit_h_zoom=False)
        self.viewChanged.emit()

    # ------------------------------------------------------------------
    # Time-window history (back / forward through zoom and pan states)
    # ------------------------------------------------------------------

    def push_view_history(self, coalesce_ms: int = 0) -> None:
        now = time.monotonic()
        recent = (now - self._last_history_push) * 1000.0 < coalesce_ms
        self._last_history_push = now
        if recent:
            return
        state = self.active_time_span()
        if self._view_history and self._view_history[-1] == state:
            return
        self._view_history.append(state)
        del self._view_history[:-50]
        self._view_future.clear()

    def view_back(self) -> bool:
        if not self._view_history:
            return False
        self._view_future.append(self.active_time_span())
        self.view_t_lo, self.view_t_hi = self._view_history.pop()
        self.invalidate_cache()
        self.viewChanged.emit()
        return True

    def view_forward(self) -> bool:
        if not self._view_future:
            return False
        self._view_history.append(self.active_time_span())
        self.view_t_lo, self.view_t_hi = self._view_future.pop()
        self.invalidate_cache()
        self.viewChanged.emit()
        return True

    # ------------------------------------------------------------------
    # Pick hit-testing
    # ------------------------------------------------------------------

    def _mag_handle_rect(self) -> Optional[QRectF]:
        vis = self.visible_indexed_traces()
        slot_i = self._slot_index(self.mag_trace_idx)
        if not vis or slot_i < 0:
            return None
        rect = self.plot_rect()
        x0 = max(rect.left(), min(rect.right(), self.time_to_x(self.mag_t_lo)))
        x1 = max(rect.left(), min(rect.right(), self.time_to_x(self.mag_t_hi)))
        y0 = self.slot_top(slot_i, self.slot_height(len(vis)))
        return QRectF(x0, y0 + 1, max(12.0, x1 - x0), self.MAG_HANDLE_H)

    def _slot_index(self, orig_idx: int) -> int:
        return next((i for i, (o, _) in enumerate(self.visible_indexed_traces()) if o == orig_idx), -1)

    def set_show_pick_error_bars(self, show: bool) -> None:
        val = bool(show)
        if val != self.show_pick_error_bars:
            self.show_pick_error_bars = val
            self.invalidate_cache()

    def pick_at(self, pos: QPointF) -> Optional[Tuple[int, object]]:
        """Return (trace_idx, PhasePick) of the pick marker within tolerance of pos, if any."""
        vis = self.visible_indexed_traces()
        rect = self.plot_rect()
        if not vis or not rect.contains(pos):
            return None
        orig_idx, tr = vis[self.y_to_visible_slot(pos.y())]
        ref_ep = self.reference_epoch()
        best: Optional[Tuple[float, object]] = None
        for pk in tr.picks:
            if not pk.onset:
                continue
            px = self.time_to_x(pk.onset_epoch + tr.display_offset - ref_ep)
            d = abs(px - pos.x())
            if d <= self.PICK_HIT_TOLERANCE_PX and (best is None or d < best[0]):
                best = (d, pk)
        return (orig_idx, best[1]) if best else None

    def pick_error_bar_at(self, pos: QPointF) -> Optional[Tuple[int, object, str]]:
        """Return (trace_idx, PhasePick, 'left'|'right') if pos is near an error-bar edge."""
        if not self.show_pick_error_bars:
            return None
        vis = self.visible_indexed_traces()
        rect = self.plot_rect()
        if not vis or not rect.contains(pos):
            return None
        orig_idx, tr = vis[self.y_to_visible_slot(pos.y())]
        ref_ep = self.reference_epoch()
        best: Optional[Tuple[float, int, object, str]] = None
        for pk in tr.picks:
            if not pk.onset:
                continue
            rel_pk = pk.onset_epoch + tr.display_offset - ref_ep
            px = self.time_to_x(rel_pk)
            if abs(px - pos.x()) <= self.PICK_HIT_TOLERANCE_PX:
                continue
            sigma = pk.sigma_t(tr.delta)
            for side, edge_x in (
                ("left", self.time_to_x(rel_pk - sigma)),
                ("right", self.time_to_x(rel_pk + sigma)),
            ):
                if rect.left() <= edge_x <= rect.right() and abs(edge_x - px) > self.PICK_HIT_TOLERANCE_PX:
                    d = abs(edge_x - pos.x())
                    if d <= self.ERROR_BAR_HIT_TOLERANCE_PX and (best is None or d < best[0]):
                        best = (d, orig_idx, pk, side)
        return (best[1], best[2], best[3]) if best else None

    def invalidate_slot(self, orig_idx: int) -> None:
        """Redraw only a single trace slot in the cached backing store if possible."""
        vis = self.visible_indexed_traces()
        slot_i = next((i for i, (o_idx, _) in enumerate(vis) if o_idx == orig_idx), -1)
        if (
            slot_i < 0
            or self._cache_dirty
            or not _pixmap_matches_widget(self._static_pixmap, self)
            or not vis
        ):
            self.invalidate_cache()
            return

        rect = self.plot_rect()
        n_vis = len(vis)
        slot_h = self.slot_height(n_vis)
        y_top = self.slot_top(slot_i, slot_h)
        if y_top + slot_h < rect.top() or y_top > rect.bottom():
            self.update()
            return
        y_mid = y_top + 0.5 * slot_h
        ref_ep = self.reference_epoch()
        t_lo, t_hi = self.active_time_span()

        painter = QPainter(self._static_pixmap)
        painter.setClipRect(QRectF(0, rect.top(), self.width(), rect.height()))
        painter.setFont(self._font_main)
        # Clear this slot's background in margin and plot area
        painter.fillRect(QRectF(0, y_top, self.MARGIN_LEFT, slot_h), self.margin_bg_color)
        painter.fillRect(QRectF(rect.left(), y_top, rect.width(), slot_h), self.bg_color)

        # Redraw vertical time grid lines clipped to this slot
        painter.save()
        painter.setClipRect(QRectF(rect.left(), y_top, rect.width(), slot_h))
        self._draw_time_grid_lines_only(painter, rect, t_lo, t_hi)
        painter.restore()

        # Redraw the single trace slot
        self._draw_single_trace_slot(
            painter,
            rect,
            slot_i,
            orig_idx,
            vis[slot_i][1],
            y_top,
            y_mid,
            slot_h,
            ref_ep,
            t_lo,
            t_hi,
            self._cached_global_max,
        )
        painter.setPen(QPen(QColor("#30363d"), 1.2))
        painter.drawRect(rect)
        painter.end()
        self._cache_state_key = self._current_state_key()
        self.update()

    def visible_indexed_traces(self) -> List[Tuple[int, Trace]]:
        if self._cached_vis is None:
            vis = [(idx, tr) for idx, tr in enumerate(self.traces) if not tr.hidden and tr.length > 0]
            self._group_ends = set()
            if self.group_by_station:
                first_pos: dict = {}
                for pos, (_i, tr) in enumerate(vis):
                    first_pos.setdefault(tr.station, pos)
                rank = {"Z": 0, "N": 1, "E": 2, "R": 3, "T": 4, "L": 5, "Q": 6}
                vis.sort(key=lambda it: (first_pos[it[1].station], rank.get(it[1].comp, 9), it[0]))
                for k in range(len(vis) - 1):
                    if vis[k][1].station != vis[k + 1][1].station:
                        self._group_ends.add(k)
            self._cached_vis = vis
        return self._cached_vis

    def reference_epoch(self) -> float:
        if self._cached_ref_ep is None:
            vis = self.visible_indexed_traces()
            self._cached_ref_ep = min(tr.start_epoch + tr.display_offset for _, tr in vis) if vis else 0.0
        return self._cached_ref_ep

    def full_time_span(self) -> Tuple[float, float]:
        vis = self.visible_indexed_traces()
        if not vis:
            return 0.0, 60.0
        ref = self.reference_epoch()
        t0 = min((tr.start_epoch + tr.display_offset - ref) for _, tr in vis)
        t1 = max((tr.start_epoch + tr.display_offset - ref + tr.duration) for _, tr in vis)
        return t0, max(t0 + 1.0, t1)

    def active_time_span(self) -> Tuple[float, float]:
        f0, f1 = self.full_time_span()
        lo = self.view_t_lo if self.view_t_lo is not None else f0
        hi = self.view_t_hi if self.view_t_hi is not None else f1
        if hi <= lo:
            return f0, f1
        return lo, hi

    def set_traces(self, traces: List[Trace], reset_view: bool = False) -> None:
        self.traces = traces
        self._cached_vis = None
        self._cached_ref_ep = None
        if reset_view or (self.view_t_lo is None and self.view_t_hi is None):
            f0, f1 = self.full_time_span()
            self.view_t_lo, self.view_t_hi = f0, f1
            dur = f1 - f0
            self.mag_t_lo = f0 + 0.15 * dur
            self.mag_t_hi = f0 + min(dur, max(5.0, 0.45 * dur))
        self._emit_vertical_range()
        self.invalidate_cache()

    def plot_rect(self) -> QRectF:
        w = max(100, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT)
        h = max(60, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        return QRectF(self.MARGIN_LEFT, self.MARGIN_TOP, w, h)

    def time_to_x(self, rel_t: float) -> float:
        rect = self.plot_rect()
        t_lo, t_hi = self.active_time_span()
        return rect.left() + (rel_t - t_lo) / max(1e-9, t_hi - t_lo) * rect.width()

    def x_to_time(self, x: float) -> float:
        rect = self.plot_rect()
        t_lo, t_hi = self.active_time_span()
        frac = (x - rect.left()) / max(1.0, rect.width())
        return t_lo + frac * (t_hi - t_lo)

    def y_to_visible_slot(self, y: float) -> int:
        vis = self.visible_indexed_traces()
        if not vis:
            return -1
        rect = self.plot_rect()
        slot = int((y - rect.top() + self.v_offset) // max(1.0, self.slot_height(len(vis))))
        return max(0, min(len(vis) - 1, slot))

    def _current_state_key(self) -> Tuple:
        vis = self.visible_indexed_traces()
        return (
            self.width(),
            self.height(),
            len(vis),
            self.norm_mode,
            round(self.global_zoom, 5),
            round(self.view_t_lo or 0.0, 4),
            round(self.view_t_hi or 0.0, 4),
            self.title,
            self.cursor_lock_mode,
            self.show_pick_error_bars,
            round(self.min_slot_h, 1),
            round(self.v_offset, 1),
            self.group_by_station,
            tuple(
                (
                    orig_idx,
                    tr.selected,
                    tr.filter_name,
                    getattr(tr, "overlay_filter_name", "NONE"),
                    getattr(tr, "overlay_filter_color", ""),
                    tuple((p.name, p.onset, p.quality, p.sign, p.spec) for p in tr.picks),
                )
                for orig_idx, tr in vis
            ),
            round(float(self.devicePixelRatioF()), 3),
        )

    def _ensure_static_pixmap(self) -> None:
        key = self._current_state_key()
        if (
            not self._cache_dirty
            and _pixmap_matches_widget(self._static_pixmap, self)
            and self._cache_state_key == key
        ):
            return

        self._cached_vis = None
        self._cached_ref_ep = None
        if not _pixmap_matches_widget(self._static_pixmap, self):
            self._static_pixmap = _alloc_pixmap_for_widget(self)

        painter = QPainter(self._static_pixmap)
        painter.fillRect(self.rect(), self.bg_color)

        rect = self.plot_rect()
        painter.fillRect(QRectF(0, 0, self.MARGIN_LEFT, self.height()), self.margin_bg_color)

        painter.setFont(self._font_main)
        painter.setPen(self.text_color)
        ref_ep = self.reference_epoch()
        hdr_text = (
            f"{self.title}   |   Ref: {format_sh_time(ref_ep) if ref_ep > 0 else 'N/A'}   |   "
            f"Norm: {self.norm_mode}   Zoom: {self.global_zoom:.2f}x   Lock: {self.cursor_lock_mode}"
        )
        painter.drawText(
            QRectF(self.MARGIN_LEFT + 6, 4, rect.width(), 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            hdr_text,
        )

        vis = self.visible_indexed_traces()
        if not vis:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                "No traces loaded. Use File -> Read (or click 'Load Example Data' / press Ctrl+R) to load waveforms.",
            )
            painter.end()
            self._cache_dirty = False
            self._cache_state_key = key
            return

        t_lo, t_hi = self.active_time_span()
        self._draw_time_grid(painter, rect, t_lo, t_hi)

        n_vis = len(vis)
        slot_h = self.slot_height(n_vis)

        # Compute global and tiered max amplitudes if norm_mode in ('C', 'AW')
        global_max = 1e-9
        tier_high_max = 1e-9
        tier_low_max = 1e-9
        has_high = False
        has_low = False
        if self.norm_mode in ("C", "AW"):
            for _, tr in vis:
                seg = self._visible_samples(tr, ref_ep, t_lo, t_hi) if self.norm_mode == "AW" else tr.data
                if seg.size > 0:
                    m = float(np.max(np.abs(seg - np.mean(seg))))
                    if m > global_max:
                        global_max = m
                    if m > 1000.0:
                        has_high = True
                        if m > tier_high_max:
                            tier_high_max = m
                    elif m > 1e-6:
                        has_low = True
                        if m > tier_low_max:
                            tier_low_max = m
        self._cached_global_max = global_max
        self._has_mixed_tiers = (has_high and has_low and (tier_high_max / max(1e-9, tier_low_max)) > 50.0)
        self._cached_tier_high_max = tier_high_max if tier_high_max > 1e-9 else global_max
        self._cached_tier_low_max = tier_low_max if tier_low_max > 1e-9 else global_max

        painter.save()
        painter.setClipRect(QRectF(0, rect.top(), self.width(), rect.height()))
        for slot_i, (orig_idx, tr) in enumerate(vis):
            y_top = self.slot_top(slot_i, slot_h)
            if y_top + slot_h < rect.top() or y_top > rect.bottom():
                continue
            y_mid = y_top + 0.5 * slot_h
            self._draw_single_trace_slot(
                painter,
                rect,
                slot_i,
                orig_idx,
                tr,
                y_top,
                y_mid,
                slot_h,
                ref_ep,
                t_lo,
                t_hi,
                global_max,
            )

        painter.restore()
        painter.setPen(QPen(QColor("#30363d"), 1.2))
        painter.drawRect(rect)
        painter.end()
        self._cache_dirty = False
        self._cache_state_key = key

    def paintEvent(self, _event) -> None:
        self._ensure_static_pixmap()

        painter = QPainter(self)
        if self._static_pixmap is not None:
            painter.drawPixmap(0, 0, self._static_pixmap)

        vis = self.visible_indexed_traces()
        if not vis:
            painter.end()
            return

        rect = self.plot_rect()
        slot_h = self.slot_height(len(vis))
        painter.setClipRect(QRectF(0, rect.top(), self.width(), rect.height()))

        # Draw Magnifier highlight box on self.mag_trace_idx (fast overlay)
        self._draw_magnifier_box(painter, rect, vis, slot_h)

        # Highlight hovered pick marker, or live preview of a pick being dragged
        self._draw_pick_overlay(painter, rect, vis, slot_h)

        # Draw active left-drag window if dragging
        if self._left_drag_start_t is not None and self._left_drag_cur_t is not None:
            x0 = self.time_to_x(min(self._left_drag_start_t, self._left_drag_cur_t))
            x1 = self.time_to_x(max(self._left_drag_start_t, self._left_drag_cur_t))
            painter.fillRect(QRectF(x0, rect.top(), max(2.0, x1 - x0), rect.height()), QColor(88, 166, 255, 45))
            painter.setPen(QPen(QColor("#58a6ff"), 1.2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x0, rect.top(), max(2.0, x1 - x0), rect.height()))

        # Draw crosshair cursor (snapped preview in PEAK_LOCK / ZERO_LOCK so pick matches visual cursor)
        if self._hover_pos is not None and rect.contains(self._hover_pos):
            draw_x = self._hover_pos.x()
            if self.cursor_lock_mode != "CROSSHAIR":
                slot_i = self.y_to_visible_slot(self._hover_pos.y())
                if 0 <= slot_i < len(vis):
                    _, hover_tr = vis[slot_i]
                    ref_ep = self.reference_epoch()
                    hover_t = self.x_to_time(draw_x)
                    hover_ep = ref_ep + hover_t - hover_tr.display_offset
                    snapped_ep = self.snap_time_to_feature(hover_tr, hover_ep)
                    draw_x = self.time_to_x(snapped_ep + hover_tr.display_offset - ref_ep)
            pen_c = QPen(QColor(230, 237, 243, 140), 0, Qt.PenStyle.DotLine)
            painter.setPen(pen_c)
            painter.drawLine(QPointF(draw_x, rect.top()), QPointF(draw_x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), self._hover_pos.y()), QPointF(rect.right(), self._hover_pos.y()))

        painter.end()

    def _visible_samples(self, tr: Trace, ref_ep: float, t_lo: float, t_hi: float) -> np.ndarray:
        tr_rel_start = tr.start_epoch + tr.display_offset - ref_ep
        i0 = max(0, int(math.floor((t_lo - tr_rel_start) / max(1e-9, tr.delta))))
        i1 = min(tr.length, int(math.ceil((t_hi - tr_rel_start) / max(1e-9, tr.delta))) + 1)
        return tr.data[i0:i1] if i1 > i0 else tr.data

    def _time_grid_step(self, t_lo: float, t_hi: float) -> Tuple[float, float]:
        span = max(1e-6, t_hi - t_lo)
        raw_step = span / 8.0
        mag = 10.0 ** math.floor(math.log10(raw_step))
        for mult in (1.0, 2.0, 5.0, 10.0):
            if raw_step <= mult * mag:
                step = mult * mag
                break
        else:
            step = 10.0 * mag
        first_tick = math.ceil(t_lo / step) * step
        return first_tick, step

    def _draw_time_grid_lines_only(self, painter: QPainter, rect: QRectF, t_lo: float, t_hi: float) -> None:
        t, step = self._time_grid_step(t_lo, t_hi)
        grid_pen = QPen(self.grid_color, 0, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        while t <= t_hi + 1e-9:
            x = self.time_to_x(t)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            t += step

    def _draw_time_grid(self, painter: QPainter, rect: QRectF, t_lo: float, t_hi: float) -> None:
        t, step = self._time_grid_step(t_lo, t_hi)
        grid_pen = QPen(self.grid_color, 0, Qt.PenStyle.DashLine)
        text_pen = QPen(self.text_color)

        while t <= t_hi + 1e-9:
            x = self.time_to_x(t)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setPen(text_pen)
            lbl = f"{t:.1f}s"
            painter.drawText(QRectF(x - 40, rect.bottom() + 4, 80, 18), Qt.AlignmentFlag.AlignCenter, lbl)
            t += step

    def _draw_single_trace_slot(
        self,
        painter: QPainter,
        rect: QRectF,
        slot_i: int,
        orig_idx: int,
        tr: Trace,
        y_top: float,
        y_mid: float,
        slot_h: float,
        ref_ep: float,
        t_lo: float,
        t_hi: float,
        global_max: float,
    ) -> None:
        # Slot separator in margin and plot
        if self.group_by_station and slot_i in self._group_ends:
            painter.setPen(QPen(QColor("#4c5d7d"), 1.6))
        else:
            painter.setPen(QPen(QColor("#21262d"), 0))
        painter.drawLine(QPointF(0, y_top + slot_h), QPointF(rect.right(), y_top + slot_h))

        # Left margin label
        if tr.selected:
            painter.fillRect(QRectF(2, y_top + 1, self.MARGIN_LEFT - 4, slot_h - 2), QColor(255, 166, 87, 35))
        lbl_color = self.selected_trace_color if tr.selected else (
            self.beam_trace_color if tr.station in ("BEAM", "ALIGN", "SUM", "PWS") else self.text_color
        )
        painter.setPen(lbl_color)
        sel_mark = "*" if tr.selected else " "
        line1 = f"{orig_idx + 1:2d}{sel_mark} {tr.station:<5s} {tr.channel3}"
        painter.drawText(
            QRectF(6, y_top + 2, self.MARGIN_LEFT - 10, slot_h * 0.55),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            line1,
        )
        ov_name = (getattr(tr, "overlay_filter_name", "") or "").strip()
        has_ov = bool(ov_name and ov_name.upper() != "NONE")
        if slot_h >= 26:
            sub_lbl = tr.filter_name if tr.filter_name != "NONE" else f"dt={tr.delta:g}s"
            sub_w = self.MARGIN_LEFT - 14
            if has_ov:
                badge_text = f"[{ov_name[:12]}]"
                fm = painter.fontMetrics()
                badge_w = min(self.MARGIN_LEFT - 24, fm.horizontalAdvance(badge_text) + 10)
                badge_h = min(16.0, max(12.0, slot_h * 0.38))
                badge_y = y_top + slot_h * 0.5 + max(0.0, (slot_h * 0.45 - badge_h) * 0.5)
                badge_x = self.MARGIN_LEFT - 6 - badge_w
                badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)
                badge_col = _resolve_overlay_filter_color(getattr(tr, "overlay_filter_color", ""), ov_name)
                painter.save()
                painter.setPen(QPen(badge_col, 1.0))
                painter.setBrush(QColor(badge_col.red(), badge_col.green(), badge_col.blue(), 45))
                painter.drawRoundedRect(badge_rect, 4.0, 4.0)
                painter.setPen(badge_col.lighter(125))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
                painter.restore()
                sub_w = max(16.0, badge_x - 12.0)
            painter.setPen(QColor("#8b949e"))
            painter.drawText(
                QRectF(10, y_top + slot_h * 0.5, sub_w, slot_h * 0.45),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                sub_lbl[:16],
            )
        elif has_ov:
            badge_text = f"[{ov_name[:10]}]"
            fm = painter.fontMetrics()
            badge_w = min(self.MARGIN_LEFT * 0.48, fm.horizontalAdvance(badge_text) + 8)
            badge_h = max(11.0, slot_h - 4.0)
            badge_rect = QRectF(self.MARGIN_LEFT - 6 - badge_w, y_top + 2.0, badge_w, badge_h)
            badge_col = _resolve_overlay_filter_color(getattr(tr, "overlay_filter_color", ""), ov_name)
            painter.save()
            painter.setPen(QPen(badge_col, 1.0))
            painter.setBrush(QColor(badge_col.red(), badge_col.green(), badge_col.blue(), 45))
            painter.drawRoundedRect(badge_rect, 3.0, 3.0)
            painter.setPen(badge_col.lighter(125))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
            painter.restore()

        # Determine visible sample range
        tr_rel_start = tr.start_epoch + tr.display_offset - ref_ep
        i0 = max(0, int(math.floor((t_lo - tr_rel_start) / max(1e-9, tr.delta))))
        i1 = min(tr.length, int(math.ceil((t_hi - tr_rel_start) / max(1e-9, tr.delta))) + 1)
        if i1 <= i0:
            return

        seg = tr.data[i0:i1]
        mean_v = float(np.mean(seg))
        local_max = float(np.max(np.abs(seg - mean_v))) if seg.size > 0 else 1e-9
        if self.norm_mode in ("C", "AW"):
            if getattr(self, "_has_mixed_tiers", False):
                is_low_tier = local_max <= 1000.0 or float(np.max(np.abs(tr.data))) <= 1000.0
                max_amp = self._cached_tier_low_max if is_low_tier else self._cached_tier_high_max
            else:
                max_amp = global_max
        else:
            max_amp = max(1e-9, local_max)

        scale_y = (0.44 * slot_h * self.global_zoom * tr.norm) / max_amp
        if local_max > 1e-6 and scale_y * local_max < 3.0:
            scale_y = 3.0 / local_max

        poly = _build_trace_polygon(
            tr.data,
            i0,
            i1,
            tr_rel_start,
            tr.delta,
            t_lo,
            t_hi,
            rect.left(),
            rect.width(),
            y_mid,
            scale_y,
            y_top + 1.0,
            y_top + slot_h - 1.0,
            mean_v,
        )

        wf_color = self.selected_trace_color if tr.selected else (
            self.beam_trace_color if tr.station in ("BEAM", "ALIGN", "SUM", "PWS") else self.trace_color
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(wf_color, 1.0))
        painter.drawPolyline(poly)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Draw theoretical arrivals
        if tr.theo_picks:
            theo_pen = QPen(self.theo_color, 0, Qt.PenStyle.DashLine)
            for ph_name, ph_ep in tr.theo_picks.items():
                rel_ph = ph_ep + tr.display_offset - ref_ep
                if t_lo <= rel_ph <= t_hi:
                    px = self.time_to_x(rel_ph)
                    painter.setPen(theo_pen)
                    painter.drawLine(QPointF(px, y_top + 2), QPointF(px, y_top + slot_h - 2))
                    painter.drawText(QPointF(px + 2, y_top + slot_h - 4), f"({ph_name})")

        # Draw manual/auto phase picks and [±sigma_t] uncertainty error bars
        if tr.picks:
            pick_pen = QPen(self.pick_color, 1.5)
            for pk in tr.picks:
                if not pk.onset:
                    continue
                rel_pk = pk.onset_epoch + tr.display_offset - ref_ep
                if self.show_pick_error_bars:
                    sigma = pk.sigma_t(tr.delta)
                    if rel_pk + sigma >= t_lo and rel_pk - sigma <= t_hi:
                        _draw_pick_uncertainty_interval(
                            painter,
                            self.time_to_x(rel_pk - sigma),
                            self.time_to_x(rel_pk + sigma),
                            y_top,
                            slot_h,
                            rect.left(),
                            rect.right(),
                            self.pick_color,
                        )
                if t_lo <= rel_pk <= t_hi:
                    px = self.time_to_x(rel_pk)
                    painter.setPen(pick_pen)
                    painter.drawLine(QPointF(px, y_top + 1), QPointF(px, y_top + slot_h - 1))
                    lbl = f"{pk.spec}{pk.sign_char}{pk.name}"
                    painter.drawText(QPointF(px + 3, y_top + 12), lbl)

    def _draw_magnifier_box(
        self,
        painter: QPainter,
        rect: QRectF,
        vis: List[Tuple[int, Trace]],
        slot_h: float,
    ) -> None:
        slot_idx = next((i for i, (orig_i, _) in enumerate(vis) if orig_i == self.mag_trace_idx), 0)
        if not (0 <= slot_idx < len(vis)):
            return
        x0 = max(rect.left(), min(rect.right(), self.time_to_x(self.mag_t_lo)))
        x1 = max(rect.left(), min(rect.right(), self.time_to_x(self.mag_t_hi)))
        if x1 <= x0:
            x1 = min(rect.right(), x0 + 12.0)
        y0 = self.slot_top(slot_idx, slot_h)
        box = QRectF(x0, y0 + 1, x1 - x0, slot_h - 2)
        painter.fillRect(box, self.mag_box_color)
        painter.setPen(QPen(self.mag_border_color, 1.4))
        painter.drawRect(box)
        # Grip bar: drag it (or Ctrl+drag anywhere) to move the window without changing its length
        painter.fillRect(QRectF(x0, y0 + 1, max(12.0, x1 - x0), self.MAG_HANDLE_H), QColor(255, 215, 0, 170))

    def _draw_pick_overlay(
        self,
        painter: QPainter,
        rect: QRectF,
        vis: List[Tuple[int, Trace]],
        slot_h: float,
    ) -> None:
        """Highlight the hovered pick marker and show a live preview while dragging a pick."""
        ref_ep = self.reference_epoch()
        if self._pick_drag is not None and self._pick_drag_epoch is not None:
            orig_idx, pk = self._pick_drag
            slot_i = self._slot_index(orig_idx)
            if slot_i >= 0:
                tr = vis[slot_i][1]
                px = self.time_to_x(self._pick_drag_epoch + tr.display_offset - ref_ep)
                y0 = self.slot_top(slot_i, slot_h)
                painter.setPen(QPen(QColor("#ffd700"), 2.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(px, y0 + 1), QPointF(px, y0 + slot_h - 1))
                painter.setPen(QColor("#ffd700"))
                painter.drawText(QPointF(px + 4, y0 + 24), format_sh_time(self._pick_drag_epoch)[-12:])
            return
        if self._hover_pick is not None:
            orig_idx, pk = self._hover_pick
            slot_i = self._slot_index(orig_idx)
            if slot_i >= 0:
                tr = vis[slot_i][1]
                px = self.time_to_x(pk.onset_epoch + tr.display_offset - ref_ep)
                y0 = self.slot_top(slot_i, slot_h)
                painter.setPen(QPen(QColor("#ffd700"), 2.5))
                painter.drawLine(QPointF(px, y0 + 1), QPointF(px, y0 + slot_h - 1))

    def snap_time_to_feature(self, tr: Trace, epoch_click: float) -> float:
        """Apply cursor_lock_mode ('CROSSHAIR', 'PEAK_LOCK', 'ZERO_LOCK') around epoch_click."""
        return snap_time_to_feature(tr, epoch_click, self.cursor_lock_mode)

    # ------------------------------------------------------------------
    # Mouse, Resize & Wheel Events
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        self._cache_dirty = True
        super().resizeEvent(event)
        self._emit_vertical_range()

    def leaveEvent(self, event) -> None:
        self._hover_pos = None
        self._hover_pick = None
        self.unsetCursor()
        self.setToolTip("")
        self.update()
        super().leaveEvent(event)

    def _pick_tooltip_cleared(self) -> bool:
        """Remove a stale grip-bar tooltip when the mouse is no longer over it."""
        if self.toolTip().startswith("Drag to move the Magnifier"):
            self.setToolTip("")
            return True
        return False

    def _update_hover_pick(self, pos: QPointF) -> None:
        hit = self.pick_at(pos)
        old_key = (self._hover_pick[0], id(self._hover_pick[1])) if self._hover_pick else None
        new_key = (hit[0], id(hit[1])) if hit else None
        if old_key == new_key:
            return
        self._hover_pick = hit
        if hit is None:
            self.unsetCursor()
            self.setToolTip("")
            return
        oi, pk = hit
        tr = self.traces[oi]
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip(
            f"{pk.spec}{pk.sign_char}{pk.name} on {tr.station} {tr.channel3}\n"
            f"Onset: {pk.onset}   Q={pk.quality}  W={pk.weight}\n"
            "Drag: move pick   |   Double-click or Del: delete pick"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        vis = self.visible_indexed_traces()
        if not vis:
            return

        rect = self.plot_rect()
        slot_i = self.y_to_visible_slot(pos.y())
        orig_idx, tr = vis[slot_i]

        # Click inside left margin -> toggle trace selection (single-slot redraw)
        if pos.x() < self.MARGIN_LEFT:
            if event.button() == Qt.MouseButton.LeftButton:
                tr.selected = not tr.selected
                self.invalidate_slot(orig_idx)
                self.traceSelectionToggled.emit(orig_idx)
            elif event.button() == Qt.MouseButton.MiddleButton:
                self.traceInvertRequested.emit(orig_idx)
            return

        if not rect.contains(pos):
            return

        self._hover_pos = pos
        rel_t = self.x_to_time(pos.x())
        ref_ep = self.reference_epoch()
        click_ep = ref_ep + rel_t - tr.display_offset
        snapped_ep = self.snap_time_to_feature(tr, click_ep)

        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._mag_handle_rect()
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if ctrl or (handle is not None and handle.contains(pos)):
                self._mag_move = (rel_t, self.mag_t_lo, self.mag_t_hi)
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return

            # Shift + left drag pans the time window
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.push_view_history()
                lo, hi = self.active_time_span()
                self._pan_state = (pos.x(), lo, hi)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

            prev_mag_idx = self.mag_trace_idx
            self.mag_trace_idx = orig_idx
            hit = self.pick_at(pos)
            if hit is not None:
                # Grab an existing pick instead of creating a new one
                self._pick_drag = hit
                self._pick_drag_epoch = None
                self._pick_drag_press_x = pos.x()
                self._left_drag_start_t = None
                self._left_drag_cur_t = None
                self._left_press_pos = None
                self._left_press_info = None
            else:
                eb_hit = self.pick_error_bar_at(pos) if self.show_pick_error_bars else None
                if eb_hit is not None:
                    self._error_bar_drag = (eb_hit[0], eb_hit[1])
                    self._left_drag_start_t = None
                    self._left_drag_cur_t = None
                    self._left_press_pos = None
                    self._left_press_info = None
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self._left_drag_start_t = rel_t
                    self._left_drag_cur_t = None
                    self._left_press_pos = pos
                    self._left_press_info = (orig_idx, snapped_ep, rel_t)
            if prev_mag_idx != orig_idx:
                self.magnifierWindowChanged.emit(self.mag_trace_idx, self.mag_t_lo, self.mag_t_hi)
            self.update()

        elif event.button() == Qt.MouseButton.MiddleButton:
            idx_mode = CURSOR_LOCK_MODES.index(self.cursor_lock_mode) if self.cursor_lock_mode in CURSOR_LOCK_MODES else 0
            nxt = CURSOR_LOCK_MODES[(idx_mode + 1) % len(CURSOR_LOCK_MODES)]
            self.cursor_lock_mode = nxt
            self._cache_dirty = True
            self.cursorModeCycled.emit(nxt)
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            # Quick click -> context menu (on release); drag -> position/resize Magnifier window
            self.mag_trace_idx = orig_idx
            self._right_drag_start_t = rel_t
            self._right_drag_trace_idx = orig_idx
            self._right_press_pos = pos
            self._right_moved = False
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().x() >= self.MARGIN_LEFT:
            hit = self.pick_at(event.position())
            if hit is not None:
                self._pick_drag = None
                self._pick_drag_epoch = None
                self._hover_pick = None
                self.unsetCursor()
                self.setToolTip("")
                self.pickDeleteRequested.emit(hit[0], hit[1].name)
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._hover_pick is not None:
            oi, pk = self._hover_pick
            self._hover_pick = None
            self.unsetCursor()
            self.setToolTip("")
            self.pickDeleteRequested.emit(oi, pk.name)
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if (
            self._hover_pos is not None
            and abs(pos.x() - self._hover_pos.x()) < 0.1
            and abs(pos.y() - self._hover_pos.y()) < 0.1
            and not event.buttons()
        ):
            return
        self._hover_pos = pos

        # Moving the yellow Magnifier window: constant length, clamped to the data span
        if self._mag_move is not None:
            press_t, lo0, hi0 = self._mag_move
            length = hi0 - lo0
            f0, f1 = self.full_time_span()
            new_lo = lo0 + (self.x_to_time(pos.x()) - press_t)
            new_lo = max(f0, min(max(f0, f1 - length), new_lo))
            self.mag_t_lo = new_lo
            self.mag_t_hi = new_lo + length
            self.magnifierWindowChanged.emit(self.mag_trace_idx, self.mag_t_lo, self.mag_t_hi)
            self.update()
            return

        # Shift + left drag: pan time window
        if self._pan_state is not None:
            x0, lo, hi = self._pan_state
            span = hi - lo
            dt = -(pos.x() - x0) / max(1.0, self.plot_rect().width()) * span
            self.view_t_lo = lo + dt
            self.view_t_hi = hi + dt
            self.invalidate_cache()
            return

        # Dragging an existing pick: live preview only, commit on release
        if self._pick_drag is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            if abs(pos.x() - self._pick_drag_press_x) > 2.0:
                oi, _pk = self._pick_drag
                tr = self.traces[oi]
                ep = self.reference_epoch() + self.x_to_time(pos.x()) - tr.display_offset
                self._pick_drag_epoch = self.snap_time_to_feature(tr, ep)
            self.update()
            return

        # Dragging pick uncertainty error-bar edge
        if self._error_bar_drag is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            oi, pk = self._error_bar_drag
            if 0 <= oi < len(self.traces):
                tr = self.traces[oi]
                hover_ep = self.reference_epoch() + self.x_to_time(pos.x()) - tr.display_offset
                sigma_sec = abs(hover_ep - pk.onset_epoch)
                pk.set_quality(PhasePick.quality_from_sigma_t(sigma_sec))
                self.invalidate_slot(oi)
            return

        if (
            (event.buttons() & Qt.MouseButton.RightButton)
            and self._right_press_pos is not None
            and not self._right_moved
        ):
            dx = pos.x() - self._right_press_pos.x()
            dy = pos.y() - self._right_press_pos.y()
            if math.hypot(dx, dy) > self.RIGHT_DRAG_THRESHOLD_PX:
                self._right_moved = True

        vis = self.visible_indexed_traces()
        rect = self.plot_rect()
        if vis and rect.contains(pos):
            slot_i = self.y_to_visible_slot(pos.y())
            orig_idx, tr = vis[slot_i]
            rel_t = self.x_to_time(pos.x())
            ref_ep = self.reference_epoch()
            ep = ref_ep + rel_t - tr.display_offset
            s_idx = max(0, min(tr.length - 1, int(round((ep - tr.start_epoch) / max(1e-9, tr.delta)))))
            amp = float(tr.data[s_idx]) if tr.length > 0 else 0.0
            self.cursorMoved.emit(orig_idx, ep, amp)

            if (event.buttons() & Qt.MouseButton.RightButton) and self._right_drag_start_t is not None:
                if self._right_moved and abs(rel_t - self._right_drag_start_t) > 0.2:
                    self.mag_t_lo = min(self._right_drag_start_t, rel_t)
                    self.mag_t_hi = max(self._right_drag_start_t, rel_t)
                    self.magnifierWindowChanged.emit(self.mag_trace_idx, self.mag_t_lo, self.mag_t_hi)
            elif (event.buttons() & Qt.MouseButton.LeftButton) and self._left_drag_start_t is not None:
                if self._left_drag_cur_t is not None or (
                    self._left_press_pos is not None
                    and abs(pos.x() - self._left_press_pos.x()) > self.LEFT_DRAG_THRESHOLD_PX
                ):
                    self._left_drag_cur_t = rel_t

        if not event.buttons():
            self._update_hover_pick(pos)
            if self._hover_pick is None:
                eb_hit = self.pick_error_bar_at(pos)
                handle = self._mag_handle_rect()
                if eb_hit is not None:
                    _oi, eb_pk, _side = eb_hit
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    self.setToolTip(
                        f"{eb_pk.name} uncertainty Q={eb_pk.quality} (±{eb_pk.sigma_t(self.traces[_oi].delta):.2g}s)\n"
                        "Drag error-bar edge to adjust pick uncertainty"
                    )
                elif handle is not None and handle.contains(pos):
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    self.setToolTip("Drag to move the Magnifier window (length stays constant)")
                elif rect.contains(pos):
                    self.setCursor(Qt.CursorShape.CrossCursor)
                    self._pick_tooltip_cleared()
                else:
                    self.unsetCursor()
                    self._pick_tooltip_cleared()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._mag_move is not None:
                self._mag_move = None
                self.unsetCursor()
                self.update()
                return
            if self._pan_state is not None:
                self._pan_state = None
                self.unsetCursor()
                self.viewChanged.emit()
                self.update()
                return
            if self._pick_drag is not None:
                oi, pk = self._pick_drag
                if self._pick_drag_epoch is not None:
                    self.pickMoved.emit(oi, pk.name, self._pick_drag_epoch)
                self._pick_drag = None
                self._pick_drag_epoch = None
                self.update()
                return
            if self._error_bar_drag is not None:
                oi, pk = self._error_bar_drag
                if 0 <= oi < len(self.traces):
                    tr = self.traces[oi]
                    hover_ep = self.reference_epoch() + self.x_to_time(event.position().x()) - tr.display_offset
                    sigma_sec = abs(hover_ep - pk.onset_epoch)
                    pk.set_quality(PhasePick.quality_from_sigma_t(sigma_sec))
                    self.invalidate_slot(oi)
                    self.pickChanged.emit(oi, pk)
                self._error_bar_drag = None
                self.unsetCursor()
                self.update()
                return
            if self._left_drag_start_t is not None and self._left_drag_cur_t is not None:
                lo = min(self._left_drag_start_t, self._left_drag_cur_t)
                hi = max(self._left_drag_start_t, self._left_drag_cur_t)
                if hi - lo > 0.3:
                    self.mag_t_lo = lo
                    self.mag_t_hi = hi
                    self.magnifierWindowChanged.emit(self.mag_trace_idx, self.mag_t_lo, self.mag_t_hi)
            elif self._left_press_info is not None:
                orig_idx, snapped_ep, rel_t = self._left_press_info
                rel_pos = event.position()
                rect = self.plot_rect()
                if 0 <= orig_idx < len(self.traces):
                    pick_x = rel_pos.x() if rect.contains(rel_pos) else (
                        self._left_press_pos.x() if self._left_press_pos is not None else rel_pos.x()
                    )
                    pick_x = max(rect.left(), min(rect.right(), pick_x))
                    tr = self.traces[orig_idx]
                    rel_t = self.x_to_time(pick_x)
                    click_ep = self.reference_epoch() + rel_t - tr.display_offset
                    snapped_ep = self.snap_time_to_feature(tr, click_ep)
                    self._hover_pos = QPointF(
                        pick_x,
                        rel_pos.y() if rect.contains(rel_pos) else (self._hover_pos.y() if self._hover_pos is not None else rect.center().y()),
                    )
                self.traceClicked.emit(orig_idx, snapped_ep, rel_t)
            self._left_drag_start_t = None
            self._left_drag_cur_t = None
            self._left_press_pos = None
            self._left_press_info = None
        elif event.button() == Qt.MouseButton.RightButton:
            if self._right_press_pos is not None and not self._right_moved:
                gp = event.globalPosition().toPoint()
                self.contextMenuRequested.emit(
                    self._right_drag_trace_idx, float(self._right_drag_start_t or 0.0), gp.x(), gp.y()
                )
            self._right_drag_start_t = None
            self._right_press_pos = None
            self._right_moved = False
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            fac = 1.25 if delta > 0 else 0.8
            self.global_zoom = max(0.05, min(200.0, self.global_zoom * fac))
            self.amplitudeZoomChanged.emit(self.global_zoom)
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            self.set_v_offset(self.v_offset - delta * 0.5)
            return
        else:
            self.push_view_history(coalesce_ms=500)
            t_lo, t_hi = self.active_time_span()
            center_t = self.x_to_time(event.position().x())
            scale = 0.8 if delta > 0 else 1.25
            new_span = max(1.0, (t_hi - t_lo) * scale)
            ratio = (center_t - t_lo) / max(1e-9, t_hi - t_lo)
            self.view_t_lo = center_t - ratio * new_span
            self.view_t_hi = self.view_t_lo + new_span
        self.invalidate_cache()


class MagnifierCanvas(QWidget):
    """Single-trace high-resolution zoomed Magnifier panel matching SHM's Magnifier Window.

    Uses an offscreen QPixmap backing store for the zoomed waveform + picks so cursor hover
    and amplitude measurement drag run with sub-millisecond latency.
    """

    pickPlaced = pyqtSignal(int, float)  # (trace_idx, epoch_time)
    pickChanged = pyqtSignal(int, object)  # (trace_idx, PhasePick)
    amplitudeMeasured = pyqtSignal(int, float, float)  # (trace_idx, rel_t_lo, rel_t_hi)
    cursorModeCycled = pyqtSignal(str)  # new cursor lock mode
    cursorOverlayModeChanged = pyqtSignal(str)  # new overlay mode ('STANDARD', 'HILBERT', 'REFERENCE')

    ERROR_BAR_HIT_TOLERANCE_PX = 7.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 145)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
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
            f"(L-click: Pick | M-click: Lock | R-drag: Amp)"
        )
        painter.drawText(QRectF(72, 2, r.width(), 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hdr)

        if i1 <= i0:
            painter.end()
            self._cache_dirty = False
            self._cache_state_key = key
            return

        seg = tr.data[i0:i1]
        mean_v = float(np.mean(seg))
        max_amp = max(1e-9, float(np.max(np.abs(seg - mean_v))))
        y_mid = r.center().y()
        scale_y = 0.44 * r.height() / max_amp

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
                    ov_dm = ov_arr - float(np.mean(ov_arr))
                    ov_max = float(np.max(np.abs(ov_dm)))
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
        self.unsetCursor()
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
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        r = self.plot_rect()
        if not event.buttons():
            eb_hit = self.pick_error_bar_at(pos) if self.show_pick_error_bars else None
            if eb_hit is not None:
                eb_pk, _side = eb_hit
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                dt = self.trace.delta if self.trace else 0.05
                self.setToolTip(
                    f"{eb_pk.name} uncertainty Q={eb_pk.quality} (±{eb_pk.sigma_t(dt):.2g}s)\n"
                    "Drag error-bar edge to adjust pick uncertainty"
                )
            elif r.contains(pos):
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.setToolTip("")
            else:
                self.unsetCursor()
                self.setToolTip("")
            if self._hover_x is not None and abs(pos.x() - self._hover_x) < 0.1:
                return

        self._hover_x = pos.x()

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
