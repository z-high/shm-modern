"""Shared rendering, geometry, snapping, and offscreen pixmap helpers for SHM waveform canvases."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QWidget

from shm_app.core.models import Trace

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
    finite_mask = np.isfinite(raw_seg)
    if not np.any(finite_mask):
        return epoch_click
    mean_raw = float(np.mean(raw_seg[finite_mask]))
    seg = np.where(finite_mask, raw_seg - mean_raw, 0.0)
    if seg.size < 2:
        return epoch_click
    if mode == "PEAK_LOCK":
        best = i0 + int(np.argmax(np.abs(seg)))
        return tr.start_epoch + best * tr.delta
    if mode == "ZERO_LOCK":
        abs_seg = np.where(finite_mask, np.abs(seg), np.inf)
        best = i0 + int(np.argmin(abs_seg))
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

    safe_mean = mean_v if math.isfinite(mean_v) else 0.0
    safe_scale = scale_y if math.isfinite(scale_y) else 1.0
    sub = np.asarray(data[i0:i1], dtype=np.float64)
    if not np.all(np.isfinite(sub)):
        sub = np.where(np.isfinite(sub), sub, safe_mean)

    inv_span = width_px / max(1e-9, t_hi - t_lo)
    max_direct = max(4000, min(10000, int(width_px * 6)))

    if n_pts <= max_direct:
        idxs = np.arange(i0, i1, dtype=np.float64)
        ts = tr_rel_start + idxs * dt
        xs = x_left + (ts - t_lo) * inv_span
        ys = np.clip(y_mid - (sub - safe_mean) * safe_scale, y_min, y_max)
        return QPolygonF(map(QPointF, xs.tolist(), ys.tolist()))

    # High-resolution vectorized min/max envelope using exact argmin/argmax timestamps
    n_bins = max(900, min(4000, int(width_px * 1.5)))
    bin_size = max(2, n_pts // n_bins)
    n_used = (n_pts // bin_size) * bin_size
    seg_2d = sub[:n_used].reshape(-1, bin_size)
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
        tail = sub[n_used:]
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
    ys = np.clip(y_mid - (vals - safe_mean) * safe_scale, y_min, y_max)
    return QPolygonF(map(QPointF, xs.tolist(), ys.tolist()))

