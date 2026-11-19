"""Headless UI tests for Phase 3 canvas enhancements (M-13, M-14, M-24)."""

from __future__ import annotations

import os
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QMouseEvent
from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(["shm_p3_test"])

from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.waveform_canvases import (
    CURSOR_LOCK_MODES,
    MAGNIFIER_CURSOR_MODES,
    MagnifierCanvas,
    MainTraceCanvas,
    _resolve_overlay_filter_color,
)
from shm_app.processing.trace_ops import hilbert_transform

L = Qt.MouseButton.LeftButton
NO = Qt.MouseButton.NoButton
NOMOD = Qt.KeyboardModifier.NoModifier


def _send_mouse(widget, kind: str, x: float, y: float, button=NO, buttons=NO, mods=NOMOD) -> None:
    ev_type = {
        "press": QEvent.Type.MouseButtonPress,
        "release": QEvent.Type.MouseButtonRelease,
        "move": QEvent.Type.MouseMove,
    }[kind]
    ev = QMouseEvent(ev_type, QPointF(x, y), widget.mapToGlobal(QPointF(x, y)), button, buttons, mods)
    QApplication.sendEvent(widget, ev)


def _pixmap_image(widget) -> QImage:
    widget.repaint()
    app.processEvents()
    return widget.grab().toImage()


def _images_differ(img_a: QImage, img_b: QImage) -> bool:
    if img_a.size() != img_b.size():
        return True
    return img_a.constBits().asstring(img_a.sizeInBytes()) != img_b.constBits().asstring(img_b.sizeInBytes())


class TestPickUncertaintyErrorBarsM13(unittest.TestCase):
    """M-13: Visual pick uncertainty error bars [±sigma_t] and interactive drag."""

    def _make_trace_with_pick(self, quality: int = 4, onset_rel: float = 15.0) -> Trace:
        dt = 0.05
        t = np.arange(800, dtype=np.float64) * dt
        data = np.sin(2.0 * np.pi * 1.0 * t) * 100.0
        t0 = 1000000.0
        tr = Trace(data=data, delta=dt, station="GRA1", comp="Z", start_epoch=t0)
        pk = PhasePick(name="P", onset=format_sh_time(t0 + onset_rel), station="GRA1", quality=quality)
        pk.set_quality(quality)
        tr.add_or_replace_pick(pk)
        return tr

    def test_main_and_magnifier_render_error_bars_across_qualities(self) -> None:
        tr_q0 = self._make_trace_with_pick(quality=0, onset_rel=15.0)
        tr_q9 = self._make_trace_with_pick(quality=8, onset_rel=15.0)

        main_cv = MainTraceCanvas()
        main_cv.resize(800, 260)
        main_cv.set_traces([tr_q9], reset_view=True)
        self.assertTrue(main_cv.show_pick_error_bars)

        key_on = main_cv._current_state_key()
        img_on = _pixmap_image(main_cv)

        main_cv.set_show_pick_error_bars(False)
        key_off = main_cv._current_state_key()
        self.assertNotEqual(key_on, key_off)
        img_off = _pixmap_image(main_cv)
        self.assertTrue(_images_differ(img_on, img_off), "Toggling show_pick_error_bars must alter MainTraceCanvas pixels")

        # Quality change must invalidate _current_state_key
        main_cv.set_show_pick_error_bars(True)
        key_q8 = main_cv._current_state_key()
        tr_q9.picks[0].set_quality(1)
        self.assertNotEqual(key_q8, main_cv._current_state_key())

        # MagnifierCanvas rendering for Q=0..9
        mag_cv = MagnifierCanvas()
        mag_cv.resize(700, 220)
        mag_cv.set_magnifier_view(tr_q0, 0, tr_q0.start_epoch, 5.0, 25.0)
        for q in range(10):
            tr_q0.picks[0].set_quality(q)
            mag_cv.invalidate_cache()
            _ = _pixmap_image(mag_cv)

        img_mag_on = _pixmap_image(mag_cv)
        key_mag_on = mag_cv._current_state_key()
        mag_cv.set_show_pick_error_bars(False)
        self.assertNotEqual(key_mag_on, mag_cv._current_state_key())
        img_mag_off = _pixmap_image(mag_cv)
        self.assertTrue(_images_differ(img_mag_on, img_mag_off), "Toggling show_pick_error_bars must alter MagnifierCanvas pixels")

    def test_magnifier_error_bar_interactive_drag_updates_quality_and_emits_signal(self) -> None:
        tr = self._make_trace_with_pick(quality=4, onset_rel=10.0)  # sigma_t = 0.50s
        mag = MagnifierCanvas()
        mag.resize(800, 220)
        mag.set_magnifier_view(tr, 0, tr.start_epoch, 5.0, 15.0)
        mag.show()
        app.processEvents()

        changed_events = []
        placed_events = []
        mag.pickChanged.connect(lambda idx, pk: changed_events.append((idx, pk.quality)))
        mag.pickPlaced.connect(lambda idx, ep: placed_events.append((idx, ep)))

        y_mid = mag.plot_rect().center().y()
        edge_x = mag.time_to_x(10.0 + tr.picks[0].sigma_t(tr.delta))  # 10.5 s
        hit = mag.pick_error_bar_at(QPointF(edge_x, y_mid))
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], "right")

        target_x = mag.time_to_x(12.0)  # sigma_sec = 2.0 s -> Q = 6
        _send_mouse(mag, "press", edge_x, y_mid, L, L)
        _send_mouse(mag, "move", target_x, y_mid, NO, L)
        _send_mouse(mag, "release", target_x, y_mid, L, NO)

        self.assertEqual(placed_events, [], "Dragging error bar must not place a new pick")
        self.assertEqual(len(changed_events), 1)
        self.assertEqual(changed_events[0][0], 0)
        self.assertEqual(tr.picks[0].quality, PhasePick.quality_from_sigma_t(2.0))
        self.assertEqual(tr.picks[0].quality, 6)
        mag.close()

    def test_main_canvas_error_bar_drag_preserves_pick_move(self) -> None:
        tr = self._make_trace_with_pick(quality=6, onset_rel=20.0)  # sigma_t = 2.0 s
        cv = MainTraceCanvas()
        cv.resize(900, 240)
        cv.set_traces([tr], reset_view=True)
        cv.show()
        app.processEvents()

        changed = []
        moved = []
        cv.pickChanged.connect(lambda idx, pk: changed.append((idx, pk.quality)))
        cv.pickMoved.connect(lambda idx, name, ep: moved.append((idx, name, ep)))

        y_mid = cv.plot_rect().center().y()
        edge_x = cv.time_to_x(20.0 + tr.picks[0].sigma_t(tr.delta))  # 22.0 s
        target_x = cv.time_to_x(24.0)  # sigma_sec = 4.0 s -> Q = 7
        _send_mouse(cv, "press", edge_x, y_mid, L, L)
        _send_mouse(cv, "move", target_x, y_mid, NO, L)
        _send_mouse(cv, "release", target_x, y_mid, L, NO)

        self.assertEqual(moved, [])
        self.assertEqual(len(changed), 1)
        self.assertEqual(tr.picks[0].quality, 7)

        # Dragging directly on pick center still emits pickMoved
        pick_x = cv.time_to_x(20.0)
        _send_mouse(cv, "press", pick_x, y_mid, L, L)
        _send_mouse(cv, "move", pick_x + 40.0, y_mid, NO, L)
        _send_mouse(cv, "release", pick_x + 40.0, y_mid, L, NO)
        self.assertEqual(len(moved), 1)
        cv.close()


class TestMagnifierCursorModesM14(unittest.TestCase):
    """M-14: Hilbert and Reference waveform cursor modes in MagnifierCanvas."""

    def _make_wave_trace(self) -> Trace:
        dt = 0.05
        t = np.arange(600, dtype=np.float64) * dt
        # Localized Ricker-like wavelet around t=10s and phase-shifted packet around t=20s
        sig = np.exp(-((t - 10.0) ** 2) / 0.5) * np.sin(2.0 * np.pi * 1.5 * (t - 10.0))
        sig += np.exp(-((t - 20.0) ** 2) / 0.5) * np.cos(2.0 * np.pi * 1.5 * (t - 20.0))
        t0 = 1000000.0
        tr = Trace(data=sig * 500.0, delta=dt, station="GRA1", comp="Z", start_epoch=t0)
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(t0 + 10.0), station="GRA1"))
        return tr

    def test_cursor_mode_constants_and_setters(self) -> None:
        self.assertEqual(CURSOR_LOCK_MODES, ("CROSSHAIR", "PEAK_LOCK", "ZERO_LOCK"))
        self.assertEqual(MAGNIFIER_CURSOR_MODES, ("STANDARD", "HILBERT", "REFERENCE"))

        mag = MagnifierCanvas()
        modes_emitted = []
        mag.cursorOverlayModeChanged.connect(modes_emitted.append)

        mag.set_cursor_mode("PEAK_LOCK")
        self.assertEqual(mag.cursor_lock_mode, "PEAK_LOCK")
        self.assertEqual(mag.cursor_overlay_mode, "STANDARD")

        mag.set_cursor_mode("HILBERT")
        self.assertEqual(mag.cursor_overlay_mode, "HILBERT")
        mag.set_cursor_overlay_mode("REFERENCE")
        self.assertEqual(mag.cursor_overlay_mode, "REFERENCE")
        mag.set_cursor_overlay_mode("STANDARD")
        self.assertEqual(mag.cursor_overlay_mode, "STANDARD")
        self.assertEqual(modes_emitted, ["HILBERT", "REFERENCE", "STANDARD"])

    def test_hilbert_cursor_mode_rendering_and_math(self) -> None:
        tr = self._make_wave_trace()
        mag = MagnifierCanvas()
        mag.resize(720, 220)
        mag.set_magnifier_view(tr, 0, tr.start_epoch, 5.0, 25.0)
        mag.show()
        app.processEvents()

        hover_x = mag.time_to_x(20.0)
        hover_y = mag.plot_rect().center().y()
        _send_mouse(mag, "move", hover_x, hover_y)
        img_std = _pixmap_image(mag)

        # Switch to HILBERT mode using an explicit reference wavelet
        ref_t = np.arange(80, dtype=np.float64) * 0.05
        ref_seg = np.sin(2.0 * np.pi * 1.5 * ref_t)
        mag.set_reference_waveform(ref_seg, delta=0.05, center_rel_sec=2.0)
        mag.set_cursor_overlay_mode("HILBERT")

        ov = mag._get_cursor_overlay_waveform()
        self.assertIsNotNone(ov)
        ov_arr, ov_dt, ov_center = ov
        self.assertAlmostEqual(ov_dt, 0.05)
        self.assertAlmostEqual(ov_center, 2.0)
        expected_h = hilbert_transform(ref_seg - np.mean(ref_seg))
        self.assertTrue(np.allclose(ov_arr, expected_h, atol=1e-9))

        img_hilb = _pixmap_image(mag)
        self.assertTrue(_images_differ(img_std, img_hilb), "HILBERT overlay must paint onto MagnifierCanvas")

        # Also verify HILBERT mode works without an explicit reference (uses active trace/pick)
        mag.set_reference_waveform(None)
        ov_trace = mag._get_cursor_overlay_waveform()
        self.assertIsNotNone(ov_trace)
        self.assertGreater(ov_trace[0].size, 10)
        _ = _pixmap_image(mag)
        mag.close()

    def test_reference_waveform_capture_and_rendering(self) -> None:
        tr = self._make_wave_trace()
        mag = MagnifierCanvas()
        mag.resize(720, 220)
        mag.set_magnifier_view(tr, 0, tr.start_epoch, 8.0, 12.0)
        mag.show()
        app.processEvents()

        key_before = mag._current_state_key()
        self.assertTrue(mag.capture_reference_window())
        self.assertNotEqual(key_before, mag._current_state_key())

        # Move to second arrival window and render REFERENCE overlay
        mag.set_magnifier_view(tr, 0, tr.start_epoch, 15.0, 25.0)
        hover_x = mag.time_to_x(20.0)
        hover_y = mag.plot_rect().center().y()
        _send_mouse(mag, "move", hover_x, hover_y)
        img_std = _pixmap_image(mag)

        mag.set_cursor_overlay_mode("REFERENCE")
        img_ref = _pixmap_image(mag)
        self.assertTrue(_images_differ(img_std, img_ref), "REFERENCE cursor mode must draw captured reference waveform")

        # Also verify passing a Trace directly to set_reference_waveform
        mag.set_reference_waveform(tr, center_rel_sec=10.0)
        ov = mag._get_cursor_overlay_waveform()
        self.assertIsNotNone(ov)
        self.assertEqual(ov[0].size, tr.length)
        self.assertAlmostEqual(ov[2], 10.0)
        mag.close()


class TestOverlayFilterBadgesM24(unittest.TestCase):
    """M-24: Trace-specific overlay filter badges (colour_addfil) in MainTraceCanvas."""

    def test_overlay_filter_badge_state_key_colors_and_rendering(self) -> None:
        dt = 0.05
        data = np.sin(np.linspace(0.0, 20.0 * np.pi, 400))
        tr0 = Trace(data=data, delta=dt, station="GRA1", comp="Z", start_epoch=1000000.0)
        tr1 = Trace(data=data, delta=dt, station="GRA2", comp="Z", start_epoch=1000000.0)

        cv = MainTraceCanvas()
        cv.resize(800, 260)
        cv.set_traces([tr0, tr1], reset_view=True)

        key_none = cv._current_state_key()
        img_none = _pixmap_image(cv)

        tr0.overlay_filter_name = "BP 1-5Hz"
        tr0.overlay_filter_color = "red"
        key_red = cv._current_state_key()
        self.assertNotEqual(key_none, key_red)
        img_red = _pixmap_image(cv)
        self.assertTrue(_images_differ(img_none, img_red), "Overlay filter badge must render into margin")

        tr0.overlay_filter_color = "blue"
        key_blue = cv._current_state_key()
        self.assertNotEqual(key_red, key_blue)
        img_blue = _pixmap_image(cv)
        self.assertTrue(_images_differ(img_red, img_blue), "Changing overlay_filter_color must alter badge rendering")

        # Verify color resolver handles named colors, hex colors, and deterministic defaults
        self.assertNotEqual(_resolve_overlay_filter_color("red", "BP"), _resolve_overlay_filter_color("green", "BP"))
        self.assertNotEqual(_resolve_overlay_filter_color("green", "BP"), _resolve_overlay_filter_color("blue", "BP"))
        self.assertEqual(_resolve_overlay_filter_color("#12ab34", "BP"), QColor("#12ab34"))
        self.assertTrue(_resolve_overlay_filter_color("", "LP_2HZ").isValid())

        # Verify signals exist
        self.assertTrue(hasattr(cv, "overlayFilterRequested"))
        self.assertTrue(hasattr(cv, "clearOverlayFilterRequested"))


if __name__ == "__main__":
    unittest.main()
