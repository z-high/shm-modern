"""Tests for the usability overhaul: menus, tutorial mode, pick editing, context menu, pan/history,
row height + scrolling, undo/redo, ALIGN/BEAM toggles and non-modal dialogs."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from shm_app.gui.help_texts import HELP_TEXTS
from shm_app.gui.main_window import SHMMainWindow

L = Qt.MouseButton.LeftButton
R = Qt.MouseButton.RightButton
NO = Qt.MouseButton.NoButton
NOMOD = Qt.KeyboardModifier.NoModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


def mouse(widget, kind, x, y, button=NO, buttons=NO, mods=NOMOD):
    """Send a synthetic mouse event (QTest cannot hold buttons during moves)."""
    t = {
        "press": QEvent.Type.MouseButtonPress,
        "release": QEvent.Type.MouseButtonRelease,
        "move": QEvent.Type.MouseMove,
        "dbl": QEvent.Type.MouseButtonDblClick,
    }[kind]
    ev = QMouseEvent(t, QPointF(x, y), widget.mapToGlobal(QPointF(x, y)), button, buttons, mods)
    QApplication.sendEvent(widget, ev)


class UsabilityBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.win = SHMMainWindow(autoload_example=True)
        self.win.resize(1500, 900)
        self.win.show()
        self.app.processEvents()
        self.cv = self.win.main_canvas

    def tearDown(self) -> None:
        for d in list(self.win._open_dialogs):
            d.close()
        self.win.close()

    def slot_center_y(self, orig_idx: int) -> float:
        vis = self.cv.visible_indexed_traces()
        slot_i = next(i for i, (o, _) in enumerate(vis) if o == orig_idx)
        h = self.cv.slot_height(len(vis))
        return self.cv.slot_top(slot_i, h) + 0.5 * h

    def add_pick(self, idx: int, rel_t: float):
        ref = self.cv.reference_epoch()
        tr = self.win.engine.traces[idx]
        self.win._on_main_trace_clicked(idx, ref + rel_t - tr.display_offset, rel_t)
        return tr.get_pick(self.win.phase_panel.active_phase)


class TestMenusAndTutorial(UsabilityBase):
    def test_menu_structure_and_samples_not_in_main_menus(self) -> None:
        titles = [a.text().replace("&", "") for a in self.win.menuBar().actions()]
        self.assertEqual(
            titles,
            ["File", "Edit", "Traces", "View", "Process", "Analysis", "Location", "Tools", "Help"],
        )

        def texts(menu):
            out = []
            for a in menu.actions():
                out.append(a.text())
                if a.menu():
                    out.extend(texts(a.menu()))
            return out

        for act in self.win.menuBar().actions():
            label = act.text().replace("&", "")
            if label == "Help":
                continue
            self.assertFalse(
                any("sample data" in t.lower() or "example" in t.lower() for t in texts(act.menu())),
                f"sample data must not appear in menu {label}",
            )
        help_menu = next(a.menu() for a in self.win.menuBar().actions() if a.text() == "&Help")
        self.assertTrue(any("Sample Data" in t for t in texts(help_menu)))

        from PyQt6.QtWidgets import QToolBar

        tb_texts = [a.text().lower() for a in self.win.findChild(QToolBar).actions()]
        self.assertTrue(tb_texts)
        self.assertFalse(any("example" in t or "sample" in t for t in tb_texts))

        # Nested tree structure exists (e.g. Process > Filter > Filter Dialog)
        proc = next(a.menu() for a in self.win.menuBar().actions() if a.text() == "&Process")
        sub_titles = [a.text().replace("&", "") for a in proc.actions() if a.menu()]
        self.assertIn("Filter", sub_titles)
        self.assertIn("Rotate Components", sub_titles)

    def test_every_action_has_help_text_and_no_duplicate_shortcuts(self) -> None:
        for act, _title, key, _sc in self.win._help_actions:
            self.assertIn(key, HELP_TEXTS, f"missing help text for {key}")
            self.assertTrue(act.statusTip())
        seqs = [
            a.shortcut().toString()
            for a, _t, _k, _s in self.win._help_actions
            if a.shortcut().toString()
        ]
        self.assertEqual(len(seqs), len(set(seqs)), "duplicate shortcuts")

    def test_tutorial_mode_toggles_rich_tooltips(self) -> None:
        act = next(a for a, _t, k, _s in self.win._help_actions if k == "beam")
        self.assertNotIn("<table", act.toolTip())
        self.assertFalse(self.win._menus[0].toolTipsVisible())
        self.win.act_tutorial.setChecked(True)
        self.assertIn("<table", act.toolTip())
        self.assertIn("BEAM", act.toolTip())
        self.assertTrue(all(m.toolTipsVisible() for m in self.win._menus))
        self.win.act_tutorial.setChecked(False)
        self.assertNotIn("<table", act.toolTip())
        self.assertFalse(any(m.toolTipsVisible() for m in self.win._menus))


class TestPickEditing(UsabilityBase):
    def test_hit_test_move_and_delete(self) -> None:
        pk = self.add_pick(2, 25.0)
        self.assertIsNotNone(pk)
        x = self.cv.time_to_x(25.0)
        y = self.slot_center_y(2)
        hit = self.cv.pick_at(QPointF(x + 3, y))
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], 2)
        self.assertIsNone(self.cv.pick_at(QPointF(x + 40, y)))

        # Dragging the pick moves it instead of creating a second one
        n_before = len(self.win.engine.traces[2].picks)
        mouse(self.cv, "press", x, y, L, L)
        x2 = x + 80
        mouse(self.cv, "move", x2, y, NO, L)
        mouse(self.cv, "release", x2, y, L, NO)
        tr = self.win.engine.traces[2]
        self.assertEqual(len(tr.picks), n_before)
        self.assertAlmostEqual(self.cv.time_to_x(tr.picks[0].onset_epoch - self.cv.reference_epoch()), x2, delta=3.0)

        # Double-click on the pick deletes only that pick
        self.add_pick(5, 25.0)
        x5 = self.cv.time_to_x(25.0)
        mouse(self.cv, "dbl", x5, self.slot_center_y(5), L, L)
        self.assertEqual(self.win.engine.traces[5].picks, [])
        self.assertEqual(len(self.win.engine.traces[2].picks), n_before)

    def test_delete_key_on_hovered_pick(self) -> None:
        self.add_pick(1, 30.0)
        x = self.cv.time_to_x(30.0)
        y = self.slot_center_y(1)
        mouse(self.cv, "move", x - 20, y)
        mouse(self.cv, "move", x, y)
        self.assertIsNotNone(self.cv._hover_pick)
        from PyQt6.QtGui import QKeyEvent

        self.cv.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, NOMOD))
        self.assertEqual(self.win.engine.traces[1].picks, [])

    def test_switching_trace_focuses_magnifier_without_creating_pick(self) -> None:
        self.cv.mag_trace_idx = 0
        x = self.cv.time_to_x(35.0)
        y3 = self.slot_center_y(3)

        # First click switches active trace to #3 without creating a pick
        mouse(self.cv, "press", x, y3, L, L)
        mouse(self.cv, "release", x, y3, L, NO)
        self.assertEqual(self.cv.mag_trace_idx, 3)
        self.assertEqual(self.win.mag_canvas.trace_idx, 3)
        self.assertEqual(self.win.engine.traces[3].picks, [])

        # Subsequent click on the already-active trace #3 places a pick
        mouse(self.cv, "press", x, y3, L, L)
        mouse(self.cv, "release", x, y3, L, NO)
        self.assertEqual(len(self.win.engine.traces[3].picks), 1)

    def test_magnifier_double_click_and_drag_pick(self) -> None:
        self.win._focus_magnifier(2, 35.0)
        mag = self.win.mag_canvas
        mag.resize(800, 220)
        self.app.processEvents()
        self.assertEqual(mag.trace_idx, 2)

        y_mid = mag.plot_rect().center().y()
        x_pick = mag.time_to_x(35.0)
        mouse(mag, "press", x_pick, y_mid, L, L)
        mouse(mag, "release", x_pick, y_mid, L, NO)
        tr = self.win.engine.traces[2]
        self.assertEqual(len(tr.picks), 1)
        t_orig = tr.picks[0].onset_epoch

        # Dragging the pick inside Magnifier moves it without duplicating
        x_moved = mag.time_to_x(38.0)
        mouse(mag, "press", x_pick, y_mid, L, L)
        mouse(mag, "move", x_moved, y_mid, NO, L)
        mouse(mag, "release", x_moved, y_mid, L, NO)
        self.assertEqual(len(tr.picks), 1)
        self.assertGreater(tr.picks[0].onset_epoch, t_orig + 2.0)

        # Full double-click sequence deletes the pick cleanly with a single Undo step
        px_now = mag.time_to_x(tr.picks[0].onset_epoch + tr.display_offset - mag.ref_epoch)
        mouse(mag, "press", px_now, y_mid, L, L)
        mouse(mag, "release", px_now, y_mid, L, NO)
        mouse(mag, "dbl", px_now, y_mid, L, L)
        mouse(mag, "release", px_now, y_mid, L, NO)
        self.assertEqual(tr.picks, [])

        self.win.action_undo()
        tr = self.win.engine.traces[2]
        self.assertIs(self.win.mag_canvas.trace, tr)
        self.assertEqual(len(tr.picks), 1)

        # Hover + Del key in Magnifier also deletes the pick
        px_restored = mag.time_to_x(tr.picks[0].onset_epoch + tr.display_offset - mag.ref_epoch)
        mouse(mag, "move", px_restored, y_mid)
        from PyQt6.QtGui import QKeyEvent

        mag.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, NOMOD))
        self.assertEqual(tr.picks, [])

    def test_undo_redo_of_picks(self) -> None:
        self.add_pick(0, 20.0)
        self.assertEqual(len(self.win.engine.traces[0].picks), 1)
        self.win.action_undo()
        self.assertEqual(self.win.engine.traces[0].picks, [])
        self.win.action_redo()
        self.assertEqual(len(self.win.engine.traces[0].picks), 1)

    def test_undo_filter_and_trace_delete(self) -> None:
        self.win.cmd_input.setText("BUTTER BP 0.8 3.0 3")
        self.win._on_run_command_line()
        self.assertIn("BUTTER", self.win.engine.traces[0].filter_name)
        self.win.action_undo()
        self.assertNotIn("BUTTER", self.win.engine.traces[0].filter_name)

        self.win.action_select_all()
        self.win.action_delete_selected()
        self.assertEqual(len(self.win.engine.traces), 0)
        self.win.action_undo()
        self.assertEqual(len(self.win.engine.traces), 19)


class TestContextMenuAndNavigation(UsabilityBase):
    def test_quick_right_click_requests_menu_but_drag_moves_magnifier(self) -> None:
        got = []
        # The real handler opens a blocking QMenu; replace it by a recorder in the test
        self.cv.contextMenuRequested.disconnect(self.win._on_trace_context_menu)
        self.cv.contextMenuRequested.connect(lambda i, t, gx, gy: got.append((i, t)))
        x, y = self.cv.time_to_x(20.0), self.slot_center_y(3)

        mouse(self.cv, "press", x, y, R, R)
        mouse(self.cv, "release", x, y, R, NO)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][0], 3)

        lo0, hi0 = self.cv.mag_t_lo, self.cv.mag_t_hi
        mouse(self.cv, "press", x, y, R, R)
        mouse(self.cv, "move", x + 60, y, NO, R)
        mouse(self.cv, "release", x + 60, y, R, NO)
        self.assertEqual(len(got), 1, "drag must not open the context menu")
        self.assertNotEqual((lo0, hi0), (self.cv.mag_t_lo, self.cv.mag_t_hi))

    def test_context_menu_actions(self) -> None:
        win = self.win
        win._invert_trace_sign(0)
        win._hide_trace(1)
        self.assertTrue(win.engine.traces[1].hidden)
        win.action_show_hidden()
        self.assertFalse(win.engine.traces[1].hidden)
        win._focus_magnifier(2, 40.0)
        self.assertEqual(self.cv.mag_trace_idx, 2)
        self.assertAlmostEqual(0.5 * (self.cv.mag_t_lo + self.cv.mag_t_hi), 40.0, places=3)

    def test_shift_drag_pans_and_history_goes_back(self) -> None:
        lo0, hi0 = self.cv.active_time_span()
        x, y = self.cv.time_to_x(lo0 + 0.5 * (hi0 - lo0)), self.slot_center_y(0)
        mouse(self.cv, "press", x, y, L, L, SHIFT)
        mouse(self.cv, "move", x - 120, y, NO, L, SHIFT)
        mouse(self.cv, "release", x - 120, y, L, NO, SHIFT)
        lo1, hi1 = self.cv.active_time_span()
        self.assertGreater(lo1, lo0)
        self.assertAlmostEqual(hi1 - lo1, hi0 - lo0, places=6)
        self.assertEqual(len(self.win.engine.traces[0].picks), 0, "pan must not create a pick")

        self.assertTrue(self.cv.view_back())
        lo2, hi2 = self.cv.active_time_span()
        self.assertAlmostEqual(lo2, lo0, places=6)
        self.assertTrue(self.cv.view_forward())
        self.assertAlmostEqual(self.cv.active_time_span()[0], lo1, places=6)

    def test_row_height_scrolling_and_hit_testing(self) -> None:
        self.cv.set_min_slot_height(0)
        self.app.processEvents()
        self.assertFalse(self.win.v_scroll.isVisibleTo(self.win))

        self.cv.set_min_slot_height(64)
        self.app.processEvents()
        n = len(self.cv.visible_indexed_traces())
        self.assertGreater(self.cv.max_v_offset(), 0)
        self.assertTrue(self.win.v_scroll.isVisibleTo(self.win))
        self.assertEqual(self.win.v_scroll.maximum(), int(self.cv.max_v_offset() + 0.999))
        self.assertAlmostEqual(self.cv.slot_height(n), 64.0)

        self.cv.set_v_offset(1e9)
        self.assertAlmostEqual(self.cv.v_offset, self.cv.max_v_offset())
        self.assertEqual(self.win.v_scroll.value(), int(self.cv.v_offset))

        # The last trace is now visible and hit-testing maps to it
        rect = self.cv.plot_rect()
        self.assertEqual(self.cv.y_to_visible_slot(rect.bottom() - 5), n - 1)
        self.cv.repaint()
        self.cv.invalidate_slot(n - 1)
        self.cv.repaint()

        # Selecting via the left margin of the last row still works
        y = rect.bottom() - 10
        mouse(self.cv, "press", 20, y, L, L)
        mouse(self.cv, "release", 20, y, L, NO)
        self.assertTrue(self.win.engine.traces[n - 1].selected)

        self.cv.set_min_slot_height(0)
        self.assertEqual(self.cv.v_offset, 0.0)


class TestToggleAndNonModal(UsabilityBase):
    def test_align_toggle_restores_traces(self) -> None:
        win = self.win
        win.engine.execute_line("PICK ALL")
        n0 = len(win.engine.traces)
        offs0 = [t.display_offset for t in win.engine.traces]
        win.action_align_phase()
        self.assertEqual(len(win.engine.traces), n0 + 1)
        self.assertEqual(win.engine.traces[-1].station, "ALIGN")
        self.assertTrue(any(abs(t.display_offset - o) > 1e-9 for t, o in zip(win.engine.traces, offs0)))
        win.action_align_phase()
        self.assertEqual(len(win.engine.traces), n0)
        for t, o in zip(win.engine.traces, offs0):
            self.assertAlmostEqual(t.display_offset, o, places=9)

    def test_beam_toggle_and_undo_of_beam(self) -> None:
        win = self.win
        n0 = len(win.engine.traces)
        win.action_form_beam()
        self.assertEqual(len(win.engine.traces), n0 + 1)
        win.action_form_beam()
        self.assertEqual(len(win.engine.traces), n0)
        win.action_form_beam()
        win.action_undo()
        self.assertEqual(len(win.engine.traces), n0)

    def test_analysis_dialogs_are_non_modal(self) -> None:
        win = self.win
        win.action_open_spectrum_dialog()
        self.assertEqual(len(win._open_dialogs), 1)
        dlg = win._open_dialogs[0]
        self.assertFalse(dlg.isModal())
        self.assertTrue(dlg.isVisible())
        # Main window stays usable while the dialog is open
        win.action_select_all()
        self.assertTrue(all(t.selected for t in win.engine.traces))
        dlg.close()
        self.app.processEvents()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        self.assertEqual(len(win._open_dialogs), 0)

    def test_state_label_and_menu_actions_do_not_crash(self) -> None:
        win = self.win
        win._update_state_label()
        self.assertIn("Phase P", win.state_lbl.text())
        win.action_view_back()
        win.action_view_forward()
        win.action_undo()
        win.action_redo()


class TestSpectrumWindowGroupingSlider(UsabilityBase):
    def test_spectrum_accumulates_curves_from_yellow_window(self) -> None:
        win, cv = self.win, self.cv
        cv.mag_trace_idx = 0
        cv.mag_t_lo, cv.mag_t_hi = 80.0, 110.0
        win.action_open_spectrum_dialog()
        self.assertEqual(len(win._open_dialogs), 1)
        dlg = win._open_dialogs[0]
        self.assertEqual(len(dlg.curves), 1)
        c0 = dlg.curves[0]
        tr = win.engine.traces[0]
        rel_start = tr.start_epoch + tr.display_offset - cv.reference_epoch()
        self.assertAlmostEqual(c0["t_lo"], 80.0 - rel_start, places=6)
        self.assertAlmostEqual(c0["t_hi"], 110.0 - rel_start, places=6)

        # Same window again: no duplicate. Moved window: second curve in the SAME dialog, other colour
        win.action_open_spectrum_dialog()
        self.assertEqual(len(dlg.curves), 1)
        cv.mag_t_lo, cv.mag_t_hi = 150.0, 180.0
        win.action_open_spectrum_dialog()
        self.assertEqual(len(win._open_dialogs), 1)
        self.assertEqual(len(dlg.curves), 2)
        self.assertNotEqual(dlg.curves[0]["color"], dlg.curves[1]["color"])

        # The curve only depends on the selected part of the trace
        from shm_app.processing.spectrum import compute_fft_spectrum

        f_full, a_full = compute_fft_spectrum(tr)
        f_win, a_win = compute_fft_spectrum(dlg.curves[0]["trace"], c0["t_lo"], c0["t_hi"])
        self.assertLess(len(f_win), len(f_full))

        # Selected traces add further curves
        win.engine.traces[1].selected = True
        win.engine.traces[2].selected = True
        win.action_open_spectrum_selected()
        self.assertEqual(len(dlg.curves), 4)
        for m in range(5):
            dlg.method_combo.setCurrentIndex(m)
        dlg.clear_curves()
        self.assertEqual(dlg.curves, [])

    def test_spectrum_window_without_data_adds_nothing(self) -> None:
        self.cv.mag_trace_idx = 0
        self.cv.mag_t_lo, self.cv.mag_t_hi = 5000.0, 5030.0
        self.win.action_open_spectrum_dialog()
        self.assertEqual(self.win._open_dialogs, [])

    def test_move_yellow_window_keeps_length_and_creates_no_pick(self) -> None:
        cv = self.cv
        cv.mag_trace_idx = 4
        cv.mag_t_lo, cv.mag_t_hi = 60.0, 90.0
        cv.update()
        handle = cv._mag_handle_rect()
        self.assertIsNotNone(handle)
        x, y = handle.center().x(), handle.center().y()
        mouse(cv, "press", x, y, L, L)
        mouse(cv, "move", x + 100, y, NO, L)
        mouse(cv, "release", x + 100, y, L, NO)
        self.assertAlmostEqual(cv.mag_t_hi - cv.mag_t_lo, 30.0, places=6)
        self.assertGreater(cv.mag_t_lo, 60.0)
        self.assertTrue(all(t.picks == [] for t in self.win.engine.traces))
        # mag canvas follows
        self.assertAlmostEqual(self.win.mag_canvas.t_lo, cv.mag_t_lo, places=6)

        # Clamped at the data edges, length still constant
        handle = cv._mag_handle_rect()
        x, y = handle.center().x(), handle.center().y()
        mouse(cv, "press", x, y, L, L)
        mouse(cv, "move", x + 5000, y, NO, L)
        mouse(cv, "release", x + 5000, y, L, NO)
        f0, f1 = cv.full_time_span()
        self.assertAlmostEqual(cv.mag_t_hi, f1, places=6)
        self.assertAlmostEqual(cv.mag_t_hi - cv.mag_t_lo, 30.0, places=6)

        # Ctrl + drag anywhere on a waveform also moves it
        lo = cv.mag_t_lo
        xx, yy = cv.time_to_x(30.0), self.slot_center_y(0)
        mouse(cv, "press", xx, yy, L, L, Qt.KeyboardModifier.ControlModifier)
        mouse(cv, "move", xx - 60, yy, NO, L, Qt.KeyboardModifier.ControlModifier)
        mouse(cv, "release", xx - 60, yy, L, NO, Qt.KeyboardModifier.ControlModifier)
        self.assertNotAlmostEqual(cv.mag_t_lo, lo, places=3)
        self.assertAlmostEqual(cv.mag_t_hi - cv.mag_t_lo, 30.0, places=6)
        self.assertTrue(all(t.picks == [] for t in self.win.engine.traces))

    def test_group_components_by_station(self) -> None:
        win, cv = self.win, self.cv
        win.engine.traces.reverse()  # scrambles Z, N, E order
        win.refresh_views()
        win.act_group.setChecked(True)
        win.action_toggle_grouping()
        vis = cv.visible_indexed_traces()
        stations = [t.station for _, t in vis]
        # each station appears as one contiguous block
        seen = []
        for st in stations:
            if not seen or seen[-1] != st:
                seen.append(st)
        self.assertEqual(len(seen), len(set(seen)))
        rank = {"Z": 0, "N": 1, "E": 2}
        for st in set(stations):
            comps = [rank[t.comp] for _, t in vis if t.station == st and t.comp in rank]
            self.assertEqual(comps, sorted(comps))
        self.assertTrue(cv._group_ends)
        cv.repaint()
        # Hit-testing still maps to the right original trace
        slot_y = self.slot_center_y(vis[3][0])
        self.assertEqual(vis[cv.y_to_visible_slot(slot_y)][0], vis[3][0])
        win.act_group.setChecked(False)
        win.action_toggle_grouping()
        self.assertEqual([i for i, _ in cv.visible_indexed_traces()], sorted(i for i, _ in cv.visible_indexed_traces()))

    def test_row_height_slider_free_adjustment_and_preset_sync(self) -> None:
        win, cv = self.win, self.cv
        win.row_slider.setValue(30)  # -> 46 px
        self.assertEqual(cv.min_slot_h, 46.0)
        self.assertEqual(win.row_label.text(), "46 px")
        self.assertTrue(all(not a.isChecked() for a in win._row_actions.values()))
        win.row_slider.setValue(24)  # -> 40 px equals the Normal preset
        self.assertEqual(cv.min_slot_h, 40.0)
        self.assertTrue(win._row_actions[40].isChecked())
        win._set_row_height(64)
        self.assertEqual(win.row_slider.value(), 48)
        self.assertEqual(win.row_label.text(), "64 px")
        win.row_slider.setValue(0)
        self.assertEqual(cv.min_slot_h, 0.0)
        self.assertEqual(win.row_label.text(), "Fit")
        self.assertTrue(win._row_actions[0].isChecked())

        # Changing row height keeps the same top trace in view
        win._set_row_height(60)
        cv.set_v_offset(60 * 5)  # trace 5 at the top
        win.row_slider.setValue(84 - 16 + 16)  # 84 px
        h = cv.slot_height()
        self.assertAlmostEqual(cv.v_offset / h, 5.0, delta=0.05)

    def test_spectrum_reopens_after_repeated_corner_x_closes(self) -> None:
        from PyQt6 import sip
        from PyQt6.QtGui import QCloseEvent
        from shm_app.gui.dialogs import SpectrumDialog

        win = self.win
        # Simulate repeated open + corner-X close cycles, both with and without event-loop flushes
        for i in range(6):
            win.action_open_spectrum_dialog()
            self.app.processEvents()
            dlgs = [d for d in win._open_dialogs if isinstance(d, SpectrumDialog)]
            self.assertEqual(len(dlgs), 1, f"cycle {i}: expected 1 open SpectrumDialog")
            dlg = dlgs[0]
            self.assertTrue(dlg.isVisible())

            QApplication.sendEvent(dlg, QCloseEvent())
            self.assertFalse(dlg.isVisible())
            self.assertEqual([d for d in win._open_dialogs if isinstance(d, SpectrumDialog)], [])
            if i % 2 == 0:
                self.app.processEvents()
                self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

        # Even if a deleted C++ dialog wrapper lingers in _open_dialogs, reopening must still work cleanly
        stale = SpectrumDialog(None, parent=win)
        win._open_dialogs.append(stale)
        sip.delete(stale)
        win.action_open_spectrum_dialog()
        self.app.processEvents()
        self.assertEqual(len([d for d in win._open_dialogs if isinstance(d, SpectrumDialog)]), 1)
        self.assertTrue(win._open_dialogs[-1].isVisible())

    def test_spectrum_cursor_value_pair_readout(self) -> None:
        from PyQt6.QtCore import QEvent as QtEvent

        win, cv = self.win, self.cv
        cv.mag_trace_idx = 0
        cv.mag_t_lo, cv.mag_t_hi = 80.0, 120.0
        win.action_open_spectrum_dialog()
        dlg = win._open_dialogs[-1]
        dlg.canvas.resize(640, 400)
        dlg.canvas.draw()

        bbox = dlg._ax.bbox
        # Canvas Qt coords use top-down y, Matplotlib bbox uses bottom-up y
        qx = 0.5 * (bbox.x0 + bbox.x1)
        qy = dlg.canvas.height() - 0.5 * (bbox.y0 + bbox.y1)
        mouse(dlg.canvas, "move", qx, qy, NO, NO)
        txt_fft = dlg.cursor_label.text()
        self.assertIn("Hz", txt_fft)
        self.assertIn("A =", txt_fft)
        self.assertIn("T =", txt_fft)

        # Switch to Spectrogram -> readout shows (t, f) + Power in dB
        dlg.method_combo.setCurrentIndex(4)
        dlg.canvas.draw()
        bbox = dlg._ax.bbox
        qx = 0.5 * (bbox.x0 + bbox.x1)
        qy = dlg.canvas.height() - 0.5 * (bbox.y0 + bbox.y1)
        mouse(dlg.canvas, "move", qx, qy, NO, NO)
        txt_spec = dlg.cursor_label.text()
        self.assertIn("s, f =", txt_spec)
        self.assertIn("Hz", txt_spec)
        self.assertIn("Power =", txt_spec)

        # Leaving canvas resets cursor readout
        QApplication.sendEvent(dlg.canvas, QEvent(QtEvent.Type.Leave))
        self.assertEqual(dlg.cursor_label.text(), "Cursor: —")

    def test_high_resolution_trace_polygon_and_extrema(self):
        import numpy as np
        from shm_app.gui.waveform_canvases import _build_trace_polygon

        # 6000 samples on a 1200px plot draws every sample directly without harsh 450-bin compression
        data_6k = np.sin(np.linspace(0.0, 40.0 * np.pi, 6000))
        poly_6k = _build_trace_polygon(data_6k, 0, 6000, 0.0, 0.05, 0.0, 300.0, 0.0, 1200.0, 50.0, 20.0, 10.0, 90.0, 0.0)
        self.assertEqual(poly_6k.count(), 6000)

        # Very large trace uses high-density true-timestamp bins and preserves spikes (including tail remainder)
        data_big = np.zeros(50003, dtype=np.float64)
        spike_idx = 49999
        data_big[spike_idx] = 1.0
        poly_big = _build_trace_polygon(data_big, 0, 50003, 0.0, 0.05, 0.0, 50003 * 0.05, 0.0, 1200.0, 50.0, 20.0, 10.0, 90.0, 0.0)
        self.assertGreaterEqual(poly_big.count(), 2400)
        ys = [poly_big.at(i).y() for i in range(poly_big.count())]
        self.assertAlmostEqual(min(ys), 30.0, places=4)


import glob
import re
import tempfile
from unittest.mock import patch

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QPushButton,
    QTabBar,
)

from shm_app.core.config_station import get_config
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace, normalize_shm_event_type
from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    FilterDialog,
    ParticleMotionDialog,
    SpectrumDialog,
    VespagramDialog,
)
from shm_app.gui.dialogs_io import ReadDataDialog
from shm_app.gui.dialogs_location import EventSummaryDialog, NetworkLocationDialog
from shm_app.gui.panels import ParameterBoxPanel
from shm_app.gui.world_map import WorldMapDialog
from shm_app.plugins.bgr_dialog import BGRFetchDialog
from shm_app.plugins.live_dialog import LiveEventDialog
from shm_app.plugins.live_events import LiveEvent
from shm_app.plugins.live_planner import evaluate_event_relevance
from shm_app.plugins.live_service import LiveEventEntry, build_analysis_param_set


class TestMainWindowLayoutAndStatus(UsabilityBase):
    """Regressions for main window height, status bar compactness, and live notification banner."""

    def test_main_window_resize_compact_and_label_widths(self) -> None:
        win = self.win
        win.resize(1380, 860)
        self.app.processEvents()
        self.assertEqual(win.size().height(), 860)

        win.resize(1380, 760)
        self.app.processEvents()
        self.assertLessEqual(win.size().height(), 800)

        min_hint = win.minimumSizeHint()
        self.assertLessEqual(min_hint.height(), 750)

        self.assertLessEqual(win.cursor_info_lbl.minimumWidth(), 200)
        self.assertLessEqual(win.state_lbl.minimumWidth(), 220)

    def test_no_duplicate_live_event_ready_message(self) -> None:
        win = self.win
        ev = LiveEvent(
            event_id="TEST_EV_1",
            origin_epoch=1700000000.0,
            origin_iso="2023-11-14T22:13:20Z",
            lat=38.0,
            lon=20.0,
            depth_km=10.0,
            magnitude=5.8,
            mag_type="Mw",
            region="GREECE",
            source_feed="emsc_ws",
            event_type="quarry blast",
        )
        eva = evaluate_event_relevance(ev, win.live_coordinator.live_config, win.config)
        entry = LiveEventEntry(event=ev, evaluation=eva, fetched_traces=[win.engine.traces[0].copy()])
        win.live_coordinator.entries["TEST_EV_1"] = entry
        win.status_bar.clearMessage()
        win._on_live_event_ready("TEST_EV_1")
        self.assertTrue(win.live_notify_btn.isVisible())
        self.assertNotIn("Live Event Ready", win.status_bar.currentMessage())
        win.live_notify_btn.setVisible(False)


class TestDialogQualityStandards(UsabilityBase):
    """Regressions for dialog size hints, autoDefault, C-locale, and dark colorbar styles across all 11 dialogs."""

    def _assert_dialog_quality(self, name: str, dlg: QDialog) -> None:
        dlg.show()
        self.app.processEvents()
        min_h = dlg.minimumSizeHint()
        sz = dlg.size()
        self.assertGreaterEqual(sz.width(), min_h.width(), f"{name}: width {sz.width()} < minHint {min_h.width()}")
        self.assertGreaterEqual(sz.height(), min_h.height(), f"{name}: height {sz.height()} < minHint {min_h.height()}")
        auto_btns = [b.text() for b in dlg.findChildren(QPushButton) if b.autoDefault()]
        self.assertEqual(auto_btns, [], f"{name}: autoDefault buttons left: {auto_btns}")
        spin_locales = [sb.locale().name() for sb in dlg.findChildren(QDoubleSpinBox)]
        self.assertTrue(
            all(loc in ("C", "en_US") or loc.startswith("C") for loc in spin_locales),
            f"{name}: unexpected spinbox locales {spin_locales}",
        )
        for sb in dlg.findChildren(QDoubleSpinBox):
            self.assertIn(".", sb.textFromValue(1.5), f"{name}: comma decimal leaked in {sb}")

    def test_all_eleven_dialogs_size_hints_autodefault_and_c_locale(self) -> None:
        win = self.win
        vis = win.engine.visible_traces()
        picks_list = [PhasePick(name="P", onset="02-NOV-2003_02:25:30.123", station=vis[0].station, comp=vis[0].comp)]

        # 1. BGRFetchDialog
        bgr_dlg = BGRFetchDialog(
            config=win.config,
            view_window=(vis[0].start_epoch, 120.0),
            origin_str="02-NOV-2003_02:25:00.000",
            workspace_stations=["GRA1", "GRA2", "GRB1"],
            parent=win,
        )
        self._assert_dialog_quality("BGRFetchDialog", bgr_dlg)
        bgr_dlg.close()

        # 2. ReadDataDialog
        read_dlg = ReadDataDialog(config=win.config, parent=win)
        self._assert_dialog_quality("ReadDataDialog", read_dlg)
        read_dlg.close()

        # 3. FKAnalysisDialog
        fk_dlg = FKAnalysisDialog(vis, t_lo=10.0, t_hi=40.0, config=win.config, parent=win)
        fk_dlg.run_fk()
        self._assert_dialog_quality("FKAnalysisDialog", fk_dlg)
        fk_dlg.close()

        # 4. VespagramDialog
        vesp_dlg = VespagramDialog(vis, config=win.config, parent=win)
        vesp_dlg.run_vespa()
        self._assert_dialog_quality("VespagramDialog", vesp_dlg)
        vesp_dlg.close()

        # 5. WorldMapDialog
        wmap_dlg = WorldMapDialog(win.engine, config=win.config, live_coordinator=win.live_coordinator, parent=win)
        self._assert_dialog_quality("WorldMapDialog", wmap_dlg)
        wmap_dlg.close()

        # 6. FilterDialog
        flt_dlg = FilterDialog(dt=0.05, config=win.config, parent=win)
        self._assert_dialog_quality("FilterDialog", flt_dlg)
        flt_dlg.close()

        # 7. SpectrumDialog
        spec_dlg = SpectrumDialog(None, parent=win)
        spec_dlg.add_curve(vis[0], 10.0, 40.0, "GRA1 BHZ")
        spec_dlg.method_combo.setCurrentIndex(4)
        self._assert_dialog_quality("SpectrumDialog", spec_dlg)
        spec_dlg.close()

        # 8. ParticleMotionDialog
        win.main_canvas.mag_trace_idx = 0
        win.main_canvas.mag_t_lo = 10.0
        win.main_canvas.mag_t_hi = 35.0
        win.action_open_pmotion_dialog()
        pm_open = [d for d in win._open_dialogs if isinstance(d, ParticleMotionDialog)]
        self.assertTrue(pm_open)
        pm_dlg = pm_open[-1]
        self._assert_dialog_quality("ParticleMotionDialog", pm_dlg)
        pm_dlg.close()

        # 9. NetworkLocationDialog
        net_dlg = NetworkLocationDialog(picks_list, initial_depth=33.0, config=win.config, parent=win)
        self._assert_dialog_quality("NetworkLocationDialog", net_dlg)
        net_dlg.close()

        # 10. EventSummaryDialog
        evsum_dlg = EventSummaryDialog(AnalysisParamSet(), picks_list, config=win.config, parent=win)
        self._assert_dialog_quality("EventSummaryDialog", evsum_dlg)
        evsum_dlg.close()

        # 11. LiveEventDialog
        live_dlg = LiveEventDialog(win.live_coordinator, parent=win)
        self._assert_dialog_quality("LiveEventDialog", live_dlg)
        live_dlg.close()

    def test_dark_colorbar_styles_in_fk_vespagram_and_spectrum(self) -> None:
        win = self.win
        vis = win.engine.visible_traces()

        # FKAnalysisDialog
        fk_dlg = FKAnalysisDialog(vis, t_lo=10.0, t_hi=40.0, config=win.config, parent=win)
        fk_dlg.run_fk()
        self.assertGreaterEqual(len(fk_dlg.fig.axes), 2)
        cbar_ax_fk = fk_dlg.fig.axes[-1]
        self.assertEqual(cbar_ax_fk.yaxis.label.get_color(), "#e6edf3")
        self.assertEqual(cbar_ax_fk.yaxis.get_tick_params().get("color"), "#8b949e")
        self.assertEqual(cbar_ax_fk.yaxis.get_tick_params().get("labelcolor"), "#8b949e")
        fk_dlg.close()

        # VespagramDialog
        vesp_dlg = VespagramDialog(vis, config=win.config, parent=win)
        vesp_dlg.run_vespa()
        self.assertGreaterEqual(len(vesp_dlg.fig.axes), 2)
        cbar_ax_vesp = vesp_dlg.fig.axes[-1]
        self.assertEqual(cbar_ax_vesp.yaxis.label.get_color(), "#e6edf3")
        self.assertEqual(cbar_ax_vesp.yaxis.get_tick_params().get("color"), "#8b949e")
        self.assertEqual(cbar_ax_vesp.yaxis.get_tick_params().get("labelcolor"), "#8b949e")
        vesp_dlg.close()

        # SpectrumDialog (method 4: Spectrogram)
        spec_dlg = SpectrumDialog(None, parent=win)
        spec_dlg.add_curve(vis[0], 10.0, 40.0, "GRA1 BHZ")
        spec_dlg.method_combo.setCurrentIndex(4)
        self.assertGreaterEqual(len(spec_dlg.fig.axes), 2)
        cbar_ax_spec = spec_dlg.fig.axes[-1]
        self.assertEqual(cbar_ax_spec.yaxis.label.get_color(), "#e6edf3")
        self.assertEqual(cbar_ax_spec.yaxis.get_tick_params().get("color"), "#8b949e")
        self.assertEqual(cbar_ax_spec.yaxis.get_tick_params().get("labelcolor"), "#8b949e")
        spec_dlg.close()


class TestNetworkLocationDialogQuality(UsabilityBase):
    """Regressions for NetworkLocationDialog NoEditTriggers, column headers, and column widths."""

    def test_network_location_dialog_properties(self) -> None:
        win = self.win
        vis = win.engine.visible_traces()
        vis[0].add_or_replace_pick(PhasePick(name="P", onset="02-NOV-2003_02:25:30.123", station=vis[0].station, comp=vis[0].comp))
        picks_list = [p for t in vis for p in t.picks]

        net_dlg = NetworkLocationDialog(picks_list, initial_depth=33.0, config=win.config, parent=win)
        net_dlg.show()
        self.app.processEvents()

        try:
            self.assertEqual(net_dlg.table.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
            headers = [net_dlg.table.horizontalHeaderItem(c).text() for c in range(net_dlg.table.columnCount())]
            self.assertEqual(headers, ["Station", "Comp", "Phase", "Onset Time", "Weight", "Dist (deg)", "Residual (s)"])

            onset_col_w = net_dlg.table.columnWidth(3)
            onset_text_w = net_dlg.table.fontMetrics().horizontalAdvance("02-NOV-2003_02:25:30.123")
            self.assertGreaterEqual(onset_col_w, onset_text_w)
        finally:
            net_dlg.close()


class TestAmpersandEscaping(UsabilityBase):
    """Regressions verifying absence of unescaped single '&' mnemonics in dialog and panel widgets."""

    def test_live_widgets_ampersand_escaping(self) -> None:
        win = self.win
        vis = win.engine.visible_traces()

        flt_dlg = FilterDialog(dt=0.05, config=win.config, parent=win)
        read_dlg = ReadDataDialog(config=win.config, parent=win)
        net_dlg = NetworkLocationDialog([], config=win.config, parent=win)
        fk_dlg = FKAnalysisDialog(vis, config=win.config, parent=win)
        evsum_dlg = EventSummaryDialog(AnalysisParamSet(), [], config=win.config, parent=win)

        pm_dlg = ParticleMotionDialog(vis[0], vis[1], vis[2], parent=win)

        # Check key buttons & groupboxes have &&
        pm_btns = [b.text() for b in pm_dlg.findChildren(QPushButton) if "Adopt" in b.text()]
        self.assertIn("Adopt Azimuth && Incidence", pm_btns)

        fk_btns = [b.text() for b in fk_dlg.findChildren(QPushButton) if "Adopt" in b.text()]
        self.assertIn("Adopt Azimuth && Slowness to Parameter Box", fk_btns)

        read_grps = [g.title() for g in read_dlg.findChildren(QGroupBox)]
        self.assertIn("Station Selection Matrix (GRF Array && GRSN Regional Network)", read_grps)
        self.assertIn("Stream && Time Window Parameters", read_grps)

        param_grps = [g.title() for g in win.param_panel.findChildren(QGroupBox)]
        self.assertIn("Display Scaling && Time", param_grps)

        from PyQt6.QtWidgets import QTabWidget
        tabs_w = evsum_dlg.findChild(QTabWidget)
        self.assertIsNotNone(tabs_w)
        self.assertEqual(tabs_w.tabText(1), "Epicenter && Station Map (epimap)")

        # Verify no unescaped single ampersand in any widget of these dialogs/panels
        unescaped_pattern = re.compile(r"(?<!&)&(?!&)")
        for dlg_or_panel in (flt_dlg, read_dlg, net_dlg, fk_dlg, evsum_dlg, pm_dlg, win.param_panel):
            for btn in dlg_or_panel.findChildren(QPushButton):
                self.assertIsNone(unescaped_pattern.search(btn.text()), f"Unescaped & in QPushButton: {btn.text()}")
            for grp in dlg_or_panel.findChildren(QGroupBox):
                self.assertIsNone(unescaped_pattern.search(grp.title()), f"Unescaped & in QGroupBox: {grp.title()}")
            for tab_bar in dlg_or_panel.findChildren(QTabBar):
                for i in range(tab_bar.count()):
                    self.assertIsNone(unescaped_pattern.search(tab_bar.tabText(i)), f"Unescaped & in tab: {tab_bar.tabText(i)}")

        for d in (flt_dlg, read_dlg, net_dlg, fk_dlg, evsum_dlg, pm_dlg):
            d.close()

    def test_static_source_files_no_unescaped_ampersand_in_widgets(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = glob.glob(os.path.join(repo_root, "shm_app", "gui", "dialogs*.py")) + [
            os.path.join(repo_root, "shm_app", "gui", "panels.py")
        ]
        unescaped_pattern = re.compile(r"(?<!&)&(?!&)")

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if any(kw in line for kw in ("QPushButton(", "QLabel(", "QGroupBox(", ".addTab(")):
                        for m in re.finditer(r"\"([^\"]*)\"|\x27([^\x27]*)\x27", line):
                            s = m.group(1) or m.group(2) or ""
                            self.assertIsNone(
                                unescaped_pattern.search(s),
                                f"Unescaped ampersand in {fpath}:{line_no} -> {s}",
                            )


class TestWorkflowAndSignalSynchronizations(UsabilityBase):
    """Regressions for ParameterBoxPanel, FilterDialog, ReadDataDialog, FK, ParticleMotion, canvas middle-click, and unfilter backup."""

    def test_parameter_box_evtype_normalization_and_sync(self) -> None:
        win = self.win
        self.assertEqual(normalize_shm_event_type("teleseismic", tier="tele"), "t")
        self.assertEqual(normalize_shm_event_type("Quarry Blast"), "b")
        self.assertEqual(normalize_shm_event_type("local_eq", tier="local"), "l")
        self.assertEqual(normalize_shm_event_type("regional event", tier="regional"), "r")
        self.assertEqual(normalize_shm_event_type("nuclear"), "n")
        self.assertEqual(normalize_shm_event_type("rockburst"), "m")

        ev = LiveEvent(
            event_id="TEST_EV_PSET",
            origin_epoch=1700000000.0,
            origin_iso="2023-11-14T22:13:20Z",
            lat=38.0,
            lon=20.0,
            depth_km=10.0,
            magnitude=5.8,
            mag_type="Mw",
            region="GREECE",
            source_feed="emsc_ws",
            event_type="quarry blast",
        )
        eva = evaluate_event_relevance(ev, win.live_coordinator.live_config, win.config)
        entry = LiveEventEntry(event=ev, evaluation=eva, fetched_traces=[win.engine.traces[0].copy()])
        pset_live = build_analysis_param_set(entry, win.config, win.live_coordinator.live_config)
        self.assertEqual(pset_live.event_type, "b")

        win.engine.param_sets[win.engine.active_set_index] = pset_live
        win.param_panel.refresh_from_model()
        self.assertEqual(win.param_panel.evtype_combo.currentData(), "b")
        self.assertIn("Quarry Blast", win.param_panel.evtype_combo.currentText())

    def test_parameter_box_treset_norm_stw_and_zoom_sync(self) -> None:
        win = self.win
        win.engine.execute_line("NORM AW")
        win.refresh_views()
        self.assertEqual(win.main_canvas.norm_mode, "AW")
        self.assertTrue(win.param_panel.norm_combo.currentText().startswith("AW "))

        win.engine.execute_line("STW 12 42")
        win.refresh_views()
        self.assertEqual((win.main_canvas.view_t_lo, win.main_canvas.view_t_hi), (12.0, 42.0))

        win.engine.execute_line("ZOOM 4.0")
        win.refresh_views()
        self.assertAlmostEqual(win.main_canvas.global_zoom, 4.0)

        btn_amp_plus = next(b for b in win.param_panel.findChildren(QPushButton) if b.text() == "Amp +")
        btn_amp_plus.click()
        self.assertAlmostEqual(win.engine.global_zoom, 6.0)
        self.assertAlmostEqual(win.main_canvas.global_zoom, 6.0)

        win.engine.execute_line("TRESET")
        win.refresh_views()
        self.assertEqual(win.main_canvas.norm_mode, "C")
        self.assertTrue(win.param_panel.norm_combo.currentText().startswith("C "))
        self.assertEqual((win.main_canvas.view_t_lo, win.main_canvas.view_t_hi), win.main_canvas.full_time_span())
        self.assertEqual(win.h_zoom_lbl.text(), "Full")
        self.assertAlmostEqual(win.main_canvas.global_zoom, 1.0)

    def test_filter_dialog_lp_hp_period_suffix_and_search(self) -> None:
        win = self.win
        flt_dlg = FilterDialog(dt=0.05, config=win.config, parent=win)
        flt_dlg.show()
        self.app.processEvents()

        try:
            flt_dlg.but_type.setCurrentText("LP (Lowpass)")
            self.assertFalse(flt_dlg.but_f2.isEnabled())

            flt_dlg.but_type.setCurrentText("HP (Highpass)")
            self.assertFalse(flt_dlg.but_f2.isEnabled())

            flt_dlg.but_type.setCurrentText("BP (Bandpass)")
            self.assertTrue(flt_dlg.but_f2.isEnabled())

            flt_dlg.but_type.setCurrentText("BS (Bandstop)")
            self.assertTrue(flt_dlg.but_f2.isEnabled())

            flt_dlg.but_f1.setValue(2.0)
            self.assertIn("0.50", flt_dlg.but_f1.suffix())

            flt_dlg.tabs.setCurrentIndex(2)
            flt_dlg.file_search.setText("WWSSN_SP")
            vis_rows = [i for i in range(flt_dlg.file_list.count()) if not flt_dlg.file_list.item(i).isHidden()]
            self.assertGreaterEqual(len(vis_rows), 1)
            self.assertIn("WWSSN_SP", flt_dlg.file_list.currentItem().text().upper())

            self.assertIn(".", flt_dlg.but_f1.textFromValue(1.5))
        finally:
            flt_dlg.close()

    def test_read_data_dialog_channels_components_and_empty_edge_case(self) -> None:
        win = self.win
        read_dlg = ReadDataDialog(config=win.config, parent=win)
        read_dlg.show()
        self.app.processEvents()

        try:
            mseed_dir = os.path.join(win.config.data_examples_dir, "mseed")
            read_dlg.path_edit.setText(mseed_dir)
            read_dlg.chk_z.setChecked(True)
            read_dlg.chk_n.setChecked(False)
            read_dlg.chk_e.setChecked(False)
            read_dlg.chan_combo.setCurrentText("BH")
            read_dlg._execute_read()

            self.assertGreater(len(read_dlg.loaded_traces), 0)
            self.assertTrue(all(tr.comp == "Z" and tr.channel3.startswith("BH") for tr in read_dlg.loaded_traces))

            # Edge case: no component selected shows warning and does not read
            read_dlg.chk_z.setChecked(False)
            read_dlg.chk_n.setChecked(False)
            read_dlg.chk_e.setChecked(False)
            with patch.object(QMessageBox, "warning") as mock_warn:
                read_dlg.loaded_traces.clear()
                read_dlg._execute_read()
                mock_warn.assert_called_once()
                self.assertEqual(len(read_dlg.loaded_traces), 0)
        finally:
            read_dlg.close()

    def test_fk_analysis_dialog_adopt_vs_locate(self) -> None:
        win = self.win
        vis = win.engine.visible_traces()
        fk_dlg = FKAnalysisDialog(vis, t_lo=10.0, t_hi=40.0, config=win.config, parent=win)
        fk_dlg.run_fk()

        old_lat, old_lon = win.engine.current_params.lat, win.engine.current_params.lon
        adopted_only = []
        adopted_loc = []
        fk_dlg.adoptParameters.connect(lambda a, s: adopted_only.append((a, s)))
        fk_dlg.adoptAndLocate.connect(lambda a, s: adopted_loc.append((a, s)))

        # 1. Unchecked: adoptParameters only
        fk_dlg.chk_auto_locate.setChecked(False)
        fk_dlg._on_adopt()
        self.assertEqual(len(adopted_only), 1)
        self.assertEqual(len(adopted_loc), 0)

        win._on_adopt_az_slo(123.4, 7.8)
        self.assertAlmostEqual(win.engine.current_params.azimuth, 123.4)
        self.assertAlmostEqual(win.engine.current_params.slowness, 7.8)
        self.assertEqual((win.engine.current_params.lat, win.engine.current_params.lon), (old_lat, old_lon))

        # 2. Checked: adoptAndLocate
        adopted_only.clear()
        adopted_loc.clear()
        fk_dlg.chk_auto_locate.setChecked(True)
        fk_dlg._on_adopt()
        self.assertEqual(len(adopted_only), 0)
        self.assertEqual(len(adopted_loc), 1)
        fk_dlg.close()

    def test_particle_motion_window_incidence_and_rot3(self) -> None:
        win = self.win
        vis = win.engine.visible_traces()
        win.main_canvas.mag_trace_idx = 0
        win.main_canvas.mag_t_lo = 10.0
        win.main_canvas.mag_t_hi = 35.0
        win.action_open_pmotion_dialog()
        self.app.processEvents()

        pm_open = [d for d in win._open_dialogs if isinstance(d, ParticleMotionDialog)]
        self.assertTrue(pm_open)
        pm_dlg = pm_open[-1]
        self.assertIsNotNone(pm_dlg.t_lo)
        self.assertIsNotNone(pm_dlg.t_hi)
        self.assertAlmostEqual(pm_dlg.t_lo, 10.0, delta=0.1)
        self.assertAlmostEqual(pm_dlg.t_hi, 35.0, delta=0.1)

        pm_dlg.az_spin.setValue(85.0)
        pm_dlg.inc_spin.setValue(32.5)
        win._on_adopt_pmotion_angles(85.0, 32.5)
        self.assertAlmostEqual(win.engine.current_params.azimuth, 85.0)
        self.assertAlmostEqual(win.engine.current_params.incidence, 32.5)
        self.assertAlmostEqual(vis[0].inci, 32.5)

        n_before = len(win.engine.traces)
        win.action_rotate_3d()
        self.assertEqual(len(win.engine.traces), n_before + 3)
        pm_dlg.close()

    def test_canvas_middle_click_trace_invert_undo_redo_and_magnifier_sync(self) -> None:
        win = self.win
        cv = win.main_canvas
        orig_val = win.engine.traces[0].data[0]

        # Middle click in left margin (< cv.MARGIN_LEFT) of slot 0
        slot_y = cv.slot_top(0, cv.slot_height()) + 10
        pt = QPointF(10.0, float(slot_y))
        global_pt = cv.mapToGlobal(pt.toPoint()).toPointF()
        ev_mid = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            pt,
            global_pt,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cv.mousePressEvent(ev_mid)
        self.app.processEvents()

        # Inverted sign
        self.assertAlmostEqual(win.engine.traces[0].data[0], -orig_val)
        self.assertIn("Invert sign", [u[0] for u in win._undo_stack])

        # Undo restores original
        win.action_undo()
        self.app.processEvents()
        self.assertAlmostEqual(win.engine.traces[0].data[0], orig_val)

        # Redo inverts again
        win.action_redo()
        self.app.processEvents()
        self.assertAlmostEqual(win.engine.traces[0].data[0], -orig_val)

        # Magnifier lock mode sync
        r_mag = win.mag_canvas.plot_rect()
        mid_pt = r_mag.center()
        global_mid_pt = win.mag_canvas.mapToGlobal(mid_pt.toPoint()).toPointF()
        ev_mag = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            mid_pt,
            global_mid_pt,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
        )
        win.mag_canvas.mousePressEvent(ev_mag)
        self.assertEqual(win.mag_canvas.cursor_lock_mode, "PEAK_LOCK")
        self.assertEqual(win.main_canvas.cursor_lock_mode, "PEAK_LOCK")

        win.mag_canvas.mousePressEvent(ev_mag)
        self.assertEqual(win.mag_canvas.cursor_lock_mode, "ZERO_LOCK")
        self.assertEqual(win.main_canvas.cursor_lock_mode, "ZERO_LOCK")

        win.mag_canvas.mousePressEvent(ev_mag)
        self.assertEqual(win.mag_canvas.cursor_lock_mode, "CROSSHAIR")
        self.assertEqual(win.main_canvas.cursor_lock_mode, "CROSSHAIR")

    def test_unfiltered_backup_stability_across_sort_filter_beam_unfilter(self) -> None:
        win = self.win
        first_raw = win.engine.traces[0].data.copy()
        first_sta = win.engine.traces[0].station
        first_ch = win.engine.traces[0].channel3

        win.action_sort_by_station()
        win.engine.apply_filter_to_traces("BP 0.7 2.0 3")
        win.action_form_beam()
        win.engine.traces[0].selected = True
        win.action_unfilter()

        matched = [t for t in win.engine.traces if t.station == first_sta and t.channel3 == first_ch]
        self.assertTrue(matched)
        self.assertTrue((matched[0].data == first_raw).all())
        self.assertTrue(all(t.filter_name == "NONE" for t in win.engine.traces))

    def test_live_event_dialog_open_cache_dir(self) -> None:
        win = self.win
        live_dlg = LiveEventDialog(win.live_coordinator, parent=win)
        live_dlg.show()
        self.app.processEvents()

        try:
            with tempfile.TemporaryDirectory() as tmpd:
                live_dlg.edit_cache_dir.setText(tmpd)
                with patch.object(QDesktopServices, "openUrl", return_value=True) as mock_open:
                    live_dlg._on_open_cache_dir()
                    mock_open.assert_called_once_with(QUrl.fromLocalFile(os.path.abspath(tmpd)))
        finally:
            live_dlg.close()


if __name__ == "__main__":
    unittest.main()

