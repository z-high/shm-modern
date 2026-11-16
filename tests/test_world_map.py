"""Tests for Seismic World Map and extended Tutorial Mode tooltip coverage."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QGroupBox, QPushButton, QToolBar, QToolButton

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shm_app.gui.dialogs import EventSummaryDialog
from shm_app.gui.help_texts import HELP_TEXTS
from shm_app.gui.main_window import SHMMainWindow
from shm_app.gui.world_map import (
    WorldMapDialog,
    distance_ring_segments,
    great_circle_segments,
    load_world_countries,
)
from shm_app.plugins.bgr_dialog import BGRFetchDialog
from shm_app.plugins.live_dialog import LiveEventDialog
from shm_app.plugins.live_events import LiveEvent
from shm_app.plugins.live_planner import evaluate_event_relevance, plan_event_fetch_window
from shm_app.plugins.live_service import LiveEventEntry, STATE_READY, STATE_WAITING


def _app() -> QApplication:
    inst = QApplication.instance()
    return inst if inst is not None else QApplication([])


class TestWorldMapGeometry(unittest.TestCase):
    def test_bundled_countries_asset_loaded(self) -> None:
        countries = load_world_countries()
        self.assertGreater(len(countries), 200)
        names = {c["n"] for c in countries}
        self.assertIn("Germany", names)
        self.assertIn("Japan", names)
        self.assertIn("Russia", names)

    def test_great_circle_and_rings_split_antimeridian(self) -> None:
        # Path crossing date line: Kamchatka (+160E) to Aleutians/Hawaii (-155W)
        segs = great_circle_segments(55.0, 162.0, 20.0, -155.0, n_pts=80)
        self.assertGreaterEqual(len(segs), 1)
        for seg in segs:
            dlons = abs(seg[1:, 0] - seg[:-1, 0])
            self.assertTrue((dlons <= 180.0).all(), "Great-circle segment crossed +-180 without splitting")

        # Concentric distance ring at Fiji (-18, 178) crosses +-180 meridian
        rsegs = distance_ring_segments(-18.0, 178.0, 30.0)
        self.assertGreaterEqual(len(rsegs), 2)
        for seg in rsegs:
            dlons = abs(seg[1:, 0] - seg[:-1, 0])
            self.assertTrue((dlons <= 180.0).all(), "Distance ring crossed +-180 without splitting")


class TestWorldMapAndTutorialUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self.win = SHMMainWindow()
        self.win.resize(1280, 820)
        self.win.load_example_dataset("GRF_031102_0225.GSE")
        self.app.processEvents()

    def tearDown(self) -> None:
        for d in list(self.win._open_dialogs):
            d.close()
        self.win.close()
        self.app.processEvents()

    def test_tutorial_mode_covers_docks_sliders_console_and_dialogs(self) -> None:
        # Initially off
        self.assertFalse(self.win.tutorial_mode)
        self.assertNotIn("<table", self.win.row_slider.toolTip())
        self.assertNotIn("<table", self.win.h_zoom_slider.toolTip())
        self.assertNotIn("<table", self.win.h_scroll.toolTip())
        self.assertNotIn("<table", self.win.cmd_input.toolTip())
        self.assertNotIn("<table", self.win.phase_panel.phase_edit.toolTip())
        self.assertNotIn("<table", self.win.param_panel.dist_spin.toolTip())
        self.assertIn("dock_phase", HELP_TEXTS)
        self.assertIn("dock_param", HELP_TEXTS)
        self.assertIn("world_map", HELP_TEXTS)

        # Turn Tutorial Mode ON -> docks, sliders, command box, actions all get rich HTML cards
        self.win.act_tutorial.setChecked(True)
        QApplication.processEvents()

        rich_widgets = [
            self.win.row_slider,
            self.win.h_zoom_slider,
            self.win.h_scroll,
            self.win.cmd_input,
            self.win.phase_panel.phase_edit,
            self.win.phase_panel.qual_spin,
            self.win.phase_panel.btn_del_pick,
            self.win.phase_panel.chk_depth_rays,
            self.win.param_panel.set_combo,
            self.win.param_panel.dist_spin,
            self.win.param_panel.azim_spin,
            self.win.param_panel.slow_spin,
            self.win.param_panel.depth_spin,
            self.win.param_panel.origin_edit,
            self.win.param_panel.norm_combo,
        ]
        for w in rich_widgets:
            self.assertIn("<table", w.toolTip(), f"Missing rich tutorial tooltip on {w}")

        # Dock toggle actions in View menu also get rich tooltips
        self.assertIn("<table", self.win.phase_dock.toggleViewAction().toolTip())
        self.assertIn("<table", self.win.param_dock.toggleViewAction().toolTip())

        # Non-modal dialogs opened while Tutorial Mode is ON get rich tooltips automatically
        self.win.action_open_world_map()
        QApplication.processEvents()
        wmap = self.win._find_open_dialog(WorldMapDialog)
        self.assertIsNotNone(wmap)
        self.assertIn("<table", wmap.chk_paths.toolTip())

        # Turning Tutorial Mode OFF restores plain tooltips everywhere including open WorldMapDialog
        self.win.act_tutorial.setChecked(False)
        QApplication.processEvents()
        for w in rich_widgets:
            self.assertNotIn("<table", w.toolTip())
        self.assertNotIn("<table", wmap.chk_paths.toolTip())

    def test_world_map_stations_events_search_and_actions(self) -> None:
        # Inject active Parameter Box event + sample Live Events
        pset = self.win.engine.param_sets[self.win.engine.active_set_index]
        pset.lat = 38.3
        pset.lon = 142.4
        pset.depth = 32.0
        pset.distance = 82.1
        pset.azimuth = 33.4
        pset.mag_mw = 6.9
        pset.region_name = "Near East Coast of Honshu, Japan"

        ev_ready = LiveEvent(
            event_id="EMSC-TONGA-TEST",
            origin_epoch=1700000000.0,
            origin_iso="2023-11-14T22:13:20Z",
            lat=-20.2,
            lon=-174.1,
            depth_km=120.0,
            magnitude=6.4,
            mag_type="Mw",
            region="Tonga Islands",
            source_feed="emsc_ws",
        )
        ev_wait = LiveEvent(
            event_id="GFZ-CRETE-TEST",
            origin_epoch=1700000500.0,
            origin_iso="2023-11-14T22:21:40Z",
            lat=35.1,
            lon=25.2,
            depth_km=15.0,
            magnitude=5.1,
            mag_type="Mw",
            region="Crete, Greece",
            source_feed="gfz_fdsn",
        )
        lcfg = self.win.live_coordinator.live_config
        ev_ready_eval = evaluate_event_relevance(ev_ready, lcfg, self.win.config)
        ev_ready_plan = plan_event_fetch_window(ev_ready, ev_ready_eval, lcfg, self.win.config, allow_network=False)
        ev_wait_eval = evaluate_event_relevance(ev_wait, lcfg, self.win.config)

        self.win.live_coordinator.entries["EMSC-TONGA-TEST"] = LiveEventEntry(
            event=ev_ready,
            evaluation=ev_ready_eval,
            plan=ev_ready_plan,
            state=STATE_READY,
        )
        self.win.live_coordinator.entries["GFZ-CRETE-TEST"] = LiveEventEntry(
            event=ev_wait,
            evaluation=ev_wait_eval,
            state=STATE_WAITING,
        )

        self.win.action_open_world_map()
        QApplication.processEvents()
        dlg = self.win._find_open_dialog(WorldMapDialog)
        self.assertIsNotNone(dlg)

        # Verify database stations + loaded traces + active event + live events discovered
        self.assertGreater(len(dlg._station_items), 1000)
        loaded_codes = {s.code for s in dlg._station_items if s.category == "loaded"}
        self.assertIn("GRA1", loaded_codes)
        self.assertTrue(any(e.is_active_param for e in dlg._event_items))
        self.assertEqual(sum(1 for e in dlg._event_items if e.is_live), 2)

        # Preset view checks
        dlg.set_view_preset("europe")
        self.assertAlmostEqual(dlg._xlim[0], -12.0, delta=0.1)
        dlg.set_view_preset("germany")
        self.assertAlmostEqual(dlg._ylim[1], 55.5, delta=0.1)
        dlg.set_view_preset("grf")
        self.assertLess(dlg._xlim[1] - dlg._xlim[0], 3.0)

        # Search station -> selects loaded station & highlights workspace traces on click
        self.assertTrue(dlg.focus_search("GRA1"))
        self.assertTrue(dlg.btn_select_traces.isEnabled())
        self.assertTrue(dlg.btn_open_bgr.isEnabled())
        dlg.btn_select_traces.click()
        QApplication.processEvents()
        self.assertTrue(any(tr.selected and tr.station.upper() == "GRA1" for tr in self.win.engine.traces))

        # Search Live READY event -> can adopt into Parameter Box
        self.assertTrue(dlg.focus_search("Crete"))
        self.assertTrue(dlg.btn_adopt_event.isEnabled())
        dlg.btn_adopt_event.click()
        QApplication.processEvents()
        new_pset = self.win.engine.param_sets[self.win.engine.active_set_index]
        self.assertAlmostEqual(new_pset.lat, 35.1, delta=0.01)
        self.assertAlmostEqual(new_pset.lon, 25.2, delta=0.01)
        self.assertEqual(new_pset.region_name, "Crete, Greece")

    def test_ui_polish_interconnections_and_no_toolbar_overflow(self) -> None:
        # 1. BGR duration carry bug fix (no "4m 60s") & Workspace preset
        self.assertEqual(BGRFetchDialog._format_duration_label(299.95), "(5 min)")
        self.assertEqual(BGRFetchDialog._format_duration_label(300.0), "(5 min)")
        self.assertEqual(BGRFetchDialog._format_duration_label(3599.9), "(1 h)")

        self.win.resize(1280, 820)
        self.win.show()
        QApplication.processEvents()

        # 2. Toolbar must fit at 1280px width without hiding actions behind extension button (>>)
        tbs = self.win.findChildren(QToolBar)
        self.assertTrue(len(tbs) >= 1)
        tb = tbs[0]
        ext_btn = tb.findChild(QToolButton, "qt_toolbar_ext_button")
        if ext_btn is not None:
            self.assertFalse(ext_btn.isVisible())
        self.assertTrue(self.win.row_slider.isVisible())

        # 3. ParameterBoxPanel Display Scaling && Time label + lat/lon -> dist/BAZ/region live sync
        grp_titles = [g.title() for g in self.win.param_panel.findChildren(QGroupBox)]
        self.assertIn("Display Scaling && Time", grp_titles)
        self.win.param_panel.lat_spin.setValue(35.1)
        self.win.param_panel.lon_spin.setValue(25.2)
        QApplication.processEvents()
        p = self.win.engine.current_params
        self.assertGreater(p.distance, 12.0)
        self.assertNotEqual(p.region_name, "")

        # 4. BGRFetchDialog workspace preset & collapsible log box
        self.win.action_open_bgr_fetch_dialog()
        QApplication.processEvents()
        bgr = self.win._find_open_dialog(BGRFetchDialog)
        self.assertIsNotNone(bgr)
        from PyQt6.QtWidgets import QPushButton
        ws_btns = [b for b in bgr.findChildren(QPushButton) if b.text().startswith("Workspace (")]
        self.assertEqual(len(ws_btns), 1)
        self.assertFalse(bgr.log_box.isVisible())
        ws_btns[0].click()
        QApplication.processEvents()
        self.assertIn("GRA1", bgr.selected_stations())

        # 5. LiveEventDialog log tab + Show on Map -> WorldMap focus -> Open Live Monitor
        ev = LiveEvent(
            event_id="live_interconnect_1",
            origin_epoch=1700000100.0,
            origin_iso="2023-11-14T22:15:00Z",
            lat=38.2,
            lon=142.3,
            depth_km=35.0,
            magnitude=6.2,
            mag_type="Mw",
            region="Near East Coast of Honshu, Japan",
            source_feed="USGS",
        )
        lcfg = self.win.live_coordinator.live_config
        ev_eval = evaluate_event_relevance(ev, lcfg, self.win.config)
        self.win.live_coordinator.entries[ev.event_id] = LiveEventEntry(
            event=ev,
            evaluation=ev_eval,
            plan=plan_event_fetch_window(ev, ev_eval, lcfg, self.win.config, allow_network=False),
            state=STATE_READY,
        )
        self.win.live_coordinator.ordered_ids.append(ev.event_id)
        self.win.action_open_live_event_dialog()
        QApplication.processEvents()
        live_dlg = self.win._find_open_dialog(LiveEventDialog)
        self.assertIsNotNone(live_dlg)
        self.assertEqual(live_dlg.tabs.count(), 3)
        live_dlg.events_table.selectRow(0)
        QApplication.processEvents()
        self.assertTrue(live_dlg.btn_show_map.isEnabled())
        live_dlg.btn_show_map.click()
        QApplication.processEvents()

        wmap = self.win._find_open_dialog(WorldMapDialog)
        self.assertIsNotNone(wmap)
        self.assertIsNotNone(wmap._selected_item)
        self.assertEqual(wmap._selected_item.get("id"), ev.event_id)

        # 6. EventSummaryDialog basemap + Open in World Map interconnection
        all_picks = [pk for tr in self.win.engine.traces for pk in tr.picks]
        ev_dlg = EventSummaryDialog(self.win.engine.current_params, all_picks, config=self.win.config, parent=self.win)
        self.win._show_nonmodal(ev_dlg)
        ev_dlg.openWorldMapRequested.connect(self.win._on_event_summary_open_world_map)
        ev_dlg.openWorldMapRequested.emit()
        QApplication.processEvents()



class TestWorldMapDialogQualityAndRegressions(unittest.TestCase):
    """Regressions for WorldMapDialog sizing, autoDefault, C-locale, and presets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def setUp(self) -> None:
        self.win = SHMMainWindow()
        self.win.resize(1280, 820)
        self.win.load_example_dataset("GRF_031102_0225.GSE")
        self.app.processEvents()

    def tearDown(self) -> None:
        for d in list(self.win._open_dialogs):
            d.close()
        self.win.close()
        self.app.processEvents()

    def test_world_map_dialog_quality_standards(self) -> None:
        wmap_dlg = WorldMapDialog(
            self.win.engine,
            config=self.win.config,
            live_coordinator=self.win.live_coordinator,
            parent=self.win,
        )
        wmap_dlg.show()
        self.app.processEvents()
        try:
            min_h = wmap_dlg.minimumSizeHint()
            sz = wmap_dlg.size()
            self.assertGreaterEqual(sz.width(), min_h.width())
            self.assertGreaterEqual(sz.height(), min_h.height())
            auto_btns = [b.text() for b in wmap_dlg.findChildren(QPushButton) if b.autoDefault()]
            self.assertEqual(auto_btns, [])
            spin_locales = [sb.locale().name() for sb in wmap_dlg.findChildren(QDoubleSpinBox)]
            self.assertTrue(all(loc in ("C", "en_US") or loc.startswith("C") for loc in spin_locales))
            for sb in wmap_dlg.findChildren(QDoubleSpinBox):
                self.assertIn(".", sb.textFromValue(1.5), f"Comma decimal leaked in {sb}")
        finally:
            wmap_dlg.close()

    def test_world_map_dialog_preset_zooms_and_path_toggles(self) -> None:
        wmap_dlg = WorldMapDialog(
            self.win.engine,
            config=self.win.config,
            live_coordinator=self.win.live_coordinator,
            parent=self.win,
        )
        wmap_dlg.show()
        self.app.processEvents()
        try:
            # Preset view bounds
            wmap_dlg.set_view_preset("world")
            self.assertAlmostEqual(wmap_dlg._xlim[0], -180.0, delta=1.0)
            self.assertAlmostEqual(wmap_dlg._xlim[1], 180.0, delta=1.0)

            wmap_dlg.set_view_preset("europe")
            self.assertAlmostEqual(wmap_dlg._xlim[0], -12.0, delta=0.5)

            wmap_dlg.set_view_preset("germany")
            self.assertAlmostEqual(wmap_dlg._ylim[1], 55.5, delta=0.5)

            wmap_dlg.set_view_preset("grf")
            self.assertLess(wmap_dlg._xlim[1] - wmap_dlg._xlim[0], 3.0)

            # Toggling layers triggers redraw without exception
            wmap_dlg.chk_paths.setChecked(False)
            wmap_dlg.chk_rings.setChecked(False)
            wmap_dlg.chk_quarries.setChecked(True)
            self.app.processEvents()
            wmap_dlg.canvas.draw()
        finally:
            wmap_dlg.close()


if __name__ == "__main__":
    unittest.main()

