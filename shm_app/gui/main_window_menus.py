"""Menu bar and main toolbar construction helpers for SHMMainWindow."""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import QMenu, QToolBar

from shm_app.gui.help_texts import status_summary


def _add_menu_action(
    win: Any,
    menu: QMenu,
    label: str,
    slot: Any,
    shortcut: Optional[str] = None,
    key: Optional[str] = None,
) -> QAction:
    act = QAction(label, win)
    if shortcut:
        act.setShortcut(QKeySequence(shortcut))
    # Do not forward the 'checked' argument of triggered() to the slot
    act.triggered.connect(lambda _checked=False, s=slot: s())
    if key:
        title = label.replace("&", "").rstrip(".")
        act.setStatusTip(status_summary(key))
        win._help_actions.append((act, title, key, shortcut or ""))
    menu.addAction(act)
    return act


def _add_toolbar_action(win: Any, tb: QToolBar, label: str, slot: Any, key: str, shortcut: str = "") -> QAction:
    act = QAction(label, win)
    act.triggered.connect(lambda _checked=False, s=slot: s())
    act.setStatusTip(status_summary(key))
    win._help_actions.append((act, label, key, shortcut))
    tb.addAction(act)
    return act


def build_menus_and_toolbar(win: Any) -> None:
    """Build task-oriented menu tree (File, Edit, Traces, View, Process, Analysis, Location, Tools, Help) and toolbar."""
    mb = win.menuBar()
    win._menus = []
    win._help_actions = []
    win.tutorial_mode = False

    def new_menu(parent: Any, title: str) -> QMenu:
        m = parent.addMenu(title)
        m.setToolTipsVisible(False)
        win._menus.append(m)
        return m

    def add(
        menu: QMenu,
        label: str,
        slot: Any,
        shortcut: Optional[str] = None,
        key: Optional[str] = None,
    ) -> QAction:
        return _add_menu_action(win, menu, label, slot, shortcut, key)

    # ---------------- File ----------------
    m_file = new_menu(mb, "&File")
    add(m_file, "&Open Waveforms...", win.action_open_read_dialog, "Ctrl+R", "open")
    add(m_file, "Download from &BGR (FDSN)...", win.action_open_bgr_fetch_dialog, None, "bgr_fetch")
    add(m_file, "Open &Event Parameters (.evt)...", win.action_read_evt, None, "open_evt")
    add(m_file, "Import &StationXML...", win.action_import_stationxml, None, "import_stationxml")
    add(m_file, "Import Quake&ML 1.2 (.xml)...", win.action_import_quakeml, None, "import_quakeml")
    add(m_file, "Restore Last Session", win.action_restore_session, None, "restore_session")
    m_file.addSeparator()
    add(m_file, "&Save Event Parameters (.evt)...", win.action_save_evt, "Ctrl+I", "save_evt")
    add(m_file, "Save Event and &Clear...", win.action_save_evt_and_clear, "Ctrl+J", "save_evt_clear")
    m_exp = new_menu(m_file, "E&xport Waveforms")
    add(m_exp, "GSE2 (.GSE)...", win.action_export_gse2, None, "export_gse")
    add(m_exp, "Mini-SEED (.mseed)...", win.action_export_mseed, None, "export_mseed")
    add(m_exp, "SH Q-File (.QHD/.QBN)...", win.action_export_qfile, None, "export_q")
    add(m_exp, "ASCII (.asc)...", win.action_export_ascii, None, "export_ascii")
    m_cat = new_menu(m_file, "Export &Catalog && Bulletins")
    add(m_cat, "QuakeML 1.2 (.xml)...", win.action_export_quakeml, None, "export_quakeml")
    add(m_cat, "GeoJSON RFC 7946 (.geojson)...", win.action_export_geojson, None, "export_geojson")
    add(m_cat, "Observatory Bulletin (PDF/SVG)...", win.action_export_bulletin, None, "export_bulletin")
    add(m_file, "Export Plot &Image...", win.action_export_plot, "Ctrl+H", "export_image")
    m_file.addSeparator()
    add(m_file, "&Quit", win.close, "Ctrl+Q", "quit")

    # ---------------- Edit ----------------
    m_edit = new_menu(mb, "&Edit")
    win.act_undo = add(m_edit, "&Undo", win.action_undo, "Ctrl+Z", "undo")
    win.act_redo = add(m_edit, "&Redo", win.action_redo, "Ctrl+Shift+Z", "redo")
    m_edit.addSeparator()
    add(m_edit, "Delete All Picks of Active &Phase", win.action_delete_active_pick, "Ctrl+Shift+Delete", "delete_phase")
    add(m_edit, "Clear All Picks and &Parameters", win.action_clear_params, "Ctrl+X", "clear_all")
    win._update_undo_actions()

    # ---------------- Traces ----------------
    m_trc = new_menu(mb, "&Traces")
    add(m_trc, "Select &All", win.action_select_all, "Ctrl+A", "select_all")
    add(m_trc, "&Deselect All", win.action_deselect_all, "Ctrl+Y", "deselect_all")
    m_trc.addSeparator()
    add(m_trc, "Delete &Selected", win.action_delete_selected, "Ctrl+D", "delete_selected")
    add(m_trc, "&Keep Only Selected", win.action_keep_selected, None, "keep_selected")
    add(m_trc, "Show All &Hidden Traces", win.action_show_hidden, None, "show_hidden")
    add(m_trc, "&Invert Sign of Selected", win.action_flip_traces, None, "flip")
    add(m_trc, "&Cut to Magnifier Window", win.action_cut_to_magnifier, None, "cut")
    m_sort = new_menu(m_trc, "S&ort Traces")
    add(m_sort, "By Epicentral Distance", win.action_sort_by_distance, None, "sort_distance")
    add(m_sort, "By Station Name", win.action_sort_by_station, None, "sort_station")

    # ---------------- View ----------------
    m_view = new_menu(mb, "&View")
    add(m_view, "&Zoom to Magnifier Window", win.action_zoom_to_magnifier, "Ctrl+W", "view_magnifier")
    add(m_view, "&Reset Full View", win.action_reset_windows, "Ctrl+E", "view_reset")
    add(m_view, "&Previous View", win.action_view_back, "Alt+Left", "view_back")
    add(m_view, "&Next View", win.action_view_forward, "Alt+Right", "view_forward")
    m_view.addSeparator()
    win.act_group = add(
        m_view, "&Group Components by Station (Z, N, E)", win.action_toggle_grouping, None, "group_components"
    )
    win.act_group.setCheckable(True)
    win.act_pick_error_bars = add(
        m_view, "Show Pick &Uncertainty Error Bars", lambda: None, None, "pick_error_bars"
    )
    win.act_pick_error_bars.setCheckable(True)
    win.act_pick_error_bars.setChecked(True)
    win.act_pick_error_bars.toggled.connect(win.action_toggle_pick_error_bars)

    m_mag_cur = new_menu(m_view, "&Magnifier Cursor Mode")
    win._mag_cursor_group = QActionGroup(win)
    win._mag_cursor_group.setExclusive(True)
    win.act_mag_cursor_standard = add(
        m_mag_cur,
        "Standard Crosshair",
        lambda: win.action_set_mag_cursor_mode("STANDARD"),
        None,
        "mag_cursor_standard",
    )
    win.act_mag_cursor_hilbert = add(
        m_mag_cur,
        "Hilbert 90° Phase Shift Cursor",
        lambda: win.action_set_mag_cursor_mode("HILBERT"),
        None,
        "mag_cursor_hilbert",
    )
    win.act_mag_cursor_reference = add(
        m_mag_cur,
        "Reference Waveform Cursor",
        lambda: win.action_set_mag_cursor_mode("REFERENCE"),
        None,
        "mag_cursor_reference",
    )
    for act_cur in (
        win.act_mag_cursor_standard,
        win.act_mag_cursor_hilbert,
        win.act_mag_cursor_reference,
    ):
        act_cur.setCheckable(True)
        win._mag_cursor_group.addAction(act_cur)
    win.act_mag_cursor_standard.setChecked(True)
    m_mag_cur.addSeparator()
    add(m_mag_cur, "Capture Reference Window", win.action_capture_reference_window, None, "mag_capture_ref")

    m_rows = new_menu(m_view, "Trace &Row Height")
    win._row_actions = {}
    for label, px, key in (
        ("Fit All Traces in Window", 0, "rows_fit"),
        ("Compact (24 px, scrolls)", 24, "rows_compact"),
        ("Normal (40 px, scrolls)", 40, "rows_normal"),
        ("Large (64 px, scrolls)", 64, "rows_large"),
    ):
        a = add(m_rows, label, lambda px=px: win._set_row_height(px), None, key)
        a.setCheckable(True)
        a.setChecked(px == 0)
        win._row_actions[px] = a
    m_rows.addSeparator()
    hint = m_rows.addAction("Fine adjustment: slider in the toolbar")
    hint.setEnabled(False)
    m_view.addSeparator()
    act_phase_dock = win.phase_dock.toggleViewAction()
    act_phase_dock.setText("Phase Picking Box")
    act_phase_dock.setStatusTip(status_summary("dock_phase"))
    m_view.addAction(act_phase_dock)
    win._help_actions.append((act_phase_dock, "Phase Picking Box", "dock_phase", ""))

    act_param_dock = win.param_dock.toggleViewAction()
    act_param_dock.setText("Analysis Parameter Box")
    act_param_dock.setStatusTip(status_summary("dock_param"))
    m_view.addAction(act_param_dock)
    win._help_actions.append((act_param_dock, "Analysis Parameter Box", "dock_param", ""))

    if hasattr(win, "fit_dock"):
        act_fit_dock = win.fit_dock.toggleViewAction()
        act_fit_dock.setText("Fit Diagnostics Window")
        act_fit_dock.setStatusTip(status_summary("dock_fit"))
        m_view.addAction(act_fit_dock)
        win._help_actions.append((act_fit_dock, "Fit Diagnostics Window", "dock_fit", ""))

    # ---------------- Process ----------------
    m_proc = new_menu(mb, "&Process")
    m_flt = new_menu(m_proc, "&Filter")
    add(m_flt, "Filter Dialog...", win.action_open_filter_dialog, "Ctrl+F", "filter")
    add(m_flt, "Trace Overlay Filter (colour_addfil)...", win.action_overlay_filter_dialog, None, "overlay_filter")
    add(m_flt, "Clear Overlay Filters on Selected", win.action_clear_overlay_filters, None, "clear_overlay_filter")
    add(m_flt, "Restore Unfiltered (Raw) Waveforms", win.action_unfilter, "Ctrl+U", "unfilter")
    add(m_flt, "3-Component Polarization Filter...", win.action_apply_polfil_interactive, None, "polfil")
    m_ops = new_menu(m_proc, "&Basic Operations")
    add(m_ops, "Demean", win.action_demean, None, "demean")
    add(m_ops, "Remove Linear Trend", win.action_detrend, None, "detrend")
    add(m_ops, "Despike", win.action_despike, None, "despike")
    add(m_ops, "Integrate", win.action_integrate, None, "integrate")
    add(m_ops, "Differentiate", win.action_differentiate, None, "differentiate")
    add(m_ops, "Hilbert Transform", win.action_hilbert, None, "hilbert")
    add(m_ops, "Analytic Envelope", win.action_envelope, None, "envelope")
    m_smp = new_menu(m_proc, "&Sampling")
    add(m_smp, "Resample...", win.action_resample, None, "resample")
    add(m_smp, "Decimate...", win.action_decimate, None, "decimate")
    m_rot = new_menu(m_proc, "&Rotate Components")
    add(m_rot, "2D: ZNE to ZRT", win.action_rotate_2d, "Ctrl+O", "rot2")
    add(m_rot, "3D: ZNE to LQT", win.action_rotate_3d, None, "rot3")

    # ---------------- Analysis ----------------
    m_ana = new_menu(mb, "&Analysis")
    m_pick = new_menu(m_ana, "&Picking")
    add(m_pick, "Auto Pick All Traces", win.action_auto_pick, "Ctrl+P", "autopick")
    add(m_pick, "Correlation Pick Across Array", win.action_corr_pick, None, "corrpick")
    add(m_pick, "Show Theoretical Phases", win.action_theoretical_phases, "Ctrl+T", "theo")
    add(m_pick, "Identify Phases (IASP91)...", win.action_identify_phases, None, "identify_phases")
    m_arr = new_menu(m_ana, "&Array")
    add(m_arr, "Beam (toggle)", win.action_form_beam, "Ctrl+B", "beam")
    add(m_arr, "Align on Phase (toggle)", win.action_align_phase, "Ctrl+N", "align")
    add(m_arr, "Sum Traces", win.action_sum_traces, None, "sum")
    add(m_arr, "Phase-Weighted Stack", win.action_pws_traces, None, "pws")
    m_arr.addSeparator()
    add(m_arr, "FK Analysis...", win.action_open_fk_dialog, "Ctrl+K", "fk")
    add(m_arr, "Vespagram...", win.action_open_vespa_dialog, "Ctrl+V", "vespa")
    m_ana.addSeparator()
    add(m_ana, "Spectrum of Magnifier Window", win.action_open_spectrum_dialog, None, "spectrum")
    add(m_ana, "Spectrum of Selected Traces (Magnifier Window)", win.action_open_spectrum_selected, None, "spectrum_selected")
    add(m_ana, "Particle Motion and Polarization (MDIR)...", win.action_open_pmotion_dialog, None, "pmotion")

    # ---------------- Location ----------------
    m_loc = new_menu(mb, "&Location")
    add(m_loc, "Plane Wave Fit (Array)", win.action_locate_plane_wave, "Ctrl+L", "loc_plane")
    add(m_loc, "From Slowness and Backazimuth", win.action_locate_by_slowness, None, "loc_slowness")
    add(m_loc, "From S-P Time", win.action_locate_by_sp_diff, None, "loc_sp")
    add(m_loc, "Depth from Depth Phases (pP-P / sP-P)", win.action_fit_depth_phase, None, "loc_depth")
    add(m_loc, "Multi-Station Network Location...", win.action_open_network_loc_dialog, None, "loc_network")
    add(m_loc, "Solve Focal Mechanism (Beachball)", win.action_solve_focal_mechanism, None, "focal_mechanism")
    add(m_loc, "Fit Diagnostics & Inversion Residuals...", win.action_open_fit_diagnostic_window, "Ctrl+Shift+D", "fit_diagnostics")
    m_loc.addSeparator()
    add(m_loc, "Compute Magnitudes (ML, mb, MS, mB)", win.action_compute_magnitudes, None, "magnitudes")
    add(m_loc, "Scan Surface Waves (18-22s Ms)", win.action_scan_surface_waves, None, "surfmag")
    add(m_loc, "Event Summary...", win.action_open_event_summary, None, "event_summary")
    m_set = new_menu(m_loc, "Parameter &Sets")
    add(m_set, "Copy Set 1 to Set 2", lambda: win._copy_param_set(0, 1), None, "copy_set12")
    add(m_set, "Copy Set 2 to Set 1", lambda: win._copy_param_set(1, 0), None, "copy_set21")

    # ---------------- Tools ----------------
    m_tool = new_menu(mb, "T&ools")
    add(m_tool, "BGR EIDA Waveform &Downloader...", win.action_open_bgr_fetch_dialog, "Ctrl+G", "bgr_fetch")
    add(m_tool, "&Live Event Monitor...", win.action_open_live_event_dialog, "Ctrl+M", "live_events")
    win.act_seedlink_stream = add(m_tool, "Seed&Link Stream...", win.action_open_seedlink_dialog, None, "seedlink_stream")
    win.act_seedlink_stop = add(m_tool, "Stop SeedLink Stream", win.action_stop_seedlink, None, "seedlink_stop")
    win.act_seedlink_stop.setEnabled(False)
    add(m_tool, "&World Map (Stations && Events)...", win.action_open_world_map, "Ctrl+Shift+W", "world_map")
    m_tool.addSeparator()
    add(m_tool, "Create Synthetic Test Waveforms...", win.action_create_synthetics, None, "synthetics")
    add(m_tool, "Generate SFD Index File...", win.action_generate_sfd, None, "sfd")

    # ---------------- Help ----------------
    m_help = new_menu(mb, "&Help")
    win.act_tutorial = add(m_help, "&Tutorial Mode", lambda: None, "Ctrl+F1", "tutorial")
    win.act_tutorial.setCheckable(True)
    win.act_tutorial.toggled.connect(win._set_tutorial_mode)
    add(m_help, "Keyboard and Mouse &Reference", win.action_show_shortcuts_help, "F1", "shortcuts")
    m_help.addSeparator()
    m_samp = new_menu(m_help, "Load &Sample Data")
    add(m_samp, "GRF Array (GSE)", lambda: win.load_example_dataset("GRF_031102_0225.GSE"), None, "sample_grf")
    add(m_samp, "GRSN Network (GSE)", lambda: win.load_example_dataset("GRSN_031102_0224.GSE"), None, "sample_grsn")
    add(m_samp, "Mini-SEED Directory", lambda: win.load_example_dataset("mseed"), None, "sample_mseed")
    m_help.addSeparator()
    add(m_help, "&About", win.action_show_about, None, "about")

    # ---------------- Toolbar: the everyday workflow, left to right ----------------
    tb = QToolBar("Main Toolbar", win)
    tb.setObjectName("main_toolbar")
    win.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

    def tbar(label: str, slot: Any, key: str, shortcut: str = "") -> QAction:
        return _add_toolbar_action(win, tb, label, slot, key, shortcut)

    tbar("Open", win.action_open_read_dialog, "open", "Ctrl+R")
    tbar("BGR Fetch", win.action_open_bgr_fetch_dialog, "bgr_fetch", "Ctrl+G")
    win.act_live_toggle = tbar("Live: OFF", win.action_toggle_live_mode, "live_toggle")
    win.act_live_toggle.setCheckable(True)
    win.act_live_toggle.setChecked(False)
    live_btn = tb.widgetForAction(win.act_live_toggle)
    if live_btn is not None:
        live_btn.setObjectName("live_toggle_btn")
    tbar("Live Monitor", win.action_open_live_event_dialog, "live_events", "Ctrl+M")
    tbar("World Map", win.action_open_world_map, "world_map", "Ctrl+Shift+W")
    tb.addSeparator()
    tbar("Filter", win.action_open_filter_dialog, "filter", "Ctrl+F")
    tbar("Auto Pick", win.action_auto_pick, "autopick", "Ctrl+P")
    tbar("Theo. Phases", win.action_theoretical_phases, "theo", "Ctrl+T")
    tb.addSeparator()
    tbar("Beam", win.action_form_beam, "beam", "Ctrl+B")
    tbar("Align", win.action_align_phase, "align", "Ctrl+N")
    tbar("FK", win.action_open_fk_dialog, "fk", "Ctrl+K")
    tbar("Vespagram", win.action_open_vespa_dialog, "vespa", "Ctrl+V")
    tb.addSeparator()
    tbar("Plane Wave", win.action_locate_plane_wave, "loc_plane", "Ctrl+L")
    tbar("Event Summary", win.action_open_event_summary, "event_summary")
    tbar("Spectrum", win.action_open_spectrum_dialog, "spectrum")

    win._refresh_tooltips()
