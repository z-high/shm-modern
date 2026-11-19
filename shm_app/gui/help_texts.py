"""Explanations shown in Tutorial Mode when hovering menu entries and toolbar buttons.

Each entry: action id -> (what it does, prerequisites, equivalent SH command or "").
Keep the texts short and in plain language; they are rendered as a rich-text hover card.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

HELP_TEXTS: Dict[str, Tuple[str, str, str]] = {
    # ---- File ----
    "open": (
        "Opens the Read dialog to load waveforms from GSE, Mini-SEED, SFD lists, Q-files or ASCII files.",
        "A data file or directory.",
        "READGSE / READS / READ",
    ),
    "open_evt": (
        "Loads a previously saved event file (.evt) and restores its phase picks and event parameters.",
        "Traces of the same stations should already be loaded.",
        "READEVT <file>",
    ),
    "import_stationxml": (
        "Imports FDSN StationXML metadata (coordinates, orientations, multi-stage pole-zero responses and sensitivities) into SHMConfig.",
        "A StationXML (.xml) file.",
        "READSXML <file>",
    ),
    "import_quakeml": (
        "Imports event origin, uncertainties, magnitudes, picks, and focal mechanisms from a QuakeML 1.2 or 1.0 XML file.",
        "A QuakeML (.xml / .qml) file.",
        "READQML <file>",
    ),
    "save_evt": (
        "Saves phase picks, location and magnitudes of the current event to an .evt file.",
        "At least one pick or filled parameter set.",
        "WRITEEVT <file>",
    ),
    "save_evt_clear": (
        "Saves the event file and then clears all picks and the active parameter set to start the next event.",
        "Same as Save Event Parameters.",
        "WRITEEVT <file>",
    ),
    "export_gse": ("Exports the traces as a GSE2 waveform file.", "Loaded traces.", "WRITEGSE <file>"),
    "export_mseed": ("Exports the traces as Mini-SEED (Steim-1).", "Loaded traces.", "WRITESEED <file>"),
    "export_q": ("Exports the traces as SeismicHandler Q-file pair (.QHD/.QBN).", "Loaded traces.", "WRITE <file>"),
    "export_ascii": ("Exports the traces as plain-text amplitude lists.", "Loaded traces.", "WRITEA <file>"),
    "export_quakeml": (
        "Exports event parameters, confidence ellipse, magnitudes, picks, and focal mechanisms to QuakeML 1.2 XML.",
        "Event parameters or picks.",
        "WRITEQML <file>",
    ),
    "export_geojson": (
        "Exports epicenter, 95% confidence ellipse polygon, and stations as RFC 7946 GeoJSON.",
        "Event parameters or traces.",
        "WRITEGEOJSON <file>",
    ),
    "export_bulletin": (
        "Generates a publication-quality vector PDF or SVG observatory bulletin asynchronously via BulletinWorker.",
        "Loaded traces or event parameters.",
        "REPORT <file>",
    ),
    "restore_session": (
        "Restores workspace traces, picks, parameters, and display settings from the saved session file.",
        "An existing .shm_session.json file.",
        "",
    ),
    "export_image": ("Saves the main waveform display as a PNG image.", "", ""),
    "quit": ("Closes the application.", "", "QUIT"),
    # ---- Edit ----
    "undo": ("Reverts the last change to traces, picks or parameters (filters, picks, beams, ...).", "", ""),
    "redo": ("Re-applies a change that was just undone.", "An earlier Undo.", ""),
    "delete_phase": (
        "Removes all picks of the phase currently selected in the Phase Box from every trace.",
        "Active phase set in the Phase Box.",
        "",
    ),
    "clear_all": (
        "Clears all picks, theoretical arrivals and the active parameter set.",
        "",
        "",
    ),
    # ---- Traces ----
    "select_all": ("Marks all visible traces as selected (orange).", "", "SELECT ALL"),
    "deselect_all": ("Removes the selection from all traces.", "", "SELECT NONE"),
    "delete_selected": ("Removes the selected traces from the workspace.", "Selected traces.", "DEL <list>"),
    "keep_selected": ("Keeps only the selected traces and removes all others.", "Selected traces.", ""),
    "show_hidden": ("Makes all hidden traces visible again.", "", "DISPLAY ALL"),
    "sort_distance": ("Sorts traces by epicentral distance.", "Event location in the Parameter Box.", ""),
    "sort_station": ("Sorts traces alphabetically by station and channel.", "", ""),
    "flip": ("Inverts the polarity of the selected traces (multiplies by -1).", "Selected traces.", "FLIP SEL"),
    "cut": (
        "Cuts all traces to the time window currently marked by the yellow Magnifier box.",
        "Magnifier window defined.",
        "CUT <lo> <hi>",
    ),
    # ---- View ----
    "view_magnifier": (
        "Zooms the main display to the time window of the Magnifier box.",
        "",
        "",
    ),
    "view_reset": ("Resets time axis, amplitude zoom and trace shifts to the full view.", "", "TRESET"),
    "view_back": ("Returns to the previous time window (zoom / pan history).", "Earlier zoom or pan.", ""),
    "view_forward": ("Steps forward again in the zoom / pan history.", "An earlier Previous View.", ""),
    "rows_fit": ("Fits all traces into the window height (classic SHM layout).", "", ""),
    "rows_compact": ("Uses at least 24 px per trace and adds a scrollbar if needed.", "", ""),
    "rows_normal": ("Uses at least 40 px per trace and adds a scrollbar if needed.", "", ""),
    "rows_large": ("Uses at least 64 px per trace and adds a scrollbar if needed.", "", ""),
    # ---- Process ----
    "filter": (
        "Opens the filter dialog: Butterworth, simulation filters (WWSSN, Wood-Anderson, ...) or filter files.",
        "Loaded traces.",
        "FILTER <name> / BUTTER BP f1 f2 n",
    ),
    "unfilter": ("Restores the raw waveforms as they were loaded.", "Previously filtered traces.", ""),
    "polfil": (
        "Applies a 3-component polarization filter that enhances rectilinear P motion.",
        "A Z, N, E triplet.",
        "POLFIL <win> <power> <mode>",
    ),
    "demean": ("Subtracts the mean value from every trace.", "", "DEMEAN ALL"),
    "detrend": ("Removes a linear trend from every trace.", "", "TREND ALL"),
    "despike": ("Removes single-sample spikes.", "", "DESPIKE ALL 5.0"),
    "integrate": ("Integrates traces (velocity to displacement).", "", "INT ALL"),
    "differentiate": ("Differentiates traces (displacement to velocity).", "", "DERIVE ALL 1"),
    "hilbert": ("Computes the Hilbert transform.", "", "HILBERT ALL"),
    "envelope": ("Replaces traces by their analytic envelope.", "", "ENVELOPE ALL"),
    "resample": ("Resamples traces to a new sample interval.", "", "RESAMPLE <dt>"),
    "decimate": ("Reduces the sampling rate by an integer factor.", "", "DECIMATE <n>"),
    "rot2": (
        "Rotates N and E components into radial and transverse (ZNE to ZRT).",
        "N and E traces; backazimuth in the Parameter Box.",
        "ROT2 <az>",
    ),
    "rot3": (
        "Rotates Z, N and E into the ray system L, Q, T.",
        "Z, N, E triplet; azimuth and incidence.",
        "ROT3 <az> <inc>",
    ),
    # ---- Analysis ----
    "autopick": (
        "Automatically picks the first onset on all traces (Baer-Kradolfer).",
        "Loaded traces.",
        "PICK ALL",
    ),
    "corrpick": (
        "Refines picks across the array by cross-correlating waveforms around a reference pick.",
        "A pick of the active phase on at least one trace.",
        "CORRPICK <phase>",
    ),
    "theo": (
        "Overlays theoretical arrival times for the current distance and depth.",
        "Distance and depth in the Parameter Box (defaults are used otherwise).",
        "THEO <dist> <depth>",
    ),
    "beam": (
        "Delays array traces by slowness and backazimuth and sums them. Using it a second time removes the beam trace.",
        "Station coordinates; slowness and azimuth in the Parameter Box.",
        "BEAM <az> <slo> CENTRE",
    ),
    "align": (
        "Shifts all traces so the active phase lines up and adds a sum trace. Using it a second time restores the original positions and removes the sum trace.",
        "Picks of the active phase.",
        "ALIGN <phase>",
    ),
    "sum": ("Adds a sum trace of all visible traces.", "", "SUM ALL"),
    "pws": ("Adds a phase-weighted stack, which suppresses incoherent noise.", "", "PWS ALL 2.0"),
    "fk": (
        "Wideband frequency-wavenumber analysis giving azimuth and slowness of a plane wave.",
        "Array traces; Magnifier window as analysis window.",
        "FK <flo> <fhi> <max slowness>",
    ),
    "vespa": (
        "Computes a slowness gather (vespagram) for a fixed azimuth.",
        "Array traces; azimuth in the Parameter Box.",
        "VESPA <az> <slo lo> <slo hi>",
    ),
    "spectrum": (
        "Computes the spectrum (FFT, Welch, MEM) of the part of the Magnifier trace inside the yellow window. Calling it again after moving the window adds another curve in a new colour to the same graph.",
        "A trace in the Magnifier and the yellow window over the signal.",
        "SPECTRUM /FFT <trace>",
    ),
    "spectrum_selected": (
        "Adds the spectrum of every selected trace, restricted to the yellow window, as separate coloured curves in the same graph.",
        "Selected traces (click their left margin) and the yellow window over the signal.",
        "",
    ),
    "group_components": (
        "Places the Z, N and E components of each station in adjacent rows, with a stronger line between stations. Switch off to return to the original order.",
        "",
        "",
    ),
    "pmotion": (
        "Shows 3-component particle motion and determines backazimuth and incidence (MDIR).",
        "Z, N, E traces.",
        "MDIR",
    ),
    # ---- Location ----
    "loc_plane": (
        "Fits a plane wave to the array picks and returns backazimuth, slowness and distance.",
        "At least 3 picks of the active phase (otherwise the auto-picker runs first).",
        "LOCATE <phase>",
    ),
    "loc_slowness": (
        "Derives the distance from slowness and backazimuth in the Parameter Box.",
        "Slowness and backazimuth.",
        "LOCATE /SLOWNESS <phase>",
    ),
    "loc_sp": (
        "Derives the distance from the S-P time difference on one trace.",
        "A P and an S pick on the same trace.",
        "",
    ),
    "loc_depth": (
        "Estimates focal depth from the pP-P or sP-P time difference.",
        "P and pP (or sP) picks on one trace; distance.",
        "",
    ),
    "loc_network": (
        "Opens the multi-station Geiger location with travel-time tables.",
        "Picks on several stations.",
        "",
    ),
    "fit_diagnostics": (
        "Opens the Fit Diagnostics & Inversion Residuals dock window displaying RMSE, reduced chi-square, covariance, error ellipses, and reactive station residual tables.",
        "Phase picks or inversion parameters.",
        "Location -> Fit Diagnostics & Inversion Residuals",
    ),
    "magnitudes": (
        "Measures amplitudes and periods and computes ML, mb, MS and mB.",
        "Picks and distance.",
        "MAG <phase>",
    ),
    "event_summary": (
        "Shows the event bulletin: location, region, magnitudes and all picks.",
        "",
        "",
    ),
    "copy_set12": ("Copies parameter set 1 to set 2.", "", ""),
    "copy_set21": ("Copies parameter set 2 to set 1.", "", ""),
    # ---- Tools ----
    "synthetics": ("Creates synthetic test waveforms (Kuepper, Gauss, sharp pulse).", "", "CREATE <kind> ..."),
    "sfd": ("Builds an SFD index file for a directory of Mini-SEED files.", "A Mini-SEED directory.", "SFD <dir>"),
    "bgr_fetch": (
        "Downloads Mini-SEED waveforms directly from BGR EIDA (FDSN-WS) for selected stations and UTC time window, saves them locally and loads them into SHM.",
        "Network access to eida.bgr.de, station selection and UTC time window.",
        "FETCH /BGR GR GRA1 BHZ <start> <dur>",
    ),
    "live_events": (
        "Opens the Live Event Monitor (real-time EMSC WebSocket, GFZ GEOFON, and USGS feeds) with distance/magnitude filtering, station planning, and Mini-SEED prefetching.",
        "Opt-in toggle; requires network access when live monitoring is ON.",
        "Tools -> Live Event Monitor (Ctrl+M)",
    ),
    "live_toggle": (
        "Toggles automated Live Event Mode ON or OFF (disabled by default on startup).",
        "Configure thresholds, station strategy, and auto-load policy in Live Event Monitor.",
        "Live Mode Toggle",
    ),
    "world_map": (
        "Opens the interactive Seismic World Map displaying all SHM database stations, loaded traces, reference station, active event epicenter, great-circle ray paths, distance rings, and Live Event Monitor earthquakes.",
        "SHMConfig station table; optional loaded traces, Parameter Box epicenter, or Live Events.",
        "Tools -> World Map",
    ),
    # ---- View, Docks, Sliders & Command Line ----
    "dock_phase": (
        "Shows or hides the Phase Picking Box dock panel (phase name, presets, quality weight, onset sharpness, polarity, depth rays).",
        "",
        "View -> Phase Picking Box",
    ),
    "dock_param": (
        "Shows or hides the Analysis Parameter Box dock panel (distance, backazimuth, slowness, depth, origin time, coordinates, normalization, zoom).",
        "",
        "View -> Analysis Parameter Box",
    ),
    "dock_fit": (
        "Shows or hides the Fit Diagnostics & Inversion Residuals dock window.",
        "",
        "View -> Fit Diagnostics Window",
    ),
    "pick_error_bars": (
        "Shows or hides [±sigma_t] pick uncertainty intervals mapped from pick quality Q (drag edges to adjust quality).",
        "Loaded traces with picks.",
        "",
    ),
    "mag_cursor_standard": (
        "Uses the standard vertical crosshair cursor in the Magnifier window.",
        "",
        "",
    ),
    "mag_cursor_hilbert": (
        "Overlays a 90-degree Hilbert-transformed cursor waveform in the Magnifier window for depth phases and core reflections.",
        "Reference window captured or Magnifier trace loaded.",
        "",
    ),
    "mag_cursor_reference": (
        "Overlays the captured reference waveform cursor in the Magnifier window for visual alignment.",
        "Reference window captured.",
        "",
    ),
    "mag_capture_ref": (
        "Captures the current Magnifier waveform window as the Reference / Hilbert cursor template.",
        "Magnifier trace loaded.",
        "",
    ),
    "overlay_filter": (
        "Applies a trace-specific secondary overlay filter (colour_addfil) with color badge to selected traces without changing global filter_name.",
        "Selected traces.",
        "ADDFIL <filter> <color>",
    ),
    "clear_overlay_filter": (
        "Clears secondary trace-specific overlay filters from selected traces.",
        "Selected traces with overlay filters.",
        "ADDFIL /CLEAR",
    ),
    "identify_phases": (
        "Matches observed pick travel times and slownesses against IASP91/AK135 tables to rank and label phases.",
        "Picked traces and epicentral distance or hypocenter.",
        "IDENTIFY",
    ),
    "surfmag": (
        "Scans vertical traces in the 18-22 s period window to measure surface-wave amplitudes and compute Prague Ms.",
        "Vertical (Z) traces and distance.",
        "SURFMAG",
    ),
    "focal_mechanism": (
        "Inverts P-wave first-motion polarities (+ / - / c / d) for a double-couple focal mechanism and renders the beachball in Fit Diagnostics Tab 5.",
        "Hypocenter/station coordinates and polarity picks.",
        "FIT /FOCAL",
    ),
    "seedlink_stream": (
        "Opens the SeedLink v3.1 TCP streaming dialog to receive live 512-byte Mini-SEED packets into the workspace.",
        "Reachable SeedLink server (host, port, streams).",
        "",
    ),
    "seedlink_stop": (
        "Stops the active SeedLink v3.1 real-time streaming worker.",
        "Active SeedLink stream.",
        "",
    ),
    "row_slider": (
        "Continuously scales vertical trace row height. Far left fits all traces into the window; dragging right gives fixed row heights with a vertical scrollbar.",
        "",
        "",
    ),
    "h_zoom_slider": (
        "Horizontally zooms the time axis from Full Span (left) up to 50x detail (right), anchored around the Magnifier or viewport center.",
        "Loaded traces.",
        "",
    ),
    "h_scroll": (
        "Scrolls horizontally across the loaded time span when zoomed in with the Horizontal Zoom slider or zoom history.",
        "Horizontal zoom > 1x.",
        "",
    ),
    "cmd_input": (
        "Classic SHM command interpreter input. Type SH commands such as FILTER SH_BP, BEAM, ALIGN P, THEO 65 30, DEMEAN ALL, or FETCH /BGR and press Enter.",
        "",
        "",
    ),
    "cmd_execute": (
        "Executes the command typed in the SHM command box.",
        "",
        "",
    ),
    "live_notify": (
        "Appears when Live Event Mode finishes prefetching qualifying Mini-SEED waveforms. Click to load the event directly into the workspace.",
        "Live Event Mode enabled with a READY event.",
        "",
    ),
    # ---- Phase Picking Box ----
    "phase_name": (
        "Active seismic phase label assigned to left-click picks on waveforms (for example P, Pg, Pn, S, PP, PKP, pP, sP).",
        "Loaded traces; left-click or Spacebar in Magnifier to pick.",
        "",
    ),
    "phase_preset": (
        "Quick-select button that sets the active picking phase label immediately.",
        "",
        "",
    ),
    "phase_spec": (
        "Onset character / specifier written into .evt picks: e (emergent), i (impulsive), + / - (amplitude direction), c / d (compression / dilatation).",
        "",
        "",
    ),
    "phase_sign": (
        "First-motion polarity of the onset: + (up/compressional), - (down/dilatational), or 0 (unclear).",
        "",
        "",
    ),
    "phase_quality": (
        "Standard SHM pick quality index from 0 (sharpest, full weight 4) to 9 (rejected, weight 0). Used in weighted location fits.",
        "",
        "",
    ),
    "phase_delete_btn": (
        "Deletes picks matching the active phase name (or opens quick phase cleanup on selected/all traces).",
        "Existing picks of the active phase.",
        "DEL <phase>",
    ),
    "phase_depth_rays": (
        "Highlights expected pP / sP depth-phase time offsets relative to primary P picks to help identify focal depth.",
        "Depth > 0 and primary P picks.",
        "",
    ),
    # ---- Analysis Parameter Box ----
    "param_set_slot": (
        "Selects one of three independent Parameter Box slots (Set 1..3), allowing side-by-side hypotheses or multiple events.",
        "",
        "",
    ),
    "param_distance": (
        "Epicentral angular distance in degrees (0 to 180 deg) between the reference station and the event epicenter.",
        "",
        "",
    ),
    "param_azimuth": (
        "Backazimuth in degrees (0 to 360 deg clockwise from North from the station array toward the earthquake epicenter).",
        "",
        "THEO / BEAM / ROT",
    ),
    "param_slowness": (
        "Horizontal slowness (ray parameter p) in s/deg, controlling plane-wave beamforming, alignment moveouts, and vespagram reference.",
        "",
        "BEAM / ALIGN",
    ),
    "param_depth": (
        "Hypocentral focal depth in km used for theoretical phase travel-time curves, slowness tables, and depth-phase fits.",
        "",
        "",
    ),
    "param_origin": (
        "Event origin time in SH UTC format (DD-MMM-YYYY_HH:MM:SS.SSS).",
        "",
        "",
    ),
    "param_coords": (
        "Event hypocenter latitude and longitude (WGS84 decimal degrees). Updates Flinn-Engdahl region and distance/backazimuth when changed.",
        "",
        "",
    ),
    "param_evtype": (
        "Event source classification written to .evt parameter output (tele, regional, local, nuclear, quarry, unknown).",
        "",
        "",
    ),
    "param_norm": (
        "Vertical amplitude scaling mode: A (Auto Trace per row), C (Constant global scale), AW (Auto shared visible Window), or SW (Single Trace visible Window).",
        "Loaded traces.",
        "NORM <A|C|AW|SW>",
    ),
    "param_amp_zoom": (
        "Increases (Amp +) or decreases (Amp -) the vertical trace gain, or resets it to unity (1:1).",
        "Loaded traces.",
        "ZOOM",
    ),
    "param_time_step": (
        "Pans the visible waveform time window left (<< Time) or right (Time >>) across the recording.",
        "Loaded traces.",
        "",
    ),
    # ---- World Map Dialog ----
    "wmap_view_preset": (
        "Jumps the map viewport directly to a preset geographic extent (Global, Europe/Alpine, Germany GRF/GRSN, or Active Event + Stations).",
        "",
        "",
    ),
    "wmap_layer_stations": (
        "Toggles map visibility of all SHM database stations alongside loaded workspace traces and array groups (GRF, GRSN).",
        "",
        "",
    ),
    "wmap_layer_events": (
        "Toggles the Active Parameter Box epicenter and Live Event Monitor earthquake feed markers.",
        "",
        "",
    ),
    "wmap_layer_paths": (
        "Draws spherical great-circle ray paths between the active/selected event epicenter and loaded or target stations.",
        "Event coordinates and at least one station.",
        "",
    ),
    "wmap_layer_rings": (
        "Draws concentric angular distance rings (Delta = 30, 60, 90, 120 deg) around the active event or reference station.",
        "",
        "",
    ),
    "wmap_layer_quarries": (
        "Toggles known Central European quarry/blasting locations from SHM's german_quarries.dat database.",
        "SHM inputs/german_quarries.dat.",
        "",
    ),
    "wmap_inspector": (
        "Shows detailed metadata (coordinates, elevation, distance Delta, backazimuth, channels, or magnitude/F-E region) for the clicked station or event.",
        "Click any marker on the map.",
        "",
    ),
    "wmap_adopt_event": (
        "Copies the selected event's epicenter, depth, origin time, distance, backazimuth, and region into the active Analysis Parameter Box.",
        "Select an event marker on the map.",
        "",
    ),
    "wmap_bgr_fetch": (
        "Opens the BGR/FDSN Waveform Downloader pre-filled with the selected event's time window and/or selected map station(s).",
        "Select a station or event on the map.",
        "",
    ),
    "wmap_select_traces": (
        "Highlights all loaded traces matching the selected map station inside the main waveform display.",
        "Select a loaded station on the map.",
        "",
    ),
    # ---- Sub-dialog controls ----
    "dlg_read_examples": (
        "One-click buttons to load SHM's bundled GSE or Mini-SEED example datasets.",
        "",
        "",
    ),
    "dlg_read_stations": (
        "Filters which stations from the selected dataset are read into memory.",
        "",
        "",
    ),
    "dlg_read_channels": (
        "Selects seismic stream code (BH, LH, HH, ...) and 3-component flags (Z, N, E).",
        "",
        "",
    ),
    "dlg_read_time": (
        "Specifies UTC start timestamp and read duration in seconds (or full file length).",
        "",
        "",
    ),
    "dlg_read_append": (
        "When checked, newly read traces are appended to the current workspace instead of replacing it.",
        "",
        "",
    ),
    "dlg_filter_sim": (
        "Optional instrument response simulation/deconvolution applied before filtering (for example WWSSN-SP or Wood-Anderson).",
        "",
        "",
    ),
    "dlg_filter_presets": (
        "Classic SHM named bandpass/highpass/lowpass filter definitions (e.g. GRSN_S+P, SH_BP, WWSSN_SP).",
        "",
        "",
    ),
    "dlg_filter_butter": (
        "Custom Butterworth recursive IIR filter parameters: pass type, corner frequencies (Hz), and filter order.",
        "",
        "",
    ),
    "dlg_filter_options": (
        "Application scope (all or selected traces), pre-filter demean, and Burg AR(20) startup extrapolation to reduce edge transients.",
        "",
        "",
    ),
    "dlg_fk_params": (
        "Frequency passband, maximum horizontal slowness s_max (s/deg), and slowness grid resolution for 2D beamforming / FK spectrum.",
        "Loaded array traces and Magnifier window.",
        "",
    ),
    "dlg_vespa_params": (
        "Fixed backazimuth angle, slowness sweep range/step, and nth-root power used to compute the time-slowness vespagram.",
        "Loaded array traces.",
        "",
    ),
    "dlg_spec_method": (
        "Chooses spectral estimation method (FFT Amplitude/Squared Spectrum, Bartlett/Welch PSD, Burg Maximum Entropy MEM AR, or Sliding-Window Hann Spectrogram) and optional log-log corner frequency fit bounds.",
        "Loaded traces and Magnifier window.",
        "",
    ),
    "dlg_pmotion_angles": (
        "Azimuth and incidence angles for 3-component L-Q-T particle motion rotation and hodogram inspection.",
        "3 components (Z, N, E) of one station.",
        "",
    ),
    "dlg_netloc_depth": (
        "Initial trial focal depth and Free-Depth toggle for iterative multi-station damped least-squares location.",
        "Picks on multiple stations.",
        "",
    ),
    "dlg_evt_save": (
        "Writes the displayed event summary, picks, magnitudes, and source parameters to a SeismicHandler .evt file.",
        "",
        "",
    ),
    "dlg_bgr_provider": (
        "Selects the FDSN Dataselect server (BGR EIDA Germany, GFZ GEOFON, IRIS EarthScope, EMSC, Orfeus) and output stream.",
        "Internet connection to the selected FDSN node.",
        "",
    ),
    "dlg_bgr_stations": (
        "Station network/code checkboxes and presets (GRF Array, GRSN, German Core) to request via FDSN-WS.",
        "",
        "",
    ),
    "dlg_bgr_time": (
        "UTC start date/time picker and window duration (or paste an SH timestamp like 02-NOV-2003_02:23:54).",
        "",
        "",
    ),
    "dlg_live_rules": (
        "Configures automated EMSC/GFZ/USGS polling, minimum teleseismic/regional/local magnitude rules (or editable Flat Min Mag), and workspace auto-load behavior.",
        "",
        "",
    ),
    "dlg_live_stations": (
        "Selects which stations Live Event Mode requests for each incoming event (Auto Smart, GRF, GRSN, Closest Europe, or editable Custom list).",
        "",
        "",
    ),
    "dlg_live_actions": (
        "Controls the live event queue: load a READY event into SHM, send its window/stations to BGR Fetch, ignore an event, or refresh feeds.",
        "",
        "",
    ),
    # ---- Help ----
    "tutorial": (
        "Switches Tutorial Mode on or off. When on, hovering any menu item, toolbar button, slider, dock control, or dialog widget explains it.",
        "",
        "",
    ),
    "shortcuts": ("Lists keyboard and mouse shortcuts.", "", ""),
    "about": ("Version, license and credits.", "", ""),
    "sample_grf": ("Loads the bundled GRF array example (GSE). For trying out the program.", "", ""),
    "sample_grsn": ("Loads the bundled GRSN network example (GSE).", "", ""),
    "sample_mseed": ("Loads the bundled Mini-SEED example directory.", "", ""),
}


def render_tooltip(title: str, key: str, shortcut: str = "") -> str:
    """Rich-text hover card for Tutorial Mode."""
    entry = HELP_TEXTS.get(key)
    if entry is None:
        return title
    what, needs, cmd = entry
    parts = [f"<b>{title}</b>", what]
    if needs:
        parts.append(f"<i>Needs:</i> {needs}")
    meta = []
    if shortcut:
        meta.append(f"<i>Shortcut:</i> {shortcut}")
    if cmd:
        meta.append(f"<i>SH command:</i> <code>{cmd}</code>")
    if meta:
        parts.append("&nbsp;&nbsp;".join(meta))
    return '<table width="340"><tr><td>' + "<br>".join(parts) + "</td></tr></table>"


def status_summary(key: str) -> str:
    entry = HELP_TEXTS.get(key)
    return entry[0].split(". ")[0].rstrip(".") + "." if entry else ""


def register_tutorial_help(
    obj: Any,
    title: str,
    key: str,
    shortcut: str = "",
    default_tooltip: str = "",
) -> Any:
    """Attach Tutorial Mode metadata to any QWidget or QAction and set its plain tooltip."""
    obj.setProperty("_shm_tutorial_meta", (title, key, shortcut, default_tooltip))
    obj.setToolTip(default_tooltip)
    return obj


def apply_tutorial_tooltips(root: Any, enabled: bool) -> None:
    """Update tooltips recursively across all registered QWidgets/QActions under root."""
    if root is None:
        return
    try:
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QWidget
    except Exception:
        return

    targets = [root]
    if hasattr(root, "findChildren"):
        targets.extend(root.findChildren(QWidget))
        targets.extend(root.findChildren(QAction))

    seen = set()
    for obj in targets:
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        meta = obj.property("_shm_tutorial_meta")
        if not meta or not isinstance(meta, tuple) or len(meta) != 4:
            continue
        title, key, shortcut, default_tooltip = meta
        if enabled:
            obj.setToolTip(render_tooltip(title, key, shortcut))
        else:
            obj.setToolTip(default_tooltip)

