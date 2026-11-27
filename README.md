# SeismicHandler Modern (`SHM` & `SHC`) — v3.1

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A ground-up modernization of Dr. Klaus Stammler's BGR **SeismicHandler Motif (`SHM`)** and **Command Interpreter (`SHC`)**, written in **Python 3.10+, PyQt6, NumPy, and Matplotlib**.

`shm-modern` combines an interactive multi-trace desktop GUI (`shm`) and a scriptable command-line interpreter (`shc`) with modern data acquisition, live earthquake monitoring, interactive global mapping, and inversion fit diagnostics.

---

## Key Features

### Waveform Display, Reporting & Guided Workflow
- **High-Performance Waveform Canvas**: Smooth multi-trace display and synced zoomed magnifier view via 2-layer `QPixmap` offscreen buffering and vectorized min/max envelope decimation, paired with horizontal time-zoom, trace row-height controls, and per-trace overlay filter rendering (`ADDFIL` / `COLOUR_ADDFIL`).
- **Interactive World Map**: Global station database (`GRF`, `GRSN`, `GERES`, and workspace traces) over a bundled vector country basemap (`shm_app/assets/world_countries.json.gz`), German quarry sites, concentric distance rings, curved great-circle ray paths, active/live event overlays, station/event search, parameter adoption, and direct handoff to waveform downloaders.
- **Publication-Grade Vector Bulletins (`REPORT`)**: Headless generation of multi-page vector **PDF** and self-contained **SVG** observatory bulletins combining waveform sections, travel-time residuals, station map, and focal mechanism beachball plots (`shm_app/gui/reporting.py`).
- **Session Autosave & Crash Recovery**: Atomic workspace persistence (`.shm_session.json` by default) with unclean-shutdown recovery prompts, configurable via `SHM_SESSION_PATH` and `SHM_NO_SESSION_PROMPT=1` (`shm_app/core/session.py`).
- **Tutorial Mode & Parameter Docks**: Toggleable guided tooltips across menus, toolbars, docks, sliders, and dialogs alongside dockable Phase Picking and Analysis Parameter panels.

### Data Formats & Real-Time Acquisition
- **Waveform & Station Formats**: Mini-SEED (`Steim-1`, `Steim-2`, `Float32`, and SFD index lists), GSE1/GSE2 (`CM6`, `INT`, `FLT`), SH Q-Files (`.QHD`/`.QBN`), ASCII, and FDSN **StationXML** (`READSXML`).
- **Catalog & Event Exchange**: Legacy `.evt` event bulletins, **QuakeML 1.2** import/export (`READQML` / `WRITEQML`), and **GeoJSON** feature export (`WRITEGEOJSON` with station geometry, 95% error ellipse, and focal mechanisms).
- **SeedLink v3.1 Streaming**: Native pure-Python TCP SeedLink streaming client and thread-safe waveform ring buffer (`shm_app/io/seedlink.py`), integrated into the GUI (`Tools -> SeedLink Stream...`) via `LiveEventCoordinator`.
- **BGR / EIDA FDSN Downloader**: Built-in dialog with `GRF` / `GRSN` / `GERES` / `Workspace` presets, UTC window synchronization, channel/component filters, progress tracking, and automatic loading/SFD generation.
- **Opt-In Live Event Mode**: Real-time monitoring of EMSC WebSocket, GFZ, and USGS feeds with tiered distance/magnitude relevance filtering, automatic observatory or epicenter-near station planning, background Mini-SEED prefetching, and configurable auto-load policies.

### Signal & Array Processing
- **Filtering, Simulation & Autocutting**: Butterworth (`LP`, `HP`, `BP`, `BR`), recursive IIR LPC filters, instrument simulation/restitution (`WWSSN`, `Wood-Anderson`, `Kirnos`, `SRO`, `LRSM`), **overlay filters** (`ADDFIL` / `COLOUR_ADDFIL`), **filter-transient autocutting** (`AUTOCUT` command and `/AUTOCUT` qualifier), demean, detrend, despike, integration, differentiation, Hilbert transform, and analytic envelopes.
- **Spectra & 3C Polarization**: FFT / PSD / MEM spectra, spectrograms, polarization filtering, and 3-component `LQT` rotation.
- **Array Processing & Calibration**: Plane-wave beamforming, phase-weighted stacking (`PWS`), **slowness/azimuth calibration** (`CALIB` with `/ARRAY` or external calibration tables), plus non-blocking background `QThread` workers for wideband **FK analysis** and **Vespagrams**.

### Earthquake Location, Focal Mechanisms, Magnitudes & Fit Diagnostics
- **Location & Phase Identification**: Plane-wave array fit, `S - P` differential inversion, multi-station depth-phase fitting (`pP - P`, `sP - P`), iterative Geiger/LocSAT network location using IASP91 travel-time tables (`.TTT`) with closed-loop residual/theoretical-pick feedback, and automated **phase identification** (`IDENTIFY`).
- **Focal Mechanism Inversion (`FIT /FOCAL`)**: First-motion polarity grid-search solver with beachball rendering and nodal-plane diagnostics (`shm_app/seismology/focal_mechanism.py`).
- **Magnitudes**: Per-station great-circle distance calculation with automated peak-to-peak amplitude/period measurement for `ML`, `mb`, `MS` (including dedicated surface-wave `SURFMAG`), and `mB`.
- **Dockable Fit Diagnostic Window**: Interactive KPI cards (`RMSE`, reduced chi-squared, outlier count, condition number), Geiger residual & 95% error-ellipse plots, array slowness residuals, Brune/Boatwright source-parameter fits, 3C hodograms, focal mechanism beachballs, interactive station re-weighting/re-fit loop, and CSV/PNG export.

---

## Quick Start

### Installation & Prerequisites
Install dependencies directly or install the `shm_app` package in editable mode via `pyproject.toml`:
```bash
# Install runtime dependencies
pip install -r requirements.txt

# Or install the shm_app package in editable mode (PEP 621 / pyproject.toml)
pip install -e .
```
*(Run the `shm` and `shc` launcher scripts from the repository root with `python3 shm` / `python3 shc`.)*

### External `SHM_ROOT` Reference & Example Data
Example waveforms and legacy SeismicHandler reference files—such as `data-examples/` (`GRF_031102_0225.GSE`, `mseed/`), `.TTT` IASP91 travel-time tables, `STATINF.DAT`, and external `filter/` definitions—are **not bundled inside `shm-modern`**. Instead, `shm_app` looks for an external SeismicHandler root directory (`SHM_ROOT`) in the following order:
1. `SHM_ROOT` environment variable (or `--shm-root` CLI option)
2. `./sh`, `../SHM/sh`, `../sh`, or `/usr/local/sh`

Synthetic trace generation, built-in Butterworth/IIR filters, the vector world map basemap, FDSN/SeedLink acquisition, and QuakeML/StationXML/Mini-SEED/GSE/Q-file I/O all work without `SHM_ROOT`. To enable `--example` and legacy `.TTT`/`STATINF.DAT` files, set `SHM_ROOT` or pass `--shm-root`:
```bash
export SHM_ROOT=/path/to/SHM/sh
```

### Launch the Desktop GUI (`shm`)
```bash
# Empty workspace
python3 shm

# Auto-load GRF array example waveforms from SHM_ROOT (data-examples/GRF_031102_0225.GSE)
python3 shm --example

# Open with specific Mini-SEED, GSE, Q-file, or .evt files
python3 shm /path/to/waveforms.GSE

# Specify a custom SHM reference directory (inputs/, filter/, data-examples/)
python3 shm --shm-root /path/to/SHM/sh --example
```

**Session Environment Variables:**
- `SHM_SESSION_PATH`: Override the default `.shm_session.json` autosave/recovery file location.
- `SHM_NO_SESSION_PROMPT=1`: Suppress the startup recovery dialog when an unclean `.shm_session.json` is present.

### Launch the Command-Line Interpreter (`shc`)
```bash
# Interactive REPL
python3 shc

# One-liner pipeline (exits automatically after the pipeline completes; QUIT is optional)
python3 shc -c "READGSE /path/to/SHM/sh/data-examples/GRF_031102_0225.GSE; PICK ALL; LOCATE P; MAG P"

# Execute an .SHC script with optional positional arguments (#1, #2, ...)
python3 shc script.SHC arg1 arg2
```
> **Note on `shc` exit codes:** Passing a nonexistent script path to `shc <script.SHC>` prints an error to `stderr` and exits with code `1`. In `-c` one-liners and `.SHC` scripts, individual command failures print `SH Error ...` diagnostics to `stderr` without aborting subsequent commands, and `shc` exits with code `1` if any `SH Error` occurred (or `0` when all commands succeed).

### Run the Automated Test Suite
```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
```

---

## Project Layout

```text
shm-modern/
├── pyproject.toml             # PEP 621 / PEP 639 package metadata (shm_app)
├── requirements.txt           # Core runtime dependencies (PyQt6, numpy, matplotlib)
├── LICENSE                    # GNU General Public License v3.0
├── NOTICE                     # Third-party attributions & original SH/SHM notices
├── README.md
├── shm                        # Desktop GUI launcher (PyQt6)
├── shc                        # Command interpreter CLI / REPL launcher
├── shm_app/
│   ├── __init__.py            # SHM_ROOT resolution & MPLCONFIGDIR setup
│   ├── __version__.py         # Package version & license metadata
│   ├── assets/                # Bundled data assets (world_countries.json.gz)
│   ├── core/                  # Models, SHM config, station/F-E lookup, time utils, session & fit models
│   ├── io/                    # Mini-SEED/SFD, GSE1/GSE2, Q-file, ASCII, .evt, QuakeML, StationXML, GeoJSON, SeedLink
│   ├── processing/            # Filters, trace ops, 3C polarization, FFT/PSD/MEM spectra
│   ├── seismology/            # Travel times, beamforming, FK, vespagram, location, focal mechanism, diagnostics
│   ├── interpreter/           # SHC command parser & execution engine
│   ├── plugins/               # BGR/FDSN fetcher and Live Event Monitor planner, service & UI
│   └── gui/                   # Main window, canvases, panels, dialogs, World Map, reporting & SVG icons/
│       └── icons/             # Bundled UI SVG icons
└── tests/                     # Headless unit, GUI, usability, and adversarial test suites
```

---

## License & Attribution

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License (Version 3 or later)** as published by the Free Software Foundation. See [`LICENSE`](LICENSE) for the full text.

This software is provided "as is", without warranty of any kind, as specified in Sections 15 and 16 of the GPL-3.0.

### Original Work & Attribution
Based on the concepts, algorithms, and file formats of **SeismicHandler (`SH` / `SHM`)**:
- **Original Author:** Dr. Klaus Stammler
- **Original Organization:** Federal Institute for Geosciences and Natural Resources (BGR), Hannover, Germany
- **Original License:** GNU General Public License (`GPL-2.0-or-later`)
- See [`NOTICE`](NOTICE) for third-party dependencies and acknowledgments.
