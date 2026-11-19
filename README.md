# SeismicHandler Modern (`SHM` & `SHC`) — v3.0

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A ground-up modernization of Dr. Klaus Stammler's BGR **SeismicHandler Motif (`SHM`)** and **Command Interpreter (`SHC`)**, written in **Python 3.10+, PyQt6, NumPy, and Matplotlib**.

`shm-modern` combines an interactive multi-trace desktop GUI (`shm`) and a scriptable command-line interpreter (`shc`) with modern data acquisition, live earthquake monitoring, interactive global mapping, and inversion fit diagnostics.

---

## Key Features

### Waveform Display & Guided Workflow
- **High-Performance Waveform Canvas**: Smooth multi-trace display and synced zoomed magnifier view via 2-layer `QPixmap` offscreen buffering and vectorized min/max envelope decimation, paired with horizontal time-zoom and trace row-height controls.
- **Interactive World Map**: Global station database (`GRF`, `GRSN`, `GERES`, and workspace traces), German quarry sites, concentric distance rings, curved great-circle ray paths, active/live event overlays, station/event search, parameter adoption, and direct handoff to waveform downloaders.
- **Tutorial Mode & Parameter Docks**: Toggleable guided tooltips across menus, toolbars, docks, sliders, and dialogs alongside dockable Phase Picking and Analysis Parameter panels.

### Data Formats & Real-Time Acquisition
- **File Formats**: Mini-SEED (Steim-1, Steim-2, Float32, and SFD index lists), GSE1/GSE2 (`CM6`, `INT`, `FLT`), SH Q-Files (`.QHD`/`.QBN`), ASCII, and `.evt` event bulletins.
- **BGR / EIDA FDSN Downloader**: Built-in dialog with `GRF` / `GRSN` / `GERES` / `Workspace` presets, UTC window synchronization, channel/component filters, progress tracking, and automatic loading/SFD generation.
- **Opt-In Live Event Mode**: Real-time monitoring of EMSC WebSocket, GFZ, and USGS feeds with tiered distance/magnitude relevance filtering, automatic observatory or epicenter-near station planning, background Mini-SEED prefetching, and configurable auto-load policies.

### Signal & Array Processing
- **Filtering & Simulation**: Butterworth (`LP`, `HP`, `BP`, `BR`), recursive IIR LPC filters, instrument simulation/restitution (`WWSSN`, `Wood-Anderson`, `Kirnos`, `SRO`, `LRSM`), demean, detrend, despike, integration, differentiation, Hilbert transform, and analytic envelopes.
- **Spectra & 3C Polarization**: FFT / PSD / MEM spectra, spectrograms, polarization filtering, and 3-component `LQT` rotation.
- **Array Processing**: Plane-wave beamforming, phase-weighted stacking (`PWS`), plus non-blocking background `QThread` workers for wideband **FK analysis** and **Vespagrams**.

### Earthquake Location, Magnitudes & Fit Diagnostics
- **Location Methods**: Plane-wave array fit, `S - P` differential inversion, multi-station depth-phase fitting (`pP - P`, `sP - P`), and iterative Geiger/LocSAT network location using IASP91 travel-time tables (`.TTT`) with closed-loop residual and theoretical-pick feedback.
- **Magnitudes**: Per-station great-circle distance calculation with automated peak-to-peak amplitude/period measurement for `ML`, `mb`, `MS`, and `mB`.
- **Dockable Fit Diagnostic Window**: Interactive KPI cards (`RMSE`, reduced chi-squared, outlier count, condition number), Geiger residual & 95% error-ellipse plots, array slowness residuals, Brune/Boatwright source-parameter fits, 3C hodograms, interactive station re-weighting/re-fit loop, and CSV/PNG export.

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

Optionally point `SHM_ROOT` (or pass `--shm-root`) to your SeismicHandler reference data directory (`inputs/`, `filter/`, `data-examples/`) if it is not placed at `./sh` or `../SHM/sh`:
```bash
export SHM_ROOT=/path/to/SHM/sh
```

### Launch the Desktop GUI (`shm`)
```bash
# Empty workspace
python3 shm

# Auto-load bundled GRF array example waveforms
python3 shm --example

# Open with specific Mini-SEED, GSE, Q-file, or .evt files
python3 shm /path/to/waveforms.GSE

# Specify a custom SHM reference directory (inputs/ and filter/)
python3 shm --shm-root /path/to/SHM/sh --example
```

### Launch the Command-Line Interpreter (`shc`)
```bash
# Interactive REPL
python3 shc

# One-liner pipeline
python3 shc -c "READGSE example.GSE; PICK ALL; LOCATE P; MAG P; QUIT"

# Execute an .SHC script with optional positional arguments
python3 shc script.SHC arg1 arg2
```

### Run the Automated Test Suite
```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
```

---

## Project Layout

```text
shm-modern/
├── shm                        # Desktop GUI launcher (PyQt6)
├── shc                        # Command interpreter CLI / REPL launcher
├── shm_app/
│   ├── core/                  # Models, SHM config, station & F-E catalog lookup, time utils
│   ├── io/                    # Mini-SEED/SFD, GSE1/GSE2, SH Q-file, ASCII, and .evt I/O
│   ├── processing/            # Filters, trace ops, 3C polarization, FFT/PSD/MEM spectra
│   ├── seismology/            # Travel times, beamforming, FK, vespagram, location, diagnostics
│   ├── interpreter/           # SHC command parser & execution engine
│   ├── plugins/               # BGR/FDSN fetcher and Live Event Monitor service & UI
│   └── gui/                   # Main window, waveform canvases, dialogs, World Map, fit diagnostics
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
