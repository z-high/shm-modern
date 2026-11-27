"""Backward-compatible facade re-exporting seismology array processing, location, picking, and magnitudes.

Individual implementations live in focused domain modules:
- `calibration.py`      : slowness/azimuth calibration tables (`calib_*.dat`)
- `array_processing.py` : plane-wave fit, beamforming, alignment, vespagram, FK, ARF
- `location.py`         : slowness-azimuth, S-P, depth-phase, and Geiger/LocSAT location
- `picking.py`          : Baer-Kradolfer + AIC onset picker, array cross-correlation, trace QC
- `magnitudes.py`       : amplitude/period measurement and ML, mb, MS, mB formulas
- `phase_id.py`         : automated IASP91 phase identification
"""

from __future__ import annotations

from shm_app.seismology.array_processing import (
    DEG_TO_KM,
    POOR_ELEMENT_CC_THRESHOLD,
    align_traces_on_phase,
    beam_delays_sec,
    build_fk_fit_diagnostic,
    compute_array_response_function,
    compute_beam,
    compute_beam_diagnostics,
    compute_fk_analysis,
    compute_vespagram,
    plane_wave_locate,
)
from shm_app.seismology.calibration import (
    CALC_STARTPOS,
    SlownessCalibrationEntry,
    SlownessCalibrationTable,
    _resolve_calib_table,
    load_calibration_for_array,
)
from shm_app.seismology.location import (
    fit_depth_phase,
    locate_by_slowness_azimuth,
    locate_by_sp_diff,
    locate_network_geiger,
)
from shm_app.seismology.magnitudes import (
    MB_SIGMA_DEPTH,
    MB_SIGMA_DIST,
    MB_SIGMA_GRID,
    ML_SIGMA_TABLE,
    compute_magnitude_mb,
    compute_magnitude_mbb,
    compute_magnitude_ml,
    compute_magnitude_ms,
    compute_trace_window_magnitudes,
    measure_amplitude_and_period,
)
from shm_app.seismology.phase_id import (
    identify_phase,
    identify_trace_picks,
)
from shm_app.seismology.picking import (
    assess_trace_quality,
    correlate_traces,
    corrpick_array,
    pick_onset_baer_kradolfer,
)

__all__ = [
    "CALC_STARTPOS",
    "DEG_TO_KM",
    "MB_SIGMA_DEPTH",
    "MB_SIGMA_DIST",
    "MB_SIGMA_GRID",
    "ML_SIGMA_TABLE",
    "POOR_ELEMENT_CC_THRESHOLD",
    "SlownessCalibrationEntry",
    "SlownessCalibrationTable",
    "_resolve_calib_table",
    "align_traces_on_phase",
    "assess_trace_quality",
    "beam_delays_sec",
    "build_fk_fit_diagnostic",
    "compute_array_response_function",
    "compute_beam",
    "compute_beam_diagnostics",
    "compute_fk_analysis",
    "compute_magnitude_mb",
    "compute_magnitude_mbb",
    "compute_magnitude_ml",
    "compute_magnitude_ms",
    "compute_trace_window_magnitudes",
    "compute_vespagram",
    "correlate_traces",
    "corrpick_array",
    "fit_depth_phase",
    "identify_phase",
    "identify_trace_picks",
    "load_calibration_for_array",
    "locate_by_slowness_azimuth",
    "locate_by_sp_diff",
    "locate_network_geiger",
    "measure_amplitude_and_period",
    "pick_onset_baer_kradolfer",
    "plane_wave_locate",
]
