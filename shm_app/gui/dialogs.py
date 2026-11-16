"""Backward-compatible facade re-exporting dock panels and analysis dialogs."""

from __future__ import annotations

from shm_app.gui.dialogs_analysis import (
    FKAnalysisDialog,
    FilterDialog,
    ParticleMotionDialog,
    SpectrumDialog,
    VespagramDialog,
)
from shm_app.gui.dialogs_io import ReadDataDialog
from shm_app.gui.dialogs_location import EventSummaryDialog, NetworkLocationDialog
from shm_app.gui.panels import ParameterBoxPanel, PhaseDialogPanel

__all__ = [
    "EventSummaryDialog",
    "FKAnalysisDialog",
    "FilterDialog",
    "NetworkLocationDialog",
    "ParameterBoxPanel",
    "ParticleMotionDialog",
    "PhaseDialogPanel",
    "ReadDataDialog",
    "SpectrumDialog",
    "VespagramDialog",
]
