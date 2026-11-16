"""Central PyQt6 signaling bridge for decoupled seismological fit diagnostics.

Implements M-06 requirements:
- Decouples computation backend from presentation layer.
- Qt signals for diagnostic delivery, interactive re-fit requests, trace highlighting,
  and parameter adoption.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from shm_app.core.fit_models import FitDiagnosticResult, FitModificationRequest


class FitDiagnosticBridge(QObject):
    """Central signaling bridge for decoupled seismological fit diagnostics."""

    # Dispatched when a fresh calculation/inversion produces diagnostic results
    fitDiagnosticAvailable = pyqtSignal(object)  # FitDiagnosticResult

    # Dispatched when user alters station inclusion, weights, or inversion parameters in the UI
    recomputeFitRequested = pyqtSignal(object)  # FitModificationRequest

    # Dispatched when user selects an item in the diagnostic table to highlight trace in canvas
    highlightTraceRequested = pyqtSignal(str)  # station code

    # Dispatched when station exclusion is explicitly toggled
    excludeStationRequested = pyqtSignal(str, bool)  # (station code, excluded_state)

    # Dispatched when weight overrides dictionary is updated
    updateWeightsRequested = pyqtSignal(dict)  # {identifier: float_weight}

    # Dispatched when user adopts solution parameters into parameter box / workspace
    adoptParametersRequested = pyqtSignal(object)  # AnalysisParamSet

    # Dispatched to zoom waveform magnifier canvas to a specific station phase
    zoomToPhaseRequested = pyqtSignal(str, str)  # (station, phase_name)

    # Dispatched when workspace is cleared
    workspaceCleared = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
