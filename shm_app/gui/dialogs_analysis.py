"""Backward-compatible facade re-exporting analysis dialogs and workers.

Individual dialogs live in focused modules:
- `dialog_filter.py`    : FilterDialog
- `dialog_fk.py`        : FKAnalysisDialog, FKWorker, _compute_fk_bundle
- `dialog_vespagram.py` : VespagramDialog, VespagramWorker
- `dialog_spectrum.py`  : SpectrumDialog, BruneInversionWorker
- `dialog_pmotion.py`   : ParticleMotionDialog, _HODOGRAM_CMAP
"""

from __future__ import annotations

from shm_app.gui.dialog_filter import FilterDialog
from shm_app.gui.dialog_fk import FKAnalysisDialog, FKWorker, _compute_fk_bundle
from shm_app.gui.dialog_pmotion import _HODOGRAM_CMAP, ParticleMotionDialog
from shm_app.gui.dialog_spectrum import BruneInversionWorker, SpectrumDialog
from shm_app.gui.dialog_vespagram import VespagramDialog, VespagramWorker
from shm_app.processing.polarization import compute_mdir

__all__ = [
    "BruneInversionWorker",
    "FKAnalysisDialog",
    "FKWorker",
    "FilterDialog",
    "ParticleMotionDialog",
    "SpectrumDialog",
    "VespagramDialog",
    "VespagramWorker",
    "_HODOGRAM_CMAP",
    "_compute_fk_bundle",
    "compute_mdir",
]
