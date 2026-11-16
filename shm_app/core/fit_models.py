"""Data contracts and models for SeismicHandler Modern Fit Diagnostics.

Defines:
- FitType: enumeration covering all 11 seismological fit domains.
- ResidualEntry: individual observation residual data record.
- ParameterUncertainty: estimated model parameter with 1-sigma error and 95% CI.
- ConfidenceEllipse: horizontal 95% confidence error ellipse representation.
- FitDiagnosticResult: unified container delivered from inversion engines to the UI.
- FitModificationRequest: container for reactive re-fitting with station exclusions and custom weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FitType(str, Enum):
    """Supported seismological modeling and fit calculation domains."""

    # 1. Hypocenter & Travel Times
    HYPOCENTER_GEIGER = "HYPOCENTER_GEIGER"
    LOCATION_TT = "HYPOCENTER_GEIGER"

    # 2. Origin Time Reduction
    ORIGIN_TIME = "ORIGIN_TIME"

    # 3. Array Plane-Wave Fitting
    PLANE_WAVE = "PLANE_WAVE"
    ARRAY_PLANE_WAVE = "PLANE_WAVE"

    # 4. Array FK & Semblance
    FK_SEMBLANCE = "FK_SEMBLANCE"
    ARRAY_FK = "FK_SEMBLANCE"

    # 5. S-P Differential Time & Depth Phase Analysis
    SP_DIFF_DEPTH = "SP_DIFF_DEPTH"
    DIFF_TIME_DEPTH = "SP_DIFF_DEPTH"

    # 6. Legacy TTFit Multi-Phase Downhill Simplex
    TTFIT_SIMPLEX = "TTFIT_SIMPLEX"

    # 7. Brune & Boatwright Spectral Source Parameters
    BRUNE_SPECTRUM = "BRUNE_SPECTRUM"
    SPECTRAL_SOURCE = "BRUNE_SPECTRUM"

    # 8. 3-Component Particle Motion & Polarization (MDIR)
    POLARIZATION_3C = "POLARIZATION_3C"

    # 9. Instrument Restitution & Pole-Zero Transfer Function Fit
    RESTITUTION_FIT = "RESTITUTION_FIT"

    # 10. Receiver Function Deconvolution Fit
    RECEIVER_FUNC = "RECEIVER_FUNC"

    # 11. Surface Wave Dispersion Curve Inversion
    SURFACE_DISPERSION = "SURFACE_DISPERSION"


@dataclass
class ResidualEntry:
    """Individual data observation residual record."""

    identifier: str  # e.g. "GRA1.BHZ.P" or "Freq_1.25Hz"
    observed: float  # Observed arrival time, amplitude, or slowness
    calculated: float  # Modeled/predicted forward value
    residual: float  # observed - calculated
    normalized_residual: float = 0.0  # residual / uncertainty
    weight: float = 1.0  # Assigned inversion weight (0.0 = excluded)
    uncertainty: float = 0.05  # Pick / observation standard error sigma
    distance_deg: float = 0.0  # Epicentral distance in degrees
    azimuth_deg: float = 0.0  # Azimuth in degrees
    station: str = ""
    phase: str = ""
    channel: str = ""
    is_outlier: bool = False  # True if |normalized_residual| > threshold
    extra_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterUncertainty:
    """Estimated model parameter value and uncertainty bounds."""

    name: str  # e.g. "latitude", "depth", "corner_freq"
    value: float
    standard_error: float  # 1-sigma formal error
    confidence_interval_95: Tuple[float, float]
    unit: str = ""


@dataclass
class ConfidenceEllipse:
    """Horizontal epicenter or slowness 95% confidence ellipse."""

    semi_major_km: float  # Semi-major axis length (km or s/deg)
    semi_minor_km: float  # Semi-minor axis length (km or s/deg)
    strike_deg: float  # Strike azimuth of semi-major axis (0..180 deg)
    confidence_level: float = 0.95
    center_x: float = 0.0  # Longitude or slowness x
    center_y: float = 0.0  # Latitude or slowness y


@dataclass
class FitDiagnosticResult:
    """Unified container delivered from any calculation engine to the diagnostic viewer."""

    fit_type: FitType
    title: str
    converged: bool
    iterations: int
    dof: int  # Degrees of freedom: N - P

    # Core Statistical Metrics
    rmse: float  # Root Mean Square Error
    wrmse: float  # Weighted RMSE
    nrmse: float = 0.0  # Normalized RMSE
    mad: float = 0.0  # Median Absolute Deviation
    chi2_reduced: float = 0.0  # Reduced Chi-Square
    r_squared: float = 0.0  # Coefficient of Determination R^2
    condition_number: float = 1.0  # Jacobian condition number kappa

    # Model Parameters & Uncertainties
    parameters: Dict[str, ParameterUncertainty] = field(default_factory=dict)
    covariance_matrix: Optional[np.ndarray] = None  # Shape (P, P)
    correlation_matrix: Optional[np.ndarray] = None  # Shape (P, P)
    confidence_ellipse: Optional[ConfidenceEllipse] = None

    # Residual Details
    residuals: List[ResidualEntry] = field(default_factory=list)
    outlier_count: int = 0

    # Visual Curves (for observed vs modelled overlays)
    overlay_curves: Dict[str, np.ndarray] = field(default_factory=dict)

    # Residual Normality Stats
    normality_p_value: float = 1.0  # Normality test p-value
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Engine Context (for re-triggering inversions)
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitModificationRequest:
    """Dispatched from the Diagnostic Window to trigger engine re-computation."""

    fit_type: FitType
    excluded_identifiers: List[str] = field(default_factory=list)  # Stations or identifiers toggled off
    weight_overrides: Dict[str, float] = field(default_factory=dict)  # Custom weights per identifier
    damping_factor: Optional[float] = None
    free_parameters: Optional[Dict[str, bool]] = None  # e.g. {"free_depth": False}
