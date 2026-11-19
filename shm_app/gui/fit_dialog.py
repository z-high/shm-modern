"""Dockable Seismological Fit Diagnostics and Inversion Residual Window.

Implements M-07 and M-08 requirements:
- FitDiagnosticWindow(QDockWidget) dockable to BottomDockWidgetArea.
- 5 KPI summary cards: Global RMSE, Chi2_red, Outlier Count, Inversion Status, Condition Number kappa.
- 6 domain tabs:
  1. Global Overview (parameter correlations, multi-fit scorecard, network geometry)
  2. Travel Time Inversion & Geiger Residuals (residuals vs distance & azimuth, 95% error ellipse plot, interactive station table)
  3. Array Beamforming & FK Analysis (slowness grid, beam residuals, ARF)
  4. Spectral Source Parameters (Brune/Boatwright omega^-2 fit curve, 2D misfit contour, source parameters)
  5. 3C Particle Motion & Polarization (hodograms with color gradient, rectilinearity, planarity, confidence cones)
  6. Instrument Restitution & Deconvolution (transfer function, SNR vs frequency, receiver functions)
- Interactive station table in Tab 2 with station checkboxes and weight inputs.
- Reactive re-fit loop emitting recomputeFitRequested(FitModificationRequest).
- Outlier rejection, parameter adoption, and CSV/PNG export.
"""

from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.gui.ui_helpers import safe_tight_layout
from shm_app.core.fit_models import (
    ConfidenceEllipse,
    FitDiagnosticResult,
    FitModificationRequest,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.seismology.focal_mechanism import compute_beachball_contours, project_schmidt


class KPICard(QFrame):
    """Reusable KPI summary metric card widget."""

    def __init__(self, title: str, value: str = "--", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "KPICard {"
            "  background-color: #161b22;"
            "  border: 1px solid #30363d;"
            "  border-radius: 6px;"
            "  padding: 4px 8px;"
            "}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold;")
        self.lbl_value = QLabel(value)
        self.lbl_value.setStyleSheet("color: #e6edf3; font-size: 14px; font-weight: bold; font-family: Monospace;")

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)

    def set_value(self, val: str, color: Optional[str] = None) -> None:
        self.lbl_value.setText(val)
        if color:
            self.lbl_value.setStyleSheet(
                f"color: {color}; font-size: 14px; font-weight: bold; font-family: Monospace;"
            )


class FitDiagnosticWindow(QDockWidget):
    """Dockable Fit Diagnostics Window attached to BottomDockWidgetArea."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        bridge: Optional[FitDiagnosticBridge] = None,
    ) -> None:
        super().__init__("Fit Diagnostics & Inversion Residuals", parent)
        self.setObjectName("FitDiagnosticWindow")
        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        self.bridge = bridge or FitDiagnosticBridge(self)
        self.current_result: Optional[FitDiagnosticResult] = None
        self._last_fm_result: Optional[Dict[str, Any]] = None
        self._block_table_signals = False
        self._syncing_table_items = False

        # Main container
        self.main_widget = QWidget(self)
        self.setWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(4)

        # ---------------- Top Control & KPI Header ----------------
        header_frame = QFrame(self.main_widget)
        header_frame.setFrameShape(QFrame.Shape.StyledPanel)
        header_frame.setStyleSheet(
            "QFrame { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; }"
        )
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)

        # Domain combo
        self.domain_combo = QComboBox()
        self.domain_combo.setStyleSheet(
            "QComboBox { background-color: #21262d; color: #e6edf3; border: 1px solid #30363d; "
            "border-radius: 4px; padding: 3px 8px; font-weight: bold; }"
        )
        self.domain_map = [
            (FitType.HYPOCENTER_GEIGER, "1. Hypocenter & Travel Times (Geiger)"),
            (FitType.ORIGIN_TIME, "2. Origin Time Reduction"),
            (FitType.PLANE_WAVE, "3. Array Plane-Wave Fit"),
            (FitType.FK_SEMBLANCE, "4. Array FK & Semblance"),
            (FitType.SP_DIFF_DEPTH, "5. S-P Diff Time & Depth Phases"),
            (FitType.TTFIT_SIMPLEX, "6. TTFit Multi-Phase Simplex"),
            (FitType.BRUNE_SPECTRUM, "7. Spectral Source Parameters (Brune)"),
            (FitType.POLARIZATION_3C, "8. 3C Particle Motion & Polarization"),
            (FitType.RESTITUTION_FIT, "9. Instrument Restitution & Pole-Zero"),
            (FitType.RECEIVER_FUNC, "10. Receiver Function Deconvolution"),
            (FitType.SURFACE_DISPERSION, "11. Surface Wave Dispersion"),
        ]
        for ftype, title in self.domain_map:
            self.domain_combo.addItem(title, ftype)
        self.domain_combo.currentIndexChanged.connect(self._on_domain_changed)
        header_layout.addWidget(self.domain_combo)

        # Status badge
        self.lbl_status_badge = QLabel("IDLE")
        self.lbl_status_badge.setStyleSheet(
            "background-color: #21262d; color: #8b949e; border: 1px solid #30363d; "
            "border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px;"
        )
        header_layout.addWidget(self.lbl_status_badge)

        # 5 KPI cards
        self.kpi_rmse = KPICard("Global RMSE", "-- s")
        self.kpi_chi2 = KPICard("Reduced χ²", "--")
        self.kpi_outliers = KPICard("Outlier Count", "--")
        self.kpi_status = KPICard("Inversion Status", "Idle")
        self.kpi_cond = KPICard("Condition κ(J)", "--")

        for card in (self.kpi_rmse, self.kpi_chi2, self.kpi_outliers, self.kpi_status, self.kpi_cond):
            header_layout.addWidget(card)

        main_layout.addWidget(header_frame)

        # ---------------- 6 Domain Tabs ----------------
        self.tabs = QTabWidget(self.main_widget)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #30363d; background: #0d1117; } "
            "QTabBar::tab { background: #161b22; color: #8b949e; padding: 6px 12px; border: 1px solid #30363d; "
            "border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; } "
            "QTabBar::tab:selected { background: #0d1117; color: #58a6ff; font-weight: bold; border-color: #58a6ff; }"
        )

        self._build_tab1_overview()
        self._build_tab2_travel_time()
        self._build_tab3_array_fk()
        self._build_tab4_spectrum()
        self._build_tab5_polarization()
        self._build_tab6_restitution()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        main_layout.addWidget(self.tabs, 1)

        # ---------------- Bottom Action Toolbar ----------------
        toolbar_frame = QFrame(self.main_widget)
        toolbar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        toolbar_frame.setStyleSheet(
            "QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; }"
        )
        tb_layout = QHBoxLayout(toolbar_frame)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(8)

        self.chk_auto_refit = QCheckBox("Auto-Refit on Table Edit")
        self.chk_auto_refit.setChecked(True)
        self.chk_auto_refit.setStyleSheet("color: #e6edf3; font-weight: bold;")
        tb_layout.addWidget(self.chk_auto_refit)

        self.btn_deactivate_outliers = QPushButton("Deactivate Outliers (> 3σ)")
        self.btn_deactivate_outliers.setStyleSheet(
            "QPushButton { background-color: #21262d; color: #f85149; border: 1px solid #f85149; "
            "border-radius: 4px; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #f85149; color: #ffffff; }"
        )
        self.btn_deactivate_outliers.clicked.connect(self._on_deactivate_outliers)
        tb_layout.addWidget(self.btn_deactivate_outliers)

        self.btn_recompute = QPushButton("Recompute Inversion")
        self.btn_recompute.setStyleSheet(
            "QPushButton { background-color: #1f6feb; color: #ffffff; border: 1px solid #388bfd; "
            "border-radius: 4px; padding: 4px 12px; font-weight: bold; } "
            "QPushButton:hover { background-color: #388bfd; }"
        )
        self.btn_recompute.clicked.connect(self._on_recompute_clicked)
        tb_layout.addWidget(self.btn_recompute)

        self.btn_adopt = QPushButton("Adopt to Parameter Box")
        self.btn_adopt.setStyleSheet(
            "QPushButton { background-color: #238636; color: #ffffff; border: 1px solid #2ea043; "
            "border-radius: 4px; padding: 4px 12px; font-weight: bold; } "
            "QPushButton:hover { background-color: #2ea043; }"
        )
        self.btn_adopt.clicked.connect(self._on_adopt_clicked)
        tb_layout.addWidget(self.btn_adopt)

        tb_layout.addStretch(1)

        self.btn_export_csv = QPushButton("Export Residuals (CSV)")
        self.btn_export_csv.setStyleSheet(
            "QPushButton { background-color: #21262d; color: #8b949e; border: 1px solid #30363d; "
            "border-radius: 4px; padding: 4px 10px; } QPushButton:hover { color: #e6edf3; }"
        )
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        tb_layout.addWidget(self.btn_export_csv)

        self.btn_export_png = QPushButton("Export Plot (PNG)")
        self.btn_export_png.setStyleSheet(
            "QPushButton { background-color: #21262d; color: #8b949e; border: 1px solid #30363d; "
            "border-radius: 4px; padding: 4px 10px; } QPushButton:hover { color: #e6edf3; }"
        )
        self.btn_export_png.clicked.connect(self._on_export_png)
        tb_layout.addWidget(self.btn_export_png)

        main_layout.addWidget(toolbar_frame)

        # Hook up bridge signals
        self.bridge.fitDiagnosticAvailable.connect(self.set_fit_result)

    # ------------------------------------------------------------------
    # Tab 1: Global Overview
    # ------------------------------------------------------------------

    def _build_tab1_overview(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Left Column: Parameter Uncertainties and Stats
        left_box = QGroupBox("Inverted Model Parameters & Uncertainties")
        left_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        left_layout = QVBoxLayout(left_box)

        self.table_params = QTableWidget(0, 4)
        self.table_params.setHorizontalHeaderLabels(["Parameter", "Value", "1-Sigma StdErr", "95% Conf. Interval"])
        self.table_params.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_params.horizontalHeader().setStretchLastSection(True)
        self.table_params.setStyleSheet("background-color: #161b22; color: #e6edf3; gridline-color: #30363d;")
        left_layout.addWidget(self.table_params)

        self.lbl_normality = QLabel("Normality: Skewness=0.00, Kurtosis=0.00, JB p-value=1.0000")
        self.lbl_normality.setStyleSheet("color: #8b949e; font-family: Monospace;")
        left_layout.addWidget(self.lbl_normality)

        layout.addWidget(left_box, 1)

        # Right Column: Correlation Matrix Heatmap
        right_box = QGroupBox("Parameter Correlation Matrix")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        right_layout = QVBoxLayout(right_box)

        self.fig_corr = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_corr = FigureCanvas(self.fig_corr)
        right_layout.addWidget(self.canvas_corr)

        layout.addWidget(right_box, 1)
        self.tabs.addTab(tab, "Global Overview")

    # ------------------------------------------------------------------
    # Tab 2: Travel Time & Geiger Inversion
    # ------------------------------------------------------------------

    def _build_tab2_travel_time(self) -> None:
        tab = QWidget()
        v_layout = QVBoxLayout(tab)
        v_layout.setContentsMargins(6, 6, 6, 6)
        v_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top Section: Two Matplotlib Canvases
        top_widget = QWidget()
        top_widget.setMinimumHeight(140)
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # Canvas 1: Residual Scatter vs Distance & Azimuth
        scatter_box = QGroupBox("Arrival Residuals vs Distance & Azimuth")
        scatter_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        scatter_layout = QVBoxLayout(scatter_box)
        self.fig_scatter = Figure(figsize=(5, 3), facecolor="#0d1117")
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        self.canvas_scatter.setMinimumHeight(110)
        scatter_layout.addWidget(self.canvas_scatter)
        top_layout.addWidget(scatter_box, 6)

        # Canvas 2: 95% Confidence Error Ellipse
        ellipse_box = QGroupBox("Horizontal 95% Confidence Error Ellipse")
        ellipse_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        ellipse_layout = QVBoxLayout(ellipse_box)
        self.fig_ellipse = Figure(figsize=(3, 3), facecolor="#0d1117")
        self.canvas_ellipse = FigureCanvas(self.fig_ellipse)
        self.canvas_ellipse.setMinimumHeight(110)
        ellipse_layout.addWidget(self.canvas_ellipse)
        top_layout.addWidget(ellipse_box, 4)

        splitter.addWidget(top_widget)

        # Bottom Section: Interactive Station Residual Table
        bot_box = QGroupBox("Interactive Station Residuals (Toggle Checkbox / Adjust Weight to Re-Fit)")
        bot_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        bot_layout = QVBoxLayout(bot_box)

        self.table_residuals = QTableWidget(0, 11)
        self.table_residuals.setHorizontalHeaderLabels(
            [
                "Active",
                "Station",
                "Comp",
                "Phase",
                "Observed (s)",
                "Calculated (s)",
                "Residual (s)",
                "Weight",
                "Dist (deg)",
                "Azim (deg)",
                "Status",
            ]
        )
        self.table_residuals.setStyleSheet(
            "QTableWidget { background-color: #161b22; color: #e6edf3; gridline-color: #30363d; } "
            "QHeaderView::section { background-color: #21262d; color: #e6edf3; border: 1px solid #30363d; }"
        )
        self.table_residuals.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_residuals.cellClicked.connect(self._on_table_row_clicked)
        self.table_residuals.cellDoubleClicked.connect(self._on_table_row_double_clicked)
        self.table_residuals.itemChanged.connect(self._on_table_item_changed)
        bot_layout.addWidget(self.table_residuals)

        splitter.addWidget(bot_box)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([200, 180])

        v_layout.addWidget(splitter)
        self.tabs.addTab(tab, "Travel Time & Geiger Residuals")

    # ------------------------------------------------------------------
    # Tab 3: Array & FK Analysis
    # ------------------------------------------------------------------

    def _build_tab3_array_fk(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        left_box = QGroupBox("2D Slowness Grid & Array Response Function (ARF)")
        left_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        l_layout = QVBoxLayout(left_box)
        self.fig_fk = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_fk = FigureCanvas(self.fig_fk)
        l_layout.addWidget(self.canvas_fk)
        layout.addWidget(left_box, 1)

        right_box = QGroupBox("Channel Beam Residuals r_m(t) & Coherence CC")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        r_layout = QVBoxLayout(right_box)
        self.fig_beam_res = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_beam_res = FigureCanvas(self.fig_beam_res)
        r_layout.addWidget(self.canvas_beam_res)
        layout.addWidget(right_box, 1)

        self.tabs.addTab(tab, "Array Beamforming & FK Analysis")

    # ------------------------------------------------------------------
    # Tab 4: Spectral Source Parameters (Brune/Boatwright)
    # ------------------------------------------------------------------

    def _build_tab4_spectrum(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        left_box = QGroupBox("Displacement Spectrum & Brune omega^-2 Fit")
        left_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        l_layout = QVBoxLayout(left_box)
        self.fig_spec = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_spec = FigureCanvas(self.fig_spec)
        l_layout.addWidget(self.canvas_spec)
        layout.addWidget(left_box, 1)

        right_box = QGroupBox("2D Misfit Surface & Source Parameters")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        r_layout = QVBoxLayout(right_box)
        self.fig_misfit = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_misfit = FigureCanvas(self.fig_misfit)
        r_layout.addWidget(self.canvas_misfit)

        self.lbl_source_params = QLabel("Source Parameters: M0=--, Mw=--, fc=-- Hz, r=-- m, Δσ=-- bar, R^2=--")
        self.lbl_source_params.setStyleSheet("color: #e6edf3; font-family: Monospace; font-weight: bold;")
        r_layout.addWidget(self.lbl_source_params)

        layout.addWidget(right_box, 1)
        self.tabs.addTab(tab, "Spectral Source Parameters")

    # ------------------------------------------------------------------
    # Tab 5: 3C Particle Motion & Polarization
    # ------------------------------------------------------------------

    def _build_tab5_polarization(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        left_box = QGroupBox("Particle Motion Hodograms (Green -> Red Time Gradient)")
        left_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        l_layout = QVBoxLayout(left_box)
        self.fig_pol = Figure(figsize=(4.5, 3), facecolor="#0d1117")
        self.canvas_pol = FigureCanvas(self.fig_pol)
        l_layout.addWidget(self.canvas_pol)
        layout.addWidget(left_box, 3)

        right_box = QGroupBox("Polarization Metrics, Confidence Cones & Beachball (M-25)")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        r_layout = QVBoxLayout(right_box)
        self.lbl_pol_metrics = QLabel(
            "Backazimuth (BAZ): --\nIncidence Angle: --\nRectilinearity: --\nPlanarity: --\n"
            "Trajectory RMS: --\nConfidence Cones (σ_θ, σ_φ): --"
        )
        self.lbl_pol_metrics.setStyleSheet("color: #e6edf3; font-family: Monospace; font-size: 12px;")
        r_layout.addWidget(self.lbl_pol_metrics)

        self.fig_beachball = Figure(figsize=(3.2, 2.8), facecolor="#0d1117")
        self.canvas_beachball = FigureCanvas(self.fig_beachball)
        self.canvas_beachball.setMinimumHeight(110)
        r_layout.addWidget(self.canvas_beachball, 1)

        self.lbl_fm_summary = QLabel("Focal Mechanism: No beachball solution loaded")
        self.lbl_fm_summary.setStyleSheet("color: #8b949e; font-family: Monospace; font-size: 11px;")
        r_layout.addWidget(self.lbl_fm_summary)

        layout.addWidget(right_box, 2)
        self.tabs.addTab(tab, "3C Particle Motion & Polarization")

    # ------------------------------------------------------------------
    # Tab 6: Instrument Restitution & Deconvolution
    # ------------------------------------------------------------------

    def _build_tab6_restitution(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        left_box = QGroupBox("Instrument Transfer Function |H(f)| & Dispersion")
        left_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        l_layout = QVBoxLayout(left_box)
        self.fig_rest = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_rest = FigureCanvas(self.fig_rest)
        l_layout.addWidget(self.canvas_rest)
        layout.addWidget(left_box, 1)

        right_box = QGroupBox("Deconvolution, SNR & Receiver Function Fit")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        r_layout = QVBoxLayout(right_box)
        self.fig_rf = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_rf = FigureCanvas(self.fig_rf)
        r_layout.addWidget(self.canvas_rf)

        self.lbl_rest_info = QLabel("Restitution / Deconvolution: Idle")
        self.lbl_rest_info.setStyleSheet("color: #e6edf3; font-family: Monospace;")
        r_layout.addWidget(self.lbl_rest_info)

        layout.addWidget(right_box, 1)
        self.tabs.addTab(tab, "Restitution & Deconvolution")

    # ------------------------------------------------------------------
    # Data Population & Rendering
    # ------------------------------------------------------------------

    def display_result(self, result: FitDiagnosticResult) -> None:
        """Display a FitDiagnosticResult, routing to the appropriate tab and refreshing KPIs/plots."""
        self.set_fit_result(result)

    def set_fit_result(self, result: FitDiagnosticResult) -> None:
        """Populate all diagnostic widgets, KPI cards, tables, and canvases with inversion result."""
        self.current_result = result

        # Switch domain combo and tab
        self._block_table_signals = True
        try:
            for idx in range(self.domain_combo.count()):
                if self.domain_combo.itemData(idx) == result.fit_type:
                    self.domain_combo.setCurrentIndex(idx)
                    break

            # Map FitType to tab index
            tab_map = {
                FitType.HYPOCENTER_GEIGER: 1,
                FitType.ORIGIN_TIME: 1,
                FitType.PLANE_WAVE: 2,
                FitType.FK_SEMBLANCE: 2,
                FitType.SP_DIFF_DEPTH: 1,
                FitType.TTFIT_SIMPLEX: 1,
                FitType.BRUNE_SPECTRUM: 3,
                FitType.POLARIZATION_3C: 4,
                FitType.RESTITUTION_FIT: 5,
                FitType.RECEIVER_FUNC: 5,
                FitType.SURFACE_DISPERSION: 5,
            }
            target_tab = tab_map.get(result.fit_type, 0)
            if self.tabs.currentIndex() != target_tab:
                self.tabs.setCurrentIndex(target_tab)

            # Update KPI cards
            time_domains = {
                FitType.HYPOCENTER_GEIGER,
                FitType.ORIGIN_TIME,
                FitType.SP_DIFF_DEPTH,
                FitType.TTFIT_SIMPLEX,
            }
            if result.fit_type in time_domains:
                rmse_str = f"{result.rmse:.3f} s"
                rmse_ok = result.rmse < 0.5
            elif result.fit_type == FitType.BRUNE_SPECTRUM:
                rmse_str = f"{result.rmse:.3f} log10"
                rmse_ok = result.rmse < 0.5
            else:
                rmse_str = f"{result.rmse:.3f}" if (result.rmse == 0.0 or abs(result.rmse) >= 0.001) else f"{result.rmse:.2e}"
                rmse_ok = result.rmse < 0.5
            self.kpi_rmse.set_value(rmse_str, "#3fb950" if rmse_ok else "#f85149")
            self.kpi_chi2.set_value(f"{result.chi2_reduced:.2f}", "#3fb950" if result.chi2_reduced < 2.0 else "#d29922")
            n_tot = len(result.residuals)
            outlier_color = "#3fb950" if result.outlier_count == 0 else "#f85149"
            self.kpi_outliers.set_value(f"{result.outlier_count} / {n_tot}", outlier_color)

            status_str = f"Converged ({result.iterations} it)" if result.converged else "Diverged / Stopped"
            self.kpi_status.set_value(status_str, "#3fb950" if result.converged else "#f85149")

            cond_str = f"{result.condition_number:.1f}" if result.condition_number < 1e5 else "Inf"
            cond_color = "#3fb950" if result.condition_number < 100.0 else "#d29922"
            self.kpi_cond.set_value(cond_str, cond_color)

            badge_text = f"CONVERGED ({result.iterations} it)" if result.converged else "NOT CONVERGED"
            badge_color = "#3fb950" if result.converged else "#f85149"
            self.lbl_status_badge.setText(badge_text)
            self.lbl_status_badge.setStyleSheet(
                f"background-color: #21262d; color: {badge_color}; border: 1px solid {badge_color}; "
                "border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px;"
            )

            # Update Tab 1: Parameters Table & Correlation Matrix
            self._update_tab1_overview(result)

            # Update Tab 2: Travel Time & Residual Table
            self._update_tab2_travel_time(result)

            # Render real domain visuals for tabs 3-6
            self._update_tab3_array_fk(result)
            self._update_tab4_spectrum(result)
            self._update_tab5_polarization(result)
            self._update_tab6_restitution(result)

        finally:
            self._block_table_signals = False

    def _update_tab1_overview(self, result: FitDiagnosticResult) -> None:
        """Update Tab 1 parameter table and correlation heatmap."""
        self.table_params.setRowCount(len(result.parameters))
        for r_i, (pname, pinfo) in enumerate(result.parameters.items()):
            ci_str = f"[{pinfo.confidence_interval_95[0]:.3f}, {pinfo.confidence_interval_95[1]:.3f}]"
            unit_str = f" {pinfo.unit}" if pinfo.unit else ""
            vals = [
                pname,
                f"{pinfo.value:.4f}{unit_str}",
                f"± {pinfo.standard_error:.4f}{unit_str}",
                ci_str,
            ]
            for c_i, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table_params.setItem(r_i, c_i, item)

        self.lbl_normality.setText(
            f"Residual Normality: Skewness={result.skewness:+.3f}, "
            f"Kurtosis={result.kurtosis:+.3f}, Jarque-Bera p={result.normality_p_value:.4f} "
            f"({'Gaussian' if result.normality_p_value > 0.05 else 'Non-Gaussian'})"
        )

        # Plot Correlation Matrix Heatmap
        self.fig_corr.clear()
        ax = self.fig_corr.add_subplot(111)
        ax.set_facecolor("#0d1117")

        if result.correlation_matrix is not None and result.correlation_matrix.size > 0:
            corr = result.correlation_matrix
            pnames = list(result.parameters.keys()) or [f"P{i+1}" for i in range(corr.shape[0])]
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(len(pnames)))
            ax.set_yticks(range(len(pnames)))
            ax.set_xticklabels(pnames, color="#8b949e", fontsize=8, rotation=30)
            ax.set_yticklabels(pnames, color="#8b949e", fontsize=8)
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="#ffffff", fontsize=8)
            self.fig_corr.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "No correlation matrix available", ha="center", va="center", color="#8b949e")

        safe_tight_layout(self.fig_corr, self.canvas_corr)
        self.canvas_corr.draw()

    def _update_tab2_travel_time(self, result: FitDiagnosticResult) -> None:
        """Update Tab 2 residual scatter subplots, error ellipse, and interactive residual table."""
        residuals = result.residuals

        # 1. Update Table
        self.table_residuals.setRowCount(len(residuals))
        for r_i, entry in enumerate(residuals):
            # Column 0: Active Checkbox
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Checked if entry.weight > 0.0 else Qt.CheckState.Unchecked)
            self.table_residuals.setItem(r_i, 0, chk_item)

            vals = [
                entry.station,
                entry.channel or "Z",
                entry.phase,
                f"{entry.observed:.3f}",
                f"{entry.calculated:.3f}",
                f"{entry.residual:+.3f}",
                f"{entry.weight:.2f}",
                f"{entry.distance_deg:.2f}",
                f"{entry.azimuth_deg:.1f}",
                "OUTLIER (>3σ)" if entry.is_outlier else "OK",
            ]
            for c_i, v in enumerate(vals, start=1):
                item = QTableWidgetItem(v)
                if c_i == 7:  # Weight is editable
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable)
                else:
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

                if c_i == 6:  # Residual color
                    res_color = QColor("#f85149" if entry.is_outlier else "#3fb950")
                    item.setForeground(res_color)
                elif c_i == 10:  # Status column
                    if entry.is_outlier:
                        item.setForeground(QColor("#f85149"))
                        item.setFont(QFont("Monospace", 9, QFont.Weight.Bold))
                    else:
                        item.setForeground(QColor("#3fb950"))
                self.table_residuals.setItem(r_i, c_i, item)

        # 2. Render Residual Scatter (Distance and Azimuth)
        self.fig_scatter.clear()
        ax1 = self.fig_scatter.add_subplot(121)
        ax2 = self.fig_scatter.add_subplot(122)
        for ax in (ax1, ax2):
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        dists = [e.distance_deg for e in residuals]
        azims = [e.azimuth_deg for e in residuals]
        res_vals = [e.residual for e in residuals]
        colors = ["#f85149" if e.is_outlier else "#3fb950" for e in residuals]

        if dists and res_vals:
            ax1.axhline(0.0, color="#8b949e", linestyle="--", linewidth=0.8)
            ax1.scatter(dists, res_vals, c=colors, edgecolors="#ffffff", s=40, zorder=3)
            ax1.set_xlabel("Epicentral Distance (deg)", color="#8b949e", fontsize=8)
            ax1.set_ylabel("Residual (s)", color="#8b949e", fontsize=8)

            ax2.axhline(0.0, color="#8b949e", linestyle="--", linewidth=0.8)
            ax2.scatter(azims, res_vals, c=colors, edgecolors="#ffffff", s=40, zorder=3)
            ax2.set_xlabel("Backazimuth (deg)", color="#8b949e", fontsize=8)
            ax2.set_ylabel("Residual (s)", color="#8b949e", fontsize=8)

        safe_tight_layout(self.fig_scatter, self.canvas_scatter)
        self.canvas_scatter.draw()

        # 3. Render 95% Error Ellipse
        self.fig_ellipse.clear()
        ax_el = self.fig_ellipse.add_subplot(111)
        ax_el.set_facecolor("#0d1117")
        ax_el.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_el.spines.values():
            spine.set_color("#30363d")

        if result.confidence_ellipse:
            el = result.confidence_ellipse
            a, b, theta = el.semi_major_km, el.semi_minor_km, el.strike_deg

            # Parametric ellipse in local km: (x = East, y = North)
            # strike azimuth theta is measured clockwise from North (+y) towards East (+x)
            # Angle in Cartesian plane: alpha = 90 - theta
            alpha_rad = math.radians(90.0 - theta)
            phi = np.linspace(0, 2 * np.pi, 100)
            x_raw = a * np.cos(phi)
            y_raw = b * np.sin(phi)
            # Rotate by alpha
            x_rot = x_raw * np.cos(alpha_rad) - y_raw * np.sin(alpha_rad)
            y_rot = x_raw * np.sin(alpha_rad) + y_raw * np.cos(alpha_rad)

            ax_el.plot(x_rot, y_rot, color="#58a6ff", linewidth=2.0, label="95% Ellipse")
            ax_el.fill(x_rot, y_rot, color="#58a6ff", alpha=0.15)
            # Plot major axis line
            x_ax = np.array([-a, a]) * np.cos(alpha_rad)
            y_ax = np.array([-a, a]) * np.sin(alpha_rad)
            ax_el.plot(x_ax, y_ax, color="#d29922", linestyle="--", linewidth=1.2, label=f"Major ({a:.1f} km)")
            ax_el.plot(0, 0, marker="o", color="#f85149", markersize=6)

            max_range = max(a * 1.3, 10.0)
            ax_el.set_xlim(-max_range, max_range)
            ax_el.set_ylim(-max_range, max_range)
            ax_el.set_aspect("equal", adjustable="box")
            ax_el.set_title(f"a={a:.1f} km, b={b:.1f} km, θ={theta:.1f}°", color="#e6edf3", fontsize=9)
            ax_el.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        else:
            ax_el.text(0.5, 0.5, "No Error Ellipse Data", ha="center", va="center", color="#8b949e")

        safe_tight_layout(self.fig_ellipse, self.canvas_ellipse)
        self.canvas_ellipse.draw()

    @staticmethod
    def _safe_curve_array(curves: Dict[str, Any], key: str) -> np.ndarray:
        val = curves.get(key) if curves else None
        if val is None:
            return np.array([], dtype=np.float64)
        arr = np.asarray(val, dtype=np.float64)
        return arr

    @staticmethod
    def _style_dark_axes(ax: Any) -> None:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    @classmethod
    def _draw_empty_placeholder(cls, fig: Figure, canvas: FigureCanvas, message: str) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        cls._style_dark_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#8b949e",
            fontsize=9,
            transform=ax.transAxes,
        )
        safe_tight_layout(fig, canvas)
        canvas.draw()

    @staticmethod
    def _add_colored_hodogram(
        ax: Any,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        cmap: LinearSegmentedColormap,
    ) -> None:
        pts = np.column_stack([x_vals, y_vals]).reshape(-1, 1, 2)
        if pts.shape[0] < 2:
            return
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=Normalize(0.0, 1.0), linewidth=1.8, zorder=3)
        lc.set_array(np.linspace(0.0, 1.0, segs.shape[0]))
        ax.add_collection(lc)
        ax.scatter([x_vals[0]], [y_vals[0]], color="#2ea043", s=28, zorder=4, label="Start")
        ax.scatter([x_vals[-1]], [y_vals[-1]], color="#f85149", s=28, zorder=4, label="End")
        ax.axhline(0.0, color="#30363d", linestyle="--", linewidth=0.7, zorder=1)
        ax.axvline(0.0, color="#30363d", linestyle="--", linewidth=0.7, zorder=1)
        x_span = float(np.nanmax(x_vals) - np.nanmin(x_vals))
        y_span = float(np.nanmax(y_vals) - np.nanmin(y_vals))
        max_lim = max(
            float(np.nanmax(np.abs(x_vals))) if x_vals.size else 0.0,
            float(np.nanmax(np.abs(y_vals))) if y_vals.size else 0.0,
            1e-9,
        )
        if x_span < 1e-12 and y_span < 1e-12:
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
        else:
            pad = max_lim * 1.15
            ax.set_xlim(-pad, pad)
            ax.set_ylim(-pad, pad)
        ax.set_aspect("equal", adjustable="box")

    def _update_tab3_array_fk(self, result: FitDiagnosticResult) -> None:
        """Render Tab 3 real 2D FK grid, ARF contours, calibration vector, and beam residuals."""
        curves = result.overlay_curves or {}
        ctx = result.context_data or {}

        s_axis = self._safe_curve_array(curves, "s_axis")
        fk_grid = self._safe_curve_array(curves, "fk_grid")
        arf_grid = self._safe_curve_array(curves, "arf_grid")

        has_fk = fk_grid.ndim == 2 and fk_grid.shape[0] >= 2 and fk_grid.shape[1] >= 2 and np.any(np.isfinite(fk_grid))
        has_arf = arf_grid.ndim == 2 and arf_grid.shape[0] >= 2 and arf_grid.shape[1] >= 2 and np.any(np.isfinite(arf_grid))

        if has_fk or has_arf:
            self.fig_fk.clear()
            ax_fk = self.fig_fk.add_subplot(111)
            self._style_dark_axes(ax_fk)

            if s_axis.ndim == 1 and s_axis.size >= 2 and np.all(np.isfinite(s_axis)):
                s_min, s_max = float(s_axis[0]), float(s_axis[-1])
            else:
                s_min, s_max = -25.0, 25.0
            extent = [s_min, s_max, s_min, s_max]

            base_grid = fk_grid if has_fk else arf_grid
            cbar_label = "Semblance / Power" if has_fk else "ARF |A(k)|²"
            vmin = float(np.nanmin(base_grid))
            vmax = float(np.nanmax(base_grid))
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0

            im = ax_fk.imshow(
                base_grid,
                extent=extent,
                origin="lower",
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            cbar = self.fig_fk.colorbar(im, ax=ax_fk, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, color="#8b949e", fontsize=8)
            cbar.ax.tick_params(colors="#8b949e", labelsize=7)

            if has_arf:
                arf_min = float(np.nanmin(arf_grid))
                arf_max = float(np.nanmax(arf_grid))
                if math.isfinite(arf_min) and math.isfinite(arf_max) and arf_max > arf_min:
                    sx_arf = np.linspace(s_min, s_max, arf_grid.shape[1])
                    sy_arf = np.linspace(s_min, s_max, arf_grid.shape[0])
                    levels = [lv for lv in (0.25, 0.5, 0.75) if arf_min < lv < arf_max]
                    if not levels:
                        levels = list(np.linspace(arf_min + 0.25 * (arf_max - arf_min), arf_max - 0.1 * (arf_max - arf_min), 3))
                    if levels:
                        ax_fk.contour(
                            sx_arf,
                            sy_arf,
                            arf_grid,
                            levels=levels,
                            colors=["#79c0ff", "#58a6ff", "#ffffff"][: len(levels)],
                            linestyles="--",
                            linewidths=0.9,
                            alpha=0.85,
                        )
                        ax_fk.plot([], [], color="#58a6ff", linestyle="--", linewidth=1.0, label="2D ARF Contours")

            calib_applied = bool(ctx.get("calib_applied", False))
            if calib_applied and ("raw_sx" in ctx and "raw_sy" in ctx) and ("cal_sx" in ctx and "cal_sy" in ctx):
                raw_sx = float(ctx["raw_sx"])
                raw_sy = float(ctx["raw_sy"])
                cal_sx = float(ctx["cal_sx"])
                cal_sy = float(ctx["cal_sy"])
                ax_fk.scatter([raw_sx], [raw_sy], color="#d29922", edgecolors="#ffffff", s=55, zorder=5, label="Raw Peak")
                ax_fk.scatter([cal_sx], [cal_sy], color="#3fb950", marker="*", s=110, zorder=6, label="Calibrated")
                if math.hypot(cal_sx - raw_sx, cal_sy - raw_sy) > 1e-9:
                    ax_fk.annotate(
                        "",
                        xy=(cal_sx, cal_sy),
                        xytext=(raw_sx, raw_sy),
                        arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2.0, shrinkA=2, shrinkB=2),
                        zorder=5,
                    )
            elif "cal_sx" in ctx and "cal_sy" in ctx:
                ax_fk.scatter([float(ctx["cal_sx"])], [float(ctx["cal_sy"])], color="#3fb950", marker="*", s=100, zorder=5, label="Peak Slowness")
            elif "raw_sx" in ctx and "raw_sy" in ctx:
                ax_fk.scatter([float(ctx["raw_sx"])], [float(ctx["raw_sy"])], color="#3fb950", marker="*", s=100, zorder=5, label="Peak Slowness")

            slo_disp = ctx.get("cal_slowness" if calib_applied else "raw_slowness")
            baz_disp = ctx.get("cal_baz" if calib_applied else "raw_baz")
            if slo_disp is not None and baz_disp is not None:
                cal_tag = " [Calibrated]" if calib_applied else ""
                ax_fk.set_title(
                    f"s={float(slo_disp):.2f} s/deg, BAZ={float(baz_disp):.1f}°{cal_tag}",
                    color="#e6edf3",
                    fontsize=9,
                )

            ax_fk.set_xlabel("sx (s/deg)", color="#8b949e", fontsize=8)
            ax_fk.set_ylabel("sy (s/deg)", color="#8b949e", fontsize=8)
            handles, _ = ax_fk.get_legend_handles_labels()
            if handles:
                ax_fk.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_fk, self.canvas_fk)
            self.canvas_fk.draw()
        else:
            self._draw_empty_placeholder(
                self.fig_fk,
                self.canvas_fk,
                "No FK / Slowness Grid Data\nRun Array FK or Plane-Wave Fit",
            )

        # Right panel: Beamforming waveform, channel residuals, and CC_m coherence
        beam_time = self._safe_curve_array(curves, "beam_time")
        beam_wf = self._safe_curve_array(curves, "beam_waveform")
        mean_res = self._safe_curve_array(curves, "mean_residual")
        chan_res = self._safe_curve_array(curves, "channel_residuals")

        raw_cc = ctx.get("channel_cc", [])
        channel_cc = [float(v) for v in raw_cc] if isinstance(raw_cc, (list, tuple, np.ndarray)) else []
        raw_names = ctx.get("channel_names", [])
        channel_names = [str(n) for n in raw_names] if isinstance(raw_names, (list, tuple)) else []
        if not channel_cc and result.fit_type in (FitType.FK_SEMBLANCE, FitType.PLANE_WAVE) and result.residuals:
            for ent in result.residuals:
                if "cc" in ent.extra_attributes:
                    channel_cc.append(float(ent.extra_attributes["cc"]))
                    channel_names.append(ent.station or ent.identifier)

        has_beam = beam_time.ndim == 1 and beam_wf.ndim == 1 and beam_time.size >= 2 and beam_time.size == beam_wf.size
        has_cc = len(channel_cc) > 0

        if has_beam or has_cc:
            self.fig_beam_res.clear()
            if has_beam and has_cc:
                ax_b = self.fig_beam_res.add_subplot(211)
                ax_cc = self.fig_beam_res.add_subplot(212)
            elif has_beam:
                ax_b = self.fig_beam_res.add_subplot(111)
                ax_cc = None
            else:
                ax_b = None
                ax_cc = self.fig_beam_res.add_subplot(111)

            if ax_b is not None:
                self._style_dark_axes(ax_b)
                if chan_res.ndim == 2 and chan_res.shape[1] == beam_time.size:
                    poor_labeled = False
                    for idx in range(chan_res.shape[0]):
                        cc_i = channel_cc[idx] if idx < len(channel_cc) else 1.0
                        if cc_i < 0.70:
                            lbl = "Poor Channel r_m(t) (CC<0.70)" if not poor_labeled else None
                            ax_b.plot(beam_time, chan_res[idx], color="#f85149", alpha=0.75, linewidth=0.9, label=lbl)
                            poor_labeled = True
                        else:
                            ax_b.plot(beam_time, chan_res[idx], color="#8b949e", alpha=0.22, linewidth=0.7)
                ax_b.plot(beam_time, beam_wf, color="#3fb950", linewidth=1.5, label="Formed Beam b(t)")
                if mean_res.ndim == 1 and mean_res.size == beam_time.size:
                    ax_b.plot(beam_time, mean_res, color="#d29922", linestyle="--", linewidth=1.2, label="Mean Residual RMS")
                ax_b.set_xlabel("Relative Time (s)", color="#8b949e", fontsize=8)
                ax_b.set_ylabel("Amplitude", color="#8b949e", fontsize=8)
                ax_b.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

            if ax_cc is not None:
                self._style_dark_axes(ax_cc)
                if len(channel_names) != len(channel_cc):
                    channel_names = [f"Ch{i + 1}" for i in range(len(channel_cc))]
                x_idx = np.arange(len(channel_cc))
                bar_colors = ["#f85149" if cc < 0.70 else "#3fb950" for cc in channel_cc]
                ax_cc.bar(x_idx, channel_cc, color=bar_colors, edgecolor="#30363d", width=0.7)
                ax_cc.axhline(0.70, color="#f85149", linestyle="--", linewidth=1.0, label="CC = 0.70")
                ax_cc.axhline(0.0, color="#30363d", linewidth=0.7)
                ax_cc.set_xticks(x_idx)
                ax_cc.set_xticklabels(channel_names, color="#8b949e", fontsize=7, rotation=25)
                ax_cc.set_ylim(-1.05, 1.05)
                ax_cc.set_ylabel("Coherence CC_m", color="#8b949e", fontsize=8)
                ax_cc.set_title("Channel Coherence CC_m (Red: CC < 0.70)", color="#e6edf3", fontsize=8)
                ax_cc.legend(loc="lower right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

            safe_tight_layout(self.fig_beam_res, self.canvas_beam_res)
            self.canvas_beam_res.draw()
        else:
            self._draw_empty_placeholder(
                self.fig_beam_res,
                self.canvas_beam_res,
                "No Beam Residual / Coherence Data Available",
            )

    def _update_tab4_spectrum(self, result: FitDiagnosticResult) -> None:
        """Render Tab 4 real Brune/Boatwright displacement spectrum and 2D misfit contour."""
        curves = result.overlay_curves or {}
        ctx = result.context_data or {}

        freqs = self._safe_curve_array(curves, "freqs")
        obs_spec = self._safe_curve_array(curves, "obs_spec")
        model_spec = self._safe_curve_array(curves, "model_spec")
        noise_spec = self._safe_curve_array(curves, "noise_spec")

        def _val(key: str, default: float = float("nan")) -> float:
            if key in ctx and ctx[key] is not None:
                return float(ctx[key])
            if key in result.parameters:
                return float(result.parameters[key].value)
            return default

        omega0 = _val("omega0")
        fc = _val("fc")
        t_star = _val("t_star")
        m0 = _val("m0")
        mw = _val("mw")
        r0_m = _val("r0_m")
        stress_drop_bar = _val("stress_drop_bar")
        r2_val = float(ctx.get("r_squared", result.r_squared)) if ("r_squared" in ctx or result.r_squared != 0.0) else float("nan")
        model_type = str(ctx.get("model_type", "Brune")).title()

        has_spec = freqs.ndim == 1 and freqs.size >= 2 and np.any(freqs > 0.0) and (
            (obs_spec.size == freqs.size and np.any(obs_spec > 0.0))
            or (model_spec.size == freqs.size and np.any(model_spec > 0.0))
        )

        if has_spec:
            self.fig_spec.clear()
            ax_s = self.fig_spec.add_subplot(111)
            self._style_dark_axes(ax_s)

            pos_f = np.isfinite(freqs) & (freqs > 0.0)
            plotted_any = False
            if obs_spec.size == freqs.size:
                m_obs = pos_f & np.isfinite(obs_spec) & (obs_spec > 0.0)
                if np.any(m_obs):
                    ax_s.loglog(freqs[m_obs], obs_spec[m_obs], "o", color="#8b949e", markersize=3, alpha=0.85, label="Observed U(f)")
                    plotted_any = True
            if model_spec.size == freqs.size:
                m_mod = pos_f & np.isfinite(model_spec) & (model_spec > 0.0)
                if np.any(m_mod):
                    fc_lbl = f" (fc={fc:.2f} Hz)" if math.isfinite(fc) else ""
                    ax_s.loglog(freqs[m_mod], model_spec[m_mod], color="#3fb950", linewidth=2.0, label=f"{model_type} Fit{fc_lbl}")
                    plotted_any = True
            if noise_spec.size == freqs.size:
                m_noi = pos_f & np.isfinite(noise_spec) & (noise_spec > 0.0)
                if np.any(m_noi):
                    ax_s.loglog(freqs[m_noi], noise_spec[m_noi], color="#f85149", linestyle=":", linewidth=1.3, label="Noise Floor")
                    plotted_any = True

            if math.isfinite(fc) and fc > 0.0 and plotted_any:
                ax_s.axvline(fc, color="#d29922", linestyle="--", linewidth=1.1, label=f"fc = {fc:.2f} Hz")

            if plotted_any:
                ann_lines = []
                if math.isfinite(omega0):
                    ann_lines.append(f"Ω0 = {omega0:.3g} nm·s")
                if math.isfinite(fc):
                    ann_lines.append(f"fc = {fc:.3f} Hz")
                if math.isfinite(t_star):
                    ann_lines.append(f"t* = {t_star:.4f} s")
                if math.isfinite(m0):
                    ann_lines.append(f"M0 = {m0:.2e} Nm")
                if math.isfinite(mw):
                    ann_lines.append(f"Mw = {mw:.2f}")
                if math.isfinite(r0_m):
                    ann_lines.append(f"r0 = {r0_m:.1f} m")
                if math.isfinite(stress_drop_bar):
                    ann_lines.append(f"Δσ = {stress_drop_bar:.2f} bar")
                if ann_lines:
                    ax_s.text(
                        0.98,
                        0.97,
                        "\n".join(ann_lines),
                        transform=ax_s.transAxes,
                        ha="right",
                        va="top",
                        color="#e6edf3",
                        fontsize=7,
                        family="monospace",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="#161b22", edgecolor="#30363d", alpha=0.9),
                    )
                ax_s.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
                ax_s.set_ylabel("Displacement (nm·s)", color="#8b949e", fontsize=8)
                ax_s.legend(loc="lower left", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
                safe_tight_layout(self.fig_spec, self.canvas_spec)
                self.canvas_spec.draw()
            else:
                self._draw_empty_placeholder(
                    self.fig_spec,
                    self.canvas_spec,
                    "No Positive Spectral Samples Available",
                )
        else:
            self._draw_empty_placeholder(
                self.fig_spec,
                self.canvas_spec,
                "No Spectral Fit Curves Available\nRun Brune / Boatwright Inversion",
            )

        # Right panel: 2D Misfit Contour over (fc_grid, omega0_grid)
        fc_grid = self._safe_curve_array(curves, "fc_grid")
        w0_grid = self._safe_curve_array(curves, "omega0_grid")
        misfit_grid = self._safe_curve_array(curves, "misfit_grid")

        valid_misfit = False
        misfit_plot = misfit_grid
        if fc_grid.ndim == 1 and w0_grid.ndim == 1 and fc_grid.size >= 2 and w0_grid.size >= 2 and misfit_grid.ndim == 2:
            if misfit_grid.shape == (w0_grid.size, fc_grid.size):
                misfit_plot = misfit_grid
                valid_misfit = True
            elif misfit_grid.shape == (fc_grid.size, w0_grid.size):
                misfit_plot = misfit_grid.T
                valid_misfit = True

        if valid_misfit and np.any(np.isfinite(misfit_plot)):
            self.fig_misfit.clear()
            ax_m = self.fig_misfit.add_subplot(111)
            self._style_dark_axes(ax_m)

            x_mesh, y_mesh = np.meshgrid(fc_grid, w0_grid)
            finite_vals = misfit_plot[np.isfinite(misfit_plot)]
            if finite_vals.size > 0 and float(np.max(finite_vals)) > float(np.min(finite_vals)):
                im = ax_m.contourf(x_mesh, y_mesh, misfit_plot, levels=15, cmap="viridis")
            else:
                im = ax_m.pcolormesh(x_mesh, y_mesh, misfit_plot, cmap="viridis", shading="auto")
            cbar = self.fig_misfit.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
            cbar.set_label("Log10 Misfit", color="#8b949e", fontsize=8)
            cbar.ax.tick_params(colors="#8b949e", labelsize=7)

            if math.isfinite(fc) and math.isfinite(omega0):
                ax_m.plot(fc, omega0, "r*", markersize=11, markeredgecolor="#ffffff", markeredgewidth=0.6, label="Optimal (fc, Ω0)")
                ax_m.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

            ax_m.set_xlabel("Corner Freq fc (Hz)", color="#8b949e", fontsize=8)
            ax_m.set_ylabel("Plateau Ω0 (nm·s)", color="#8b949e", fontsize=8)
            safe_tight_layout(self.fig_misfit, self.canvas_misfit)
            self.canvas_misfit.draw()
        else:
            self._draw_empty_placeholder(
                self.fig_misfit,
                self.canvas_misfit,
                "No 2D Spectral Misfit Contour Available",
            )

        if any(math.isfinite(v) for v in (omega0, fc, m0, mw, r0_m, stress_drop_bar)):
            self.lbl_source_params.setText(
                f"Source Params ({model_type}): Ω0={omega0:.3g} nm·s, fc={fc:.3f} Hz, t*={t_star:.4f} s, "
                f"M0={m0:.2e} Nm, Mw={mw:.2f}, r0={r0_m:.1f} m, Δσ={stress_drop_bar:.2f} bar, R²={r2_val:.3f}"
            )
        else:
            self.lbl_source_params.setText(
                "Source Parameters: M0=--, Mw=--, fc=-- Hz, r=-- m, Δσ=-- bar, R^2=--"
            )

    def display_focal_mechanism(self, fm_result: Dict[str, Any]) -> None:
        """Public entry point to render a solved double-couple focal mechanism beachball in Tab 5."""
        self._last_fm_result = dict(fm_result) if fm_result else None
        self._render_beachball(self._last_fm_result)
        self.tabs.setCurrentIndex(4)

    def _render_beachball(self, fm_result: Optional[Dict[str, Any]]) -> None:
        """Render M-25 double-couple equal-area Schmidt beachball on self.fig_beachball."""
        if not fm_result or not all(k in fm_result for k in ("strike", "dip", "rake")):
            self._draw_empty_placeholder(
                self.fig_beachball,
                self.canvas_beachball,
                "No Focal Mechanism Beachball\nRun First-Motion Solver (M-25)",
            )
            self.lbl_fm_summary.setText("Focal Mechanism: No beachball solution loaded")
            return

        strike = float(fm_result["strike"])
        dip = float(fm_result["dip"])
        rake = float(fm_result["rake"])
        bb = compute_beachball_contours(strike, dip, rake, n_grid=121)

        self.fig_beachball.clear()
        ax_bb = self.fig_beachball.add_subplot(111)
        ax_bb.set_facecolor("#0d1117")

        ax_bb.contourf(
            bb["x_grid"],
            bb["y_grid"],
            bb["sign_grid"],
            levels=[-1.5, 0.0, 1.5],
            colors=["#e6edf3", "#1f6feb"],
            zorder=1,
        )
        ax_bb.plot(bb["plane1_x"], bb["plane1_y"], color="#0d1117", linewidth=1.5, zorder=2)
        ax_bb.plot(bb["plane2_x"], bb["plane2_y"], color="#0d1117", linewidth=1.5, zorder=2)
        ax_bb.add_patch(Circle((0.0, 0.0), 1.0, fill=False, edgecolor="#8b949e", linewidth=1.6, zorder=3))

        # Plot station polarity observations if provided
        stations = fm_result.get("station_matches") or fm_result.get("polarities") or []
        for obs in stations:
            if isinstance(obs, dict):
                az = obs.get("azimuth", obs.get("azimuth_deg"))
                ih = obs.get("takeoff", obs.get("takeoff_deg"))
                pol = obs.get("observed", obs.get("polarity", 0))
            elif isinstance(obs, (tuple, list)) and len(obs) >= 3:
                az, ih, pol = obs[0], obs[1], obs[2]
            else:
                continue
            if az is None or ih is None or not pol:
                continue
            px, py = project_schmidt(float(az), 90.0 - float(ih))
            if int(pol) > 0:
                ax_bb.scatter([px], [py], marker="+", color="#f85149", s=26, linewidths=1.2, zorder=4)
            else:
                ax_bb.scatter([px], [py], marker="o", facecolors="none", edgecolors="#2ea043", s=20, linewidths=1.0, zorder=4)

        p_xy = bb["p_xy"]
        t_xy = bb["t_xy"]
        ax_bb.scatter([p_xy[0]], [p_xy[1]], color="#f85149", marker="o", s=34, zorder=5)
        ax_bb.text(p_xy[0], p_xy[1], " P", color="#f85149", fontsize=8, fontweight="bold", zorder=6)
        ax_bb.scatter([t_xy[0]], [t_xy[1]], color="#3fb950", marker="^", s=34, zorder=5)
        ax_bb.text(t_xy[0], t_xy[1], " T", color="#3fb950", fontsize=8, fontweight="bold", zorder=6)

        ax_bb.set_xlim(-1.14, 1.14)
        ax_bb.set_ylim(-1.14, 1.14)
        ax_bb.set_aspect("equal", adjustable="box")
        ax_bb.axis("off")

        s2, d2, r2 = bb["plane2"]
        misfit_ratio = fm_result.get("misfit_ratio")
        n_obs = fm_result.get("n_obs")
        n_misfit = fm_result.get("n_misfit")
        ax_bb.set_title(
            f"NP1: {strike:.0f}/{dip:.0f}/{rake:.0f}° | NP2: {s2:.0f}/{d2:.0f}/{r2:.0f}°",
            color="#e6edf3",
            fontsize=8,
        )
        summary_parts = [f"NP1 ({strike:.1f}°, {dip:.1f}°, {rake:.1f}°) | NP2 ({s2:.1f}°, {d2:.1f}°, {r2:.1f}°)"]
        if misfit_ratio is not None:
            if n_obs is not None and n_misfit is not None:
                summary_parts.append(f"Misfit: {int(n_misfit)}/{int(n_obs)} ({float(misfit_ratio):.2%})")
            else:
                summary_parts.append(f"Misfit: {float(misfit_ratio):.2%}")
        self.lbl_fm_summary.setText(" | ".join(summary_parts))

        safe_tight_layout(self.fig_beachball, self.canvas_beachball)
        self.canvas_beachball.draw()

    def _update_tab5_polarization(self, result: FitDiagnosticResult) -> None:
        """Render Tab 5 real 3C hodograms (Green->Yellow->Red), metrics, and M-25 beachball."""
        curves = result.overlay_curves or {}
        ctx = result.context_data or {}

        z = self._safe_curve_array(curves, "z")
        n = self._safe_curve_array(curves, "n")
        e = self._safe_curve_array(curves, "e")
        q = self._safe_curve_array(curves, "q")

        baz = ctx.get("baz")
        inc = ctx.get("incidence")
        rect = ctx.get("rectilinearity")
        plan = ctx.get("planarity")
        rms_pol = ctx.get("rms_pol")
        sig_th = ctx.get("sigma_theta_deg")
        sig_ph = ctx.get("sigma_phi_deg")

        has_ne = n.ndim == 1 and e.ndim == 1 and n.size >= 2 and n.size == e.size
        if has_ne:
            gyr_cmap = LinearSegmentedColormap.from_list("gyr", ["#2ea043", "#f1e05a", "#f85149"])
            self.fig_pol.clear()
            ax1 = self.fig_pol.add_subplot(121)
            ax2 = self.fig_pol.add_subplot(122)
            self._style_dark_axes(ax1)
            self._style_dark_axes(ax2)

            self._add_colored_hodogram(ax1, e, n, gyr_cmap)
            if baz is not None and math.isfinite(float(baz)):
                az_rad = math.radians(float(baz))
                lim_h = max(abs(ax1.get_xlim()[0]), abs(ax1.get_xlim()[1]), 1e-6)
                ax1.annotate(
                    "",
                    xy=(0.65 * lim_h * math.sin(az_rad), 0.65 * lim_h * math.cos(az_rad)),
                    xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.4),
                    zorder=5,
                )
            ax1.set_xlabel("East", color="#8b949e", fontsize=8)
            ax1.set_ylabel("North", color="#8b949e", fontsize=8)
            ax1.set_title("N-E (Horizontal)", color="#e6edf3", fontsize=8)

            if z.ndim == 1 and z.size == n.size:
                if baz is not None and math.isfinite(float(baz)):
                    baz_rad = math.radians(float(baz))
                    r_arr = -(n * math.cos(baz_rad) + e * math.sin(baz_rad))
                    x2_arr = r_arr
                    x2_lbl = "Radial (R)"
                    t2_lbl = "Z-R (Vertical-Radial)"
                elif q.ndim == 1 and q.size == z.size:
                    x2_arr = q
                    x2_lbl = "Q (Transverse-V)"
                    t2_lbl = "Z-Q Hodogram"
                else:
                    x2_arr = n
                    x2_lbl = "North"
                    t2_lbl = "Z-N (Vertical)"
                self._add_colored_hodogram(ax2, x2_arr, z, gyr_cmap)
                ax2.set_xlabel(x2_lbl, color="#8b949e", fontsize=8)
                ax2.set_ylabel("Vertical (Z)", color="#8b949e", fontsize=8)
                ax2.set_title(t2_lbl, color="#e6edf3", fontsize=8)
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.text(0.5, 0.5, "No Z Component", ha="center", va="center", color="#8b949e", transform=ax2.transAxes)

            safe_tight_layout(self.fig_pol, self.canvas_pol)
            self.canvas_pol.draw()
        else:
            self._draw_empty_placeholder(
                self.fig_pol,
                self.canvas_pol,
                "No 3C Polarization Hodogram Data Available",
            )

        if any(v is not None for v in (baz, inc, rect, plan, rms_pol, sig_th, sig_ph)):
            baz_s = f"{float(baz):.1f}°" if baz is not None else "--"
            inc_s = f"{float(inc):.1f}°" if inc is not None else "--"
            sth_s = f"{float(sig_th):.2f}°" if sig_th is not None else "--"
            sph_s = f"{float(sig_ph):.2f}°" if sig_ph is not None else "--"
            rect_s = f"{float(rect):.3f}" if rect is not None else "--"
            plan_s = f"{float(plan):.3f}" if plan is not None else "--"
            rms_s = f"{float(rms_pol):.4g}" if rms_pol is not None else "--"
            self.lbl_pol_metrics.setText(
                f"Backazimuth (BAZ): {baz_s} ± {sth_s}\n"
                f"Incidence Angle: {inc_s} ± {sph_s}\n"
                f"Rectilinearity: {rect_s}\n"
                f"Planarity: {plan_s}\n"
                f"Trajectory RMS (rms_pol): {rms_s}\n"
                f"Confidence Cones (σ_θ, σ_φ): ({sth_s}, {sph_s})"
            )
        else:
            self.lbl_pol_metrics.setText(
                "Backazimuth (BAZ): --\nIncidence Angle: --\nRectilinearity: --\nPlanarity: --\n"
                "Trajectory RMS: --\nConfidence Cones (σ_θ, σ_φ): --"
            )

        # Render M-25 focal mechanism beachball if present in result/context or retained
        fm_dict = ctx.get("focal_mechanism")
        if not isinstance(fm_dict, dict):
            fm_attr = getattr(result, "focal_mechanism", None)
            if isinstance(fm_attr, dict):
                fm_dict = fm_attr
            elif all(k in ctx for k in ("strike", "dip", "rake")):
                fm_dict = ctx
        if isinstance(fm_dict, dict):
            self._last_fm_result = dict(fm_dict)
        self._render_beachball(self._last_fm_result)

    def _update_tab6_restitution(self, result: FitDiagnosticResult) -> None:
        """Render Tab 6 restitution transfer function, SNR, receiver function, and dispersion curves."""
        curves = result.overlay_curves or {}
        ctx = result.context_data or {}

        freqs = self._safe_curve_array(curves, "freqs")
        obs_amp = self._safe_curve_array(curves, "obs_amp")
        resp_amp = self._safe_curve_array(curves, "resp_amp")
        h_amp = self._safe_curve_array(curves, "h_amp")
        transfer_amp = self._safe_curve_array(curves, "transfer_amp")
        phase_deg = self._safe_curve_array(curves, "phase_deg")

        snr_freqs = self._safe_curve_array(curves, "snr_freqs")
        snr_db = self._safe_curve_array(curves, "snr_db")

        rf_time = self._safe_curve_array(curves, "rf_time")
        rf_amp = self._safe_curve_array(curves, "rf_amp")
        rf_obs = self._safe_curve_array(curves, "rf_obs")
        rf_syn = self._safe_curve_array(curves, "rf_syn")

        periods = self._safe_curve_array(curves, "periods")
        obs_vel = self._safe_curve_array(curves, "obs_vel")
        model_vel = self._safe_curve_array(curves, "model_vel")
        group_vel = self._safe_curve_array(curves, "group_vel")
        phase_vel = self._safe_curve_array(curves, "phase_vel")

        # Left Canvas: Transfer Function / Amplitude Spectrum or Dispersion Curve
        has_amp = freqs.ndim == 1 and freqs.size >= 2 and np.any(freqs > 0.0) and any(
            arr.size == freqs.size and np.any(arr > 0.0)
            for arr in (obs_amp, resp_amp, h_amp, transfer_amp)
        )
        has_disp = periods.ndim == 1 and periods.size >= 2 and any(
            arr.size == periods.size and arr.size >= 2
            for arr in (obs_vel, model_vel, group_vel, phase_vel)
        )
        has_snr = snr_freqs.ndim == 1 and snr_db.ndim == 1 and snr_freqs.size >= 2 and snr_freqs.size == snr_db.size
        has_rf = rf_time.ndim == 1 and rf_time.size >= 2 and any(
            arr.size == rf_time.size and arr.size >= 2
            for arr in (rf_amp, rf_obs, rf_syn)
        )

        left_used_snr = False
        if has_amp:
            self.fig_rest.clear()
            ax_r = self.fig_rest.add_subplot(111)
            self._style_dark_axes(ax_r)
            pos_f = np.isfinite(freqs) & (freqs > 0.0)
            for arr, lbl, col, ls in (
                (obs_amp, "Observed |U(f)|", "#8b949e", "-"),
                (resp_amp, "|H(f)| Response", "#58a6ff", "-"),
                (h_amp, "|H(f)| Amplitude", "#58a6ff", "-"),
                (transfer_amp, "Transfer |T(f)|", "#3fb950", "--"),
            ):
                if arr.size == freqs.size:
                    m = pos_f & np.isfinite(arr) & (arr > 0.0)
                    if np.any(m):
                        ax_r.loglog(freqs[m], arr[m], color=col, linestyle=ls, linewidth=1.7, label=lbl)
            ax_r.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
            ax_r.set_ylabel("Amplitude / Sensitivity", color="#8b949e", fontsize=8)
            ax_r.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_rest, self.canvas_rest)
            self.canvas_rest.draw()
        elif has_disp:
            self.fig_rest.clear()
            ax_r = self.fig_rest.add_subplot(111)
            self._style_dark_axes(ax_r)
            for arr, lbl, col, ls in (
                (obs_vel, "Observed Velocity", "#8b949e", "o"),
                (model_vel, "Model Fit", "#3fb950", "-"),
                (group_vel, "Group Velocity U(T)", "#58a6ff", "-"),
                (phase_vel, "Phase Velocity c(T)", "#d29922", "--"),
            ):
                if arr.size == periods.size:
                    if ls == "o":
                        ax_r.plot(periods, arr, "o", color=col, markersize=3.5, label=lbl)
                    else:
                        ax_r.plot(periods, arr, color=col, linestyle=ls, linewidth=1.7, label=lbl)
            ax_r.set_xlabel("Period (s)", color="#8b949e", fontsize=8)
            ax_r.set_ylabel("Velocity (km/s)", color="#8b949e", fontsize=8)
            ax_r.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_rest, self.canvas_rest)
            self.canvas_rest.draw()
        elif has_snr:
            left_used_snr = True
            self.fig_rest.clear()
            ax_r = self.fig_rest.add_subplot(111)
            self._style_dark_axes(ax_r)
            pos_sf = np.isfinite(snr_freqs) & (snr_freqs > 0.0)
            if np.any(pos_sf):
                ax_r.semilogx(snr_freqs[pos_sf], snr_db[pos_sf], color="#58a6ff", linewidth=1.7, label="SNR (dB)")
            else:
                ax_r.plot(snr_freqs, snr_db, color="#58a6ff", linewidth=1.7, label="SNR (dB)")
            ax_r.axhline(3.0, color="#f85149", linestyle="--", linewidth=1.0, label="3 dB Threshold")
            ax_r.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
            ax_r.set_ylabel("SNR (dB)", color="#8b949e", fontsize=8)
            ax_r.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_rest, self.canvas_rest)
            self.canvas_rest.draw()
        else:
            self._draw_empty_placeholder(
                self.fig_rest,
                self.canvas_rest,
                "No Transfer Function / Dispersion Data Available",
            )

        # Right Canvas: Receiver Function and/or SNR vs Frequency / Phase Response
        has_phase = freqs.ndim == 1 and phase_deg.ndim == 1 and freqs.size >= 2 and phase_deg.size == freqs.size
        show_snr_right = has_snr and not left_used_snr

        if has_rf and show_snr_right:
            self.fig_rf.clear()
            ax_snr = self.fig_rf.add_subplot(211)
            ax_rf = self.fig_rf.add_subplot(212)
            self._style_dark_axes(ax_snr)
            self._style_dark_axes(ax_rf)

            pos_sf = np.isfinite(snr_freqs) & (snr_freqs > 0.0)
            if np.any(pos_sf):
                ax_snr.semilogx(snr_freqs[pos_sf], snr_db[pos_sf], color="#58a6ff", linewidth=1.5, label="SNR (dB)")
            else:
                ax_snr.plot(snr_freqs, snr_db, color="#58a6ff", linewidth=1.5, label="SNR (dB)")
            ax_snr.axhline(3.0, color="#f85149", linestyle="--", linewidth=0.9)
            ax_snr.set_ylabel("SNR (dB)", color="#8b949e", fontsize=8)
            ax_snr.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

            if rf_obs.size == rf_time.size:
                ax_rf.plot(rf_time, rf_obs, color="#8b949e", linewidth=1.2, label="Observed RF")
            if rf_amp.size == rf_time.size:
                ax_rf.plot(rf_time, rf_amp, color="#3fb950", linewidth=1.5, label="Receiver Function")
            if rf_syn.size == rf_time.size:
                ax_rf.plot(rf_time, rf_syn, color="#58a6ff", linestyle="--", linewidth=1.4, label="Synthetic Fit")
            ax_rf.axhline(0.0, color="#30363d", linewidth=0.7)
            ax_rf.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
            ax_rf.set_ylabel("RF Amp", color="#8b949e", fontsize=8)
            ax_rf.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

            safe_tight_layout(self.fig_rf, self.canvas_rf)
            self.canvas_rf.draw()
        elif has_rf:
            self.fig_rf.clear()
            ax_rf = self.fig_rf.add_subplot(111)
            self._style_dark_axes(ax_rf)
            if rf_obs.size == rf_time.size:
                ax_rf.plot(rf_time, rf_obs, color="#8b949e", linewidth=1.2, label="Observed RF")
            if rf_amp.size == rf_time.size:
                ax_rf.plot(rf_time, rf_amp, color="#3fb950", linewidth=1.6, label="Receiver Function")
            if rf_syn.size == rf_time.size:
                ax_rf.plot(rf_time, rf_syn, color="#58a6ff", linestyle="--", linewidth=1.5, label="Synthetic Fit")
            ax_rf.axhline(0.0, color="#30363d", linewidth=0.7)
            ax_rf.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
            ax_rf.set_ylabel("Amplitude", color="#8b949e", fontsize=8)
            ax_rf.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_rf, self.canvas_rf)
            self.canvas_rf.draw()
        elif show_snr_right:
            self.fig_rf.clear()
            ax_snr = self.fig_rf.add_subplot(111)
            self._style_dark_axes(ax_snr)
            pos_sf = np.isfinite(snr_freqs) & (snr_freqs > 0.0)
            if np.any(pos_sf):
                ax_snr.semilogx(snr_freqs[pos_sf], snr_db[pos_sf], color="#58a6ff", linewidth=1.7, label="SNR (dB)")
            else:
                ax_snr.plot(snr_freqs, snr_db, color="#58a6ff", linewidth=1.7, label="SNR (dB)")
            ax_snr.axhline(3.0, color="#f85149", linestyle="--", linewidth=1.0, label="3 dB Threshold")
            ax_snr.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
            ax_snr.set_ylabel("SNR (dB)", color="#8b949e", fontsize=8)
            ax_snr.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_rf, self.canvas_rf)
            self.canvas_rf.draw()
        elif has_phase:
            self.fig_rf.clear()
            ax_ph = self.fig_rf.add_subplot(111)
            self._style_dark_axes(ax_ph)
            pos_f = np.isfinite(freqs) & (freqs > 0.0)
            if np.any(pos_f):
                ax_ph.semilogx(freqs[pos_f], phase_deg[pos_f], color="#d29922", linewidth=1.6, label="Phase (deg)")
            else:
                ax_ph.plot(freqs, phase_deg, color="#d29922", linewidth=1.6, label="Phase (deg)")
            ax_ph.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
            ax_ph.set_ylabel("Phase (deg)", color="#8b949e", fontsize=8)
            ax_ph.legend(loc="best", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
            safe_tight_layout(self.fig_rf, self.canvas_rf)
            self.canvas_rf.draw()
        else:
            self._draw_empty_placeholder(
                self.fig_rf,
                self.canvas_rf,
                "No Deconvolution / SNR / Receiver Function Data Available",
            )

        info_parts: List[str] = []
        if "station" in ctx:
            info_parts.append(f"Station: {ctx['station']}")
        if "status" in ctx:
            info_parts.append(f"Status: {ctx['status']}")
        if "passband_snr_db" in ctx:
            info_parts.append(f"Passband SNR: {float(ctx['passband_snr_db']):.1f} dB")
        elif has_snr:
            info_parts.append(f"Mean SNR: {float(np.nanmean(snr_db)):.1f} dB")
        if "waterlevel" in ctx:
            info_parts.append(f"Waterlevel: {float(ctx['waterlevel']):.3g}")
        if "fit_pct" in ctx:
            info_parts.append(f"Variance Red: {float(ctx['fit_pct']):.1f}%")
        if info_parts:
            self.lbl_rest_info.setText(" | ".join(info_parts))
        elif result.fit_type in (FitType.RESTITUTION_FIT, FitType.RECEIVER_FUNC, FitType.SURFACE_DISPERSION):
            self.lbl_rest_info.setText(f"{result.title}: {'Converged' if result.converged else 'Incomplete'}")
        else:
            self.lbl_rest_info.setText("Restitution / Deconvolution: Idle")

    def _on_tab_changed(self, idx: int) -> None:
        tab_canvases = {
            0: [(self.fig_corr, self.canvas_corr)],
            1: [(self.fig_scatter, self.canvas_scatter), (self.fig_ellipse, self.canvas_ellipse)],
            2: [(self.fig_fk, self.canvas_fk), (self.fig_beam_res, self.canvas_beam_res)],
            3: [(self.fig_spec, self.canvas_spec), (self.fig_misfit, self.canvas_misfit)],
            4: [(self.fig_pol, self.canvas_pol), (self.fig_beachball, self.canvas_beachball)],
            5: [(self.fig_rest, self.canvas_rest), (self.fig_rf, self.canvas_rf)],
        }
        for fig, canvas in tab_canvases.get(idx, []):
            safe_tight_layout(fig, canvas)
            canvas.draw_idle()

    # ------------------------------------------------------------------
    # User Interactions & Reactive Loop
    # ------------------------------------------------------------------

    def _on_domain_changed(self, index: int) -> None:
        selected_type = self.domain_combo.itemData(index)
        tab_map = {
            FitType.HYPOCENTER_GEIGER: 1,
            FitType.ORIGIN_TIME: 1,
            FitType.PLANE_WAVE: 2,
            FitType.FK_SEMBLANCE: 2,
            FitType.SP_DIFF_DEPTH: 1,
            FitType.TTFIT_SIMPLEX: 1,
            FitType.BRUNE_SPECTRUM: 3,
            FitType.POLARIZATION_3C: 4,
            FitType.RESTITUTION_FIT: 5,
            FitType.RECEIVER_FUNC: 5,
            FitType.SURFACE_DISPERSION: 5,
        }
        target_tab = tab_map.get(selected_type, 0)
        self.tabs.setCurrentIndex(target_tab)

    def _on_table_row_clicked(self, row: int, col: int) -> None:
        if self.current_result and row < len(self.current_result.residuals):
            entry = self.current_result.residuals[row]
            self.bridge.highlightTraceRequested.emit(entry.station)

    def _on_table_row_double_clicked(self, row: int, col: int) -> None:
        if self.current_result and row < len(self.current_result.residuals):
            entry = self.current_result.residuals[row]
            self.bridge.zoomToPhaseRequested.emit(entry.station, entry.phase)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._block_table_signals or self._syncing_table_items or not self.current_result:
            return
        col = item.column()
        if col not in (0, 7):
            return

        row = item.row()
        self._syncing_table_items = True
        try:
            if col == 0:
                weight_item = self.table_residuals.item(row, 7)
                if item.checkState() == Qt.CheckState.Checked:
                    if weight_item is not None:
                        txt = weight_item.text().strip()
                        try:
                            w = float(txt) if txt else 0.0
                        except ValueError:
                            w = 0.0
                        if not txt or txt == "0.00" or w <= 0.0:
                            weight_item.setText("1.00")
                elif item.checkState() == Qt.CheckState.Unchecked:
                    if weight_item is not None:
                        weight_item.setText("0.00")
            elif col == 7:
                chk_item = self.table_residuals.item(row, 0)
                if chk_item is not None:
                    txt = item.text().strip()
                    try:
                        w = float(txt)
                    except ValueError:
                        w = 1.0
                    if w > 0.0:
                        chk_item.setCheckState(Qt.CheckState.Checked)
                    else:
                        chk_item.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self._syncing_table_items = False

        # Trigger refit if checkbox in col 0 or weight in col 7 is edited
        if self.chk_auto_refit.isChecked():
            self._dispatch_recompute_request()

    def _dispatch_recompute_request(self) -> None:
        """Collect current table state and emit recomputeFitRequested."""
        if not self.current_result:
            return

        excluded_ids: List[str] = []
        weight_overrides: Dict[str, float] = {}

        n_rows = self.table_residuals.rowCount()
        for r in range(n_rows):
            chk_item = self.table_residuals.item(r, 0)
            st_item = self.table_residuals.item(r, 1)
            comp_item = self.table_residuals.item(r, 2)
            phase_item = self.table_residuals.item(r, 3)
            weight_item = self.table_residuals.item(r, 7)

            if not chk_item or not st_item or not phase_item:
                continue

            station = st_item.text()
            comp = comp_item.text() if comp_item else "Z"
            phase = phase_item.text()
            ident = f"{station}.{comp}.{phase}"

            try:
                w_val = float(weight_item.text()) if weight_item else 1.0
            except ValueError:
                w_val = 1.0

            is_checked = (chk_item.checkState() == Qt.CheckState.Checked)
            if is_checked and w_val <= 0.0:
                w_val = 1.0
                if weight_item is not None:
                    prev_sync = self._syncing_table_items
                    self._syncing_table_items = True
                    try:
                        weight_item.setText("1.00")
                    finally:
                        self._syncing_table_items = prev_sync

            if not is_checked or (w_val <= 0.0 and not is_checked):
                excluded_ids.append(ident)
                excluded_ids.append(station)
                weight_overrides[ident] = 0.0
                weight_overrides[station] = 0.0
            else:
                weight_overrides[ident] = w_val
                weight_overrides[station] = w_val

        req = FitModificationRequest(
            fit_type=self.current_result.fit_type,
            excluded_identifiers=excluded_ids,
            weight_overrides=weight_overrides,
        )
        self.bridge.recomputeFitRequested.emit(req)

    def _on_deactivate_outliers(self) -> None:
        """Deactivate all outlier rows and trigger refit."""
        if not self.current_result:
            return
        self._block_table_signals = True
        try:
            for r in range(self.table_residuals.rowCount()):
                status_item = self.table_residuals.item(r, 10)
                if status_item and "OUTLIER" in status_item.text():
                    chk_item = self.table_residuals.item(r, 0)
                    if chk_item:
                        chk_item.setCheckState(Qt.CheckState.Unchecked)
                    w_item = self.table_residuals.item(r, 7)
                    if w_item:
                        w_item.setText("0.00")
        finally:
            self._block_table_signals = False

        self._dispatch_recompute_request()

    def _on_recompute_clicked(self) -> None:
        self._dispatch_recompute_request()

    def _on_adopt_clicked(self) -> None:
        if self.current_result and "lat" in self.current_result.context_data:
            # We can package param or notify bridge
            from shm_app.core.models import AnalysisParamSet
            from shm_app.core.time_utils import format_sh_time

            ctx = self.current_result.context_data
            t0 = ctx.get("t0", 0.0)
            lat = ctx.get("lat", 0.0)
            lon = ctx.get("lon", 0.0)
            z = ctx.get("depth", 33.0)

            param = AnalysisParamSet(
                lat=lat,
                lon=lon,
                depth=z,
                origin=format_sh_time(t0),
                rmse=self.current_result.rmse,
                wrmse=self.current_result.wrmse,
                chi2_reduced=self.current_result.chi2_reduced,
                condition_number=self.current_result.condition_number,
            )
            if self.current_result.confidence_ellipse:
                el = self.current_result.confidence_ellipse
                param.error_ellipse = (el.semi_major_km, el.semi_minor_km, el.strike_deg)
            self.bridge.adoptParametersRequested.emit(param)

    def _on_export_csv(self) -> None:
        if not self.current_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Residuals to CSV", "residuals.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Station",
                        "Component",
                        "Phase",
                        "Observed",
                        "Calculated",
                        "Residual_s",
                        "Weight",
                        "Distance_deg",
                        "Azimuth_deg",
                        "Is_Outlier",
                    ]
                )
                for e in self.current_result.residuals:
                    writer.writerow(
                        [
                            e.station,
                            e.channel,
                            e.phase,
                            e.observed,
                            e.calculated,
                            e.residual,
                            e.weight,
                            e.distance_deg,
                            e.azimuth_deg,
                            e.is_outlier,
                        ]
                    )
        except Exception:
            pass

    def _on_export_png(self) -> None:
        curr_tab = self.tabs.currentIndex()
        canvas_map = {
            0: self.canvas_corr,
            1: self.canvas_scatter,
            2: self.canvas_fk,
            3: self.canvas_spec,
            4: self.canvas_pol,
            5: self.canvas_rest,
        }
        canvas = canvas_map.get(curr_tab, self.canvas_scatter)
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot Image", "diagnostic_plot.png", "PNG Files (*.png)")
        if not path:
            return
        try:
            canvas.figure.savefig(path, dpi=200, facecolor=canvas.figure.get_facecolor())
        except Exception:
            pass
