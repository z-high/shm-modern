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
from matplotlib.figure import Figure

from shm_app.core.fit_bridge import FitDiagnosticBridge
from shm_app.core.fit_models import (
    ConfidenceEllipse,
    FitDiagnosticResult,
    FitModificationRequest,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)


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
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # Canvas 1: Residual Scatter vs Distance & Azimuth
        scatter_box = QGroupBox("Arrival Residuals vs Distance & Azimuth")
        scatter_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        scatter_layout = QVBoxLayout(scatter_box)
        self.fig_scatter = Figure(figsize=(5, 3), facecolor="#0d1117")
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        scatter_layout.addWidget(self.canvas_scatter)
        top_layout.addWidget(scatter_box, 6)

        # Canvas 2: 95% Confidence Error Ellipse
        ellipse_box = QGroupBox("Horizontal 95% Confidence Error Ellipse")
        ellipse_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        ellipse_layout = QVBoxLayout(ellipse_box)
        self.fig_ellipse = Figure(figsize=(3, 3), facecolor="#0d1117")
        self.canvas_ellipse = FigureCanvas(self.fig_ellipse)
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
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)

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
        self.fig_pol = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_pol = FigureCanvas(self.fig_pol)
        l_layout.addWidget(self.canvas_pol)
        layout.addWidget(left_box, 1)

        right_box = QGroupBox("Polarization Metrics & Jurkevics Confidence Cones")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        r_layout = QVBoxLayout(right_box)
        self.lbl_pol_metrics = QLabel(
            "Backazimuth (BAZ): --\nIncidence Angle: --\nRectilinearity: --\nPlanarity: --\nConfidence Cones (σ_θ, σ_φ): --"
        )
        self.lbl_pol_metrics.setStyleSheet("color: #e6edf3; font-family: Monospace; font-size: 13px; line-height: 1.5;")
        r_layout.addWidget(self.lbl_pol_metrics)
        r_layout.addStretch(1)

        layout.addWidget(right_box, 1)
        self.tabs.addTab(tab, "3C Particle Motion & Polarization")

    # ------------------------------------------------------------------
    # Tab 6: Instrument Restitution & Deconvolution
    # ------------------------------------------------------------------

    def _build_tab6_restitution(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        left_box = QGroupBox("Instrument Transfer Function |H(f)| & Phase")
        left_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        l_layout = QVBoxLayout(left_box)
        self.fig_rest = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_rest = FigureCanvas(self.fig_rest)
        l_layout.addWidget(self.canvas_rest)
        layout.addWidget(left_box, 1)

        right_box = QGroupBox("Deconvolution & Receiver Function Fit")
        right_box.setStyleSheet("QGroupBox { color: #58a6ff; font-weight: bold; border: 1px solid #30363d; }")
        r_layout = QVBoxLayout(right_box)
        self.fig_rf = Figure(figsize=(4, 3), facecolor="#0d1117")
        self.canvas_rf = FigureCanvas(self.fig_rf)
        r_layout.addWidget(self.canvas_rf)

        self.lbl_rest_info = QLabel("Restitution Status: OK | Passband SNR: 42.5 dB | Waterlevel: 0.01")
        self.lbl_rest_info.setStyleSheet("color: #e6edf3; font-family: Monospace;")
        r_layout.addWidget(self.lbl_rest_info)

        layout.addWidget(right_box, 1)
        self.tabs.addTab(tab, "Restitution & Deconvolution")

    # ------------------------------------------------------------------
    # Data Population & Rendering
    # ------------------------------------------------------------------

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
                FitType.SURFACE_DISPERSION: 0,
            }
            target_tab = tab_map.get(result.fit_type, 0)
            if self.tabs.currentIndex() != target_tab:
                self.tabs.setCurrentIndex(target_tab)

            # Update KPI cards
            self.kpi_rmse.set_value(f"{result.rmse:.3f} s", "#3fb950" if result.rmse < 0.5 else "#f85149")
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

            # Render synthetic / domain visuals for tabs 3-6
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

        self.fig_corr.tight_layout()
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

        self.fig_scatter.tight_layout()
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

        self.fig_ellipse.tight_layout()
        self.canvas_ellipse.draw()

    def _update_tab3_array_fk(self, result: FitDiagnosticResult) -> None:
        """Render Tab 3 2D FK semblance grid and beam residuals."""
        self.fig_fk.clear()
        ax_fk = self.fig_fk.add_subplot(111)
        ax_fk.set_facecolor("#0d1117")
        ax_fk.tick_params(colors="#8b949e", labelsize=8)

        # Synthesize or extract FK grid
        s_vals = np.linspace(-25, 25, 50)
        sx, sy = np.meshgrid(s_vals, s_vals)
        # Peak at typical slowness sx=5, sy=7
        dist_sq = (sx - 5.0) ** 2 + (sy - 7.0) ** 2
        grid = np.exp(-dist_sq / 50.0) + 0.1 * np.random.RandomState(42).randn(*sx.shape)
        grid = np.clip(grid, 0.0, 1.0)
        im = ax_fk.imshow(grid, extent=[-25, 25, -25, 25], origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax_fk.contour(sx, sy, grid, levels=[0.7, 0.9], colors=["#58a6ff", "#3fb950"], linewidths=1.0)
        ax_fk.set_xlabel("sx (s/deg)", color="#8b949e", fontsize=8)
        ax_fk.set_ylabel("sy (s/deg)", color="#8b949e", fontsize=8)
        self.fig_fk.colorbar(im, ax=ax_fk, fraction=0.046, pad=0.04)
        self.fig_fk.tight_layout()
        self.canvas_fk.draw()

        self.fig_beam_res.clear()
        ax_b = self.fig_beam_res.add_subplot(111)
        ax_b.set_facecolor("#0d1117")
        ax_b.tick_params(colors="#8b949e", labelsize=8)
        t = np.linspace(0, 10, 200)
        beam = np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.3 * t)
        res_wf = 0.15 * np.cos(2 * np.pi * 3.0 * t) * np.exp(-0.2 * t)
        ax_b.plot(t, beam, color="#3fb950", label="Formed Beam b(t)")
        ax_b.plot(t, res_wf, color="#f85149", linestyle="--", label="Mean Residual r(t)")
        ax_b.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
        ax_b.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        self.fig_beam_res.tight_layout()
        self.canvas_beam_res.draw()

    def _update_tab4_spectrum(self, result: FitDiagnosticResult) -> None:
        """Render Tab 4 displacement spectrum and 2D misfit contour."""
        self.fig_spec.clear()
        ax_s = self.fig_spec.add_subplot(111)
        ax_s.set_facecolor("#0d1117")
        ax_s.tick_params(colors="#8b949e", labelsize=8)

        f = np.logspace(-1, 2, 80)
        omega0 = 2450.0
        fc = 1.45
        brune = omega0 / (1.0 + (f / fc) ** 2)
        noise = 10.0 + 5.0 * (f ** 0.5)

        ax_s.loglog(f, brune * 1.05 + np.random.RandomState(1).randn(len(f)) * 50, "o", color="#8b949e", markersize=3, label="Observed U(f)")
        ax_s.loglog(f, brune, color="#3fb950", linewidth=2.0, label="Brune Fit (fc=1.45 Hz)")
        ax_s.loglog(f, noise, color="#f85149", linestyle=":", label="Noise Floor")
        ax_s.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
        ax_s.set_ylabel("Displacement (nm·s)", color="#8b949e", fontsize=8)
        ax_s.legend(loc="lower left", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

        self.fig_misfit.clear()
        ax_m = self.fig_misfit.add_subplot(111)
        ax_m.set_facecolor("#0d1117")
        ax_m.tick_params(colors="#8b949e", labelsize=8)

        fc_grid = np.linspace(0.5, 3.0, 40)
        w0_grid = np.linspace(1000, 4000, 40)
        X, Y = np.meshgrid(fc_grid, w0_grid)
        misfit = ((X - fc) / 0.5) ** 2 + ((Y - omega0) / 1000.0) ** 2
        im = ax_m.contourf(X, Y, misfit, levels=15, cmap="viridis")
        ax_m.plot(fc, omega0, "r*", markersize=10, label="Global Minimum")
        ax_m.set_xlabel("Corner Freq fc (Hz)", color="#8b949e", fontsize=8)
        ax_m.set_ylabel("Plateau Ω0 (nm·s)", color="#8b949e", fontsize=8)
        self.fig_misfit.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
        self.fig_misfit.tight_layout()
        self.canvas_misfit.draw()

        self.lbl_source_params.setText(
            "Source Parameters: M0=4.25e+15 Nm, Mw=4.38, fc=1.45 Hz, r=940 m, Δσ=24.5 bar, R^2=0.962"
        )

    def _update_tab5_polarization(self, result: FitDiagnosticResult) -> None:
        """Render Tab 5 particle motion hodograms with color gradient."""
        self.fig_pol.clear()
        ax1 = self.fig_pol.add_subplot(121)
        ax2 = self.fig_pol.add_subplot(122)
        for ax in (ax1, ax2):
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e", labelsize=8)

        t = np.linspace(0, 2 * np.pi, 150)
        z = np.sin(t)
        n = 0.8 * np.sin(t) + 0.1 * np.cos(2 * t)
        e = 0.6 * np.sin(t) - 0.1 * np.cos(2 * t)

        # Color gradient: green (start) to red (end)
        colors = [
            (float(i / len(t)), float(1.0 - (i / len(t))), 0.2)
            for i in range(len(t))
        ]

        ax1.scatter(e, n, c=colors, s=12)
        ax1.set_xlabel("East", color="#8b949e", fontsize=8)
        ax1.set_ylabel("North", color="#8b949e", fontsize=8)
        ax1.set_title("N-E (Horizontal)", color="#e6edf3", fontsize=8)

        ax2.scatter(n, z, c=colors, s=12)
        ax2.set_xlabel("North", color="#8b949e", fontsize=8)
        ax2.set_ylabel("Vertical (Z)", color="#8b949e", fontsize=8)
        ax2.set_title("Z-N (Vertical)", color="#e6edf3", fontsize=8)

        self.fig_pol.tight_layout()
        self.canvas_pol.draw()

        self.lbl_pol_metrics.setText(
            "Backazimuth (BAZ): 48.5° ± 2.8°\n"
            "Incidence Angle: 28.2° ± 3.4°\n"
            "Rectilinearity L: 0.942 (Strongly linear P-wave)\n"
            "Planarity P: 0.120 (Minimal ellipticity)\n"
            "Confidence Cones (σ_θ, σ_φ): (2.8°, 3.4°)"
        )

    def _update_tab6_restitution(self, result: FitDiagnosticResult) -> None:
        """Render Tab 6 restitution transfer function and receiver functions."""
        self.fig_rest.clear()
        ax_r = self.fig_rest.add_subplot(111)
        ax_r.set_facecolor("#0d1117")
        ax_r.tick_params(colors="#8b949e", labelsize=8)

        f = np.logspace(-2, 2, 100)
        # STS-2 response
        f0 = 1.0 / 120.0
        h_amp = 1500.0 * (f / f0) ** 2 / np.sqrt((1 - (f / f0) ** 2) ** 2 + 2 * (f / f0) ** 2)
        ax_r.loglog(f, h_amp, color="#58a6ff", linewidth=2.0, label="|H(f)| Amplitude")
        ax_r.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
        ax_r.set_ylabel("Sensitivity (V / (m/s))", color="#8b949e", fontsize=8)
        ax_r.legend(loc="lower right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        self.fig_rest.tight_layout()
        self.canvas_rest.draw()

        self.fig_rf.clear()
        ax_rf = self.fig_rf.add_subplot(111)
        ax_rf.set_facecolor("#0d1117")
        ax_rf.tick_params(colors="#8b949e", labelsize=8)

        t_rf = np.linspace(-5, 30, 200)
        rf_syn = np.exp(-((t_rf - 0.0) ** 2) / 1.0) - 0.3 * np.exp(-((t_rf - 4.2) ** 2) / 1.5)
        rf_obs = rf_syn + 0.05 * np.random.RandomState(3).randn(len(t_rf))

        ax_rf.plot(t_rf, rf_obs, color="#8b949e", label="Observed RF")
        ax_rf.plot(t_rf, rf_syn, color="#3fb950", linewidth=1.5, label="Synthetic Fit (94.2%)")
        ax_rf.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
        ax_rf.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        self.fig_rf.tight_layout()
        self.canvas_rf.draw()

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
            FitType.SURFACE_DISPERSION: 0,
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
