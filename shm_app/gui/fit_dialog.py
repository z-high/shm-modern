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
from PyQt6.QtGui import QCloseEvent, QColor, QFont
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
    QMessageBox,
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
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.fit_dialog_plots import _FitDialogPlotsMixin
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


class FitDiagnosticWindow(_FitDialogPlotsMixin, QDockWidget):
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
        self._has_pol_data: bool = False
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
            (FitType.FOCAL_MECHANISM, "12. Focal Mechanism (First-Motion Beachball)"),
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
            "QPushButton:disabled { background-color: #161b22; color: #484f58; border-color: #21262d; }"
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

        # Register Tutorial Mode tooltips
        register_tutorial_help(self.domain_combo, "Fit Domain", "fit_domain", "", "Select seismological fit domain")
        register_tutorial_help(self.tabs, "Diagnostic Tabs", "fit_tabs", "", "Switch between diagnostic views")
        register_tutorial_help(self.table_residuals, "Observation Residuals Table", "fit_residuals_table", "", "Toggle active stations, edit weights (>= 0), or double-click to zoom")
        register_tutorial_help(self.chk_auto_refit, "Auto-Refit", "fit_auto_refit", "", "Automatically recompute inversion on table edit")
        register_tutorial_help(self.btn_deactivate_outliers, "Deactivate Outliers", "fit_deactivate_outliers", "", "Uncheck >3-sigma outlier rows and re-fit")
        register_tutorial_help(self.btn_recompute, "Recompute Inversion", "fit_recompute", "", "Recompute active fit domain")
        register_tutorial_help(self.btn_adopt, "Adopt to Parameter Box", "fit_adopt", "", "Adopt solution parameters to the workspace")
        register_tutorial_help(self.btn_export_csv, "Export Residuals (CSV)", "fit_export_csv", "", "Export observation residuals to CSV")
        register_tutorial_help(self.btn_export_png, "Export Plot (PNG)", "fit_export_png", "", "Export active diagnostic plot as PNG")

        # Hook up bridge signals
        self.bridge.fitDiagnosticAvailable.connect(self.set_fit_result)
        self.bridge.workspaceCleared.connect(self.clear_diagnostics)

    def clear_diagnostics(self) -> None:
        """Reset all diagnostic tables, KPI cards, and plots to idle state."""
        self.current_result = None
        self._last_fm_result = None
        self._has_pol_data = False
        self._block_table_signals = True
        try:
            self.table_params.setRowCount(0)
            self.table_residuals.setRowCount(0)
            self.kpi_rmse.set_value("-- s", "#e6edf3")
            self.kpi_chi2.set_value("--", "#e6edf3")
            self.kpi_outliers.set_value("--", "#e6edf3")
            self.kpi_status.set_value("Idle", "#e6edf3")
            self.kpi_cond.set_value("--", "#e6edf3")
            self.lbl_status_badge.setText("IDLE")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #21262d; color: #8b949e; border: 1px solid #30363d; "
                "border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px;"
            )
            self.lbl_normality.setText("Normality: Skewness=0.00, Kurtosis=0.00, JB p-value=1.0000")
            self.lbl_source_params.setText("Source Parameters: M0=--, Mw=--, fc=-- Hz, r=-- m, Δσ=-- bar, R^2=--")
            self.lbl_pol_metrics.setText(
                "Backazimuth (BAZ): --\nIncidence Angle: --\nRectilinearity: --\nPlanarity: --\n"
                "Trajectory RMS: --\nConfidence Cones (σ_θ, σ_φ): --"
            )
            self.lbl_fm_summary.setText("Focal Mechanism: No beachball solution loaded")
            self.lbl_rest_info.setText("Restitution / Deconvolution: Idle")
            for fig, canvas, msg in (
                (self.fig_corr, self.canvas_corr, "No correlation matrix available"),
                (self.fig_scatter, self.canvas_scatter, "No Residual Data"),
                (self.fig_ellipse, self.canvas_ellipse, "No Error Ellipse Data"),
                (self.fig_fk, self.canvas_fk, "No FK / Slowness Grid Data"),
                (self.fig_beam_res, self.canvas_beam_res, "No Beam Residual / Coherence Data Available"),
                (self.fig_spec, self.canvas_spec, "No Spectral Fit Curves Available"),
                (self.fig_misfit, self.canvas_misfit, "No 2D Spectral Misfit Contour Available"),
                (self.fig_pol, self.canvas_pol, "No 3C Polarization Hodogram Data Available"),
                (self.fig_beachball, self.canvas_beachball, "No Focal Mechanism Beachball\nRun First-Motion Solver (M-25)"),
                (self.fig_rest, self.canvas_rest, "No Transfer Function / Dispersion Data Available"),
                (self.fig_rf, self.canvas_rf, "No Deconvolution / SNR / Receiver Function Data Available"),
            ):
                self._draw_empty_placeholder(fig, canvas, msg)
            self._update_adopt_button_state()
        finally:
            self._block_table_signals = False

    def cleanup_canvases(self) -> None:
        """Stop pending Matplotlib idle draws and detach Qt window event filters before teardown."""
        win_handles = []
        wh = self.windowHandle()
        if wh is not None:
            win_handles.append(wh)
        parent_w = self.parentWidget()
        if parent_w is not None and parent_w.windowHandle() is not None:
            pwh = parent_w.windowHandle()
            if pwh not in win_handles:
                win_handles.append(pwh)
        for canvas in self.findChildren(FigureCanvas):
            try:
                canvas._draw_pending = False
            except Exception:
                pass
            for handle in win_handles:
                try:
                    handle.removeEventFilter(canvas)
                except Exception:
                    pass

    def closeEvent(self, event: QCloseEvent) -> None:
        self.cleanup_canvases()
        super().closeEvent(event)

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
                FitType.FOCAL_MECHANISM: 4,
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
            elif result.fit_type == FitType.FOCAL_MECHANISM:
                rmse_str = f"{result.rmse * 100.0:.1f}% misfit"
                rmse_ok = result.rmse <= 0.25
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
            self._update_adopt_button_state()

        finally:
            self._block_table_signals = False

    def _is_result_adoptable(self, result: Optional[FitDiagnosticResult]) -> bool:
        if result is None:
            return False
        ctx = result.context_data or {}
        if "lat" in ctx:
            return True
        if result.fit_type == FitType.FOCAL_MECHANISM and isinstance(ctx.get("focal_mechanism"), dict):
            return True
        return False

    def _update_adopt_button_state(self) -> None:
        if self.current_result is None:
            self.btn_adopt.setEnabled(False)
            self.btn_adopt.setToolTip("Run a Location or Focal Mechanism fit first to adopt parameters.")
            return
        if self._is_result_adoptable(self.current_result):
            self.btn_adopt.setEnabled(True)
            if self.current_result.fit_type == FitType.FOCAL_MECHANISM:
                self.btn_adopt.setToolTip("Adopt the solved double-couple focal mechanism into the workspace.")
            else:
                self.btn_adopt.setToolTip("Adopt the inverted hypocenter, depth, origin, and ellipse into Parameter Box.")
        else:
            dom_lbl = getattr(self.current_result.fit_type, "value", str(self.current_result.fit_type))
            self.btn_adopt.setEnabled(False)
            self.btn_adopt.setToolTip(
                f"Adopt to Parameter Box is only supported for Location and Focal Mechanism fits (not {dom_lbl})."
            )

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
            FitType.FOCAL_MECHANISM: 4,
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
        invalid_weight = False
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
                        if not txt or not math.isfinite(w) or w <= 0.0:
                            weight_item.setText("1.00")
                            weight_item.setData(Qt.ItemDataRole.UserRole, 1.0)
                        else:
                            weight_item.setData(Qt.ItemDataRole.UserRole, float(w))
                elif item.checkState() == Qt.CheckState.Unchecked:
                    if weight_item is not None:
                        weight_item.setText("0.00")
                        weight_item.setData(Qt.ItemDataRole.UserRole, 0.0)
            elif col == 7:
                chk_item = self.table_residuals.item(row, 0)
                txt = item.text().strip()
                try:
                    w = float(txt)
                except ValueError:
                    w = float("nan")
                if not math.isfinite(w) or w < 0.0:
                    prev_raw = item.data(Qt.ItemDataRole.UserRole)
                    if prev_raw is not None and math.isfinite(float(prev_raw)) and float(prev_raw) >= 0.0:
                        prev_w = float(prev_raw)
                    elif chk_item is not None and chk_item.checkState() == Qt.CheckState.Checked:
                        prev_w = 1.0
                    else:
                        prev_w = 0.0
                    item.setText(f"{prev_w:.2f}")
                    item.setData(Qt.ItemDataRole.UserRole, prev_w)
                    if chk_item is not None:
                        chk_item.setCheckState(
                            Qt.CheckState.Checked if prev_w > 0.0 else Qt.CheckState.Unchecked
                        )
                    invalid_weight = True
                else:
                    item.setText(f"{w:.2f}")
                    item.setData(Qt.ItemDataRole.UserRole, float(w))
                    if chk_item is not None:
                        chk_item.setCheckState(
                            Qt.CheckState.Checked if w > 0.0 else Qt.CheckState.Unchecked
                        )
        finally:
            self._syncing_table_items = False

        if invalid_weight:
            return

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
            if not math.isfinite(w_val) or w_val < 0.0:
                w_val = 0.0

            is_checked = (chk_item.checkState() == Qt.CheckState.Checked)
            if is_checked and w_val <= 0.0:
                w_val = 1.0
                if weight_item is not None:
                    prev_sync = self._syncing_table_items
                    self._syncing_table_items = True
                    try:
                        weight_item.setText("1.00")
                        weight_item.setData(Qt.ItemDataRole.UserRole, 1.0)
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
                        w_item.setData(Qt.ItemDataRole.UserRole, 0.0)
        finally:
            self._block_table_signals = False

        self._dispatch_recompute_request()

    def _on_recompute_clicked(self) -> None:
        self._dispatch_recompute_request()

    def _on_adopt_clicked(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "Adopt Parameters", "No diagnostic result is currently loaded.")
            return
        ctx = self.current_result.context_data or {}
        if "lat" in ctx:
            from shm_app.core.models import AnalysisParamSet
            from shm_app.core.time_utils import format_sh_time

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
            return

        if self.current_result.fit_type == FitType.FOCAL_MECHANISM and isinstance(ctx.get("focal_mechanism"), dict):
            self.bridge.adoptFocalMechanismRequested.emit(dict(ctx["focal_mechanism"]))
            return

        dom_lbl = getattr(self.current_result.fit_type, "value", str(self.current_result.fit_type))
        QMessageBox.information(
            self,
            "Adopt Parameters",
            f"Direct parameter adoption is supported for Location and Focal Mechanism fits, not {dom_lbl}.",
        )

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
        except Exception as exc:
            QMessageBox.warning(self, "Export Residuals Error", f"Failed to export CSV to '{path}':\n{exc}")

    def _on_export_png(self) -> None:
        curr_tab = self.tabs.currentIndex()
        is_focal_tab4 = curr_tab == 4 and (
            (self.current_result is not None and self.current_result.fit_type == FitType.FOCAL_MECHANISM)
            or (self._last_fm_result is not None and not self._has_pol_data)
        )
        canvas_map = {
            0: self.canvas_corr,
            1: self.canvas_scatter,
            2: self.canvas_fk,
            3: self.canvas_spec,
            4: self.canvas_beachball if is_focal_tab4 else self.canvas_pol,
            5: self.canvas_rest,
        }
        canvas = canvas_map.get(curr_tab, self.canvas_scatter)
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot Image", "diagnostic_plot.png", "PNG Files (*.png)")
        if not path:
            return
        try:
            canvas.figure.savefig(path, dpi=200, facecolor=canvas.figure.get_facecolor())
        except Exception as exc:
            QMessageBox.warning(self, "Export Plot Error", f"Failed to export PNG to '{path}':\n{exc}")
