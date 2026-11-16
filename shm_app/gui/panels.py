"""Dockable phase-picking and parameter panels for modernized SHM."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shm_app.core.config_station import SHMConfig, get_config, resolve_reference_station
from shm_app.core.models import (
    SHM_EVENT_TYPES,
    AnalysisParamSet,
    PhasePick,
    Trace,
    normalize_shm_event_type,
)
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.help_texts import register_tutorial_help
from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault
from shm_app.seismology.traveltime import sphere_diff


class PhaseDialogPanel(QWidget):
    """Dockable SHM Phase Box for configuring active phase name, onset spec, polarity, quality, and weight."""

    phaseConfigChanged = pyqtSignal()
    deleteActivePickRequested = pyqtSignal()
    depthPhaseCheckToggled = pyqtSignal(bool)

    PHASE_PRESETS = [
        "P", "pP", "sP", "PP", "PcP", "PKP",
        "S", "sS", "SS", "ScS", "SKS", "PKiKP",
        "Pg", "Pn", "Sg", "Sn", "Lg", "Rg",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.active_phase: str = "P"
        self.active_spec: str = "e"
        self.active_sign: int = 0
        self.active_source: str = "m"
        self.active_quality: int = 4
        self.active_weight: int = 2

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Phase name quick buttons
        grp_ph = QGroupBox("Phase Name")
        grid = QGridLayout(grp_ph)
        grid.setContentsMargins(4, 3, 4, 3)
        grid.setSpacing(2)
        self.phase_edit = QLineEdit("P")
        self.phase_edit.textChanged.connect(self._on_phase_text_changed)
        grid.addWidget(QLabel("Active:"), 0, 0)
        grid.addWidget(self.phase_edit, 0, 1, 1, 3)

        for idx, ph in enumerate(self.PHASE_PRESETS):
            btn = QPushButton(ph)
            btn.setStyleSheet("padding: 1px 6px; min-height: 20px;")
            btn.clicked.connect(lambda _checked=False, name=ph: self.set_phase_name(name))
            row = 1 + idx // 4
            col = idx % 4
            grid.addWidget(btn, row, col)
        layout.addWidget(grp_ph)

        # Onset Spec, Sign, Quality, Weight
        grp_attr = QGroupBox("Pick Attributes")
        form = QFormLayout(grp_attr)
        form.setContentsMargins(4, 3, 4, 3)
        form.setVerticalSpacing(2)
        form.setHorizontalSpacing(6)

        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["e (Emergent)", "i (Impulsive)"])
        self.spec_combo.currentIndexChanged.connect(self._sync_from_widgets)
        form.addRow("Onset Type:", self.spec_combo)

        self.sign_combo = QComboBox()
        self.sign_combo.addItems(["0 (None)", "+ / c (Compression)", "- / d (Dilatation)"])
        self.sign_combo.currentIndexChanged.connect(self._sync_from_widgets)
        form.addRow("First Motion:", self.sign_combo)

        self.qual_spin = QSpinBox()
        self.qual_spin.setRange(0, 10)
        self.qual_spin.setValue(4)
        self.qual_spin.valueChanged.connect(self._on_quality_changed)
        form.addRow("Quality (0-10):", self.qual_spin)

        self.weight_label = QLabel("2")
        form.addRow("SHM Weight:", self.weight_label)
        layout.addWidget(grp_attr)

        # Actions
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 1, 0, 1)
        self.btn_del_pick = QPushButton("Delete Pick")
        self.btn_del_pick.setStyleSheet("padding: 2px 8px; min-height: 22px;")
        self.btn_del_pick.clicked.connect(self.deleteActivePickRequested.emit)
        btn_row.addWidget(self.btn_del_pick)

        self.chk_depth_rays = QCheckBox("Depth Phase Check")
        self.chk_depth_rays.toggled.connect(self.depthPhaseCheckToggled.emit)
        btn_row.addWidget(self.chk_depth_rays)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        register_tutorial_help(self.phase_edit, "Active Phase Name", "phase_name", "", "Active phase label for picking (e.g. P, S, PP)")
        register_tutorial_help(self.spec_combo, "Onset Sharpness / Specifier", "phase_spec", "", "Onset sharpness (e/i/+/-/c/d)")
        register_tutorial_help(self.sign_combo, "First-Motion Polarity", "phase_sign", "", "First-motion polarity (+, -, 0)")
        register_tutorial_help(self.qual_spin, "Pick Quality (0-9)", "phase_quality", "", "Pick quality index 0 (best) to 9 (unusable)")
        register_tutorial_help(self.btn_del_pick, "Delete Phase Pick", "phase_delete_btn", "", "Delete active phase picks")
        register_tutorial_help(self.chk_depth_rays, "Depth Phase Check", "phase_depth_rays", "", "Show depth-phase (pP/sP) guide offsets")
        for btn in self.findChildren(QPushButton):
            if btn is not self.btn_del_pick:
                register_tutorial_help(btn, f"Select Phase '{btn.text()}'", "phase_preset", "", f"Switch active picking phase to {btn.text()}")
        disable_button_autodefault(self)
        apply_c_locale(self)

    def set_phase_name(self, name: str) -> None:
        self.active_phase = name
        self.phase_edit.setText(name)
        self.phaseConfigChanged.emit()

    def _on_phase_text_changed(self, txt: str) -> None:
        if txt.strip():
            self.active_phase = txt.strip()
            self.phaseConfigChanged.emit()

    def _on_quality_changed(self, q: int) -> None:
        dummy = PhasePick()
        dummy.set_quality(q)
        self.active_quality = dummy.quality
        self.active_weight = dummy.weight
        self.weight_label.setText(str(self.active_weight))
        self.phaseConfigChanged.emit()

    def _sync_from_widgets(self) -> None:
        self.active_spec = "i" if self.spec_combo.currentIndex() == 1 else "e"
        s_idx = self.sign_combo.currentIndex()
        self.active_sign = 1 if s_idx == 1 else (-1 if s_idx == 2 else 0)
        self.phaseConfigChanged.emit()

    def create_pick_for_trace(self, tr: Trace, epoch: float) -> PhasePick:
        pk = PhasePick(
            name=self.active_phase,
            onset=format_sh_time(epoch),
            spec=self.active_spec,
            sign=self.active_sign,
            source=self.active_source,
            filter_name=tr.filter_name,
            station=tr.station,
            comp=tr.comp,
            chan1=tr.chan1,
            chan2=tr.chan2,
        )
        pk.set_quality(self.active_quality)
        return pk


class ParameterBoxPanel(QWidget):
    """Dockable SHM Parameter Box with Sets 1/2/3, Zoom, Normalization, and Event Parameters."""

    parametersUpdated = pyqtSignal()
    normChanged = pyqtSignal(str)
    zoomRequested = pyqtSignal(float)
    timeStepRequested = pyqtSignal(float)

    def __init__(
        self,
        param_sets: List[AnalysisParamSet],
        parent: Optional[QWidget] = None,
        config: Optional[SHMConfig] = None,
    ) -> None:
        super().__init__(parent)
        self.param_sets = param_sets
        self.config = config
        self.active_set_idx = 0
        self._updating_ui = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Parameter Set selector (Set 1, Set 2, Set 3)
        set_row = QHBoxLayout()
        set_row.setSpacing(4)
        set_row.addWidget(QLabel("Param Set:"))
        self.set_combo = QComboBox()
        self.set_combo.addItems(["Set 1 (Primary)", "Set 2 (Secondary)", "Set 3 (Scratch)"])
        self.set_combo.currentIndexChanged.connect(self._on_set_switched)
        set_row.addWidget(self.set_combo)
        layout.addLayout(set_row)

        # Event & Array Parameters Form
        grp_p = QGroupBox("Analysis Parameters")
        form = QFormLayout(grp_p)
        form.setContentsMargins(4, 4, 4, 4)
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(6)

        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0.0, 180.0)
        self.dist_spin.setDecimals(2)
        self.dist_spin.setSuffix(" deg")
        self.dist_spin.valueChanged.connect(self._on_field_edited)
        form.addRow("Distance:", self.dist_spin)

        self.azim_spin = QDoubleSpinBox()
        self.azim_spin.setRange(0.0, 360.0)
        self.azim_spin.setDecimals(1)
        self.azim_spin.setSuffix(" deg")
        self.azim_spin.valueChanged.connect(self._on_field_edited)
        form.addRow("Backazimuth:", self.azim_spin)

        self.slow_spin = QDoubleSpinBox()
        self.slow_spin.setRange(0.0, 40.0)
        self.slow_spin.setDecimals(2)
        self.slow_spin.setSuffix(" s/deg")
        self.slow_spin.valueChanged.connect(self._on_field_edited)
        form.addRow("Slowness:", self.slow_spin)

        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 800.0)
        self.depth_spin.setDecimals(1)
        self.depth_spin.setValue(33.0)
        self.depth_spin.setSuffix(" km")
        self.depth_spin.valueChanged.connect(self._on_field_edited)
        form.addRow("Depth:", self.depth_spin)

        self.origin_edit = QLineEdit()
        self.origin_edit.editingFinished.connect(self._on_field_edited)
        form.addRow("Origin Time:", self.origin_edit)

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(3)
        self.lat_spin.valueChanged.connect(self._on_field_edited)
        form.addRow("Latitude:", self.lat_spin)

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(3)
        self.lon_spin.valueChanged.connect(self._on_field_edited)
        form.addRow("Longitude:", self.lon_spin)

        self.region_lbl = QLabel("-")
        self.region_lbl.setWordWrap(True)
        form.addRow("F-E Region:", self.region_lbl)

        self.mags_lbl = QLabel("ML: 0.0  mb: 0.0  MS: 0.0")
        form.addRow("Magnitudes:", self.mags_lbl)

        self.evtype_combo = QComboBox()
        for code, label in SHM_EVENT_TYPES:
            self.evtype_combo.addItem(label, code)
        self.evtype_combo.currentIndexChanged.connect(self._on_field_edited)
        form.addRow("Event Type:", self.evtype_combo)
        layout.addWidget(grp_p)

        # Display Normalization & Zoom controls
        grp_disp = QGroupBox("Display Scaling && Time")
        disp_v = QVBoxLayout(grp_disp)
        disp_v.setContentsMargins(4, 4, 4, 4)
        disp_v.setSpacing(3)

        norm_h = QHBoxLayout()
        norm_h.addWidget(QLabel("Norm:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["A (Auto Trace)", "C (Constant)", "AW (Auto Window)", "SW (Single Window)"])
        self.norm_combo.currentIndexChanged.connect(self._on_norm_changed)
        norm_h.addWidget(self.norm_combo)
        disp_v.addLayout(norm_h)

        zoom_h = QHBoxLayout()
        btn_zm_in = QPushButton("Amp +")
        btn_zm_in.clicked.connect(lambda: self.zoomRequested.emit(1.5))
        btn_zm_out = QPushButton("Amp -")
        btn_zm_out.clicked.connect(lambda: self.zoomRequested.emit(1.0 / 1.5))
        btn_zm_rst = QPushButton("1:1")
        btn_zm_rst.clicked.connect(lambda: self.zoomRequested.emit(0.0))
        zoom_h.addWidget(btn_zm_in)
        zoom_h.addWidget(btn_zm_out)
        zoom_h.addWidget(btn_zm_rst)
        disp_v.addLayout(zoom_h)

        step_h = QHBoxLayout()
        btn_t_left = QPushButton("<< Time")
        btn_t_left.clicked.connect(lambda: self.timeStepRequested.emit(-0.25))
        btn_t_right = QPushButton("Time >>")
        btn_t_right.clicked.connect(lambda: self.timeStepRequested.emit(0.25))
        step_h.addWidget(btn_t_left)
        step_h.addWidget(btn_t_right)
        disp_v.addLayout(step_h)

        register_tutorial_help(self.set_combo, "Parameter Set Slot", "param_set_slot", "", "Select Analysis Parameter Set 1..3")
        register_tutorial_help(self.dist_spin, "Epicentral Distance (deg)", "param_distance", "", "Epicentral distance in degrees (0-180)")
        register_tutorial_help(self.azim_spin, "Backazimuth (deg)", "param_azimuth", "", "Station-to-event backazimuth (0-360 deg)")
        register_tutorial_help(self.slow_spin, "Horizontal Slowness", "param_slowness", "", "Horizontal slowness (s/deg)")
        register_tutorial_help(self.depth_spin, "Source Depth (km)", "param_depth", "", "Focal depth in kilometers")
        register_tutorial_help(self.origin_edit, "Origin Time (UTC)", "param_origin", "", "Event origin timestamp in UTC")
        register_tutorial_help(self.lat_spin, "Epicenter Latitude", "param_coords", "", "Event latitude (WGS84 decimal deg)")
        register_tutorial_help(self.lon_spin, "Epicenter Longitude", "param_coords", "", "Event longitude (WGS84 decimal deg)")
        register_tutorial_help(self.evtype_combo, "Event Type", "param_evtype", "", "Source classification (tele, regional, local, ...)")
        register_tutorial_help(self.norm_combo, "Normalization Mode", "param_norm", "", "Vertical trace amplitude normalization mode")
        register_tutorial_help(btn_t_left, "Pan Time Left", "param_time_step", "", "Shift time window backward")
        register_tutorial_help(btn_t_right, "Pan Time Right", "param_time_step", "", "Shift time window forward")
        for btn in self.findChildren(QPushButton):
            if btn.text() in ("Amp +", "Amp -", "1:1"):
                register_tutorial_help(btn, f"Amplitude Gain ({btn.text()})", "param_amp_zoom", "", f"Vertical trace scaling: {btn.text()}")

        layout.addWidget(grp_disp)
        layout.addStretch(1)
        disable_button_autodefault(self)
        apply_c_locale(self)
        self.refresh_from_model()

    NORM_MODES = ("A", "C", "AW", "SW")

    def set_norm_mode(self, mode: str) -> None:
        """Synchronize the normalization combo box with an engine/workspace mode change."""
        m = (mode or "A").strip().upper()
        if m not in self.NORM_MODES:
            return
        idx = self.NORM_MODES.index(m)
        if self.norm_combo.currentIndex() != idx:
            self.norm_combo.blockSignals(True)
            self.norm_combo.setCurrentIndex(idx)
            self.norm_combo.blockSignals(False)

    def _on_set_switched(self, idx: int) -> None:
        self.active_set_idx = max(0, min(2, idx))
        self.refresh_from_model()
        self.parametersUpdated.emit()

    def refresh_from_model(self) -> None:
        self._updating_ui = True
        try:
            if self.set_combo.currentIndex() != self.active_set_idx:
                self.set_combo.blockSignals(True)
                self.set_combo.setCurrentIndex(self.active_set_idx)
                self.set_combo.blockSignals(False)
            p = self.param_sets[self.active_set_idx]
            self.dist_spin.setValue(p.distance)
            self.azim_spin.setValue(p.azimuth % 360.0)
            self.slow_spin.setValue(p.slowness)
            self.depth_spin.setValue(p.depth)
            self.origin_edit.setText(p.origin)
            self.lat_spin.setValue(p.lat)
            self.lon_spin.setValue(p.lon)
            reg_txt = f"[{p.region_id}] {p.region_name}" if p.region_name else "-"
            self.region_lbl.setText(reg_txt)
            self.mags_lbl.setText(f"ML:{p.mag_ml:.2f} mb:{p.mag_mb:.2f} MS:{p.mag_ms:.2f}")
            ev_code = normalize_shm_event_type(p.event_type, dist_deg=p.distance)
            ev_codes = [c for c, _ in SHM_EVENT_TYPES]
            ev_idx = ev_codes.index(ev_code) if ev_code in ev_codes else 0
            if self.evtype_combo.currentIndex() != ev_idx:
                self.evtype_combo.setCurrentIndex(ev_idx)
        finally:
            self._updating_ui = False

    def _on_field_edited(self) -> None:
        if self._updating_ui:
            return
        sender = self.sender()
        p = self.param_sets[self.active_set_idx]
        p.distance = self.dist_spin.value()
        p.azimuth = self.azim_spin.value()
        p.slowness = self.slow_spin.value()
        p.depth = self.depth_spin.value()
        p.origin = self.origin_edit.text().strip()
        p.lat = self.lat_spin.value()
        p.lon = self.lon_spin.value()
        ev_idx = max(0, min(len(SHM_EVENT_TYPES) - 1, self.evtype_combo.currentIndex()))
        p.event_type = SHM_EVENT_TYPES[ev_idx][0]

        if sender in (self.lat_spin, self.lon_spin):
            cfg = self.config or get_config()
            ref_lat, ref_lon = p.ref_lat, p.ref_lon
            if ref_lat == 0.0 and ref_lon == 0.0:
                _, ref_lat, ref_lon = resolve_reference_station(
                    cfg, ref_station="GRA1", default_code="GRA1"
                )
            dist_deg, baz_deg, _ = sphere_diff(ref_lat, ref_lon, p.lat, p.lon)
            igreg, grname, isreg, srname = cfg.fer.lookup(p.lat, p.lon)
            p.distance = dist_deg
            p.azimuth = baz_deg % 360.0
            p.region_id = igreg
            p.region_name = grname
            p.sregion_id = isreg
            p.sregion_name = srname
            self._updating_ui = True
            try:
                self.dist_spin.setValue(p.distance)
                self.azim_spin.setValue(p.azimuth)
                self.region_lbl.setText(f"[{p.region_id}] {p.region_name}" if p.region_name else "-")
            finally:
                self._updating_ui = False

        self.parametersUpdated.emit()

    def _on_norm_changed(self, idx: int) -> None:
        if self._updating_ui:
            return
        if 0 <= idx < len(self.NORM_MODES):
            self.normChanged.emit(self.NORM_MODES[idx])
