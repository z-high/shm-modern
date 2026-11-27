"""SeedLink v3.1 live waveform stream configuration and control dialog."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shm_app.gui.ui_helpers import apply_c_locale, disable_button_autodefault


class SeedLinkStreamDialog(QDialog):
    """Configuration and control dialog for native SeedLink v3.1 TCP streaming (M-21)."""

    def __init__(self, live_coordinator, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.live_coordinator = live_coordinator
        self.setWindowTitle("SeedLink v3.1 Live Waveform Stream")
        self.resize(520, 260)

        layout = QVBoxLayout(self)

        grp = QGroupBox("SeedLink Server && Stream Selection")
        form = QFormLayout(grp)

        self.host_edit = QLineEdit("geofon.gfz-potsdam.de")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(18000)

        self.streams_edit = QLineEdit("GR_GRA1:BHZ,GR_GRB1:BHZ")
        self.streams_edit.setPlaceholderText("Comma-separated NET_STA:LOCCHA (e.g. GR_GRA1:BHZ)")

        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(200, 500000)
        self.max_samples_spin.setValue(36000)
        self.max_samples_spin.setSingleStep(1000)

        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("Streams:", self.streams_edit)
        form.addRow("Buffer Samples / Stream:", self.max_samples_spin)
        layout.addWidget(grp)

        self.status_lbl = QLabel(f"Status: {getattr(live_coordinator, 'seedlink_status', 'idle')}")
        self.status_lbl.setStyleSheet("color: #58a6ff; font-family: Monospace;")
        layout.addWidget(self.status_lbl)

        btn_h = QHBoxLayout()
        self.btn_start = QPushButton("Start Stream")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop Stream")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_h.addStretch(1)
        btn_h.addWidget(self.btn_start)
        btn_h.addWidget(self.btn_stop)
        btn_h.addWidget(btn_close)
        layout.addLayout(btn_h)

        if hasattr(live_coordinator, "seedlinkStatusChanged"):
            live_coordinator.seedlinkStatusChanged.connect(self._on_status_changed)

        self._sync_buttons()
        disable_button_autodefault(self)
        apply_c_locale(self)

    def parsed_streams(self) -> List[tuple]:
        out: List[tuple] = []
        for raw in self.streams_edit.text().split(","):
            tok = raw.strip()
            if not tok:
                continue
            if ":" in tok:
                left, sel = tok.split(":", 1)
                parts = [p for p in left.replace(".", "_").split("_") if p]
                if len(parts) >= 2:
                    sel_norm = sel.strip().upper()
                    if "." not in sel_norm:
                        sel_norm = f"{sel_norm}.D"
                    out.append((parts[0].upper(), parts[1].upper(), sel_norm))
                    continue
            if "." in tok:
                parts = tok.split(".")
                if len(parts) >= 4:
                    loc = parts[2].strip().upper()
                    cha = parts[3].strip().upper() or "BHZ"
                    sel = f"{loc}{cha}.D" if loc else f"{cha}.D"
                    out.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                    continue
                if len(parts) >= 2:
                    sel = f"{parts[2].strip().upper()}.D" if len(parts) == 3 and parts[2].strip() else "BHZ.D"
                    out.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                    continue
            if "_" in tok:
                parts = [p for p in tok.split("_") if p]
                if len(parts) >= 2:
                    sel = f"{parts[2].upper()}.D" if len(parts) >= 3 else "BHZ.D"
                    out.append((parts[0].upper(), parts[1].upper(), sel))
        return out

    def _on_start(self) -> None:
        host = self.host_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "SeedLink", "Please enter a SeedLink host.")
            return
        streams = self.parsed_streams()
        self.live_coordinator.start_seedlink(
            host=host,
            port=int(self.port_spin.value()),
            streams=streams or None,
            max_samples_per_stream=int(self.max_samples_spin.value()),
        )
        self._sync_buttons()

    def _on_stop(self) -> None:
        self.live_coordinator.stop_seedlink()
        self._sync_buttons()

    def _on_status_changed(self, status: str) -> None:
        self.status_lbl.setText(f"Status: {status}")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        running = bool(getattr(self.live_coordinator, "is_seedlink_running", False))
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
