"""Shortcuts reference and About dialogs for SHMMainWindow."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class _MainWindowHelpMixin:
    """Shortcuts reference and About dialogs."""

    def _build_shortcuts_help_dialog(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("SHM Keyboard & Mouse Shortcuts Reference")
        dlg.setMinimumSize(720, 520)
        dlg.resize(860, 680)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        viewer = QTextEdit(dlg)
        viewer.setReadOnly(True)
        viewer.setFrameShape(QFrame.Shape.StyledPanel)
        viewer.setHtml(
            """
            <html>
            <head>
            <style>
                body {
                    background-color: #10141d;
                    color: #e6edf3;
                    font-family: sans-serif;
                    font-size: 12px;
                    margin: 4px 6px;
                }
                h3 {
                    color: #79c0ff;
                    margin: 12px 0 6px 0;
                    padding-bottom: 3px;
                    border-bottom: 1px solid #30363d;
                    font-size: 13px;
                }
                h3.first {
                    margin-top: 2px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 6px;
                }
                td {
                    padding: 4px 8px;
                    vertical-align: top;
                    border-bottom: 1px solid #21262d;
                }
                td.key {
                    white-space: nowrap;
                    font-family: monospace;
                    font-weight: bold;
                    color: #f0f6fc;
                    width: 34%;
                }
                td.key-sm {
                    white-space: nowrap;
                    font-family: monospace;
                    font-weight: bold;
                    color: #f0f6fc;
                    width: 22%;
                }
                td.desc-sm {
                    width: 28%;
                }
                .kbd {
                    background-color: #21262d;
                    border: 1px solid #30363d;
                    padding: 1px 4px;
                }
            </style>
            </head>
            <body>
            <h3 class="first">Mouse Controls (Main Trace Canvas)</h3>
            <table>
                <tr><td class="key"><span class="kbd">Left-Click</span> (other trace)</td><td>Switch Magnifier focus to trace (no pick placed)</td></tr>
                <tr><td class="key"><span class="kbd">Left-Click</span> (active trace)</td><td>Pick active phase onset</td></tr>
                <tr><td class="key"><span class="kbd">Drag</span> pick marker / error-bar edge</td><td>Move pick marker &nbsp;|&nbsp; Adjust pick quality Q</td></tr>
                <tr><td class="key"><span class="kbd">Double-Click</span> / <span class="kbd">Del</span> (marker)</td><td>Delete pick marker</td></tr>
                <tr><td class="key"><span class="kbd">Shift + Left-Drag</span></td><td>Pan time window &nbsp;|&nbsp; <span class="kbd">Alt+Left</span> / <span class="kbd">Alt+Right</span>: previous / next view</td></tr>
                <tr><td class="key"><span class="kbd">Right-Click</span> (no drag)</td><td>Open trace context menu</td></tr>
                <tr><td class="key"><span class="kbd">Right-Drag</span></td><td>Move / resize Magnifier time window</td></tr>
                <tr><td class="key"><span class="kbd">Left-Click</span> (left margin)</td><td>Select / deselect trace (*)</td></tr>
                <tr><td class="key"><span class="kbd">Middle-Click</span> (waveform)</td><td>Cycle Cursor Lock (Crosshair / Peak / Zero)</td></tr>
                <tr><td class="key"><span class="kbd">Mouse Wheel</span></td><td>Zoom time axis (<span class="kbd">Ctrl+Wheel</span>: zoom amplitude)</td></tr>
                <tr><td class="key"><span class="kbd">Shift + Mouse Wheel</span></td><td>Scroll rows vertically (View &gt; Trace Row Height)</td></tr>
            </table>

            <h3>Mouse Controls (Magnifier Window)</h3>
            <table>
                <tr><td class="key"><span class="kbd">Left-Click</span> / <span class="kbd">Spacebar</span></td><td>High-precision phase onset pick</td></tr>
                <tr><td class="key"><span class="kbd">Drag</span> / <span class="kbd">Double-Click</span> / <span class="kbd">Del</span></td><td>Move or delete a pick marker</td></tr>
                <tr><td class="key"><span class="kbd">Right-Click &amp; Drag</span></td><td>Measure peak-to-peak Amplitude &amp; Period &rarr; Magnitudes</td></tr>
                <tr><td class="key">View &gt; Magnifier Cursor Mode</td><td>Standard, Hilbert 90&deg;, Reference Waveform</td></tr>
            </table>

            <h3>Single-Key Accelerators</h3>
            <table>
                <tr><td class="key"><span class="kbd">+</span> / <span class="kbd">-</span> / <span class="kbd">0</span></td><td>Set first-motion polarity (+ compression 'c', - dilatation 'd', 0 none)</td></tr>
                <tr><td class="key"><span class="kbd">i</span> / <span class="kbd">e</span></td><td>Set onset type (i impulsive, e emergent)</td></tr>
                <tr><td class="key"><span class="kbd">1 .. 9</span></td><td>Set pick quality (1..9) and automatic SHM weight</td></tr>
                <tr><td class="key"><span class="kbd">Left</span> / <span class="kbd">Right</span> &nbsp;&bull;&nbsp; <span class="kbd">Up</span> / <span class="kbd">Down</span></td><td>Scroll time window &nbsp;&bull;&nbsp; Zoom amplitude</td></tr>
            </table>

            <h3>Ctrl Accelerators</h3>
            <table>
                <tr><td class="key-sm"><span class="kbd">Ctrl+R</span></td><td class="desc-sm">Read Dialog</td><td class="key-sm"><span class="kbd">Ctrl+G</span></td><td class="desc-sm">BGR Download</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+F</span></td><td class="desc-sm">Filter Dialog</td><td class="key-sm"><span class="kbd">Ctrl+U</span></td><td class="desc-sm">Unfilter</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+P</span></td><td class="desc-sm">AutoPick</td><td class="key-sm"><span class="kbd">Ctrl+T</span></td><td class="desc-sm">Theo Phases</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+Shift+Del</span></td><td class="desc-sm">Delete Phase Picks</td><td class="key-sm"><span class="kbd">Ctrl+Z</span> / <span class="kbd">Ctrl+Shift+Z</span></td><td class="desc-sm">Undo / Redo</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+B</span></td><td class="desc-sm">Beam</td><td class="key-sm"><span class="kbd">Ctrl+N</span></td><td class="desc-sm">Align</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+K</span></td><td class="desc-sm">FK Analysis</td><td class="key-sm"><span class="kbd">Ctrl+V</span></td><td class="desc-sm">Vespagram</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+L</span></td><td class="desc-sm">Plane Wave Loc</td><td class="key-sm"><span class="kbd">Ctrl+O</span></td><td class="desc-sm">2D Rotation</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+Shift+D</span></td><td class="desc-sm">Fit Diagnostics Window</td><td class="key-sm"><span class="kbd">Ctrl+A</span></td><td class="desc-sm">Select All</td></tr>
                <tr><td class="key-sm"><span class="kbd">Ctrl+Y</span></td><td class="desc-sm">Deselect All</td><td class="key-sm"><span class="kbd">Ctrl+D</span></td><td class="desc-sm">Delete Selected</td></tr>
            </table>
            </body>
            </html>
            """
        )
        layout.addWidget(viewer, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("Close", dlg)
        btn_close.setDefault(True)
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        return dlg

    def action_show_shortcuts_help(self) -> None:
        dlg = self._build_shortcuts_help_dialog()
        dlg.exec()

    def action_show_about(self) -> None:
        QMessageBox.about(
            self,
            "About SeismicHandler Motif (Modernized)",
            "SeismicHandler Motif (SHM) & Command Interpreter (SHC) v3.1\n\n"
            "Reverse-engineered and modernized from Klaus Stammler's BGR SeismicHandler\n"
            "using Python 3, PyQt6, NumPy, and Matplotlib.\n\n"
            "Supports Mini-SEED (Steim-1/2, SFD), SeedLink v3.1 streaming, StationXML, QuakeML 1.2, GeoJSON,\n"
            "GSE1/GSE2, SH Q-Files, ASCII traces, .evt files, SOS Butterworth & recursive IIR filters,\n"
            "trace-specific overlay filters, Brune/Boatwright source spectra, 3C polarization & beachball solver,\n"
            "array beamforming with slowness calibration, IASP91 phase ID, vector PDF/SVG bulletins, and session recovery.",
        )

