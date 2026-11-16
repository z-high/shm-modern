"""Shared Qt styling and UI helper utilities across SHM dialogs and panels."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional
import warnings

from numpy.linalg import LinAlgError

from PyQt6.QtCore import QLocale
from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QPushButton, QWidget

_ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons").replace("\\", "/")

_DARK_QSS_TEMPLATE = """
QMainWindow, QDialog, QDockWidget, QWidget {
    background-color: #10141d;
    color: #e6edf3;
    font-size: 12px;
}
QMenuBar {
    background-color: #161b22;
    color: #e6edf3;
    border-bottom: 1px solid #30363d;
}
QMenuBar::item:selected, QMenu::item:selected {
    background-color: #1f6feb;
    color: #ffffff;
}
QMenu {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
    color: #79c0ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 10px;
    outline: none;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}
QPushButton:pressed {
    background-color: #161b22;
    border-color: #58a6ff;
}
QPushButton:focus, QPushButton:default {
    border-color: #58a6ff;
    outline: none;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}
QPushButton::menu-indicator {
    image: none;
    width: 0px;
}
QSizeGrip {
    width: 0px;
    height: 0px;
    background: transparent;
    image: none;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    background-color: #0d1117;
    border: 1px solid #484f58;
    border-radius: 3px;
}
QCheckBox::indicator:hover {
    border-color: #58a6ff;
}
QCheckBox::indicator:checked {
    background-color: #1f6feb;
    border: 1px solid #58a6ff;
    image: url(__ICONS__/check.svg);
}
QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit, QComboBox, QTextEdit, QListWidget, QTableWidget {
    background-color: #0d1117;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QTimeEdit:focus, QDateTimeEdit:focus, QComboBox:focus {
    border-color: #58a6ff;
}
QSpinBox, QDoubleSpinBox, QTimeEdit {
    padding-right: 22px;
}
QDateEdit, QDateTimeEdit, QComboBox {
    padding-right: 24px;
}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: border;
    width: 18px;
    background-color: #21262d;
    border-left: 1px solid #30363d;
}
QAbstractSpinBox::up-button {
    subcontrol-position: top right;
    border-top-right-radius: 2px;
    border-bottom: 1px solid #30363d;
}
QAbstractSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 2px;
}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {
    background-color: #30363d;
}
QAbstractSpinBox::up-arrow {
    image: url(__ICONS__/arrow-up.svg);
    width: 10px;
    height: 10px;
}
QAbstractSpinBox::down-arrow {
    image: url(__ICONS__/arrow-down.svg);
    width: 10px;
    height: 10px;
}
QDateEdit::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    background-color: #21262d;
    border-left: 1px solid #30363d;
    border-top-right-radius: 2px;
    border-bottom-right-radius: 2px;
}
QDateEdit::drop-down:hover {
    background-color: #30363d;
}
QDateEdit::down-arrow {
    image: url(__ICONS__/calendar.svg);
    width: 12px;
    height: 12px;
}
QTimeEdit::down-arrow {
    image: url(__ICONS__/arrow-down.svg);
    width: 10px;
    height: 10px;
}
QComboBox::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    background-color: #21262d;
    border-left: 1px solid #30363d;
    border-top-right-radius: 2px;
    border-bottom-right-radius: 2px;
}
QComboBox::drop-down:hover {
    background-color: #30363d;
}
QComboBox::down-arrow {
    image: url(__ICONS__/arrow-down.svg);
    width: 10px;
    height: 10px;
}
QCalendarWidget QWidget {
    alternate-background-color: #161b22;
    background-color: #10141d;
    color: #e6edf3;
}
QCalendarWidget QWidget#qt_calendar_navigationbar {
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 2px;
}
QCalendarWidget QToolButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 3px;
    padding: 3px 6px;
    margin: 2px;
}
QCalendarWidget QToolButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}
QCalendarWidget QToolButton#qt_calendar_prevmonth {
    qproperty-icon: url(__ICONS__/arrow-left.svg);
    qproperty-iconSize: 12px 12px;
}
QCalendarWidget QToolButton#qt_calendar_nextmonth {
    qproperty-icon: url(__ICONS__/arrow-right.svg);
    qproperty-iconSize: 12px 12px;
}
QCalendarWidget QAbstractItemView:enabled {
    background-color: #0d1117;
    color: #e6edf3;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    outline: none;
}
QCalendarWidget QAbstractItemView:disabled {
    color: #484f58;
}
QToolBar {
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    spacing: 4px;
}
QToolButton#live_toggle_btn:checked {
    background-color: #1b4d2e;
    color: #e6ffed;
    border: 1px solid #3fb950;
    border-radius: 3px;
    font-weight: bold;
}
QPushButton#live_notify_btn {
    background-color: #1b4d2e;
    color: #e6ffed;
    border: 1px solid #3fb950;
    border-radius: 4px;
    padding: 2px 10px;
    font-weight: bold;
}
QPushButton#live_notify_btn:hover {
    background-color: #23633c;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #10141d;
}
QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #21262d;
    color: #e6edf3;
    border-bottom: 2px solid #58a6ff;
}
QHeaderView::section {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    padding: 4px 6px;
    font-weight: bold;
}
QScrollBar:horizontal {
    background: #0d1117;
    height: 14px;
    border-top: 1px solid #30363d;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background: #3b434f;
    min-width: 28px;
    border-radius: 4px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background: #58a6ff;
}
QScrollBar::handle:horizontal:disabled {
    background: #21262d;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0px;
}
QScrollBar:vertical {
    background: #0d1117;
    width: 14px;
    border-left: 1px solid #30363d;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #3b434f;
    min-height: 28px;
    border-radius: 4px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: #58a6ff;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0px;
}
QSlider::groove:horizontal {
    background: #21262d;
    border: 1px solid #30363d;
    height: 6px;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #1f6feb;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #58a6ff;
    border: 1px solid #0d1117;
    width: 12px;
    margin: -4px 0px;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #79c0ff;
}
"""


def get_dark_stylesheet() -> str:
    """Return the global dark observatory Qt stylesheet."""
    return _DARK_QSS_TEMPLATE.replace("__ICONS__", _ICONS_DIR)


def widen_combo(combo: QComboBox, min_width: int = 180, popup_width: int = 240) -> QComboBox:
    """Ensure a QComboBox and its dropdown popup are wide enough for readable labels."""
    combo.setMinimumWidth(min_width)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    view = combo.view()
    if view is not None:
        view.setMinimumWidth(max(min_width, popup_width))
    return combo


def disable_button_autodefault(root: QWidget) -> None:
    """Prevent Enter-focus default button artifacts across all QPushButton children."""
    for btn in root.findChildren(QPushButton):
        btn.setAutoDefault(False)
        btn.setDefault(False)


def apply_c_locale(root: Optional[QWidget] = None) -> None:
    """Ensure C locale (dot decimals) globally and on all spinbox/date/time controls."""
    c_loc = QLocale.c()
    QLocale.setDefault(c_loc)
    if root is not None:
        for sb in root.findChildren(QAbstractSpinBox):
            sb.setLocale(c_loc)


def add_dark_colorbar(fig: Any, mappable: Any, ax: Any, label: str) -> Any:
    """Attach a dark-themed Matplotlib colorbar matching the observatory UI palette."""
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label, color="#e6edf3")
    cbar.ax.yaxis.set_tick_params(color="#8b949e", labelcolor="#8b949e")
    cbar.ax.tick_params(colors="#8b949e", labelcolor="#8b949e")
    cbar.outline.set_edgecolor("#30363d")
    return cbar


def make_push_button(
    text: str,
    slot: Optional[Callable[..., Any]] = None,
    *,
    checkable: bool = False,
    tooltip: str = "",
) -> QPushButton:
    """Create a QPushButton with autoDefault disabled and optional click binding."""
    btn = QPushButton(text)
    btn.setAutoDefault(False)
    btn.setDefault(False)
    if checkable:
        btn.setCheckable(True)
    if tooltip:
        btn.setToolTip(tooltip)
    if slot is not None:
        btn.clicked.connect(slot)
    return btn


def safe_tight_layout(fig: Any, canvas: Optional[Any] = None, min_px: int = 80) -> None:
    """Apply Matplotlib tight_layout only when the target canvas is large enough to avoid singular transforms."""
    if fig is None:
        return
    if canvas is not None:
        try:
            if canvas.width() < min_px or canvas.height() < min_px:
                return
        except Exception:
            pass
    try:
        dpi = float(fig.dpi)
        if fig.get_figwidth() * dpi < min_px or fig.get_figheight() * dpi < min_px:
            return
    except Exception:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except (LinAlgError, ValueError, RuntimeError):
            pass
