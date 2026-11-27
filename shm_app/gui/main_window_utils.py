"""Shared main-window utilities: command quoting, Qt excepthook, dark theme, and undo decorator."""

from __future__ import annotations

import functools
import os
import sys
import traceback
from typing import Any, Optional

from PyQt6 import sip
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox, QStyleFactory

from shm_app.gui.ui_helpers import apply_c_locale


def _quote_sh_arg(path: str) -> str:
    """Quote a filesystem path safely for SHCommandEngine._tokenize_command."""
    if '"' not in path:
        return f'"{path}"'
    if "'" not in path:
        return f"'{path}'"
    return '"' + path.replace('"', '\\"') + '"'


def _install_qt_excepthook() -> None:
    """Install a PyQt6-safe sys.excepthook once so uncaught slot errors warn instead of SIGABRT."""
    if QApplication.instance() is None:
        return
    if getattr(sys.excepthook, "_shm_excepthook_installed", False):
        return

    prev_hook = sys.excepthook

    def _shm_qt_excepthook(exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            prev_hook(exc_type, exc_value, exc_tb)
            return
        msg_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        sys.stderr.write("".join(msg_lines))
        sys.stderr.flush()

        app = QApplication.instance()
        if app is not None:
            active_win = app.activeWindow()
            for w in app.topLevelWidgets():
                _mw_cls = getattr(sys.modules.get("shm_app.gui.main_window"), "SHMMainWindow", None)
                if _mw_cls is not None and isinstance(w, _mw_cls) and not sip.isdeleted(w):
                    try:
                        w._append_console_log(f"Unhandled GUI error: {exc_type.__name__}: {exc_value}")
                        if hasattr(w, "status_bar") and w.status_bar is not None:
                            w.status_bar.showMessage(f"Error: {exc_value}", 6000)
                    except Exception:
                        pass
                    if active_win is None:
                        active_win = w
                    break
            is_offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            if (not is_offscreen) or os.environ.get("SHM_SHOW_EXCEPTHOOK_DIALOG") == "1":
                try:
                    QMessageBox.warning(
                        active_win,
                        "Unexpected SHM Error",
                        f"{exc_type.__name__}: {exc_value}",
                    )
                except Exception:
                    pass

    _shm_qt_excepthook._shm_excepthook_installed = True  # type: ignore[attr-defined]
    _shm_qt_excepthook._prev_hook = prev_hook  # type: ignore[attr-defined]
    sys.excepthook = _shm_qt_excepthook


def apply_dark_theme(app: Optional[QApplication]) -> None:
    """Force consistent Fusion style and dark QPalette across desktop environments."""
    apply_c_locale()
    if app is None:
        return
    if app.property("_shm_dark_theme_applied"):
        return
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#10141d"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#0d1117"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#21262d"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#58a6ff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#1f6feb"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Light, QColor("#3b434f"))
    pal.setColor(QPalette.ColorRole.Midlight, QColor("#2c333d"))
    pal.setColor(QPalette.ColorRole.Mid, QColor("#30363d"))
    pal.setColor(QPalette.ColorRole.Dark, QColor("#484f58"))
    pal.setColor(QPalette.ColorRole.Shadow, QColor("#090c10"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8b949e"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#484f58"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#484f58"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6e7681"))
    app.setPalette(pal)
    app.setProperty("_shm_dark_theme_applied", True)


def _undoable(label: str):
    """Decorator: snapshot traces and parameters before running a state-changing action."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            self._push_undo(label)
            return fn(self, *args, **kwargs)

        return wrapper

    return deco
