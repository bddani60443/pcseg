from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QWidget,
)

_GUARDED_TYPES = (QSpinBox, QDoubleSpinBox, QComboBox, QSlider)


class _AppWheelGuard(QObject):
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False

        # Walk up from the ACTUAL event target - which may be an internal
        # child of a guarded control (e.g. a QSpinBox's own lineEdit()), not
        # the guarded control itself - see module docstring for why this,
        # not a findChildren()-based per-widget filter, is what's needed.
        guarded: QWidget | None = None
        w = obj if isinstance(obj, QWidget) else None
        while w is not None:
            if isinstance(w, _GUARDED_TYPES):
                guarded = w
                break
            w = w.parentWidget()
        if guarded is None:
            return False

        scroll_area = guarded.parentWidget()
        while scroll_area is not None and not isinstance(scroll_area, QAbstractScrollArea):
            scroll_area = scroll_area.parentWidget()
        if scroll_area is not None:
            QApplication.sendEvent(scroll_area.viewport(), event)
        return True  # swallow here either way - the guarded control itself never reacts


_installed_guard: _AppWheelGuard | None = None


def install_wheel_guard() -> _AppWheelGuard:
    """Installs the single application-wide wheel guard, if not already
    installed. Idempotent - safe to call more than once (e.g. once per
    Viewer3DWidget constructed), returns the same instance every time."""
    global _installed_guard
    app = QApplication.instance()
    assert app is not None, "install_wheel_guard() needs a QApplication to already exist"
    if _installed_guard is None:
        _installed_guard = _AppWheelGuard()
        app.installEventFilter(_installed_guard)
    return _installed_guard
