"""
A small dual-handle range slider - Qt has no built-in one. Plain QWidget +
QPainter (not OpenGL), so unlike the 3D view this renders and can be verified
normally in any environment, including this sandbox's software-GL setup.

Used for the Region of Interest theta/phi ranges (see viewer_widget.py) -
paired with, not replacing, the existing spinboxes: dragging a handle updates
the paired spinbox live and vice versa, so typing an exact number is still
available when that's more precise than dragging.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

_HANDLE_RADIUS = 7.0
_TRACK_MARGIN = _HANDLE_RADIUS + 3.0


class RangeSlider(QWidget):
    """A horizontal 0...x-------x...N slider with two independently
    draggable handles bounding [low, high] within [minimum, maximum].
    `rangeChanged(low, high)` fires continuously while dragging (matching the
    paired spinboxes' own live valueChanged behavior), not just on release."""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, minimum: float, maximum: float, low: float, high: float, parent=None) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._low = self._clamp(low)
        self._high = self._clamp(high)
        self._dragging: str | None = None  # 'low' | 'high' | None
        self.setMinimumHeight(26)
        self.setMinimumWidth(80)
        self.setMouseTracking(True)

    def values(self) -> tuple[float, float]:
        return self._low, self._high

    def setValues(self, low: float, high: float) -> None:
        """Programmatic update (e.g. from a paired spinbox) - does NOT
        re-emit rangeChanged, so callers can sync both directions without
        feedback loops; the caller already knows the new values."""
        low, high = self._clamp(low), self._clamp(high)
        if (low, high) == (self._low, self._high):
            return
        self._low, self._high = low, high
        self.update()

    def _clamp(self, v: float) -> float:
        return max(self._min, min(self._max, float(v)))

    def _value_to_x(self, value: float) -> float:
        span = self.width() - 2 * _TRACK_MARGIN
        if self._max <= self._min or span <= 0:
            return _TRACK_MARGIN
        frac = (value - self._min) / (self._max - self._min)
        return _TRACK_MARGIN + frac * span

    def _x_to_value(self, x: float) -> float:
        span = self.width() - 2 * _TRACK_MARGIN
        if span <= 0:
            return self._min
        frac = (x - _TRACK_MARGIN) / span
        frac = max(0.0, min(1.0, frac))
        return self._min + frac * (self._max - self._min)

    def paintEvent(self, ev) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mid_y = self.height() / 2

        painter.setPen(QPen(QColor("#aaaaaa"), 3, cap=Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(_TRACK_MARGIN, mid_y), QPointF(self.width() - _TRACK_MARGIN, mid_y))

        x_lo = self._value_to_x(self._low)
        x_hi = self._value_to_x(self._high)
        painter.setPen(QPen(QColor("#5b8fd6"), 4, cap=Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(x_lo, mid_y), QPointF(x_hi, mid_y))

        for x, active in ((x_lo, self._dragging == "low"), (x_hi, self._dragging == "high")):
            painter.setBrush(QColor("#5b8fd6") if active else QColor("#ffffff"))
            painter.setPen(QPen(QColor("#33507a"), 1.5))
            painter.drawEllipse(QPointF(x, mid_y), _HANDLE_RADIUS, _HANDLE_RADIUS)
        painter.end()

    def _nearest_handle(self, x: float) -> str:
        d_lo = abs(x - self._value_to_x(self._low))
        d_hi = abs(x - self._value_to_x(self._high))
        return "low" if d_lo <= d_hi else "high"

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        x = ev.position().x()
        self._dragging = self._nearest_handle(x)
        self._drag_to(x)
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if self._dragging is not None:
            self._drag_to(ev.position().x())
            ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if self._dragging is not None:
            self._dragging = None
            self.update()
            ev.accept()

    def _drag_to(self, x: float) -> None:
        value = self._x_to_value(x)
        if self._dragging == "low":
            self._low = min(value, self._high)
        else:
            self._high = max(value, self._low)
        self.update()
        self.rangeChanged.emit(self._low, self._high)
