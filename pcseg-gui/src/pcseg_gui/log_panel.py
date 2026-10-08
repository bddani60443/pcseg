from __future__ import annotations

from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

MAX_LINES = 5000  # caps unbounded growth over a long session


class LogPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(MAX_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.text)

        # Standard log-viewer behavior: stick to the bottom on new output, but
        # stop doing that the moment the user scrolls up to read something -
        # otherwise a fast-arriving sweep would yank the view out from under them.
        self._auto_scroll = True
        self.text.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    def _on_scrolled(self, value: int) -> None:
        bar = self.text.verticalScrollBar()
        self._auto_scroll = value >= bar.maximum() - 2

    def append_line(self, message: str) -> None:
        self.text.appendPlainText(message)
        if self._auto_scroll:
            bar = self.text.verticalScrollBar()
            bar.setValue(bar.maximum())

    def clear(self) -> None:
        self.text.clear()
        self._auto_scroll = True
