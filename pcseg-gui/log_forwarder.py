from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal


class LogForwarder(QObject):
    message = pyqtSignal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self, forwarder: LogForwarder) -> None:
        super().__init__()
        self._forwarder = forwarder

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._forwarder.message.emit(msg)


def attach_log_forwarder(forwarder: LogForwarder, logger: logging.Logger | None = None) -> _QtLogHandler:
    """Adds a handler forwarding this logger's records to `forwarder.message`,
    alongside whatever handlers are already there (e.g. main.py's console
    logging.basicConfig) - doesn't replace or reconfigure anything. Returns the
    handler so the caller can detach_log_forwarder() it once the task that
    wanted live status text is done, rather than forwarding for the rest of
    the app's lifetime."""
    logger = logger or logging.getLogger()
    handler = _QtLogHandler(forwarder)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return handler


def detach_log_forwarder(handler: _QtLogHandler, logger: logging.Logger | None = None) -> None:
    (logger or logging.getLogger()).removeHandler(handler)
