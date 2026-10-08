from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from pcseg.exceptions import SweepCancelled
from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, str)

    def __init__(self, fn: Callable, args: tuple, kwargs: dict) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            if "progress_callback" in inspect.signature(self._fn).parameters:
                kwargs["progress_callback"] = lambda pct, msg: self.progress.emit(pct, msg)
            result = self._fn(*self._args, **kwargs)
        except SweepCancelled:
            self.cancelled.emit()
            return
        except Exception as e:
            logging.exception(f"Background task failed: {self._fn}")
            self.error.emit(str(e))
            return
        self.finished.emit(result)


def run_in_background(
    fn: Callable,
    *args: Any,
    on_success: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_progress: Callable[[int, str], None] | None = None,
    on_cancelled: Callable[[], None] | None = None,
    **kwargs: Any,
) -> tuple[QThread, _Worker]:
    """
    Runs `fn(*args, **kwargs)` on a new QThread. If `fn` accepts a
    `progress_callback` keyword argument, one is injected automatically that
    forwards to `on_progress` on the main thread.

    If `fn` raises `pcseg.exceptions.SweepCancelled` (a Run Prompt/Auto-Segment
    sweep's own cooperative-cancellation signal, from its `cancel_check`
    callback returning True), `on_cancelled` fires instead of `on_error` - a
    deliberate user cancellation isn't a failure, callers shouldn't show an
    error dialog for it. Omit `on_cancelled` if `fn` never raises it (e.g.
    Pick Point's single non-sweep call).

    Returns (thread, worker) - the CALLER must keep both alive (e.g. as
    instance attributes) until `finished`/`error`/`cancelled` fires, or
    Python's GC can tear down the thread mid-run.
    """
    thread = QThread()
    worker = _Worker(fn, args, kwargs)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    worker.cancelled.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.error.connect(worker.deleteLater)
    worker.cancelled.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    worker.finished.connect(on_success)
    worker.error.connect(on_error)
    if on_progress is not None:
        worker.progress.connect(on_progress)
    if on_cancelled is not None:
        worker.cancelled.connect(on_cancelled)

    thread.start()
    return thread, worker
