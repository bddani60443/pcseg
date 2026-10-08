from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path.home() / ".pcseg-view" / "logs"
LOG_FILE = LOG_DIR / "pcseg-view.log"              # every logging.* call, INFO and up - includes full excepthook tracebacks
NATIVE_CRASH_FILE = LOG_DIR / "native_crash.log"   # faulthandler's raw dump - segfaults/native crashes a Python hook can't see

_native_crash_fh = None  # kept open + referenced for the app's whole lifetime - see module docstring, point 3


def install() -> Path:
    """Call once, as early as possible in main() - after logging.basicConfig()
    (so the existing console handler is set up first; basicConfig() is a
    no-op once the root logger already has handlers, so doing this before it
    would silently break console logging) but before QApplication/any GL
    setup, so even a crash during Qt/GL initialization itself gets caught.
    Returns LOG_FILE, so the caller can also mention it wherever's visible."""
    global _native_crash_fh
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    _native_crash_fh = open(NATIVE_CRASH_FILE, "a", encoding="utf-8")
    _native_crash_fh.write(f"\n--- session started {datetime.now().isoformat()} ---\n")
    _native_crash_fh.flush()
    faulthandler.enable(file=_native_crash_fh, all_threads=True)

    sys.excepthook = _excepthook
    threading.excepthook = _threading_excepthook

    logging.info(f"Crash log: {LOG_FILE}")
    logging.info(f"Native-crash log (segfaults/driver crashes): {NATIVE_CRASH_FILE}")
    return LOG_FILE


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    logging.critical(
        "Uncaught exception on the main thread:\n"
        + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    )
    _show_crash_dialog(exc_type, exc_value)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _threading_excepthook(args: "threading.ExceptHookArgs") -> None:
    thread_name = args.thread.name if args.thread is not None else "?"
    logging.critical(
        f"Uncaught exception on thread '{thread_name}':\n"
        + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    )


def _show_crash_dialog(exc_type, exc_value) -> None:
    """Best-effort only - the process may already be in a bad state by the
    time this runs, so it must never itself raise or block the crash from
    being reported/logged above."""
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is None:
            return
        QMessageBox.critical(
            None, "pcseg-view - unexpected error",
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"Details were written to:\n{LOG_FILE}\n\n"
            "Please attach that file if you report this.",
        )
    except Exception:
        pass
