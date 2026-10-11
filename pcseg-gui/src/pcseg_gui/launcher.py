# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
Entry point of the `pcseg-view` command. The viewer's own dependencies are an
optional extra, so a plain `pip install pcseg` still installs this command -
check for them first and say how to get them, instead of failing with a bare
ImportError from deep inside the Qt/OpenGL imports.
"""
from __future__ import annotations

import importlib.util
import sys

GUI_MODULES = {"PyQt6": "PyQt6", "pyqtgraph": "pyqtgraph", "OpenGL": "PyOpenGL"}


def missing_gui_packages() -> list[str]:
    missing = []
    for module, package in GUI_MODULES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    return missing


def main() -> int:
    missing = missing_gui_packages()
    if missing:
        print(f"pcseg-view needs the optional GUI dependencies (missing: {', '.join(missing)}).\n"
              f"Install them from the pcseg directory with:  pip install -e \".[gui]\"", file=sys.stderr)
        return 1
    from pcseg_gui.main import main as run_viewer

    return run_viewer()


if __name__ == "__main__":
    sys.exit(main())
