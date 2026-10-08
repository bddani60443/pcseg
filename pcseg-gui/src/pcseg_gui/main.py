"""
pcseg-view - minimal 3D viewer/screenshot tool for pcseg point clouds and
segmentation results. This is a working tool for producing figures, not a
citable research artifact.

Usage:
    pcseg-view [path-to-scan.e57-or-result.las] [--decimation N]
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

from pcseg_gui.crash_log import install as install_crash_logging
from pcseg_gui.main_window import MainWindow


def _configure_gl_backend() -> None:
    """
    Must run before QApplication is constructed. On Windows, Qt's default
    OpenGL backend is often ANGLE (a Direct3D-backed OpenGL ES translation
    layer) rather than the native driver - ANGLE has no legacy fixed-function
    matrix stack at all, so pyqtgraph's own GLViewWidget.setProjection() (which
    calls glMatrixMode/glLoadIdentity internally, independent of this app's own
    shaders) fails with GL_INVALID_OPERATION no matter what QSurfaceFormat
    profile is requested. Forcing desktop OpenGL avoids that; Linux/macOS
    already default to a native GL context, so this is a no-op there.
    """
    if platform.system() == "Windows":
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)


def _configure_gl_surface_format() -> None:
    """
    Must run before QApplication is constructed. Without an explicit format,
    Qt's default QSurfaceFormat negotiates a context that reports as OpenGL
    <2.1 to pyqtgraph's own version check (ctx.format().version()) even on a
    driver that actually supports much more - pyqtgraph then silently skips
    GL initialization and nothing ever renders. Compatibility Profile 3.0
    matches this app's #version 130 shaders.

    On Windows, Qt creates contexts through WGL_ARB_create_context. Without
    explicitly requesting DeprecatedFunctions, NVIDIA's driver hands back a
    forward-compatible context that has dropped the legacy fixed-function
    matrix stack (glMatrixMode etc.) even though CompatibilityProfile was
    requested - pyqtgraph's GLViewWidget still calls those internally, so
    every paintGL raised GL_INVALID_OPERATION (confirmed on Windows/NVIDIA).
    Harmless to request on Linux/macOS too, so this isn't Windows-gated.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 0)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setOption(QSurfaceFormat.FormatOption.DeprecatedFunctions, True)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcseg-view", description=__doc__)
    parser.add_argument("path", nargs="?", help="Path to a .e57 scan or a pcseg-exported .las result.")
    parser.add_argument("--decimation", type=int, default=2,
                         help="E57 grid decimation step (ignored for .las; default: 2).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # After basicConfig() (which no-ops if the root logger already has a
    # handler - installing crash logging first would silently kill console
    # output) but before any Qt/GL setup, so even a crash during THAT gets caught.
    install_crash_logging()

    _configure_gl_backend()
    _configure_gl_surface_format()
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()

    if args.path:
        window.open_file(args.path, decimation=args.decimation)
        window._last_decimation = args.decimation

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
