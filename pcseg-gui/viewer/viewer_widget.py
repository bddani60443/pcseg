from __future__ import annotations

import html
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import OpenGL.GL as gl
import pyqtgraph.opengl as gl_module
from core.scan_node import ScanNode
from PyQt6.QtCore import QPoint, QRect, QRectF, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
    QVector3D,
)
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.Qt import QtCore as pg_QtCore
from pyqtgraph.Qt import QtGui as pg_QtGui

from layers import Layer, LayerInstance
from log_forwarder import (
    LogForwarder,
    attach_log_forwarder,
    detach_log_forwarder,
)
from viewer.layer_panel import LayerPanel
from viewer.point_cloud_item import (
    COLOR_MODES,
    RENDER_MODES,
    ShaderPointCloudItem,
)
from viewer.range_slider import RangeSlider
from viewer.view_cube import FACE_LABELS, ViewCubeOverlay, face_snap_target
from wheel_guard import install_wheel_guard
from workers import run_in_background

CHANNEL_CHOICES = ["auto", "rgb", "intensity", "normals"]
DEFAULT_BACKGROUND_COLOR = "#202020"
# Committed instance selection: a deeper, more saturated blue - the "usual"
# selected-item color most OS/app UIs use (matches Windows' own default
# accent blue, since that's what this app is mostly tested on). Deliberately
# a different shade from HOVER_COLOR below (paler/lighter) so a hovered vs.
# an actually-selected instance stay visually distinguishable even though
# both read as "blue" per the same design request.
SELECTION_COLOR = (0, 120, 215)
# Live hover preview - see ShaderPointCloudItem.hover_preview_points's own
# matching default and Viewer3DWidget._on_hover_moved/_update_instance_overlay.
HOVER_COLOR = (100, 170, 255)


def _to_unit_color(rgb_255: tuple[int, int, int]) -> tuple[float, float, float]:
    """0-255 int RGB (SELECTION_COLOR/HOVER_COLOR's own convention, matching
    highlight_points'/hover_preview_points' `color` arg) -> 0-1 float RGB,
    what the outline shader's `u_color` uniform expects (see
    ShaderPointCloudItem.set_selection_outline/set_hover_outline)."""
    return (rgb_255[0] / 255.0, rgb_255[1] / 255.0, rgb_255[2] / 255.0)


def _union_indices(index_arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenates several instances' own `indices` arrays for
    highlight_points/set_selection_outline - deliberately NOT deduplicated
    via np.unique (real, reported freeze on selecting one large detected
    object: np.unique sorts, ~1 SECOND for a single ~2M-point instance on a
    20M-point scan, vs ~1ms for a plain concatenate - see git history for
    the profiling that found this). A duplicate index (two selected
    instances sharing a point, or the same instance appearing twice) is
    harmless here: highlight_points/clear_highlight round-trip correctly
    through fancy-index gather/scatter regardless (every duplicate position
    reads/writes the SAME value, since they all key off the same original
    point), and the outline pass just draws that one point's mask twice, not
    incorrectly. The single-array case (by far the most common - one clicked
    instance) skips the concatenate call entirely too, returning the array
    itself with no copy at all, matching hover_preview_points' own zero-copy
    cost for the identical situation."""
    if len(index_arrays) == 1:
        return index_arrays[0]
    return np.concatenate(index_arrays)


@dataclass
class _PendingPick:
    """One accepted-from-the-candidate-menu pick that ISN'T in the target
    layer yet - see Viewer3DWidget._pending_picks. Deliberately a separate
    type from LayerInstance rather than reusing it directly: a pending pick
    only ever becomes a real LayerInstance once Accept commits it (see
    _accept_pending_picks), and keeping the two distinct makes "is this
    real yet" a type-level fact, not a flag to check."""
    indices: np.ndarray
    label: str
    color: tuple
    score: float | None


class PickableGLView(gl_module.GLViewWidget):
    """Emits point_clicked on right mouse press; interaction-mode gating (View
    vs. Pick) lives in Viewer3DWidget, not here - this widget only reports
    the click, it doesn't decide what a click means.

    LEFT button is ALWAYS selection (no separate toggle), never navigation:
    plain click selects the instance under the cursor; plain drag
    rubber-band box-selects every instance the box touches. Click-vs-drag is
    told apart purely by how far the mouse moves before release: stays
    within `_CLICK_DRAG_THRESHOLD_PX` -> click; past it -> drag.

    Navigation lives entirely on MIDDLE button instead (Blender-style, no
    modifier needed for the everyday case): plain MMB-drag orbits;
    Shift+MMB-drag pans in the screen x/y plane; Ctrl+MMB-drag pans in the
    other (mostly-vertical) plane - the exact same two pan variants
    pyqtgraph's own default MMB/Ctrl+MMB binding offered, just moved one
    modifier level up now that unmodified MMB means orbit instead of the
    weaker of the two pans. Driven manually here (not forwarded to
    super().mouseMoveEvent()) since pyqtgraph's own hardcoded button
    dispatch has no "plain MMB = orbit" case to reuse - it only orbits on
    LEFT - though the underlying orbit()/pan() math is the exact same calls
    pyqtgraph's own LMB/MMB handling already used, just re-triggered here.

    Also owns the corner ViewCube overlay (see view_cube.py) - drawn as an
    extra pass after the normal scene paint, and hit-tested directly against
    raw mouse events here (ahead of the normal right-click-pick / left-drag
    handling below), since it's a fixed screen-space overlay unrelated to
    anything in the 3D scene itself.

    Also emits `hover_moved` on every plain (no-button) mouse move and
    `hover_left` on leaving the widget - Viewer3DWidget uses these for a
    live "if I clicked now, this is what I'd select" hover preview (see its
    own `_on_hover_moved`); this widget doesn't decide what hovering means
    any more than it decides what a click means."""

    _CLICK_DRAG_THRESHOLD_PX = 4
    _deselect_badge_cursor_cache: QCursor | None = None  # lazy class-level singleton - see _deselect_badge_cursor()

    point_clicked = pyqtSignal(QPoint)
    cube_face_clicked = pyqtSignal(int)
    instance_click = pyqtSignal(QPoint, bool)  # (position, shift_held)
    box_select = pyqtSignal(QRect, bool)  # (rect, shift_held)
    escape_pressed = pyqtSignal()
    delete_requested = pyqtSignal()
    hover_moved = pyqtSignal(QPoint)  # mouse moved with no button held - live "what would I select" preview
    hover_left = pyqtSignal()  # cursor left the 3D view entirely - clear any hover preview
    shift_toggled = pyqtSignal(bool)  # Shift pressed/released with no mouse movement involved
    recenter_requested = pyqtSignal()  # middle-button double-click - see mouseDoubleClickEvent

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.view_cube = ViewCubeOverlay()
        self.setMouseTracking(True)  # needed to get moveEvents with no button held, for hover
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # needed to actually receive key events (Escape et al)
        self._cube_press_face: int | None = None  # set while a LEFT press that hit the cube is held
        self._mid_press_pos = None  # QPointF, set while a MIDDLE orbit/pan-drag is held
        self._click_press_pos: QPoint | None = None  # set while a plain LEFT press is pending click-vs-box-drag
        self._click_became_drag = False
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.roi_overlay = None  # _ROIOverlayPanel, set by Viewer3DWidget._build_roi_overlay

    @classmethod
    def _deselect_badge_cursor(cls) -> QCursor:
        """The normal arrow cursor with a small minus badge added at its
        bottom-right - NOT a full replacement icon (an earlier version swapped
        the whole cursor for a solid glyph; reported as not reading as "the
        cursor" any more, just a random icon). Qt can't composite a custom
        pixmap on top of the OS's own native arrow (a QCursor pixmap IS the
        whole cursor), so this draws its own approximation of a plain arrow
        and adds the badge to the same image - close enough to the real thing
        that the badge reads as "attached to the cursor" rather than "instead
        of it." Hollow/outline badge (reported: a solid red disc read as an
        alarming "error" glyph, too heavy for a routine deselect indicator) -
        just a thin ring with the minus inside, transparent center so it
        doesn't block the view under the cursor either.

        Shown while: (a) plain hover, only when the hovered point actually
        belongs to a currently-selected instance (see
        Viewer3DWidget._update_deselect_badge) - not just "Shift is held
        somewhere over the view," which said nothing about whether a click
        would actually remove anything; or (b) an active Shift+drag box-select
        (see PickableGLView.mouseMoveEvent's box-drag branch), where there's
        no single hovered instance to check against, so it's shown for the
        whole drag purely on Shift being held."""
        if cls._deselect_badge_cursor_cache is None:
            size = 24
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            arrow = pg_QtGui.QPolygonF([
                pg_QtCore.QPointF(1, 1), pg_QtCore.QPointF(1, 15), pg_QtCore.QPointF(5, 11.5),
                pg_QtCore.QPointF(7.5, 16.5), pg_QtCore.QPointF(9.5, 15.5), pg_QtCore.QPointF(7, 10.5),
                pg_QtCore.QPointF(12, 10),
            ])
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(QColor(20, 20, 20), 1.2))
            painter.drawPolygon(arrow)

            badge_cx, badge_cy, badge_r = 18, 18, 5
            painter.setBrush(Qt.BrushStyle.NoBrush)  # hollow - no fill
            painter.setPen(QPen(QColor(40, 40, 40), 1.3))
            painter.drawEllipse(pg_QtCore.QPointF(badge_cx, badge_cy), badge_r, badge_r)
            painter.setPen(QPen(QColor(255, 255, 255), 1.0))
            painter.drawEllipse(pg_QtCore.QPointF(badge_cx, badge_cy), badge_r - 1.1, badge_r - 1.1)
            painter.setPen(QPen(QColor(40, 40, 40), 1.6))
            painter.drawLine(
                pg_QtCore.QPointF(badge_cx - 2.3, badge_cy), pg_QtCore.QPointF(badge_cx + 2.3, badge_cy),
            )
            painter.end()
            cls._deselect_badge_cursor_cache = QCursor(pixmap, 1, 1)  # hotspot at the arrow tip
        return cls._deselect_badge_cursor_cache

    _pan_plane_cursor_cache: dict[bool, QCursor] = {}  # keyed by `vertical` - see _pan_plane_cursor()

    @classmethod
    def _pan_plane_cursor(cls, vertical: bool) -> QCursor:
        """A small 3-axis tripod (isometric layout - X right-down, Y
        left-down, Z straight up, the standard "axis gizmo" convention most
        3D tools already use, so it reads as genuinely perpendicular axes
        rather than 2 arbitrary flat shapes) - shown as the cursor ONLY
        while a middle-drag pan is actually in progress (see
        mouseMoveEvent's middle-drag branch and mouseReleaseEvent's reset),
        not just while the modifier is held. Shift pans the X/Y plane, Ctrl
        pans the X/Z plane (see PickableGLView's own class docstring) - the
        two axes of whichever plane is CURRENTLY active are drawn bold and
        colored, with a faint line connecting their tips as a plane hint;
        the third, uninvolved axis stays a thin gray line - so which pan
        variant is active reads at a glance, without losing the other two
        axes as a spatial reference. Same reasoning as _deselect_badge_cursor
        for explicit (not OS-theme) colors: this floats over an arbitrary,
        user-controlled 3D scene background."""
        if vertical in cls._pan_plane_cursor_cache:
            return cls._pan_plane_cursor_cache[vertical]

        size = 28
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = size // 2, size // 2 + 2
        center = pg_QtCore.QPointF(cx, cy)
        length = 11.0

        def _tip(dx: float, dy: float) -> "pg_QtCore.QPointF":
            return pg_QtCore.QPointF(cx + dx * length, cy + dy * length)

        x_tip = _tip(0.87, 0.5)    # isometric right-down
        y_tip = _tip(-0.87, 0.5)   # isometric left-down
        z_tip = _tip(0.0, -1.0)    # straight up

        # Shift (vertical=False) pans X/Y (matches mouseMoveEvent's
        # pan(dx, dy, 0)); Ctrl (vertical=True) pans X/Z (pan(dx, 0, dy)) -
        # X is always one of the two active axes either way.
        active_tips = (x_tip, z_tip) if vertical else (x_tip, y_tip)
        dim_tip = y_tip if vertical else z_tip

        dim_pen = QPen(QColor(210, 210, 210, 190), 1.2)
        plane_hint_pen = QPen(QColor(90, 170, 255, 130), 1.0)
        active_pen = QPen(QColor(90, 170, 255, 255), 2.2)

        # Faint line between the active plane's two tips - hints "these two
        # axes span a plane" without drawing a full flat shape.
        painter.setPen(plane_hint_pen)
        painter.drawLine(active_tips[0], active_tips[1])

        # Dim (inactive) axis drawn first, so the active ones layer on top
        # at the shared center point.
        painter.setPen(dim_pen)
        painter.drawLine(center, dim_tip)

        painter.setPen(active_pen)
        painter.drawLine(center, active_tips[0])
        painter.drawLine(center, active_tips[1])

        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(40, 40, 40), 1.0))
        painter.drawEllipse(center, 1.8, 1.8)

        painter.end()
        cursor = QCursor(pixmap, cx, cy)  # hotspot at the shared center, not the pixmap's own center
        cls._pan_plane_cursor_cache[vertical] = cursor
        return cursor

    def set_deselect_badge(self, active: bool) -> None:
        """Called by Viewer3DWidget (the only thing that knows whether the
        currently-hovered point is part of a SELECTED instance) - this class
        just draws whatever it's told, same division of labor as
        instance_click/box_select not deciding what a click means."""
        if active:
            self.setCursor(self._deselect_badge_cursor())
        else:
            self.unsetCursor()

    def paintGL(self) -> None:
        super().paintGL()
        ratio = self.devicePixelRatio()  # matches _pick_point_index's own convention
        self.view_cube.paint(
            gl, self.opts["azimuth"], self.opts["elevation"],
            int(self.width() * ratio), int(self.height() * ratio), ratio,
        )

    def _cube_hit_test(self, ev) -> int | None:
        pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        return self.view_cube.hit_test(
            pos.x(), pos.y(), self.width(), self.height(), self.opts["azimuth"], self.opts["elevation"],
        )

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            face = self._cube_hit_test(ev)
            if face is not None:
                # Standard "clickable button" semantics: the press is fully
                # consumed by the cube - deliberately does NOT call
                # super().mousePressEvent(ev), so pyqtgraph's own orbit-drag
                # state (self.mousePos) never gets initialized for this press.
                # The actual snap action fires on release, not press (see
                # mouseReleaseEvent below), so a press-then-drag-off-the-cube
                # cancels instead of acting.
                self._cube_press_face = face
                ev.accept()
                return
            # Whether this press ends up being a click (select) or a drag
            # (box-select) isn't decided yet - mouseMoveEvent decides once it
            # sees how far the mouse actually travels. Deliberately does NOT
            # call super().mousePressEvent(ev): LEFT never navigates any more
            # (navigation moved to MIDDLE, see class docstring), so
            # pyqtgraph's own orbit-drag state must never be touched here.
            self._click_press_pos = ev.pos()
            self._click_became_drag = False
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.MiddleButton:
            # Deliberately does NOT call super().mousePressEvent(ev) either -
            # this class drives orbit()/pan() itself in mouseMoveEvent (see
            # class docstring for why pyqtgraph's own dispatch can't be
            # reused here), so seeding ITS internal state would be pointless.
            self.set_deselect_badge(False)  # Shift here means pan, not deselect - see mouseMoveEvent
            self._mid_press_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.RightButton:
            self.point_clicked.emit(ev.pos())
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._mid_press_pos is not None:
            lpos = ev.position() if hasattr(ev, "position") else ev.localPos()
            diff = lpos - self._mid_press_pos
            self._mid_press_pos = lpos
            mods = ev.modifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                self.pan(diff.x(), diff.y(), 0, relative="view-upright")
                self.setCursor(self._pan_plane_cursor(vertical=False))
            elif mods & Qt.KeyboardModifier.ControlModifier:
                self.pan(diff.x(), 0, diff.y(), relative="view-upright")
                self.setCursor(self._pan_plane_cursor(vertical=True))
            else:
                self.orbit(-diff.x(), diff.y())
                self.unsetCursor()  # plain orbit - no pan-plane cursor, see class docstring
            ev.accept()
            return
        if self._click_press_pos is not None:
            if not self._click_became_drag:
                moved = (ev.pos() - self._click_press_pos).manhattanLength()
                if moved <= self._CLICK_DRAG_THRESHOLD_PX:
                    ev.accept()
                    return  # still within click tolerance - not a drag (yet)
                self._click_became_drag = True
                self._rubber_band.setGeometry(QRect(self._click_press_pos, QSize()))
                self._rubber_band.show()
            self._rubber_band.setGeometry(QRect(self._click_press_pos, ev.pos()).normalized())
            # Box-select has no single hovered instance to gate on (unlike
            # plain hover - see Viewer3DWidget._update_deselect_badge), so
            # show the badge for the whole drag purely on Shift being held.
            self.set_deselect_badge(bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            ev.accept()
            return
        if self._cube_press_face is not None:
            # A cube-owned press is being held - keep the hover highlight live
            # (nice-to-have) but deliberately do NOT call
            # super().mouseMoveEvent(ev) at all: that's exactly what would
            # otherwise feed pyqtgraph's own orbit-drag math and rotate the
            # main camera through/under the cube while the button is held -
            # confirmed live as the reported bug before this fix.
            face = self._cube_hit_test(ev)
            if face != self.view_cube.hovered_face:
                self.view_cube.hovered_face = face
                self.update()
            ev.accept()
            return
        face = self._cube_hit_test(ev)
        if face != self.view_cube.hovered_face:
            self.view_cube.hovered_face = face
            self.update()
        self.hover_moved.emit(ev.pos())
        super().mouseMoveEvent(ev)  # harmless with no button held - see GLViewWidget.mouseMoveEvent

    def leaveEvent(self, ev) -> None:
        self.set_deselect_badge(False)
        self.hover_left.emit()
        super().leaveEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        """Middle-button double-click re-centers the view (see
        Viewer3DWidget.center_camera) - reported live that panning/zooming
        over a session gradually leaves the ORBIT PIVOT (pyqtgraph's own
        opts['center']) drifted away from the scan itself, since panning
        moves that same point along with the view; a quick, standard 3D-
        tool gesture (most viewers bind SOME form of "recenter" to a
        double-click) to snap it back without hunting for the toolbar
        button. Only resets the pivot itself, not zoom distance or
        camera angle - center_camera() doesn't touch either of those."""
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.recenter_requested.emit()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.MiddleButton and self._mid_press_pos is not None:
            self._mid_press_pos = None
            self.unsetCursor()  # pan-plane cursor (if any) only shows DURING the drag
            ev.accept()
            self.update()
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._click_press_pos is not None:
            was_drag = self._click_became_drag
            rect = QRect(self._click_press_pos, ev.pos()).normalized()
            pos = ev.pos()
            self._click_press_pos = None
            self._click_became_drag = False
            self._rubber_band.hide()
            # The drag-only badge (set unconditionally on Shift during the
            # drag - see mouseMoveEvent) needs an explicit clear here: the
            # box_select handler has early-return paths (no layer selected,
            # box touched nothing) that never reach the instance-aware
            # _update_deselect_badge downstream, which would otherwise leave
            # a stale badge showing after release.
            self.set_deselect_badge(False)
            ev.accept()
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if was_drag:
                self.box_select.emit(rect, shift)
                self.update()
            else:
                self.instance_click.emit(pos, shift)
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._cube_press_face is not None:
            pressed_face = self._cube_press_face
            self._cube_press_face = None
            if self._cube_hit_test(ev) == pressed_face:
                self.cube_face_clicked.emit(pressed_face)
            # cube_face_clicked's own handler (Viewer3DWidget._on_cube_face_clicked)
            # already calls update() itself after re-orienting the camera, but
            # force one here too regardless (harmless if redundant) - real
            # usage reported the cube visually not settling into its new
            # orientation after a click, and this repaint is the one place
            # both the main scene AND the cube (PickableGLView.paintGL) get
            # redrawn from the live, just-updated camera state.
            self.update()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)
        self.update()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            ev.accept()
            return
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_requested.emit()
            ev.accept()
            return
        if ev.key() == Qt.Key.Key_Shift and not ev.isAutoRepeat():
            # Covers toggling Shift while the mouse is already sitting still
            # over an instance (mouseMoveEvent-driven hover only re-evaluates
            # on actual movement) - requires this widget to actually have
            # focus, which StrongFocus + a prior click/hover normally gives
            # it. Doesn't decide the badge itself - Viewer3DWidget is the one
            # that knows whether the currently-hovered point is even part of
            # a selected instance (see shift_toggled/set_deselect_badge).
            self.shift_toggled.emit(True)
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Shift and not ev.isAutoRepeat():
            self.shift_toggled.emit(False)
        super().keyReleaseEvent(ev)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self.roi_overlay is not None:
            self.roi_overlay.reposition()


class _ROIOverlayPanel(QFrame):
    """Floating Region-of-interest panel drawn ON TOP of the 3D view itself -
    a real child QWidget of PickableGLView (QOpenGLWidget composites plain
    child widgets over its own GL content correctly, same as any other
    parent/child pair) rather than a control living in the right-hand Run
    settings panel. Moved here (design review 2026-09-11) after ROI was
    reported as feeling out of place sharing a panel with "what the next Run
    will do" settings, when it's really about "what's currently visible/
    swept" - the same category as the ViewCube, which is why this docks
    directly below it. Anchored to the top-right corner via reposition()
    (called from PickableGLView.resizeEvent, since a plain child widget with
    no layout manager doesn't reposition itself when its parent resizes).

    Collapsible via a small tab (collapsed: just the tab; expanded: tab +
    controls) - independent of roi_group's own checkbox, which toggles
    whether the crop is ACTUALLY applied. That distinction matters: leaving
    the panel collapsed shouldn't hide the fact that a crop is currently
    active, so the tab's TOOLTIP says so (see set_active_indicator) even
    though the tab itself stays plain.

    No "ROI" text label on the tab - just a small arrow that flips direction
    on expand/collapse, since the expanded content's own title ("Crop")
    already says what it is; the tab only needs to say "there's more here."

    Two earlier versions of the tab both failed live testing: a colored
    "chip" background read as not matching the rest of the app's buttons;
    switching to a fully transparent native QToolButton (system arrow icon,
    no backdrop) then turned out invisible against the app's own actual
    default 3D-view background (a dark gray) - it only worked by accident
    against a light one, since a NATIVE arrow's color already follows
    whatever the OS theme happens to be, and this button floats over
    content that has nothing to do with that theme. The fix that actually
    holds regardless of scene OR OS theme: give the tab the SAME fixed,
    explicit-colored backdrop the expanded panel already needs (color
    parity, reported as the right direction) - permanently, not toggled off
    while collapsed - and draw the arrow itself as an explicit-colored text
    glyph rather than a native icon, so its contrast against that backdrop
    doesn't depend on the OS theme either. A small, fixed, always-visible
    chip, same as this whole panel already is once expanded - just smaller."""

    _MARGIN_PX = 10  # matches ViewCubeOverlay's own margin_px - see view_cube.py
    _ARROW_COLLAPSED = "▸"
    _ARROW_EXPANDED = "▾"

    def __init__(self, roi_group: QGroupBox, parent_view) -> None:
        super().__init__(parent_view)
        self._parent_view = parent_view
        self.setObjectName("roiOverlayPanel")
        self.setStyleSheet(
            "#roiOverlayPanel { background-color: rgba(48, 48, 52, 250); "
            "border: 1px solid #8a8a8a; border-radius: 6px; }"
            "#roiOverlayPanel QGroupBox { border: none; margin-top: 4px; color: #f0f0f0; }"
            "#roiOverlayPanel QGroupBox::title { color: #f0f0f0; }"
            "#roiOverlayPanel QLabel { color: #f0f0f0; }"
            # Unchecking the box auto-disables its contents (Qt's own
            # checkable-QGroupBox behavior) - reported illegible: native
            # disabled-state rendering dims text toward the OS's own default
            # disabled color, which against THIS panel's fixed dark
            # background (not the OS theme's own background) can land at
            # very low contrast. Pick an explicit disabled color known to
            # read against rgba(48,48,52) specifically, same reasoning as
            # the enabled colors above.
            "#roiOverlayPanel QGroupBox::title:disabled { color: #999999; }"
            "#roiOverlayPanel QLabel:disabled { color: #999999; }"
            "#roiOverlayPanel QDoubleSpinBox:disabled { color: #999999; "
            "background-color: #3a3a3e; border: 1px solid #666666; }"
            "#roiOverlayPanel QPushButton:disabled { color: #888888; }"
            # The tab: same backdrop color as the panel above (color
            # parity), explicit text color for the arrow glyph itself
            # (setArrowType's native icon would follow the OS theme, not
            # this fixed dark backdrop - the exact contrast bug this whole
            # redesign is fixing).
            "#roiOverlayPanel QToolButton#roiTab { background-color: rgba(60, 60, 64, 255); "
            "color: #f0f0f0; border: 1px solid #8a8a8a; border-radius: 4px; font-weight: bold; }"
            "#roiOverlayPanel QToolButton#roiTab:hover { background-color: rgba(80, 80, 86, 255); }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(2)

        self._tab = QToolButton()
        self._tab.setObjectName("roiTab")
        self._tab.setText(self._ARROW_COLLAPSED)
        self._tab.setCheckable(True)
        self._tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tab.setFixedSize(18, 18)  # reported the default toolbutton size was too big for a corner tab
        self._base_tooltip = "Crop (θ/φ region of interest) - click to expand/collapse."
        self._tab.setToolTip(self._base_tooltip)
        self._tab.toggled.connect(self._on_tab_toggled)
        outer.addWidget(self._tab)

        roi_group.setFlat(True)
        outer.addWidget(roi_group)
        self._content = roi_group
        self._content.setVisible(False)  # starts collapsed - a corner control, not a permanent fixture

        self.adjustSize()

    def _on_tab_toggled(self, expanded: bool) -> None:
        self._tab.setText(self._ARROW_EXPANDED if expanded else self._ARROW_COLLAPSED)
        self._content.setVisible(expanded)
        self.adjustSize()
        self.reposition()

    def set_active_indicator(self, active: bool) -> None:
        """The tab's own look stays the same either way - only its tooltip
        notes that a crop is currently active, so collapsing the panel
        doesn't silently hide that fact."""
        suffix = " (currently active)" if active else ""
        self._tab.setToolTip(self._base_tooltip + suffix)

    def reposition(self) -> None:
        """Anchors to the top-right corner of the 3D view, directly below
        where ViewCubeOverlay draws itself (same margin, same corner - see
        view_cube.py's corner_rect/paint for the cube's own placement)."""
        self.adjustSize()
        cube = self._parent_view.view_cube
        cube_bottom = self._MARGIN_PX + cube.size_px
        x = self._parent_view.width() - self._MARGIN_PX - self.width()
        y = int(cube_bottom + self._MARGIN_PX)
        self.move(max(0, x), y)
        self.raise_()


def _channel_arg(combo: QComboBox, node: ScanNode) -> str:
    """
    The services take a concrete channel (rgb/intensity/normals), not 'auto' -
    the CLI resolves 'auto' itself via _resolve_channel() before calling them
    (see cli.py), so the GUI must do the same rather than passing 'auto' through.

    Reads currentData() (the UserRole value _channel_combo() attaches to each
    item), NOT currentText() - the displayed text for "auto" gets rewritten to
    show what it resolves to (e.g. "auto (rgb)"), so comparing display text
    against the literal string "auto" would silently break once that happens.
    """
    choice = combo.currentData(Qt.ItemDataRole.UserRole)
    if choice == "auto":
        from core.projection import default_channel
        return default_channel(node)
    return choice


class _OutlinedGLTextItem(gl_module.GLTextItem):
    """
    GLTextItem draws its text exactly once via a bare QPainter.drawText() -
    no outline/shadow at all, so a label can blend into similarly-toned
    points or the background color behind it (reported live). Overridden to
    draw a few pixel-offset copies in a fixed outline color first, then the
    real text on top, in the SAME paint() call - real SCREEN-SPACE pixel
    offsets applied to the already-projected 2D text position, not a
    3D-world offset (which would scale confusingly with camera distance
    instead of staying a crisp, constant-width outline).

    Fixed white-on-black rather than computing contrast against the app's
    background color: GLTextItem has no way to see what's actually rendered
    BEHIND the text (it doesn't read the framebuffer) - only the background
    color setting, which wouldn't account for the point cloud itself. An
    outline gives contrast against literally anything on either side (dark
    OR light OR colorful points) by construction - the standard technique
    games/subtitles use for exactly this "readable over any content" need.
    """

    _OUTLINE_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def paint(self) -> None:
        if len(self.text) < 1:
            return
        self.setupGLState()

        project = self.compute_projection()
        vec3 = pg_QtGui.QVector3D(*self.pos)
        text_pos = self.align_text(project.map(vec3).toPointF())

        painter = pg_QtGui.QPainter(self.view())
        painter.setFont(self.font)
        painter.setRenderHints(
            pg_QtGui.QPainter.RenderHint.Antialiasing | pg_QtGui.QPainter.RenderHint.TextAntialiasing
        )

        painter.setPen(pg_QtCore.Qt.GlobalColor.black)
        for dx, dy in self._OUTLINE_OFFSETS:
            painter.drawText(text_pos + pg_QtCore.QPointF(dx, dy), self.text)

        painter.setPen(pg_QtCore.Qt.GlobalColor.white)
        painter.drawText(text_pos, self.text)
        painter.end()


class Viewer3DWidget(QFrame):
    """
    Loads one point cloud at a time and shows it in an orbit-controlled 3D
    view. All segmentation (prompt-driven sweep, single-point pick) runs
    directly against the already-loaded ScanNode via services -
    in-process, no subprocess, no re-loading the file - and is drawn back
    onto the same point cloud instead of requiring an export/reopen roundtrip.

    The right-hand panel is ALWAYS the Run (DINO+SAM2 prompt-driven sweep)
    settings, matching `pcseg run` - no Mode selector any more (design review
    2026-09-10: this used to switch between View/Run/Auto-Segment via a combo
    box; View-only was never really a "mode" - it had no settings of its own
    - and Auto-Segment was dropped from the GUI entirely, per the same
    review, so with only one real mode left there was nothing left to
    select). Orbit/pan/zoom/select are never mode-gated - they always work
    regardless of anything in this panel (see PickableGLView). Run starts
    from its own Run button and runs on a background thread (see workers.py)
    so the UI - and the progress bar/status label here - stay responsive
    during a sweep.

    Single-point picking (pcseg pick-segment) is NOT part of this panel at
    all (same 2026-09-10 review that removed the Mode selector - it was
    pulling right-panel space for a workflow that's really about ONE
    specific layer, not a whole separate segmentation approach) - it lives
    entirely in the Layers panel instead: right-click an existing layer (or
    "+ New empty layer..." for a brand new one) -> "Pick into this
    layer..." arms it immediately, no settings dialog (uses the 3D view's
    OWN currently-displayed color channel and a fixed 90 deg FOV - see
    _start_pick_into_layer). Right-clicking the 3D view then segments the
    point under the cursor and shows its (always 3, SAM2-ranked) candidate
    masks in a small menu at the click position - hovering one previews it
    live with a faint highlight, choosing one queues it as a PENDING pick
    (see `_PendingPick`), NOT added to the layer yet, so several objects can
    be picked as one batch. Shift+right-click runs the SAME SAM2 segmentation
    but SUBTRACTS the chosen candidate's points from every pending pick it
    overlaps instead of adding a new one (see _subtract_pick_from_pending) -
    an eraser for over-segmentation: if an earlier pick swallowed more than
    the one object (e.g. a wall along with its window), shift-click the
    excess region and let SAM2 trace ITS boundary rather than trimming it by
    hand. Ctrl+right-click instead ACCUMULATES points for one shared SAM2
    multi-point prompt (see _add_multipoint_click) - SAM2 supports several
    points at once and uses them to disambiguate an object that a single
    click can't (e.g. two touching, similarly-colored panels): the first
    Ctrl+right-click anchors the same fixed-FOV view single-point picking
    already uses and draws its boundary as a cyan outline in the 3D view
    (_update_multipoint_indicator, reusing the ROI overlay's own boundary-
    tracing), since every point of one SAM2 prompt has to land on that SAME
    rendered image; further Ctrl+right-clicks must stay inside that outline
    (rejected otherwise, with a status message) and just add another point,
    while a plain right-click adds ITS point too and then actually runs SAM2
    with everything accumulated, showing the (now single, no longer
    ambiguous) result the same way a single-point pick does. Escape cancels
    just an in-progress multi-point buffer if one's active, without
    discarding pending picks already accepted earlier. The Layers panel's
    "Picking..." section (shown
    only while armed) lists the pending picks, each individually droppable
    via its own [x], plus three buttons: **Finalize** (runs the same noise
    filter Run always applies on each pending pick individually - never
    merged together, since each pick is its own deliberately-chosen object -
    to preview the cleaned-up result; greyed out with nothing pending),
    **Accept** (finalizes if not already, then commits everything for real,
    marked `manual=True` on the resulting LayerInstances so a hand-picked
    object reads as such even inside an otherwise-automatic layer, and ends
    the session), and **Cancel** (discards every pending pick, no commit).
    Escape is the same as Cancel; right-click does nothing while unarmed,
    same "explicit or it does nothing" convention as before.
    """

    # "Export at Full Resolution..." (Layers panel) - MainWindow owns the
    # actual reload-at-full-detail + label-transfer + export sequence (see
    # MainWindow.export_at_full_resolution) since it needs open_file's
    # background-loading machinery, which lives there, not here - same
    # division of labor as LayerPanel.pick_into_requested -> Viewer3DWidget.
    export_full_resolution_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        # Per-viewer preference (not session/layer data) - QSettings persists it
        # across app restarts for free, same mechanism Qt apps normally use for
        # this kind of thing.
        self._settings = QSettings("pcseg", "pcseg-view")
        # The hover/selection outline post-process - a real GPU render effect
        # (see ShaderPointCloudItem's outline_* methods), not free, and not
        # to everyone's taste visually either - made toggleable (View menu,
        # see MainWindow._build_menu) after being reported as wanted-
        # optional. ShaderPointCloudItem is recreated on every scan LOAD (a
        # new node = a new GL item), so this widget's own copy of the
        # setting is what survives a reload and gets reapplied to the new
        # point_item each time (see load_node) - point_item itself has no
        # memory across that recreation.
        self.outline_effect_enabled = self._settings.value("outline_effect_enabled", True, type=bool)
        # Debug images for Runs and picks (Tools menu - see MainWindow._build_menu,
        # set_debug_sweep_images/set_debug_output_folder below). `debug_dir`
        # is the BASE folder every run's own timestamped subfolder lands in -
        # remembered across restarts so re-enabling this later doesn't
        # require re-picking a folder. `debug_sweep_images` starts False even
        # if a folder is already remembered - turning this on is a deliberate
        # per-session choice, not something that should silently resume.
        self.debug_sweep_images: bool = False
        self.debug_dir: str | None = self._settings.value("debug_dir", None, type=str) or None

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setLayout(outer)

        self._build_view_toolbar(outer)

        # THREE regions side by side (design review 2026-09: superseded the
        # two-column-panel-beside-the-3D-view layout - the user actually
        # wanted Layers on the window's far-left EDGE and the 3D view
        # sandwiched in the CENTER, not Layers+Settings sharing one
        # container off to one side of the 3D view). One top-level
        # horizontal QSplitter, three independent siblings: Layers,
        # gl_view, Mode+settings+ROI - not two nested splitters, so the 3D
        # view is a true peer of each panel, not a big pane living inside
        # what's still conceptually "the same side" as one of them.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        layers_panel = self._build_layers_panel()
        layers_panel.setMinimumWidth(240)
        layers_panel.setMaximumWidth(420)
        splitter.addWidget(layers_panel)

        self.gl_view = PickableGLView()
        saved_bg = self._settings.value("background_color", DEFAULT_BACKGROUND_COLOR, type=str)
        self.gl_view.setBackgroundColor(saved_bg)
        self.gl_view.setCameraPosition(distance=35, elevation=30, azimuth=45)
        self.gl_view.point_clicked.connect(self._on_point_clicked)
        self.gl_view.cube_face_clicked.connect(self._on_cube_face_clicked)
        self.gl_view.instance_click.connect(self._on_instance_click)
        self.gl_view.box_select.connect(self._on_instance_box_select)
        self.gl_view.escape_pressed.connect(self._on_escape)
        self.gl_view.delete_requested.connect(self._on_delete_selected_instances)
        self.gl_view.hover_moved.connect(self._on_hover_moved)
        self.gl_view.hover_left.connect(self._clear_hover_preview)
        self.gl_view.shift_toggled.connect(self._on_shift_toggled)
        self.gl_view.recenter_requested.connect(self.center_camera)
        self.gl_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build_roi_overlay()  # floating panel, must come after gl_view exists to parent onto it
        splitter.addWidget(self.gl_view)

        run_settings_panel = self._build_run_settings_panel()
        run_settings_panel.setMinimumWidth(280)
        run_settings_panel.setMaximumWidth(420)
        splitter.addWidget(run_settings_panel)

        splitter.setStretchFactor(0, 0)  # Layers - fixed-ish width
        splitter.setStretchFactor(1, 1)  # 3D view - absorbs all extra space
        splitter.setStretchFactor(2, 0)  # Mode/settings/ROI - fixed-ish width
        splitter.setSizes([300, 700, 340])

        # A FIXED-height container, not a plain addLayout() - the progress
        # bar/Cancel button toggle setVisible() on every sweep start/stop,
        # and an ordinary layout row collapses to ~0 height while its
        # children are hidden, so the splitter above it (stretch=1, i.e.
        # "take whatever's left") grew/shrank by that same amount every time
        # - reported live as the 3D view visibly jumping/resizing whenever a
        # sweep started. Reserving a constant height here regardless of
        # what's inside keeps the splitter's available space, and so the 3D
        # view itself, stable across the whole session.
        progress_container = QWidget()
        progress_container.setFixedHeight(28)
        progress_row = QHBoxLayout(progress_container)
        progress_row.setContentsMargins(4, 2, 4, 2)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_row.addWidget(self.progress_bar, stretch=1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setToolTip(
            "Stops the current Run sweep. Takes effect at the "
            "next view boundary, not instantly (DINO/SAM2 inference itself can't be "
            "interrupted mid-call) - no layer is added for a cancelled run."
        )
        self.btn_cancel.clicked.connect(self._cancel_current_task)
        progress_row.addWidget(self.btn_cancel)
        outer.addWidget(progress_container)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #cccccc; padding: 2px 5px;")
        outer.addWidget(self.status_label)

        self.axis_item = gl_module.GLAxisItem()
        self.axis_item.setSize(1, 1, 1)
        self.gl_view.addItem(self.axis_item)

        self.point_item: ShaderPointCloudItem | None = None
        self.node: ScanNode | None = None
        self.point_size = 2
        self._sam_service = None
        self._dino_service = None
        self.dino_size = "tiny"  # changed via models_dialog.py -> set_dino_size()
        self.sam_size = "tiny"   # changed via models_dialog.py -> set_sam_size()
        self._task_thread = None
        self._task_worker = None
        self._busy = False
        self._cancel_event: threading.Event | None = None
        self._pick_target_layer = None  # Layer currently armed for pick-into - see _start_pick_into_layer
        self._pending_picks: list[_PendingPick] = []  # accepted-from-menu, not committed yet - see _add_pending_pick
        # Multi-point pick-in-progress (Ctrl+right-click accumulates points for
        # ONE shared SAM2 prompt - see _add_multipoint_click). None/empty/None
        # when no accumulation is in progress (the common single-point case).
        self._multipoint_anchor_index: int | None = None
        self._multipoint_extra_indices: list[int] = []
        self._multipoint_window: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._multipoint_indicator_item = None
        self._hovered_instance = None  # LayerInstance currently shown as a hover preview - see _on_hover_moved
        self._hover_throttle_last = 0.0  # time.monotonic() of the last hover GPU pick - see _on_hover_moved
        self._log_forwarder = LogForwarder()
        self._log_forwarder.message.connect(self._on_log_message)
        self._log_handler = None
        self.roi_indicator_item = None
        self.overlay_box_item = None
        self.overlay_text_items: list = []
        self._selected_instance_ids: set[int] = set()  # id(instance) of each currently-selected instance
        # None = normal per-layer visibility/points_hidden rules apply.
        # Otherwise, the ONLY point indices allowed to render - see
        # _focus_on_selected/_clear_focus and _recompute_composition's own
        # hidden_mask computation, which checks this FIRST and ignores
        # every layer's own points_hidden while it's set (an exclusive
        # "isolate" override, not something that composes with per-layer
        # hiding - simpler and less surprising than trying to combine both).
        self._focus_indices: np.ndarray | None = None
        self._last_deletion: dict | None = None  # one-level undo - see _on_delete_selected_instances
        self._overlay_label_size = 16  # set via View menu > Label Size... (pick_label_size)

    # ── Top toolbar: view controls only ─────────────────────────────────────

    def _build_view_toolbar(self, outer: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(5, 5, 5, 5)

        btn_minus = QPushButton("- Size")
        btn_minus.setToolTip("Shrink the rendered point size.")
        btn_minus.clicked.connect(lambda: self._change_point_size(-1))
        btn_plus = QPushButton("+ Size")
        btn_plus.setToolTip("Grow the rendered point size.")
        btn_plus.clicked.connect(lambda: self._change_point_size(1))
        row.addWidget(btn_minus)
        row.addWidget(btn_plus)
        row.addSpacing(15)

        row.addWidget(QLabel("Color:"))
        self.combo_color = QComboBox()
        self.combo_color.setToolTip(
            "RGB / intensity / normals for a raw .e57 scan (whichever it actually has); "
            "a .las result shows true RGB if present, or its per-point classification "
            "field otherwise - one color per segmented object/class."
        )
        self.combo_color.currentTextChanged.connect(self._on_color_mode_changed)
        row.addWidget(self.combo_color)
        row.addSpacing(15)

        row.addWidget(QLabel("Render mode:"))
        self.combo_render = QComboBox()
        self.combo_render.addItems(RENDER_MODES)
        self.combo_render.setToolTip(
            "Solid / Transparent / Clipped - fades or hides points whose normal faces "
            "away from the camera, so you can see \"through\" a surface toward what's "
            "behind it. Only has an effect where real normals exist (a raw .e57 scan); "
            "LAS results have none, so this stays a no-op there."
        )
        self.combo_render.currentIndexChanged.connect(self._on_render_mode_changed)
        row.addWidget(self.combo_render)
        row.addSpacing(15)

        row.addWidget(QLabel("View detail:"))
        self.view_decimation_spin = _int_spin(1, 32, 1)
        self.view_decimation_spin.setToolTip(
            "Rendering-only subsample of the ALREADY-LOADED points, for interactive "
            "performance (1 = show every loaded point). Density-adaptive, not a uniform "
            "stride: thins out areas with a lot of local point overlap (e.g. heavily "
            "overlapping sweep views) much more than already-sparse areas, which stay at "
            "close to full detail instead of losing the same fraction everywhere. NOT the "
            "same as Load detail: segmentation always runs against the full loaded point "
            "set regardless of this value - it only changes how much you're currently "
            "choosing to look at, live, without reloading the file. Changing this can take "
            "several seconds to recompute on a very large scan (tens of millions of "
            "points) - the view will catch up shortly."
        )
        self.view_decimation_spin.valueChanged.connect(self._on_view_decimation_changed)
        row.addWidget(self.view_decimation_spin)

        # Moved out of the Layers box (design review 2026-09) - the Layers
        # box is layer DATA (the list + Instance details), this is a
        # rendering/display toggle, same category as View detail next to
        # it, which the Layers box was never really the right home for. Renamed from "Show instance boxes +
        # labels (selected layer)" to fit this row's terser naming
        # convention - full explanation moved into the tooltip instead.
        self.show_instance_overlay_checkbox = QCheckBox("Instance labels")
        self.show_instance_overlay_checkbox.setToolTip(
            "Draws a wireframe box and a label (name + score) over each instance "
            "of the layer currently selected in the Layers panel - only that one "
            "layer's instances, not the whole stack, to keep it from getting "
            "cluttered/slow with many objects. Font size: View menu > Label Size..."
        )
        self.show_instance_overlay_checkbox.toggled.connect(self._update_instance_overlay)
        row.addWidget(self.show_instance_overlay_checkbox)

        row.addStretch()

        btn_center = QPushButton("Center View")
        btn_center.setToolTip("Re-centers the camera on the loaded scan's origin.")
        btn_center.clicked.connect(self.center_camera)
        row.addWidget(btn_center)

        # Background color moved to MainWindow's View menu (see
        # main_window.py._build_menu) to free up panel/toolbar space - purely
        # a relocation, same pick_background_color()/reset_background_color()
        # underneath.

        outer.addLayout(row)

    # ── Region of interest: floating overlay on the 3D view ────────────────
    # Moved out of the right-hand Run-settings panel (design review
    # 2026-09-11: reported as feeling out of place there - ROI is about
    # "what's currently visible/swept," not "what the next Run will do,"
    # even though Run does respect it). Now a collapsible panel drawn
    # directly on top of the 3D view, anchored below the ViewCube - another
    # "about the current view" control already living in that corner.

    def _build_roi_controls(self) -> QGroupBox:
        """Builds self.roi_group and its theta/phi controls - unchanged
        from before the move, just relocated out of _build_run_settings_panel
        into its own method so _build_roi_overlay can wrap it."""
        # "Crop" not "Region of interest" (design review 2026-09-11 -
        # reported the longer title reads better as this short a label,
        # and the tooltip below still spells out what it actually does).
        self.roi_group = QGroupBox("Crop")
        self.roi_group.setCheckable(True)
        self.roi_group.setChecked(False)
        self.roi_group.setToolTip(
            "Region of interest (θ/φ crop). Restricts BOTH what's drawn in the 3D view AND "
            "what actually gets swept by Run, to this angular wedge - e.g. crop to just one "
            "building instead of sweeping the full 360 degrees so unrelated classes never mix "
            "in. A yellow outline traced through real scanned points shows the current wedge. "
            "Bounds where each swept view is CENTERED, not its full field of view - a view "
            "whose center sits right at the wedge edge still sees (and can classify) a bit "
            "beyond it, same as the sweep's own overlap between neighboring views; shrink "
            "the wedge a little past what you actually need if a hard boundary matters. "
            "Single-point picking is unaffected (a single named point, no sweep to "
            "restrict)."
        )
        self.roi_group.toggled.connect(self._on_roi_changed)
        roi_layout = QVBoxLayout(self.roi_group)
        # Extra top margin - the checkable title's own reserved height gets
        # miscalculated once _ROIOverlayPanel's QSS sets `border: none` on
        # this box (removing the frame also drops the native style's usual
        # auto-reservation for the title row), reported live as the title
        # text visually overlapping the theta row right below it. Padding
        # the content layout directly sidesteps relying on that native
        # calculation at all.
        roi_layout.setContentsMargins(6, 16, 6, 6)
        theta_tip = "Horizontal (azimuth) bound, degrees - 0 is straight ahead of the scan origin."
        phi_tip = (
            "Vertical (elevation) bound, degrees - negative looks up, positive looks down. "
            "Restricting theta/phi only bounds each swept view's CENTER, not its FOV cone - "
            "a view pitched near +-90 (straight up/down) can still see a wide spread of "
            "theta at its extreme rows (an equirectangular-projection effect, not a bug) - "
            "keep phi away from the poles for a tight, predictable wedge."
        )

        # Compact single-row-per-axis layout: [min field] [slider] [max field] -
        # the slider is the primary drag interaction, the flanking fields are
        # for typing an exact boundary (synced both ways). Replaced an earlier
        # stacked layout (separate spinbox rows above a full-width slider row)
        # that ate too much panel height for what it showed.
        self.roi_theta_lo = _double_spin(-180.0, 180.0, 5.0, -45.0)
        self.roi_theta_hi = _double_spin(-180.0, 180.0, 5.0, 45.0)
        self.roi_theta_slider = RangeSlider(-180.0, 180.0, -45.0, 45.0)
        theta_row = QHBoxLayout()
        theta_row.addWidget(QLabel("θ:"))
        for w in (self.roi_theta_lo, self.roi_theta_slider, self.roi_theta_hi):
            w.setToolTip(theta_tip)
        # Plain editable numeric fields, no up/down stepper arrows - the
        # slider is the primary way to adjust these, typing is for an exact
        # value, not incremental nudging.
        self.roi_theta_lo.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.roi_theta_hi.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.roi_theta_lo.setMaximumWidth(64)
        self.roi_theta_hi.setMaximumWidth(64)
        theta_row.addWidget(self.roi_theta_lo)
        theta_row.addWidget(self.roi_theta_slider, stretch=1)
        theta_row.addWidget(self.roi_theta_hi)
        roi_layout.addLayout(theta_row)

        self.roi_phi_lo = _double_spin(-90.0, 90.0, 5.0, -45.0)
        self.roi_phi_hi = _double_spin(-90.0, 90.0, 5.0, 45.0)
        self.roi_phi_slider = RangeSlider(-90.0, 90.0, -45.0, 45.0)
        phi_row = QHBoxLayout()
        phi_row.addWidget(QLabel("φ:"))
        for w in (self.roi_phi_lo, self.roi_phi_slider, self.roi_phi_hi):
            w.setToolTip(phi_tip)
        self.roi_phi_lo.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.roi_phi_hi.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.roi_phi_lo.setMaximumWidth(64)
        self.roi_phi_hi.setMaximumWidth(64)
        phi_row.addWidget(self.roi_phi_lo)
        phi_row.addWidget(self.roi_phi_slider, stretch=1)
        phi_row.addWidget(self.roi_phi_hi)
        roi_layout.addLayout(phi_row)

        btn_roi_reset = QPushButton("Reset to Defaults")
        btn_roi_reset.setToolTip("Unchecks Region of interest and restores theta/phi to -45..45.")
        btn_roi_reset.clicked.connect(self.reset_roi_defaults)
        roi_layout.addWidget(btn_roi_reset)

        for spin in (self.roi_theta_lo, self.roi_theta_hi):
            spin.valueChanged.connect(self._on_roi_theta_spin_changed)
        for spin in (self.roi_phi_lo, self.roi_phi_hi):
            spin.valueChanged.connect(self._on_roi_phi_spin_changed)
        self.roi_theta_slider.rangeChanged.connect(self._on_roi_theta_slider_changed)
        self.roi_phi_slider.rangeChanged.connect(self._on_roi_phi_slider_changed)
        return self.roi_group

    def _build_roi_overlay(self) -> None:
        """Wraps _build_roi_controls() in a floating, collapsible panel
        parented directly onto self.gl_view (a real child QWidget - Qt
        composites plain child widgets over a QOpenGLWidget's GL content
        correctly) and anchors it just below the ViewCube. Must run AFTER
        self.gl_view exists (see __init__ - called right after gl_view's
        own setup, before the splitter picks up the right-hand panel)."""
        self.roi_overlay = _ROIOverlayPanel(self._build_roi_controls(), self.gl_view)
        self.roi_group.toggled.connect(self.roi_overlay.set_active_indicator)
        self.gl_view.roi_overlay = self.roi_overlay  # see PickableGLView.resizeEvent
        self.roi_overlay.reposition()

    def _build_run_settings_panel(self) -> QWidget:
        """The Run (DINO+SAM2) prompt settings - what the NEXT run will do.
        Its own top-level splitter pane, a peer of the 3D view and of
        _build_layers_panel(), not sharing a container with either (design
        review 2026-09: superseded the earlier two-columns-in-one-panel
        layout - see this method's caller). ROI used to live here too, but
        moved to a floating overlay on the 3D view itself (design review
        2026-09-11, see _build_roi_overlay) - it's a "what's currently
        visible/swept" control, not really "what to run next."

        No Mode selector any more (design review 2026-09-10: this panel WAS
        View/Run/Auto-Segment, switched via a combo box - now it's just Run,
        permanently, since that's the viewer's one standard, dedicated
        workflow; View-only was never really a "mode" (no settings of its
        own), and Auto-Segment/Pick Point moved out entirely - Pick Point to
        the Layers panel's right-click menu, Auto-Segment dropped from the
        GUI, both per the same review). Orbit/pan/zoom/select always work
        regardless of anything here - they were never mode-gated to begin
        with (see PickableGLView)."""
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # ROI itself now lives in a floating overlay on the 3D view, not
        # here - see _build_roi_controls/_build_roi_overlay. This panel is
        # just the Run prompt settings now.
        setup_container = QWidget()
        setup_layout = QVBoxLayout(setup_container)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.addWidget(self._build_manual_pick_section())
        setup_layout.addWidget(self._build_run_prompt_page())

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(setup_container)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_layout.addWidget(settings_scroll, stretch=1)

        return right_col

    def _build_layers_panel(self) -> QWidget:
        """Layers (what past runs produced) - its own top-level splitter
        pane, on the window's far-left edge, a peer of the 3D view and of
        _build_run_settings_panel() (design review 2026-09 - see
        _build_run_settings_panel's docstring for why these used to share
        one container and no longer do)."""
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.layer_panel = LayerPanel()
        self.layer_panel.changed.connect(self._recompute_composition)
        self.layer_panel.changed.connect(self._update_instance_overlay)
        self.layer_panel.selectionChanged.connect(self._update_instance_overlay)
        self.layer_panel.selectionChanged.connect(self._clear_instance_selection)
        self.layer_panel.pick_into_requested.connect(self._start_pick_into_layer)
        self.layer_panel.instance_clicked.connect(self._on_layer_panel_instance_clicked)
        layers_box = QGroupBox("Layers")
        layers_box_layout = QVBoxLayout(layers_box)

        # The mouse/keyboard help used to be a permanent multi-line caption
        # here - now behind the same hover/click info icon every other
        # section uses, so it no longer takes up the panel's space.
        controls_header = QHBoxLayout()
        controls_header.addWidget(_info_icon(
            "Left button: click a detection to select it, drag a box to "
            "select several (Shift+click/drag removes instead of adding). "
            "Middle button: drag to orbit, Shift+drag to pan, Ctrl+drag to "
            "pan vertically. Delete removes the selected instance(s), "
            "Escape clears the selection."
        ))
        controls_header.addWidget(_section_hint_label("Mouse and keyboard controls"))
        controls_header.addStretch(1)
        layers_box_layout.addLayout(controls_header)

        # Hidden until a pick session is armed (see _start_pick_into_layer) -
        # lets picking several objects into one layer happen as a batch
        # (pending, NOT added for real yet - see _PendingPick) instead of
        # each candidate-menu choice committing immediately, so a bad pick
        # can be dropped before it ever touches the real layer.
        self.pending_picks_box = QGroupBox("Picking...")
        self.pending_picks_box.setVisible(False)
        pending_layout = QVBoxLayout(self.pending_picks_box)

        self.pending_picks_list = QListWidget()
        self.pending_picks_list.setToolTip(
            "Pending picks for this session - not added to the layer yet. Each row's "
            "✕ drops just that one."
        )
        pending_layout.addWidget(self.pending_picks_list)

        pending_btn_row = QHBoxLayout()
        self.btn_pick_finalize = QPushButton("Finalize")
        self.btn_pick_finalize.setToolTip(
            "Cleans each pending pick on its own (voxel-connectivity + Local Outlier "
            "Factor, the same noise filter Run always applies) - never merged together, "
            "each pick stays its own element. Lets you preview the cleaned-up result "
            "before committing. Optional: Accept runs this itself first regardless, so "
            "skipping this button is fine."
        )
        self.btn_pick_finalize.clicked.connect(self._finalize_pending_picks)
        pending_btn_row.addWidget(self.btn_pick_finalize)

        self.btn_pick_accept = QPushButton("Accept")
        self.btn_pick_accept.setToolTip(
            "Finalizes (if not already) and adds every pending pick into the target layer "
            "for real, then ends this pick session."
        )
        self.btn_pick_accept.clicked.connect(self._accept_pending_picks)
        pending_btn_row.addWidget(self.btn_pick_accept)

        btn_pick_cancel = QPushButton("Cancel")
        btn_pick_cancel.setToolTip("Discards every pending pick and ends this pick session - nothing is added.")
        btn_pick_cancel.clicked.connect(self._cancel_pending_picks)
        pending_btn_row.addWidget(btn_pick_cancel)

        pending_layout.addLayout(pending_btn_row)
        layers_box_layout.addWidget(self.pending_picks_box)

        self.btn_undo_deletion = QPushButton("Undo Last Deletion")
        self.btn_undo_deletion.setEnabled(False)
        self.btn_undo_deletion.setToolTip(
            "Restores the instance(s) removed by the most recent deletion - "
            "one level only, overwritten by the next deletion. This app has "
            "no general Undo yet; this is specific to instance deletion."
        )
        self.btn_undo_deletion.clicked.connect(self._undo_last_deletion)
        layers_box_layout.addWidget(self.btn_undo_deletion)

        layers_scroll = QScrollArea()
        layers_scroll.setWidgetResizable(True)
        layers_scroll.setWidget(self.layer_panel)
        layers_scroll.setFrameShape(QFrame.Shape.NoFrame)
        layers_box_layout.addWidget(layers_scroll)

        # This panel's own content fills its own pane's height now that
        # Layers has a whole splitter pane to itself rather than sharing one
        # with Settings - no competition for vertical space to split
        # stretch=0/1 between them any more.
        layers_box.setMinimumHeight(120)
        left_layout.addWidget(layers_box, stretch=1)

        bottom_row = QHBoxLayout()
        btn_clear = QPushButton("Clear All Layers")
        btn_clear.setToolTip("Removes every layer and reverts the 3D view to the plain color mode.")
        btn_clear.clicked.connect(self.clear_segmentation)
        bottom_row.addWidget(btn_clear)
        btn_export = QPushButton("Export Segmentation...")
        btn_export.setToolTip(
            "Saves the FULL composed layer stack to a .las, with a per-point "
            "classification field - same as the CLI's own export, honoring every "
            "layer's order/visibility and every instance's included/excluded state."
        )
        btn_export.clicked.connect(self.export_segmentation)
        bottom_row.addWidget(btn_export)
        left_layout.addLayout(bottom_row)

        btn_export_full = QPushButton("Export at Full Resolution...")
        btn_export_full.setToolTip(
            "For a scan currently loaded at a reduced Load detail (a speed/memory "
            "tradeoff while segmenting - see Open scan's own dialog): reloads the SAME "
            "source file at full resolution, transfers the current layer stack onto the "
            "newly loaded points (a spatial/voxel transfer, not exact - same as File > "
            "Reload at Different Detail...), then exports THAT to .las - so segmenting "
            "can stay fast at a coarse Load detail while the exported result is always "
            "the full, undecimated point cloud. Only applies to a raw .e57 scan; already "
            "at full resolution just runs the normal export directly."
        )
        btn_export_full.clicked.connect(self.export_full_resolution_requested.emit)
        left_layout.addWidget(btn_export_full)

        # Single application-wide filter (idempotent - see wheel_guard.py for
        # why this needs to be app-level, not a findChildren() sweep over
        # this container: a per-widget filter attached only to the outer
        # QSpinBox/QComboBox instances misses wheel events Qt delivers to
        # their own INTERNAL child widgets instead, which is most of a
        # spinbox's actual clickable area). Installed here rather than in
        # _build_run_settings_panel purely because this method happens to
        # run second in __init__ now - idempotent, so it doesn't matter
        # which panel's builder actually calls it.
        install_wheel_guard()

        return left_col

    def _build_manual_pick_section(self) -> QWidget:
        """A dedicated, clearly-visible entry point for the SAME single-point
        SAM2 picking workflow the Layers panel's own right-click "Pick into
        this layer..." menu already offers (see _start_pick_into_layer) -
        that context-menu entry is easy to never discover. This is its own
        AI segmentation mode, distinct from Run Prompt below (plain SAM2 off
        a clicked point, no GroundingDINO text prompt involved) - its own
        bordered section says so up front, visually separated from the
        Prompt page right below it."""
        box = QGroupBox("Add Points by Clicking")
        layout = QVBoxLayout(box)

        hint_header = QHBoxLayout()
        hint_header.addWidget(_info_icon(
            "A separate AI segmentation mode from Run Prompt below (SAM2 only, no "
            "text prompt) - right-click an object in the 3D view and SAM2 segments "
            "it under the cursor. Adds into the currently selected layer (see "
            "Layers on the left); if none is selected, you'll be asked to name a "
            "new one first."
        ))
        hint_header.addWidget(_section_hint_label())
        hint_header.addStretch(1)
        layout.addLayout(hint_header)

        btn = QPushButton("Start Picking...")
        btn.setToolTip(
            "Arms single-point picking into the currently selected layer (or a new "
            "one you name, if none is selected) - same underlying workflow as a "
            "layer's own right-click \"Pick into this layer...\" menu entry."
        )
        btn.clicked.connect(self._start_manual_pick_section)
        layout.addWidget(btn)

        return box

    def _start_manual_pick_section(self) -> None:
        """Handler for Add Points by Clicking's own "Start Picking..."
        button - targets the currently selected layer if there is one,
        otherwise prompts for a new layer's name first (same as the Layers
        panel's own "+ New empty layer..." button), then arms picking into
        it exactly like the right-click "Pick into this layer..." menu entry
        would (see _start_pick_into_layer, which owns everything past this
        point)."""
        if self.node is None:
            self.status_label.setText("Load a scan first.")
            return
        layer = self.layer_panel.selected_layer
        if layer is None:
            name, ok = QInputDialog.getText(self, "New layer", "Layer name:")
            name = name.strip()
            if not ok or not name:
                return
            layer = self.layer_panel.add_empty_layer(name)
        self._start_pick_into_layer(layer)

    def _build_sweep_settings_dialog(self) -> QDialog:
        """FOV/Overlap - the base view grid every sweep uses (including
        Multi-scale coverage's own middle tier, see _build_view_grid) -
        moved off the main Run page into their own gear-accessible popup
        (see the "Sweep geometry" row above), same reasoning as
        _build_multiscale_settings_dialog below."""
        self.run_overlap = _double_spin(0.0, 0.95, 0.05, 0.2)
        self.run_overlap.setToolTip(
            "Fractional angular overlap between neighboring views in the automatically "
            "computed view grid (matches --overlap) - the practical equivalent of a "
            "manual step size: smaller overlap means fewer, more widely-spaced views. "
            "Still used exactly the same way when Multi-scale coverage is on, too - EVERY "
            "one of its three tiers (far/middle/near) builds its own grid with this same "
            "overlap fraction, just at that tier's own FOV. Turning Multi-scale coverage "
            "on never bypasses this setting."
        )
        self.run_fov = _double_spin(10.0, 170.0, 5.0, 90.0)
        self.run_fov.setToolTip(
            "Field of view per view, in degrees (matches --fov). With Multi-scale "
            "coverage OFF, this is simply the whole sweep's own FOV. With it ON, this "
            "value doesn't stop mattering - it becomes the middle tier's own FOV, and the "
            "base the far/near tiers scale relative to (their own FOV = this x Far/Near "
            "FOV scale, in Multi-scale coverage's own settings). So changing this still "
            "reshapes the ENTIRE multi-scale grid, not just a part of it."
        )
        return _build_settings_popup(
            self, "Sweep geometry settings",
            [
                ("Overlap:", _row_with_reset(self.run_overlap, 0.2)),
                ("FOV (deg):", _row_with_reset(self.run_fov, 90.0)),
            ],
        )

    def _build_multiscale_settings_dialog(self) -> QDialog:
        """Multi-scale coverage's own far/near percentile + FOV scale
        settings, in their own gear-accessible popup (see the checkbox row
        above) - kept off the main Run page since the defaults already work
        well day-to-day (design review 2026-09-14)."""
        self.run_far_percentile = _double_spin(0.0, 100.0, 5.0, 75.0)
        self.run_far_percentile.setToolTip(
            "Points beyond this percentile of the scan's own distance distribution count "
            "as \"far\" - e.g. 75 = the farthest quarter of the scan's real points."
        )
        self.run_far_fov_scale = _double_spin(0.1, 1.0, 0.05, 0.5)
        self.run_far_fov_scale.setToolTip(
            "The far pass's own FOV, as a fraction of Sweep geometry's own FOV (0.5 = half)."
        )
        self.run_near_percentile = _double_spin(0.0, 100.0, 5.0, 25.0)
        self.run_near_percentile.setToolTip(
            "Points within this percentile of the scan's own distance distribution count "
            "as \"near\" - e.g. 25 = the closest quarter of the scan's real points."
        )
        self.run_near_fov_scale = _double_spin(1.0, 2.0, 0.05, 1.3)
        self.run_near_fov_scale.setToolTip(
            "The near pass's own FOV, as a multiple of Sweep geometry's own FOV (1.3 = "
            "30% wider). Not pushed too high by default: a rectilinear crop distorts "
            "increasingly toward its own edges as FOV grows, so an overly large near FOV "
            "can lose more to that distortion than it gains in context."
        )
        return _build_settings_popup(
            self, "Multi-scale coverage settings",
            [
                ("Far percentile:", _row_with_reset(self.run_far_percentile, 75.0)),
                ("Far FOV scale:", _row_with_reset(self.run_far_fov_scale, 0.5)),
                ("Near percentile:", _row_with_reset(self.run_near_percentile, 25.0)),
                ("Near FOV scale:", _row_with_reset(self.run_near_fov_scale, 1.3)),
            ],
        )

    def _build_run_prompt_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. door, window")
        self.prompt_input.setToolTip(
            "Text prompt for GroundingDINO, e.g. 'door, window' (matches --prompt). "
            "Each comma-separated category is queried separately for full recall."
        )
        self.prompt_input.textChanged.connect(self._refresh_category_order_list)
        layout.addWidget(self.prompt_input)

        order_box = QGroupBox("Category draw order within this layer")
        order_box.setToolTip(
            "Only matters for a multi-category prompt (e.g. 'window, building') - each "
            "category becomes its own layer, and this decides which of those layers ends "
            "up highest in the stack, winning any point/pixel a lower one also claims. "
            "Automatic (default) ranks by each category's own total point count - "
            "whichever detected the most real estate wins. Manual order (below) lets you "
            "rank the parsed categories yourself instead."
        )
        order_layout = QVBoxLayout(order_box)
        order_header = QHBoxLayout()
        order_header.addWidget(_info_icon(order_box.toolTip()))
        order_header.addWidget(_section_hint_label())
        order_header.addStretch(1)
        order_layout.addLayout(order_header)
        self.category_manual_order = QCheckBox("Manual order (default: automatic, by size)")
        self.category_manual_order.setToolTip(
            "Off (default): within this layer, a smaller/more specific detection (a "
            "window) always wins overlap against a bigger one (its wall), regardless "
            "of prompt order - matches the CLI's own default. On: rank the parsed "
            "categories below yourself instead - top of the list wins overlaps."
        )
        self.category_manual_order.toggled.connect(self._on_category_manual_order_toggled)
        order_layout.addWidget(_row_with_reset(self.category_manual_order, False))

        cat_row = QHBoxLayout()
        self.category_list = QListWidget()
        self.category_list.setEnabled(False)
        self.category_list.setMaximumHeight(90)
        self.category_list.setToolTip("Top = applied last = wins overlaps against categories below it.")
        cat_row.addWidget(self.category_list, stretch=1)
        cat_btns = QVBoxLayout()
        btn_cat_up = QPushButton("▲")
        btn_cat_up.setFixedWidth(24)
        btn_cat_up.setToolTip("Move the selected category up (toward winning more overlaps).")
        btn_cat_up.clicked.connect(lambda: self._move_category(-1))
        cat_btns.addWidget(btn_cat_up)
        btn_cat_down = QPushButton("▼")
        btn_cat_down.setFixedWidth(24)
        btn_cat_down.setToolTip("Move the selected category down.")
        btn_cat_down.clicked.connect(lambda: self._move_category(1))
        cat_btns.addWidget(btn_cat_down)
        cat_row.addLayout(cat_btns)
        order_layout.addLayout(cat_row)

        # The view grid is always the automatically-computed, overlap-
        # guaranteed one now (matches `pcseg run --auto-coverage`) - the GUI
        # no longer exposes the manual step-deg/pitch-angle grid at all, and
        # the sweep's range is always the active Region of interest's
        # theta/phi (or the scan's own real-data extent when ROI is off) -
        # see run_prompt_segmentation. "Restrict vertical range" was removed
        # too: ROI's own phi range is now the one place that's set (and it
        # hard-clips the RESULT to match, not just the sweep - see the ROI
        # box's own tooltip), so a second, separate vertical-range control
        # here would just be redundant with it.
        #
        # FOV/Overlap themselves moved off the main page into their own gear-
        # accessible popup (design review 2026-09-14, same reasoning as
        # Multi-scale coverage's own settings just below) - config for what
        # a plain single view of the sweep looks like, sensible left at its
        # defaults day-to-day, so it doesn't need to permanently occupy page
        # space. Still the base FOV/overlap Multi-scale coverage's own
        # middle tier uses too (see _build_view_grid) - moving it doesn't
        # change what it's used for, just where it's tucked away.
        sweep_row = QHBoxLayout()
        sweep_row.addWidget(QLabel("Sweep geometry"))
        sweep_row.addStretch(1)
        btn_sweep_settings = QToolButton()
        btn_sweep_settings.setAutoRaise(True)
        btn_sweep_settings.setText("⚙")  # gear
        btn_sweep_settings.setToolTip(
            "Sweep geometry settings (FOV, overlap) - the base view grid every sweep "
            "uses, including Multi-scale coverage's own middle tier."
        )
        self._sweep_settings_dialog = self._build_sweep_settings_dialog()
        btn_sweep_settings.clicked.connect(lambda: self._sweep_settings_dialog.exec())
        sweep_row.addWidget(btn_sweep_settings)
        layout.addLayout(sweep_row)

        channels_box = QGroupBox("Channels")
        channels_box.setToolTip(
            "Which rendered view of the scan GroundingDINO and SAM2 each actually look "
            "at - RGB, intensity, or surface normals, chosen independently per model "
            "since one model can do better on a channel the other doesn't. 'auto' (the "
            "default for both) picks whichever channel this particular scan actually has "
            "available, the same resolution order the CLI itself uses - see the combo's "
            "own 'auto (...)' label for what that resolves to on the CURRENTLY loaded "
            "scan."
        )
        channels_form = _form(channels_box)
        channels_form.addRow(_info_icon(channels_box.toolTip()), _section_hint_label())
        self.run_dino_channel = _channel_combo()
        self.run_dino_channel.setToolTip(
            "Rendered channel GroundingDINO runs detection on (matches --dino-channel; "
            "auto = best available)."
        )
        channels_form.addRow("DINO channel:", _row_with_reset(self.run_dino_channel, 0))
        self.run_sam_channel = _channel_combo()
        self.run_sam_channel.setToolTip(
            "Rendered channel SAM2 runs segmentation on (matches --sam-channel; "
            "auto = best available)."
        )
        channels_form.addRow("SAM2 channel:", _row_with_reset(self.run_sam_channel, 0))

        thresh_box = QGroupBox("Thresholds")
        thresh_box.setToolTip(
            "Confidence/quality filters applied at each stage of the detection pipeline, "
            "in the order they actually run: DINO box/text thresholds decide what counts "
            "as a detection AT ALL (before SAM2 ever runs); Max box area fraction drops a "
            "degenerate box that claims almost the whole view; SAM2 score threshold "
            "filters by the resulting MASK's own boundary quality (a wrong-but-clean mask "
            "can still score high); Min score is the LAST filter, after merging, on the "
            "final detection's own semantic (DINO text-match) confidence. Each field's own "
            "tooltip below has the specifics."
        )
        thresh_form = _form(thresh_box)
        thresh_form.addRow(_info_icon(thresh_box.toolTip()), _section_hint_label())
        self.run_dino_box_thresh = _double_spin(0.0, 1.0, 0.05, 0.3)
        self.run_dino_box_thresh.setToolTip(
            "Minimum GroundingDINO box confidence to keep a detection (matches "
            "--dino-box-threshold). Lower = more (and more speculative) detections."
        )
        thresh_form.addRow("DINO box threshold:", _row_with_reset(self.run_dino_box_thresh, 0.3))
        self.run_dino_text_thresh = _double_spin(0.0, 1.0, 0.05, 0.25)
        self.run_dino_text_thresh.setToolTip(
            "Minimum per-token text-match confidence for a phrase to attach to a box "
            "(matches --dino-text-threshold)."
        )
        thresh_form.addRow("DINO text threshold:", _row_with_reset(self.run_dino_text_thresh, 0.25))
        self.run_max_box_area = _double_spin(0.1, 1.0, 0.05, 0.95)
        self.run_max_box_area.setToolTip(
            "Drops a DINO box covering this fraction of its own view's area or more - "
            "GroundingDINO occasionally claims the ENTIRE crop is one object, almost "
            "always a degenerate detection rather than something genuinely worth "
            "keeping. Lower this if that still gets through; raise it (toward 1.0) only "
            "if a real object that legitimately fills most of a view is being dropped."
        )
        thresh_form.addRow("Max box area fraction:", _row_with_reset(self.run_max_box_area, 0.95))
        self.run_sam_score_thresh = _double_spin(0.0, 1.0, 0.05, 0.0)
        self.run_sam_score_thresh.setToolTip(
            "SAM2's own mask-quality score, not detection correctness - a wrong-but-"
            "clean-edged mask can score as high as a correct one. 0 = no filtering."
        )
        thresh_form.addRow("SAM2 score threshold:", _row_with_reset(self.run_sam_score_thresh, 0.0))
        self.run_min_score = _double_spin(0.0, 1.0, 0.05, 0.0)
        self.run_min_score.setToolTip(
            "Drops a FINAL detection (after outlier-filtering and merging) whose score - "
            "GroundingDINO's own text-match confidence, averaged across fragments if "
            "'Merge overlapping detections' merged more than one into it - is below this "
            "(matches --min-score). The semantic-confidence counterpart to 'SAM2 score "
            "threshold' above: a low SAM2 score means a messy mask boundary; a low score "
            "here means DINO itself wasn't confident the detection matches the prompt at "
            "all. 0 = no filtering."
        )
        thresh_form.addRow("Min score:", _row_with_reset(self.run_min_score, 0.0))

        # Advanced: rarely-changed thresholds/channel overrides and the
        # optional manual category order - collapsed by default so the
        # panel's primary space goes to prompt + sweep geometry + Run.
        advanced_box, advanced_layout = _collapsible_section(
            "Advanced settings",
            tooltip=(
                "Rarely-changed knobs, collapsed by default: category draw order, which "
                "rendered channel DINO/SAM2 each use, and the detection/filtering "
                "thresholds. Every field in here still applies to every Run exactly the "
                "same whether this section happens to be expanded or collapsed right now "
                "- collapsing it only hides the controls, it never resets or disables "
                "anything (see the individual sections/fields below for what each one "
                "actually does)."
            ),
        )
        advanced_layout.addWidget(order_box)
        advanced_layout.addWidget(channels_box)
        advanced_layout.addWidget(thresh_box)
        layout.addWidget(advanced_box)

        # NOT under Advanced, on purpose (design review 2026-09-10): whether
        # to merge at all is a decision worth seeing every time, not a
        # rarely-touched knob to hide away - and the IoU field right below
        # only matters once merging is actually on, so it stays collapsed
        # until then instead of cluttering the panel with a number that does
        # nothing yet.
        self.run_merge_duplicates = QCheckBox("Merge overlapping detections")
        self.run_merge_duplicates.setChecked(True)  # GUI default ON (2026-09-10) - unlike the CLI's own
        # off-by-default (see run_sweep's docstring: deliberately raw/unmerged for a first, unfamiliar
        # look at what a sweep finds) - by the time this reached the GUI it had already been tuned and
        # trusted through several rounds of real-data verification, so defaulting to on here matches how
        # it's actually used day to day. `pcseg run` on the command line is unaffected either way.
        self.run_merge_duplicates.setToolTip(
            "ON by default in the GUI (the CLI's own --merge-duplicates defaults off - see "
            "--help): a sweep detects the same physical object once per overlapping view, so "
            "one real object can show up as several instances - this fuses same-label "
            "instances that are really overlapping fragments of one object back into a "
            "single one (3D point-index IoU, same mechanism Auto-Segment uses internally; "
            "matches --merge-duplicates/--merge-iou). Turn it OFF if you're debugging what "
            "the pipeline is finding raw - merging can hide exactly the kind of mistake "
            "worth seeing unmerged (two classes fighting over the same points, a mask "
            "leaking onto a neighbor)."
        )
        layout.addWidget(_row_with_reset(self.run_merge_duplicates, True))

        merge_box = QGroupBox()
        merge_box.setFlat(True)
        merge_form = _form(merge_box)
        self.run_merge_iou = _double_spin(0.0, 1.0, 0.05, 0.10)
        self.run_merge_iou.setToolTip(
            "3D index-overlap (IoU) threshold above which two views' detections of the "
            "same label are treated as one physical object (matches --merge-iou)."
        )
        merge_form.addRow("IoU score:", _row_with_reset(self.run_merge_iou, 0.10))
        layout.addWidget(merge_box)
        merge_box.setVisible(self.run_merge_duplicates.isChecked())
        self.run_merge_duplicates.toggled.connect(merge_box.setVisible)

        # Off by default, same reasoning as Refine boundary-cut detections
        # below: an extra opt-in pass, not a change to what every existing
        # sweep already does. A plain on/off toggle (design review
        # 2026-09-14) - unlike Refine below, its own settings form does NOT
        # expand inline under the checkbox any more (reported as not needed
        # day-to-day - the defaults just work) - a small gear button next to
        # it opens them in their own popup dialog instead (see
        # _build_multiscale_settings_dialog), completely independent of the
        # checkbox's own checked state, so tuning them never requires
        # toggling the feature off and back on, and toggling it doesn't
        # imply anything about wanting to see/tune them right then.
        self.run_multiscale_coverage = QCheckBox("Multi-scale coverage")
        self.run_multiscale_coverage.setToolTip(
            "Layers two EXTRA coverage passes on top of the normal sweep grid, at "
            "different FOVs depending on how far the real points in a given direction "
            "are from the scanner (see core.projection.compute_multiscale_coverage_grid): "
            "a tighter-zoomed pass over the scan's own farthest points (a distant object "
            "subtends fewer panorama pixels than a near one, so it needs a closer zoom to "
            "become legible to DINO), and a wider pass over its own closest points (already "
            "big/legible, so trading some resolution for more surrounding context costs "
            "little). Percentile-based, not a fixed distance, so it scales automatically "
            "across very different scan sizes. Refine (below, if also on) still runs "
            "afterward on everything either pass finds. Costs extra sweep views - a scan "
            "with a lot of genuinely distant OR close material takes noticeably longer. "
            "Its own far/near percentile and FOV scale settings are in the gear button "
            "next to this checkbox, not shown here - the defaults already work well."
        )
        multiscale_row = QHBoxLayout()
        multiscale_row.addWidget(_row_with_reset(self.run_multiscale_coverage, False))
        btn_multiscale_settings = QToolButton()
        btn_multiscale_settings.setAutoRaise(True)
        btn_multiscale_settings.setText("⚙")  # gear
        btn_multiscale_settings.setToolTip("Multi-scale coverage settings (far/near percentile, FOV scale).")
        btn_multiscale_settings.clicked.connect(lambda: self._multiscale_settings_dialog.exec())
        multiscale_row.addWidget(btn_multiscale_settings)
        layout.addLayout(multiscale_row)
        self._multiscale_settings_dialog = self._build_multiscale_settings_dialog()

        # Off by default (unlike Merge overlapping detections above) - this
        # changes the coarse sweep itself into a DINO-only pass (see
        # SegmentationService.run_sweep's own refine_detections docstring),
        # experimental enough to opt into deliberately rather than changing
        # what every existing sweep does out of the box.
        self.run_refine_detections = QCheckBox("Refine boundary-cut detections")
        self.run_refine_detections.setToolTip(
            "Turns the coarse sweep into a DINO-ONLY box-finding pass - SAM2 "
            "never runs there at all - then runs the real DINO+SAM2 pass "
            "exactly once per found box, on a view re-centered on that box's "
            "own middle and sized to just cover its extent. Fixes an object "
            "the coarse sweep's own view boundaries cut through (only a "
            "sliver visible right at a view's edge), and avoids ever running "
            "SAM2 on a coarse-pass mask that would just get thrown away once "
            "re-segmented anyway - one real DINO+SAM2 call per object, not "
            "two."
        )
        layout.addWidget(_row_with_reset(self.run_refine_detections, False))

        refine_box = QGroupBox()
        refine_box.setFlat(True)
        refine_form = _form(refine_box)
        self.run_refine_margin = _double_spin(0.0, 3.0, 0.1, 0.5)
        self.run_refine_margin.setToolTip(
            "Padding added around each detection's own angular extent before "
            "re-viewing it, as a fraction of that extent (0.5 = 50% margin on "
            "every side) - some room around the object, not just its exact "
            "outline."
        )
        refine_form.addRow("Margin:", _row_with_reset(self.run_refine_margin, 0.5))
        self.run_refine_min_fov = _double_spin(5.0, 90.0, 1.0, 25.0)
        self.run_refine_min_fov.setToolTip(
            "Floor on the refined view's own FOV, degrees - keeps a small "
            "object's re-view from getting so tightly cropped that DINO/SAM2 "
            "lose useful surrounding context (too much focus is its own "
            "problem, not just too little)."
        )
        refine_form.addRow("Min FOV:", _row_with_reset(self.run_refine_min_fov, 25.0))
        self.run_refine_box_thresh = _double_spin(0.0, 1.0, 0.05, 0.45)
        self.run_refine_box_thresh.setToolTip(
            "DINO box threshold for JUST the refine pass's own detections - "
            "higher than the coarse sweep's own DINO box threshold above by "
            "default, since a zoomed-in, re-centered view naturally raises "
            "DINO's confidence for the SAME real object (fills more of the "
            "frame, less occluded by whatever sat at the coarse view's own "
            "edge) - reusing the coarse threshold here would let through "
            "more low-quality boxes than the coarse pass itself would have."
        )
        refine_form.addRow("Refine box threshold:", _row_with_reset(self.run_refine_box_thresh, 0.45))
        layout.addWidget(refine_box)
        refine_box.setVisible(self.run_refine_detections.isChecked())
        self.run_refine_detections.toggled.connect(refine_box.setVisible)

        btn_run = QPushButton("Run")
        btn_run.setToolTip("Sweeps and segments every match, added as a new layer on top of the stack.")
        btn_run.clicked.connect(self.run_prompt_segmentation)
        layout.addWidget(btn_run)

        # No page-wide "Reset to Defaults" button any more (design review
        # 2026-09-11) - each field above resets itself via its own small
        # icon (see _row_with_reset), which only appears once that
        # particular field has actually been changed. reset_run_prompt_defaults
        # (below) still exists and still resets everything at once - it's
        # just no longer exposed here, only via the global "Reset All
        # Settings" (Tools menu, see reset_all_settings).

        layout.addStretch()
        return w

    def _parse_categories(self, prompt: str) -> list[str]:
        return [p.strip() for p in prompt.split(",") if p.strip()]

    def _refresh_category_order_list(self, prompt: str) -> None:
        """Re-parses the prompt into categories on every edit. Deliberately simple:
        always rebuilds in prompt order, so a manual reorder is lost if the prompt
        text is edited afterward - acceptable since re-typing the prompt is itself
        a bigger change than the ranking, and this keeps the list from ever holding
        a stale category no longer in the prompt."""
        self.category_list.clear()
        self.category_list.addItems(self._parse_categories(prompt))

    def _on_category_manual_order_toggled(self, checked: bool) -> None:
        self.category_list.setEnabled(checked)

    def _move_category(self, delta: int) -> None:
        row = self.category_list.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < self.category_list.count()):
            return
        item = self.category_list.takeItem(row)
        self.category_list.insertItem(target, item)
        self.category_list.setCurrentRow(target)

    def _category_label_order(self) -> list[str] | None:
        """None means 'use the default automatic size ordering' - see _add_result_by_label."""
        if not self.category_manual_order.isChecked():
            return None
        return [self.category_list.item(i).text() for i in range(self.category_list.count())]

    # ── Loading ──────────────────────────────────────────────────────────

    def load_node(self, node: ScanNode, view_decimation: int = 1) -> None:
        if self.point_item is not None:
            self.gl_view.removeItem(self.point_item)
            self.gl_view.makeCurrent()
            self.point_item.cleanup()
            self.gl_view.doneCurrent()
            self.point_item = None

        self.node = node
        self.layer_panel.clear()  # a new scan's point indices don't correspond to the old one's layers
        self.status_label.setText("")

        # Everything below is point-index/object state left over from the
        # PREVIOUS scan - layer_panel.clear() above only clears the layer
        # stack itself; none of this is implicitly tied to it. Left stale,
        # this ranges from silently-wrong (a pick session still armed at a
        # layer object that's no longer in the panel, an old multi-point
        # accumulation reused against the new scan's different point
        # indices) to a real crash (_focus_indices holding indices past the
        # new node's own, smaller point_count - see _recompute_composition's
        # `new_hidden_mask[self._focus_indices] = False`).
        self._pick_target_layer = None
        self._pending_picks = []
        self.pending_picks_box.setVisible(False)
        self._clear_multipoint_session()
        self._clear_hover_preview()
        self._clear_instance_selection()
        self._focus_indices = None
        self._last_deletion = None
        self.btn_undo_deletion.setEnabled(False)

        if self.roi_indicator_item is not None:
            self.gl_view.removeItem(self.roi_indicator_item)
            self.roi_indicator_item = None
        self.roi_group.blockSignals(True)
        self.roi_group.setChecked(False)
        self.roi_group.blockSignals(False)

        available = [m for m in COLOR_MODES if self._mode_available(node, m)]
        if not available:
            available = ["rgb"]
        self.combo_color.blockSignals(True)
        self.combo_color.clear()
        self.combo_color.addItems(available)
        self.combo_color.blockSignals(False)

        self.view_decimation_spin.blockSignals(True)
        self.view_decimation_spin.setValue(view_decimation)
        self.view_decimation_spin.blockSignals(False)

        color_mode = available[0]
        self.point_item = ShaderPointCloudItem(
            node, color_mode=color_mode, point_size=self.point_size,
            render_mode=self.combo_render.currentIndex(), view_decimation=view_decimation,
        )
        # Initialize its GL state (shader compile, VBOs) HERE, synchronously,
        # rather than leaving it to its own lazy first paint() - a caller
        # that restores a layer stack right after load_node() returns (see
        # MainWindow.export_at_full_resolution/reload_at_different_detail_dialog/
        # open_project, all of which set layer_panel.layers + call refresh()
        # synchronously, before the event loop gets a turn to actually paint
        # anything) would otherwise call apply_layer_colors()/
        # upload_color_data() while vbo_col is still None - a real reported
        # bug: those calls silently no-op (see their own `if self.vbo_col is
        # None: return` guard), so the transferred layers exist in the
        # Layers panel but the 3D view keeps showing plain base colors until
        # something else happens to re-trigger composition later. Doing this
        # eagerly means vbo_col (and everything else) is always ready by the
        # time load_node() returns, for every caller.
        self.gl_view.makeCurrent()
        self.point_item.initializeGL()
        self.gl_view.doneCurrent()
        self.gl_view.addItem(self.point_item)
        self.center_camera()
        self._update_auto_channel_labels(node)
        # Carry this widget's own outline-toggle setting over to the new
        # point item - a fresh ShaderPointCloudItem always starts at ITS OWN default
        # (enabled), which would silently override an explicit "off" the
        # user set via the View menu on every new load otherwise. Pure
        # state, no GL context needed (see set_outline_enabled).
        self.point_item.set_outline_enabled(self.outline_effect_enabled)
        logging.info(f"Loaded '{node.scan_name}': {node.point_count:,} points, color modes: {available}")

    def _update_auto_channel_labels(self, node: ScanNode) -> None:
        """Rewrites every channel combo's "auto" entry to show what it actually
        resolves to for THIS scan (e.g. "auto (rgb)") - using the exact same
        default_channel() the CLI's own _resolve_channel() uses, so the GUI and
        CLI never disagree about what "auto" means for a given scan. Different
        scans can resolve differently (depends on which channels that scan
        actually has), so this re-runs on every load, not just once at startup."""
        from core.projection import default_channel
        resolved = default_channel(node)
        for combo in (self.run_dino_channel, self.run_sam_channel):
            model = combo.model()
            auto_item = model.item(0)  # CHANNEL_CHOICES[0] == "auto", always
            auto_item.setText(f"auto ({resolved})")

    def _on_view_decimation_changed(self, step: int) -> None:
        if self.point_item is None:
            return
        # The density-adaptive recompute this can trigger (see
        # ShaderPointCloudItem._ensure_decimation_mask) runs synchronously on
        # the UI thread and can take several real seconds on a very large
        # scan - a wait cursor at least reads as "working," not "frozen,"
        # for that stretch. Not worth a full background-thread treatment
        # (see workers.py) for a single infrequent slider change, but a
        # silent multi-second freeze with no cursor feedback at all would
        # read as a hang.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.gl_view.makeCurrent()
            self.point_item.set_view_decimation(step)
            self.gl_view.doneCurrent()
            self.gl_view.update()
        finally:
            QApplication.restoreOverrideCursor()

    # ── Region of interest (theta/phi crop, used by Run) ────────────────────

    def _roi_ranges(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """None, None when the ROI box is unchecked (no restriction, unchanged
        behavior). Otherwise both axes are always returned together as a single
        2D wedge - lo/hi are sorted defensively so which spinbox holds the larger
        value never matters."""
        if not self.roi_group.isChecked():
            return None, None
        t_lo, t_hi = sorted((self.roi_theta_lo.value(), self.roi_theta_hi.value()))
        p_lo, p_hi = sorted((self.roi_phi_lo.value(), self.roi_phi_hi.value()))
        if t_hi <= t_lo:
            t_hi = t_lo + 0.01
        if p_hi <= p_lo:
            p_hi = p_lo + 0.01
        return (t_lo, t_hi), (p_lo, p_hi)

    def _on_roi_theta_spin_changed(self, *_args) -> None:
        self.roi_theta_slider.setValues(self.roi_theta_lo.value(), self.roi_theta_hi.value())
        self._on_roi_changed()

    def _on_roi_phi_spin_changed(self, *_args) -> None:
        self.roi_phi_slider.setValues(self.roi_phi_lo.value(), self.roi_phi_hi.value())
        self._on_roi_changed()

    def _on_roi_theta_slider_changed(self, lo: float, hi: float) -> None:
        self.roi_theta_lo.blockSignals(True)
        self.roi_theta_hi.blockSignals(True)
        self.roi_theta_lo.setValue(lo)
        self.roi_theta_hi.setValue(hi)
        self.roi_theta_lo.blockSignals(False)
        self.roi_theta_hi.blockSignals(False)
        self._on_roi_changed()

    def _on_roi_phi_slider_changed(self, lo: float, hi: float) -> None:
        self.roi_phi_lo.blockSignals(True)
        self.roi_phi_hi.blockSignals(True)
        self.roi_phi_lo.setValue(lo)
        self.roi_phi_hi.setValue(hi)
        self.roi_phi_lo.blockSignals(False)
        self.roi_phi_hi.blockSignals(False)
        self._on_roi_changed()

    def _on_roi_changed(self, *_args) -> None:
        theta_range, phi_range = self._roi_ranges()
        if self.point_item is not None:
            self.gl_view.makeCurrent()
            self.point_item.set_angular_roi(theta_range, phi_range)
            self.gl_view.doneCurrent()
        self._update_roi_indicator(theta_range, phi_range)
        self.gl_view.update()

    def reset_roi_defaults(self) -> None:
        """Restores the Region of interest box to its as-built defaults
        (unchecked, theta -45..45, phi -45..45) - setValue()/setChecked()
        only actually fire their connected signals when the value truly
        changes, so this is safe to call even when some fields are already
        at their default."""
        self.roi_group.setChecked(False)
        self.roi_theta_lo.setValue(-45.0)
        self.roi_theta_hi.setValue(45.0)
        self.roi_phi_lo.setValue(-45.0)
        self.roi_phi_hi.setValue(45.0)

    def reset_run_prompt_defaults(self) -> None:
        """Restores every Run Prompt SETTING (not the typed prompt text
        itself - that's user content, not a parameter with a "default") to
        its as-built default."""
        self.category_manual_order.setChecked(False)
        self.run_overlap.setValue(0.2)
        self.run_fov.setValue(90.0)
        self.run_dino_channel.setCurrentIndex(0)  # "auto"
        self.run_sam_channel.setCurrentIndex(0)
        self.run_dino_box_thresh.setValue(0.3)
        self.run_dino_text_thresh.setValue(0.25)
        self.run_max_box_area.setValue(0.95)
        self.run_sam_score_thresh.setValue(0.0)
        self.run_min_score.setValue(0.0)
        self.run_merge_duplicates.setChecked(True)
        self.run_merge_iou.setValue(0.10)
        self.run_refine_detections.setChecked(False)
        self.run_refine_margin.setValue(0.5)
        self.run_refine_min_fov.setValue(25.0)
        self.run_refine_box_thresh.setValue(0.45)
        self.run_max_box_area.setValue(0.95)
        self.run_multiscale_coverage.setChecked(False)
        self.run_far_percentile.setValue(75.0)
        self.run_far_fov_scale.setValue(0.5)
        self.run_near_percentile.setValue(25.0)
        self.run_near_fov_scale.setValue(1.3)

    def reset_all_settings(self) -> None:
        """Run's settings AND the ROI box - the single global reset offered
        alongside Run's own local one (see main_window.py's Tools menu).
        Single-point picking has no persistent settings at all any more (see
        _start_pick_into_layer - it always uses the current view channel and
        a fixed FOV) - nothing here for it to reset."""
        self.reset_roi_defaults()
        self.reset_run_prompt_defaults()
        self.status_label.setText("All settings reset to defaults.")

    def save_project(self, path: str) -> None:
        """Gathers everything a project file needs to describe the CURRENT
        editing state (see project_file.py's own module docstring for the
        format) and writes it - the raw point cloud itself is never
        duplicated, just referenced by node.file_path + the exact Load
        detail it was loaded with, reloaded the normal way on open (see
        MainWindow.open_project)."""
        if self.node is None:
            self.status_label.setText("Nothing to save - load a scan first.")
            return
        import project_file
        theta_range, phi_range = self._roi_ranges()
        project_file.save_project(
            path,
            source_path=self.node.file_path,
            load_decimation=self.node.decimation_step,
            point_count=self.node.point_count,
            view_decimation=self.point_item.view_decimation if self.point_item is not None else 1,
            color_mode=self.point_item.current_color_mode if self.point_item is not None else "rgb",
            render_mode=self.combo_render.currentIndex(),
            roi_enabled=self.roi_group.isChecked(),
            roi_theta_range=theta_range,
            roi_phi_range=phi_range,
            layers=self.layer_panel.layers,
        )
        self.status_label.setText(f"Project saved: {path}")

    def restore_project_state(self, manifest: dict, layers: list) -> None:
        """Applies everything from an opened project file EXCEPT the scan
        itself - MainWindow.open_project already reloaded that via the
        normal load_node path (using load_decimation/view_decimation FROM
        this same manifest, so point indices below line up) before calling
        this. Restores the layer stack, then color/render mode, then ROI -
        that order matters: restoring layers first means the
        color/render-mode changes' own recompose picks up the RESTORED
        layers, not whatever was on screen before (nothing, right after a
        fresh load)."""
        self.layer_panel.layers = layers
        self.layer_panel.selected_layer = layers[0] if layers else None
        self.layer_panel.refresh()  # rebuilds rows + recomposes the 3D view's colors

        color_mode = manifest.get("color_mode")
        if color_mode:
            idx = self.combo_color.findText(color_mode)
            if idx >= 0:
                self.combo_color.setCurrentIndex(idx)
        self.combo_render.setCurrentIndex(manifest.get("render_mode", 0))

        theta_range = manifest.get("roi_theta_range")
        phi_range = manifest.get("roi_phi_range")
        if theta_range is not None:
            self.roi_theta_lo.setValue(theta_range[0])
            self.roi_theta_hi.setValue(theta_range[1])
        if phi_range is not None:
            self.roi_phi_lo.setValue(phi_range[0])
            self.roi_phi_hi.setValue(phi_range[1])
        self.roi_group.setChecked(bool(manifest.get("roi_enabled", False)))

    def _to_world_points(self, indices: np.ndarray) -> np.ndarray:
        n = self.node
        return n.points_quantized[indices].astype(np.float32) * n.quant_scale + n.quant_center

    def snapshot_for_label_transfer(self) -> tuple[list, "label_transfer.VoxelOwnerMap"]:
        """Captures everything MainWindow.reload_at_different_detail needs to
        carry the current Layers stack over to a RE-LOAD of this same scan
        at a DIFFERENT Load detail - see label_transfer.py's own module
        docstring for why this can't just reuse the old point indices (a
        different decimation samples an entirely different set of points).

        Returns (layers_meta, voxel_owner_map). `layers_meta` is
        [(name, source_mode, visible, expanded, [(key, label, color,
        included, score, manual), ...]), ...] - everything about the
        current layer/instance structure EXCEPT indices, which get
        recomputed against the newly (re)loaded scan by apply_label_transfer
        instead. `voxel_owner_map` (see label_transfer.build_voxel_owner_map/
        VoxelOwnerMap - pass the whole object through unchanged, don't pull
        `.owner` back out of it) is built HERE, before the reload, while
        self.node still holds the OLD points these instances' indices
        actually refer to - it's the only piece that needs those old points
        at all, and once built it doesn't need them again."""
        from viewer import label_transfer

        layers_meta = []
        all_points: list[np.ndarray] = []
        all_ids: list[np.ndarray] = []
        key = 0
        for layer in self.layer_panel.layers:
            inst_metas = []
            for inst in layer.instances:
                valid = inst.indices[(inst.indices >= 0) & (inst.indices < self.node.point_count)]
                if len(valid):
                    all_points.append(self._to_world_points(valid))
                    all_ids.append(np.full(len(valid), key, dtype=np.int64))
                inst_metas.append((key, inst.label, inst.color, inst.included, inst.score, inst.manual))
                key += 1
            layers_meta.append((layer.name, layer.source_mode, layer.visible, layer.expanded, inst_metas))

        old_points_xyz = np.concatenate(all_points) if all_points else np.zeros((0, 3), dtype=np.float32)
        old_instance_ids = np.concatenate(all_ids) if all_ids else np.zeros(0, dtype=np.int64)
        voxel_owner_map = label_transfer.build_voxel_owner_map(old_points_xyz, old_instance_ids)
        return layers_meta, voxel_owner_map

    def apply_label_transfer(self, layers_meta: list, voxel_owner_map: "label_transfer.VoxelOwnerMap") -> tuple[list, dict]:
        """Rebuilds the Layer/LayerInstance structure from `layers_meta`
        (see snapshot_for_label_transfer), assigning each instance NEW
        indices from self.node - which must already be the freshly
        (re)loaded scan at this point (MainWindow.reload_at_different_detail
        calls this from open_file's on_loaded hook, same pattern
        open_project uses for restore_project_state). An instance that ends
        up with zero transferred points (none of its old voxels had a match
        among the new points - see label_transfer.py's own docstring for
        why that can happen) is dropped, not kept empty - counted in the
        returned stats so the caller can tell the user.

        Returns (new_layers, stats) - stats = {"kept": n, "dropped": n}."""
        from viewer import label_transfer

        new_points_xyz = self._to_world_points(np.arange(self.node.point_count))
        result = label_transfer.transfer_owners(new_points_xyz, voxel_owner_map)

        new_layers = []
        stats = {"kept": 0, "dropped": 0}
        for name, source_mode, visible, expanded, inst_metas in layers_meta:
            new_instances = []
            for key, label, color, included, score, manual in inst_metas:
                new_indices = np.where(result == key)[0]
                if len(new_indices) == 0:
                    stats["dropped"] += 1
                    continue
                stats["kept"] += 1
                new_instances.append(LayerInstance(
                    indices=new_indices, label=label, color=color,
                    included=included, score=score, manual=manual,
                ))
            new_layers.append(Layer(
                name=name, source_mode=source_mode, instances=new_instances,
                visible=visible, expanded=expanded,
            ))
        return new_layers, stats

    def _update_roi_indicator(self, theta_range: tuple[float, float] | None,
                               phi_range: tuple[float, float] | None) -> None:
        if self.roi_indicator_item is not None:
            self.gl_view.removeItem(self.roi_indicator_item)
            self.roi_indicator_item = None
        if theta_range is None or phi_range is None or self.node is None:
            return
        pts = _wedge_boundary_points(self.node, theta_range, phi_range)
        if pts is None:
            return
        self.roi_indicator_item = gl_module.GLLinePlotItem(
            pos=pts, color=(1.0, 0.85, 0.1, 1.0), width=2.0, mode="line_strip", antialias=True,
        )
        self.gl_view.addItem(self.roi_indicator_item)

    def _update_instance_overlay(self) -> None:
        """Camera-facing wireframe box + text label per instance, for ONLY the
        Layers panel's currently-selected layer (its row's select radio) - not
        the whole stack, so it doesn't get cluttered/slow with many objects.

        The box is the AXIS-ALIGNED extent of the instance's own real points
        (min/max xyz), not a re-projection of whatever 2D DINO box produced
        it - that box belongs to one specific sweep view/angle and has no
        inherent 3D pose of its own once the instance is expressed as a set
        of 3D point indices."""
        if self.overlay_box_item is not None:
            self.gl_view.removeItem(self.overlay_box_item)
            self.overlay_box_item = None
        for text_item in self.overlay_text_items:
            self.gl_view.removeItem(text_item)
        self.overlay_text_items = []

        if not self.show_instance_overlay_checkbox.isChecked():
            return
        layer = self.layer_panel.selected_layer
        if layer is None or self.node is None:
            return

        pts_all = self.node.points_quantized.astype(np.float32) * self.node.quant_scale + self.node.quant_center
        edge_verts = []
        edge_colors = []
        for i, inst in enumerate(layer.instances):
            if not inst.included or len(inst.indices) == 0:
                continue
            pts = pts_all[inst.indices]
            mn, mx = pts.min(axis=0), pts.max(axis=0)
            edges = _aabb_edge_vertices(mn, mx)
            edge_verts.append(edges)
            # Hovered/selected instance's box switches to the SAME tint its
            # points get (see ShaderPointCloudItem.hover_preview_points /
            # _refresh_instance_selection_highlight) - "if I clicked now,
            # this is what I'd select"/"this is what's selected" reads
            # consistently whether you're looking at the points or the box
            # around them. Hover wins over selected (it's the more
            # up-to-the-moment state) if somehow both apply at once.
            if inst is self._hovered_instance:
                box_color = tuple(c / 255.0 for c in HOVER_COLOR)
            elif id(inst) in self._selected_instance_ids:
                box_color = tuple(c / 255.0 for c in SELECTION_COLOR)
            else:
                box_color = tuple(c / 255.0 for c in inst.color[:3])
            color_rgba = (*box_color, 1.0)
            edge_colors.append(np.tile(np.array(color_rgba, dtype=np.float32), (len(edges), 1)))

            label_text = f"{inst.label} #{i + 1}"
            if inst.score is not None:
                label_text += f" - {inst.score:.2f}"
            cx, cy = (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0
            text_item = _OutlinedGLTextItem(
                pos=np.array([cx, cy, mx[2]], dtype=np.float32),
                text=label_text, font=QFont("Helvetica", self._overlay_label_size),
            )
            self.gl_view.addItem(text_item)
            self.overlay_text_items.append(text_item)

        if edge_verts:
            self.overlay_box_item = gl_module.GLLinePlotItem(
                pos=np.concatenate(edge_verts), color=np.concatenate(edge_colors),
                width=2.0, mode="lines", antialias=True,
            )
            self.gl_view.addItem(self.overlay_box_item)

    @staticmethod
    def _mode_available(node: ScanNode, mode: str) -> bool:
        return {
            "rgb": node.colors_rgb is not None,
            "intensity": node.colors_intensity is not None,
            "normals": node.normals is not None and node.normals.any(),
        }[mode]

    # ── View toolbar callbacks ───────────────────────────────────────────

    def _on_color_mode_changed(self, mode: str) -> None:
        if self.point_item is not None and mode:
            # set_color_mode() re-uploads a VBO internally - needs gl_view's
            # context current, same as _recompute_composition (which this
            # calls right after, and which already wraps its own GL calls).
            self.gl_view.makeCurrent()
            self.point_item.set_color_mode(mode)  # already reverts to the plain base color
            self.gl_view.doneCurrent()
            self._recompute_composition()  # ...then re-overlay the current layer stack on top of it
            self.gl_view.update()

    def _on_render_mode_changed(self, index: int) -> None:
        if self.point_item is not None:
            self.point_item.render_mode = index
            self.gl_view.update()

    def _change_point_size(self, delta: int) -> None:
        self.point_size = max(1, self.point_size + delta)
        if self.point_item is not None:
            self.point_item.point_size = self.point_size
            self.gl_view.update()

    def center_camera(self) -> None:
        if self.node is None:
            return
        c = self.node.quant_center
        self.gl_view.opts["center"] = QVector3D(float(c[0]), float(c[1]), float(c[2]))
        self.gl_view.update()

    def _on_cube_face_clicked(self, face_index: int) -> None:
        azimuth, elevation = face_snap_target(face_index, self.gl_view.opts["azimuth"])
        self.gl_view.setCameraPosition(azimuth=azimuth, elevation=elevation)
        self.gl_view.update()
        self.status_label.setText(f"View: {FACE_LABELS[face_index]}")

    def pick_background_color(self) -> None:
        """Opens a QColorDialog defaulting to the CURRENT background (whatever
        pyqtgraph's own GLViewWidget.opts['bgcolor'] holds - it's what its paint
        loop actually passes to glClearColor), applies the choice live, and
        persists it via QSettings so it survives an app restart. Called from
        MainWindow's View menu (moved out of this widget's own toolbar to free
        up panel space)."""
        current_rgba = self.gl_view.opts.get("bgcolor", (0.125, 0.125, 0.125, 1.0))
        initial = QColor.fromRgbF(*current_rgba)
        color = QColorDialog.getColor(initial, self, "Choose 3D view background color")
        if not color.isValid():
            return
        self.gl_view.setBackgroundColor(color)
        self.gl_view.update()
        self._settings.setValue("background_color", color.name())

    def reset_background_color(self) -> None:
        """Restores the documented default (#202020) - an explicit, named
        default rather than an open-ended color picker with nothing to fall
        back to."""
        self.gl_view.setBackgroundColor(DEFAULT_BACKGROUND_COLOR)
        self.gl_view.update()
        self._settings.setValue("background_color", DEFAULT_BACKGROUND_COLOR)

    def set_outline_effect_enabled(self, enabled: bool) -> None:
        """Toggles the hover/selection outline post-process on/off (View
        menu, a checkable action - see MainWindow._build_menu) - it's a real
        GPU render effect, not free, and not everyone wants it visually
        either. Persists via QSettings (same pattern as background color)
        so the choice survives an app restart; also carried over to a fresh
        ShaderPointCloudItem on every scan load (see load_node), since that
        object itself doesn't survive a reload to remember its own state."""
        self.outline_effect_enabled = enabled
        self._settings.setValue("outline_effect_enabled", enabled)
        if self.point_item is not None:
            self.gl_view.makeCurrent()
            self.point_item.set_outline_enabled(enabled)
            self.gl_view.doneCurrent()
            self.gl_view.update()

    def set_debug_output_folder(self) -> None:
        """Opens a folder picker to choose (or change) the BASE folder sweep
        debug images save into (see `debug_sweep_images`/SegmentationService.
        run_sweep's own `debug_dir`) - each Run gets its own timestamped
        subfolder inside it (see `_new_run_debug_dir`), so repeated runs
        never overwrite each other's images. Persists via QSettings, same
        pattern as background color/outline effect."""
        chosen = QFileDialog.getExistingDirectory(self, "Choose debug images folder", self.debug_dir or "")
        if not chosen:
            return
        self.debug_dir = chosen
        self._settings.setValue("debug_dir", chosen)

    def set_debug_sweep_images(self, enabled: bool) -> bool:
        """Toggles saving annotated debug images of Runs and picks (Tools menu - a
        checkable action, see MainWindow._build_menu). Turning it ON with no
        folder chosen yet (this session or a remembered one) prompts for one
        immediately via `set_debug_output_folder` - cancelling that prompt
        leaves this OFF, since there'd be nowhere to save to. Returns the
        actual resulting state (not just an echo of `enabled`) so the menu's
        own checkbox can self-correct against a cancelled folder prompt
        instead of trusting the click that triggered this call."""
        if enabled and not self.debug_dir:
            self.set_debug_output_folder()
            if not self.debug_dir:
                enabled = False
        self.debug_sweep_images = enabled
        return enabled

    def _new_run_debug_dir(self) -> str | None:
        """The per-run debug folder for THIS Run, or None if debug images are
        off - a fresh timestamped subfolder under `debug_dir` every time, so
        one Run's images never overwrite a previous Run's (see
        SegmentationService.run_sweep's own `debug_dir`, which further splits
        this into prefilter/refine subfolders)."""
        if not (self.debug_sweep_images and self.debug_dir):
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.debug_dir, f"sweep_{stamp}")

    def _pick_debug_dir(self) -> str | None:
        """Where point-click (pick) debug images go, or None if debug images
        are off: one shared `picks` subfolder under `debug_dir` - each pick
        is a single timestamped PNG (see point_segmentation_service.
        segment_at_point's `debug_dir`), unlike a Run's folder of views."""
        if not (self.debug_sweep_images and self.debug_dir):
            return None
        return os.path.join(self.debug_dir, "picks")

    def pick_label_size(self) -> None:
        """Font size (points) for the instance-box overlay's billboard labels
        (name + score) - same "menu action opens a small picker, applies live"
        treatment as pick_background_color, moved here from the Layers box's
        own Advanced settings (design review 2026-09: the user wanted it out
        of the Layers panel entirely, not just tucked away within it) - not
        persisted across restarts, unlike background color, since this is a
        much lower-stakes, rarely-touched cosmetic tweak."""
        value, accepted = QInputDialog.getInt(
            self, "Label Size", "Instance-box overlay label font size (points):",
            self._overlay_label_size, 6, 48,
        )
        if not accepted:
            return
        self._overlay_label_size = value
        self._update_instance_overlay()

    # ── Busy-state helpers (shared by all 3 background tasks) ──────────────

    def _set_busy(self, busy: bool, message: str = "", cancellable: bool = False) -> None:
        self._busy = busy
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)  # indeterminate until the first real progress signal arrives
            self.status_label.setText(message)
        else:
            self._cancel_event = None
        self.btn_cancel.setVisible(busy and cancellable)
        self.btn_cancel.setEnabled(True)  # re-enable for the next run, even if a cancel click disabled it below

    def _cancel_current_task(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.btn_cancel.setEnabled(False)  # already requested - the sweep stops at its next view boundary
            self.status_label.setText("Cancelling... (takes effect at the next view)")

    def _on_task_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_log_message(self, message: str) -> None:
        self.status_label.setText(message)

    def _start_log_forwarding(self) -> None:
        """Surfaces model-loading (and other) log messages in the status label
        during the gap between clicking Run and the sweep's own progress_callback
        starting (DINO/SAM2 loading on first use happens inside run_sweep, before
        its per-view loop) - detach_log_forwarding() must be called from every
        completion path (success/error/cancelled) once the task is done."""
        self._log_handler = attach_log_forwarder(self._log_forwarder)

    def _stop_log_forwarding(self) -> None:
        if self._log_handler is not None:
            detach_log_forwarder(self._log_handler)
            self._log_handler = None

    # ── Layer stack (pick / run all funnel here) ─────────────────────────────

    def _add_result_by_label(self, result, source_mode: str, what: str,
                              label_order: list[str] | None = None,
                              roi_ranges: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None) -> None:
        """Splits `result` into ONE LAYER PER DISTINCT LABEL - a multi-
        category prompt like "window, building" (see SegmentationService.
        _detect_all_phrases) makes a "window" layer and a "building" layer,
        never one mixed-label layer (design review 2026-09-10). Appends into
        an already-existing SAME-NAMED layer instead of creating a duplicate
        if a previous run (or a manual "+ New empty layer...") already made
        one for that label - same "add into an existing bucket, not always
        a fresh one" pattern _start_pick_into_layer's "Pick into this
        layer..." already established for single-point picking.

        `roi_ranges`, when given (a Run call made while the ROI box was
        active), HARD-CLIPS every instance's indices to the exact requested
        theta/phi box before any of this - via the SAME angle computation/
        mask the ROI's own visual crop already uses
        (ShaderPointCloudItem._angular_mask, itself built on
        ensure_point_angles' index_mapper.point_to_pano-derived theta/phi,
        NOT world XYZ). Necessary because the sweep itself only restricts
        view CENTERS, not each view's FOV cone - a detection can legitimately
        back-project a few points outside the requested box even though
        every view that contributed to it was correctly centered inside it
        (see the ROI group's own tooltip). An instance left with zero points
        after clipping is dropped entirely.

        `label_order`, if given (the user's own drag-ordered category
        ranking, top = wins - see _category_label_order), decides which
        label's LAYER ends up highest in the stack instead of the default
        largest-total-points-wins: this used to rank instances WITHIN one
        shared layer (see git history's layer_from_result), now it ranks
        LAYERS against each other instead, since each label gets its own."""
        if roi_ranges is not None and self.point_item is not None:
            theta_range, phi_range = roi_ranges
            mask = self.point_item._angular_mask(theta_range=theta_range, phi_range=phi_range)
            kept = []
            n_clipped_points = 0
            for inst in result.instances:
                clipped = inst.indices[mask[inst.indices]]
                n_clipped_points += len(inst.indices) - len(clipped)
                if len(clipped) == 0:
                    continue
                inst.indices = clipped
                kept.append(inst)
            result.instances = kept
            if n_clipped_points:
                logging.info(f"ROI hard-clip: removed {n_clipped_points:,} point(s) outside the requested box.")

        if not result.instances:
            self.status_label.setText(f"{what}: no objects found - no layer added.")
            return

        by_label: dict[str, list] = {}
        for inst in result.instances:
            by_label.setdefault(inst.label, []).append(inst)

        if label_order:
            ranked = [lbl for lbl in label_order if lbl in by_label]
            unranked = sorted(
                (lbl for lbl in by_label if lbl not in label_order),
                key=lambda lbl: sum(len(i.indices) for i in by_label[lbl]), reverse=True,
            )
            # add_layer always inserts at the TOP of the stack - process
            # LOSER-first, WINNER-last, so label_order[0] (top of the list,
            # "wins") ends up added dead last and so highest in the stack.
            process_order = unranked + list(reversed(ranked))
        else:
            process_order = sorted(
                by_label, key=lambda lbl: sum(len(i.indices) for i in by_label[lbl]), reverse=True,
            )

        new_layers = 0
        appended_layers = 0
        for label in process_order:
            group = sorted(by_label[label], key=lambda inst: len(inst.indices), reverse=True)
            existing = next((l for l in self.layer_panel.layers if l.name == label), None)
            if existing is not None:
                for inst in group:
                    self.layer_panel.add_instance(existing, LayerInstance(
                        indices=inst.indices, label=inst.label, color=inst.color,
                        score=getattr(inst, "score", None),
                    ))
                appended_layers += 1
            else:
                new_instances = [
                    LayerInstance(indices=inst.indices, label=inst.label, color=inst.color,
                                  score=getattr(inst, "score", None))
                    for inst in group
                ]
                self.layer_panel.add_layer(Layer(name=label, source_mode=source_mode, instances=new_instances))
                new_layers += 1
        # triggers `changed` -> _recompute_composition, per add_layer/add_instance

        n_objects = len(result.instances)
        n_points = len(result.combined_indices())
        layer_summary = f"{new_layers} new layer(s)" if not appended_layers else \
            f"{new_layers} new, {appended_layers} appended layer(s)"
        self.status_label.setText(
            f"{what}: {n_objects} object(s), {n_points:,} points total across "
            f"{len(by_label)} label(s) - {layer_summary}."
        )

    def _recompute_composition(self) -> None:
        """Re-derives node.point_classes and the 3D view's colors from the CURRENT
        layer stack, from scratch, every time - a plain overlay-on-top-of-previous
        update couldn't handle a deleted/hidden/reordered layer's points reverting
        to unclassified, since compose_layers only ever touches indices it's given."""
        if self.node is None or self.point_item is None:
            return

        from core.classification import compose_layers

        entries = self.layer_panel.compose_entries()  # [(label, indices, color), ...] bottom-of-stack first
        self.node.point_classes = None
        self.node.class_mapping = None

        # apply_layer_colors/upload_color_data re-upload a VBO - need gl_view's
        # context current. This method runs as a signal handler off arbitrary
        # Layers-panel actions (a checkbox toggle, a button click, an
        # add/reorder/delete) with no GL context guaranteed current at that
        # point - same real crash class as the instance-selection highlight
        # fix (GL_INVALID_OPERATION on glBindBuffer), caught proactively here
        # before it was ever actually reported for THIS call site.
        self.gl_view.makeCurrent()
        if entries:
            compose_layers(self.node, [(label, indices) for label, indices, _color in entries])
            self.point_item.apply_layer_colors(entries)
        else:
            self.point_item.upload_color_data()
        # Layers with points_hidden=True get folded into node.hidden_mask
        # here too - same "recompute from the whole current stack every
        # time" reasoning as the coloring just above (a layer that just
        # turned points_hidden back off needs its points to reappear, which
        # an additive-only update couldn't handle any more than the color
        # composition above could). Requires `visible` too, not points_hidden
        # alone (design review 2026-09-12) - "hide points" is a refinement of
        # a layer actually being on, not an independent switch: turning the
        # layer's own visibility off should make its points reappear
        # (as unclassified) even if "hide points" was still checked, rather
        # than leaving them invisible for a reason no longer visible anywhere
        # in the UI once visible is off.
        #
        # _focus_indices (see _focus_on_selected/_clear_focus) is an
        # EXCLUSIVE override when set: only those points render, full stop,
        # regardless of what any individual layer's own points_hidden says -
        # simpler and less surprising than trying to combine "hide these"
        # (per layer) with "show ONLY these" (focus) at the same time.
        if self._focus_indices is not None:
            new_hidden_mask = np.ones(self.node.point_count, dtype=bool)
            new_hidden_mask[self._focus_indices] = False
        else:
            hidden_arrays = [
                layer.hidable_indices() for layer in self.layer_panel.layers
                if layer.visible and layer.points_hidden
            ]
            new_hidden_mask = np.zeros(self.node.point_count, dtype=bool)
            if hidden_arrays:
                # No np.unique - a boolean assignment like this is
                # duplicate-safe (see _union_indices' own docstring), and
                # np.unique's sort was a real, reported freeze on a large
                # hidden instance.
                new_hidden_mask[_union_indices(hidden_arrays)] = True
        self.point_item.set_hidden_mask(new_hidden_mask)
        self.gl_view.doneCurrent()
        self.gl_view.update()
        # apply_layer_colors/upload_color_data both reset point_item's OWN
        # hover-preview bookkeeping (the base array they just replaced makes
        # any prior hover backup stale) - resync this side's tracking too, or
        # _on_hover_moved would see "already showing this" for the still-
        # hovered instance and skip re-applying a tint that's actually gone.
        self._hovered_instance = None
        self.point_item.set_hover_outline(None, _to_unit_color(HOVER_COLOR))  # pure state - no GL context needed

    def clear_segmentation(self) -> None:
        self.layer_panel.clear()  # triggers `changed` -> _recompute_composition (reverts to plain colors)
        self.status_label.setText("")

    def export_segmentation(self) -> None:
        mask = self.layer_panel.combined_indices()
        if self.node is None or len(mask) == 0:
            self.status_label.setText("Nothing to export - add at least one layer first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Segmentation", "segmentation.las", "LAS point cloud (*.las)")
        if not path:
            return

        from fileio.exporters import export_to_las

        message = export_to_las(self.node, path, mask=mask)
        self.status_label.setText(message)
        logging.info(message)

    # ── Prompt-driven sweep (whole-sweep, no mouse interaction) ────────────

    def set_dino_size(self, size: str) -> None:
        """Changes which GroundingDINO weight size the NEXT sweep uses (see
        models_dialog.py) - drops any already-loaded service so
        _ensure_dino_service() constructs a fresh one against the new
        config/checkpoint path instead of keeping the old size loaded."""
        if size == self.dino_size:
            return
        self.dino_size = size
        self._dino_service = None

    def set_sam_size(self, size: str) -> None:
        """Same as set_dino_size, for SAM2."""
        if size == self.sam_size:
            return
        self.sam_size = size
        self._sam_service = None

    def _ensure_dino_service(self):
        if self._dino_service is None:
            from services.dino_service import DinoService

            import model_sizes
            config_path, checkpoint_path = model_sizes.dino_paths(self.dino_size)
            logging.info(f"Loading GroundingDINO ({self.dino_size}, first use of this size this session)...")
            self._dino_service = DinoService(config_path=str(config_path), checkpoint_path=str(checkpoint_path))
        return self._dino_service

    def _ensure_sam_service(self):
        if self._sam_service is None:
            from services.sam_service import SamService

            import model_sizes
            config_path, checkpoint_path = model_sizes.sam_paths(self.sam_size)
            logging.info(f"Loading SAM2 ({self.sam_size}, first use of this size this session)...")
            self._sam_service = SamService(config_path=config_path, checkpoint_path=str(checkpoint_path))
        return self._sam_service

    def _build_view_grid(self):
        """The coverage grid Run's own current settings would actually sweep -
        shared by run_prompt_segmentation and save_coverage_grid_preview so
        the preview always shows EXACTLY what a real Run would do, never a
        second, potentially-drifting computation of the same thing. Scoped to
        the active Region of interest's own theta/phi range directly (not
        just post-hoc-clipped after a full-scan sweep - real compute savings,
        not just a tidier result): when ROI is off, theta_range/phi_range are
        both None, which compute_coverage_grid already treats as "the scan's
        own real-data extent"."""
        from core.projection import (
            compute_coverage_grid,
            compute_multiscale_coverage_grid,
            real_scan_mask,
        )

        roi_theta, roi_phi = self._roi_ranges()
        if self.run_multiscale_coverage.isChecked():
            # (theta, phi, fov) triples - run_sweep's own view_grid accepts
            # this mixed-arity form to override its global fov per view (see
            # its own docstring) - the plain compute_coverage_grid path below
            # (theta, phi) pairs only uses run_sweep's single global fov.
            return compute_multiscale_coverage_grid(
                self.node, fov=self.run_fov.value(), overlap=self.run_overlap.value(),
                far_percentile=self.run_far_percentile.value(), far_fov_scale=self.run_far_fov_scale.value(),
                near_percentile=self.run_near_percentile.value(), near_fov_scale=self.run_near_fov_scale.value(),
                phi_range=roi_phi, theta_range=roi_theta,
            )
        return compute_coverage_grid(
            real_scan_mask(self.node), fov=self.run_fov.value(),
            overlap=self.run_overlap.value(), phi_range=roi_phi, theta_range=roi_theta,
        )

    def save_coverage_grid_preview(self) -> None:
        """Tools > Save Coverage Grid Preview...: saves a PNG of the scan's
        full panorama showing how the CURRENT Run settings' coverage grid
        (see `_build_view_grid`) covers it - every scanned pixel tinted by how
        many views see it (gaps in red), each view's true footprint outlined
        and numbered in sweep order (see
        `core.projection.render_coverage_grid_preview`) - so you can inspect
        exactly what a Run would sweep, including Multi-scale coverage's
        extra zoom tiers, without spending the time (or GPU memory) actually
        running DINO/SAM2.

        The save path is picked FIRST (cheap, instant) so a cancelled dialog
        never wastes the render itself - reconstructing the scan's full
        panorama can take a while on a large scan, so this runs on a
        background thread like every other heavy operation here."""
        if self.node is None:
            self.status_label.setText("Load a scan first.")
            return
        if self._busy:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Coverage Grid Preview", "coverage_grid.png", "PNG Image (*.png)",
        )
        if not path:
            return

        from core.projection import render_coverage_grid_preview

        node = self.node
        fov = self.run_fov.value()
        channel = _channel_arg(self.run_dino_channel, node)
        view_grid = self._build_view_grid()

        def task(progress_callback=None):
            import cv2
            image = render_coverage_grid_preview(node, view_grid, default_fov=fov, channel=channel)
            if not cv2.imwrite(path, image):
                raise RuntimeError(f"Failed to write {path!r} - check the path/permissions.")
            return path

        self._set_busy(True, "Rendering coverage grid preview...")

        def on_success(saved_path) -> None:
            self._set_busy(False)
            self.status_label.setText(f"Coverage grid preview saved to {saved_path}")

        def on_error(message: str) -> None:
            self._set_busy(False)
            self.status_label.setText(f"Coverage grid preview failed: {message}")

        self._task_thread, self._task_worker = run_in_background(
            task, on_success=on_success, on_error=on_error,
        )

    def run_prompt_segmentation(self) -> None:
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.status_label.setText("Enter a prompt first (e.g. 'door, window').")
            return
        if self.node is None or self._busy:
            return

        from services.segmentation_service import SegmentationService

        view_grid = self._build_view_grid()
        dino = self._ensure_dino_service()
        sam = self._ensure_sam_service()
        node = self.node

        # Every QWidget read happens here, on the main thread, BEFORE the worker
        # thread starts - Qt widgets aren't safe to touch from a non-GUI thread,
        # so task() below must only close over plain Python values, never self.
        # fov is ALSO needed here (not just for compute_coverage_grid above) -
        # run_sweep uses it again for every view's own perspective-crop
        # projection, independent of how the view grid was built.
        fov = self.run_fov.value()
        dino_channel = _channel_arg(self.run_dino_channel, node)
        sam_channel = _channel_arg(self.run_sam_channel, node)
        box_threshold = self.run_dino_box_thresh.value()
        text_threshold = self.run_dino_text_thresh.value()
        max_box_area_fraction = self.run_max_box_area.value()
        sam_score_threshold = self.run_sam_score_thresh.value()
        min_score = self.run_min_score.value()
        label_order = self._category_label_order()  # None -> _add_result_by_label's default size order
        merge_duplicates = self.run_merge_duplicates.isChecked()
        merge_iou = self.run_merge_iou.value()
        refine_detections = self.run_refine_detections.isChecked()
        refine_margin = self.run_refine_margin.value()
        refine_min_fov = self.run_refine_min_fov.value()
        refine_box_threshold = self.run_refine_box_thresh.value()
        debug_dir = self._new_run_debug_dir()
        roi_theta, roi_phi = self._roi_ranges()

        self._cancel_event = threading.Event()
        cancel_check = self._cancel_event.is_set

        def task(progress_callback=None):
            return SegmentationService(dino, sam).run_sweep(
                node, prompt,
                fov=fov,
                dino_channel=dino_channel,
                sam_channel=sam_channel,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_box_area_fraction=max_box_area_fraction,
                sam_score_threshold=sam_score_threshold,
                min_score=min_score,
                merge_duplicates=merge_duplicates,
                merge_iou=merge_iou,
                refine_detections=refine_detections,
                refine_margin=refine_margin,
                refine_min_fov=refine_min_fov,
                refine_box_threshold=refine_box_threshold,
                debug_dir=debug_dir,
                view_grid=view_grid,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

        self._set_busy(True, f"Running sweep for '{prompt}'...", cancellable=True)
        self._start_log_forwarding()  # covers the DINO/SAM2-loading gap before the sweep's own progress starts

        roi_ranges = (roi_theta, roi_phi) if roi_theta is not None else None

        def on_success(result) -> None:
            self._stop_log_forwarding()
            self._set_busy(False)
            self._add_result_by_label(result, source_mode="Run Prompt",
                                       what=f"Prompt '{prompt}'", label_order=label_order, roi_ranges=roi_ranges)
            if debug_dir is not None:
                self.status_label.setText(f"{self.status_label.text()} Debug images: {debug_dir}")

        def on_error(message: str) -> None:
            self._stop_log_forwarding()
            self._set_busy(False)
            self.status_label.setText(f"Run failed: {message}")

        def on_cancelled() -> None:
            self._stop_log_forwarding()
            self._set_busy(False)
            self.status_label.setText(f"Prompt '{prompt}': cancelled - no layer added.")

        self._task_thread, self._task_worker = run_in_background(
            task, on_success=on_success, on_error=on_error, on_progress=self._on_task_progress,
            on_cancelled=on_cancelled,
        )

    # ── Pick mode ────────────────────────────────────────────────────────

    def _on_point_clicked(self, pos: QPoint) -> None:
        """Right-click in the 3D view. Two entirely different meanings,
        never both applicable at once: mid-pick-session (see
        _start_pick_into_layer), it's "run SAM2 at this point" - otherwise
        (the common case), right-click was previously just idle here, so
        it's repurposed for a lightweight context menu offering "Focus on
        Selected"/"Show All" (see _show_viewport_context_menu) - an
        "isolate selection" toggle most 3D tools offer, invoked right on the
        object in the viewport rather than from a side panel."""
        if self._pick_target_layer is not None and self.point_item is not None and not self._busy:
            self._pick_point_index(pos)
            return
        self._show_viewport_context_menu(pos)

    def _show_viewport_context_menu(self, pos: QPoint) -> None:
        if self.point_item is None:
            return
        menu = QMenu(self)
        if self._selected_instance_ids:
            focus_action = menu.addAction("Focus on Selected (hide everything else)")
            focus_action.triggered.connect(self._focus_on_selected)
        if self._focus_indices is not None:
            show_all_action = menu.addAction("Show All (clear focus)")
            show_all_action.triggered.connect(self._clear_focus)
        if menu.isEmpty():
            return
        menu.exec(self.gl_view.mapToGlobal(pos))

    def _focus_on_selected(self) -> None:
        """Isolates the CURRENTLY selected instance(s) - see
        _on_point_clicked's own context menu - hiding every other point in
        the scan, an exclusive override independent of any layer's own
        points_hidden (see _recompute_composition). Stays in effect until
        _clear_focus, regardless of what gets clicked/selected afterward -
        the same "sticky until explicitly turned off" behavior other 3D
        tools' isolate-selection mode already uses, rather than silently
        reverting the moment the selection itself changes."""
        layer = self.layer_panel.selected_layer
        if layer is None or not self._selected_instance_ids:
            return
        selected_insts = [inst for inst in layer.instances if id(inst) in self._selected_instance_ids]
        if not selected_insts:
            return
        # No np.unique - only ever used as a boolean mask index (see
        # _recompute_composition), which is duplicate-safe - see
        # _union_indices' own docstring for why this matters (np.unique's
        # sort was a real, reported freeze on a large selected instance).
        self._focus_indices = _union_indices([inst.indices for inst in selected_insts])
        self._recompute_composition()
        self.status_label.setText(
            f"Focused on {len(selected_insts)} instance(s) - everything else hidden. "
            "Right-click > Show All to restore."
        )

    def _clear_focus(self) -> None:
        if self._focus_indices is None:
            return
        self._focus_indices = None
        self._recompute_composition()
        self.status_label.setText("Focus cleared.")

    _PICK_FOV_DEG = 90.0  # fixed FOV _run_pick_segmentation's SAM2 view uses - see _start_pick_into_layer

    def _pick_point_index(self, pos: QPoint) -> None:
        point_index = self._gpu_pick_point_index(pos)
        self.gl_view.update()  # the selection pass overwrote the visible frame - redraw normally
        if point_index is None:
            self.status_label.setText("No point under cursor.")
            return
        # Modifiers read at click time, not inside the background SAM2 task -
        # by the time the result comes back the key(s) may already be up.
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._add_multipoint_click(point_index)
            return
        subtract = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if self._multipoint_anchor_index is not None:
            self._finish_multipoint_click(point_index, pos, subtract=subtract)
            return
        self._run_pick_segmentation(point_index, pos, subtract=subtract)

    def _start_multipoint_session(self, point_index: int) -> None:
        """First Ctrl+right-click of a fresh multi-point pick: anchors the
        SAM2 view to this point (same FOV single-point picking already uses)
        and shows its boundary as a wedge outline (see
        _update_multipoint_indicator, reusing the ROI overlay's own
        _wedge_boundary_points) - SAM2 needs every point of one multi-point
        prompt to land on the SAME rendered image, so points outside this
        boundary can't be added to it (see _point_within_multipoint_window)."""
        self.point_item.ensure_point_angles()
        theta0 = float(self.point_item._point_theta[point_index])
        phi0 = float(self.point_item._point_phi[point_index])
        if not (np.isfinite(theta0) and np.isfinite(phi0)):
            self.status_label.setText(
                "That point has no usable scan position - can't anchor a multi-point pick there."
            )
            return
        half = self._PICK_FOV_DEG / 2.0
        self._multipoint_anchor_index = point_index
        self._multipoint_extra_indices = []
        self._multipoint_window = ((theta0 - half, theta0 + half), (phi0 - half, phi0 + half))
        self._update_multipoint_indicator()
        self.status_label.setText(
            "1 point queued for a multi-point pick - Ctrl+right-click more points within "
            "the shown outline, right-click to run, Escape to cancel."
        )

    def _point_within_multipoint_window(self, point_index: int) -> bool:
        if self._multipoint_window is None:
            return False
        theta = float(self.point_item._point_theta[point_index])
        phi = float(self.point_item._point_phi[point_index])
        (t_lo, t_hi), (p_lo, p_hi) = self._multipoint_window
        return t_lo <= theta <= t_hi and p_lo <= phi <= p_hi

    def _add_multipoint_click(self, point_index: int) -> None:
        """Ctrl+right-click: adds one more point to the in-progress multi-
        point pick (starting a new one if none is active yet) instead of
        running SAM2 immediately - see _pick_point_index/_finish_multipoint_click
        for the plain right-click that actually runs it."""
        if self._multipoint_anchor_index is None:
            self._start_multipoint_session(point_index)
            return
        if not self._point_within_multipoint_window(point_index):
            self.status_label.setText(
                "That point falls outside the current pick view (see the outline) - can't "
                "combine it with this multi-point pick. Right-click to run with what you "
                "have, or Escape to start over."
            )
            return
        self._multipoint_extra_indices.append(point_index)
        n = 1 + len(self._multipoint_extra_indices)
        self.status_label.setText(
            f"{n} points queued for one multi-point pick - Ctrl+right-click to add more, "
            "right-click to run, Escape to cancel."
        )

    def _finish_multipoint_click(self, point_index: int, screen_pos: QPoint, subtract: bool = False) -> None:
        """Plain right-click while a multi-point pick is in progress: this
        click's own point joins the prompt as its LAST point, then the
        buffered session is cleared (so a new one can start fresh) and SAM2
        runs once with the anchor plus every queued extra point."""
        if not self._point_within_multipoint_window(point_index):
            self.status_label.setText(
                "That point falls outside the current pick view (see the outline) - can't "
                "combine it with this multi-point pick. Right-click INSIDE the outline to "
                "run, or Escape to start over."
            )
            return
        anchor = self._multipoint_anchor_index
        extras = self._multipoint_extra_indices + [point_index]
        self._clear_multipoint_session()
        self._run_pick_segmentation(anchor, screen_pos, subtract=subtract, extra_indices=extras)

    def _clear_multipoint_session(self) -> None:
        self._multipoint_anchor_index = None
        self._multipoint_extra_indices = []
        self._multipoint_window = None
        self._update_multipoint_indicator()

    def _cancel_multipoint_buffer(self) -> None:
        self._clear_multipoint_session()
        self.status_label.setText("Multi-point pick cancelled.")

    def _update_multipoint_indicator(self) -> None:
        """Draws (or clears) the current multi-point pick's boundary as a
        wedge outline - a SEPARATE GLLinePlotItem from the real ROI's own
        roi_indicator_item (see _update_roi_indicator), so an active ROI
        selection and an in-progress multi-point pick can each show their own
        outline without clobbering the other's."""
        if self._multipoint_indicator_item is not None:
            self.gl_view.removeItem(self._multipoint_indicator_item)
            self._multipoint_indicator_item = None
        if self._multipoint_window is None or self.node is None:
            return
        theta_range, phi_range = self._multipoint_window
        pts = _wedge_boundary_points(self.node, theta_range, phi_range)
        if pts is None:
            return
        self._multipoint_indicator_item = gl_module.GLLinePlotItem(
            pos=pts, color=(0.3, 0.85, 1.0, 1.0), width=2.0, mode="line_strip", antialias=True,
        )
        self.gl_view.addItem(self._multipoint_indicator_item)

    _PICK_RADIUS_PX = 4  # tolerance around the click/hover point - see _gpu_pick_point_index

    @staticmethod
    def _drain_gl_errors(max_iterations: int = 16) -> None:
        """Repeatedly calls glGetError() until clear - a single call only
        returns ONE pending error at a time, so this is the only way to be
        sure none are left queued. Real reported crash: glViewport (in
        _gpu_pick_point_index, called right after this) raised
        GL_INVALID_ENUM - an error glViewport's own all-integer arguments
        can never actually produce, meaning it was a STALE error from
        somewhere ELSE on the shared GL context (most likely the outline-
        overlay effect - see ShaderPointCloudItem._drain_gl_errors, which
        has the same fix on that side), only surfacing here because this
        happened to be the next checked call. Called at the start of both
        GPU-pick entry points (the two call sites this has actually been
        seen crashing from) as a defensive second layer, independent of
        whatever else may or may not have already drained it upstream."""
        for _ in range(max_iterations):
            if gl.glGetError() == gl.GL_NO_ERROR:
                return

    def _gpu_pick_point_index(self, pos: QPoint) -> int | None:
        """GPU-based pick: renders each point's index as its color into the
        current framebuffer, then reads back a small square patch
        (`_PICK_RADIUS_PX` in each direction) centered on `pos` - not just
        the exact pixel - and returns whichever non-background point is
        CLOSEST to the click/hover center. An exact single pixel made
        picking a small or mostly-occluded object (a thin sliver only a few
        pixels wide peeking between other geometry) frustratingly precision-
        dependent - reported live. Clobbers the visible frame - caller must
        trigger a normal repaint afterward. Must run on the main thread
        (touches GL). Shared by Pick Point mode's segmentation click,
        instance-selection's click, AND the hover preview - all three just
        need "which point is under the cursor," they differ only in what
        they do with the answer."""
        self.gl_view.makeCurrent()
        self._drain_gl_errors()
        ratio = self.gl_view.devicePixelRatio()
        w_phys = int(self.gl_view.width() * ratio)
        h_phys = int(self.gl_view.height() * ratio)

        gl.glViewport(0, 0, w_phys, h_phys)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        region = self.gl_view.getViewport()
        self.gl_view.setProjection(region, region)
        self.gl_view.setModelview()
        self.point_item.render_for_selection()

        # GL-space center (y-flipped from screen space - glReadPixels' origin
        # is bottom-left) - everything below stays in this SAME space, so
        # the "nearest to center" distance never needs converting back and
        # forth between the two conventions.
        cx = int(pos.x() * ratio)
        cy = int(h_phys - (pos.y() * ratio))
        r = self._PICK_RADIUS_PX
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w_phys, cx + r + 1), min(h_phys, cy + r + 1)
        w, h = x1 - x0, y1 - y0

        point_index = None
        if w > 0 and h > 0:
            pixels = gl.glReadPixels(x0, y0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            # glReadPixels returns rows bottom-up starting at y0 - row 0 of
            # this reshape IS gl-y y0, matching the gl_y formula below directly.
            arr = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 4).astype(np.int64)
            not_background = ~((arr[..., 0] == 255) & (arr[..., 1] == 255) & (arr[..., 2] == 255))
            idx_grid = arr[..., 0] + (arr[..., 1] << 8) + (arr[..., 2] << 16) + (arr[..., 3] << 24)
            valid = not_background & (idx_grid < self.point_item.count)
            if valid.any():
                rows, cols = np.nonzero(valid)
                gl_x, gl_y = x0 + cols, y0 + rows
                dist2 = (gl_x - cx) ** 2 + (gl_y - cy) ** 2
                best = np.argmin(dist2)
                point_index = int(idx_grid[rows[best], cols[best]])

        self.gl_view.doneCurrent()
        return point_index

    # ── Instance selection (click-to-select a whole detection, not points) ──
    # Always on (no mode toggle) - PickableGLView disambiguates a plain click
    # from a drag itself, purely by how far the mouse moves; see its class
    # docstring.

    def _on_escape(self) -> None:
        """Escape does triple duty, innermost first: clears the current
        instance selection, cancels an in-progress MULTI-POINT buffer (Ctrl+
        right-clicks not yet run - see _cancel_multipoint_buffer) if one is
        active, otherwise cancels the whole pick session (discarding any
        pending picks - see _cancel_pending_picks) - harmless no-op for
        whichever of these wasn't active. A multi-point buffer cancels on its
        own rather than falling through to the whole-session cancel, so
        backing out of one multi-point pick never discards pending picks
        already accepted earlier in the same session."""
        self._clear_instance_selection()
        if self._multipoint_anchor_index is not None:
            self._cancel_multipoint_buffer()
            return
        if self._pick_target_layer is not None:
            self._cancel_pending_picks()

    def _clear_instance_selection(self) -> None:
        self._selected_instance_ids = set()
        self.layer_panel.set_highlighted_instance_ids(set())
        if self.point_item is not None:
            # clear_highlight() re-uploads a VBO, so it needs gl_view's context
            # current - unlike _on_instance_click (which calls this right after
            # its own _gpu_pick_point_index, itself already inside a
            # makeCurrent/doneCurrent pair), this method is ALSO called
            # standalone from Escape and layer_panel.selectionChanged - neither
            # of which have any GL context
            # current at all (real crash: GL_INVALID_OPERATION on glBindBuffer,
            # same root cause as the box-select fix, different call site - see
            # _refresh_instance_selection_highlight for the same fix).
            self.gl_view.makeCurrent()
            # Flush the hover slot FIRST, before touching the primary one -
            # see _refresh_instance_selection_highlight for why (same fix,
            # same reasoning, this is the "clear" counterpart).
            self.point_item.clear_hover_preview()
            self._hovered_instance = None
            self.point_item.clear_highlight()
            self.point_item.set_selection_outline(None, _to_unit_color(SELECTION_COLOR))
            self.gl_view.doneCurrent()
            self.gl_view.update()
        self._update_deselect_badge(None)
        self._update_instance_overlay()  # box colors follow selection too, not just points

    def _on_instance_click(self, pos: QPoint, shift: bool) -> None:
        """Click-to-select at INSTANCE (bounding-box) granularity, not
        point-wise: whichever detected instance owns the point under the
        cursor gets toggled into/out of the selection as a whole. A plain
        click ADDS the instance under the cursor to the selection; Shift+click
        REMOVES it instead. Missing every instance (background/empty click)
        does NOT clear the current selection - only Escape does that
        explicitly; a stray click next to what you're trying to grab
        shouldn't cost you the selection you already built up. This is what's
        actually useful for weeding out false-positive detections - grab the
        box, delete it - which the earlier point-wise box-drag selection (see
        git history) was more machinery than the job needed."""
        if self.node is None or self.point_item is None or self._busy:
            return
        layer = self.layer_panel.selected_layer
        if layer is None:
            self.status_label.setText("Select a layer first (Layers panel) before selecting its instances.")
            return

        point_index = self._gpu_pick_point_index(pos)
        self.gl_view.update()  # the selection pass overwrote the visible frame - redraw normally

        inst = None
        if point_index is not None:
            # Only INCLUDED instances - matches _update_instance_overlay's and
            # the box-select's own filter, so "what you can click" always
            # agrees with "what's currently shown as this layer's boxes."
            inst = next(
                (i for i in layer.instances if i.included and point_index in i.indices), None,
            )

        if inst is None:
            self.status_label.setText(f"No instance under cursor in '{layer.name}'.")
            return

        key = id(inst)
        if shift:
            self._selected_instance_ids.discard(key)
            action = "Deselected"
        else:
            self._selected_instance_ids.add(key)
            action = "Selected"

        self._refresh_instance_selection_highlight(layer)
        self.status_label.setText(
            f"{action} '{inst.label}' ({len(inst.indices):,} pts) - "
            f"{len(self._selected_instance_ids)} instance(s) selected. Shift+click/drag to deselect, "
            "Delete to remove selected, Escape to clear."
        )

    def _gpu_pick_indices_in_rect(self, rect: QRect) -> np.ndarray:
        """GPU-based rectangle pick: same render_for_selection() pass
        _gpu_pick_point_index uses for a single pixel, read back over the
        whole dragged rectangle instead. Must run on the main thread (touches
        GL); caller must trigger a normal repaint afterward (the selection
        pass overwrites the visible frame)."""
        self.gl_view.makeCurrent()
        self._drain_gl_errors()
        ratio = self.gl_view.devicePixelRatio()
        w_phys = int(self.gl_view.width() * ratio)
        h_phys = int(self.gl_view.height() * ratio)

        gl.glViewport(0, 0, w_phys, h_phys)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        region = self.gl_view.getViewport()
        self.gl_view.setProjection(region, region)
        self.gl_view.setModelview()
        self.point_item.render_for_selection()

        x0 = max(0, int(rect.left() * ratio))
        w = min(w_phys - x0, int(rect.width() * ratio))
        # glReadPixels' origin is bottom-left; rect.top() is measured from
        # the screen TOP (matches _gpu_pick_point_index's own y-flip convention).
        y0 = max(0, h_phys - int(rect.bottom() * ratio))
        h = min(h_phys - y0, int(rect.height() * ratio))

        decoded = np.array([], dtype=np.int64)
        if w > 0 and h > 0:
            pixels = gl.glReadPixels(x0, y0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            arr = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 4).astype(np.int64)
            not_background = ~((arr[:, 0] == 255) & (arr[:, 1] == 255) & (arr[:, 2] == 255))
            idx = arr[:, 0] + (arr[:, 1] << 8) + (arr[:, 2] << 16) + (arr[:, 3] << 24)
            decoded = np.unique(idx[not_background & (idx < self.point_item.count)])

        self.gl_view.doneCurrent()
        return decoded

    def _on_instance_box_select(self, rect: QRect, shift: bool) -> None:
        """Windows-style rubber-band selection at INSTANCE granularity:
        every detected instance touched ANYWHERE by the dragged box is added
        to (or, with Shift, removed from) the selection - not point-wise,
        same bounding-box-level convention as a plain click. Missing every
        instance leaves the current selection untouched, same as a plain
        click (see _on_instance_click)."""
        if self.node is None or self.point_item is None or self._busy:
            return
        layer = self.layer_panel.selected_layer
        if layer is None:
            self.status_label.setText("Select a layer first (Layers panel) before selecting its instances.")
            return

        decoded = self._gpu_pick_indices_in_rect(rect)
        self.gl_view.update()  # the selection pass overwrote the visible frame - redraw normally

        touched = []
        if len(decoded):
            touched = [
                inst for inst in layer.instances
                if inst.included and np.intersect1d(inst.indices, decoded, assume_unique=False).size > 0
            ]

        if not touched:
            self.status_label.setText(f"No instance touched by that box in '{layer.name}'.")
            return

        if shift:
            for inst in touched:
                self._selected_instance_ids.discard(id(inst))
            action = "Deselected"
        else:
            for inst in touched:
                self._selected_instance_ids.add(id(inst))
            action = "Selected"

        self._refresh_instance_selection_highlight(layer)
        self.status_label.setText(
            f"{action} {len(touched)} instance(s) in the box - "
            f"{len(self._selected_instance_ids)} instance(s) selected total. Delete to remove selected, "
            "Escape to clear."
        )

    def _on_layer_panel_instance_clicked(self, layer, instance) -> None:
        """Clicking an instance sub-row in the Layers panel (see
        LayerPanel.instance_clicked) toggles it into/out of the SAME
        `_selected_instance_ids` a 3D-view click or box-select already
        drives - one selection concept, two entry points. Selects `layer`
        first if some other layer was active, matching how the 3D-view click
        handlers are also always scoped to the currently selected layer
        (`layer_panel.selectionChanged` already clears any old selection
        when that happens - see _build_layers_panel's wiring)."""
        if layer is not self.layer_panel.selected_layer:
            self.layer_panel.select(layer)
        key = id(instance)
        if key in self._selected_instance_ids:
            self._selected_instance_ids.discard(key)
        else:
            self._selected_instance_ids.add(key)
        self._refresh_instance_selection_highlight(layer)

    def _refresh_instance_selection_highlight(self, layer) -> None:
        """Recomputes the SELECTION_COLOR point highlight from
        `_selected_instance_ids`, dropping any id that no longer resolves to
        an instance still in `layer` (e.g. a prior deletion/undo swapped in
        different instance objects) - self-healing rather than accumulating
        stale ids."""
        valid_ids = {id(inst) for inst in layer.instances}
        self._selected_instance_ids &= valid_ids
        selected_insts = [inst for inst in layer.instances if id(inst) in self._selected_instance_ids]
        self.layer_panel.set_highlighted_instance_ids(self._selected_instance_ids)

        # highlight_points/clear_highlight re-upload a VBO - need gl_view's
        # context current. _gpu_pick_point_index (called just before this, in
        # _on_instance_click) already did its own makeCurrent/doneCurrent and
        # released it, so it's NOT current any more here without this - same
        # real crash as the earlier box-select fix (GL_INVALID_OPERATION on
        # glBindBuffer), reported again live because this method is a
        # separate call, not inlined into the same current-context block.
        self.gl_view.makeCurrent()
        # Flush the hover slot BEFORE touching the primary one, and forget
        # what was hovered. Real reported bug: click an instance while it's
        # already showing its hover tint -> highlight_points() below backs up
        # the CURRENT (hover-tinted) colors as its own "original", and later,
        # if the cursor drifts off it, _on_hover_moved's clear_hover_preview()
        # restores ITS OWN (older, pre-selection) backup on the same indices -
        # stomping the just-applied SELECTION_COLOR back to nothing, even
        # though `_selected_instance_ids` (and Delete) still reflect the real,
        # correct selection - only the paint was wrong. Clearing hover first
        # means highlight_points() snapshots the true un-tinted color, and
        # resetting `_hovered_instance` forces the next hover check to
        # re-evaluate (and re-snapshot) against these NEW colors instead of
        # reusing a stale backup.
        self.point_item.clear_hover_preview()
        self._hovered_instance = None
        if not selected_insts:
            self.point_item.clear_highlight()
            self.point_item.set_selection_outline(None, _to_unit_color(SELECTION_COLOR))
        else:
            union = _union_indices([inst.indices for inst in selected_insts])
            self.point_item.highlight_points(union, color=SELECTION_COLOR)
            self.point_item.set_selection_outline(union, _to_unit_color(SELECTION_COLOR))
        self.gl_view.doneCurrent()
        self.gl_view.update()
        self._update_deselect_badge(None)  # _hovered_instance was just reset above too
        self._update_instance_overlay()  # box colors follow selection too, not just points

    def _on_delete_selected_instances(self) -> None:
        """Removes the currently selected WHOLE instance(s) from the selected
        layer, for real - after a confirmation (this app has no general
        Undo, so a destructive action needs its own safety net) and with one
        level of undo (see _undo_last_deletion)."""
        layer = self.layer_panel.selected_layer
        if layer is None or not self._selected_instance_ids:
            return
        to_delete = [inst for inst in layer.instances if id(inst) in self._selected_instance_ids]
        if not to_delete:
            return

        n = len(to_delete)
        # A plain QMessageBox.question() can't style individual parts of its
        # text (the count, the undo caveat) or leave BOTH buttons visually
        # plain - built explicitly instead (design review 2026-09-11): the
        # count reads as the thing to actually double-check before
        # confirming, the undo note is real but secondary (de-emphasized,
        # not competing for attention with the question itself), and
        # neither button gets Qt's own bold "default button" treatment - No
        # was reported as reading too heavy/discouraging for a question
        # that's routine to say yes to. Escape still safely maps to No
        # regardless (Qt's own convention for a Yes/No box, independent of
        # setDefaultButton - not changed here).
        box = QMessageBox(self)
        box.setWindowTitle("Delete selected instance(s)?")
        box.setIcon(QMessageBox.Icon.Question)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            f"Delete <b>{n}</b> selected instance(s) from layer '{layer.name}'?<br><br>"
            "<span style='color:#888888; font-style:italic;'>This can be undone once, "
            "via \"Undo Last Deletion\" - but only until the next deletion overwrites "
            "it.</span>"
        )
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        reply = box.exec()
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._last_deletion = {"layer": layer, "original_instances": list(layer.instances)}
        layer.instances = [inst for inst in layer.instances if id(inst) not in self._selected_instance_ids]

        self._clear_instance_selection()
        self.layer_panel.refresh()  # emits `changed` -> _recompute_composition + _update_instance_overlay
        self.btn_undo_deletion.setEnabled(True)
        self.status_label.setText(f"Deleted {n} instance(s) from '{layer.name}'. Undo Last Deletion to restore.")

    def _undo_last_deletion(self) -> None:
        if self._last_deletion is None:
            return
        layer = self._last_deletion["layer"]
        if layer not in self.layer_panel.layers:
            # The layer itself was deleted (Layers panel's own "X" button)
            # after the instance-deletion but before Undo - restoring its
            # instance data wouldn't make the layer reappear in the panel,
            # so there's nothing sensible to undo into any more.
            self._last_deletion = None
            self.btn_undo_deletion.setEnabled(False)
            self.status_label.setText("Can't undo - that layer was deleted since.")
            return

        layer.instances = self._last_deletion["original_instances"]

        self._last_deletion = None
        self.btn_undo_deletion.setEnabled(False)
        self.layer_panel.refresh()
        self.status_label.setText(f"Undid the last deletion in '{layer.name}'.")

    def _start_pick_into_layer(self, layer) -> None:
        """Arms single-point picking targeted at `layer` - see the class
        docstring for the overall design. No settings dialog any more
        (design review 2026-09-10): arms immediately, using the 3D view's
        OWN currently-displayed color channel and a fixed 90 deg FOV, same
        "standard settings" pcseg's own defaults use. Right-click in the 3D
        view then segments the point under the cursor and shows its
        candidates in a small menu (see _run_pick_segmentation/
        _show_pick_candidate_menu) - choosing one queues it as a PENDING pick
        (see _PendingPick), not committed to `layer` yet, so several objects
        can be picked as one batch before Accept commits them all at once
        (see the Picking... panel: Finalize/Accept/Cancel, wired in
        _build_layers_panel). Escape or re-arming disarms it and discards
        any pending picks (see _cancel_pending_picks)."""
        if self.node is None:
            self.status_label.setText("Load a scan first.")
            return
        self._pick_target_layer = layer
        self._pending_picks = []
        self._update_pending_picks_panel()
        self._refresh_highlight_overlay()  # clears any leftover pending-pick highlight from a prior session
        self.status_label.setText(
            f"Armed: right-click an object in the 3D view to pick into '{layer.name}' "
            "(Shift+right-click subtracts from an over-segmented pending pick instead; "
            "Ctrl+right-click builds a multi-point pick). Escape cancels."
        )

    def _disarm_pick_into_layer(self) -> None:
        self._pick_target_layer = None
        self._pending_picks = []
        self._clear_multipoint_session()
        self._update_pending_picks_panel()
        self._refresh_highlight_overlay()

    def _run_pick_segmentation(self, point_index: int, screen_pos: QPoint, subtract: bool = False,
                                extra_indices: list[int] | None = None) -> None:
        from services.point_segmentation_service import segment_at_point

        node = self.node
        layer = self._pick_target_layer
        # "Whatever channel we're currently on" - the 3D view's OWN active
        # color mode, not a separately chosen one (see _start_pick_into_layer).
        channel = self.point_item.current_color_mode
        sam = self._ensure_sam_service()
        debug_dir = self._pick_debug_dir()

        def task(progress_callback=None):
            # Single-point picks always request all 3 SAM2-ranked candidates -
            # one point alone is often ambiguous (door vs. door panel vs.
            # wall) and the candidate menu below is exactly what lets you
            # pick the right one instead of silently committing to SAM2's own
            # top score. A multi-point pick (extra_indices non-empty) already
            # resolves that ambiguity itself, so segment_at_point requests
            # only its single best mask there instead (see its own docstring).
            return segment_at_point(
                sam, node, point_index, fov=self._PICK_FOV_DEG, channel=channel,
                keep_all_masks=True, extra_indices=extra_indices, debug_dir=debug_dir,
            )

        extra_note = f" (+{len(extra_indices)} more point(s))" if extra_indices else ""
        self._set_busy(True, f"Segmenting point {point_index}{extra_note}...")
        self._start_log_forwarding()  # covers SAM2's first-use loading gap here too

        def on_success(result) -> None:
            self._stop_log_forwarding()
            self._set_busy(False)
            self._show_pick_candidate_menu(layer, result.candidates, screen_pos, subtract=subtract)

        def on_error(message: str) -> None:
            self._stop_log_forwarding()
            self._set_busy(False)
            self.status_label.setText(f"Pick failed: {message}")

        self._task_thread, self._task_worker = run_in_background(
            task, on_success=on_success, on_error=on_error, on_progress=self._on_task_progress,
        )

    def _show_pick_candidate_menu(self, layer, candidates, screen_pos: QPoint, subtract: bool = False) -> None:
        """Shows the pick's (up to 3) SAM2 candidates as a small menu at the
        click position, one row each with its score and point count -
        hovering a row live-previews that candidate's own points with a
        faint colored overlay in the 3D view (see _preview_pick_candidate),
        so you can compare before committing. Choosing one normally queues it
        as a new PENDING pick for `layer` (see _add_pending_pick); with
        `subtract` (Shift was held at click time - see _pick_point_index) it
        instead REMOVES the chosen candidate's points from whichever pending
        picks already contain them (see _subtract_pick_from_pending) - the
        other candidates are discarded either way, never kept as extra
        instances. Dismissing the menu (click away, Escape) does nothing."""
        if not candidates:
            self.status_label.setText("No object found under that point.")
            return

        menu = QMenu(self)
        if subtract:
            hint = menu.addAction("Subtract from pending pick(s):")
            hint.setEnabled(False)
            menu.addSeparator()
        for rank, cand in enumerate(candidates):
            action = menu.addAction(f"Candidate {rank + 1} - score {cand.score:.2f} ({len(cand.indices):,} pts)")
            action.setData(cand)
            action.hovered.connect(lambda cand=cand: self._preview_pick_candidate(cand))

        chosen = menu.exec(self.gl_view.mapToGlobal(screen_pos))
        # Runs AFTER exec() returns, i.e. after whatever hover-preview was
        # showing and after `chosen`'s own trigger, if any - not tied to
        # QMenu's aboutToHide/triggered signal ORDER (unspecified/fragile),
        # guaranteed correct sequencing this way instead. Hover lives in its
        # OWN slot now (see _preview_pick_candidate) - clearing it here never
        # touches the primary selection/pending-pick highlight underneath.
        self._clear_hover_preview()
        if chosen is not None:
            if subtract:
                self._subtract_pick_from_pending(layer, chosen.data())
            else:
                self._add_pending_pick(layer, chosen.data())

    def _preview_pick_candidate(self, cand) -> None:
        """Live preview while hovering one candidate in the pick menu - the
        SAME hover-preview slot _on_hover_moved drives for the general
        "hovering an existing instance in the 3D view" feature (see
        ShaderPointCloudItem.hover_preview_points), just fed a specific
        candidate instead of whatever's under the live cursor. A separate
        overlay from the primary selection/pending-pick highlight, so it
        never disturbs whatever that's currently showing."""
        if self.point_item is None:
            return
        self.gl_view.makeCurrent()
        self.point_item.hover_preview_points(cand.indices, color=cand.color)
        self.gl_view.doneCurrent()
        self.gl_view.update()

    _HOVER_THROTTLE_S = 0.08  # ~12 checks/sec while the mouse moves - see _on_hover_moved

    def _update_deselect_badge(self, hovered_inst) -> None:
        """Shows the small minus badge on the cursor (PickableGLView.
        set_deselect_badge) only when Shift is held AND the point currently
        under the cursor belongs to an instance that's actually selected
        right now - not just "Shift is down somewhere over the view," which
        said nothing about whether a click there would remove anything.
        `hovered_inst` is passed explicitly (rather than always reading
        `self._hovered_instance`) since _on_hover_moved calls this with a
        freshly GPU-picked instance that hasn't been stored as the new
        `_hovered_instance` yet at every call site."""
        shift_held = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        active = shift_held and hovered_inst is not None and id(hovered_inst) in self._selected_instance_ids
        self.gl_view.set_deselect_badge(active)

    def _on_shift_toggled(self, _held: bool) -> None:
        """Shift pressed/released while the mouse sits still over an already-
        hovered instance (see PickableGLView.shift_toggled) - re-evaluate the
        badge against whatever's currently tracked as hovered, since no
        mouse-move event will do it for us here."""
        self._update_deselect_badge(self._hovered_instance)

    def _on_hover_moved(self, pos: QPoint) -> None:
        """Live "if I clicked now, this is what I'd select" preview: as the
        mouse moves over the 3D view with no button held, whichever instance
        of the SELECTED layer is under the cursor gets a light blue tint
        (see ShaderPointCloudItem.hover_preview_points) - same GPU pick
        _gpu_pick_point_index/_on_instance_click already use for the actual
        click, just read-only and continuous. Throttled (_HOVER_THROTTLE_S)
        rather than run on every raw mouse-move event: each check is a real
        GPU render-for-selection + synchronous readback, cheap once but not
        free enough for every pixel of mouse travel. Skipped entirely while
        a pick session is armed (that has its own hover preview - see
        _preview_pick_candidate) or a task is running."""
        if self.point_item is None or self._busy or self._pick_target_layer is not None:
            return
        now = time.monotonic()
        if now - self._hover_throttle_last < self._HOVER_THROTTLE_S:
            return
        self._hover_throttle_last = now

        layer = self.layer_panel.selected_layer
        if layer is None:
            self._clear_hover_preview()
            return

        point_index = self._gpu_pick_point_index(pos)
        self.gl_view.update()  # the selection pass overwrote the visible frame - redraw normally

        inst = None
        if point_index is not None:
            inst = next((i for i in layer.instances if i.included and point_index in i.indices), None)

        self._update_deselect_badge(inst)

        if inst is self._hovered_instance:
            return  # already showing this (both None counts as "already showing nothing")

        if inst is None:
            self._clear_hover_preview()
            return
        self._hovered_instance = inst
        self.gl_view.makeCurrent()
        self.point_item.hover_preview_points(inst.indices)
        self.point_item.set_hover_outline(inst.indices, _to_unit_color(HOVER_COLOR))
        self.gl_view.doneCurrent()
        self.gl_view.update()
        self._update_instance_overlay()  # the box around it follows the same tint

    def _clear_hover_preview(self) -> None:
        was_hovering = self._hovered_instance is not None
        self._hovered_instance = None
        self._update_deselect_badge(None)
        if self.point_item is not None:
            self.gl_view.makeCurrent()
            self.point_item.clear_hover_preview()
            self.point_item.set_hover_outline(None, _to_unit_color(HOVER_COLOR))
            self.gl_view.doneCurrent()
            self.gl_view.update()
        if was_hovering:
            self._update_instance_overlay()

    def _refresh_highlight_overlay(self) -> None:
        """Single source of truth for what the PRIMARY highlight slot shows
        (SELECTION_COLOR instance selection, or orange pending picks) - call
        after any change to `_pending_picks` or `_selected_instance_ids`.
        Priority: pending picks (while the current pick session has any) >
        instance selection > nothing - a pick session is a distinct enough
        interaction that its own in-progress state should win over a stale
        instance selection underneath it. Independent of the separate HOVER
        slot (see hover_preview_points/_clear_hover_preview) - hover always
        layers on top of whatever this leaves showing."""
        if self.point_item is None:
            return
        if self._pending_picks:
            union = _union_indices([p.indices for p in self._pending_picks])
            self.gl_view.makeCurrent()
            # Flush hover first - see _refresh_instance_selection_highlight
            # for why a stale hover backup can otherwise stomp this.
            self.point_item.clear_hover_preview()
            self._hovered_instance = None
            # Orange: distinct from SELECTION_COLOR (instance selection) and
            # from each candidate's own hover-preview color, so "this is a
            # pending, not-yet-real pick" reads as its own consistent state.
            self.point_item.highlight_points(union, color=(255, 165, 0), blend=0.6)
            self.gl_view.doneCurrent()
            self.gl_view.update()
            return
        layer = self.layer_panel.selected_layer
        if layer is not None and self._selected_instance_ids:
            self._refresh_instance_selection_highlight(layer)
        else:
            self._clear_instance_selection()

    def _pending_pick_label_and_color(self, layer) -> tuple[str, tuple]:
        """The label/color EVERY pending pick in one session shares - the
        target layer's own most common existing label (or its name, for a
        brand new empty layer), and its first instance's color (or a freshly
        generated one if it's still empty) - a manually added object of the
        same class should look and read the same as the rest of the layer,
        with no extra typing needed."""
        if layer.instances:
            label = Counter(inst.label for inst in layer.instances).most_common(1)[0][0]
            color = layer.instances[0].color
        else:
            label = layer.name
            from core.colors import generate_label_color
            color = generate_label_color([label])[0]
        return label, color

    def _add_pending_pick(self, layer, cand) -> None:
        """Queues one chosen PickCandidate as a _PendingPick for `layer` -
        NOT added for real yet (see the class docstring and the Picking...
        panel's Finalize/Accept/Cancel)."""
        label, color = self._pending_pick_label_and_color(layer)
        self._pending_picks.append(_PendingPick(indices=cand.indices, label=label, color=color, score=cand.score))
        self._update_pending_picks_panel()
        self._refresh_highlight_overlay()
        self.status_label.setText(
            f"{len(self._pending_picks)} pending pick(s) for '{layer.name}' - Finalize to preview the "
            "cleaned result, Accept to commit, right-click to pick another."
        )

    def _remove_pending_pick(self, index: int) -> None:
        if 0 <= index < len(self._pending_picks):
            del self._pending_picks[index]
            self._update_pending_picks_panel()
            self._refresh_highlight_overlay()

    def _subtract_pick_from_pending(self, layer, cand) -> None:
        """Shift+right-click pick (see _show_pick_candidate_menu): removes
        `cand`'s points from every pending pick that overlaps them, instead
        of queuing `cand` as its own new pending pick - an eraser for
        over-segmentation using SAM2's own object boundary rather than a
        manual brush (an earlier pick swallowed more than one object; shift-
        click the excess region and subtract whatever SAM2 traces there). A
        pending pick left with zero points after subtraction is dropped
        entirely; one untouched by `cand` is kept exactly as-is."""
        if not self._pending_picks:
            self.status_label.setText(
                "Nothing pending to subtract from yet - shift-pick trims existing pending picks."
            )
            return

        updated: list[_PendingPick] = []
        trimmed = 0
        for pick in self._pending_picks:
            remaining = np.setdiff1d(pick.indices, cand.indices, assume_unique=False)
            if len(remaining) == len(pick.indices):
                updated.append(pick)
                continue
            trimmed += 1
            if len(remaining) > 0:
                updated.append(_PendingPick(indices=remaining, label=pick.label, color=pick.color, score=pick.score))

        self._pending_picks = updated
        self._update_pending_picks_panel()
        self._refresh_highlight_overlay()
        self.status_label.setText(
            f"Subtracted {len(cand.indices):,} pt(s) from {trimmed} pending pick(s) for '{layer.name}'."
        )

    def _update_pending_picks_panel(self) -> None:
        """Rebuilds the Picking... panel (see _build_layers_panel) from
        `_pick_target_layer`/`_pending_picks` - hidden entirely while no pick
        session is armed. Each row gets its own ✕ button (same pattern
        LayerRow's delete button already uses) - the "clickable element to
        drop just this one" the pending-pick list needs, since a pending
        pick has no committed LayerInstance/Instance-details row yet to
        click on."""
        armed = self._pick_target_layer is not None
        self.pending_picks_box.setVisible(armed)
        self.pending_picks_list.clear()
        if not armed:
            return

        self.pending_picks_box.setTitle(f"Picking into '{self._pick_target_layer.name}'")
        for i, pick in enumerate(self._pending_picks):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(2, 0, 2, 0)
            score_txt = f", score {pick.score:.2f}" if pick.score is not None else ""
            row_layout.addWidget(QLabel(f"{pick.label} ({len(pick.indices):,} pts{score_txt})"), stretch=1)
            btn_remove = QPushButton("✕")
            btn_remove.setFixedWidth(22)
            btn_remove.setToolTip("Drop this pending pick - not added to the layer, so nothing to undo.")
            btn_remove.clicked.connect(lambda checked=False, i=i: self._remove_pending_pick(i))
            row_layout.addWidget(btn_remove)

            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.pending_picks_list.addItem(item)
            self.pending_picks_list.setItemWidget(item, row)

        has_pending = bool(self._pending_picks)
        self.btn_pick_finalize.setEnabled(has_pending)
        self.btn_pick_accept.setEnabled(has_pending)

    def _finalize_pending_picks(self) -> None:
        """Cleans each pending pick on its OWN (core.mask_filters.filter_mask_outliers:
        voxel-connectivity + Local Outlier Factor) - one finalized element per
        pending pick, never merged together first. Unlike Run's own detections
        (an automatic sweep, where two overlapping boxes usually ARE the same
        real object split in two, so IoU-merging them back together is
        correct), a manual pick is a person deliberately clicking one object
        at a time; two picks landing close enough to trip a 3D-IoU merge
        would silently collapse them into a single instance even though each
        click was meant to stay its own element. In place, so Finalize can be
        clicked to preview the cleaned-up result before Accept commits it.
        Safe/idempotent to call more than once - Accept always calls this
        itself first regardless of whether you clicked it."""
        if not self._pending_picks or self.node is None:
            return

        from core.mask_filters import filter_mask_outliers

        finalized: list[_PendingPick] = []
        for pick in self._pending_picks:
            clean = filter_mask_outliers(self.node, pick.indices)
            if len(clean) == 0:
                continue
            finalized.append(_PendingPick(indices=clean, label=pick.label, color=pick.color, score=pick.score))

        self._pending_picks = finalized
        self._update_pending_picks_panel()
        self._refresh_highlight_overlay()
        self.status_label.setText(
            f"Finalized: {len(finalized)} pending pick(s) ready for '{self._pick_target_layer.name}'."
        )

    def _accept_pending_picks(self) -> None:
        """Finalizes (if not already - see _finalize_pending_picks, safe to
        call again) and adds every pending pick into the target layer for
        real, each marked `manual=True` so it reads as hand-picked even
        inside an otherwise-automatic layer (see LayerInstance.manual and
        _populate_instance_list's display of it) - then ends the pick
        session entirely (same as Cancel, minus the discarding)."""
        if not self._pending_picks or self._pick_target_layer is None:
            return
        self._finalize_pending_picks()
        layer = self._pick_target_layer
        for pick in self._pending_picks:
            self.layer_panel.add_instance(layer, LayerInstance(
                indices=pick.indices, label=pick.label, color=pick.color, score=pick.score, manual=True,
            ))
        n = len(self._pending_picks)
        self._pick_target_layer = None
        self._pending_picks = []
        self._update_pending_picks_panel()
        self._refresh_highlight_overlay()
        self.status_label.setText(f"Added {n} object(s) to '{layer.name}'.")

    def _cancel_pending_picks(self) -> None:
        if self._pick_target_layer is None:
            return
        name = self._pick_target_layer.name
        self._disarm_pick_into_layer()
        self.status_label.setText(f"Pick cancelled - discarded pending pick(s) for '{name}'.")


def _nearest_valid_pano_index(mapper, row: int, col: int, max_radius: int = 6) -> int | None:
    """
    mapper.pano_map[row, col] if that EXACT cell has a real point; otherwise
    searches an expanding square neighborhood (up to max_radius cells) for
    the nearest cell that does, picking the true-nearest by (row, col)
    distance rather than just the first one found. An exact-cell-only lookup
    (the original implementation) meant a single missed grid cell dropped
    that boundary sample entirely - reported live as the yellow ROI outline
    sometimes not showing up at all; confirmed live on the real dataset that
    a phi range extending past the scan's real vertical extent (e.g.
    (60, 89), well beyond what this TLS scan's row grid actually captured)
    returned NO boundary points whatsoever, and even (-90, 90) (partially
    beyond it) silently dropped ~36% of the samples, producing a visibly
    gapped, incorrectly-connected outline (line_strip joins whatever survived,
    not what was actually adjacent along the real boundary).
    """
    if 0 <= row < mapper.h and 0 <= col < mapper.w and mapper.pano_map[row, col] >= 0:
        return int(mapper.pano_map[row, col])
    for radius in range(1, max_radius + 1):
        r0, r1 = max(0, row - radius), min(mapper.h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(mapper.w, col + radius + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        block = mapper.pano_map[r0:r1, c0:c1]
        valid_rows, valid_cols = np.where(block >= 0)
        if len(valid_rows) == 0:
            continue
        dr = (valid_rows + r0) - row
        dc = (valid_cols + c0) - col
        nearest = np.argmin(dr * dr + dc * dc)
        return int(block[valid_rows[nearest], valid_cols[nearest]])
    return None


def _wedge_boundary_points(node: ScanNode, theta_range: tuple[float, float],
                            phi_range: tuple[float, float], samples: int = 48) -> np.ndarray | None:
    """
    Traces the region-of-interest wedge's boundary (top/right/bottom/left edges,
    closed back to the start) as a polyline through REAL scanned points, by
    stepping along the boundary in (theta, phi) - the scan's own row/col-grid
    convention, see ShaderPointCloudItem.ensure_point_angles - converting each
    step back to a (row, col) grid cell and looking up whatever real point
    actually sits there via node.index_mapper.pano_map.

    Deliberately NOT a synthetic sphere traced via a lon/lat->XYZ formula: the
    scan's row/col grid has no single coherent center in world XYZ (verified live
    - see ensure_point_angles's docstring), so there is no formula that could
    place a theta/phi coordinate at a correct world position without a real point
    to anchor it. A boundary step with no real point nearby (a gap in the scan
    data at that grid cell) is simply skipped, so the traced outline may be a
    partial/gapped polyline rather than a perfectly closed rectangle - accurate
    is preferred over complete here. Returns None if fewer than 2 real points
    were found along the whole boundary (nothing useful to draw).
    """
    mapper = node.index_mapper
    w = mapper.w
    h = max(mapper.h, mapper.w // 2)
    t_lo, t_hi = theta_range
    p_lo, p_hi = phi_range

    thetas = np.linspace(t_lo, t_hi, samples)
    phis = np.linspace(p_lo, p_hi, samples)
    edge_tp = (
        [(t, p_lo) for t in thetas]
        + [(t_hi, p) for p in phis]
        + [(t, p_hi) for t in thetas[::-1]]
        + [(t_lo, p) for p in phis[::-1]]
    )

    pts = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center
    out = []
    for theta, phi in edge_tp:
        # Wrap horizontally (the 0/360 seam) instead of dropping a sample that
        # lands exactly on it (e.g. theta=180 computes col==w, one past the
        # end) - live-verified this alone doesn't explain a fully-missing
        # line, but it's a real, if minor, source of dropped samples.
        col = int(round((theta / 360.0 + 0.5) * w)) % w
        # Clamp vertically to the scan's REAL row range (mapper.h - the raw,
        # unpadded capture grid), not the padded h used for the phi formula
        # itself - h is bigger than mapper.h specifically because
        # reconstruct_panorama_for_channel pads rows with no real data at
        # all (see core.projection.pad_img) - so a phi request that reaches
        # into pure padding has no exact cell to find in the first place,
        # only a real one nearest the scan's genuine edge.
        row = min(max(int(round((phi / 180.0 + 0.5) * h)), 0), mapper.h - 1)
        idx = _nearest_valid_pano_index(mapper, row, col)
        if idx is not None:
            out.append(pts[idx])

    if len(out) < 2:
        return None
    out.append(out[0])  # close the loop
    return np.array(out, dtype=np.float32)


def _aabb_edge_vertices(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """The 12 edges of an axis-aligned box spanning [mn, mx], as 24 vertices
    (each consecutive pair is one edge) - the layout GLLinePlotItem's
    `mode="lines"` (GL_LINES, disconnected segments) wants, so every
    instance's box can be uploaded as one slice of a single combined
    GLLinePlotItem rather than one GL item per instance."""
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    corners = {
        (0, 0, 0): (x0, y0, z0), (1, 0, 0): (x1, y0, z0),
        (1, 1, 0): (x1, y1, z0), (0, 1, 0): (x0, y1, z0),
        (0, 0, 1): (x0, y0, z1), (1, 0, 1): (x1, y0, z1),
        (1, 1, 1): (x1, y1, z1), (0, 1, 1): (x0, y1, z1),
    }
    edges = [
        ((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 0)),  # bottom face
        ((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)), ((0, 1, 1), (0, 0, 1)),  # top face
        ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
        ((1, 1, 0), (1, 1, 1)), ((0, 1, 0), (0, 1, 1)),  # verticals
    ]
    verts = [corners[c] for edge in edges for c in edge]
    return np.array(verts, dtype=np.float32)


# ── Small widget-construction helpers (used only within this module) ───────

def _collapsible_section(title: str, expanded: bool = False, tooltip: str = "") -> tuple[QWidget, QVBoxLayout]:
    """A disclosure section - a plain arrow + title header (NOT a checkbox)
    that expands/collapses its content on click, content actually hidden
    (not just greyed-out) when collapsed so it reclaims the vertical space.
    Used for "Advanced settings" - deliberately NOT built on a checkable
    QGroupBox any more (an earlier version was, reported live as
    misleading): a checkbox/tick visually reads as an on/off TOGGLE for
    whatever's inside, but collapsing this never disables or resets
    anything - every field in "Advanced settings" keeps whatever it's set
    to and is read exactly the same by Run whether this happens to be
    expanded or collapsed right now, same as the ROI box's OWN checkbox
    genuinely does gate its feature - a plain arrow (same pattern the ROI
    overlay panel's own tab already uses) doesn't carry that false
    implication."""
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(2)

    header = QToolButton()
    header.setText(title)
    header.setCheckable(True)
    header.setChecked(expanded)
    header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
    header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    header.setAutoRaise(True)  # flat until hovered - matches every other plain toolbar button in the app
    if tooltip:
        header.setToolTip(tooltip)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(header)
        header_row.addWidget(_info_icon(tooltip))
        header_row.addStretch(1)
        outer.addLayout(header_row)
    else:
        outer.addWidget(header)

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(4, 0, 0, 0)
    outer.addWidget(content)
    content.setVisible(expanded)

    def _on_toggled(checked: bool) -> None:
        header.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        content.setVisible(checked)

    header.toggled.connect(_on_toggled)
    return container, content_layout


def _build_settings_popup(parent_widget, title: str, rows: list[tuple[str, QWidget]]) -> QDialog:
    """A small popup QDialog holding a QFormLayout of `rows` (label -> an
    already-built row widget, e.g. from _row_with_reset) plus a single Close
    button - the "gear icon next to a checkbox/section" pattern (design
    review 2026-09-14, see Sweep geometry/Multi-scale coverage in
    _build_run_prompt_page): settings fine left at their defaults day-to-day
    don't need to permanently occupy space on the main scrollable page, but
    should still be one click away, not buried in a menu. Built once at
    __init__ time and reused (`.exec()`ed again) rather than rebuilt per
    click - its row widgets (the real spin boxes Run itself reads) need to
    stay alive for the widget's whole lifetime, not just while the dialog is
    open."""
    dialog = QDialog(parent_widget)
    dialog.setWindowTitle(title)
    form = _form(dialog)
    for label, widget in rows:
        form.addRow(label, widget)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.clicked.connect(lambda _btn: dialog.accept())
    form.addRow(buttons)
    return dialog


def _form(parent):
    from PyQt6.QtWidgets import QFormLayout
    layout = QFormLayout(parent) if parent is not None else QFormLayout()
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return layout


def _info_icon_image() -> QIcon:
    """The info button's icon: a white "i" on a blue disc, painted here
    rather than using the circled-i character (U+24D8) as the button's text -
    that glyph is missing from many Linux font sets, where the button then
    rendered completely blank. Drawn at 2x for high-DPI screens. Deliberately
    NOT cached in a module-level global: a QIcon/QPixmap that outlives the
    QApplication is destroyed after Qt itself at interpreter exit, which
    crashes the process (seen as a segfault on quitting the viewer)."""
    size, ratio = 16, 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#6fa8dc"))
    painter.drawEllipse(QRectF(1, 1, size - 2, size - 2))
    font = QFont()
    font.setBold(True)
    font.setPixelSize(11)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "i")
    painter.end()
    return QIcon(pixmap)


def _info_icon(tooltip: str) -> QToolButton:
    """A small, deliberately visible "there's an explanation here" affordance
    (design review 2026-09-16, user request: plain setToolTip() on a field
    works but is invisible until you happen to hover exactly that widget -
    nothing on screen signals it's documented at all). Hover shows `tooltip`
    like any normal tooltip; a click ALSO force-shows the same text right at
    the cursor via QToolTip.showText - same content, just for anyone who'd
    rather click than hover-and-wait, or who's using a touch/no-hover input.
    Purely informational - never gates or changes anything itself."""
    btn = QToolButton()
    btn.setIcon(_info_icon_image())
    btn.setIconSize(QSize(16, 16))
    btn.setAutoRaise(True)
    btn.setFixedSize(18, 18)
    btn.setCursor(Qt.CursorShape.WhatsThisCursor)
    # Qt only word-wraps RICH-text tooltips - a long plain-text one is shown
    # as a single line as wide as the text, so wrap it in <qt> (escaped).
    rich = "<qt>" + html.escape(tooltip).replace("\n", "<br>") + "</qt>"
    btn.setToolTip(rich)
    btn.setStyleSheet("QToolButton { border: none; }")
    btn.clicked.connect(lambda: QToolTip.showText(QCursor.pos(), rich, btn))
    return btn


def _section_hint_label(text: str = "What is this section?") -> QLabel:
    """The small gray caption next to a section-level _info_icon (order_box/
    channels_box/thresh_box) - just names what the icon is for, since a lone
    icon with no caption at all reads as orphaned/unclear what it belongs
    to."""
    label = QLabel(text)
    label.setStyleSheet("color: #888; font-size: 11px;")
    return label


def _row_with_reset(widget, default) -> QWidget:
    """Wraps `widget` (a QDoubleSpinBox, QComboBox, or QCheckBox) with a
    small reset-to-default icon button right after it - shown only once the
    value has actually changed from `default`. Replaces the single "Reset
    to Defaults" button that used to sit at the bottom of the Run prompt
    page (design review 2026-09-11: reported as unnecessary now that each
    field can reset itself - most edits only touch one or two settings, not
    the whole page, so a per-field affordance that only shows up exactly
    where it's relevant is more useful than one big button that always
    resets everything at once).

    The button itself stays in the layout ALWAYS (never setVisible(False))
    - only its icon/enabled state toggles, leaving it blank-and-inert when
    not needed. An earlier version DID setVisible(False) it, which pulls
    the button out of the row's layout flow entirely - reported live as
    the spin box (and its up/down arrows specifically) shifting width every
    time the icon appeared/disappeared, moving the arrows out from under
    the cursor mid-click if you were repeatedly nudging a value right at
    its default. Reserving the space permanently fixes the row's width for
    good, so nothing next to it ever moves.

    Also adds a visible _info_icon right before the reset button, reusing
    `widget`'s OWN setToolTip() text - every call site already sets a real
    one on the field before wrapping it here, so this needs no separate
    text to maintain; a widget with no tooltip set (toolTip() == "") simply
    gets no icon rather than a blank, useless one."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(widget, stretch=1)

    tooltip = widget.toolTip()
    if tooltip:
        row.addWidget(_info_icon(tooltip))

    btn = QToolButton()
    btn.setAutoRaise(True)
    btn.setFixedSize(20, 20)
    btn.setToolTip(f"Reset to default ({default})")
    btn.setEnabled(False)  # blank + inert until the value actually differs from default
    row.addWidget(btn)

    def get_value():
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentIndex()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        raise TypeError(f"_row_with_reset: unsupported widget type {type(widget)}")

    def set_value(value) -> None:
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(value)
        elif isinstance(widget, QComboBox):
            widget.setCurrentIndex(value)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(value)

    def refresh(*_args) -> None:
        modified = get_value() != default
        btn.setText("↺" if modified else "")
        btn.setEnabled(modified)

    if isinstance(widget, QDoubleSpinBox):
        widget.valueChanged.connect(refresh)
    elif isinstance(widget, QComboBox):
        widget.currentIndexChanged.connect(refresh)
    elif isinstance(widget, QCheckBox):
        widget.toggled.connect(refresh)
    else:
        raise TypeError(f"_row_with_reset: unsupported widget type {type(widget)}")

    btn.clicked.connect(lambda: set_value(default))
    return container


def _int_spin(lo: int, hi: int, default: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(default)
    return box


def _double_spin(lo: float, hi: float, step: float, default: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setDecimals(2)
    box.setValue(default)
    return box


def _channel_combo() -> QComboBox:
    """
    Backed by a QStandardItemModel storing each choice's real value ("auto",
    "rgb", ...) as Qt.ItemDataRole.UserRole item data, kept separate from the
    DISPLAYED text - lets "auto"'s displayed text be rewritten to show what it
    currently resolves to (e.g. "auto (rgb)", see
    Viewer3DWidget._update_auto_channel_labels) without that display change
    breaking _channel_arg's reading of the actual selected value.
    """
    box = QComboBox()
    model = QStandardItemModel(box)
    for choice in CHANNEL_CHOICES:
        item = QStandardItem(choice)
        item.setData(choice, Qt.ItemDataRole.UserRole)
        model.appendRow(item)
    box.setModel(model)
    return box
