"""
AutoCAD/Blender/SolidWorks-style corner "ViewCube" navigation aid: a small
6-face cube overlay, top-right corner of the 3D view, that always shows the
SAME rotation as the main camera (built from the exact same
azimuth/elevation this app's camera already uses - see PickableGLView/
pyqtgraph.opengl.GLViewWidget.viewMatrix()/cameraPosition()) and whose faces
can be clicked to snap the main camera to a standard orthogonal view.

No pre-existing rotation-sync mechanism to reuse: the "axis rotates like the
scan" the user referred to (viewer_widget.py's `self.axis_item =
GLAxisItem()`) is just a normal object placed IN the main 3D scene at the
origin - it "rotates" only because it's swept along with everything else by
the same camera, not because of any separate sync code. This module builds
that sync deliberately, once, for a small SEPARATE corner viewport.

Deliberately 6-face only (no edge/corner clicking) - see viewer_widget.py's
ViewCube integration comment for why that's a scoped-out stretch goal, not an
oversight.

All the geometry/hit-testing math here is pure (no GL calls, no Qt widget
dependency beyond QMatrix4x4/QVector4D) so it's directly unit-testable without
a real GL context - see tests exercising face_snap_target/ray_cube_intersect/
screen_point_to_ray directly.
"""
from __future__ import annotations

from math import atan2, degrees

import numpy as np
from PyQt6.QtGui import QMatrix4x4, QVector4D

# (face index, outward unit normal, label, base RGB color 0-1). Order fixed -
# index IS the face identity used everywhere below (ray_cube_intersect,
# hovered_face, face_snap_target).
FACES: list[tuple[int, tuple[float, float, float], str, tuple[float, float, float]]] = [
    (0, (1.0, 0.0, 0.0), "RIGHT", (0.80, 0.30, 0.30)),
    (1, (-1.0, 0.0, 0.0), "LEFT", (0.30, 0.55, 0.80)),
    (2, (0.0, 1.0, 0.0), "BACK", (0.35, 0.70, 0.35)),
    (3, (0.0, -1.0, 0.0), "FRONT", (0.85, 0.70, 0.25)),
    (4, (0.0, 0.0, 1.0), "TOP", (0.88, 0.88, 0.88)),
    (5, (0.0, 0.0, -1.0), "BOTTOM", (0.45, 0.45, 0.45)),
]
FACE_LABELS = {idx: label for idx, _n, label, _c in FACES}


def face_snap_target(face_index: int, current_azimuth: float) -> tuple[float, float]:
    """
    (azimuth, elevation) that puts the main camera looking straight at this
    face - i.e. camera position along the face's own outward normal.

    This is deliberately the exact inverse of pyqtgraph.opengl.GLViewWidget.
    cameraPosition()'s own formula:
        pos = center + dist*(cos(elev)*cos(azim), cos(elev)*sin(azim), sin(elev))
    so "snap to this face" and "where the camera already sits at this
    azimuth/elevation" are the same operation, not two independently invented
    ones that could drift apart.

    Top/Bottom (normal along Z) are a gimbal pole - azimuth doesn't affect
    camera POSITION there, only which way is "up" on screen. Snapping it to
    the nearest 90 degrees (rather than preserving whatever arbitrary value
    was active before the click) was a deliberate later change: preserving it
    verbatim left the view - and the cube - sitting at a non-canonical roll
    after a Top/Bottom click, which real usage reported as looking like the
    cube hadn't actually finished re-orienting. Snapping the OTHER 4 faces
    already produces a clean axis-aligned azimuth (0/90/180/270) by
    construction (see below) - rounding here makes Top/Bottom consistent
    with that instead of the one asymmetric case.
    """
    _idx, normal, _label, _color = FACES[face_index]
    nx, ny, nz = normal
    if abs(nz) > 0.999:
        return round(current_azimuth / 90.0) * 90.0, 90.0 if nz > 0 else -90.0
    return degrees(atan2(ny, nx)), 0.0


def build_cube_view_matrix(azimuth: float, elevation: float, distance: float) -> QMatrix4x4:
    """
    Same rotation as pyqtgraph.opengl.GLViewWidget.viewMatrix() - translate,
    then rotate(elevation-90, 1,0,0), then rotate(azimuth+90, 0,0,-1) - but
    with a FIXED `distance` and no center/pan translation, so the cube's
    apparent size/position never changes with the main view's zoom/pan, only
    its rotation tracks the main camera.
    """
    view = QMatrix4x4()
    view.translate(0.0, 0.0, -distance)
    view.rotate(elevation - 90, 1, 0, 0)
    view.rotate(azimuth + 90, 0, 0, -1)
    return view


def build_cube_projection_matrix(distance: float, fov: float) -> QMatrix4x4:
    proj = QMatrix4x4()
    proj.perspective(fov, 1.0, distance * 0.01, distance * 100.0)
    return proj


def screen_point_to_ray(local_x: float, local_y: float, viewport_size: float,
                         view: QMatrix4x4, proj: QMatrix4x4) -> tuple[np.ndarray, np.ndarray]:
    """
    Unprojects a point in the cube's OWN small square viewport - top-left-origin
    local pixel coordinates, (0,0) to (viewport_size, viewport_size) - into a
    ray (origin, unit direction) in the cube's local/world space, via the
    standard inverse-projection*view technique.
    """
    ndc_x = (2.0 * local_x / viewport_size) - 1.0
    ndc_y = 1.0 - (2.0 * local_y / viewport_size)  # local_y grows down, NDC y grows up

    inv, invertible = (proj * view).inverted()
    if not invertible:
        # Degenerate (shouldn't happen for a well-formed perspective*rotation
        # matrix) - point the ray straight down -Z from far away so it simply
        # misses the cube rather than raising.
        return np.array([0.0, 0.0, 1000.0]), np.array([0.0, 0.0, -1.0])

    near = inv.map(QVector4D(ndc_x, ndc_y, -1.0, 1.0))
    far = inv.map(QVector4D(ndc_x, ndc_y, 1.0, 1.0))
    near_pt = np.array([near.x(), near.y(), near.z()]) / near.w()
    far_pt = np.array([far.x(), far.y(), far.z()]) / far.w()
    direction = far_pt - near_pt
    direction /= np.linalg.norm(direction)
    return near_pt, direction


def ray_cube_intersect(origin: np.ndarray, direction: np.ndarray, half_extent: float = 1.0) -> int | None:
    """
    Standard slab-method ray/axis-aligned-box intersection against the unit
    cube [-half_extent, half_extent]^3 centered at the origin, returning the
    FACES index of the nearest entry face, or None if the ray misses (or the
    cube is behind the ray's origin).
    """
    t_near, t_far = -np.inf, np.inf
    entry_axis, entry_sign = -1, 0

    for axis in range(3):
        o, d = float(origin[axis]), float(direction[axis])
        if abs(d) < 1e-12:
            if o < -half_extent or o > half_extent:
                return None
            continue
        t1 = (-half_extent - o) / d
        t2 = (half_extent - o) / d
        sign1, sign2 = -1, 1
        if t1 > t2:
            t1, t2 = t2, t1
            sign1, sign2 = sign2, sign1
        if t1 > t_near:
            t_near, entry_axis, entry_sign = t1, axis, sign1
        if t2 < t_far:
            t_far = t2
        if t_near > t_far:
            return None

    if entry_axis == -1 or t_near < 0:
        return None

    normal = [0.0, 0.0, 0.0]
    normal[entry_axis] = float(entry_sign)
    return _face_index_for_normal(tuple(normal))


def _face_index_for_normal(normal: tuple[float, float, float]) -> int | None:
    for idx, n, _label, _color in FACES:
        if n == normal:
            return idx
    return None


def _face_quad_vertices(normal: tuple[float, float, float], scale: float = 1.0) -> list[np.ndarray]:
    """The 4 corners (CCW as seen from outside) of a cube face with this
    outward normal, for immediate-mode GL_QUADS rendering. `scale` shrinks the
    cube uniformly about the origin - see ViewCubeOverlay's own docstring for
    why this needs to be less than 1.0 (a cube's projected corner-to-corner
    extent, at a diagonal/corner-on orientation, is larger than its projected
    face-to-face extent - a frustum sized to just fit the face-on case clips
    the corners at other orientations)."""
    n = np.array(normal)
    up_hint = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
    tangent1 = np.cross(n, up_hint)
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(n, tangent1)
    return [scale * (n + su * tangent1 + sv * tangent2) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def corner_rect(widget_w: float, widget_h: float, size: float, margin: float) -> tuple[float, float, float, float]:
    """Top-right corner square (x0, y0, size, size), in the SAME top-left-origin
    coordinate system Qt widget/mouse-event coordinates already use."""
    return widget_w - margin - size, margin, size, size


class ViewCubeOverlay:
    """
    Owns the ViewCube's geometry/state (size, hover) and both its math
    (hit-testing) and its raw-GL drawing. Deliberately has no Qt widget/signal
    of its own - PickableGLView (a thin GLViewWidget subclass) owns the actual
    mouse events and GL context, and calls into this for both.

    `cube_scale` shrinks the cube uniformly (applied to both the drawn
    geometry AND the hit-test's cube bounds, so what's clickable always
    matches what's drawn) relative to the fixed corner viewport/frustum below
    1.0 - real, on-hardware feedback caught something a face-on-only check
    missed: a cube's screen-space extent isn't constant as it rotates. At a
    diagonal/corner-on orientation you're seeing the body-diagonal-to-corner
    distance rather than the face-to-face one, so a frustum sized to just fit
    the face-on case (scale=1.0) clips the corners everywhere else - verified
    numerically by sweeping every (azimuth, elevation) combination and taking
    the worst-case projected NDC extent over all 8 corners: 1.32 unscaled
    (i.e. 32% past the visible edge) at a corner-on orientation, vs. 0.92
    face-on. 0.6 brings the worst case down to ~0.75, comfortably inside the
    visible [-1, 1] NDC range at every orientation, not just the ones a
    developer happens to try by hand.
    """

    def __init__(self, size_px: float = 64.0, margin_px: float = 10.0,
                 distance: float = 4.0, fov: float = 40.0, cube_scale: float = 0.6) -> None:
        self.size_px = size_px
        self.margin_px = margin_px
        self.distance = distance
        self.fov = fov
        self.cube_scale = cube_scale
        self.hovered_face: int | None = None

    def hit_test(self, mouse_x: float, mouse_y: float, widget_w: float, widget_h: float,
                 azimuth: float, elevation: float) -> int | None:
        """mouse_x/mouse_y/widget_w/widget_h are all in the SAME (Qt logical
        pixel, top-left-origin) space as QMouseEvent.position()/self.width()/
        self.height() - deliberately not device-pixel-ratio-scaled, so this
        never needs to know the ratio at all (only paint()'s actual glViewport
        call does, since that's a real framebuffer operation)."""
        x0, y0, size, _ = corner_rect(widget_w, widget_h, self.size_px, self.margin_px)
        if not (x0 <= mouse_x <= x0 + size and y0 <= mouse_y <= y0 + size):
            return None
        view = build_cube_view_matrix(azimuth, elevation, self.distance)
        proj = build_cube_projection_matrix(self.distance, self.fov)
        origin, direction = screen_point_to_ray(mouse_x - x0, mouse_y - y0, size, view, proj)
        return ray_cube_intersect(origin, direction, half_extent=self.cube_scale)

    def paint(self, gl, azimuth: float, elevation: float, widget_w_px: int, widget_h_px: int,
              device_ratio: float) -> None:
        """Draws the cube into its own small corner sub-viewport, on top of
        whatever the main scene already rendered this frame. Caller
        (PickableGLView.paintGL) must call this AFTER the main scene paint,
        with a current GL context; restores the full-widget viewport before
        returning so later GL work (e.g. the next frame) isn't affected."""
        size = int(self.size_px * device_ratio)
        margin = int(self.margin_px * device_ratio)
        x0 = widget_w_px - margin - size
        y0 = widget_h_px - margin - size  # GL viewport Y grows up - "top" = high y
        if size <= 0 or x0 < 0 or y0 < 0:
            return

        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(x0, y0, size, size)
        gl.glViewport(x0, y0, size, size)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)

        view = build_cube_view_matrix(azimuth, elevation, self.distance)
        proj = build_cube_projection_matrix(self.distance, self.fov)
        mvp = np.array((proj * view).data(), dtype=np.float32)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixf(mvp)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        for idx, normal, _label, color in FACES:
            brightness = 1.3 if idx == self.hovered_face else 1.0
            gl.glColor3f(*(min(1.0, c * brightness) for c in color))
            verts = _face_quad_vertices(normal, scale=self.cube_scale)
            gl.glBegin(gl.GL_QUADS)
            for v in verts:
                gl.glVertex3f(float(v[0]), float(v[1]), float(v[2]))
            gl.glEnd()

            # A thin dark outline per face, so adjacent faces (and the hover
            # highlight) read clearly even where two base colors are close.
            gl.glColor3f(0.08, 0.08, 0.08)
            gl.glBegin(gl.GL_LINE_LOOP)
            for v in verts:
                gl.glVertex3f(float(v[0]), float(v[1]), float(v[2]))
            gl.glEnd()

        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, widget_w_px, widget_h_px)
