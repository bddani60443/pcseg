"""
GPU point cloud renderer (VBO + GLSL), adapted from the original interactive
application's "Classic" renderer. Kept: per-point color (RGB/intensity/normals),
the render-mode ghosting effect (Solid/Transparent/Clipped), and (for Pick
mode) GPU point picking + temporary highlight-on-pick. Still dropped, since no
interactive pipeline other than single-point pick attaches to this viewer:
live SAM2-mask overlay blending and highlight-on-hover.
"""
from __future__ import annotations

import ctypes
import logging
from typing import Literal

import numpy as np
import OpenGL.GL as gl
from OpenGL.GL import shaders
from OpenGL.raw.GL.VERSION.GL_2_0 import (
    glVertexAttribPointer as raw_glVertexAttribPointer,
)
from core.scan_node import ScanNode
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

from viewer import density_decimation

VERTEX_SHADER = """
#version 130

in vec3 position;
in vec3 color;
in vec3 normal;

uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_scale;
uniform vec3 camPos;
uniform float basePointSize;

out vec4 v_color;
out float v_is_ghost;

void main() {
    vec3 worldPos = (position * u_scale) + u_center;
    vec3 viewDir = normalize(camPos - worldPos);

    // A LAS-loaded point cloud has no real normals (LAS doesn't carry them) -
    // every normal is exactly (0,0,0) there. Without this branch, diffuse
    // lighting evaluates to 0 for every point and the ambient-only term
    // (0.4) flatly dims the whole cloud to 40% brightness - which looked
    // like broken classification colors rather than a lighting artifact
    // until checked against a real LAS file.
    float lightIntensity;
    float facing;
    if (length(normal) < 0.1) {
        lightIntensity = 1.0;
        facing = 1.0;
    } else {
        float diffuse = max(dot(normal, viewDir), 0.0);
        lightIntensity = 0.4 + diffuse;
        facing = dot(normal, viewDir);
    }

    // Distance-based point size: closer points appear larger, far points smaller.
    float dist = length(camPos - worldPos);
    float refDist = 10.0;
    float distScale = clamp(refDist / max(dist, 0.1), 0.3, 3.0);
    float scaledSize = basePointSize * distScale;

    if (facing > 0.0 || length(normal) < 0.1) {
        gl_PointSize = scaledSize;
        v_is_ghost = 0.0;
    } else {
        gl_PointSize = max(scaledSize * 0.3, 1.0);
        v_is_ghost = 1.0;
    }

    gl_Position = u_mvp * vec4(worldPos, 1.0);
    v_color = vec4(color * lightIntensity, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 130
in vec4 v_color;
in float v_is_ghost;

uniform int u_render_mode;  // 0=Solid, 1=Transparent (stipple), 2=Clipped

out vec4 fragColor;

void main() {
    // Bug fixed here: the grey-tint blend used to apply to every ghost
    // (back-facing) point UNCONDITIONALLY, before even checking
    // u_render_mode - so Solid (0) rendered ghost points tinted/dimmed
    // exactly like Transparent (1) did, with only the extra checkerboard
    // discard actually differing between them. Reported live as "Render
    // Mode doesn't seem to do anything" / "always looks transparent" -
    // correct, since Solid's ghost points were never actually rendered
    // normally. Solid must render ghost points completely normally (no
    // tint, no discard) - it's the "no special back-face treatment" mode,
    // not "Transparent but without the stipple".
    if (v_is_ghost > 0.5) {
        if (u_render_mode == 0) {
            fragColor = v_color;
        } else if (u_render_mode == 2) {
            discard;
        } else {
            if (mod(gl_FragCoord.x + gl_FragCoord.y, 2.0) == 0.0) discard;
            fragColor = vec4(mix(v_color.rgb, vec3(0.5), 0.5), 1.0);
        }
    } else {
        fragColor = v_color;
    }
}
"""

SELECTION_VERTEX_SHADER = """
#version 130

in vec3 position;

uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_scale;
uniform float basePointSize;

flat out int v_index;

void main() {
    vec3 worldPos = (position * u_scale) + u_center;
    gl_Position = u_mvp * vec4(worldPos, 1.0);
    gl_PointSize = basePointSize + 4.0;  // slightly larger hit target than the visible point
    v_index = gl_VertexID;
}
"""

SELECTION_FRAGMENT_SHADER = """
#version 130

flat in int v_index;
out vec4 fragColor;

void main() {
    // Encode the point index into RGBA so glReadPixels can recover it exactly.
    float r = float((v_index >> 0) & 0xFF) / 255.0;
    float g = float((v_index >> 8) & 0xFF) / 255.0;
    float b = float((v_index >> 16) & 0xFF) / 255.0;
    float a = float((v_index >> 24) & 0xFF) / 255.0;
    fragColor = vec4(r, g, b, a);
}
"""

OUTLINE_MASK_VERTEX_SHADER = """
#version 130

in vec3 position;

uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_scale;
uniform float u_point_size;

void main() {
    vec3 worldPos = (position * u_scale) + u_center;
    gl_Position = u_mvp * vec4(worldPos, 1.0);
    // Deliberately much bigger than the real on-screen point size (see
    // ShaderPointCloudItem._OUTLINE_MASK_POINT_SIZE) - a point cloud isn't
    // watertight, so rendering at the true size would leave a dotted,
    // holey mask. Oversized points overlapping their neighbors on screen
    // IS the "fill the gaps into one solid silhouette" step, done for free
    // by point rasterization instead of a separate dilate/blur pass.
    gl_PointSize = u_point_size;
}
"""

OUTLINE_MASK_FRAGMENT_SHADER = """
#version 130
out vec4 fragColor;

void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

OUTLINE_COMPOSITE_VERTEX_SHADER = """
#version 130

in vec2 position;  // already NDC, a [-1,1] full-viewport quad - see _outline_quad_vbo
out vec2 v_uv;

void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

OUTLINE_COMPOSITE_FRAGMENT_SHADER = """
#version 130

in vec2 v_uv;
uniform sampler2D u_mask;
uniform vec2 u_texel;       // 1.0 / (viewport width, height) in pixels
uniform vec3 u_color;
uniform float u_thickness;  // outline thickness, in pixels

out vec4 fragColor;

void main() {
    // Inside the (oversized, near-solid) silhouette - not an edge, nothing
    // to draw here; the outline only traces the BOUNDARY.
    if (texture(u_mask, v_uv).a > 0.5) {
        discard;
    }
    // Outside the silhouette: ring-sample a circle of radius u_thickness
    // around this pixel - if any sample lands inside the mask, this pixel
    // is within u_thickness of the boundary, so it's part of the outline.
    // This is "dilate the mask by u_thickness, then keep only what's new"
    // done in a single pass (no separate dilated-mask texture needed).
    const int N = 12;
    float hit = 0.0;
    for (int i = 0; i < N; i++) {
        float ang = 6.28318530718 * float(i) / float(N);
        vec2 offset = vec2(cos(ang), sin(ang)) * u_thickness * u_texel;
        hit = max(hit, texture(u_mask, v_uv + offset).a);
    }
    if (hit > 0.5) {
        fragColor = vec4(u_color, 1.0);
    } else {
        discard;
    }
}
"""

RENDER_MODES = ["Solid", "Transparent", "Clipped"]
COLOR_MODES: tuple[str, ...] = ("rgb", "intensity", "normals")


class _OutlineGroup:
    """Per-outline-group GL state - one instance for the current selection,
    one for the current hover (see ShaderPointCloudItem._outline_selection/
    _outline_hover). A persistent element buffer holding just that group's
    point indices, re-uploaded only when the indices actually changed
    (set_selection_outline/set_hover_outline mark `dirty`; paint() re-
    uploads lazily right before drawing - same "only touch the GPU buffer
    when something really changed" pattern _upload_ebo()'s own callers
    already use elsewhere in this class)."""

    def __init__(self) -> None:
        self.indices: np.ndarray | None = None
        self.color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.ebo = None
        self.count = 0
        self.dirty = False


class ShaderPointCloudItem(GLGraphicsItem):
    def __init__(self, node: ScanNode, color_mode: Literal["rgb", "intensity", "normals"] = "rgb",
                 point_size: int = 2, render_mode: int = 0, view_decimation: int = 1) -> None:
        super().__init__()
        self.node = node
        self.point_size = point_size
        self.render_mode = render_mode
        self.count = node.point_count
        self.current_color_mode = color_mode
        # Rendering-only subsample of the already-loaded points, for interactive
        # performance - segmentation always runs against the FULL loaded point set
        # (node.point_count / node.point_classes) regardless of this value; it only
        # ever changes what gets uploaded to the EBO below, never node's own data.
        # Density-adaptive, not a uniform stride (see density_decimation.py) - a
        # keep-mask over ALL points, cached and only recomputed when THIS VALUE
        # itself changes (_ensure_decimation_mask), never on a ROI-
        # triggered _upload_ebo() call - live-measured at ~9s on a real 58M-point
        # load, far too slow to redo on anything but a deliberate "View detail"
        # change.
        self.view_decimation = max(1, view_decimation)
        self._decimation_keep_mask: np.ndarray | None = None
        self._decimation_computed_for_step: int | None = None

        # Rendering-only angular crop (region of interest), same theta/phi convention
        # as core.projection.get_face_projection_map - see _ensure_point_angles(). Like
        # view_decimation, this never touches node.point_count/point_classes; it only
        # changes what gets uploaded to the EBO below.
        self.theta_range: tuple[float, float] | None = None
        self.phi_range: tuple[float, float] | None = None
        self._point_theta: np.ndarray | None = None
        self._point_phi: np.ndarray | None = None

        self.shader_program: int | None = None
        self.selection_program: int | None = None
        self.vbo_pos = None
        self.vbo_col = None
        self.vbo_norm = None
        self.ebo = None
        self.visible_count = self.count

        self.active_color_array: np.ndarray | None = None
        self._highlighted_indices: np.ndarray | None = None
        self._highlight_backup_colors: np.ndarray | None = None
        # A SEPARATE slot from the pair above, composited on top of
        # whatever they're currently showing (base colors, an instance
        # selection, or a pending pick) rather than fighting over the same
        # one - see hover_preview_points.
        self._hover_indices: np.ndarray | None = None
        self._hover_backup_colors: np.ndarray | None = None

        # Silhouette-outline post-process (see set_selection_outline/
        # set_hover_outline/_paint_outline_overlay) - draws a solid-color
        # outline traced around the on-screen footprint of a whole instance,
        # not per-point. Two independent groups (their own persistent EBO
        # each) so a hovered-but-not-selected instance and the actual
        # selection can show distinct-colored outlines at once. Everything
        # here is lazily created on first real use (outline_mask_program
        # etc. stay None, and _paint_outline_overlay() is a no-op, for the
        # (usual) case where nothing is hovered or selected at all).
        self.outline_mask_program: int | None = None
        self.outline_composite_program: int | None = None
        self._outline_quad_vbo = None
        self._outline_fbo = None
        self._outline_tex = None
        self._outline_fbo_size = (0, 0)
        self._outline_selection = _OutlineGroup()
        self._outline_hover = _OutlineGroup()
        # User-facing toggle (View menu - see Viewer3DWidget.set_outline_effect_enabled)
        # - gates the WHOLE effect at paint time, not by clearing the groups
        # above, so re-enabling shows the current hover/selection's outline
        # again immediately, without needing a fresh hover/click to
        # recompute it.
        self.outline_enabled: bool = True

    def initializeGL(self) -> None:
        # Idempotency guard: load_node() calls this explicitly (eagerly, so
        # vbo_col etc. are ready before load_node() returns - see its own
        # comment) AND THEN calls gl_view.addItem(self.point_item), whose
        # pyqtgraph internals (GLViewWidget.addItem -> item.initialize())
        # unconditionally call initializeGL() a SECOND time on the very same
        # instance once the view is isValid() - which it usually already is,
        # since a big scan's background load leaves plenty of time for the
        # widget to paint at least once first. That second call used to
        # crash outright (it runs AFTER load_node()'s own doneCurrent(), so
        # there's no current GL context at all - even _drain_gl_errors()'s
        # own glGetError() raised GL_INVALID_OPERATION) and, even on a
        # driver where it happened not to crash, would silently leak the
        # first call's shader programs/VBOs by overwriting their handles.
        # Real work only ever needs to happen once per instance.
        if self.shader_program is not None:
            return

        # Real reported crash: GL_INVALID_OPERATION on glCreateShader here,
        # right after a scan reload (a new ShaderPointCloudItem's very first
        # paint - see load_node) - shaders.compileShader is a CHECKED call
        # (calls glGetError() internally), so it inherited and misreported a
        # STALE error left over from something EARLIER on the shared
        # context, most likely the just-cleaned-up PREVIOUS point_item's own
        # cleanup() (its glDelete* calls weren't drained either - see the
        # matching fix there) - same "a pending error blames whatever
        # CHECKED call happens to run next" class of bug _drain_gl_errors'
        # own docstring already documents for the outline pass below.
        # Draining here too means THIS shader compile always starts from a
        # genuinely clean error state, regardless of what ran before it.
        self._drain_gl_errors()
        try:
            vs = shaders.compileShader(VERTEX_SHADER, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vs, fs, validate=False)

            vs_sel = shaders.compileShader(SELECTION_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
            fs_sel = shaders.compileShader(SELECTION_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
            self.selection_program = shaders.compileProgram(vs_sel, fs_sel, validate=False)
        except Exception as e:
            logging.error(f"Shader compilation error: {e}")
            return

        self.vbo_pos = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.node.points_quantized.nbytes,
                         self.node.points_quantized, gl.GL_STATIC_DRAW)

        self.vbo_norm = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_norm)
        normals = self.node.normals if self.node.normals is not None else np.zeros((self.count, 3), dtype=np.int8)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STATIC_DRAW)

        self.vbo_col = gl.glGenBuffers(1)
        self.upload_color_data()

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.ebo = gl.glGenBuffers(1)
        self._upload_ebo()

        # Separate try/except from the main shaders above: a driver that
        # somehow can't compile these (an old/unusual OpenGL 3.0 driver,
        # say) should only lose the outline effect, not the entire point
        # cloud renderer - outline_mask_program/outline_composite_program
        # simply stay None and _paint_outline_overlay() no-ops forever.
        # ALSO wrapped in _drain_gl_errors() (finally, below) - real reported
        # crash: a compile/link problem here on some real driver left a GL
        # error flag set even though the Python-level exception was caught
        # and logged correctly; that undrained flag then surfaced at the
        # very first _gpu_pick_point_index call (a completely unrelated
        # method, misattributing the error to its own glViewport call) since
        # this runs at the START of the object's life, before any hover/pick
        # has happened yet - see _drain_gl_errors' own docstring for the
        # general mechanism.
        try:
            vs_mask = shaders.compileShader(OUTLINE_MASK_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
            fs_mask = shaders.compileShader(OUTLINE_MASK_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
            self.outline_mask_program = shaders.compileProgram(vs_mask, fs_mask, validate=False)

            vs_comp = shaders.compileShader(OUTLINE_COMPOSITE_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
            fs_comp = shaders.compileShader(OUTLINE_COMPOSITE_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
            self.outline_composite_program = shaders.compileProgram(vs_comp, fs_comp, validate=False)

            # A single [-1,1] NDC quad (GL_TRIANGLE_STRIP order), reused every
            # frame the outline effect draws - built once here, not per-paint.
            quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
            self._outline_quad_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._outline_quad_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        except Exception as e:
            logging.error(f"Outline shader compilation error (outline effect disabled): {e}")
            self.outline_mask_program = None
            self.outline_composite_program = None
        finally:
            self._drain_gl_errors()

    def _upload_ebo(self) -> None:
        """(Re)builds the element buffer from node.hidden_mask, cropped by the
        angular ROI (if any), and subsampled by view_decimation, in that
        order. Must run with a current GL context - initializeGL() calls it
        once at setup, set_view_decimation()/set_angular_roi()/
        set_hidden_mask() call it again live."""
        visible = np.where(~self.node.hidden_mask)[0].astype(np.uint32)
        if self.theta_range is not None or self.phi_range is not None:
            angular = self._angular_mask()
            visible = visible[angular[visible]]
        if self.view_decimation > 1:
            self._ensure_decimation_mask()
            if self._decimation_keep_mask is not None:
                visible = visible[self._decimation_keep_mask[visible]]
        self.visible_count = len(visible)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, visible.nbytes, visible, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _ensure_decimation_mask(self) -> None:
        """Density-adaptive keep-mask over ALL points (see density_decimation.py),
        cached and recomputed ONLY when view_decimation's own value changed since
        the last call - NOT on every _upload_ebo() (ROI and hidden-point
        changes call that far more often than the user actually changes
        "View detail", and this is expensive: ~9s measured on a real
        58M-point load). Computed globally (not restricted to whatever's
        currently ROI-visible) - simpler and much safer against
        exactly that repeated-recompute risk, at the minor cost of the kept
        fraction being exact overall rather than exactly 1/step within an
        arbitrarily small ROI crop specifically."""
        if self._decimation_computed_for_step == self.view_decimation:
            return
        if self.view_decimation <= 1:
            self._decimation_keep_mask = None  # None = "keep everything", cheapest representation
        else:
            points_3d = self.node.points_quantized.astype(np.float32) * self.node.quant_scale + self.node.quant_center
            voxel_ids = density_decimation.pack_voxel_ids(points_3d, density_decimation.DEFAULT_VOXEL_SIZE)
            self._decimation_keep_mask = density_decimation.select_density_adaptive(
                voxel_ids, 1.0 / self.view_decimation,
            )
        self._decimation_computed_for_step = self.view_decimation

    def set_view_decimation(self, step: int) -> None:
        """Changes the rendering-only subsample live, without touching node.point_count,
        node.point_classes, or anything segmentation-related - purely how much of the
        already-loaded/already-classified point set gets drawn. Needs a current GL
        context; the caller (Viewer3DWidget) is responsible for makeCurrent/doneCurrent
        around this, same as its other direct GL entry points."""
        step = max(1, step)
        if step == self.view_decimation or self.ebo is None:
            self.view_decimation = step
            return
        self.view_decimation = step
        self._upload_ebo()
        self.update()

    def ensure_point_angles(self) -> None:
        """
        Lazily computes each point's azimuth (theta) and elevation (phi), degrees,
        from its position in the scan's OWN captured row/col grid
        (node.index_mapper.point_to_pano) - the exact inverse of
        core.projection.get_face_projection_map's uf/vf formulas (uf = (lon/2pi +
        0.5)*w -> theta = 360*(col/w - 0.5); vf = (0.5 - lat/pi)*h combined with
        phi = -degrees(lat) -> phi = 180*(row/h - 0.5), same convention
        phi_extent_of_valid_data already uses).

        Deliberately NOT atan2(x, z) on real world XYZ (points_quantized*quant_scale
        + quant_center), despite that seeming like the obvious "azimuth relative to
        the scan origin" - verified live on the real dataset that the two disagree by
        a wildly non-constant amount (std ~60 degrees across sampled points, not a
        single fixed rotation), consistent with a registered/multi-station scan where
        the row/col capture grid has no single coherent center in world XYZ at all.
        The sweep restriction (core.projection.compute_indexed_coverage_grid's
        theta_range/phi_range) and the back-projection that turns a mask into point
        indices both operate entirely within this SAME row/col grid, never touching
        world XYZ - so matching it here is what makes an angular view crop actually
        correspond to what a theta/phi-restricted sweep will process, instead of an
        unrelated-looking crop that happens to also be called "theta/phi".

        A point with no valid grid cell (mapper.point_to_pano == -1, e.g. it fell
        outside the structured capture grid) gets NaN for both - a NaN account never
        satisfies a `>=`/`<=` bound, so such a point is excluded whenever ANY angular
        restriction is active, which is the safe default (never wrongly shown/kept).
        """
        if self._point_theta is not None:
            return
        mapper = self.node.index_mapper
        row = mapper.point_to_pano[:, 0].astype(np.float64)
        col = mapper.point_to_pano[:, 1].astype(np.float64)
        valid = (row >= 0) & (col >= 0)
        w = float(mapper.w)
        # reconstruct_panorama_for_channel pads to width/2 if shorter - see
        # core.projection.pad_img - and point_to_pano's rows are unaffected by that
        # bottom-only padding, so this matches the padded panorama's own row->phi scale.
        h = float(max(mapper.h, mapper.w // 2))
        theta = 360.0 * (col / w - 0.5)
        phi = 180.0 * (row / h - 0.5)
        theta[~valid] = np.nan
        phi[~valid] = np.nan
        self._point_theta = theta.astype(np.float32)
        self._point_phi = phi.astype(np.float32)

    def _angular_mask(self, theta_range: tuple[float, float] | None = None,
                       phi_range: tuple[float, float] | None = None) -> np.ndarray:
        """Boolean mask (len self.count), True where the point's theta/phi
        (see ensure_point_angles) falls within the given ranges - defaults to
        self.theta_range/self.phi_range (the visual-crop's own state) when not
        given explicitly, so this stays a drop-in no-arg call at its original
        _upload_ebo() call site. Also reused (with explicit ranges) by
        Viewer3DWidget to HARD-CLIP a finished sweep's detected instances to
        the exact ROI box - see run_prompt_segmentation/run_auto_segmentation
        - so the visual crop and the actual segmentation-result boundary are
        always the same computation, never two independently-drifting ones."""
        if theta_range is None:
            theta_range = self.theta_range
        if phi_range is None:
            phi_range = self.phi_range
        self.ensure_point_angles()
        mask = np.ones(self.count, dtype=bool)
        if theta_range is not None:
            lo, hi = theta_range
            mask &= (self._point_theta >= lo) & (self._point_theta <= hi)
        if phi_range is not None:
            lo, hi = phi_range
            mask &= (self._point_phi >= lo) & (self._point_phi <= hi)
        return mask

    def set_angular_roi(self, theta_range: tuple[float, float] | None,
                         phi_range: tuple[float, float] | None) -> None:
        """Crops what's drawn to the given angular wedge (None on either axis = no
        crop on that axis) - purely a rendering filter, same non-effect on node's
        own data as set_view_decimation. Needs a current GL context; caller handles
        makeCurrent/doneCurrent."""
        if theta_range == self.theta_range and phi_range == self.phi_range:
            return
        self.theta_range = theta_range
        self.phi_range = phi_range
        if self.ebo is None:
            return
        self._upload_ebo()
        self.update()

    def set_hidden_mask(self, mask: np.ndarray) -> None:
        """Replaces node.hidden_mask (a boolean array, True = never rendered
        at all, regardless of view_decimation/ROI - see
        _upload_ebo's own read of it) and re-uploads the EBO to match. Used
        for "hide these classified points from the 3D view entirely," not
        just reverting their coloring - see Layer.points_hidden's own
        docstring and Viewer3DWidget._recompute_composition, which computes
        the union of every points_hidden layer's own points and calls this.
        Same convention as set_view_decimation/set_angular_roi: needs a
        current GL context, caller handles makeCurrent/doneCurrent."""
        if np.array_equal(mask, self.node.hidden_mask):
            return
        self.node.hidden_mask = mask
        if self.ebo is None:
            return
        self._upload_ebo()
        self.update()

    def _compute_base_color_data(self) -> np.ndarray:
        """The plain per-point color for the current color mode, with no segmentation overlay."""
        data = None
        if self.current_color_mode == "rgb" and self.node.colors_rgb is not None:
            data = self.node.colors_rgb
        elif self.current_color_mode == "intensity" and self.node.colors_intensity is not None:
            i = self.node.colors_intensity
            data = np.column_stack((i, i, i))
        elif self.current_color_mode == "normals" and self.node.normals is not None:
            data = (self.node.normals.astype(np.float32) + 128).clip(0, 255).astype(np.uint8)

        if data is None:
            data = np.full((self.count, 3), 200, dtype=np.uint8)
        return np.ascontiguousarray(data)

    def upload_color_data(self) -> None:
        """Reverts to the plain color mode - clears any segmentation overlay or highlight."""
        if self.vbo_col is None:
            return

        self.active_color_array = self._compute_base_color_data().copy()
        self._highlighted_indices = None
        self._highlight_backup_colors = None
        self._hover_indices = None
        self._hover_backup_colors = None
        self._reupload_active_colors()

    def apply_layer_colors(self, entries: list[tuple[str, np.ndarray, tuple]]) -> None:
        """
        Overlays each detected object's own color on top of the current color
        mode - the GUI counterpart of pcseg's PNG preview export, but on the
        live 3D view. Call upload_color_data() to clear it.

        `entries` is an ORDERED (label, indices, color) sequence, applied in
        exactly the given order (earlier first, later wins any overlap) - the
        same ordering contract as core.classification.compose_layers, so
        this always matches what node.point_classes/an export would say. Unlike
        a {label: indices} dict, this allows the same label to appear more than
        once (two separate layers can legitimately both be named "window").
        """
        if self.vbo_col is None:
            return

        data = self._compute_base_color_data().copy()
        for _label, indices, color in entries:
            valid = indices[(indices >= 0) & (indices < self.count)]
            if len(valid):
                data[valid] = np.array(color[:3], dtype=np.uint8)

        self.active_color_array = data
        self._highlighted_indices = None
        self._highlight_backup_colors = None
        self._hover_indices = None
        self._hover_backup_colors = None
        self._reupload_active_colors()
        self.update()

    def _reupload_active_colors(self) -> None:
        """CALLER'S CONTRACT: gl_view's GL context must already be current
        (via QOpenGLWidget.makeCurrent()) before calling this, or anything
        that calls this - upload_color_data(), apply_layer_colors(),
        highlight_points(), clear_highlight(), set_color_mode(). Real,
        repeatedly-hit crash: called from a Viewer3DWidget signal
        handler/event callback (a Layers-panel checkbox, a color-mode combo,
        Escape, a click) with no GL context current raises
        GLError(GL_INVALID_OPERATION) on glBindBuffer below, because
        `self.vbo_col`'s integer id isn't valid in whatever context (if any)
        happens to be bound at that moment. paintGL() itself is always safe
        (Qt makes the context current before calling it) - it's every OTHER
        caller that must bracket the call in gl_view.makeCurrent() /
        gl_view.doneCurrent() explicitly."""
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_col)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.active_color_array.nbytes, self.active_color_array, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    _HIGHLIGHT_BLEND = 0.9  # fraction of `color` mixed in - see highlight_points

    def highlight_points(self, indices: np.ndarray, color: tuple[int, int, int] = (0, 255, 0),
                          blend: float | None = None) -> None:
        """Temporarily recolors `indices` - blends `color` in at `blend`
        strength (default `_HIGHLIGHT_BLEND`) rather than a flat overwrite,
        so a highlighted point still reads as recognizably ITS OWN color
        (segmentation class, RGB, intensity, whatever the active color mode
        is) tinted, not a featureless solid block that erases every visual
        cue about what's actually selected. This is the PRIMARY highlight
        slot (instance selection, or pending picks - see
        Viewer3DWidget._refresh_highlight_overlay, which picks a `blend`
        per use) - a live HOVER preview is a separate, independent slot (see
        hover_preview_points) that layers on top of this one without
        disturbing it. Call clear_highlight() to restore the exact original
        colors (the backup below is the true pre-highlight color, not the
        blended one)."""
        if self.active_color_array is None:
            return
        self.clear_highlight()

        valid = indices[(indices >= 0) & (indices < self.count)]
        if len(valid) == 0:
            return

        strength = self._HIGHLIGHT_BLEND if blend is None else blend
        self._highlighted_indices = valid
        self._highlight_backup_colors = self.active_color_array[valid].copy()
        original = self._highlight_backup_colors.astype(np.float32)
        tint = np.array(color, dtype=np.float32)
        blended = original * (1.0 - strength) + tint * strength
        self.active_color_array[valid] = blended.clip(0, 255).astype(np.uint8)
        self._reupload_active_colors()
        self.update()

    def clear_highlight(self) -> None:
        if self._highlighted_indices is None:
            return
        self.active_color_array[self._highlighted_indices] = self._highlight_backup_colors
        self._highlighted_indices = None
        self._highlight_backup_colors = None
        self._reupload_active_colors()
        self.update()

    def hover_preview_points(self, indices: np.ndarray, color: tuple[int, int, int] = (100, 170, 255),
                              blend: float = 0.35) -> None:
        """Live "if I clicked now, this is what I'd get" preview while the
        mouse hovers a detected instance (see Viewer3DWidget._on_hover_moved)
        - a light BLUE tint, deliberately its own separate highlight slot
        (`_hover_indices`/`_hover_backup_colors`) rather than reusing
        `highlight_points`'s (a deeper-blue selection / orange pending-pick): hover
        changes on every mouse move and must layer ON TOP of whatever the
        primary highlight is currently showing without disturbing it, not
        replace/fight over the same backup. clear_hover_preview() restores
        to exactly what was showing right before this call, whatever that
        was - the primary highlight, or plain base colors.

        This only stays correct if the primary slot never changes WHILE a
        hover backup is outstanding without going through this slot first -
        a real reported bug (Viewer3DWidget._refresh_instance_selection_highlight)
        was exactly that: select an already-hovered instance, and this
        stale backup (captured before the selection) would later restore
        the PRE-selection color over it once the cursor moved off. Every
        Viewer3DWidget call site that touches the primary slot now clears
        this one first - keep that discipline for any new call site."""
        if self.active_color_array is None:
            return
        self.clear_hover_preview()

        valid = indices[(indices >= 0) & (indices < self.count)]
        if len(valid) == 0:
            return

        self._hover_indices = valid
        self._hover_backup_colors = self.active_color_array[valid].copy()
        original = self._hover_backup_colors.astype(np.float32)
        tint = np.array(color, dtype=np.float32)
        blended = original * (1.0 - blend) + tint * blend
        self.active_color_array[valid] = blended.clip(0, 255).astype(np.uint8)
        self._reupload_active_colors()
        self.update()

    def clear_hover_preview(self) -> None:
        if self._hover_indices is None:
            return
        self.active_color_array[self._hover_indices] = self._hover_backup_colors
        self._hover_indices = None
        self._hover_backup_colors = None
        self._reupload_active_colors()
        self.update()

    def set_color_mode(self, mode: str) -> None:
        if mode == self.current_color_mode:
            return
        self.current_color_mode = mode
        self.upload_color_data()
        self.update()

    # ── Silhouette outline post-process (hover + selection only) ───────────
    # A solid-color line traced around the on-screen SHAPE a group of points
    # forms, not a per-point outline (reported live that per-point outlines
    # "look bad" - noisy, no sense of the object as a whole). Two independent
    # groups so hover and selection can show distinct-colored outlines at
    # the same time; see _OutlineGroup, _paint_outline_overlay, and the
    # OUTLINE_*_SHADER sources at the top of this file for the technique
    # (oversized-point mask render + single-pass ring-sample edge detect,
    # both standard, portable GLSL - no platform-specific extensions).

    def set_selection_outline(self, indices: np.ndarray | None, color: tuple[float, float, float]) -> None:
        """Pure state - no GL calls, so (unlike highlight_points/
        hover_preview_points) this is safe to call with NO current GL
        context: the actual upload happens lazily in paint(), which always
        has one. `indices=None` (or empty) turns this group's outline off."""
        self._outline_selection.indices = indices if indices is not None and len(indices) else None
        self._outline_selection.color = color
        self._outline_selection.dirty = True

    def set_hover_outline(self, indices: np.ndarray | None, color: tuple[float, float, float]) -> None:
        self._outline_hover.indices = indices if indices is not None and len(indices) else None
        self._outline_hover.color = color
        self._outline_hover.dirty = True

    def set_outline_enabled(self, enabled: bool) -> None:
        """User-facing on/off for the whole effect (View menu). Pure state -
        no GL calls, safe without a current context - _paint_outline_overlay
        just checks this flag first. update() so toggling off clears
        whatever's currently drawn immediately, not just from the next
        unrelated repaint."""
        if enabled == self.outline_enabled:
            return
        self.outline_enabled = enabled
        self.update()

    # These two control DIFFERENT things, tuned independently per live
    # feedback: _OUTLINE_MASK_POINT_SIZE is what pushes the outline AWAY
    # from the true point cluster edge (bigger mask points = the silhouette
    # itself balloons outward, before any ring-sampling even happens) -
    # keep this as SMALL as still reliably merges neighboring points into
    # one continuous silhouette (no dotted gaps), since every extra pixel
    # here is pure distance-from-the-real-edge. _OUTLINE_THICKNESS_PX only
    # controls how BOLD the ring itself reads once drawn, starting from
    # wherever the (now much tighter) mask edge already is - free to make
    # this generous without dragging the outline further from the object
    # the way growing the mask point size does. Final values below tuned
    # live (2026-09-11) - mask size taken all the way down to ~1px (barely
    # bigger than a real point, just enough to still merge into a solid
    # silhouette) with thickness pushed up to compensate for boldness.
    _OUTLINE_MASK_POINT_SIZE = 1.0
    _OUTLINE_THICKNESS_PX = 3.5

    def _ensure_outline_fbo(self, width: int, height: int) -> bool:
        """(Re)allocates the mask FBO/texture to match the current viewport
        size - lazy, and only touched again if the viewport actually resized.
        Returns False (caller should skip drawing this frame) if anything
        about FBO setup isn't usable, e.g. an incomplete framebuffer on some
        unusual driver - fails safe (no outline) rather than raising into
        the middle of a paint() call."""
        if width <= 0 or height <= 0:
            return False
        if self._outline_fbo is not None and self._outline_fbo_size == (width, height):
            return True

        if self._outline_tex is None:
            self._outline_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._outline_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self._outline_fbo is None:
            self._outline_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._outline_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._outline_tex, 0)
        complete = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if not complete:
            logging.error("Outline mask FBO incomplete - outline effect disabled for this session.")
            self.outline_mask_program = None  # stop retrying every frame
            return False

        self._outline_fbo_size = (width, height)
        return True

    def _draw_outline_group(self, group: "_OutlineGroup", mvp_data: np.ndarray,
                             center: np.ndarray, scale: float, vp_w: int, vp_h: int, prev_fbo: int) -> None:
        if group.indices is None:
            return
        if group.dirty or group.ebo is None:
            if group.ebo is None:
                group.ebo = gl.glGenBuffers(1)
            valid = group.indices[(group.indices >= 0) & (group.indices < self.count)].astype(np.uint32)
            group.count = len(valid)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, group.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, valid.nbytes, valid, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            group.dirty = False
        if group.count == 0:
            return

        # --- Pass 1: this group's points, oversized, into the mask texture ---
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._outline_fbo)
        gl.glViewport(0, 0, vp_w, vp_h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # No depth test here on purpose - the mask traces this GROUP's own
        # screen footprint (what it would look like with nothing in front
        # of it), not what's currently actually visible of it; occlusion by
        # other geometry is a separate concern the outline isn't meant to
        # reflect (an outline that flickered/broke apart behind a nearer
        # object would read as a rendering bug, not a feature).
        gl.glDisable(gl.GL_DEPTH_TEST)

        gl.glUseProgram(self.outline_mask_program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.outline_mask_program, "u_mvp"), 1, gl.GL_FALSE, mvp_data)
        gl.glUniform3f(gl.glGetUniformLocation(self.outline_mask_program, "u_center"), center[0], center[1], center[2])
        gl.glUniform1f(gl.glGetUniformLocation(self.outline_mask_program, "u_scale"), scale)
        gl.glUniform1f(
            gl.glGetUniformLocation(self.outline_mask_program, "u_point_size"), self._OUTLINE_MASK_POINT_SIZE,
        )

        pos_loc = gl.glGetAttribLocation(self.outline_mask_program, "position")
        if pos_loc < 0:
            # -1 means "not found" - passing that straight to the UNCHECKED
            # raw_glVertexAttribPointer below would wrap to a huge GLuint
            # index instead of raising cleanly, exactly the kind of silent
            # GL error that leaked into unrelated code (see
            # _drain_gl_errors' docstring for the real crash this guards
            # against). Should never actually happen ("position" drives
            # gl_Position directly, so it can't be optimized away) - this is
            # cheap insurance, not an expected path.
            gl.glUseProgram(0)
            return
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        raw_glVertexAttribPointer(pos_loc, 3, gl.GL_SHORT, False, 0, ctypes.c_void_p(0))
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, group.ebo)
        gl.glDrawElements(gl.GL_POINTS, group.count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glDisableVertexAttribArray(pos_loc)
        gl.glUseProgram(0)

        # --- Pass 2: composite the outline back onto the real frame ---
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, prev_fbo)
        gl.glViewport(0, 0, vp_w, vp_h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glUseProgram(self.outline_composite_program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._outline_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self.outline_composite_program, "u_mask"), 0)
        gl.glUniform2f(gl.glGetUniformLocation(self.outline_composite_program, "u_texel"), 1.0 / vp_w, 1.0 / vp_h)
        gl.glUniform3f(gl.glGetUniformLocation(self.outline_composite_program, "u_color"), *group.color)
        gl.glUniform1f(
            gl.glGetUniformLocation(self.outline_composite_program, "u_thickness"), self._OUTLINE_THICKNESS_PX,
        )

        quad_loc = gl.glGetAttribLocation(self.outline_composite_program, "position")
        if quad_loc < 0:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glUseProgram(0)
            gl.glDisable(gl.GL_BLEND)
            return
        gl.glEnableVertexAttribArray(quad_loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._outline_quad_vbo)
        raw_glVertexAttribPointer(quad_loc, 2, gl.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glDisableVertexAttribArray(quad_loc)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_BLEND)

    @staticmethod
    def _drain_gl_errors(max_iterations: int = 16) -> None:
        """Repeatedly calls glGetError() until clear (bounded, just in case) -
        the correct way to fully flush pending GL errors, since a single
        glGetError() call only returns ONE error at a time and more can
        already be queued. Real reported crash: an error from somewhere in
        this outline pass (glVertexAttribPointer is called via the
        UNCHECKED "raw" binding - see paint()'s own use of the same
        raw_glVertexAttribPointer import - which bypasses PyOpenGL's normal
        per-call error check entirely) went undetected here and only
        surfaced later at a completely unrelated call
        (_on_hover_moved -> _gpu_pick_point_index's glViewport, in a
        different method, sometimes even a different PAINT cycle) - a
        pending GL error just sits on the shared context until the next
        CHECKED call anywhere notices it, and blames THAT call instead of
        whatever actually caused it. Draining before AND after this whole
        effect's own passes guarantees its error state can never leak into
        anything else again, regardless of which exact line caused it."""
        for _ in range(max_iterations):
            if gl.glGetError() == gl.GL_NO_ERROR:
                return

    def _paint_outline_overlay(self) -> None:
        """Called at the end of paint() (see below) - draws on top of the
        already-rendered normal scene, using the SAME MVP paint() just used
        (view_widget.currentProjection()/currentModelView() are still valid
        mid-paint-cycle, exactly like paint() itself reads them). Wrapped in
        one try/except AND bracketed by _drain_gl_errors() (see its own
        docstring - fixes a real reported crash where this pass's own error
        state leaked into unrelated code): this is a from-scratch multi-pass
        FBO effect that couldn't be visually verified in a real GL context
        while writing it (this sandbox has no working GPU/QOpenGLWidget) -
        if anything about it is wrong on some real driver, the effect should
        cleanly no-op, never take the rest of the 3D view down with it."""
        if not self.outline_enabled:
            return
        if self.outline_mask_program is None or self.outline_composite_program is None:
            return
        if self._outline_selection.indices is None and self._outline_hover.indices is None:
            return
        view_widget = self.view()
        if view_widget is None:
            return
        self._drain_gl_errors()  # start clean - don't let a stale error from elsewhere confuse this pass
        try:
            # NOT view_widget.getViewport() - that returns LOGICAL pixels
            # (width()/height()), but by the time an item's paint() runs,
            # Qt has already called the real glViewport with the DEVICE
            # (HiDPI-scaled) pixel size (see GLViewWidget.paintGL's own
            # comment: "when called by Qt, glViewport has already been
            # called with device pixel ratio taken of"). Reading the live
            # GL state directly guarantees this matches whatever the normal
            # scene just rendered at, on any DPI scale, without duplicating
            # that scaling math here.
            live_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            vp_w, vp_h = int(live_viewport[2]), int(live_viewport[3])
            if not self._ensure_outline_fbo(vp_w, vp_h):
                return

            prev_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)
            mvp = view_widget.currentProjection() * view_widget.currentModelView()
            mvp_data = np.array(mvp.data(), dtype=np.float32)
            center = self.node.quant_center
            scale = self.node.quant_scale

            # Selection drawn after hover, so if the same instance is both
            # hovered AND selected, the selection's outline color wins (it's
            # the more "committed" state of the two) rather than flickering
            # between them depending on draw order.
            self._draw_outline_group(self._outline_hover, mvp_data, center, scale, vp_w, vp_h, prev_fbo)
            self._draw_outline_group(self._outline_selection, mvp_data, center, scale, vp_w, vp_h, prev_fbo)

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, prev_fbo)
            gl.glViewport(0, 0, vp_w, vp_h)
            gl.glEnable(gl.GL_DEPTH_TEST)
        except Exception as e:
            logging.error(f"Outline overlay render error (skipping this frame): {e}")
        finally:
            self._drain_gl_errors()  # never let this pass's error state escape to anything after it

    def paint(self) -> None:
        if self.shader_program is None:
            self.initializeGL()
        if self.shader_program is None:
            return

        # Start this frame's render clean - a leftover GL error from
        # whatever ran immediately before (another item's own paint() in
        # the same frame, or anything else touching the shared GL context)
        # would otherwise get blamed on whatever this method's own first
        # checked call happens to be. See _drain_gl_errors' own docstring;
        # this is the same fix as the one already applied around the
        # outline overlay, now applied unconditionally to the whole
        # per-frame render path instead of only within the outline's own
        # try/except - a real crash was reported reaching glClearColor
        # (pyqtgraph's OWN call, at the very start of the NEXT frame) with
        # GL_INVALID_ENUM, which glClearColor's all-float arguments can
        # never actually produce - so the leak wasn't being fully contained
        # by the narrower, outline-only drains alone.
        self._drain_gl_errors()

        gl.glUseProgram(self.shader_program)

        view_widget = self.view()

        if view_widget:
            pos = view_widget.cameraPosition()
            gl.glUniform3f(gl.glGetUniformLocation(self.shader_program, "camPos"), pos.x(), pos.y(), pos.z())

            # pyqtgraph doesn't populate the legacy gl_ModelViewProjectionMatrix
            # built-in (no glLoadMatrix calls in this version) - build and pass
            # it explicitly instead.
            mvp = view_widget.currentProjection() * view_widget.currentModelView()
            mvp_data = np.array(mvp.data(), dtype=np.float32)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader_program, "u_mvp"), 1, gl.GL_FALSE, mvp_data)

        c = self.node.quant_center
        gl.glUniform3f(gl.glGetUniformLocation(self.shader_program, "u_center"), c[0], c[1], c[2])
        gl.glUniform1f(gl.glGetUniformLocation(self.shader_program, "u_scale"), self.node.quant_scale)
        gl.glUniform1f(gl.glGetUniformLocation(self.shader_program, "basePointSize"), self.point_size)
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "u_render_mode"), self.render_mode)

        pos_loc = gl.glGetAttribLocation(self.shader_program, "position")
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        raw_glVertexAttribPointer(pos_loc, 3, gl.GL_SHORT, False, 0, ctypes.c_void_p(0))

        col_loc = gl.glGetAttribLocation(self.shader_program, "color")
        gl.glEnableVertexAttribArray(col_loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_col)
        raw_glVertexAttribPointer(col_loc, 3, gl.GL_UNSIGNED_BYTE, True, 0, ctypes.c_void_p(0))

        norm_loc = gl.glGetAttribLocation(self.shader_program, "normal")
        gl.glEnableVertexAttribArray(norm_loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_norm)
        raw_glVertexAttribPointer(norm_loc, 3, gl.GL_BYTE, True, 0, ctypes.c_void_p(0))

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)

        if self.ebo is not None and self.visible_count > 0:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glDrawElements(gl.GL_POINTS, self.visible_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        gl.glDisableVertexAttribArray(pos_loc)
        gl.glDisableVertexAttribArray(col_loc)
        gl.glDisableVertexAttribArray(norm_loc)
        gl.glUseProgram(0)

        self._paint_outline_overlay()  # no-ops instantly unless hover/selection outline is set
        self._drain_gl_errors()  # unconditional - leave this whole frame clean for whatever runs next

    def render_for_selection(self) -> None:
        """
        Draws every point with its index encoded as its RGBA color, into
        whatever framebuffer is currently bound. The caller is responsible for
        setting up that framebuffer/viewport and reading back the pixel under
        the cursor afterward - see Viewer3DWidget.handle_point_click.

        Real reported crash: this uses the same UNCHECKED raw_glVertexAttribPointer
        as paint()'s own vertex setup (see _drain_gl_errors' docstring for why that
        bypasses PyOpenGL's normal per-call error check), but unlike paint() never
        drained afterward - any error it left queued survived all the way past this
        call's own caller (_gpu_pick_point_index skips its next checked call,
        glReadPixels, whenever the pick rectangle clips to zero size at the
        viewport's edge) and surfaced at a COMPLETELY unrelated later call
        (glBindBuffer in _reupload_active_colors, via clear_hover_preview) as a
        misattributed GL_INVALID_ENUM. Draining at the end (below) means this pass
        can never leak its own error state into whatever runs after it, same fix as
        paint()'s own trailing _drain_gl_errors().
        """
        if self.selection_program is None:
            return

        gl.glUseProgram(self.selection_program)

        view_widget = self.view()
        if view_widget:
            mvp = view_widget.currentProjection() * view_widget.currentModelView()
            mvp_data = np.array(mvp.data(), dtype=np.float32)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.selection_program, "u_mvp"), 1, gl.GL_FALSE, mvp_data)

        c = self.node.quant_center
        gl.glUniform3f(gl.glGetUniformLocation(self.selection_program, "u_center"), c[0], c[1], c[2])
        gl.glUniform1f(gl.glGetUniformLocation(self.selection_program, "u_scale"), self.node.quant_scale)
        gl.glUniform1f(gl.glGetUniformLocation(self.selection_program, "basePointSize"), self.point_size)

        pos_loc = gl.glGetAttribLocation(self.selection_program, "position")
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        raw_glVertexAttribPointer(pos_loc, 3, gl.GL_SHORT, False, 0, ctypes.c_void_p(0))

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)

        gl.glDisableVertexAttribArray(pos_loc)
        gl.glUseProgram(0)
        self._drain_gl_errors()  # unconditional - leave this whole pass clean for whatever runs next

    def cleanup(self) -> None:
        buffers = [
            b for b in (
                self.vbo_pos, self.vbo_col, self.vbo_norm, self.ebo, self._outline_quad_vbo,
                self._outline_selection.ebo, self._outline_hover.ebo,
            ) if b is not None
        ]
        if buffers:
            try:
                gl.glDeleteBuffers(len(buffers), buffers)
            except Exception as e:
                logging.error(f"Error cleaning up point cloud GL buffers: {e}")
        for program in (self.shader_program, self.selection_program,
                        self.outline_mask_program, self.outline_composite_program):
            if program is not None:
                try:
                    gl.glDeleteProgram(program)
                except Exception as e:
                    logging.error(f"Error cleaning up point cloud GL program: {e}")
        if self._outline_tex is not None:
            try:
                gl.glDeleteTextures(1, [self._outline_tex])
            except Exception as e:
                logging.error(f"Error cleaning up outline texture: {e}")
        if self._outline_fbo is not None:
            try:
                gl.glDeleteFramebuffers(1, [self._outline_fbo])
            except Exception as e:
                logging.error(f"Error cleaning up outline FBO: {e}")
        self.vbo_pos = self.vbo_col = self.vbo_norm = self.ebo = None
        self.shader_program = self.selection_program = None
        self.outline_mask_program = self.outline_composite_program = None
        self._outline_quad_vbo = self._outline_fbo = self._outline_tex = None
        self._outline_selection = _OutlineGroup()
        self._outline_hover = _OutlineGroup()
        self.node = None
        # Same reasoning as initializeGL()'s own drain - a stray error from
        # any of this method's own glDelete* calls above must not leak onto
        # the shared context and get misattributed to whatever GL call runs
        # next (typically the REPLACEMENT ShaderPointCloudItem's own
        # initializeGL, called from load_node right after this).
        self._drain_gl_errors()
