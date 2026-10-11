from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from pcseg.core.mapping import IndexMapper
from pcseg.core.scan_node import ScanNode

Channel = Literal["rgb", "intensity", "normals"]


@dataclass
class ProjectionContext:
    """Everything needed to map a rendered 2D crop back onto the point cloud."""
    scan_id: str
    pano_shape: tuple
    fov: float
    map_x: np.ndarray
    map_y: np.ndarray
    image_2d: np.ndarray
    index_mapper: IndexMapper
    center_px: int = -1
    center_py: int = -1
    rot_theta_deg: float = 0.0
    rot_phi_deg: float = 0.0


def find_nearest_point_index(node: ScanNode, xyz: tuple[float, float, float]) -> int:
    """
    Returns the point-cloud index whose 3D position is closest to `xyz`. The
    non-interactive counterpart of clicking a point in a viewer: a CLI has no
    mouse, but a caller can still name a real-world coordinate.
    """
    from scipy.spatial import cKDTree

    points_3d = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center
    tree = cKDTree(points_3d)
    _, idx = tree.query(np.asarray(xyz, dtype=np.float32))
    return int(idx)


def default_channel(node: ScanNode) -> Channel:
    """Picks the best available channel: RGB > intensity > normals (normals always exist)."""
    if node.colors_rgb is not None:
        return "rgb"
    if node.colors_intensity is not None:
        return "intensity"
    return "normals"


def _view_rotation(theta: float, phi: float) -> np.ndarray:
    """Rotation from a perspective view's local frame (x right, y up, z
    forward) to the panorama's frame, for a view centred at yaw `theta` and
    pitch `phi` (degrees; phi > 0 looks down). Shared by the forward map
    (get_face_projection_map), its inverse (project_point_into_view) and the
    coverage footprints (view_coverage_counts), so all three agree exactly."""
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    r_theta = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                        [0, 1, 0],
                        [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
    r_phi = np.array([[1, 0, 0],
                      [0, np.cos(phi_rad), -np.sin(phi_rad)],
                      [0, np.sin(phi_rad), np.cos(phi_rad)]])
    return r_theta @ r_phi


def get_face_projection_map(h: int, w: int, theta: float, phi: float, size: int, fov: float = 90) -> tuple[np.ndarray, np.ndarray]:
    """Builds the cv2.remap coordinate maps for a perspective crop out of an equirectangular panorama."""
    fov_rad = np.radians(fov)

    step = np.tan(fov_rad / 2)
    x = np.linspace(-step, step, size)
    y = np.linspace(step, -step, size)

    xv, yv = np.meshgrid(x, y)
    z = np.ones_like(xv)
    vec = np.stack([xv, yv, z], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    r = _view_rotation(theta, phi)
    vec_rot = vec @ r.T

    lon = np.arctan2(vec_rot[..., 0], vec_rot[..., 2])
    lat = np.arcsin(vec_rot[..., 1])

    uf = (lon / (2 * np.pi) + 0.5) * w
    vf = (0.5 - lat / np.pi) * h

    map_x = uf.astype(np.float32)
    map_y = vf.astype(np.float32)

    return map_x, map_y


def real_point_mask(node: ScanNode, min_distance: float = 0.1) -> NDArray:
    """
    True for points at least `min_distance` from the scanner origin, False for
    the rest - many TLS scanners don't drop a grid cell when a pulse gets no
    return (e.g. aimed at open sky) but instead fill it with a degenerate
    point sitting exactly at the scanner's own origin, rather than leaving a
    genuine gap (measured on a real scan: every point in the topmost rows had
    distance exactly 0.0). LIDAR has a real minimum range anyway, so this costs
    no genuine near-scanner coverage. Use this to filter a mask's back-projected
    point indices (`indices_3d[real_point_mask(node)[indices_3d]]`) so a
    detected instance never silently includes points that aren't real geometry.
    """
    pts = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center
    return np.linalg.norm(pts, axis=1) > min_distance


def real_scan_mask(node: ScanNode, min_distance: float = 0.1) -> NDArray:
    """
    A stricter, padded version of `node.index_mapper.pano_map >= 0` for use as
    compute_coverage_grid's valid_mask - see real_point_mask for why a plain
    "does this cell have a point index" check alone isn't enough. Pads to the
    (h, w) reconstruct_panorama_for_channel(pad=True) uses by default, since
    that's what a sweep actually renders against.
    """
    mapper = node.index_mapper
    real = real_point_mask(node, min_distance)

    grid = np.zeros((mapper.h, mapper.w), dtype=bool)
    valid_cells = mapper.pano_map >= 0
    grid[valid_cells] = real[mapper.pano_map[valid_cells]]

    return pad_img(grid, fill_value=False)


def phi_extent_of_valid_data(valid_mask: NDArray) -> tuple[float, float]:
    """
    Returns the (phi_lo, phi_hi) span that actually contains valid data in
    `valid_mask`, derived from its populated row range - i.e. exactly how far
    up/down a scan's real measurements reach, with no guessed bound. Rows
    without any valid pixel (a scanner's genuine vertical limit, or the
    padding reconstruct_panorama_for_channel adds - both are "no data" the
    same way) are excluded from both ends.
    """
    h = valid_mask.shape[0]
    rows_with_data = np.where(valid_mask.any(axis=1))[0]
    if len(rows_with_data) == 0:
        return (0.0, 0.0)
    top_row, bottom_row = int(rows_with_data.min()), int(rows_with_data.max()) + 1
    # Inverse of get_face_projection_map's row<->latitude mapping (lat = pi*(0.5 - row/h)),
    # combined with phi = -lat (see get_face_projection_map: phi=+30 looks down, i.e. to
    # negative latitude) -> phi = 180*(row/h - 0.5).
    phi_lo = 180.0 * (top_row / h - 0.5)
    phi_hi = 180.0 * (bottom_row / h - 0.5)
    return (phi_lo, phi_hi)


def compute_coverage_grid(valid_mask: NDArray, fov: float = 90.0, overlap: float = 0.2,
                           probe_size: int = 32,
                           phi_range: tuple[float, float] | None = None,
                           theta_range: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """
    Computes the (theta, phi) view centers needed to sweep every valid
    (non-empty) region of an equirectangular panorama with `fov`-sized
    perspective crops, with `overlap` fractional angular overlap between
    neighboring crops so an object near one crop's edge still falls fully
    inside a neighbor - while skipping candidate crops whose entire footprint
    lands on pixels the scan never captured (valid_mask False everywhere in
    that crop), which a fixed theta/phi sweep has no way to know about.

    `valid_mask` should be `real_scan_mask(node)` (or, at minimum,
    `node.index_mapper.pano_map >= 0` if a plain "does this grid cell have a
    point at all" check is genuinely what's wanted) - see real_scan_mask for
    why a plain pano_map check alone isn't enough on real scanner output.

    `probe_size` only needs to be big enough to reliably catch a crop that
    contains any real data at all - it doesn't need to match the resolution
    later used to actually render/detect on that crop.

    `phi_range` restricts how far up/down candidate ring *centers* may be
    placed (phi > 0 looks down, phi < 0 looks up - see get_face_projection_map).
    Left as None (the default), it's derived from `valid_mask` itself via
    phi_extent_of_valid_data - i.e. go exactly as far up/down as the scan
    actually has data, no further, no manually guessed bound needed. Pass an
    explicit, narrower range only to deliberately exclude a sub-region that
    does have real data but isn't useful to sweep for some other reason (e.g.
    a detector that's known to misbehave on a specific viewing angle for a
    specific prompt).

    `theta_range` restricts the sweep to one azimuthal wedge (e.g. "just this
    building") instead of the full 360 - left as None (the default), the
    sweep covers the full circle as before this parameter existed. See
    compute_indexed_coverage_grid for the exact (theta_lo, theta_hi) contract.
    """
    entries, _n_theta = compute_indexed_coverage_grid(
        valid_mask, fov=fov, overlap=overlap, probe_size=probe_size,
        phi_range=phi_range, theta_range=theta_range,
    )
    return [(theta, phi) for _theta_idx, _phi_idx, theta, phi in entries]


def compute_indexed_coverage_grid(
    valid_mask: NDArray, fov: float = 90.0, overlap: float = 0.2, probe_size: int = 32,
    phi_range: tuple[float, float] | None = None,
    theta_range: tuple[float, float] | None = None,
) -> tuple[list[tuple[int, int, float, float]], int]:
    """
    Same coverage-grid computation as compute_coverage_grid, but also returns
    each kept tile's (theta_idx, phi_idx) position in the full candidate grid
    (before any tiles were dropped for having no valid data) plus that grid's
    `n_theta` - what core.merge's view-adjacency gating needs to tell genuinely
    neighboring views apart from unrelated ones, the same way it already does
    for a manually chosen step_deg x pitch sweep. A tile skipped for having no
    data simply leaves a gap in these indices; two surviving tiles that were
    already adjacent in the full grid are still detected as adjacent regardless
    of what got dropped elsewhere.

    `theta_range` restricts the sweep to one azimuthal wedge (e.g. "only the
    building in front of me") instead of the full 360 - left as None (the
    default), the sweep covers the full circle, unchanged from before this
    parameter existed. Given explicitly as (theta_lo, theta_hi) with
    theta_lo < theta_hi (pick a representation that doesn't need to cross the
    0/360 seam, e.g. -45..45 rather than 315..45 for a wedge straddling
    forward), a restricted wedge has two open ends rather than wrapping into
    itself - the returned n_theta is therefore 0 in that case (rather than the
    real step count), which tells core.merge's adjacency gating (`d == n_theta
    - 1` wraps only when n_theta > 0) not to treat the wedge's two open ends
    as neighbors just because a full circle's matching indices would be.
    """
    h, w = valid_mask.shape[:2]
    if not valid_mask.any():
        return [], 0

    step = fov * (1.0 - overlap)
    if not (0 < step <= fov):
        raise ValueError(f"overlap must be in [0, 1) (got {overlap!r})")

    if phi_range is None:
        phi_range = phi_extent_of_valid_data(valid_mask)
    phi_lo, phi_hi = phi_range
    if not (-90.0 <= phi_lo < phi_hi <= 90.0):
        raise ValueError(f"phi_range must satisfy -90 <= lo < hi <= 90 (got {phi_range!r})")

    # ceil (not round): a coverage guarantee must never end up with fewer steps
    # than requested - rounding down at an exact half-step (e.g. 180/72 = 2.5)
    # would silently produce zero overlap instead of the requested amount.
    if theta_range is None:
        n_theta = max(1, math.ceil(360.0 / step))
        thetas = [i * 360.0 / n_theta for i in range(n_theta)]
        wrap_n_theta = n_theta
    else:
        theta_lo, theta_hi = theta_range
        if not theta_lo < theta_hi:
            raise ValueError(f"theta_range must satisfy lo < hi (got {theta_range!r})")
        theta_span = theta_hi - theta_lo
        n_theta = max(1, math.ceil(theta_span / step))
        thetas = [theta_lo + (i + 0.5) * (theta_span / n_theta) for i in range(n_theta)]
        wrap_n_theta = 0  # an open wedge, not a closed ring - never wrap-adjacent

    phi_span = phi_hi - phi_lo
    n_phi = max(1, math.ceil(phi_span / step))
    phis = [phi_lo + (j + 0.5) * (phi_span / n_phi) for j in range(n_phi)]

    entries = []
    for phi_idx, phi in enumerate(phis):
        for theta_idx, theta in enumerate(thetas):
            map_x, map_y = get_face_projection_map(h, w, theta, phi, probe_size, fov)
            xs = np.clip(map_x.astype(np.int32), 0, w - 1)
            ys = np.clip(map_y.astype(np.int32), 0, h - 1)
            if valid_mask[ys, xs].any():
                entries.append((theta_idx, phi_idx, theta, phi))
    return entries, wrap_n_theta


def compute_distance_panorama(node: ScanNode, valid_mask: NDArray | None = None) -> NDArray:
    """
    Per-panorama-cell distance from the scanner origin (world (0,0,0) in the
    dequantized frame - same convention `real_point_mask` already uses), 0 for
    any cell with no real point there. Cheap: the scan's own capture grid is
    (at most) one point per cell, so this is a straight placement via
    IndexMapper.map_to_panorama, not an aggregation over several points.

    `valid_mask`, if given (typically `real_scan_mask(node)`), additionally
    zeroes out any cell it marks False - without this, a cell excluded for
    holding only a degenerate "no return" placeholder point (distance exactly
    0 at the scanner's own origin - see `real_point_mask`) is already 0 and
    harmless, but an otherwise-valid cell that `valid_mask` excludes for some
    OTHER reason (e.g. a caller-restricted phi/theta range) should read as
    "no data" here too, not whatever raw distance its point happens to have.

    Padded to the same (mapper.w // 2, mapper.w) shape `real_scan_mask` itself
    pads to (see `pad_img`) - `IndexMapper.map_to_panorama` on its own returns
    the RAW, unpadded (mapper.h, mapper.w) shape (real crash on a real scan:
    mapper.h=1194 vs. real_scan_mask's own padded 1524, a broadcast failure
    the moment the two were combined below), so this pads to match BEFORE
    `valid_mask` (already padded) is applied against it.
    """
    pts = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center
    distances = np.linalg.norm(pts, axis=1)
    dist_panorama = node.index_mapper.map_to_panorama(distances, default_value=0.0)
    dist_panorama = pad_img(dist_panorama, fill_value=0.0)
    if valid_mask is not None:
        dist_panorama = np.where(valid_mask, dist_panorama, 0.0)
    return dist_panorama


def compute_multiscale_coverage_grid(
    node: ScanNode, fov: float = 90.0, overlap: float = 0.2,
    far_percentile: float = 75.0, far_fov_scale: float = 0.5,
    near_percentile: float = 25.0, near_fov_scale: float = 1.3,
    probe_size: int = 32,
    phi_range: tuple[float, float] | None = None,
    theta_range: tuple[float, float] | None = None,
) -> list[tuple[float, float, float]]:
    """
    A three-way partition of the scan's own coverage (not three overlapping
    passes): every direction is swept by exactly one tier's appropriately
    scaled tiles, based on how far the real point there actually is.
    Combines into a single (theta, phi, fov) list -
    `SegmentationService.run_sweep`'s `view_grid` accepts this shape
    directly, alongside its plain (theta, phi) form.

    - FAR tier: cells beyond `far_percentile` of the scan's own distance
      distribution (not a fixed meter value, so it scales across wildly
      different scan sizes), rendered at a smaller `fov * far_fov_scale` -
      a distant object subtends fewer panorama pixels, so a tighter zoom
      keeps it resolvable.
    - NEAR tier: cells within `near_percentile`, at a larger
      `fov * near_fov_scale` - a close object is already legible, so
      trading some resolution for more surrounding context helps
      disambiguate it. Kept modest by default since a rectilinear crop
      distorts increasingly toward its edges as FOV grows.
    - MIDDLE tier: the plain `fov`, covering whatever's left once the far
      and near tiers' cells are carved out, so every direction is covered
      by exactly one tier, never two.

    Any tier restricted to a mask with no real data anywhere in it (e.g. a
    scan with no meaningfully distant OR close material after percentile
    thresholding, or - now that the middle tier is itself restricted - a scan
    where the far/near tiers happen to already cover everything) simply
    contributes zero tiles, same as `compute_coverage_grid` already does for
    any single empty tile.

    `phi_range`/`theta_range`, if given, restrict ALL THREE tiers to that
    SAME region (e.g. an active ROI) - each tier still independently narrows
    ITS OWN candidate area within that restriction via `valid_mask`, so a
    far-only or near-only tier never probes tiles outside where its own kind
    of material actually exists, even inside a shared ROI.
    """
    valid_mask = real_scan_mask(node)
    real_mask = real_point_mask(node)
    if not real_mask.any():
        # No real distance data to tier by at all - fall back to a single,
        # scan-wide pass at the plain fov (the same result this function
        # always returned in this case).
        return [(theta, phi, fov) for theta, phi in compute_coverage_grid(
            valid_mask, fov=fov, overlap=overlap, probe_size=probe_size,
            phi_range=phi_range, theta_range=theta_range,
        )]

    pts = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center
    real_distances = np.linalg.norm(pts[real_mask], axis=1)
    dist_panorama = compute_distance_panorama(node, valid_mask)

    far_threshold = float(np.percentile(real_distances, far_percentile))
    far_fov = fov * far_fov_scale
    far_mask = valid_mask & (dist_panorama >= far_threshold)
    far = [(theta, phi, far_fov) for theta, phi in compute_coverage_grid(
        far_mask, fov=far_fov, overlap=overlap, probe_size=probe_size,
        phi_range=phi_range, theta_range=theta_range,
    )]

    near_threshold = float(np.percentile(real_distances, near_percentile))
    near_fov = fov * near_fov_scale
    near_mask = valid_mask & (dist_panorama > 0) & (dist_panorama <= near_threshold)
    near = [(theta, phi, near_fov) for theta, phi in compute_coverage_grid(
        near_mask, fov=near_fov, overlap=overlap, probe_size=probe_size,
        phi_range=phi_range, theta_range=theta_range,
    )]

    middle_mask = valid_mask & ~far_mask & ~near_mask
    middle = [(theta, phi, fov) for theta, phi in compute_coverage_grid(
        middle_mask, fov=fov, overlap=overlap, probe_size=probe_size,
        phi_range=phi_range, theta_range=theta_range,
    )]

    return middle + far + near


def view_coverage_counts(h: int, w: int, view_grid: list[tuple[float, float] | tuple[float, float, float]],
                         default_fov: float = 90.0) -> list[NDArray]:
    """
    One boolean (h, w) mask per view of `view_grid`: True for every
    equirectangular-panorama pixel that view's perspective crop actually
    sees. Computed per pixel by the exact inverse of get_face_projection_map
    (pixel -> direction -> the view's local frame -> inside its square
    frustum), so a view pitched up or down gets its true, curved footprint -
    wider towards the pole - rather than a theta/phi rectangle, and a view
    straddling the 0/360 seam comes out as the two pieces it really is.
    Entries are (theta, phi) with `default_fov`, or (theta, phi, fov).
    """
    lon = ((np.arange(w) + 0.5) / w - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon_g, lat_g = np.meshgrid(lon, lat)
    directions = np.stack([np.cos(lat_g) * np.sin(lon_g), np.sin(lat_g), np.cos(lat_g) * np.cos(lon_g)], axis=-1)

    masks = []
    for entry in view_grid:
        theta, phi = entry[0], entry[1]
        fov = entry[2] if len(entry) == 3 else default_fov
        local = directions @ _view_rotation(theta, phi)
        step = np.tan(np.radians(fov) / 2.0)
        z = local[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            inside = (z > 1e-6) & (np.abs(local[..., 0]) <= step * z) & (np.abs(local[..., 1]) <= step * z)
        masks.append(inside)
    return masks



_COVERAGE_COUNT_COLORS_BGR = [(40, 40, 230), (80, 190, 60), (0, 190, 255), (200, 80, 210)]
_COVERAGE_COUNT_LABELS = ["not covered", "1 view", "2 views", "3+ views"]
_COVERAGE_OUTLINE_PALETTE_BGR = [(255, 255, 255), (255, 255, 0), (180, 105, 255), (0, 255, 128), (0, 128, 255)]


def render_coverage_grid_preview(
    node: ScanNode,
    view_grid: list[tuple[float, float] | tuple[float, float, float]],
    default_fov: float = 90.0,
    channel: Channel | None = None,
) -> NDArray:
    """
    Renders the scan's own full 360-degree equirectangular panorama with
    `view_grid`'s own tiles (see `compute_coverage_grid`/
    `compute_multiscale_coverage_grid` - accepts either's output, and the
    same mixed (theta, phi) / (theta, phi, fov) entry shape
    `SegmentationService.run_sweep`'s own `view_grid` parameter does) drawn
    on top as colored rectangles - lets you SEE exactly what a sweep's own
    coverage grid covers, and where its different zoom tiers (if any) land,
    before actually running anything. `default_fov` is what a plain
    (theta, phi) entry uses (matching `run_sweep`'s own global `fov` for
    such entries).

    Each DISTINCT fov value present gets its own color (see
    `_COVERAGE_PREVIEW_PALETTE_BGR`) and its own legend line (with degree
    value + tile count) in the top-left corner. A tile whose angular span
    crosses the panorama's own 0/360 seam is drawn as two separate
    rectangles rather than one wrapping incorrectly across the whole image.

    Returns a BGR uint8 image (ready for cv2.imwrite) of the panorama's own
    (padded) size; never modifies `node`.
    """
    channel = channel or default_channel(node)
    panorama = reconstruct_panorama_for_channel(node, channel)
    if panorama is None:
        raise ValueError(f"Channel {channel!r} is not available for this scan.")
    h, w = panorama.shape[:2]
    if channel == "rgb":
        # reconstruct_panorama_for_channel leaves "rgb" in RGB order but already
        # converts "intensity"/"normals" to BGR; cv2 wants BGR.
        panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(np.ascontiguousarray(panorama), cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor((gray * 0.6).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    entries = [(e[0], e[1], e[2] if len(e) == 3 else default_fov) for e in view_grid]

    # Footprints on a reduced grid (fast even for a full-resolution panorama),
    # scaled back up with nearest-neighbour.
    scale = max(1, int(np.ceil(w / 1024)))
    hs, ws = max(1, h // scale), max(1, w // scale)
    masks = view_coverage_counts(hs, ws, entries, default_fov)
    counts = np.sum(masks, axis=0) if masks else np.zeros((hs, ws), dtype=np.int64)
    counts = cv2.resize(counts.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    scanned = real_scan_mask(node)
    if scanned.shape != (h, w):
        scanned = cv2.resize(scanned.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    tint = np.zeros_like(canvas)
    level = np.minimum(counts, 3)
    for n, color in enumerate(_COVERAGE_COUNT_COLORS_BGR):
        tint[scanned & (level == n)] = color
    canvas[scanned] = cv2.addWeighted(canvas, 0.5, tint, 0.5, 0)[scanned]

    font_scale = max(0.5, w / 3000.0)
    thickness = max(1, int(round(2 * font_scale)))
    distinct_fovs = sorted({round(e[2], 3) for e in entries})
    outline_color = {f: _COVERAGE_OUTLINE_PALETTE_BGR[i % len(_COVERAGE_OUTLINE_PALETTE_BGR)]
                     for i, f in enumerate(distinct_fovs)}

    for index, ((theta, phi, fov), mask) in enumerate(zip(entries, masks), start=1):
        color = outline_color[round(fov, 3)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.polylines(canvas, [(contour * scale).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
        cx = int(((theta / 360.0 + 0.5) % 1.0) * w)
        cy = int(min(max(phi / 180.0 + 0.5, 0.0), 1.0) * h)
        _put_outlined_text(canvas, str(index), (cx, cy), 1.3 * font_scale, color, thickness + 1, centered=True)

    covered = float((counts[scanned] > 0).mean() * 100.0) if scanned.any() else 0.0
    lines = [(color, label) for color, label in zip(_COVERAGE_COUNT_COLORS_BGR, _COVERAGE_COUNT_LABELS)]
    lines += [(outline_color[f], f"{f:g} deg views ({sum(1 for e in entries if round(e[2], 3) == f)})")
              for f in distinct_fovs]
    _draw_coverage_legend(canvas, lines, f"scanned area covered: {covered:.1f}%", font_scale)
    return canvas


def _put_outlined_text(image: NDArray, text: str, org: tuple[int, int], scale: float, color: tuple,
                       thickness: int, centered: bool = False) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if centered:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        org = (org[0] - tw // 2, org[1] + th // 2)
    cv2.putText(image, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _draw_coverage_legend(image: NDArray, lines: list[tuple[tuple, str]], footer: str, font_scale: float) -> None:
    """Swatch + label per line, then a footer, on a dark box in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9 * font_scale
    thickness = max(1, int(round(2 * font_scale)))
    line_h = int(40 * font_scale)
    pad = int(15 * font_scale)
    swatch = int(26 * font_scale)
    texts = [label for _, label in lines] + [footer]
    text_w = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in texts)
    box_w = pad * 3 + swatch + text_w
    box_h = pad * 2 + line_h * len(texts)
    x0, y0 = pad, pad
    region = image[y0:y0 + box_h, x0:x0 + box_w]
    region[:] = (region * 0.25).astype(np.uint8)
    for i, (color, label) in enumerate(lines):
        top = y0 + pad + i * line_h
        cv2.rectangle(image, (x0 + pad, top), (x0 + pad + swatch, top + swatch), color, -1)
        cv2.putText(image, label, (x0 + pad * 2 + swatch, top + swatch - 4), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
    cv2.putText(image, footer, (x0 + pad, y0 + pad + len(lines) * line_h + swatch - 4), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def pad_img(img: NDArray, fill_value: int = 255) -> NDArray:
    """Pads an image from the bottom so its height is at least width/2 (full equirectangular ratio)."""
    h, w = img.shape[:2]
    target_h = int(w / 2)

    if h >= target_h:
        return img

    pad_bottom = target_h - h

    if img.ndim == 2:
        pad_width = ((0, pad_bottom), (0, 0))
    elif img.ndim == 3:
        pad_width = ((0, pad_bottom), (0, 0), (0, 0))
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")

    return np.pad(img, pad_width=pad_width, mode="constant", constant_values=fill_value)


def reconstruct_panorama_for_channel(node: ScanNode, channel: Channel, pad: bool = True) -> NDArray | None:
    """
    Rebuilds the full equirectangular panorama image for one specific channel
    (rgb / intensity / normals), independent of any "currently active" UI state.

    `pad=False` returns the image at exactly (mapper.h, mapper.w) - the same shape
    IndexMapper.map_to_panorama produces - so callers that need to align a mask
    (built via map_to_panorama) with this image pixel-for-pixel can do so safely.
    Padding to a full equirectangular aspect ratio is a cosmetic step and must be
    applied *after* any such mask alignment, not before.
    """
    mapper = node.index_mapper
    image_data = None

    if channel == "rgb":
        if node.colors_rgb is not None:
            image_data = mapper.map_to_panorama(node.colors_rgb, default_value=0)

    elif channel == "intensity":
        if node.colors_intensity is not None:
            gray_img = mapper.map_to_panorama(node.colors_intensity, default_value=0)
            gray_norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            image_data = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)

    elif channel == "normals":
        normals_grid = mapper.map_to_panorama(node.normals, default_value=0)
        if normals_grid.dtype.kind == "f":
            image_data = ((normals_grid + 1.0) * 127.5).astype(np.uint8)
        else:
            image_data = (normals_grid.astype(np.int16) + 128).clip(0, 255).astype(np.uint8)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError(f"Unknown channel: {channel!r}")

    if image_data is None:
        return None

    if node.hidden_mask is not None:
        hidden_2d = mapper.map_to_panorama(node.hidden_mask, default_value=False)
        if hidden_2d.any():
            image_data[hidden_2d] = [0, 0, 0]

    return pad_img(image_data, fill_value=0) if pad else image_data


def extract_perspective_view(node: ScanNode, center_index: int, out_size: int | None = None, fov: float = 90.0,
                              channel: Channel | None = None) -> ProjectionContext | None:
    """Crops one perspective view out of the panorama, centered on a given point index."""
    channel = channel or default_channel(node)
    full_panorama = reconstruct_panorama_for_channel(node, channel)
    if full_panorama is None:
        return None

    h_final, w_final = full_panorama.shape[:2]

    if out_size is None:
        native_size = int((fov / 360.0) * w_final)
        out_size = min(native_size, 2048)

    pano_y, pano_x = node.index_mapper.point_to_pano[center_index]
    if pano_y < 0:
        logging.warning(f"Point {center_index} has no panorama position; cannot extract view.")
        return None

    theta_deg = (int(pano_x) / w_final - 0.5) * 360.0
    phi_deg = (0.5 - int(pano_y) / h_final) * 180.0

    map_x, map_y = get_face_projection_map(h=h_final, w=w_final, theta=theta_deg, phi=-phi_deg, size=out_size, fov=fov)
    result_image = cv2.remap(full_panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return ProjectionContext(
        scan_id=node.scan_id,
        pano_shape=full_panorama.shape,
        fov=fov,
        map_x=map_x,
        map_y=map_y,
        image_2d=result_image,
        index_mapper=node.index_mapper,
        center_px=out_size // 2,
        center_py=out_size // 2,
        rot_theta_deg=theta_deg,
        rot_phi_deg=-phi_deg,
    )


def project_point_into_view(node: ScanNode, ctx: ProjectionContext, point_index: int) -> tuple[int, int] | None:
    """
    Inverse of the remap `extract_perspective_view` used to build `ctx`:
    given a DIFFERENT point index, returns its own (px, py) pixel location
    within `ctx.image_2d`, or None if that point falls outside this view's
    FOV entirely (behind the view direction, or beyond its fov/2 half-angle
    on either axis).

    Needed for multi-point SAM2 prompting (see
    point_segmentation_service.segment_at_point's `extra_indices`): SAM2
    takes every point of one prompt as pixel coordinates on a SINGLE image,
    so a second/third clicked point has to be expressed in the SAME crop the
    anchor point's own view already produced, not re-extracted as its own
    fresh view (which would just be a different crop, with no guarantee the
    anchor point is even IN it).

    Rederives the point's own (theta, phi) from its panorama position the
    same way extract_perspective_view derives the anchor's, converts both to
    a unit direction vector, undoes `ctx`'s own rotation (its `rot_theta_deg`/
    `rot_phi_deg`, exactly what get_face_projection_map's forward rotation
    used) to get back into the crop's local (pre-rotation) frame, then
    inverts that function's own tan(fov/2)-normalized linspace to recover a
    pixel coordinate - the exact algebraic inverse of get_face_projection_map,
    not an approximation.
    """
    h_final, w_final = ctx.pano_shape[:2]
    pano_y, pano_x = node.index_mapper.point_to_pano[point_index]
    if pano_y < 0:
        return None

    lon = (int(pano_x) / w_final - 0.5) * 2 * np.pi
    lat = (0.5 - int(pano_y) / h_final) * np.pi
    vec_target = np.array([
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
        np.cos(lat) * np.cos(lon),
    ])

    r = _view_rotation(ctx.rot_theta_deg, ctx.rot_phi_deg)

    # get_face_projection_map computes vec_rot = vec @ r.T, so the inverse
    # (r is orthogonal, r.T == r^-1) is vec = vec_rot @ r.
    vec_local = vec_target @ r

    if vec_local[2] <= 1e-6:
        return None  # behind the view direction entirely

    step = np.tan(np.radians(ctx.fov) / 2.0)
    x_frac = vec_local[0] / vec_local[2]
    y_frac = vec_local[1] / vec_local[2]
    if abs(x_frac) > step or abs(y_frac) > step:
        return None

    size = ctx.image_2d.shape[0]
    px = int(round((x_frac + step) / (2 * step) * (size - 1)))
    py = int(round((step - y_frac) / (2 * step) * (size - 1)))
    px = min(max(px, 0), size - 1)
    py = min(max(py, 0), size - 1)
    return px, py
