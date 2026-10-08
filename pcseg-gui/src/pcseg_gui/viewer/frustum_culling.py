"""
Coarse, conservative frustum culling for the point-cloud renderer.

Self-contained on purpose, so the whole feature is trivial to rip out: this
module holds ALL of the spatial-partition-building and plane math, and
ShaderPointCloudItem (point_cloud_item.py) only touches it through a handful
of clearly-marked call sites gated behind a single `frustum_culling_enabled`
flag (see `set_frustum_culling()`), checked in `_upload_ebo()` exactly like
`view_decimation`'s own existing flag already is - it's one more index-set
filter feeding the same EBO-building pipeline, not a new rendering path.
Deleting this file and reverting those few call sites removes the feature
completely, with zero effect on shaders, picking, or layer compositing.

Deliberately NOT per-point exact and NOT a real octree: a regular grid, built
once at load time (build_spatial_grid), broad-phase-tested on camera change
against the current camera's 6 frustum planes (extract_frustum_planes,
cull_cells). A cell is REJECTED only if it is PROVABLY entirely outside at
least one plane (the standard "most favorable corner" / p-vertex test);
partially-overlapping and fully-inside cells are always kept. This "when in
doubt, keep it" bias is deliberate, per the feature request: a bug that hides
visible geometry is far worse than one that renders a bit extra near the
frustum edge, and it keeps the per-cell test itself a single cheap check per
plane rather than needing exact polygon clipping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtGui import QMatrix4x4


@dataclass
class SpatialGrid:
    """One regular 3D grid, built once at load time. `cell_point_indices[i]`
    are the point indices in cell i; `cell_aabbs[i]` is that cell's own
    (min_x,min_y,min_z,max_x,max_y,max_z) - computed from its actual member
    points, not the nominal grid-cell bounds, so a sparse cell's AABB is never
    bigger than the real geometry it contains. Only occupied cells exist (no
    empty-cell entries)."""
    cell_point_indices: list[np.ndarray]
    cell_aabbs: np.ndarray  # (n_cells, 6) float32


def build_spatial_grid(points_xyz: np.ndarray, target_cells: int = 512) -> SpatialGrid:
    """
    Bins `points_xyz` (N,3), real (unquantized) world coordinates, into a
    regular grid sized so the total OCCUPIED-cell count is roughly
    `target_cells` - a coarse broad-phase partition, not a real octree (see
    module docstring for why that's deliberate). `target_cells` translates to
    `round(target_cells**(1/3))` cells per axis, so e.g. the default 512 gives
    an 8x8x8 candidate grid (fewer than that many cells actually end up
    occupied on a typical scan's non-cubical extent).
    """
    n = len(points_xyz)
    if n == 0:
        return SpatialGrid([], np.zeros((0, 6), dtype=np.float32))

    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)

    cells_per_axis = max(1, round(target_cells ** (1.0 / 3.0)))
    cell_size = extent / cells_per_axis

    cell_idx = np.floor((points_xyz - mins) / cell_size).astype(np.int64)
    cell_idx = np.clip(cell_idx, 0, cells_per_axis - 1)
    flat_idx = (cell_idx[:, 0] * cells_per_axis + cell_idx[:, 1]) * cells_per_axis + cell_idx[:, 2]

    order = np.argsort(flat_idx, kind="stable")
    sorted_flat = flat_idx[order]
    _unique_cells, start_positions = np.unique(sorted_flat, return_index=True)
    cell_point_indices = np.split(order, start_positions[1:])

    cell_aabbs = np.zeros((len(cell_point_indices), 6), dtype=np.float32)
    for i, idxs in enumerate(cell_point_indices):
        pts = points_xyz[idxs]
        cell_aabbs[i, :3] = pts.min(axis=0)
        cell_aabbs[i, 3:] = pts.max(axis=0)

    return SpatialGrid(cell_point_indices, cell_aabbs)


def extract_frustum_planes(view_projection: QMatrix4x4) -> np.ndarray:
    """
    The 6 frustum clipping planes of `view_projection` (= projection * view,
    the SAME combined matrix ShaderPointCloudItem.paint() already builds for
    its own u_mvp uniform), as a (6,4) array of (a,b,c,d) rows - standard
    Gribb & Hartmann (2001) row-combination extraction. A world-space point
    p=(x,y,z) is on the INSIDE of plane i iff a*x + b*y + c*z + d >= 0. Rows
    are normalized so (a,b,c) is a unit normal, making that value a true
    signed distance and keeping cull_cells' comparisons well-conditioned
    regardless of the matrix's own scale.

    Uses QMatrix4x4.row(i), NOT a manual reshape of .data() - .data() is
    OpenGL-native COLUMN-major (see point_cloud_item.py's own u_mvp upload
    via glUniformMatrix4fv(..., GL_FALSE, ...)), so reinterpreting it by hand
    as a plain row-major 4x4 would silently transpose the matrix. .row()
    gives the mathematically-standard row Gribb-Hartmann expects regardless
    of the underlying storage layout - verified directly against a known
    translation matrix while writing this (row(0) of a pure +X translation
    is (1,0,0,tx), matching the row-major convention the formula assumes).
    """
    def _vec4(v) -> np.ndarray:
        return np.array([v.x(), v.y(), v.z(), v.w()], dtype=np.float64)

    m0 = _vec4(view_projection.row(0))
    m1 = _vec4(view_projection.row(1))
    m2 = _vec4(view_projection.row(2))
    m3 = _vec4(view_projection.row(3))

    planes = np.array([
        m3 + m0,  # left
        m3 - m0,  # right
        m3 + m1,  # bottom
        m3 - m1,  # top
        m3 + m2,  # near
        m3 - m2,  # far
    ])
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return planes / norms


def cull_cells(cell_aabbs: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """
    Broad-phase-only AABB/frustum test: returns a bool mask (n_cells,), True
    for cells to KEEP. A cell is rejected ONLY if it is provably entirely
    outside at least one plane (the standard "most favorable corner"/p-vertex
    test: for each plane, pick the AABB corner that's furthest in the
    plane-normal direction - if even THAT corner fails the plane's inside
    test, the whole box must be outside it). A cell that's fully inside or
    merely intersecting the frustum always survives every plane and is kept -
    see module docstring for why that generous bias is deliberate.
    """
    if len(cell_aabbs) == 0:
        return np.zeros(0, dtype=bool)

    mins, maxs = cell_aabbs[:, :3], cell_aabbs[:, 3:]
    keep = np.ones(len(cell_aabbs), dtype=bool)
    for a, b, c, d in planes:
        px = np.where(a >= 0, maxs[:, 0], mins[:, 0])
        py = np.where(b >= 0, maxs[:, 1], mins[:, 1])
        pz = np.where(c >= 0, maxs[:, 2], mins[:, 2])
        signed_dist = a * px + b * py + c * pz + d
        keep &= signed_dist >= 0
    return keep


def cull_point_indices(grid: SpatialGrid, planes: np.ndarray) -> np.ndarray:
    """Union of every surviving cell's point indices, as one flat int64
    array - what ShaderPointCloudItem folds into its existing hidden_mask/
    ROI/view_decimation index-set filtering in _upload_ebo()."""
    keep_cells = cull_cells(grid.cell_aabbs, planes)
    if not keep_cells.any():
        return np.array([], dtype=np.int64)
    kept = [grid.cell_point_indices[i] for i in np.where(keep_cells)[0]]
    return np.concatenate(kept) if kept else np.array([], dtype=np.int64)
