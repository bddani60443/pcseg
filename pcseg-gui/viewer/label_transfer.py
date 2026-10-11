"""
Carries classified instances over to a RE-LOAD of the same scan at a
DIFFERENT Load detail (decimation) - "I classified this at a coarse
decimation, now let me load the full-resolution scan (or a smaller one) and
keep what I already found." A different decimation samples an entirely
different set of points from the source file (E57 decimation strides through
the structured capture grid), so the OLD point indices are simply meaningless
against the NEW point set - there is no valid index-based carryover here, by
construction (unlike project_file.py's project-reopen case, which always
reloads at the EXACT SAME decimation the indices were saved against).

Self-contained on purpose (pure numpy, same design as density_decimation.py)
- the technique is a voxel-grid
SPATIAL label transfer, reusing density_decimation.py's own voxel-hashing
(pack_voxel_ids): bin the OLD classified points into voxels, remembering
which instance "owns" each occupied voxel (smaller instances win ties over
bigger ones sharing a voxel - same "more specific wins overlap" convention
layers.py's own module docstring already describes for composing the Layers
stack), then for every NEW point, look up whether its own voxel was
previously owned by an instance and inherit that ownership if so.

This is coarse (voxel-resolution, not exact) by construction - a genuinely
fine, sub-voxel boundary between two instances can lose or misassign a thin
fringe of points either way, and a NEW point whose voxel had no OLD
classified point at all (extra detail the coarser load never resolved, or a
handful of empty gaps) simply gets no label. Good enough for "keep going at
a different resolution without starting over," not a claim of exactness.

IMPORTANT: build_voxel_owner_map and transfer_owners are always used as a
PAIR, in that order, because voxel ids from the two calls must be directly
comparable - see VoxelOwnerMap's own docstring for why that needs an
explicit SHARED origin (not pack_voxel_ids' own default, self-computed-min
behavior, which is only self-consistent WITHIN one call).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viewer.density_decimation import DEFAULT_VOXEL_SIZE, pack_voxel_ids

__all__ = ["DEFAULT_VOXEL_SIZE", "VoxelOwnerMap", "build_voxel_owner_map", "transfer_owners"]

# Added to the OLD points' own minimum corner (see VoxelOwnerMap.origin) as a
# safety margin, so the NEW (independently loaded) point set - almost
# certainly close to the old one for the same physical scan re-decimated,
# but not guaranteed bit-identical - can extend a bit further in the
# negative direction without its voxel coordinates clipping to 0 and
# silently colliding with real near-origin voxels. Generous on purpose (a
# real scan's re-decimated bounding box shifting by even a few meters would
# be surprising) - it only costs unused id space, of which there is a lot
# (see density_decimation.py's own _BITS_PER_AXIS headroom).
_ORIGIN_MARGIN_M = 1000.0


@dataclass
class VoxelOwnerMap:
    """`owner_map`'s voxel ids and any `transfer_owners` call against it
    MUST use the exact same `origin` (and `voxel_size`) - see
    density_decimation.pack_voxel_ids' own docstring for why voxel ids from
    two INDEPENDENT calls (one over the OLD points, one over the NEW ones)
    are only comparable when given the same explicit origin; each call's
    own default (self-computed min) would silently misalign two entirely
    different point sets' id spaces. Bundling the three together here (one
    object returned by build_voxel_owner_map, threaded straight into
    transfer_owners) makes that pairing impossible to get wrong by
    accident - there's no loose origin/voxel_size parameter for a caller to
    forget or mismatch."""
    owner: dict[int, int]
    origin: np.ndarray
    voxel_size: float


def build_voxel_owner_map(
    old_points_xyz: np.ndarray,
    old_instance_ids: np.ndarray,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> VoxelOwnerMap:
    """
    `old_points_xyz` (N,3): world-space positions of every OLD point that
    belongs to at least one instance (unclassified points don't need to be
    passed in at all - they can never transfer anything). `old_instance_ids`
    (N,): a caller-assigned integer key per point identifying WHICH instance
    it belongs to (the same point can only belong to one instance here - if
    a point is in several overlapping instances in the real Layers stack,
    the caller decides which key wins before calling this, same as any
    other point-ownership question in this app).

    Returns a VoxelOwnerMap - {voxel_id: instance_id} for each occupied
    voxel's OWNING instance, plus the origin/voxel_size that produced those
    ids (pass the WHOLE object to transfer_owners, not just `.owner`). Ties
    (two different instances both having old points fall in the exact same
    voxel) go to whichever instance has FEWER total points overall -
    processed largest-instance-first so a later, smaller instance's own
    voxels always overwrite a bigger one's in the shared cell, mirroring the
    "more specific wins overlap" convention used everywhere else layers
    compose (see this module's own docstring).
    """
    if len(old_points_xyz) == 0:
        return VoxelOwnerMap({}, np.zeros(3, dtype=np.float64), voxel_size)

    origin = old_points_xyz.min(axis=0).astype(np.float64) - _ORIGIN_MARGIN_M
    voxel_ids = pack_voxel_ids(old_points_xyz, voxel_size, origin=origin)

    unique_instances, counts = np.unique(old_instance_ids, return_counts=True)
    # Largest first, so the loop below processes smaller instances LATER -
    # their voxel writes land last and win the dict-overwrite for any voxel
    # a bigger instance also touched.
    order = unique_instances[np.argsort(-counts)]

    owner: dict[int, int] = {}
    for inst_id in order:
        inst_voxels = np.unique(voxel_ids[old_instance_ids == inst_id])
        for v in inst_voxels.tolist():
            owner[v] = int(inst_id)
    return VoxelOwnerMap(owner, origin, voxel_size)


def transfer_owners(new_points_xyz: np.ndarray, voxel_owner_map: VoxelOwnerMap) -> np.ndarray:
    """
    For every point in `new_points_xyz`, looks up whether its voxel appears
    in `voxel_owner_map.owner` (see build_voxel_owner_map - MUST be the
    object that call returned, not a bare dict, so the same origin/
    voxel_size are reused) and returns the owning instance id, or -1 if
    that voxel was never classified. Vectorized end to end (no per-point
    Python loop) via sorted-array + searchsorted, the standard technique
    for a fast array-based "dict lookup" at real point-cloud scale - a
    plain per-point dict lookup would not be fast enough here (this runs
    against every point in a freshly (re)loaded scan, potentially tens of
    millions)."""
    n = len(new_points_xyz)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    owner = voxel_owner_map.owner
    if not owner:
        return np.full(n, -1, dtype=np.int64)

    new_voxel_ids = pack_voxel_ids(new_points_xyz, voxel_owner_map.voxel_size, origin=voxel_owner_map.origin)

    keys = np.fromiter(owner.keys(), dtype=np.int64, count=len(owner))
    values = np.fromiter((owner[int(k)] for k in keys), dtype=np.int64, count=len(owner))
    sort_order = np.argsort(keys)
    keys, values = keys[sort_order], values[sort_order]

    pos = np.searchsorted(keys, new_voxel_ids)
    pos_clamped = np.clip(pos, 0, len(keys) - 1)
    matched = keys[pos_clamped] == new_voxel_ids

    result = np.full(n, -1, dtype=np.int64)
    result[matched] = values[pos_clamped[matched]]
    return result
