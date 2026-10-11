"""
Density-adaptive rendering decimation.

Self-contained on purpose: pure numpy, ShaderPointCloudItem only touches it through
_upload_ebo() (one call site, replacing the old uniform `visible[::step]`
stride-slice), gated by the exact same `view_decimation` control that's
always been there - the UI knob and its meaning ("how aggressively to
subsample for interactive performance") don't change, only HOW the points to
drop get chosen.

Why: a uniform stride keeps the same FRACTION of points everywhere, but real
TLS scans are wildly non-uniform in local density - many overlapping sweep
views can pack an enormous number of points into a small physical area (a
live-measured real example elsewhere in this codebase: one "road" detection
had 11.6M points collapse into ~1,200 voxels at 0.15m - an average of ~9,700
points per voxel), while other areas (grazing angles, scan edges) are
comparatively sparse. A uniform stride wastes render budget re-showing
near-duplicate points from the dense areas while contributing nothing extra
visually, and can't spend that budget on the sparse areas that would
actually benefit from staying at full detail. This bins points into a voxel
grid and caps how many points ANY single voxel is allowed to contribute,
computed so the TOTAL kept count matches what the requested decimation step
would already give - a voxel already under the cap is untouched (full detail
kept exactly where it was already sparse), a voxel over the cap is thinned
down to it (exactly where thinning is actually free, visually).
"""
from __future__ import annotations

import numpy as np

# Deliberately independent of mask_filters.py's own 0.15m voxel default,
# even though it's the same order of magnitude - that one is tuned for
# outlier-filtering CORRECTNESS on a segmented instance; this one is a
# rendering-only heuristic for "how close together do points have to be
# before they're visually redundant on screen," a different (looser)
# concern with no need to stay numerically tied to the other value.
DEFAULT_VOXEL_SIZE = 0.15

# Bits per axis for packing (offset, non-negative) voxel coordinates into a
# single sortable int64 key - see pack_voxel_ids. 21 bits = up to ~2.1M
# distinct cells per axis, i.e. a scan could span over 300km at the default
# 0.15m voxel size before this becomes a real constraint on any one axis -
# effectively unlimited for any real TLS/handheld scan.
_BITS_PER_AXIS = 21
_AXIS_LIMIT = 1 << _BITS_PER_AXIS


def pack_voxel_ids(points_xyz: np.ndarray, voxel_size: float = DEFAULT_VOXEL_SIZE,
                    origin: np.ndarray | None = None) -> np.ndarray:
    """
    One int64 per point, EXACTLY identifying its voxel (no hash collisions,
    unlike a fixed-size hash table) - bit-packs the 3 (offset to
    non-negative) integer grid coordinates into a single sortable key. This
    is what makes grouping points by voxel a single `np.argsort` on ONE
    int64 column instead of `np.unique(voxel_coords, axis=0, ...)`'s
    multi-column lexicographic comparison - live-measured elsewhere in this
    codebase (mask_filters.py) that the axis=0 form alone took ~10s on
    "just" 11.5M points; a real full-resolution load here can be several
    times that many, where the difference matters for something meant to
    run on an interactive UI action, not a one-off filter.

    `origin` defaults to `points_xyz.min(axis=0)` (the original, single-call
    behavior every existing caller in this module relies on: voxel ids only
    ever need to be self-consistent WITHIN that one call). Pass it
    EXPLICITLY instead whenever voxel ids from two SEPARATE calls need to be
    compared against each other (e.g. label_transfer.py, matching voxels
    between an OLD and a NEW, independently-loaded point set) - each call
    computing its own local min would silently misalign the two id spaces
    (two physically different voxels, one from each point set, landing on
    the same packed id purely because they happen to sit at the same
    OFFSET from their own respective set's minimum - a real bug live-caught
    via a deliberately spatially-disjoint round-trip test before this
    parameter existed: two point sets with nothing in common still "matched"
    every point, silently, no exception, just wrong).

    Clamps to _AXIS_LIMIT per axis rather than raising - an all but
    impossible extent/voxel-size combination shouldn't crash the viewer,
    just silently lose exactness (rare hash-collision-like behavior at the
    clamped extreme) in a case realistic scans will never actually hit.
    """
    if len(points_xyz) == 0:
        return np.zeros(0, dtype=np.int64)

    if origin is None:
        origin = points_xyz.min(axis=0)
    voxel_coords = np.floor((points_xyz - origin) / voxel_size).astype(np.int64)
    voxel_coords = np.clip(voxel_coords, 0, _AXIS_LIMIT - 1)

    return (
        (voxel_coords[:, 0] << (2 * _BITS_PER_AXIS))
        | (voxel_coords[:, 1] << _BITS_PER_AXIS)
        | voxel_coords[:, 2]
    )


def _solve_cap(counts: np.ndarray, target_total: int) -> int:
    """Largest integer cap C such that capping every voxel's point count at
    C keeps at most `target_total` points overall - i.e. the loosest cap
    (touches the FEWEST voxels, thins each over-cap voxel as LITTLE as
    possible) that still hits the target. Binary search over `counts`
    ALONE (per-voxel counts, not per-point) - cheap regardless of the real
    point count, since the number of occupied voxels is always far smaller
    than the number of points."""
    if len(counts) == 0 or target_total <= 0:
        return 0
    total = int(counts.sum())
    if target_total >= total:
        return int(counts.max())

    lo, hi = 0, int(counts.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        kept = int(np.minimum(counts, mid).sum())
        if kept <= target_total:
            lo = mid
        else:
            hi = mid - 1
    return lo


def select_density_adaptive(voxel_ids: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Boolean mask (len(voxel_ids),), True for points to KEEP - `keep_fraction`
    of the input overall (matching what a uniform 1/N stride would have kept
    for the same N), but distributed by thinning dense voxels down to a
    shared cap rather than a uniform stride over the whole array. A voxel at
    or under the cap keeps every one of its points; over the cap, keeps
    exactly `cap` of them (the first `cap` in voxel_ids' own original
    order - deterministic, not random, so the same input always produces
    the same output, and no RNG state to manage).

    `keep_fraction >= 1.0` returns an all-True mask (no decimation) without
    doing any of the grouping work below - the common case (View detail = 1,
    the default) should cost nothing extra over the old behavior.
    """
    n = len(voxel_ids)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if keep_fraction >= 1.0:
        return np.ones(n, dtype=bool)

    target_total = int(round(n * keep_fraction))
    if target_total <= 0:
        return np.zeros(n, dtype=bool)

    order = np.argsort(voxel_ids, kind="stable")
    sorted_ids = voxel_ids[order]
    _unique_ids, start_positions, counts = np.unique(sorted_ids, return_index=True, return_counts=True)

    cap = _solve_cap(counts, target_total)
    if cap >= int(counts.max()):
        return np.ones(n, dtype=bool)  # target isn't actually tighter than "keep everything"
    # A real edge case, not just theoretical: if the scan is already close
    # to one point per voxel almost everywhere (little redundancy for this
    # technique to exploit - e.g. an already-sparse/subsampled load, or a
    # genuinely uniform-density scan), the number of OCCUPIED voxels alone
    # can exceed target_total, and _solve_cap's binary search correctly
    # concludes no cap >= 1 hits the target - returning 0. Rather than
    # "cap=0 keeps literally nothing" (an empty/blank view is a far worse
    # failure than showing more points than requested), the graceful
    # degradation is a floor of 1: every occupied voxel keeps AT LEAST one
    # point, even if that overshoots target_total - live-caught via a
    # stress test (20M points spread thin over a large area at the default
    # voxel size, asking for a keep-fraction well below what the data's own
    # redundancy could actually supply).
    cap = max(cap, 1)

    group_start_for_each = np.repeat(start_positions, counts)
    rank_in_group = np.arange(n) - group_start_for_each
    per_voxel_cap = np.repeat(np.minimum(counts, cap), counts)
    keep_sorted = rank_in_group < per_voxel_cap

    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[order[keep_sorted]] = True
    return keep_mask
