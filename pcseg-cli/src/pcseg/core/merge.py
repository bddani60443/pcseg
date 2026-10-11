from __future__ import annotations

import numpy as np

ViewKey = tuple[int, int]


def merge_by_3d_iou(
    index_arrays: list[np.ndarray],
    iou_threshold: float = 0.5,
    overlap_coeff_threshold: float | None = None,
    touches_border: list[bool] | None = None,
    border_overlap_threshold: float = 0.6,
    points_3d: np.ndarray | None = None,
    surface_merge_distance: float | None = None,
    spatial_sample_size: int = 1500,
    view_keys: list[ViewKey] | list[frozenset[ViewKey]] | None = None,
    n_theta: int = 0,
    return_view_keys: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[frozenset[ViewKey]]]:
    """
    Greedy clustering (largest-first) of point-index arrays.

    Default rule: symmetric Jaccard IoU against a cluster's current union,
    merging once it reaches `iou_threshold`. This protects small,
    fully-contained objects (e.g. a window) from being swallowed by a much
    bigger overlapping mask (its wall), and needs no view-gating since
    masks from unrelated views essentially never share point indices.

    Optional rules, each layered on top of Jaccard IoU:
    - `overlap_coeff_threshold`: also merges on overlap coefficient
      (intersection / smaller set's size) alone - catches the same real
      object captured at very different sizes across views, where Jaccard
      under-merges. Not yet validated against a real scan, so opt-in.
    - `touches_border` + `border_overlap_threshold`: the same
      overlap-coefficient rule, but only between two masks that both
      touched their view's crop edge - catches one physical surface cut
      into near-disjoint fragments by the sweep's view boundaries, which
      can have zero index overlap.
    - `points_3d` + `surface_merge_distance`: merges two border-touching
      masks with zero index overlap if any of their (subsampled) points
      lie within this 3D distance - catches a surface resampled into
      disjoint point sets across views. Unlike the other rules this isn't
      inherently local (ground near the scanner is close to every view),
      so it relies on `view_keys` gating.

    `view_keys[i]` restricts the border-overlap and spatial rules to only fire
    between grid-adjacent views (4-connected on the sweep's theta/phi grid,
    wrapping at the 360/0 seam via `n_theta`) - i.e. actually neighboring pieces
    of the sweep, not any two views that happen to share a bit of nearby ground.
    Each entry is normally a single `(theta_idx, phi_idx)` tuple (one raw mask
    came from exactly one view); it may also be a set of such tuples, which is
    what lets this same function run a second "merge the merges" pass: call it
    once, take `return_view_keys=True`'s per-cluster key sets (each cluster's
    combined view footprint) as the *next* call's `view_keys`, and clusters whose
    footprints are adjacent - even if no single original view pair was - become
    eligible to merge further. Leave `view_keys` unset to disable this gating
    (every pair is eligible, as before).
    """
    if not index_arrays:
        return ([], [], []) if return_view_keys else []

    n = len(index_arrays)
    touches_border = touches_border if touches_border is not None else [False] * n
    use_spatial = points_3d is not None and surface_merge_distance is not None
    gate_by_view = view_keys is not None
    norm_keys = [_as_key_set(k) for k in view_keys] if gate_by_view else None
    rng = np.random.default_rng(0)

    sets = [set(arr.tolist()) for arr in index_arrays]
    order = sorted(range(n), key=lambda i: len(sets[i]), reverse=True)

    clusters: list[set[int]] = []
    cluster_is_border: list[bool] = []
    cluster_view_keys: list[frozenset[ViewKey]] = []

    for i in order:
        candidate = sets[i]
        candidate_is_border = touches_border[i]
        candidate_keys = norm_keys[i] if gate_by_view else None
        best_cluster_idx = None
        best_score = 0.0

        for cluster_idx, cluster in enumerate(clusters):
            both_border = candidate_is_border and cluster_is_border[cluster_idx]
            border_rules_eligible = both_border and (
                not gate_by_view or _key_sets_adjacent(candidate_keys, cluster_view_keys[cluster_idx], n_theta)
            )
            intersection = len(candidate & cluster)

            score = None
            if intersection > 0:
                iou = intersection / len(candidate | cluster)
                if iou >= iou_threshold:
                    score = iou

                if overlap_coeff_threshold is not None:
                    overlap_coeff = intersection / min(len(candidate), len(cluster))
                    if overlap_coeff >= overlap_coeff_threshold and (score is None or overlap_coeff > score):
                        score = overlap_coeff

                if score is None and border_rules_eligible:
                    overlap_coeff = intersection / min(len(candidate), len(cluster))
                    if overlap_coeff >= border_overlap_threshold:
                        score = overlap_coeff
            elif border_rules_eligible and use_spatial:
                a = _subsample(candidate, spatial_sample_size, rng)
                b = _subsample(cluster, spatial_sample_size, rng)
                if _spatially_adjacent(a, b, points_3d, surface_merge_distance):
                    score = 1e-6  # weakest possible match: only wins if nothing else does

            if score is not None and score > best_score:
                best_score = score
                best_cluster_idx = cluster_idx

        if best_cluster_idx is not None:
            clusters[best_cluster_idx] |= candidate
            cluster_is_border[best_cluster_idx] = cluster_is_border[best_cluster_idx] or candidate_is_border
            if gate_by_view:
                cluster_view_keys[best_cluster_idx] |= candidate_keys
        else:
            clusters.append(set(candidate))
            cluster_is_border.append(candidate_is_border)
            cluster_view_keys.append(candidate_keys if gate_by_view else frozenset())

    merged = [np.array(sorted(c), dtype=np.int64) for c in clusters]
    return (merged, cluster_view_keys, cluster_is_border) if return_view_keys else merged


def merge_by_3d_iou_iterative(index_arrays: list[np.ndarray], max_rounds: int = 3, **kwargs) -> list[np.ndarray]:
    """
    Repeats merge_by_3d_iou, feeding each round's resulting clusters (and
    their combined view-key footprints) back in as the next round's input,
    so clusters that only become view-adjacent after growing in an earlier
    round can keep merging. Stops early once a round doesn't reduce the
    cluster count (converged).

    Each cluster's border-eligibility carries over as-is - a cluster stays
    "border" only if some original fragment inside it actually touched a
    crop edge. Forcing every cluster to "border" after round 1 was tried
    and made every already-closed object eligible for the loose merge
    rules in later rounds, collapsing distinct objects together - so that's
    deliberately avoided.
    """
    touches_border = kwargs.pop("touches_border", None)
    view_keys = kwargs.pop("view_keys", None)

    merged, keys, is_border = merge_by_3d_iou(index_arrays, touches_border=touches_border, view_keys=view_keys,
                                               return_view_keys=True, **kwargs)

    for _ in range(max_rounds - 1):
        if len(merged) <= 1:
            break
        new_merged, new_keys, new_is_border = merge_by_3d_iou(
            merged, touches_border=is_border, view_keys=keys, return_view_keys=True, **kwargs,
        )
        if len(new_merged) == len(merged):
            break
        merged, keys, is_border = new_merged, new_keys, new_is_border

    return merged


def mask_touches_border(mask_2d: np.ndarray, margin: int = 2) -> bool:
    """True if any True pixel of a 2D mask falls within `margin` pixels of its crop's edge."""
    m = max(1, margin)
    return bool(
        mask_2d[:m, :].any() or mask_2d[-m:, :].any() or mask_2d[:, :m].any() or mask_2d[:, -m:].any()
    )


def _as_key_set(key: ViewKey | frozenset[ViewKey] | set[ViewKey]) -> frozenset[ViewKey]:
    return frozenset(key) if isinstance(key, (set, frozenset)) else frozenset([key])


def _grid_adjacent(a: ViewKey, b: ViewKey, n_theta: int) -> bool:
    """4-connected adjacency on the (theta_idx, phi_idx) sweep grid, wrapping theta at the seam."""
    a_theta, a_phi = a
    b_theta, b_phi = b
    if a_phi == b_phi:
        d = abs(a_theta - b_theta)
        return d == 0 or d == 1 or (n_theta > 0 and d == n_theta - 1)
    if a_theta == b_theta:
        return abs(a_phi - b_phi) <= 1
    return False


def _key_sets_adjacent(keys_a: frozenset[ViewKey], keys_b: frozenset[ViewKey], n_theta: int) -> bool:
    return any(_grid_adjacent(a, b, n_theta) for a in keys_a for b in keys_b)


def _subsample(index_set: set[int], sample_size: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.fromiter(index_set, dtype=np.int64, count=len(index_set))
    if len(arr) > sample_size:
        arr = rng.choice(arr, size=sample_size, replace=False)
    return arr


def _spatially_adjacent(a_idx: np.ndarray, b_idx: np.ndarray, points_3d: np.ndarray, max_distance: float) -> bool:
    from scipy.spatial import cKDTree

    tree = cKDTree(points_3d[b_idx])
    dists, _ = tree.query(points_3d[a_idx], k=1)
    return bool(dists.min() <= max_distance)
