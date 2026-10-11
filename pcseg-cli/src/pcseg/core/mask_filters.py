from __future__ import annotations

import logging
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.neighbors import LocalOutlierFactor

from pcseg.core.scan_node import ScanNode

_LOF_SUBSAMPLE_THRESHOLD = 200_000


def _to_points_3d(node: ScanNode, indices: np.ndarray) -> np.ndarray:
    return node.points_quantized[indices].astype(np.float32) * node.quant_scale + node.quant_center


def filter_mask_outliers(node: ScanNode, indices: np.ndarray, voxel_size: float = 0.15,
                          k: int = 16) -> np.ndarray:
    """
    Two-stage noise filter:

    Stage 1 - Voxel connected components: discretize into a voxel grid, keep
        the largest 26-connected component (by total POINT count, not voxel
        count). Vectorized via a KD-tree pair query feeding
        scipy.sparse.csgraph.connected_components.
    Stage 2 - Local Outlier Factor (LOF): flags a point as an outlier by
        comparing its own local density against its k nearest neighbors',
        rather than a single global threshold over the whole instance. This
        replaced an earlier global-threshold Statistical Outlier Removal
        (SOR) pass, which over-removed legitimate points on large
        continuous surfaces (e.g. the far, sparser end of a road) because a
        single global density threshold can't distinguish a surface that's
        legitimately sparser in places from isolated speckle.

        Points are deduplicated to unique coordinates before any KD-tree
        work (duplicate decisions are broadcast back afterwards) - real
        scans can contain large numbers of points that quantize to
        identical coordinates, which otherwise degrades KD-tree queries
        toward their worst case. Above `_LOF_SUBSAMPLE_THRESHOLD` unique
        points, LOF fits on a random subsample instead of every point, and
        the rest inherit their nearest subsampled point's decision, since
        LOF's own cost grows super-linearly with n.

    Thin wrapper around `_filter_outliers_core` - extracts this instance's own
    real-world points from `node` (the only thing that needs `node` at all)
    and hands off to the core algorithm below, which has no ScanNode
    dependency and is what `segmentation_service._run_outlier_filtering`
    calls directly on its worker threads for parallel filtering (see its own
    docstring).
    """
    if len(indices) < 2:
        return indices
    points_3d = _to_points_3d(node, indices)
    return _filter_outliers_core(points_3d, indices, voxel_size, k)


def _filter_outliers_core(points_3d: np.ndarray, indices: np.ndarray, voxel_size: float = 0.15,
                           k: int = 16) -> np.ndarray:
    """The actual two-stage filter body - see `filter_mask_outliers`'s own
    docstring for the full algorithm rationale. Takes ALREADY-EXTRACTED
    `points_3d` (same convention as `_to_points_3d`'s own output) instead of
    a ScanNode + indices pair, so a parallel caller only needs each
    instance's own small points_3d/indices pair, never the whole scan."""
    if len(indices) < 2:
        return indices

    original_count = len(indices)
    verbose = original_count > 20_000
    t_start = time.perf_counter()

    voxel_coords = np.floor(points_3d / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    num_voxels = len(unique_voxels)
    if verbose:
        logging.info(f"Outlier filter: {original_count} points -> {num_voxels} voxels "
                     f"({time.perf_counter() - t_start:.1f}s so far) - starting connectivity...")

    if num_voxels > 1:
        tree = cKDTree(unique_voxels)
        pairs = tree.query_pairs(r=1.8, output_type="ndarray")
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(num_voxels, num_voxels))
        n_components, labels = connected_components(graph, directed=False)

        # Largest by total POINT count, not voxel count (a sparse but
        # spatially large component shouldn't lose to a small dense one).
        points_per_component = np.bincount(labels, weights=counts, minlength=n_components)
        best_label = np.argmax(points_per_component)

        stage1_mask = (labels == best_label)[inverse_indices]
        indices = indices[stage1_mask]
        points_3d = points_3d[stage1_mask]

    if verbose:
        logging.info(f"Outlier filter: connectivity done ({time.perf_counter() - t_start:.1f}s so far), "
                      f"{len(indices)} points remain - starting LOF (k={min(k, max(len(indices) - 1, 0))})...")

    effective_k = min(k, len(indices) - 1)
    if effective_k >= 2:

        unique_pts, unique_inverse = np.unique(points_3d, axis=0, return_inverse=True)
        n_unique = len(unique_pts)
        unique_effective_k = min(k, n_unique - 1)

        if unique_effective_k >= 2:
            if n_unique > _LOF_SUBSAMPLE_THRESHOLD:

                rng = np.random.default_rng(0)
                sample_idx = rng.choice(n_unique, size=_LOF_SUBSAMPLE_THRESHOLD, replace=False)
                sample_pts = unique_pts[sample_idx]
                sample_k = min(k, len(sample_pts) - 1)
                lof = LocalOutlierFactor(n_neighbors=sample_k)
                sample_is_inlier = lof.fit_predict(sample_pts) == 1

                sample_tree = cKDTree(sample_pts, balanced_tree=False, compact_nodes=False)
                _, nearest = sample_tree.query(unique_pts, k=1, workers=-1)
                unique_is_inlier = sample_is_inlier[nearest]
            else:
                lof = LocalOutlierFactor(n_neighbors=unique_effective_k)
                unique_is_inlier = lof.fit_predict(unique_pts) == 1
            is_inlier = unique_is_inlier[unique_inverse]
            indices = indices[is_inlier]

    removed = original_count - len(indices)
    if removed > 0 or verbose:
        logging.info(f"Outlier filter: {original_count} -> {len(indices)} points "
                     f"({removed} removed, {time.perf_counter() - t_start:.1f}s total)")

    return indices
