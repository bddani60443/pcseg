import numpy as np
import pytest

from core.mask_filters import filter_mask_outliers
from core.scan_node import ScanNode


def _make_node(points_xyz: np.ndarray) -> ScanNode:
    """Minimal ScanNode with just enough real data for the mask filters to run on."""
    quant_scale = 1.0
    quant_center = np.zeros(3, dtype=np.float32)
    points_quantized = points_xyz.astype(np.int16)

    return ScanNode(
        scan_id="test",
        scan_name="test",
        file_path="test.e57",
        points_quantized=points_quantized,
        quant_scale=quant_scale,
        quant_center=quant_center,
        normals=np.zeros((len(points_xyz), 3), dtype=np.int8),
        colors_rgb=None,
        colors_intensity=None,
        point_count=len(points_xyz),
        original_point_count=len(points_xyz),
        decimation_step=1,
        available_modes=["normals"],
        row_max=1,
        col_max=len(points_xyz),
        hidden_mask=np.zeros(len(points_xyz), dtype=bool),
        index_mapper=None,  # not needed by the filters under test
    )


def test_filter_mask_outliers_reduces_or_keeps_clean_input():
    rng = np.random.default_rng(42)
    points = rng.normal(loc=0, scale=1.0, size=(200, 3)) * 10  # spread out so scale isn't degenerate
    node = _make_node(points)
    indices = np.arange(len(points))

    result = filter_mask_outliers(node, indices)

    assert len(result) <= len(indices)
    assert len(result) > 0


def test_filter_mask_outliers_passthrough_for_tiny_input():
    node = _make_node(np.array([[0, 0, 0]]))
    result = filter_mask_outliers(node, np.array([0]))
    assert list(result) == [0]


def test_filter_mask_outliers_subsample_branch_for_large_instance():
    # Exceeds _LOF_SUBSAMPLE_THRESHOLD (200k) so this exercises the
    # fit-on-subsample-then-propagate LOF path, not the direct one.
    rng = np.random.default_rng(7)
    n_inliers = 210_000
    inliers = rng.normal(loc=0, scale=1.0, size=(n_inliers, 3)) * 10
    # A small, tight, far-away speck: dense locally (so stage 1's
    # voxel-connectivity wouldn't drop it) but should still read as an
    # outlier relative to the main cloud once propagated.
    outliers = np.tile([500, 500, 500], (20, 1)) + rng.normal(scale=0.05, size=(20, 3))
    points = np.vstack([inliers, outliers])
    outlier_start = n_inliers

    node = _make_node(points)
    indices = np.arange(len(points))

    result = filter_mask_outliers(node, indices, voxel_size=50.0)

    # Stage 1 (voxel connectivity) alone would already drop the far speck
    # since it's a separate connected component - this just confirms the
    # subsample path runs end-to-end without error and doesn't blow up the
    # main cloud.
    assert len(result) <= len(indices)
    assert len(result) > 0
    assert not any(i >= outlier_start for i in result)


def test_filter_mask_outliers_handles_mass_duplicate_points():
    # Real full-resolution scans can hand this filter a huge block of
    # points that all quantize to the EXACT same coordinate (invalid/
    # no-return LiDAR samples, or heavy sweep-view overlap) - live-
    # reproduced on a real "road" instance where 11.5M+ points collapsed
    # to a single unique coordinate. A naive KD-tree-based subsample+
    # propagate approach degrades toward its worst case on data this
    # duplicate-heavy (took ~2950s on the real instance); dedup-first
    # avoids that pathology. This test uses a smaller duplicate block
    # (300k, still above _LOF_SUBSAMPLE_THRESHOLD) so it stays fast in
    # CI while still exercising the same code path.
    n_dupes = 300_000
    duplicated = np.tile([1.0, 2.0, 3.0], (n_dupes, 1))
    node = _make_node(duplicated)
    indices = np.arange(len(duplicated))

    result = filter_mask_outliers(node, indices, voxel_size=50.0)

    # All points are identical, so nothing should be flagged as an outlier.
    assert len(result) == n_dupes
