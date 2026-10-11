"""
Unit tests for core.classification.upsample_classification, using small
hand-built grids (no E57/model dependency) so the exact grid-cell arithmetic
can be checked point-by-point.

Scenario: a 4x4 original scanner grid (rows/cols 0..3). The "low-res" node
kept only even rows/cols (decimation_step=2, so a 2x2 grid of samples, each
representing one 2x2 block of the original grid). The "high-res" node kept
everything (decimation_step=1, all 16 original points). Each low-res sample
sits at the top-left corner of its block, so upsampling should assign every
point in a 2x2 block the label of that block's low-res corner sample.
"""
import numpy as np

from core.classification import upsample_classification
from core.mapping import IndexMapper
from core.scan_node import ScanNode


def _make_node(rows_dec: np.ndarray, cols_dec: np.ndarray, row_max: int, col_max: int,
                decimation_step: int) -> ScanNode:
    """`rows_dec`/`cols_dec` are already-decimated grid coordinates (i.e. original // decimation_step),
    matching what fileio.e57_loader._prepare_grid_info produces."""
    n = len(rows_dec)
    mapper = IndexMapper({"row": rows_dec, "col": cols_dec}, row_max=row_max - 1, col_max=col_max - 1)
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=np.zeros((n, 3), dtype=np.int16),
        quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.zeros((n, 3), dtype=np.int8),
        colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=decimation_step,
        available_modes=["normals"], row_max=row_max, col_max=col_max,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=mapper,
    )


def _low_and_high_res_nodes():
    # Low-res: 2x2 decimated grid (decimation_step=2), one point per cell,
    # in (row, col) order: (0,0), (0,1), (1,0), (1,1).
    low = _make_node(
        rows_dec=np.array([0, 0, 1, 1]), cols_dec=np.array([0, 1, 0, 1]),
        row_max=2, col_max=2, decimation_step=2,
    )
    low.point_classes = np.array([1, 2, 3, 4], dtype=np.uint32)
    low.class_mapping = {0: "unclassified", 1: "A", 2: "B", 3: "C", 4: "D"}

    # High-res: full 4x4 original grid (decimation_step=1), row-major order.
    rows_high = np.repeat(np.arange(4), 4)
    cols_high = np.tile(np.arange(4), 4)
    high = _make_node(rows_dec=rows_high, cols_dec=cols_high, row_max=4, col_max=4, decimation_step=1)

    return low, high


def _label_at(high: ScanNode, propagated: np.ndarray, orig_row: int, orig_col: int) -> int:
    idx = orig_row * 4 + orig_col  # matches the row-major construction above
    return int(propagated[idx])


def test_upsample_assigns_each_original_point_its_blocks_label():
    low, high = _low_and_high_res_nodes()
    propagated, class_mapping = upsample_classification(low, high)

    assert class_mapping == low.class_mapping

    # Top-left 2x2 block (orig rows/cols 0-1) -> low-res sample at decimated (0,0) -> class 1 ("A").
    for r in (0, 1):
        for c in (0, 1):
            assert _label_at(high, propagated, r, c) == 1

    # Top-right block (rows 0-1, cols 2-3) -> decimated (0,1) -> class 2 ("B").
    for r in (0, 1):
        for c in (2, 3):
            assert _label_at(high, propagated, r, c) == 2

    # Bottom-left block (rows 2-3, cols 0-1) -> decimated (1,0) -> class 3 ("C").
    for r in (2, 3):
        for c in (0, 1):
            assert _label_at(high, propagated, r, c) == 3

    # Bottom-right block (rows 2-3, cols 2-3) -> decimated (1,1) -> class 4 ("D").
    for r in (2, 3):
        for c in (2, 3):
            assert _label_at(high, propagated, r, c) == 4


def test_upsample_exact_matches_at_original_sample_points():
    """Points that coincide exactly with a low-res sample must get that sample's own label."""
    low, high = _low_and_high_res_nodes()
    propagated, _ = upsample_classification(low, high)

    assert _label_at(high, propagated, 0, 0) == 1
    assert _label_at(high, propagated, 0, 2) == 2
    assert _label_at(high, propagated, 2, 0) == 3
    assert _label_at(high, propagated, 2, 2) == 4


def test_upsample_output_size_matches_target_not_source():
    low, high = _low_and_high_res_nodes()
    propagated, _ = upsample_classification(low, high)
    assert len(propagated) == high.point_count
    assert high.point_count > low.point_count


def test_upsample_raises_without_source_classification():
    low, high = _low_and_high_res_nodes()
    low.point_classes = None
    try:
        upsample_classification(low, high)
        assert False, "expected ValueError"
    except ValueError:
        pass
