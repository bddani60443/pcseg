import numpy as np
import pytest

from pcseg.cli import build_parser
from pcseg.core.projection import find_nearest_point_index
from pcseg.core.scan_node import ScanNode


def _make_node(points_xyz: np.ndarray) -> ScanNode:
    n = len(points_xyz)
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=points_xyz.astype(np.int16),
        quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.zeros((n, 3), dtype=np.int8),
        colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=1,
        available_modes=["normals"], row_max=1, col_max=n,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=None,
    )


def test_find_nearest_point_index_exact_match():
    points = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    node = _make_node(points)
    assert find_nearest_point_index(node, (10.0, 0.0, 0.0)) == 1


def test_find_nearest_point_index_approximate():
    points = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    node = _make_node(points)
    assert find_nearest_point_index(node, (8.5, 0.0, 0.0)) == 1


def test_find_nearest_point_index_respects_quant_scale_and_center():
    points = np.array([[0, 0, 0], [10, 0, 0]])
    node = _make_node(points)
    node.quant_scale = 2.0
    node.quant_center = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    # real coords: point0 -> (5,0,0), point1 -> (25,0,0)
    assert find_nearest_point_index(node, (24.0, 0.0, 0.0)) == 1
    assert find_nearest_point_index(node, (6.0, 0.0, 0.0)) == 0


def test_pick_segment_requires_point_xyz_or_point_index():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pick-segment", "-i", "scan.e57", "-o", "out.las"])


def test_pick_segment_point_xyz_and_point_index_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "pick-segment", "-i", "scan.e57", "-o", "out.las",
            "--point-xyz", "1", "2", "3", "--point-index", "5",
        ])


def test_pick_segment_parses_point_xyz():
    parser = build_parser()
    args = parser.parse_args([
        "pick-segment", "-i", "scan.e57", "-o", "out.las", "--point-xyz", "1.5", "2.5", "3.5",
    ])
    assert args.point_xyz == [1.5, 2.5, 3.5]
    assert args.point_index is None
    assert args.keep_all_masks is False


def test_pick_segment_parses_point_index_and_keep_all_masks():
    parser = build_parser()
    args = parser.parse_args([
        "pick-segment", "-i", "scan.e57", "-o", "out.las", "--point-index", "42", "--keep-all-masks",
    ])
    assert args.point_index == 42
    assert args.point_xyz is None
    assert args.keep_all_masks is True
