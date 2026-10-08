"""
Regression test for a real bug found while smoke-testing on an actual scan:
reconstruct_panorama_for_channel() pads the image to a full equirectangular
aspect ratio, but IndexMapper.map_to_panorama() (used to project a 3D mask
back onto that image) does not - so overlaying a mask onto a padded panorama
raised an IndexError whenever the raw grid was short enough to need padding
(width > 2x height, which is the common case for a single TLS station scan).
"""
import laspy
import numpy as np

from pcseg.core.mapping import IndexMapper
from pcseg.core.scan_node import ScanNode
from pcseg.core.projection import compute_coverage_grid
from pcseg.io.exporters import (
    compose_layers_preview_image,
    export_coverage_grid_preview,
    export_mask_preview_image,
    export_to_las,
)


def _make_wide_node() -> ScanNode:
    # A grid wide enough (w > 2*h) that reconstruct_panorama_for_channel() must pad.
    h, w = 4, 20
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows = rows.ravel().astype(np.int32)
    cols = cols.ravel().astype(np.int32)
    n = h * w

    mapper = IndexMapper({"row": rows, "col": cols}, row_max=h - 1, col_max=w - 1)

    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=np.zeros((n, 3), dtype=np.int16),
        quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.tile(np.array([0, 0, 127], dtype=np.int8), (n, 1)),
        colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=1,
        available_modes=["normals"], row_max=h, col_max=w,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=mapper,
    )


def test_export_mask_preview_image_on_a_padded_panorama(tmp_path):
    node = _make_wide_node()
    out_path = tmp_path / "preview.png"

    indices_by_label = {"thing": np.array([0, 1, 2])}
    colors_by_label = {"thing": (255, 0, 0)}

    export_mask_preview_image(node, str(out_path), indices_by_label, colors_by_label, base_channel="normals")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_export_coverage_grid_preview_draws_every_computed_tile(tmp_path):
    node = _make_wide_node()
    out_path = tmp_path / "coverage.png"

    valid_mask = node.index_mapper.pano_map >= 0
    grid = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2)
    assert len(grid) > 0  # sanity: the fixture's grid actually has data to sweep

    msg = export_coverage_grid_preview(node, str(out_path), grid, base_channel="normals")

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert str(len(grid)) in msg


def test_export_mask_preview_image_lets_smaller_mask_win_pixel_overlap(tmp_path):
    """
    Regression test: overlapping labels used to be composited in dict-iteration
    order (i.e. sweep/detection order), so a large background label (e.g.
    "building") could visually paint over a smaller, more specific one (e.g.
    "window") purely by chance. Compositing must instead go largest-first,
    smallest-last, so a smaller label always wins the pixels it shares with a
    bigger one - matching what assign_classifications does for the exported
    point cloud's classification field.
    """
    node = _make_wide_node()
    out_path = tmp_path / "preview.png"

    small_indices = np.array([0])  # a single point
    large_indices = np.arange(node.point_count)  # every point, including index 0

    # Dict order deliberately puts the *large* label last, so a naive
    # "last-in-dict-wins" implementation would make this test fail.
    indices_by_label = {"small": small_indices, "large": large_indices}
    colors_by_label = {"small": (0, 200, 0), "large": (200, 0, 0)}

    export_mask_preview_image(node, str(out_path), indices_by_label, colors_by_label, base_channel="normals")

    import cv2
    saved = cv2.imread(str(out_path))  # BGR

    mask_2d = node.index_mapper.map_to_panorama(
        np.isin(np.arange(node.point_count), small_indices), default_value=False
    )
    ys, xs = np.where(mask_2d)
    assert len(ys) == 1
    pixel_bgr = saved[ys[0], xs[0]].astype(np.float32)

    small_bgr = np.array([0, 200, 0], dtype=np.float32)  # RGB (0,200,0) -> BGR unchanged (symmetric)
    large_bgr = np.array([0, 0, 200], dtype=np.float32)  # RGB (200,0,0) -> BGR (0,0,200)

    assert np.linalg.norm(pixel_bgr - small_bgr) < np.linalg.norm(pixel_bgr - large_bgr)


def test_export_mask_preview_image_overlap_pixel_matches_winner_only_render(tmp_path):
    """
    Regression test: overlapping labels used to be composited by blending onto
    the already-blended overlay for every label that touched a pixel, so an
    overlap pixel ended up a smeared mix of every label that touched it (each
    diluted further by whichever came after) instead of a single, clean color
    for the pixel's one winning label. A pixel where a winning (smaller) label
    overlaps a losing (larger) one must now render IDENTICALLY to a scene where
    only the winning label was ever present - proving zero color leaks in from
    the loser, not just "mostly" the winner's color.
    """
    node = _make_wide_node()
    small_indices = np.array([0])
    large_indices = np.arange(node.point_count)

    out_both = tmp_path / "both.png"
    export_mask_preview_image(
        node, str(out_both),
        {"small": small_indices, "large": large_indices},
        {"small": (0, 200, 0), "large": (200, 0, 0)},
        base_channel="normals",
    )

    out_small_only = tmp_path / "small_only.png"
    export_mask_preview_image(
        node, str(out_small_only),
        {"small": small_indices}, {"small": (0, 200, 0)},
        base_channel="normals",
    )

    import cv2
    both = cv2.imread(str(out_both))
    small_only = cv2.imread(str(out_small_only))

    mask_2d = node.index_mapper.map_to_panorama(
        np.isin(np.arange(node.point_count), small_indices), default_value=False
    )
    ys, xs = np.where(mask_2d)
    assert len(ys) == 1
    assert np.array_equal(both[ys[0], xs[0]], small_only[ys[0], xs[0]])


def test_compose_layers_preview_image_respects_explicit_order_not_size(tmp_path):
    """
    Unlike export_mask_preview_image, compose_layers_preview_image must NOT
    reorder by size - a GUI layer stack's explicit order is the whole point.
    Put the larger array LAST here (the opposite of size-sorting) and confirm
    it still wins the shared pixel, proving order alone decided it.
    """
    node = _make_wide_node()
    out_path = tmp_path / "layers.png"
    shared = np.array([0])

    layers = [
        ("small", shared, (0, 200, 0)),
        ("large", np.arange(node.point_count), (200, 0, 0)),
    ]
    compose_layers_preview_image(node, str(out_path), layers, base_channel="normals")

    import cv2
    saved = cv2.imread(str(out_path))
    mask_2d = node.index_mapper.map_to_panorama(
        np.isin(np.arange(node.point_count), shared), default_value=False
    )
    ys, xs = np.where(mask_2d)
    large_bgr = np.array([0, 0, 200], dtype=np.float32)  # RGB (200,0,0) -> BGR
    small_bgr = np.array([0, 200, 0], dtype=np.float32)
    pixel = saved[ys[0], xs[0]].astype(np.float32)
    assert np.linalg.norm(pixel - large_bgr) < np.linalg.norm(pixel - small_bgr)


def _make_flat_node(n: int) -> ScanNode:
    """A minimal ScanNode for export_to_las tests - no panorama/index_mapper needed."""
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=np.zeros((n, 3), dtype=np.int16),
        quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.zeros((n, 3), dtype=np.int8),
        colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=1,
        available_modes=["normals"], row_max=1, col_max=n,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=None,
    )


def test_export_to_las_with_many_classes_does_not_overflow(tmp_path):
    """
    Regression test for a real bug found while smoke-testing auto-segment on an
    actual scan: LAS point format 3's classification field is bit-packed to 5 bits
    (max 31), but auto-segmentation routinely finds 100+ distinct objects, so
    `las.classification = classes` raised laspy.OverflowError. Export must instead
    switch to point format 7 (full 8-bit classification + RGB) once class count
    exceeds 31 - format 6 alone would fix the overflow but silently drop RGB support.
    """
    n = 300
    node = _make_flat_node(n)
    node.point_classes = (np.arange(n) % 194).astype(np.uint8)  # 194 distinct classes, mirrors the real failure
    node.class_mapping = {i: f"object_{i:04d}" for i in range(194)}

    out_path = tmp_path / "many_classes.las"
    message = export_to_las(node, str(out_path))

    assert "Saved LAS" in message
    with laspy.open(str(out_path)) as f:
        assert f.header.point_format.id == 7
    read_back = laspy.read(str(out_path))
    assert np.array_equal(np.asarray(read_back.classification), node.point_classes)


def test_export_to_las_with_few_classes_keeps_point_format_3(tmp_path):
    n = 10
    node = _make_flat_node(n)
    node.point_classes = np.array([0, 1, 2] * 3 + [0], dtype=np.uint8)
    node.class_mapping = {0: "unclassified", 1: "door", 2: "window"}

    out_path = tmp_path / "few_classes.las"
    export_to_las(node, str(out_path))

    with laspy.open(str(out_path)) as f:
        assert f.header.point_format.id == 3
