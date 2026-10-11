"""
Regression test for a real bug found while smoke-testing a denser auto-segment
sweep: node.point_classes was allocated as uint8, so assigning the 256th distinct
label raised OverflowError before export even ran. Assignment must use a wide
dtype; only the LAS exporter's format-specific ceiling should clip values.
"""
import numpy as np

from core.classification import assign_classifications, compose_layers
from core.scan_node import ScanNode


def _make_node(n: int) -> ScanNode:
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


def test_assign_classifications_handles_more_than_255_labels():
    n = 400
    node = _make_node(n)
    indices_by_label = {f"object_{i:04d}": np.array([i]) for i in range(n)}

    assign_classifications(node, indices_by_label)  # must not raise OverflowError

    assert node.point_classes.dtype != np.uint8
    assert node.point_classes.max() >= 300
    assert len(node.class_mapping) == n + 1  # + the implicit "unclassified" 0 class


def test_assign_classifications_reuses_class_id_for_repeated_label():
    node = _make_node(4)
    assign_classifications(node, {"door": np.array([0, 1])})
    assign_classifications(node, {"door": np.array([2, 3])})

    assert len(node.class_mapping) == 2  # unclassified + door, not a duplicate "door" entry
    assert node.point_classes[0] == node.point_classes[2]


def test_assign_classifications_lets_smaller_label_win_overlap_regardless_of_dict_order():
    """
    A smaller/more specific detection (e.g. "window") commonly spatially overlaps
    a larger one (e.g. "building") on the same points. Overlapping points must end
    up classified as the smaller label, not whichever label happened to be last in
    dict iteration order (which reflects sweep/detection order, carrying no size
    information) - verified in both insertion orders so this isn't accidental.
    """
    shared = np.array([5])
    small = {"window": shared, "building": np.arange(100)}
    large_first = {"building": np.arange(100), "window": shared}

    node_a = _make_node(200)
    assign_classifications(node_a, small)
    node_b = _make_node(200)
    assign_classifications(node_b, large_first)

    window_id_a = [k for k, v in node_a.class_mapping.items() if v == "window"][0]
    window_id_b = [k for k, v in node_b.class_mapping.items() if v == "window"][0]

    assert node_a.point_classes[5] == window_id_a
    assert node_b.point_classes[5] == window_id_b


def test_compose_layers_applies_in_the_given_order_not_by_size():
    """
    Unlike assign_classifications, compose_layers must NOT reorder by size -
    a GUI layer stack's explicit user order is the whole point. Put the
    LARGER array last here (the opposite of what size-sorting would do) and
    confirm it still wins the overlap, proving order alone decided it.
    """
    node = _make_node(200)
    small_first = [("small", np.array([5])), ("large", np.arange(100))]

    compose_layers(node, small_first)

    large_id = [k for k, v in node.class_mapping.items() if v == "large"][0]
    assert node.point_classes[5] == large_id


def test_compose_layers_keeps_two_layers_with_the_same_label_distinct():
    """
    A dict can't represent "two separate 'window' layers" (duplicate keys
    collapse) - compose_layers takes a list specifically so this works: both
    entries share one class id (same label = same exported class), but each
    contributes its own indices at its own position in the stack.
    """
    node = _make_node(10)
    layers = [("window", np.array([0, 1])), ("building", np.array([2, 3])), ("window", np.array([4, 5]))]

    compose_layers(node, layers)

    window_id = [k for k, v in node.class_mapping.items() if v == "window"][0]
    assert len(node.class_mapping) == 3  # unclassified + window + building, not 4
    assert node.point_classes[0] == window_id
    assert node.point_classes[4] == window_id


def test_compose_layers_matches_assign_classifications_for_a_dict_input():
    """assign_classifications(node, d) must behave exactly like
    compose_layers(node, size-sorted d.items()) - it's defined in terms of it now."""
    node_a = _make_node(200)
    node_b = _make_node(200)
    indices_by_label = {"window": np.array([5]), "building": np.arange(100)}

    assign_classifications(node_a, indices_by_label)
    compose_layers(node_b, sorted(indices_by_label.items(), key=lambda kv: len(kv[1]), reverse=True))

    assert node_a.class_mapping == node_b.class_mapping
    assert np.array_equal(node_a.point_classes, node_b.point_classes)
