from __future__ import annotations

import numpy as np

from core.scan_node import ScanNode


def assign_classifications(node: ScanNode, indices_by_label: dict[str, np.ndarray]) -> None:
    """
    Populates node.point_classes / node.class_mapping from a {label: point_indices}
    mapping. Shared by every pipeline that produces labeled 3D masks (prompted or
    automatic), so LAS/E57 export always sees the same classification field shape.

    Uses a wide dtype (not the LAS format's eventual 5-/8-bit field width) because
    auto-segmentation routinely proposes more than 255 objects on a dense sweep;
    any clipping to a specific export format's ceiling happens at export time
    (see io.exporters.export_to_las), not here.

    Apply largest labels first, smallest last: where two detections spatially
    overlap (e.g. a "window" mask inside a "building" mask), whichever is
    applied last wins the shared points. Ordering by size means the smaller,
    more specific detection wins instead of whichever happened to come last in
    sweep/detection order (which carries no size information at all). For an
    explicitly user-chosen order instead (e.g. a GUI's reorderable layer
    stack, where the same label may legitimately appear more than once - two
    separate "window" runs, kept as two distinct layers), see compose_layers.
    """
    ordered = sorted(indices_by_label.items(), key=lambda item: len(item[1]), reverse=True)
    compose_layers(node, ordered)


def compose_layers(node: ScanNode, layers: list[tuple[str, np.ndarray]]) -> None:
    """
    Like assign_classifications, but takes an explicit ordered sequence of
    (label, indices) entries instead of a dict, applied in exactly the given
    order - earlier entries first, later entries win any overlap - and allows
    the same label to appear more than once (a dict couldn't represent that),
    which a reorderable multi-layer stack needs: e.g. two separate "window"
    layers from two different runs, kept as two distinct entries rather than
    merged into one, with their relative order still meaningful against
    whatever other layers sit between them in the stack.
    """
    if node.point_classes is None:
        node.point_classes = np.zeros(node.point_count, dtype=np.uint32)
        node.class_mapping = {0: "unclassified"}

    for label, indices in layers:
        existing = list(node.class_mapping.values())
        if label not in existing:
            class_id = max(node.class_mapping.keys()) + 1
            node.class_mapping[class_id] = label
        else:
            class_id = list(node.class_mapping.keys())[existing.index(label)]
        node.point_classes[indices] = class_id


def upsample_classification(source_node: ScanNode, target_node: ScanNode) -> tuple[np.ndarray, dict]:
    """
    Propagates source_node.point_classes (typically computed on a fast, coarsely
    decimated load) onto target_node (a higher-resolution - typically full-res -
    load of the SAME underlying scan) by nearest-grid-cell lookup: every target
    point inherits the label of whichever source grid cell it falls into.

    This is exact and O(N) rather than an approximate spatial-proximity search,
    because a TLS scan's decimation is a regular row/col subsampling - every
    point's exact position in the *original* scanner grid is recoverable from
    its (already stored) panorama-grid coordinate and its node's decimation_step,
    with no need to keep the raw E57 row/col arrays around after loading. Points
    are grouped into source-resolution "cells" of `source.decimation_step` x
    `source.decimation_step` original grid units.

    Both nodes must be independent loads of the same E57 file, differing only in
    `decimation_step` - loading the same scan file at two different steps is the
    caller's responsibility (see cli.py's --export-decimation).
    """
    if source_node.point_classes is None:
        raise ValueError("source_node has no classification to propagate (run a segmentation first).")

    label_grid = source_node.index_mapper.map_to_panorama(source_node.point_classes, default_value=0)
    h_low, w_low = label_grid.shape

    p2p = target_node.index_mapper.point_to_pano
    valid = p2p[:, 0] >= 0

    # Recover each target point's exact position in the *original* scanner grid.
    orig_row = (target_node.row_max - 1 - p2p[:, 0]) * target_node.decimation_step
    orig_col = p2p[:, 1] * target_node.decimation_step

    # Which source-resolution cell does that position fall into?
    dec_row_low = orig_row // source_node.decimation_step
    pano_y_low = np.clip((h_low - 1) - dec_row_low, 0, h_low - 1)
    pano_x_low = np.clip(orig_col // source_node.decimation_step, 0, w_low - 1)

    propagated = np.zeros(target_node.point_count, dtype=label_grid.dtype)
    propagated[valid] = label_grid[pano_y_low[valid], pano_x_low[valid]]

    return propagated, dict(source_node.class_mapping)
