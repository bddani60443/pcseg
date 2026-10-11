from __future__ import annotations

import json
import logging
import os

import cv2
import laspy
import numpy as np

from core.projection import Channel, reconstruct_panorama_for_channel
from core.scan_node import ScanNode


def export_to_las(node: ScanNode, output_filepath: str, mask: np.ndarray | None = None, progress_callback=None) -> str:
    """Exports the (optionally masked) point cloud to a standard .las file."""
    if progress_callback:
        progress_callback(10, "De-quantizing coordinates...")

    pts_float = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center
    colors = node.colors_rgb
    classes = node.point_classes

    if mask is not None:
        if len(mask) == 0:
            return "Export skipped: mask selects 0 points."
        pts_float = pts_float[mask, :].copy()
        if colors is not None:
            colors = colors[mask, :].copy()
        if classes is not None:
            classes = classes[mask].copy()

    if len(pts_float) == 0:
        return "Export skipped: 0 points left after masking."

    if progress_callback:
        progress_callback(40, "Creating LAS header...")

    # Point format 3's classification field is bit-packed (5 bits -> max 31): fine
    # for a handful of prompt labels, but auto-segmentation can produce hundreds of
    # objects. Fall back to format 7 (LAS 1.4, full 8-bit classification + RGB,
    # max 255) whenever that ceiling would otherwise be exceeded - format 6 would
    # also give 8-bit classification but silently drops RGB support entirely.
    n_classes = len(node.class_mapping) if node.class_mapping else 0
    point_format, version, max_class = (7, "1.4", 255) if n_classes > 31 else (3, "1.2", 31)

    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.min(pts_float, axis=0)

    las = laspy.LasData(header)
    las.x = pts_float[:, 0]
    las.y = pts_float[:, 1]
    las.z = pts_float[:, 2]

    if colors is not None:
        las.red = colors[:, 0].astype(np.uint16) * 256
        las.green = colors[:, 1].astype(np.uint16) * 256
        las.blue = colors[:, 2].astype(np.uint16) * 256

    if classes is not None:
        overflow = classes > max_class
        if overflow.any():
            logging.warning(f"{overflow.sum()} points have class IDs beyond {max_class}; clipping to {max_class}.")
            classes = np.minimum(classes, max_class)
        las.classification = classes

    if progress_callback:
        progress_callback(80, "Writing file to disk...")

    las.write(output_filepath)

    if node.class_mapping is not None:
        json_path = output_filepath.rsplit(".", 1)[0] + "_classes.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(node.class_mapping, f, indent=4)

    if progress_callback:
        progress_callback(100, "Export complete!")
    return f"Saved LAS ({len(pts_float):,} points) to {os.path.basename(output_filepath)}"


def export_to_e57(node: ScanNode, output_filepath: str, mask: np.ndarray | None = None, progress_callback=None) -> str:
    """
    Exports as a structured E57 file, preserving the grid layout (rowIndex/columnIndex)
    so the file reloads with full panorama support. Hidden points are excluded.
    """
    import pye57  # optional, heavy dependency - imported lazily

    if progress_callback:
        progress_callback(5, "Preparing E57 export...")

    mapper = node.index_mapper
    h = mapper.h

    visible_mask = ~node.hidden_mask if node.hidden_mask is not None else np.ones(node.point_count, dtype=bool)

    row_indices = np.full(node.point_count, -1, dtype=np.int32)
    col_indices = np.full(node.point_count, -1, dtype=np.int32)

    valid_mask = mapper.pano_map >= 0
    pano_rows, pano_cols = np.where(valid_mask)
    point_ids = mapper.pano_map[pano_rows, pano_cols]

    original_rows = (h - 1) - pano_rows
    row_indices[point_ids] = original_rows * node.decimation_step
    col_indices[point_ids] = pano_cols * node.decimation_step

    if progress_callback:
        progress_callback(20, "De-quantizing coordinates...")

    has_grid = row_indices >= 0
    export_mask = visible_mask & has_grid
    if mask is not None and len(mask) > 0:
        layer_mask = np.zeros(node.point_count, dtype=bool)
        layer_mask[mask] = True
        export_mask = export_mask & layer_mask

    pts_float = node.points_quantized[export_mask].astype(np.float64) * node.quant_scale + node.quant_center

    if progress_callback:
        progress_callback(40, "Building E57 data dictionary...")

    data = {
        "cartesianX": pts_float[:, 0],
        "cartesianY": pts_float[:, 1],
        "cartesianZ": pts_float[:, 2],
        "rowIndex": row_indices[export_mask],
        "columnIndex": col_indices[export_mask],
    }

    if node.colors_rgb is not None:
        rgb = node.colors_rgb[export_mask]
        data["colorRed"] = rgb[:, 0].astype(np.int32)
        data["colorGreen"] = rgb[:, 1].astype(np.int32)
        data["colorBlue"] = rgb[:, 2].astype(np.int32)

    has_classes = node.point_classes is not None
    if has_classes:
        data["intensity"] = node.point_classes[export_mask].astype(np.float64)
    elif node.colors_intensity is not None:
        data["intensity"] = node.colors_intensity[export_mask].astype(np.float64)

    if progress_callback:
        progress_callback(60, "Writing E57 file...")

    e57 = pye57.E57(output_filepath, mode="w")
    try:
        e57.write_scan_raw(data, name=node.scan_name)
    finally:
        e57.close()

    if has_classes and node.class_mapping is not None:
        txt_path = output_filepath.rsplit(".", 1)[0] + "_classes.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for class_id, class_name in sorted(node.class_mapping.items()):
                f.write(f"{class_id} {class_name}\n")

    n_exported = int(export_mask.sum())
    if progress_callback:
        progress_callback(100, "E57 export complete!")
    return f"Saved structured E57 ({n_exported:,} points) to {os.path.basename(output_filepath)}"


def export_mask_preview_image(node: ScanNode, output_filepath: str, indices_by_label: dict[str, np.ndarray],
                               colors_by_label: dict[str, tuple], base_channel: Channel | None = None) -> str:
    """
    Renders the full panorama with each detected instance highlighted, and saves it as
    a .png/.jpg. Lets `--out preview.png` produce a visual sanity check next to the
    point-cloud export, without opening any GUI.

    Composites largest-first/smallest-last (same order as assign_classifications),
    so a smaller/more specific label still wins an overlap. For an explicitly
    user-chosen order instead (e.g. a GUI's reorderable layer stack, where the
    same label may legitimately appear more than once), see
    compose_layers_preview_image.
    """
    ordered = sorted(indices_by_label.items(), key=lambda item: len(item[1]), reverse=True)
    layers = [(label, indices, colors_by_label[label]) for label, indices in ordered]
    return compose_layers_preview_image(node, output_filepath, layers, base_channel=base_channel)


def compose_layers_preview_image(node: ScanNode, output_filepath: str,
                                  layers: list[tuple[str, np.ndarray, tuple]],
                                  base_channel: Channel | None = None) -> str:
    """
    Like export_mask_preview_image, but takes an explicit ordered sequence of
    (label, indices, color) entries instead of two dicts, applied in exactly
    the given order - earlier entries first, later entries win any overlap -
    and allows the same label (and color) to appear more than once, the same
    way compose_layers does for the classification field itself. Keep this in
    sync with compose_layers's compositing order so a preview image and its
    matching LAS/E57 export agree on which layer wins any given point.
    """
    from core.projection import default_channel, pad_img

    channel = base_channel or default_channel(node)
    # pad=False: keep the same (mapper.h, mapper.w) shape map_to_panorama produces,
    # so the per-label masks below align pixel-for-pixel with this image.
    panorama = reconstruct_panorama_for_channel(node, channel, pad=False)
    if panorama is None:
        raise ValueError(f"Could not reconstruct panorama for channel {channel!r}.")

    # Decide each pixel's winning color first and blend with the ORIGINAL
    # panorama exactly once at the end - blending onto the already-blended
    # overlay on every layer instead would compound: an overlap pixel would
    # end up a smeared mix of every layer that touched it (each one diluted
    # further by the next), rather than a single, clean color for whichever
    # layer actually owns that pixel per the given order.
    color_layer = np.zeros_like(panorama, dtype=np.float32)
    has_label = np.zeros(panorama.shape[:2], dtype=bool)
    for label, indices, color in layers:
        if len(indices) == 0:
            continue
        full_mask = np.zeros(node.point_count, dtype=bool)
        full_mask[indices] = True
        mask_2d = node.index_mapper.map_to_panorama(full_mask, default_value=False)
        color_layer[mask_2d] = np.array(color, dtype=np.float32)
        has_label |= mask_2d

    overlay = panorama.astype(np.float32)
    overlay[has_label] = overlay[has_label] * 0.5 + color_layer[has_label] * 0.5

    # Pad to the full equirectangular aspect ratio only now, after the mask overlay.
    overlay_bgr = cv2.cvtColor(pad_img(overlay.astype(np.uint8), fill_value=0), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filepath, overlay_bgr)
    logging.info(f"Preview image saved: {output_filepath}")
    return f"Saved preview image to {os.path.basename(output_filepath)}"
