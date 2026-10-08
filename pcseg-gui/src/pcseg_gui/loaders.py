from __future__ import annotations

import uuid
from pathlib import Path

import laspy
import numpy as np
from pcseg.core.colors import generate_label_color
from pcseg.core.mapping import IndexMapper
from pcseg.core.scan_node import ScanNode


def load_point_cloud(path: str, decimation: int = 2) -> ScanNode:
    suffix = Path(path).suffix.lower()
    if suffix == ".e57":
        return load_e57(path, decimation=decimation)
    if suffix == ".las":
        return load_las(path)
    raise ValueError(f"Unsupported file type '{suffix}' (expected .e57 or .las).")


def load_e57(path: str, decimation: int = 2) -> ScanNode:
    from pcseg.io.e57_loader import ScanData

    return ScanData(path, decimation_step=decimation).load()


def load_las(path: str) -> ScanNode:
    """
    Builds a viewer-compatible ScanNode from a LAS file. Since a flat point
    cloud has no scanner-grid rows/columns, `index_mapper` is a dummy
    (1 x N) mapper - fine for the viewer, which never calls
    apply_mask/panorama methods on it, only the renderer's direct point
    arrays.
    """
    las = laspy.read(path)
    n = len(las.points)

    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    center = xyz.mean(axis=0)
    centered = xyz - center
    max_dist = np.max(np.abs(centered)) or 1.0
    scale = 32767.0 / max_dist
    points_quantized = np.rint(centered * scale).astype(np.int16)

    colors_rgb = None
    dims = set(las.point_format.dimension_names)
    if {"red", "green", "blue"} <= dims:
        rgb16 = np.column_stack([las.red, las.green, las.blue])
        if rgb16.max() > 255:
            colors_rgb = (rgb16 / 256).astype(np.uint8)
        else:
            colors_rgb = rgb16.astype(np.uint8)

    if colors_rgb is None and "classification" in dims:
        classes = np.asarray(las.classification)
        unique_classes = np.unique(classes)
        if len(unique_classes) > 1:
            labels = [f"class_{c}" for c in unique_classes]
            palette = generate_label_color(labels)
            colors_rgb = np.zeros((n, 3), dtype=np.uint8)
            for class_id, color in zip(unique_classes, palette):
                colors_rgb[classes == class_id] = color

    return ScanNode(
        scan_id=str(uuid.uuid4()),
        scan_name=Path(path).stem,
        file_path=path,
        points_quantized=points_quantized,
        quant_scale=float(1.0 / scale),
        quant_center=center.astype(np.float32),
        normals=np.zeros((n, 3), dtype=np.int8),  # LAS carries no normals
        colors_rgb=colors_rgb,
        colors_intensity=None,
        point_count=n,
        original_point_count=n,
        decimation_step=1,
        available_modes=["rgb"] if colors_rgb is not None else [],
        row_max=1,
        col_max=n,
        hidden_mask=np.zeros(n, dtype=bool),
        index_mapper=IndexMapper({"row": np.zeros(n, dtype=np.int32), "col": np.arange(n, dtype=np.int32)},
                                  row_max=0, col_max=n - 1),
    )
