# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
A small synthetic structured E57 scan for tests: a sphere of points around the
scanner, one point per grid cell, with RGB and intensity channels.
"""
from __future__ import annotations

import numpy as np

SYNTHETIC_ROWS = 80
SYNTHETIC_COLS = 160
SYNTHETIC_RADIUS = 5.0


def write_synthetic_e57(path, rows: int = SYNTHETIC_ROWS, cols: int = SYNTHETIC_COLS,
                        radius: float = SYNTHETIC_RADIUS) -> dict:
    """
    Writes a structured (grid) E57 scan of a sphere of `radius` meters around
    the scanner - one point per grid cell, with RGB and intensity - and returns
    the raw arrays that were written, for comparison after loading.
    """
    import pye57

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    row_idx, col_idx = row_idx.ravel(), col_idx.ravel()
    elevation = np.radians(-90.0 + (row_idx + 0.5) * 180.0 / rows)
    azimuth = np.radians(-180.0 + (col_idx + 0.5) * 360.0 / cols)

    data = {
        "cartesianX": radius * np.cos(elevation) * np.cos(azimuth),
        "cartesianY": radius * np.cos(elevation) * np.sin(azimuth),
        "cartesianZ": radius * np.sin(elevation),
        "rowIndex": row_idx.astype(np.int32),
        "columnIndex": col_idx.astype(np.int32),
        "colorRed": (col_idx * 255 // cols).astype(np.uint8),
        "colorGreen": (row_idx * 255 // rows).astype(np.uint8),
        "colorBlue": np.full(rows * cols, 128, dtype=np.uint8),
        "intensity": np.linspace(0.0, 1.0, rows * cols),
    }
    e57 = pye57.E57(str(path), mode="w")
    try:
        e57.write_scan_raw(data, name="synthetic")
    finally:
        e57.close()
    return data
