from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.mapping import IndexMapper


@dataclass
class ScanNode:
    """In-memory representation of one loaded, structured point cloud scan."""

    scan_id: str
    scan_name: str
    file_path: str

    points_quantized: NDArray[np.int16]
    quant_scale: float
    quant_center: NDArray[np.float32]

    normals: NDArray[np.int8]
    colors_rgb: Optional[NDArray[np.uint8]]
    colors_intensity: Optional[NDArray[np.uint8]]

    point_count: int
    original_point_count: int
    decimation_step: int
    available_modes: list[str]

    row_max: int
    col_max: int

    hidden_mask: NDArray[np.bool_]
    index_mapper: IndexMapper

    # Populated later by the segmentation pipeline (optional). Wide dtype since
    # auto-segmentation can propose more objects than an 8-bit id would hold;
    # export formats clip to their own ceiling at export time, not here.
    point_classes: Optional[NDArray[np.uint32]] = None
    class_mapping: Optional[dict] = field(default=None)

    def __str__(self) -> str:
        ratio = 0.0
        if self.original_point_count > 0:
            ratio = (self.point_count / self.original_point_count) * 100
        return (
            f"ScanNode '{self.scan_name}': {self.point_count:,}/{self.original_point_count:,} "
            f"points ({ratio:.1f}%), grid {self.row_max}x{self.col_max}, "
            f"modes={self.available_modes}"
        )
