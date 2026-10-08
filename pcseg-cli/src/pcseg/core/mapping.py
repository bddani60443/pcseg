from typing import Dict

import numpy as np
from numpy.typing import NDArray


class IndexMapper:
    """
    Core logic for mapping 1D point cloud data to 2D panorama grids.
    Assumes strictly structured data.

    The pano_map stores (row, col) -> point_index mapping where rows are
    vertically flipped so that pano_map coordinates match the post-rot90
    panorama image used in the projection pipeline.
    """
    def __init__(self, raw_indices: Dict[str, np.ndarray], row_max: int, col_max: int) -> None:
        if "row" not in raw_indices or "col" not in raw_indices:
            raise ValueError("[IndexMapper] Critical Error: Missing row/col indices!")

        self.h = row_max + 1
        self.w = col_max + 1
        self.pano_map = np.full((self.h, self.w), -1, dtype=np.int32)

        rows = raw_indices["row"].copy()
        cols = raw_indices["col"]

        rows = (self.h - 1) - rows

        point_indices = np.arange(len(rows), dtype=np.int32)

        valid_mask = (rows >= 0) & (rows < self.h) & (cols >= 0) & (cols < self.w)

        self.pano_map[rows[valid_mask], cols[valid_mask]] = point_indices[valid_mask]

        # Inverse lookup: point_index -> (pano_y, pano_x) in already-flipped image coords.
        # Lets callers find a point's panorama position without assuming a column-major-dense
        # E57 layout (the `i // H, i % H` trick used elsewhere).
        self.point_to_pano = np.full((len(point_indices), 2), -1, dtype=np.int32)
        self.point_to_pano[point_indices[valid_mask], 0] = rows[valid_mask]
        self.point_to_pano[point_indices[valid_mask], 1] = cols[valid_mask]

    def map_to_panorama(self, source_data_1d: NDArray, default_value: float = 0) -> NDArray:
        """
        Maps any 1D array to a 2D panorama image using the pre-calculated map.

        Args:
            source_data_1d: (N,) or (N, C) numpy array (Intensity, RGB, Normals).
            default_value: Value for empty pixels (e.g., 0 for black).

        Returns:
            NDArray: (H, W) or (H, W, C) array.
        """
        if source_data_1d is None:
            return np.ndarray([])

        if source_data_1d.ndim == 1:
            out_shape = (self.h, self.w)
        else:
            out_shape = (self.h, self.w, source_data_1d.shape[1])

        panorama = np.full(out_shape, default_value, dtype=source_data_1d.dtype)

        valid_pixels_mask = self.pano_map >= 0
        point_indices = self.pano_map[valid_pixels_mask]

        panorama[valid_pixels_mask] = source_data_1d[point_indices]
        return panorama

    def map_from_panorama(self, panorama_2d: NDArray, default_value: float = 0) -> NDArray:
        """
        Maps a 2D panorama image (or mask) back to the 1D point cloud array.
        Essential for converting SAM2 masks to 3D selections.

        Args:
            panorama_2d: (H, W) or (H, W, C) numpy array.

        Returns:
            NDArray: 1D array matching the original point cloud count.
        """
        num_points = np.max(self.pano_map) + 1

        if panorama_2d.ndim == 3:
            out_shape = (num_points, panorama_2d.shape[2])
        else:
            out_shape = (num_points,)

        source_1d = np.full(out_shape, default_value, dtype=panorama_2d.dtype)
        valid_mask = self.pano_map >= 0
        point_indices = self.pano_map[valid_mask]
        pixel_values = panorama_2d[valid_mask]
        source_1d[point_indices] = pixel_values

        return source_1d

    def apply_mask(self, mask_2d: NDArray, map_x: NDArray, map_y: NDArray, fov: float = 90.0) -> NDArray:
        """
        Back-projects a 2D segmentation mask to point cloud indices.

        Projects the masked perspective pixels onto the panorama via map_x/map_y,
        uses morphological close to fill sampling gaps (no dilate - does not
        expand beyond the mask boundary), then looks up point indices directly
        from pano_map.

        Args:
            mask_2d: (H_out, W_out) bool/uint8 segmentation mask.
            map_x:   (H_out, W_out) float32 - panorama X per perspective pixel.
            map_y:   (H_out, W_out) float32 - panorama Y per perspective pixel.
            fov:     Field of view in degrees (for zoom ratio calculation).

        Returns:
            Unique 1D point cloud indices belonging to the masked object.
        """
        import cv2

        mask_ys, mask_xs = np.where(mask_2d > 0)
        if len(mask_ys) == 0:
            return np.array([], dtype=np.int64)

        # 1. Project masked pixels onto panorama space
        pano_px = np.round(map_x[mask_ys, mask_xs]).astype(np.int32)
        pano_py = np.round(map_y[mask_ys, mask_xs]).astype(np.int32)

        valid = (pano_px >= 0) & (pano_px < self.w) & (pano_py >= 0) & (pano_py < self.h)
        pano_px = pano_px[valid]
        pano_py = pano_py[valid]

        # 2. Build sparse panorama mask
        sparse_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        sparse_mask[pano_py, pano_px] = 255

        # 3. Dilate + close to fill gaps between sparse projected pixels
        panorama_fov_pixels = self.w * (fov / 360.0)
        zoom_ratio = panorama_fov_pixels / mask_2d.shape[1]
        kernel_size = int(np.ceil(zoom_ratio)) + 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        dense_mask = cv2.dilate(sparse_mask, kernel, iterations=1)
        dense_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_CLOSE, kernel)

        # 4. Look up point indices from pano_map
        filled_ys, filled_xs = np.where(dense_mask > 0)

        point_indices = self.pano_map[filled_ys, filled_xs]
        return np.unique(point_indices[point_indices >= 0])
