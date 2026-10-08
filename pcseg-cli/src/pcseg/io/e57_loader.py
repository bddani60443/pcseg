from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pye57
from numpy.typing import NDArray

from pcseg.core.mapping import IndexMapper
from pcseg.core.scan_node import ScanNode


class ScanData:
    """Loads a structured (grid) E57 scan into a ScanNode."""

    def __init__(self, filepath: str | Path, decimation_step: int = 2, progress_callback=None) -> None:
        self.filepath = str(filepath)
        self.decimation_step = decimation_step
        self.progress_callback = progress_callback

    def load(self) -> ScanNode:
        logging.info(f"Loading structured scan: {Path(self.filepath).name}")

        e57 = pye57.E57(self.filepath, mode="r")
        try:
            header = e57.get_header(0)
            row_max = header.rowMaximum
            col_max = header.columnMaximum

            if row_max == 0 or col_max == 0:
                raise ValueError(f"'{self.filepath}' is not a structured (grid) point cloud E57 file.")

            if self.progress_callback:
                self.progress_callback(-1, "Reading E57 file from disk...")

            raw_data = e57.read_scan_raw(0, ignore_unsupported_fields=True)
            if self.progress_callback:
                self.progress_callback(50, "Read from disk, processing...")

            node = self._process_raw_data(raw_data, row_max, col_max, header)

            logging.info(f"Loaded {node.point_count:,} points.")
            if self.progress_callback:
                self.progress_callback(100, "Ready")

            return node
        finally:
            e57.close()

    @staticmethod
    def peek_metadata(filepath: str | Path) -> dict:
        info = {"point_count": 0, "file_size_mb": 0.0, "has_color": False, "has_intensity": False}
        info["file_size_mb"] = os.path.getsize(filepath) / (1024**2)

        e57 = pye57.E57(str(filepath))
        try:
            header = e57.get_header(0)
            info["point_count"] = header.rowMaximum * header.columnMaximum
            field_names = header.point_fields
            info["has_color"] = "colorRed" in field_names
            info["has_intensity"] = "intensity" in field_names
        finally:
            e57.close()
        return info

    def _compute_normals(self, pts: NDArray[np.int16], rows: NDArray[np.int16], cols: NDArray[np.int16],
                          h: int, w: int, normal_step: int = 2) -> NDArray[np.int8]:
        """Computes per-point normals via a screen-space cross product on the range-image grid."""
        step = normal_step if normal_step > 0 else 1

        dh = (h // step) + 1
        dw = (w // step) + 1

        d_rows = np.clip(rows // step, 0, dh - 1)
        d_cols = np.clip(cols // step, 0, dw - 1)

        grid = np.zeros((dh, dw, 3), dtype=np.float32)
        grid[d_rows, d_cols] = pts.astype(np.float32)

        dy, dx = np.gradient(grid, axis=(0, 1))
        normals_grid = np.cross(dx, dy)

        norm = np.linalg.norm(normals_grid, axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normals_grid = np.divide(normals_grid, norm, out=np.zeros_like(normals_grid), where=norm > 0)

        final_normals = normals_grid[d_rows, d_cols]

        dot_prod = np.einsum("ij,ij->i", final_normals, pts.astype(np.float32))
        signs = np.where(dot_prod > 0, -1, 1)[:, np.newaxis]
        final_normals *= signs

        return (final_normals * 127).astype(np.int8)

    def _process_raw_data(self, raw: dict, orig_rows: int, orig_cols: int, header: "pye57.ScanHeader") -> ScanNode:
        if self.progress_callback:
            self.progress_callback(60, "Preparing grid information...")
        mask, rows_masked, cols_masked, dec_h, dec_w = self._prepare_grid_info(raw, orig_rows, orig_cols)

        if self.progress_callback:
            self.progress_callback(70, "Processing 3D coordinates...")
        points_quantized, scale, center = self._process_coordinates(raw, mask, header)

        if self.progress_callback:
            self.progress_callback(80, "Computing normals...")
        normals_int8 = self._compute_normals(points_quantized, rows_masked, cols_masked, dec_h, dec_w)

        if self.progress_callback:
            self.progress_callback(85, "Extracting colors...")
        rgb_uint8 = self._process_colors(raw, mask)

        if self.progress_callback:
            self.progress_callback(90, "Extracting intensities...")
        intensity_uint8 = self._process_intensity(raw, mask)

        mapper = IndexMapper(
            raw_indices={"row": rows_masked, "col": cols_masked},
            row_max=dec_h - 1,
            col_max=dec_w - 1,
        )

        hidden_mask = np.full(len(points_quantized), False, dtype=np.bool_)

        modes = []
        if rgb_uint8 is not None:
            modes.append("rgb")
        if intensity_uint8 is not None:
            modes.append("intensity")
        modes.append("normals")

        scan_name = Path(self.filepath).stem
        orig_count = orig_rows * orig_cols

        return ScanNode(
            scan_id=str(uuid.uuid4()),
            scan_name=scan_name,
            file_path=self.filepath,
            points_quantized=points_quantized,
            quant_scale=float(1.0 / scale),
            quant_center=center,
            normals=normals_int8,
            colors_rgb=rgb_uint8,
            colors_intensity=intensity_uint8,
            point_count=len(points_quantized),
            original_point_count=orig_count,
            decimation_step=self.decimation_step,
            available_modes=modes,
            row_max=dec_h,
            col_max=dec_w,
            hidden_mask=hidden_mask,
            index_mapper=mapper,
        )

    def _prepare_grid_info(self, raw: dict, orig_rows: int, orig_cols: int):
        row_idx = raw["rowIndex"]
        col_idx = raw["columnIndex"]

        mask = (row_idx % self.decimation_step == 0) & (col_idx % self.decimation_step == 0)

        dec_h = (orig_rows // self.decimation_step) + 1
        dec_w = (orig_cols // self.decimation_step) + 1

        rows_masked = row_idx[mask] // self.decimation_step
        cols_masked = col_idx[mask] // self.decimation_step

        return mask, rows_masked, cols_masked, dec_h, dec_w

    def _process_coordinates(self, raw: dict, mask, header: "pye57.ScanHeader"):
        if "cartesianX" in raw:
            x = raw["cartesianX"][mask]
            y = raw["cartesianY"][mask]
            z = raw["cartesianZ"][mask]
        elif "sphericalRange" in raw:
            r = raw["sphericalRange"][mask]
            az = raw["sphericalAzimuth"][mask]
            el = raw["sphericalElevation"][mask]
            cos_el = np.cos(el)
            x = r * cos_el * np.cos(az)
            y = r * cos_el * np.sin(az)
            z = r * np.sin(el)
        else:
            raise ValueError(f"Unsupported coordinate format in: {self.filepath}")

        pts_world = np.empty((len(x), 3), dtype=np.float64)
        pts_world[:, 0] = x
        pts_world[:, 1] = y
        pts_world[:, 2] = z

        r_mat, t_vec = self._extract_pose(header)
        if not (np.array_equal(r_mat, np.eye(3)) and np.array_equal(t_vec, np.zeros(3))):
            np.dot(pts_world, r_mat.T, out=pts_world)
            pts_world += t_vec

        center_f64 = np.mean(pts_world, axis=0)
        pts_world -= center_f64

        max_dist = max(abs(np.max(pts_world)), abs(np.min(pts_world)))
        if max_dist == 0:
            max_dist = 1.0
        scale = 32767.0 / max_dist

        pts_world *= scale
        np.rint(pts_world, out=pts_world)
        points_quantized = pts_world.astype(np.int16)

        return points_quantized, scale, center_f64.astype(np.float32)

    def _process_colors(self, raw: dict, mask) -> Optional[NDArray[np.uint8]]:
        if "colorRed" not in raw:
            return None

        rgb_temp = np.column_stack((raw["colorRed"][mask], raw["colorGreen"][mask], raw["colorBlue"][mask]))

        if rgb_temp.max() <= 1.0:
            return (rgb_temp * 255).astype(np.uint8)
        if rgb_temp.max() > 255:
            return (rgb_temp / 256).astype(np.uint8)
        return rgb_temp.astype(np.uint8)

    def _process_intensity(self, raw: dict, mask) -> Optional[NDArray[np.uint8]]:
        if "intensity" not in raw:
            return None
        i = raw["intensity"][mask]
        i_min, i_max = i.min(), i.max()
        if i_max > i_min:
            return ((i - i_min) / (i_max - i_min) * 255).astype(np.uint8)
        return np.zeros_like(i, dtype=np.uint8)

    def _extract_pose(self, header: "pye57.ScanHeader") -> tuple[np.ndarray, np.ndarray]:
        r_mat = np.eye(3, dtype=np.float64)
        t_vec = np.zeros(3, dtype=np.float64)

        try:
            if hasattr(header, "translation"):
                t_vec = np.array(header.translation, dtype=np.float64)

            if hasattr(header, "rotation"):
                w, x, y, z = header.rotation
                nq = w * w + x * x + y * y + z * z
                if nq > 0.0:
                    s = 2.0 / nq
                    xs, ys, zs = x * s, y * s, z * s
                    wx, wy, wz = w * xs, w * ys, w * zs
                    xx, xy, xz = x * xs, x * ys, x * zs
                    yy, yz, zz = y * ys, y * zs, z * zs
                    r_mat = np.array([
                        [1.0 - (yy + zz), xy - wz, xz + wy],
                        [xy + wz, 1.0 - (xx + zz), yz - wx],
                        [xz - wy, yz + wx, 1.0 - (xx + yy)],
                    ], dtype=np.float64)
        except Exception as e:
            logging.warning(f"Pose extraction failed, falling back to identity: {e}")

        return r_mat, t_vec
