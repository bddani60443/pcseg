from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from core.colors import generate_label_color
from core.merge import mask_touches_border, merge_by_3d_iou_iterative
from core.projection import (
    Channel,
    get_face_projection_map,
    real_point_mask,
    reconstruct_panorama_for_channel,
)
from core.scan_node import ScanNode
from exceptions import SweepCancelled

# Interface an automatic-segmentation backend must satisfy: given one RGB (or
# false-color) image, propose object masks with no prompt of any kind. SAM2 is
# the only implementation today (SamService.automatic_generate, bound via
# functools.partial in the CLI to pin its extra kwargs); a future backend (e.g.
# FastSAM) just needs to match this same `image -> list[mask]` signature and get
# registered in the CLI's --backend choices - AutoSegmentationService below is
# written against this signature, not against SAM2 specifically.
AutoMaskBackend = Callable[[np.ndarray], list[np.ndarray]]


@dataclass
class AutoInstance:
    indices: np.ndarray
    label: str
    color: tuple


@dataclass
class AutoSweepResult:
    instances: list[AutoInstance] = field(default_factory=list)

    def combined_indices(self) -> np.ndarray:
        if not self.instances:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([inst.indices for inst in self.instances]))

    def indices_by_label(self) -> dict[str, np.ndarray]:
        return {inst.label: inst.indices for inst in self.instances}

    def color_by_label(self) -> dict[str, tuple]:
        return {inst.label: inst.color for inst in self.instances}


class AutoSegmentationService:
    """
    Class-agnostic 'segment everything' pipeline: sweeps a full 360 view around
    the scan, runs an AutoMaskBackend on each (undistorted, perspective) crop with
    no prompt, back-projects every proposed mask to 3D, then merges duplicate
    detections of the same physical object seen from overlapping views.
    """

    def __init__(self, backend: AutoMaskBackend) -> None:
        self.backend = backend

    def run_sweep(
        self,
        node: ScanNode,
        step_deg: int = 30,
        fov: float = 90.0,
        out_size: int = 1024,
        channel: Channel = "rgb",
        phi_angles: list[float] | None = None,
        min_points: int = 50,
        merge_iou: float = 0.5,
        border_merge_overlap: float = 0.6,
        surface_merge_distance: float | None = 0.3,
        merge_rounds: int = 3,
        progress_callback=None,
        view_grid: tuple[list[tuple[int, int, float, float]], int] | None = None,
        cancel_check=None,
    ) -> AutoSweepResult:
        """
        `cancel_check`, if given, is called before each view (same cadence as
        `progress_callback`) - returning True raises SweepCancelled immediately,
        discarding whatever partial results this call had gathered so far. Meant
        for a GUI's Cancel button; a sweep can take minutes, and the backend's
        own inference call can't be interrupted mid-call, so between-view is the
        finest granularity available.

        `view_grid`, if given, is an (entries, n_theta) pair from
        core.projection.compute_indexed_coverage_grid, used as the exact views to
        sweep instead of the step_deg x phi_angles cartesian product. Each entry's
        (theta_idx, phi_idx) is its position in the *full* candidate grid (before
        any empty-region tiles were dropped) - passing those straight through as
        this method's own view_keys is what lets the merge's view-adjacency gating
        keep working correctly even though the grid isn't fully dense.

        `border_merge_overlap` controls how aggressively fragments of one large
        surface (e.g. pavement cut into pieces by the sweep's view boundaries) get
        stitched back together: it only applies between two masks that each
        touched their own crop's edge, so compact fully-visible objects (a window
        inside a wall's view) are unaffected.

        `surface_merge_distance` is the fallback for when even that fails because
        two views resampled the same physical surface into near-disjoint point
        indices (common for a ground plane far from any view's center): two
        border-touching masks with zero index overlap still merge if they come
        within this many meters of each other in 3D. Set to None to disable.

        `merge_rounds` re-runs the merge on its own previous output: a pavement
        patch that only grew to touch a second patch in round 1 may only become
        view-adjacent to a *third* patch after that growth, so a second round can
        keep stitching without ever loosening the adjacency check that keeps
        unrelated regions apart. See core.merge.merge_by_3d_iou_iterative.
        """
        panorama = reconstruct_panorama_for_channel(node, channel)
        if panorama is None:
            raise ValueError(f"Channel {channel!r} is not available for this scan.")

        real_mask = real_point_mask(node)
        h, w = panorama.shape[:2]
        if view_grid is not None:
            views, n_theta = view_grid
        else:
            phi_angles = phi_angles or [0.0]
            thetas = list(range(0, 360, step_deg))
            n_theta = len(thetas)
            views = [(theta_idx, phi_idx, theta, phi)
                     for phi_idx, phi in enumerate(phi_angles)
                     for theta_idx, theta in enumerate(thetas)]

        candidates: list[np.ndarray] = []
        touches_border: list[bool] = []
        view_keys: list[tuple[int, int]] = []

        for i, (theta_idx, phi_idx, theta, phi) in enumerate(views):
            if cancel_check is not None and cancel_check():
                raise SweepCancelled(f"Cancelled after {i}/{len(views)} views.")
            if progress_callback:
                progress_callback(int((i / len(views)) * 90),
                                   f"view {i + 1}/{len(views)} (theta={theta:.0f} deg, phi={phi:.0f} deg)")

            map_x, map_y = get_face_projection_map(h, w, float(theta), float(phi), out_size, fov)
            view_crop = cv2.remap(panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

            masks_2d = self.backend(view_crop)
            for mask_2d in masks_2d:
                indices_3d = node.index_mapper.apply_mask(mask_2d, map_x, map_y, fov)
                indices_3d = indices_3d[real_mask[indices_3d]]
                if len(indices_3d) >= min_points:
                    candidates.append(indices_3d)
                    touches_border.append(mask_touches_border(mask_2d))
                    view_keys.append((theta_idx, phi_idx))

        logging.info(f"Sweep produced {len(candidates)} raw mask candidates across {len(views)} views. Merging...")
        if progress_callback:
            progress_callback(95, f"Merging {len(candidates)} candidates...")

        points_3d = None
        if surface_merge_distance is not None:
            points_3d = node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center

        merged = merge_by_3d_iou_iterative(
            candidates, max_rounds=merge_rounds, iou_threshold=merge_iou,
            touches_border=touches_border, border_overlap_threshold=border_merge_overlap,
            points_3d=points_3d, surface_merge_distance=surface_merge_distance,
            view_keys=view_keys, n_theta=n_theta,
        )
        labels = [f"object_{i:04d}" for i in range(len(merged))]
        colors = generate_label_color(labels)

        result = AutoSweepResult(instances=[
            AutoInstance(indices=indices, label=label, color=color)
            for indices, label, color in zip(merged, labels, colors)
        ])

        if progress_callback:
            progress_callback(100, f"Done: {len(result.instances)} objects")

        logging.info(f"Auto-segmentation complete: {len(candidates)} candidates merged into {len(result.instances)} objects.")
        return result
