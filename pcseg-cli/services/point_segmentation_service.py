from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from core.colors import generate_label_color
from core.projection import (
    Channel,
    default_channel,
    extract_perspective_view,
    project_point_into_view,
    real_point_mask,
)
from core.scan_node import ScanNode
from services.sam_service import SamService


@dataclass
class PickCandidate:
    indices: np.ndarray
    label: str
    score: float
    color: tuple


@dataclass
class PickResult:
    candidates: list[PickCandidate] = field(default_factory=list)

    @property
    def instances(self) -> list[PickCandidate]:
        """Alias so cli._finish can treat every result type (sweep, auto-segment, pick) the same way."""
        return self.candidates

    def combined_indices(self) -> np.ndarray:
        if not self.candidates:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([c.indices for c in self.candidates]))

    def indices_by_label(self) -> dict[str, np.ndarray]:
        return {c.label: c.indices for c in self.candidates}

    def color_by_label(self) -> dict[str, tuple]:
        return {c.label: c.color for c in self.candidates}


def segment_at_point(
    sam: SamService,
    node: ScanNode,
    point_index: int,
    out_size: int | None = None,
    fov: float = 90.0,
    channel: Channel | None = None,
    keep_all_masks: bool = False,
    extra_indices: list[int] | None = None,
    debug_dir: str | None = None,
) -> PickResult:
    """
    Single- or multi-point SAM2 segmentation - the CLI counterpart of a GUI's
    "click a point, get the object under it" interaction. There's no mouse in
    a non-interactive pipeline, so `point_index` is resolved beforehand from a
    3D coordinate (see core.projection.find_nearest_point_index) or a raw
    point-cloud index.

    With a single point (the common case, `extra_indices` empty/None), SAM2
    returns 3 candidate masks ranked by its own confidence score - the point
    alone is often ambiguous (e.g. "the door" vs. "the door panel" vs. "the
    whole wall"). By default only the best-scoring one is kept;
    `keep_all_masks=True` exports all 3 as separate labeled candidates
    instead, letting you inspect and pick offline.

    `extra_indices` adds more POSITIVE point-prompts on the SAME rendered view
    `point_index` (the anchor) already produced - see
    core.projection.project_point_into_view, which projects each extra index
    into that view's own pixel space (a point that falls outside the anchor's
    FOV is silently skipped, matching the tolerant "a boundary gap just isn't
    included" convention core.projection's own ROI/coverage-grid code already
    uses elsewhere - the caller, e.g. a GUI's multi-point pick session, is
    expected to have already constrained which points it could offer here in
    the first place). Multiple points already resolve SAM2's single-point
    ambiguity itself, so only the single best mask is requested in that case
    regardless of `keep_all_masks`.

    `debug_dir`, if given, saves one annotated PNG of the pick there (see
    `_save_pick_debug_image`): the rendered view with the clicked point(s),
    then every mask SAM2 returned, best score first, each with its score and
    whether - and with how many 3D points - it was kept. None (the default)
    skips this at no cost.
    """
    sam._ensure_loaded()

    ctx = extract_perspective_view(node, point_index, out_size=out_size, fov=fov, channel=channel)
    if ctx is None:
        raise ValueError(f"Point {point_index} has no valid panorama position; cannot extract a view around it.")

    point_coords = [[ctx.center_px, ctx.center_py]]
    for extra_index in (extra_indices or []):
        px_py = project_point_into_view(node, ctx, extra_index)
        if px_py is not None:
            point_coords.append(list(px_py))
    is_multi_point = len(point_coords) > 1

    masks, scores, _ = sam.predict(
        ctx.image_2d,
        point_coords=np.array(point_coords),
        point_labels=np.ones(len(point_coords), dtype=np.int64),
        multimask_output=not is_multi_point,
    )

    order = np.argsort(scores)[::-1]  # best SAM2 score first
    keep_all = keep_all_masks and not is_multi_point
    kept = order if keep_all else order[:1]

    labels = [f"candidate_{rank}" for rank in range(len(kept))] if keep_all else ["picked_object"]
    colors = generate_label_color(labels)

    real_mask = real_point_mask(node)
    candidates = []
    kept_point_counts: dict[int, int] = {}
    for rank, mask_idx in enumerate(kept):
        mask_2d = masks[mask_idx].astype(bool)
        indices_3d = ctx.index_mapper.apply_mask(mask_2d, ctx.map_x, ctx.map_y, ctx.fov)
        indices_3d = indices_3d[real_mask[indices_3d]]
        kept_point_counts[int(mask_idx)] = len(indices_3d)
        if len(indices_3d) == 0:
            continue
        candidates.append(PickCandidate(
            indices=indices_3d, label=labels[rank], score=float(scores[mask_idx]), color=colors[rank],
        ))

    if debug_dir is not None:
        path = _save_pick_debug_image(
            ctx.image_2d, channel or default_channel(node), point_coords, masks, np.ravel(scores),
            order, kept_point_counts, debug_dir, point_index,
        )
        logging.info(f"Pick debug image saved: {path}")

    logging.info(f"Point {point_index}: {len(candidates)} candidate mask(s) kept "
                 f"(scores: {[round(c.score, 3) for c in candidates]}).")
    return PickResult(candidates=candidates)


_PICK_DEBUG_PANEL_MAX = 768  # px per panel side - keeps the composite small for large views


def _save_pick_debug_image(
    image: np.ndarray, channel: Channel, point_coords: list, masks: np.ndarray, scores: np.ndarray,
    order: np.ndarray, kept_point_counts: dict[int, int], debug_dir: str, point_index: int,
) -> str:
    """
    Writes `<debug_dir>/pick_<timestamp>_pt<point_index>.png`: side-by-side
    panels of the rendered view SAM2 saw - first plain, with the clicked
    point(s) marked (the anchor red, extra points orange), then one panel per
    returned mask in score order, the mask tinted and outlined, titled with
    its rank, SAM2 score and what became of it ("kept: N points" in 3D, or
    "not kept"). Returns the file path.
    """
    os.makedirs(debug_dir, exist_ok=True)
    base = image.copy()
    if channel == "rgb":
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)  # "intensity"/"normals" are already BGR
    size = base.shape[0]
    shrink = min(1.0, _PICK_DEBUG_PANEL_MAX / size)
    marker = max(4, size // 60)

    def draw_points(panel: np.ndarray) -> None:
        for i, (px, py) in enumerate(point_coords):
            color = (0, 0, 255) if i == 0 else (0, 165, 255)
            cv2.circle(panel, (int(px), int(py)), marker, (0, 0, 0), marker // 2 + 2, cv2.LINE_AA)
            cv2.circle(panel, (int(px), int(py)), marker, color, marker // 2, cv2.LINE_AA)

    def title(panel: np.ndarray, text: str) -> np.ndarray:
        panel = cv2.resize(panel, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA) if shrink < 1 else panel
        bar = np.full((32, panel.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, panel])

    view = base.copy()
    draw_points(view)
    panels = [title(view, f"point {point_index}: view SAM2 saw ({channel})")]

    for rank, mask_idx in enumerate(order, start=1):
        mask = masks[mask_idx].astype(bool)
        panel = base.copy()
        tint = panel.copy()
        tint[mask] = (60, 200, 60)
        panel = cv2.addWeighted(tint, 0.45, panel, 0.55, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (60, 255, 60), max(1, size // 400), cv2.LINE_AA)
        draw_points(panel)
        n_points = kept_point_counts.get(int(mask_idx))
        fate = "not kept" if n_points is None else f"kept: {n_points:,} points"
        panels.append(title(panel, f"#{rank}  score {float(scores[mask_idx]):.3f}  {fate}"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(debug_dir, f"pick_{stamp}_pt{point_index}.png")
    if not cv2.imwrite(path, np.hstack(panels)):
        raise OSError(f"Could not write the pick debug image {path!r}")
    return path

