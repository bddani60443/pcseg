from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pcseg.core.colors import generate_label_color
from pcseg.core.projection import Channel, extract_perspective_view, project_point_into_view, real_point_mask
from pcseg.core.scan_node import ScanNode
from pcseg.services.sam_service import SamService


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
        """Alias so cli._finish's per-instance filtering works unchanged for every result type."""
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
    for rank, mask_idx in enumerate(kept):
        mask_2d = masks[mask_idx].astype(bool)
        indices_3d = ctx.index_mapper.apply_mask(mask_2d, ctx.map_x, ctx.map_y, ctx.fov)
        indices_3d = indices_3d[real_mask[indices_3d]]
        if len(indices_3d) == 0:
            continue
        candidates.append(PickCandidate(
            indices=indices_3d, label=labels[rank], score=float(scores[mask_idx]), color=colors[rank],
        ))

    logging.info(f"Point {point_index}: {len(candidates)} candidate mask(s) kept "
                 f"(scores: {[round(c.score, 3) for c in candidates]}).")
    return PickResult(candidates=candidates)
