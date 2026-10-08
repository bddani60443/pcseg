from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import cv2
import numpy as np

from pcseg.core.colors import generate_label_color
from pcseg.core.mask_filters import _filter_outliers_core, _to_points_3d, filter_mask_outliers
from pcseg.core.merge import merge_by_3d_iou_iterative
from pcseg.core.projection import Channel, get_face_projection_map, real_point_mask, reconstruct_panorama_for_channel
from pcseg.core.scan_node import ScanNode
from pcseg.exceptions import SweepCancelled
from pcseg.services.dino_service import DinoService
from pcseg.services.sam_service import SamService


@dataclass
class DetectedInstance:
    indices: np.ndarray
    label: str
    score: float
    color: tuple


@dataclass
class SweepResult:
    instances: list[DetectedInstance] = field(default_factory=list)

    def combined_indices(self) -> np.ndarray:
        if not self.instances:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([inst.indices for inst in self.instances]))

    def indices_by_label(self) -> dict[str, np.ndarray]:
        out: dict[str, list[np.ndarray]] = {}
        for inst in self.instances:
            out.setdefault(inst.label, []).append(inst.indices)
        return {label: np.unique(np.concatenate(chunks)) for label, chunks in out.items()}

    def color_by_label(self) -> dict[str, tuple]:
        out = {}
        for inst in self.instances:
            out.setdefault(inst.label, inst.color)
        return out


class SegmentationService:
    """Orchestrates DINO + SAM2 inference over a full 360 sweep. No UI dependency."""

    def __init__(self, dino_service: DinoService, sam_service: SamService) -> None:
        self.dino = dino_service
        self.sam = sam_service

    def ensure_models_loaded(self) -> None:
        self.dino._ensure_loaded()
        self.sam._ensure_loaded()

    def _detect_all_phrases(self, view: np.ndarray, prompt: str, box_threshold: float, text_threshold: float,
                             max_box_area_fraction: float = 0.95):
        """
        Runs GroundingDINO once per comma-separated phrase in `prompt`, instead
        of once for the whole caption, then concatenates the results.

        Measured on a real scan: querying "window, building" as one caption
        found ~3x fewer "window" boxes than querying "window" alone at the same
        thresholds (each phrase's confidence competes with the others within a
        single DINO forward pass) - splitting recovers each category's full,
        independent recall, at the cost of one extra DINO forward pass per
        phrase per view (cheap relative to the SAM2 pass that follows).

        `max_box_area_fraction` drops any box covering that fraction of the
        view's own area or more - reported live as DINO occasionally claiming
        the ENTIRE crop is one object, almost always a degenerate detection
        (a confused/empty view, a heavily distorted crop edge) rather than a
        real single object genuinely filling the whole frame - the latter IS
        possible in principle (a wall filling a close-up view, say) but isn't
        what a near-full-frame box from an otherwise object-level prompt
        ("window", "door") is actually catching in practice, so it's dropped
        rather than kept as a rare-but-real edge case.
        """
        h, w = view.shape[:2]
        view_area = float(h * w)
        phrases = [p.strip() for p in prompt.split(",") if p.strip()]
        all_boxes, all_scores, all_labels = [], [], []
        for phrase in phrases:
            boxes, scores, labels = self.dino.run_inference(
                view, phrase, box_thresh=box_threshold, text_thresh=text_threshold,
            )
            if len(boxes) == 0:
                continue
            # GroundingDINO's box_threshold and text_threshold are independent gates -
            # a box can clear box_threshold (objectness) while no token of the queried
            # phrase clears text_threshold for it, in which case the library reports an
            # empty phrase for that box. That's not a confirmed match to anything the
            # user asked for (observed live: with box_threshold loosened well below
            # text_threshold, up to ~75% of boxes in a sweep came back with label=""),
            # so drop those boxes here rather than surface them as unlabeled instances.
            keep = np.array([bool(label) for label in labels])
            if view_area > 0:
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep &= (box_areas / view_area) < max_box_area_fraction
            if not keep.any():
                continue
            boxes, scores = boxes[keep], scores[keep]
            labels = [label for label, k in zip(labels, keep) if k]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.extend(labels)

        if not all_boxes:
            return np.array([]), np.array([]), []
        return np.concatenate(all_boxes), np.concatenate(all_scores), all_labels

    def run_sweep(
        self,
        node: ScanNode,
        prompt: str,
        step_deg: int = 30,
        fov: float = 90.0,
        out_size: int = 1024,
        dino_channel: Channel = "rgb",
        sam_channel: Channel = "rgb",
        phi_angles: list[float] | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        max_box_area_fraction: float = 0.95,
        sam_score_threshold: float = 0.0,
        min_score: float = 0.0,
        merge_duplicates: bool = False,
        merge_iou: float = 0.5,
        merge_overlap: float | None = 0.3,
        merge_rounds: int = 3,
        refine_detections: bool = False,
        refine_margin: float = 0.5,
        refine_min_fov: float = 25.0,
        refine_box_threshold: float = 0.45,
        debug_dir: str | None = None,
        progress_callback=None,
        view_grid: list[tuple[float, float] | tuple[float, float, float]] | None = None,
        cancel_check=None,
    ) -> SweepResult:
        """
        Rotates 360 (at one or more pitch angles), runs DINO on `dino_channel` to find
        candidate boxes and SAM2 on `sam_channel` to segment them, then back-projects
        every mask to 3D point indices. DINO and SAM2 can run on different rendered
        channels of the same view (e.g. DINO on intensity, SAM2 on normals) since the
        two are independent renders sharing the same projection geometry.

        `view_grid`, if given, is used as the exact views to sweep instead of the
        step_deg x phi_angles cartesian product - pass the output of
        core.projection.compute_coverage_grid for overlap-guaranteed, empty-region-aware
        coverage instead of a manually chosen step_deg/phi_angles combination. Each entry
        is normally (theta, phi), using this call's own `fov` for every view like always;
        an entry may instead be (theta, phi, view_fov) to override `fov` for just that ONE
        view - see core.projection.compute_multiscale_coverage_grid, which returns exactly
        this mixed-arity form to layer extra, differently-zoomed passes (tighter for
        distant material, wider for close material - see its own docstring for why) on top
        of the normal coverage grid, all swept in the one `run_sweep` call.

        `max_box_area_fraction` drops any DINO box covering that fraction of its
        OWN view's area or more (see `_detect_all_phrases`) - GroundingDINO
        occasionally claims the entire crop is one object, almost always a
        degenerate detection rather than something genuinely worth keeping.
        Applied identically to every DINO call this makes, coarse sweep and
        refine pass alike.

        `sam_score_threshold` drops a detection whose SAM2 mask scored below it - this is
        SAM2's own confidence in the mask BOUNDARY it drew, not in whether it segmented the
        right object (measured on a real scan: a wrong mask - pavement mistaken for a
        building - scored 0.99, higher than many correct window masks around 0.86-0.95), so
        it filters low-quality/fragmented mask shapes, not semantically wrong detections.

        `min_score` drops a FINAL instance (after outlier filtering and merging) whose
        `.score` - GroundingDINO's own text-match confidence, averaged across fragments if
        `merge_duplicates` merged more than one into it - is below this. This is the
        semantic-confidence counterpart to `sam_score_threshold`'s mask-quality one: a low
        `min_score` instance is one DINO itself wasn't sure actually matches the prompt,
        as opposed to a low `sam_score_threshold` one, which DINO was confident about but
        SAM2 drew a messy boundary for. Applied last (post-merge) rather than per-raw-box
        so a merge that averaged a confident fragment with a weak one is judged by its
        final combined confidence, not the confidence of whichever raw box happened to
        seed the cluster. Off by default (`0.0`) - `box_threshold`/`text_threshold`
        already gate DINO's raw output at detection time; this is an additional, coarser
        knob for "drop the low-confidence tail of what's left," not a replacement.

        `merge_duplicates`, off by default, fuses same-label instances that are
        really overlapping fragments of one physical object seen from multiple
        sweep views/pitch angles (each view runs DINO+SAM2 independently, so an
        object visible across two adjacent views' overlap gets detected - and
        counted - once per view; merging is the union of those overlapping
        fragments back into one instance, not a pick-a-winner dedup). Off by
        default deliberately: merging can hide exactly the kind of segmentation
        mistake worth seeing raw (e.g. two classes fighting over the same points,
        or a mask leaking onto a neighboring object) - turn it on once you trust
        the detections and just want one instance per real object, not while
        you're still debugging what the pipeline is finding. Only merges within
        the same label (a "tree" instance never merges into a "building" one).
        Merging is by 3D point-index IoU (see core.merge.merge_by_3d_iou_iterative)
        - `merge_iou` is the overlap fraction required, `merge_rounds` lets
        clusters that grow into newly-overlapping neighbors keep merging. A
        merged instance's score is the plain average of the DINO confidences of
        the fragments that went into it.

        `merge_overlap`, if not None, adds a second, looser merge rule ON TOP of
        `merge_iou` (either one being satisfied is enough - see
        core.merge.merge_by_3d_iou's `overlap_coeff_threshold`): two fragments
        also merge once the SMALLER one is `merge_overlap`-covered by the other,
        regardless of how different their sizes are - plain Jaccard IoU
        (`merge_iou`) never fires for a same-object pair where one view saw it
        near-complete and another only a sliver at its crop edge, since the
        union balloons relative to the intersection. NOT YET VALIDATED against a
        real scan (see TODO.md) - the default is a first guess, not tuned.

        Independently of `merge_duplicates`, every instance is always run through
        `filter_mask_outliers` (see core.mask_filters) before it's returned - each
        instance against only its own points, never the whole scan, so a mask
        that leaked onto a spatially disjoint surface (e.g. SAM2 bleeding from a
        window onto the wall behind it) gets trimmed back down regardless of
        whether merging is on. This is a targeted, always-on cleanup of each raw
        detection's own points, not the general-purpose `--filter` mechanism (see
        TODO.md - that one's scoped for removal for a different reason). Runs
        before merging specifically so a leaked mask can't skew a merge's 3D-IoU
        overlap calculation either.

        `refine_detections`, off by default, changes the coarse sweep above into
        a DINO-ONLY box-finding pass (see `_detect_boxes_view`) - SAM2 never
        runs there at all, and neither does the usual post-sweep outlier
        filtering, since there's no mask yet to filter. Every kept box's own
        angular extent (see `_box_angular_extent`) becomes a candidate for the
        SECOND, real pass (`_refine_boxes`): a NEW view is extracted, re-centered
        on that box's own middle and sized to just cover its extent plus
        `refine_margin` (e.g. 0.5 = 50% padding), floored at `refine_min_fov`
        degrees (so a tiny object still gets enough surrounding context) and
        capped at `max_fov` (the sweep's own `fov` - never WIDER than the
        coarse sweep already was, though still always re-centered even when
        capped, which alone already helps an object the coarse grid happened
        to cut off at its own tile edge). DINO+SAM2 then run on that view -
        every DetectedInstance in the final result comes from exactly one
        such call; there is no separate "coarse instance kept as a fallback"
        path any more, since the coarse pass never produces one. Reasoning:
        a coarse-pass SAM2 mask would only ever get thrown away the instant a
        real object gets re-segmented from a better-centered/zoomed view
        anyway, so running SAM2 at all in the coarse pass is pure waste once
        refine is on - one DINO+SAM2 call per real object, not two.

        `refine_box_threshold` is `box_threshold`'s own counterpart for JUST the
        refine pass's DINO calls (`text_threshold`/`sam_score_threshold` stay
        shared with the coarse sweep) - a re-centered, zoomed-in view naturally
        raises DINO's own box confidence for the SAME real object (it fills more
        of the frame, is less occluded by whatever's at the coarse tile's own
        edge, etc.), so reusing the coarse sweep's own (lower) `box_threshold`
        here would let through more low-quality/false-positive boxes than the
        coarse pass itself would have at that same threshold - a stricter
        default (0.45 vs. `box_threshold`'s own 0.3) keeps the refine pass held
        to a standard that accounts for that shift, rather than the coarse
        pass's own threshold effectively loosening the moment refine is on.

        `debug_dir`, if given, saves one annotated PNG per processed view (box
        rectangles + label/score text, plus a translucent colored mask overlay
        wherever a mask actually exists - see `_save_debug_view`) - None (the
        default) skips this entirely, at zero extra cost. Views land in two
        separate subfolders so the coarse pass and the refine pass stay
        independently traceable: `<debug_dir>/prefilter/` for the coarse
        pass's own views, `<debug_dir>/refine/` for the refine pass's. With
        `refine_detections` on, prefilter images show boxes ONLY (no mask -
        the coarse pass never runs SAM2 - see `refine_detections`'s own
        docstring), while refine images show the real box+mask result, same
        as always. A view that found nothing still gets saved (unannotated) -
        seeing which parts of the sweep genuinely found nothing is as useful
        for debugging as seeing what they did find.

        `cancel_check`, if given, is called before each view (same cadence as
        `progress_callback`) - returning True raises SweepCancelled immediately,
        discarding whatever partial results this call had gathered so far. Meant
        for a GUI's Cancel button; a sweep can take minutes, and DINO/SAM2 inference
        itself can't be interrupted mid-call, so between-view is the finest
        granularity available.
        """
        self.ensure_models_loaded()
        real_mask = real_point_mask(node)

        dino_panorama = reconstruct_panorama_for_channel(node, dino_channel)
        if dino_panorama is None:
            raise ValueError(f"Channel {dino_channel!r} is not available for this scan.")

        sam_panorama = dino_panorama if sam_channel == dino_channel else reconstruct_panorama_for_channel(node, sam_channel)
        if sam_panorama is None:
            raise ValueError(f"Channel {sam_channel!r} is not available for this scan.")

        h, w = dino_panorama.shape[:2]
        if view_grid is not None:
            views = view_grid
        else:
            phi_angles = phi_angles or [0.0]
            views = [(theta, phi) for phi in phi_angles for theta in range(0, 360, step_deg)]

        result = SweepResult()
        prefilter_debug_dir = os.path.join(debug_dir, "prefilter") if debug_dir else None
        refine_debug_dir = os.path.join(debug_dir, "refine") if debug_dir else None
        pending_boxes: list[_DetectedBox] = []

        for i, view_entry in enumerate(views):
            # Most entries are (theta, phi), using this call's own `fov` - a
            # (theta, phi, view_fov) entry (see compute_multiscale_coverage_grid)
            # overrides it for just this one view instead.
            if len(view_entry) == 3:
                theta, phi, view_fov = view_entry
            else:
                theta, phi = view_entry
                view_fov = fov

            if cancel_check is not None and cancel_check():
                raise SweepCancelled(f"Cancelled after {i}/{len(views)} views.")
            if progress_callback:
                progress_callback(int((i / len(views)) * 100),
                                   f"view {i + 1}/{len(views)} (theta={theta:.0f} deg, phi={phi:.0f} deg, "
                                   f"fov={view_fov:.0f} deg)")

            debug_name = f"view_{i:03d}_theta{theta:.0f}_phi{phi:.0f}_fov{view_fov:.0f}"
            if refine_detections:
                # DINO only - no SAM2 here at all (see _detect_boxes_view's
                # own docstring): whatever this coarse pass would have
                # segmented gets thrown away the moment _refine_boxes
                # re-segments the same object from a better-centered view
                # anyway, so running SAM2 on it here would be pure waste.
                pending_boxes.extend(self._detect_boxes_view(
                    dino_panorama, dino_channel, h, w, float(theta), float(phi), float(view_fov), out_size,
                    prompt, box_threshold, text_threshold, max_box_area_fraction,
                    debug_dir=prefilter_debug_dir, debug_name=debug_name,
                ))
            else:
                result.instances.extend(self._process_view(
                    node, dino_panorama, sam_panorama, dino_channel, sam_channel, h, w,
                    float(theta), float(phi), float(view_fov), out_size, prompt, box_threshold, text_threshold,
                    sam_score_threshold, real_mask, max_box_area_fraction,
                    debug_dir=prefilter_debug_dir, debug_name=debug_name,
                ))

        if refine_detections:
            before = len(pending_boxes)
            # max_fov here is always this call's own BASE `fov`, even for a
            # box that actually came from a multiscale far/near pass view at
            # a different fov (see view_grid's own docstring) - _DetectedBox
            # doesn't track which view produced it, so this is a deliberate
            # simplification: worst case a refine view ends up no tighter
            # than that specific view's own (larger) fov would have allowed,
            # never an incorrect action (see _refine_boxes: every box still
            # gets a real, better-centered SAM2 pass regardless).
            result.instances = self._refine_boxes(
                node, pending_boxes, dino_panorama, sam_panorama, dino_channel, sam_channel,
                h, w, out_size, prompt, refine_box_threshold, text_threshold, sam_score_threshold, real_mask,
                margin=refine_margin, min_fov=refine_min_fov, max_fov=fov,
                max_box_area_fraction=max_box_area_fraction, debug_dir=refine_debug_dir,
                progress_callback=progress_callback, cancel_check=cancel_check,
            )
            logging.info(f"Refine pass: {before} candidate box(es) -> {len(result.instances)} instance(s).")
        elif result.instances:
            before_points = sum(len(inst.indices) for inst in result.instances)
            result.instances = _filter_instance_outliers(node, result.instances)
            after_points = sum(len(inst.indices) for inst in result.instances)
            logging.info(f"Outlier removal: {before_points} -> {after_points} points "
                         f"across {len(result.instances)} instances.")

        if merge_duplicates and result.instances:
            before = len(result.instances)
            result.instances = _merge_duplicate_instances(result.instances, merge_iou, merge_overlap, merge_rounds)
            logging.info(f"Merged duplicates: {before} -> {len(result.instances)} instances.")

        if min_score > 0.0 and result.instances:
            before = len(result.instances)
            result.instances = [inst for inst in result.instances if inst.score >= min_score]
            dropped = before - len(result.instances)
            if dropped:
                logging.info(f"Score filter: dropped {dropped}/{before} instance(s) below min_score={min_score}.")

        if progress_callback:
            progress_callback(100, f"Sweep complete: {len(result.instances)} instances")

        logging.info(f"Sweep complete: {len(result.instances)} instances detected across {len(views)} views.")
        return result

    def _process_view(
        self, node: ScanNode, dino_panorama: np.ndarray, sam_panorama: np.ndarray,
        dino_channel: Channel, sam_channel: Channel, h: int, w: int,
        theta: float, phi: float, fov: float, out_size: int, prompt: str,
        box_threshold: float, text_threshold: float, sam_score_threshold: float,
        real_mask: np.ndarray, max_box_area_fraction: float = 0.95,
        debug_dir: str | None = None, debug_name: str | None = None,
    ) -> list[DetectedInstance]:
        """
        DINO+SAM2 on ONE rendered view, back-projected to 3D - the exact per-view
        body `run_sweep`'s own loop used to run inline, factored out so
        `_refine_boxes` can run the identical logic on an extra, adaptively
        placed view instead of only the coarse sweep's own grid (see
        `refine_detections`). Behavior-preserving extraction (when `debug_dir`
        is None, its own default) - this must stay byte-for-byte equivalent to
        what the loop did before, since it's still the ONLY code path a normal
        (non-refine, non-debug) sweep, and the refine pass itself, run.

        `debug_dir`/`debug_name`, if given (see `run_sweep`'s own `debug_dir`),
        save one annotated PNG of this view via `_save_debug_view` - a no-op,
        zero-cost branch whenever `debug_dir` is None.
        """
        map_x, map_y = get_face_projection_map(h, w, theta, phi, out_size, fov)
        dino_view = cv2.remap(dino_panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        boxes, scores, labels = self._detect_all_phrases(
            dino_view, prompt, box_threshold, text_threshold, max_box_area_fraction,
        )
        if len(boxes) == 0:
            if debug_dir is not None:
                _save_debug_view(dino_view, dino_channel, [], [], [], [], [], debug_dir, debug_name)
            return []

        sam_view = dino_view if sam_channel == dino_channel else cv2.remap(
            sam_panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
        )

        masks, sam_scores, _ = self.sam.predict(sam_view, box=boxes, multimask_output=False)
        if masks.ndim == 3:
            masks = masks[:, np.newaxis]
        # One score per box regardless of whether predict() returned it as (N,) or (N, 1).
        sam_scores = np.ravel(sam_scores)

        colors = generate_label_color(labels)
        instances = []
        debug_boxes, debug_masks, debug_labels, debug_scores, debug_colors = [], [], [], [], []
        for j in range(masks.shape[0]):
            if sam_scores[j] < sam_score_threshold:
                continue
            mask_2d = masks[j, 0].astype(bool)
            indices_3d = node.index_mapper.apply_mask(mask_2d, map_x, map_y, fov)
            indices_3d = indices_3d[real_mask[indices_3d]]
            if len(indices_3d) == 0:
                continue
            label = labels[j] if j < len(labels) else "unknown"
            score = float(scores[j]) if j < len(scores) else 0.0
            color = colors[j] if j < len(colors) else (128, 128, 128)
            instances.append(DetectedInstance(indices=indices_3d, label=label, score=score, color=color))
            if debug_dir is not None:
                debug_boxes.append(boxes[j])
                debug_masks.append(mask_2d)
                debug_labels.append(label)
                debug_scores.append(score)
                debug_colors.append(color)

        if debug_dir is not None:
            _save_debug_view(sam_view, sam_channel, debug_boxes, debug_labels, debug_scores,
                              debug_masks, debug_colors, debug_dir, debug_name)

        return instances

    def _detect_boxes_view(
        self, dino_panorama: np.ndarray, dino_channel: Channel, h: int, w: int,
        theta: float, phi: float, fov: float, out_size: int, prompt: str,
        box_threshold: float, text_threshold: float, max_box_area_fraction: float,
        debug_dir: str | None = None, debug_name: str | None = None,
    ) -> list[_DetectedBox]:
        """
        DINO-only pass on ONE rendered view - the coarse-pass counterpart of
        `_process_view` used instead of it when `refine_detections` is on
        (see its own docstring for why SAM2 is skipped here entirely). Each
        kept box's own angular extent (see `_box_angular_extent`) is returned
        as a `_DetectedBox`, not yet a real 3D detection - `_refine_boxes`
        turns these into actual DetectedInstances via a real DINO+SAM2 pass.
        """
        map_x, map_y = get_face_projection_map(h, w, theta, phi, out_size, fov)
        dino_view = cv2.remap(dino_panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        boxes, scores, labels = self._detect_all_phrases(
            dino_view, prompt, box_threshold, text_threshold, max_box_area_fraction,
        )
        if debug_dir is not None:
            colors = generate_label_color(labels) if len(labels) else []
            _save_debug_view(dino_view, dino_channel, list(boxes), labels, [float(s) for s in scores],
                              [], colors, debug_dir, debug_name)
        if len(boxes) == 0:
            return []

        detected = []
        for box, score, label in zip(boxes, scores, labels):
            extent = _box_angular_extent(map_x, map_y, box, h, w)
            if extent is None:
                continue
            theta_lo, theta_hi, phi_lo, phi_hi = extent
            detected.append(_DetectedBox(theta_lo, theta_hi, phi_lo, phi_hi, label, float(score)))
        return detected

    def _refine_boxes(
        self, node: ScanNode, boxes: list[_DetectedBox], dino_panorama: np.ndarray, sam_panorama: np.ndarray,
        dino_channel: Channel, sam_channel: Channel, h: int, w: int, out_size: int, prompt: str,
        box_threshold: float, text_threshold: float, sam_score_threshold: float, real_mask: np.ndarray,
        margin: float, min_fov: float, max_fov: float, max_box_area_fraction: float,
        debug_dir: str | None = None, progress_callback=None, cancel_check=None,
    ) -> list[DetectedInstance]:
        """
        The real DINO+SAM2 pass, run exactly once per coarse-detected box (see
        `_detect_boxes_view`), on a view re-centered on that box's own middle
        and sized to just cover its extent plus `margin`, floored at `min_fov`
        and capped at `max_fov` - always re-centered regardless of whether
        capping means no actual zoom happens, since re-centering ALONE already
        helps a box the coarse grid happened to cut off at its own tile edge.
        Every DetectedInstance in the final result comes from here - there is
        no separate "coarse instance, kept if this fails" fallback, since the
        coarse pass never produced one in the first place (see
        `refine_detections`'s own docstring for the full reasoning).

        Deliberately one `_process_view` call per box, not batched across
        views: SAM2's own batched image-encoder call (set_image_batch) holds
        every batched image's embeddings in VRAM at once, and live-measured
        this capping out on real hardware for a scan with many candidate
        boxes - one full DINO+SAM2 call per box, sequentially, keeps peak
        VRAM usage flat regardless of how many boxes a sweep finds.
        """
        instances: list[DetectedInstance] = []
        for i, box in enumerate(boxes):
            if cancel_check is not None and cancel_check():
                raise SweepCancelled(f"Cancelled during refine pass after {i}/{len(boxes)} box(es).")
            if progress_callback:
                progress_callback(95, f"refine box {i + 1}/{len(boxes)}")

            extent = max(box.theta_hi - box.theta_lo, box.phi_hi - box.phi_lo)
            target_fov = extent * (1.0 + margin)
            refine_fov = min(max(target_fov, min_fov), max_fov)
            theta_center = (box.theta_lo + box.theta_hi) / 2.0
            phi_center = (box.phi_lo + box.phi_hi) / 2.0

            instances.extend(self._process_view(
                node, dino_panorama, sam_panorama, dino_channel, sam_channel, h, w,
                theta_center, phi_center, refine_fov, out_size, prompt, box_threshold, text_threshold,
                sam_score_threshold, real_mask, max_box_area_fraction, debug_dir=debug_dir,
                debug_name=f"refine_{i:03d}_theta{theta_center:.0f}_phi{phi_center:.0f}_fov{refine_fov:.0f}",
            ))

        if instances:
            instances = _filter_instance_outliers(node, instances)
        return instances


@dataclass
class _DetectedBox:
    """A coarse-pass DINO box's own angular extent (see `_box_angular_extent`)
    plus its label/score - the DINO-only coarse pass's output when
    `refine_detections` is on, consumed by `_refine_boxes` to build the actual
    re-centered view each box gets its real DINO+SAM2 pass on."""
    theta_lo: float
    theta_hi: float
    phi_lo: float
    phi_hi: float
    label: str
    score: float


def _box_angular_extent(map_x: np.ndarray, map_y: np.ndarray, box, h: int, w: int,
                         n_samples: int = 8) -> tuple[float, float, float, float] | None:
    """
    (theta_lo, theta_hi, phi_lo, phi_hi) for a 2D DINO box, given in VIEW
    pixel space, via the SAME map_x/map_y remap coordinates that produced the
    view it was detected on. Samples points along the box's own PERIMETER
    (not just its 4 corners) since a straight-edged box in rectilinear pixel
    space maps to a CURVED region in angular space away from the view's own
    center - 4 corners alone can under-estimate the true extent; a perimeter
    sample catches that curvature.

    theta is unwrapped relative to the box's own first sample (same 0/360-
    seam handling `_compute_refine_view` used for mask-index-derived extents
    before this - see git history) - safe since a single DINO box's real
    extent is always far under 180 degrees.

    Returns None if every sampled pixel lands outside `map_x`/`map_y`'s own
    panorama bounds - shouldn't normally happen for a box GroundingDINO
    itself derived from this same view, but guarded rather than assumed.
    """
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    out_h, out_w = map_x.shape[:2]

    xs = np.linspace(x0, x1, n_samples)
    ys = np.linspace(y0, y1, n_samples)
    # Perimeter only (top+bottom edges, left+right edges) - cheap and
    # sufficient to bound a convex-ish region; no need to sample the interior.
    px = np.concatenate([xs, xs, np.full(n_samples, x0), np.full(n_samples, x1)])
    py = np.concatenate([np.full(n_samples, y0), np.full(n_samples, y1), ys, ys])
    px = np.clip(px.astype(np.int32), 0, out_w - 1)
    py = np.clip(py.astype(np.int32), 0, out_h - 1)

    pano_x = map_x[py, px]
    pano_y = map_y[py, px]
    valid = (pano_x >= 0) & (pano_x < w) & (pano_y >= 0) & (pano_y < h)
    if not valid.any():
        return None
    pano_x, pano_y = pano_x[valid], pano_y[valid]

    theta_pts = 360.0 * (pano_x / w - 0.5)
    phi_pts = 180.0 * (pano_y / h - 0.5)
    theta_ref = theta_pts[0]
    theta_pts = theta_ref + (((theta_pts - theta_ref + 180.0) % 360.0) - 180.0)

    return float(theta_pts.min()), float(theta_pts.max()), float(phi_pts.min()), float(phi_pts.max())


def _save_debug_view(
    image: np.ndarray, channel: Channel, boxes: list, labels: list[str], scores: list[float],
    masks: list[np.ndarray], colors: list[tuple], out_dir: str, name: str | None,
) -> None:
    """
    Writes one PNG of `image` (a rendered sweep view - see `_process_view`/
    `_detect_boxes_view`) with each KEPT detection's box (rectangle +
    "label score" text) drawn on top, PLUS a translucent color fill for each
    mask if any were given - `masks` is empty for a DINO-only coarse-pass
    view under `refine_detections` (see `_detect_boxes_view`), where boxes
    still draw on their own. Empty `boxes`/`masks`/etc. just saves the plain
    view, so a view that found nothing is still visible in the debug output,
    not silently missing. `colors` are the SAME per-detection colors already
    assigned to the resulting DetectedInstance where one exists (not
    regenerated here) - deterministic from the label otherwise (see
    core.colors.generate_label_color) - so a box in this image and the
    object it becomes in the 3D view are recognizably the same color even
    across the coarse (box-only) and refine (box+mask) debug folders.

    `channel` decides one color-space fix: `reconstruct_panorama_for_channel`
    leaves an "rgb"-channel panorama in RGB order but explicitly converts
    "intensity"/"normals" to BGR (see its own docstring) - cv2.imwrite always
    wants BGR, so only the "rgb" case needs converting here.
    """
    os.makedirs(out_dir, exist_ok=True)
    canvas = image.copy()
    if channel == "rgb":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if masks:
        overlay = canvas.copy()
        for mask, color in zip(masks, colors):
            overlay[mask] = color
        canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)
    for box, label, score, color in zip(boxes, labels, scores, colors):
        bgr = tuple(int(c) for c in color)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), bgr, 2)
        cv2.putText(canvas, f"{label} {score:.2f}", (x0, max(12, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    file_name = f"{name}.png" if name else f"view_{id(image):x}.png"
    cv2.imwrite(os.path.join(out_dir, file_name), canvas)


_OUTLIER_MP_MIN_POINTS = 5_000  # below this, an instance runs inline - pool/IPC overhead isn't worth it
_OUTLIER_MP_MIN_INSTANCES = 2   # only spin up a pool if there's more than one big-enough instance to split across it


def _filter_instance_outliers(node: ScanNode, instances: list[DetectedInstance]) -> list[DetectedInstance]:
    """
    Runs every instance through filter_mask_outliers (voxel-connectivity + Local
    Outlier Factor - see core.mask_filters), each against only its own points,
    never the whole scan - a mask that leaked onto a spatially
    disjoint surface gets trimmed back down to its real object regardless of
    whether merge_duplicates is on. Drops any instance that ends up with zero
    points left (shouldn't normally happen - filter_mask_outliers always keeps
    at least the largest connected component - but a truly degenerate/tiny
    instance is possible in principle).
    """
    filtered = []
    for inst, indices in zip(instances, _run_outlier_filtering(node, instances)):
        if len(indices) > 0:
            filtered.append(DetectedInstance(indices=indices, label=inst.label, score=inst.score, color=inst.color))
    return filtered


def _run_outlier_filtering(node: ScanNode, instances: list[DetectedInstance]) -> list[np.ndarray]:
    """
    Filters every instance (see filter_mask_outliers), returned in the SAME
    order as `instances`. Instances at or above `_OUTLIER_MP_MIN_POINTS` run
    in a ProcessPoolExecutor, in parallel across CPU cores - the actual
    filter (core.mask_filters._filter_outliers_core) has no ScanNode
    dependency at all (see its own docstring), so only each instance's own
    already-extracted points_3d/indices pair has to cross the process
    boundary, never the whole (potentially huge) scan. That matters
    specifically on Windows, where multiprocessing always starts each worker
    as a fresh interpreter (spawn, not fork) rather than a cheap
    copy-on-write fork - re-pickling a large ScanNode per task there would
    likely cost more than the parallelism saves. Smaller instances run
    inline, serially, regardless - not worth a worker round-trip.

    Falls back to filtering EVERYTHING inline/serially (discarding any
    partial pool results) if the pool itself raises for any reason (e.g. a
    sandboxed environment with no process-spawn permission) - an
    optimization failing shouldn't fail the whole sweep.
    """
    results: list[np.ndarray | None] = [None] * len(instances)
    big_positions = [i for i, inst in enumerate(instances) if len(inst.indices) >= _OUTLIER_MP_MIN_POINTS]

    if len(big_positions) >= _OUTLIER_MP_MIN_INSTANCES:
        try:
            points_by_pos = {i: _to_points_3d(node, instances[i].indices) for i in big_positions}
            max_workers = min(len(big_positions), os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_filter_outliers_core, points_by_pos[i], instances[i].indices): i
                    for i in big_positions
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
        except Exception as e:
            logging.warning(f"Parallel outlier filtering failed ({e!r}) - falling back to serial.")
            results = [None] * len(instances)

    for i, inst in enumerate(instances):
        if results[i] is None:
            results[i] = filter_mask_outliers(node, inst.indices)
    return results


def _merge_duplicate_instances(instances: list[DetectedInstance], merge_iou: float,
                                merge_overlap: float | None = None,
                                merge_rounds: int = 3) -> list[DetectedInstance]:
    """
    Deduplicates same-label instances via 3D IoU (see core.merge). Grouped by
    label first since merge_by_3d_iou has no label concept of its own - it would
    happily merge a "tree" into an overlapping "building" if asked to compare
    across labels, which is never what we want here.
    """
    by_label: dict[str, list[DetectedInstance]] = {}
    for inst in instances:
        by_label.setdefault(inst.label, []).append(inst)

    merged_instances: list[DetectedInstance] = []
    for label, group in by_label.items():
        merged_arrays = merge_by_3d_iou_iterative(
            [inst.indices for inst in group], max_rounds=merge_rounds, iou_threshold=merge_iou,
            overlap_coeff_threshold=merge_overlap,
        )
        for merged_indices in merged_arrays:
            merged_set = set(merged_indices.tolist())
            # The merged point set is a union of overlapping fragments of the same
            # physical object (each sweep view only ever saw and scored its own
            # slice of it), not a fresh single detection - there's no one
            # "correct" score for it, so it's the plain average of the original
            # instances that actually went into this cluster (matched by index
            # overlap with the merged set; falls back to the whole label group in
            # the - extremely unlikely - case a border/spatial merge rule joined
            # two instances with zero direct index overlap). Deliberately not
            # weighted by each contributor's point count or per-pixel confidence -
            # a simple mean of the DINO confidences involved.
            contributors = [inst for inst in group if merged_set & set(inst.indices.tolist())] or group
            avg_score = sum(inst.score for inst in contributors) / len(contributors)
            merged_instances.append(DetectedInstance(
                indices=merged_indices, label=label, score=avg_score, color=contributors[0].color,
            ))
    return merged_instances
