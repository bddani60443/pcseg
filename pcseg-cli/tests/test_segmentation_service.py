"""
Regression test for a real recall problem found while testing on an actual scan:
passing a multi-category prompt (e.g. "window, building") to GroundingDINO as one
caption measurably suppresses recall for the weaker category - "window" found
~3x fewer boxes combined with "building" in one caption than queried alone at
the same thresholds. SegmentationService._detect_all_phrases splits the prompt
on "," and queries each phrase separately, then merges - this locks in that a
multi-phrase prompt results in one DINO call per phrase, not one call for the
whole caption.

segmentation_service.py imports dino_service.py, which imports torch at module
scope (needed for the real model wrapper) - so, like the rest of the
segmentation-backend code, this test requires the optional `segmentation`
extra and is skipped (not failed) without it.
"""
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch")

from pcseg.core.mapping import IndexMapper  # noqa: E402
from pcseg.core.projection import get_face_projection_map  # noqa: E402
from pcseg.core.scan_node import ScanNode  # noqa: E402
from pcseg.exceptions import SweepCancelled  # noqa: E402
from pcseg.services.segmentation_service import (  # noqa: E402
    DetectedInstance,
    SegmentationService,
    _box_angular_extent,
    _filter_instance_outliers,
    _merge_duplicate_instances,
    _run_outlier_filtering,
)


def _make_node(distance: float = 100.0) -> ScanNode:
    """`distance` places every point that far from the origin along x - real (non-degenerate)
    by default; pass 0.0 to simulate a scanner's "no return" placeholder points."""
    h, w = 8, 8
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows, cols = rows.ravel().astype(np.int32), cols.ravel().astype(np.int32)
    n = h * w
    mapper = IndexMapper({"row": rows, "col": cols}, row_max=h - 1, col_max=w - 1)
    points = np.zeros((n, 3), dtype=np.int16)
    points[:, 0] = int(distance)
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=points,
        quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.tile(np.array([0, 0, 127], dtype=np.int8), (n, 1)),
        colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=1,
        available_modes=["normals"], row_max=h, col_max=w,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=mapper,
    )


class _FakeDino:
    """Records every run_inference call and returns one fixed box per phrase."""

    def __init__(self):
        self.calls: list[str] = []
        self.box_threshs: list[float] = []  # parallel to self.calls - see run_inference

    def _ensure_loaded(self) -> None:
        pass

    def run_inference(self, view, prompt, box_thresh, text_thresh):
        self.calls.append(prompt)
        self.box_threshs.append(box_thresh)
        box = np.array([[1, 1, 5, 5]])
        score = np.array([0.9])
        return box, score, [prompt]


class _FakeDinoWithEmptyPhrase:
    """Simulates GroundingDINO returning a box whose box_threshold cleared but no
    token cleared text_threshold - a real, reproducible library behavior, not a
    pcseg-cli bug (see _detect_all_phrases's comment)."""

    def _ensure_loaded(self) -> None:
        pass

    def run_inference(self, view, prompt, box_thresh, text_thresh):
        boxes = np.array([[1, 1, 5, 5], [2, 2, 6, 6]])
        scores = np.array([0.9, 0.7])
        return boxes, scores, [prompt, ""]


class _FakeSam:
    """Fills each candidate mask with exactly the box it was given (clipped
    to the view's own bounds) - zero overflow by construction, regardless of
    which box a particular test uses, so select_best_box_mask always accepts
    it without needing per-test tuning. Honors `multimask_output` (3
    candidates vs. 1) the same way the real SAM2 predictor does."""

    def __init__(self, score: float = 1.0):
        self.score = score

    def _ensure_loaded(self) -> None:
        pass

    def predict(self, view, box, multimask_output=False):
        n = len(box)
        h, w = view.shape[:2]
        c = 3 if multimask_output else 1
        masks = np.zeros((n, c, h, w), dtype=bool)
        for i in range(n):
            x0, y0, x1, y1 = box[i]
            x0, x1 = sorted((int(round(x0)), int(round(x1))))
            y0, y1 = sorted((int(round(y0)), int(round(y1))))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                masks[i, :, y0:y1, x0:x1] = True
        return masks, np.full((n, c), self.score), None


def test_run_sweep_queries_each_comma_separated_phrase_separately():
    node = _make_node()
    dino = _FakeDino()
    svc = SegmentationService(dino, _FakeSam())

    result = svc.run_sweep(
        node, prompt="window, building, tree",
        step_deg=360, phi_angles=[0.0],  # a single view is enough to check the call pattern
        dino_channel="normals", sam_channel="normals",
    )

    # One DINO call per phrase per view, not one call with the whole caption.
    assert dino.calls == ["window", "building", "tree"]
    assert {inst.label for inst in result.instances} == {"window", "building", "tree"}


class _FakeDinoConfigurableBox:
    """Returns exactly ONE box per call, at the given [x0, y0, x1, y1] pixel
    coordinates - lets a test control a box's own area relative to the
    view's shape directly, instead of depending on _FakeDino's fixed
    small corner box."""

    def __init__(self, box):
        self.box = box
        self.calls: list[str] = []

    def _ensure_loaded(self) -> None:
        pass

    def run_inference(self, view, prompt, box_thresh, text_thresh):
        self.calls.append(prompt)
        return np.array([self.box], dtype=np.float64), np.array([0.9]), [prompt]


def test_detect_all_phrases_drops_a_box_covering_the_entire_view():
    view = np.zeros((100, 100, 3), dtype=np.uint8)
    svc = SegmentationService(_FakeDinoConfigurableBox([0, 0, 100, 100]), _FakeSam())

    boxes, _scores, _labels = svc._detect_all_phrases(view, "window", box_threshold=0.3, text_threshold=0.25)

    assert len(boxes) == 0


def test_detect_all_phrases_keeps_a_normal_sized_box():
    view = np.zeros((100, 100, 3), dtype=np.uint8)
    svc = SegmentationService(_FakeDinoConfigurableBox([10, 10, 40, 40]), _FakeSam())  # 9% of the view

    boxes, _scores, _labels = svc._detect_all_phrases(view, "window", box_threshold=0.3, text_threshold=0.25)

    assert len(boxes) == 1


def test_detect_all_phrases_respects_a_custom_max_box_area_fraction():
    view = np.zeros((100, 100, 3), dtype=np.uint8)
    # A 60x60 box covers 36% of the view - below the default 0.95 cutoff
    # (kept), but above a stricter, explicitly requested 0.3.
    svc = SegmentationService(_FakeDinoConfigurableBox([0, 0, 60, 60]), _FakeSam())

    default_boxes, _s, _l = svc._detect_all_phrases(view, "window", box_threshold=0.3, text_threshold=0.25)
    strict_boxes, _s, _l = svc._detect_all_phrases(
        view, "window", box_threshold=0.3, text_threshold=0.25, max_box_area_fraction=0.3,
    )

    assert len(default_boxes) == 1
    assert len(strict_boxes) == 0


def test_run_sweep_drops_a_detection_whose_box_covers_the_entire_view():
    """End-to-end: the same whole-frame-box case, through run_sweep - the
    reported problem was DINO occasionally claiming an entire crop is one
    object, which must never survive as a detected instance."""
    node = _make_node()
    svc = SegmentationService(_FakeDinoConfigurableBox([0, 0, 1024, 1024]), _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert result.instances == []


def test_run_sweep_drops_masks_that_only_cover_degenerate_placeholder_points():
    """
    Regression test: many TLS scanners fill a "no return" grid cell (e.g. open
    sky) with a degenerate point at the scanner's own origin instead of
    leaving it empty (see core.projection.real_point_mask) - a detected mask
    that only covers such points isn't real geometry and must be dropped
    entirely, not counted as a genuine instance.
    """
    node = _make_node(distance=0.0)  # every point is a degenerate "no return" placeholder
    svc = SegmentationService(_FakeDino(), _FakeSam())

    result = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert result.instances == []


def test_run_sweep_drops_masks_below_the_sam_score_threshold():
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam(score=0.5))

    kept = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        sam_score_threshold=0.4,
    )
    dropped = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        sam_score_threshold=0.6,
    )

    assert len(kept.instances) == 1
    assert dropped.instances == []


def test_run_sweep_drops_final_instances_below_min_score():
    """min_score is the DINO-confidence counterpart to sam_score_threshold - _FakeDino
    always scores 0.9, so a threshold just below keeps it, just above drops it."""
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    kept = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        min_score=0.5,
    )
    dropped = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        min_score=0.95,
    )

    assert len(kept.instances) == 1
    assert dropped.instances == []


def test_run_sweep_min_score_is_off_by_default():
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    result = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert len(result.instances) == 1


def test_run_sweep_min_score_applies_to_the_merged_averaged_score():
    """min_score must be applied AFTER merge_duplicates, judged by the merged instance's
    final averaged score, not by either raw fragment's own score."""
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())
    view_grid = [(0.0, 0.0), (0.0, 0.0)]

    result = svc.run_sweep(
        node, prompt="window", dino_channel="normals", sam_channel="normals", view_grid=view_grid,
        merge_duplicates=True, min_score=0.5,
    )

    assert len(result.instances) == 1  # _FakeDino's fixed 0.9 score clears 0.5 after averaging


def test_run_sweep_raises_sweep_cancelled_when_cancel_check_returns_true():
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    with pytest.raises(SweepCancelled):
        svc.run_sweep(
            node, prompt="window",
            step_deg=90, phi_angles=[0.0],  # 4 views, so cancellation mid-sweep is observable
            dino_channel="normals", sam_channel="normals",
            cancel_check=lambda: True,
        )


def test_run_sweep_never_cancels_when_cancel_check_is_none():
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    result = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        cancel_check=None,
    )
    assert len(result.instances) == 1


def test_run_sweep_drops_boxes_whose_dino_phrase_is_empty():
    """A box that cleared box_threshold but matched no token above text_threshold
    comes back from GroundingDINO with an empty phrase - not a confirmed detection
    of anything the user asked for, so it must not surface as an "unlabeled"
    instance."""
    node = _make_node()
    svc = SegmentationService(_FakeDinoWithEmptyPhrase(), _FakeSam())

    result = svc.run_sweep(
        node, prompt="window",
        step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert len(result.instances) == 1
    assert result.instances[0].label == "window"


def test_run_sweep_merges_duplicate_instances_of_the_same_label_when_enabled():
    """Same (theta, phi) passed twice via an explicit view_grid renders identically both
    times, so the fake DINO/SAM produce identical detections - exactly what "the same
    physical object seen from two overlapping views" looks like in index space."""
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())
    view_grid = [(0.0, 0.0), (0.0, 0.0)]

    unmerged = svc.run_sweep(
        node, prompt="window", dino_channel="normals", sam_channel="normals", view_grid=view_grid,
    )
    merged = svc.run_sweep(
        node, prompt="window", dino_channel="normals", sam_channel="normals", view_grid=view_grid,
        merge_duplicates=True,
    )

    assert len(unmerged.instances) == 2
    assert len(merged.instances) == 1
    assert set(merged.instances[0].indices.tolist()) == set(unmerged.instances[0].indices.tolist())


def test_merge_duplicate_instances_never_merges_across_labels():
    idx = np.array([1, 2, 3])
    tree = DetectedInstance(indices=idx, label="tree", score=0.9, color=(0, 255, 0))
    building = DetectedInstance(indices=idx, label="building", score=0.8, color=(0, 0, 255))

    result = _merge_duplicate_instances([tree, building], merge_iou=0.5, merge_rounds=3)

    assert len(result) == 2
    assert {inst.label for inst in result} == {"tree", "building"}


def test_merge_duplicate_instances_overlap_rule_merges_very_different_sized_fragments():
    """Simulates the same real object seen near-complete in one view (0..99) and
    only a sliver of it in another (0..9) - Jaccard IoU never fires here
    (10/100 = 0.1, far below any sane merge_iou), but the smaller fragment is
    100% covered by the bigger one, which is exactly what merge_overlap should
    catch."""
    big = DetectedInstance(indices=np.arange(0, 100), label="window", score=0.8, color=(1, 1, 1))
    small = DetectedInstance(indices=np.arange(0, 10), label="window", score=0.6, color=(1, 1, 1))

    unmerged = _merge_duplicate_instances([big, small], merge_iou=0.5, merge_overlap=None, merge_rounds=3)
    merged = _merge_duplicate_instances([big, small], merge_iou=0.5, merge_overlap=0.9, merge_rounds=3)

    assert len(unmerged) == 2  # confirms Jaccard alone genuinely doesn't merge this pair
    assert len(merged) == 1
    assert set(merged[0].indices.tolist()) == set(range(100))


def test_merge_duplicate_instances_averages_the_contributing_scores():
    low = DetectedInstance(indices=np.array([1, 2, 3, 4]), label="window", score=0.4, color=(1, 1, 1))
    high = DetectedInstance(indices=np.array([2, 3, 4, 5]), label="window", score=0.9, color=(1, 1, 1))

    result = _merge_duplicate_instances([low, high], merge_iou=0.5, merge_rounds=3)

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.65)
    assert set(result[0].indices.tolist()) == {1, 2, 3, 4, 5}


def test_merge_duplicate_instances_leaves_non_overlapping_instances_separate():
    a = DetectedInstance(indices=np.array([1, 2, 3]), label="window", score=0.5, color=(1, 1, 1))
    b = DetectedInstance(indices=np.array([60, 61, 62]), label="window", score=0.6, color=(1, 1, 1))

    result = _merge_duplicate_instances([a, b], merge_iou=0.5, merge_rounds=3)

    assert len(result) == 2


def test_filter_instance_outliers_scopes_each_instance_to_itself_not_the_whole_scan():
    """A handful of stray points far from an instance's own main cluster (simulating a
    mask that leaked onto a spatially disjoint surface) should be dropped, evaluated
    against that instance's own points only - never against the whole scan."""
    node = _make_node(distance=0.0)  # every point sits at the same (0,0,0) position
    # Give the "leaked" points a distinct real position so they're spatially
    # separable from the rest, unlike every other point in this synthetic node.
    node.points_quantized[60:64] = np.array([[500, 500, 500]] * 4, dtype=np.int16)

    main_cluster = np.arange(0, 20)  # all at (0,0,0)
    leaked = np.array([60, 61, 62, 63])  # all at (500,500,500) - spatially disjoint
    inst = DetectedInstance(indices=np.concatenate([main_cluster, leaked]), label="window",
                             score=0.8, color=(1, 1, 1))

    result = _filter_instance_outliers(node, [inst])

    assert len(result) == 1
    result_indices = set(result[0].indices.tolist())
    assert set(leaked.tolist()) - result_indices  # at least some leaked points got dropped
    assert set(main_cluster.tolist()) <= result_indices  # the real cluster survives intact


def test_filter_instance_outliers_runs_independently_of_other_instances():
    """One instance's own leaked points shouldn't be judged against a completely
    different instance's points - each instance is filtered in isolation."""
    node = _make_node(distance=0.0)
    node.points_quantized[60:64] = np.array([[500, 500, 500]] * 4, dtype=np.int16)

    clean = DetectedInstance(indices=np.arange(0, 20), label="tree", score=0.7, color=(0, 1, 0))
    leaky = DetectedInstance(indices=np.concatenate([np.arange(20, 40), np.array([60, 61, 62, 63])]),
                              label="window", score=0.8, color=(1, 1, 1))

    result = _filter_instance_outliers(node, [clean, leaky])

    by_label = {inst.label: inst for inst in result}
    assert set(by_label["tree"].indices.tolist()) == set(clean.indices.tolist())  # untouched
    assert not ({60, 61, 62, 63} & set(by_label["window"].indices.tolist()))  # leak dropped


# ── refine_detections (DINO-only coarse pass + real DINO+SAM2 refine) ────

def _make_fine_node(h: int = 64, w: int = 64, distance: float = 100.0) -> ScanNode:
    """Like _make_node but at a much finer grid resolution - keeps the fixed-
    pixel fake SAM mask a small, realistic fraction of the panorama, matching
    what a real scan looks like (a few other tests below deliberately reuse
    the coarser _make_node instead, where that's the point)."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows, cols = rows.ravel().astype(np.int32), cols.ravel().astype(np.int32)
    n = h * w
    mapper = IndexMapper({"row": rows, "col": cols}, row_max=h - 1, col_max=w - 1)
    points = np.zeros((n, 3), dtype=np.int16)
    points[:, 0] = int(distance)
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=points,
        quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.tile(np.array([0, 0, 127], dtype=np.int8), (n, 1)),
        colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=1,
        available_modes=["normals"], row_max=h, col_max=w,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=mapper,
    )


def test_box_angular_extent_full_view_box_matches_the_views_own_fov():
    h, w, size, fov = 180, 360, 64, 90.0
    map_x, map_y = get_face_projection_map(h, w, theta=0.0, phi=0.0, size=size, fov=fov)

    result = _box_angular_extent(map_x, map_y, [0, 0, size - 1, size - 1], h, w)

    assert result is not None
    theta_lo, theta_hi, phi_lo, phi_hi = result
    # A box covering the WHOLE view should span close to its own fov in both
    # axes (measured: exactly 90 for theta, ~89.4 for phi on this grid).
    assert (theta_hi - theta_lo) >= fov * 0.9
    assert (phi_hi - phi_lo) >= fov * 0.9


def test_box_angular_extent_small_centered_box_is_small_and_correctly_centered():
    h, w, size, fov = 180, 360, 65, 90.0  # odd size -> exact center pixel
    map_x, map_y = get_face_projection_map(h, w, theta=30.0, phi=10.0, size=size, fov=fov)
    c = size // 2

    result = _box_angular_extent(map_x, map_y, [c - 1, c - 1, c + 1, c + 1], h, w)

    assert result is not None
    theta_lo, theta_hi, phi_lo, phi_hi = result
    assert abs((theta_lo + theta_hi) / 2 - 30.0) < 3.0
    assert abs((phi_lo + phi_hi) / 2 - 10.0) < 3.0
    assert (theta_hi - theta_lo) < 10.0
    assert (phi_hi - phi_lo) < 10.0


def test_box_angular_extent_unwraps_the_0_360_seam():
    """A view centered at theta=180 sits exactly on the panorama's own
    column-wrap seam - a box straddling it must still measure as the few
    degrees it really is, not the ~350-degree spread a naive (unwrapped)
    reading of the raw pano columns would produce."""
    h, w, size, fov = 180, 360, 65, 90.0
    map_x, map_y = get_face_projection_map(h, w, theta=180.0, phi=0.0, size=size, fov=fov)
    c = size // 2

    result = _box_angular_extent(map_x, map_y, [c - 2, c - 2, c + 2, c + 2], h, w)

    assert result is not None
    theta_lo, theta_hi, _phi_lo, _phi_hi = result
    assert (theta_hi - theta_lo) < 20.0  # correctly small (measured ~7), nowhere near ~350


def test_box_angular_extent_returns_none_when_entirely_outside_panorama_bounds():
    size = 8
    map_x = np.full((size, size), -1.0, dtype=np.float32)
    map_y = np.full((size, size), -1.0, dtype=np.float32)

    result = _box_angular_extent(map_x, map_y, [0, 0, size - 1, size - 1], 10, 20)

    assert result is None


class _FakeDinoThenEmpty:
    """First call returns a fixed box (the coarse pass's own box-finding
    call); every call after that finds nothing - simulates the refine pass
    failing to re-confirm a detection at its own (stricter) threshold."""

    def __init__(self):
        self.calls: list[str] = []

    def _ensure_loaded(self) -> None:
        pass

    def run_inference(self, view, prompt, box_thresh, text_thresh):
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return np.array([[1, 1, 5, 5]]), np.array([0.9]), [prompt]
        return np.array([]), np.array([]), []


class _CountingFakeSam:
    """Same fixed-mask behavior as _FakeSam, but counts predict() calls -
    used to prove the coarse pass makes ZERO SAM2 calls once
    refine_detections is on (see run_sweep's own docstring: SAM2 only ever
    runs from the refine pass in that mode, exactly once per candidate box)."""

    def __init__(self, score: float = 1.0):
        self.score = score
        self.predict_calls = 0

    def _ensure_loaded(self) -> None:
        pass

    def predict(self, view, box, multimask_output=False):
        self.predict_calls += 1
        n = len(box)
        h, w = view.shape[:2]
        c = 3 if multimask_output else 1
        masks = np.zeros((n, c, h, w), dtype=bool)
        for i in range(n):
            x0, y0, x1, y1 = box[i]
            x0, x1 = sorted((int(round(x0)), int(round(x1))))
            y0, y1 = sorted((int(round(y0)), int(round(y1))))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                masks[i, :, y0:y1, x0:x1] = True
        return masks, np.full((n, c), self.score), None


def test_run_sweep_refine_detections_never_calls_sam2_during_the_coarse_pass():
    node = _make_fine_node()
    dino = _FakeDino()
    sam = _CountingFakeSam()
    svc = SegmentationService(dino, sam)

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        refine_detections=True,
    )

    # Exactly one SAM2 call total, from the refine pass's own single box -
    # never from the coarse (DINO-only) pass.
    assert sam.predict_calls == 1


def test_run_sweep_refine_detections_false_still_calls_sam2_in_the_coarse_pass():
    """Isolates that skipping SAM2 in the coarse pass is refine_detections's
    own doing - without it, SAM2 runs there exactly as before."""
    node = _make_fine_node()
    dino = _FakeDino()
    sam = _CountingFakeSam()
    svc = SegmentationService(dino, sam)

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert sam.predict_calls == 1  # the one (and only, refine is off) coarse-pass call


def test_run_sweep_refine_detections_true_produces_real_instances_via_the_refine_pass():
    node = _make_fine_node()
    dino = _FakeDino()
    svc = SegmentationService(dino, _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        refine_detections=True,
    )

    # One coarse DINO-only call, one refine DINO+SAM2 call - proves the
    # second pass actually ran and produced the final result.
    assert dino.calls == ["window", "window"]
    assert len(result.instances) >= 1


def test_run_sweep_refine_detections_off_by_default_matches_pre_refine_behavior():
    """The default (refine_detections unset) must run exactly the coarse
    sweep, one DINO call - no behavior change for existing callers."""
    node = _make_node()
    dino = _FakeDino()
    svc = SegmentationService(dino, _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert dino.calls == ["window"]
    assert len(result.instances) == 1


def test_run_sweep_refine_pass_uses_its_own_higher_box_threshold():
    """A zoomed-in refine view naturally raises DINO's own box confidence for
    the same real object, so the refine pass must be judged against its OWN
    (by default stricter) box_threshold, not the coarse pass's - otherwise
    turning refine_detections on would effectively loosen the coarse
    threshold's own guarantee."""
    node = _make_fine_node()
    dino = _FakeDino()
    svc = SegmentationService(dino, _FakeSam())

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        box_threshold=0.3, refine_detections=True, refine_box_threshold=0.45,
    )

    assert dino.box_threshs == [0.3, 0.45]  # coarse call, then the refine call


def test_run_sweep_refine_box_threshold_defaults_higher_than_box_threshold():
    node = _make_fine_node()
    dino = _FakeDino()
    svc = SegmentationService(dino, _FakeSam())

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        refine_detections=True,  # box_threshold/refine_box_threshold both left at their own defaults
    )

    coarse_thresh, refine_thresh = dino.box_threshs
    assert refine_thresh > coarse_thresh


def test_run_sweep_refine_detections_false_makes_no_extra_call_on_the_same_fine_node():
    """Same node/prompt as the test above, refine off - isolates that the
    extra call really is refine_detections's own doing, not something about
    the finer grid."""
    node = _make_fine_node()
    dino = _FakeDino()
    svc = SegmentationService(dino, _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert dino.calls == ["window"]
    assert len(result.instances) == 1


def test_run_sweep_refine_detections_drops_a_box_the_refine_pass_could_not_confirm():
    """Once refine_detections is on, a box that WAS looked at again but
    didn't clear the refine pass's own (stricter) bar must be dropped
    entirely - there's no coarse SAM2 mask left to fall back on any more
    (the coarse pass never produces one), so "not confirmed" now means
    "absent," not "kept at its old, lower-confidence quality"."""
    node = _make_fine_node()
    dino = _FakeDinoThenEmpty()
    svc = SegmentationService(dino, _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        refine_detections=True,
    )

    assert len(dino.calls) == 2  # coarse box-finding call, then the failed refine call
    assert len(result.instances) == 0


def test_run_sweep_refine_detections_clamps_but_never_skips_a_box_that_fills_the_whole_view():
    """A box whose own extent already needs max_fov or more must still get a
    real, re-centered SAM2 pass - clamped to max_fov, never skipped outright.
    Unlike the old mask-index-based design, there is no coarse SAM2 mask left
    to fall back on if this one were skipped, so silently dropping it would
    just lose the object entirely."""
    node = _make_fine_node()
    dino = _FakeDinoConfigurableBox([10, 10, 900, 900])  # ~77% of a 1024x1024 view - large, real angular extent
    svc = SegmentationService(dino, _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        refine_detections=True,
    )

    assert dino.calls == ["window", "window"]  # refine still ran - no silent skip
    assert len(result.instances) >= 1  # still produced something, not silently empty


# ── debug_dir ────────────────────────────────────────────────────────────

def test_run_sweep_debug_dir_none_writes_nothing(tmp_path):
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
    )

    assert list(tmp_path.iterdir()) == []  # nothing written anywhere near tmp_path either


def test_run_sweep_debug_dir_writes_prefilter_images(tmp_path):
    node = _make_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        debug_dir=str(tmp_path),
    )

    prefilter_dir = tmp_path / "prefilter"
    assert prefilter_dir.is_dir()
    images = list(prefilter_dir.glob("*.png"))
    assert len(images) == 1  # one coarse view
    assert not (tmp_path / "refine").exists()  # refine_detections was off


def test_run_sweep_debug_dir_writes_separate_prefilter_and_refine_folders(tmp_path):
    node = _make_fine_node()
    svc = SegmentationService(_FakeDino(), _FakeSam())

    svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        refine_detections=True, debug_dir=str(tmp_path),
    )

    prefilter_images = list((tmp_path / "prefilter").glob("*.png"))
    refine_images = list((tmp_path / "refine").glob("*.png"))
    assert len(prefilter_images) == 1
    assert len(refine_images) == 1
    # Different folders, so no risk of a refine view overwriting a same-named coarse one.
    assert prefilter_images[0].parent != refine_images[0].parent


def test_run_sweep_debug_dir_saves_an_image_even_for_an_empty_view(tmp_path):
    """A view that found nothing must still show up in the debug output -
    otherwise a genuinely empty region of the sweep looks identical to one
    that was simply never rendered, which defeats the point of a coverage
    trace."""
    node = _make_node()

    class _EmptyDino:
        def _ensure_loaded(self):
            pass

        def run_inference(self, view, prompt, box_thresh, text_thresh):
            return np.array([]), np.array([]), []

    svc = SegmentationService(_EmptyDino(), _FakeSam())

    result = svc.run_sweep(
        node, prompt="window", step_deg=360, phi_angles=[0.0],
        dino_channel="normals", sam_channel="normals",
        debug_dir=str(tmp_path),
    )

    assert result.instances == []
    images = list((tmp_path / "prefilter").glob("*.png"))
    assert len(images) == 1


# ── parallel outlier filtering ──────────────────────────────────────────

def _make_big_node(n_points: int, seed: int = 0) -> ScanNode:
    """A ScanNode with `n_points` real, spatially-varied points (gaussian
    jitter, not all identical like _make_node's fixed-position grid) - big
    enough to cross _OUTLIER_MP_MIN_POINTS and actually exercise the
    parallel-outlier-filtering path."""
    side = int(np.ceil(np.sqrt(n_points)))
    h = w = side
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows, cols = rows.ravel()[:n_points].astype(np.int32), cols.ravel()[:n_points].astype(np.int32)
    mapper = IndexMapper({"row": rows, "col": cols}, row_max=h - 1, col_max=w - 1)
    rng = np.random.default_rng(seed)
    points = (rng.normal(loc=0.0, scale=100.0, size=(n_points, 3))).astype(np.int16)
    points[:, 0] += 2000  # comfortably away from the scanner origin
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=points, quant_scale=0.01, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.tile(np.array([0, 0, 127], dtype=np.int8), (n_points, 1)),
        colors_rgb=None, colors_intensity=None,
        point_count=n_points, original_point_count=n_points, decimation_step=1,
        available_modes=["normals"], row_max=h, col_max=w,
        hidden_mask=np.zeros(n_points, dtype=bool), index_mapper=mapper,
    )


def test_run_outlier_filtering_parallel_path_matches_serial_results():
    """Correctness of the real multiprocessing path: two instances big enough
    to trigger it must come back with EXACTLY the same filtered indices as
    calling filter_mask_outliers directly (serially) on each."""
    node = _make_big_node(6000, seed=1)
    inst_a = DetectedInstance(indices=np.arange(0, 6000), label="window", score=0.8, color=(1, 1, 1))
    inst_b = DetectedInstance(indices=np.arange(0, 6000)[::-1].copy(), label="door", score=0.7, color=(2, 2, 2))

    parallel_results = _run_outlier_filtering(node, [inst_a, inst_b])

    from pcseg.core.mask_filters import filter_mask_outliers as _fmo
    expected_a = _fmo(node, inst_a.indices)
    expected_b = _fmo(node, inst_b.indices)

    assert sorted(parallel_results[0].tolist()) == sorted(expected_a.tolist())
    assert sorted(parallel_results[1].tolist()) == sorted(expected_b.tolist())


def test_run_outlier_filtering_stays_serial_below_the_instance_count_threshold():
    """Only ONE big instance (below _OUTLIER_MP_MIN_INSTANCES) must never
    touch the pool at all - proven by making pool construction itself raise
    and confirming that doesn't propagate."""
    node = _make_big_node(6000, seed=2)
    inst = DetectedInstance(indices=np.arange(0, 6000), label="window", score=0.8, color=(1, 1, 1))

    with patch("pcseg.services.segmentation_service.ProcessPoolExecutor",
               side_effect=AssertionError("pool should never be constructed for a single big instance")):
        results = _run_outlier_filtering(node, [inst])

    assert len(results) == 1
    assert len(results[0]) > 0


def test_run_outlier_filtering_small_instances_never_touch_the_pool():
    node = _make_node(distance=100.0)
    instances = [
        DetectedInstance(indices=np.arange(0, 20), label="window", score=0.8, color=(1, 1, 1)),
        DetectedInstance(indices=np.arange(20, 40), label="door", score=0.7, color=(2, 2, 2)),
    ]

    with patch("pcseg.services.segmentation_service.ProcessPoolExecutor",
               side_effect=AssertionError("pool should never be constructed for small instances")):
        results = _run_outlier_filtering(node, instances)

    assert len(results) == 2


def test_run_outlier_filtering_falls_back_to_serial_when_the_pool_fails():
    """The pool failing (e.g. no process-spawn permission in some sandboxed
    environment) must not fail the whole sweep - every instance still gets
    filtered, just serially instead."""
    node = _make_big_node(6000, seed=3)
    inst_a = DetectedInstance(indices=np.arange(0, 6000), label="window", score=0.8, color=(1, 1, 1))
    inst_b = DetectedInstance(indices=np.arange(0, 6000)[::-1].copy(), label="door", score=0.7, color=(2, 2, 2))

    with patch("pcseg.services.segmentation_service.ProcessPoolExecutor",
               side_effect=OSError("no process-spawn permission")):
        results = _run_outlier_filtering(node, [inst_a, inst_b])

    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0
