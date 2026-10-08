import numpy as np

from pcseg.core.merge import mask_touches_border, merge_by_3d_iou, merge_by_3d_iou_iterative


def test_empty_input():
    assert merge_by_3d_iou([]) == []


def test_empty_input_with_return_view_keys_still_unpacks_to_three():
    """
    Regression test: the empty-input early return used to give back a 2-tuple
    ([], []) even when return_view_keys=True, while the normal path always
    returns 3 - merge_by_3d_iou_iterative unconditionally does
    `merged, keys, is_border = merge_by_3d_iou(..., return_view_keys=True)`,
    so a sweep with zero candidates (a real, reachable case, not just this
    test) raised ValueError: not enough values to unpack.
    """
    merged, keys, is_border = merge_by_3d_iou([], return_view_keys=True)
    assert merged == []
    assert keys == []
    assert is_border == []
    assert merge_by_3d_iou_iterative([]) == []


def test_single_mask_passes_through():
    result = merge_by_3d_iou([np.array([1, 2, 3])])
    assert len(result) == 1
    assert set(result[0].tolist()) == {1, 2, 3}


def test_disjoint_masks_stay_separate():
    a = np.arange(0, 100)
    b = np.arange(1000, 1100)
    result = merge_by_3d_iou([a, b], iou_threshold=0.5)
    assert len(result) == 2


def test_heavily_overlapping_masks_merge_into_one():
    a = np.arange(0, 100)
    b = np.arange(10, 110)  # 90/110 = ~0.82 IoU with a
    result = merge_by_3d_iou([a, b], iou_threshold=0.5)
    assert len(result) == 1
    assert set(result[0].tolist()) == set(a.tolist()) | set(b.tolist())


def test_merge_result_is_union_not_intersection():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([3, 4, 5, 6, 7])  # IoU = 3/7 ~= 0.43
    result = merge_by_3d_iou([a, b], iou_threshold=0.4)
    assert len(result) == 1
    assert set(result[0].tolist()) == {1, 2, 3, 4, 5, 6, 7}


def test_borderline_iou_stays_separate_below_threshold():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([3, 4, 5, 6, 7])  # IoU ~= 0.43
    result = merge_by_3d_iou([a, b], iou_threshold=0.9)
    assert len(result) == 2


def test_overlap_coeff_threshold_is_off_by_default():
    """Same size-mismatch pair as test_merge_duplicate_instances_overlap_rule_merges_...
    in test_segmentation_service.py, but at this lower level: without passing
    overlap_coeff_threshold, behavior must be identical to before this rule
    existed - Jaccard-only, so a heavily-size-mismatched but fully-contained
    pair stays separate."""
    big = np.arange(0, 100)
    small = np.arange(0, 10)  # fully contained in `big`, but IoU = 10/100 = 0.1
    result = merge_by_3d_iou([big, small], iou_threshold=0.5)
    assert len(result) == 2


def test_overlap_coeff_threshold_merges_a_fully_contained_small_fragment():
    big = np.arange(0, 100)
    small = np.arange(0, 10)  # intersection/min(size) = 10/10 = 1.0
    result = merge_by_3d_iou([big, small], iou_threshold=0.5, overlap_coeff_threshold=0.9)
    assert len(result) == 1
    assert set(result[0].tolist()) == set(big.tolist())


def test_overlap_coeff_threshold_still_requires_some_overlap():
    """overlap_coeff_threshold must not turn into a spatial/proximity rule - two
    genuinely disjoint masks (zero shared indices) must stay separate no matter
    how low the threshold is asked to go."""
    a = np.arange(0, 100)
    b = np.arange(1000, 1010)
    result = merge_by_3d_iou([a, b], iou_threshold=0.5, overlap_coeff_threshold=0.01)
    assert len(result) == 2


def test_overlap_coeff_threshold_is_additive_not_a_replacement_for_iou():
    """A pair that already clears iou_threshold must still merge even when
    overlap_coeff_threshold is set much stricter than what this pair achieves -
    the two rules are OR'd together, neither one can veto the other."""
    a = np.arange(0, 100)
    b = np.arange(10, 110)  # ~0.82 IoU, clears iou_threshold=0.5 on its own
    result = merge_by_3d_iou([a, b], iou_threshold=0.5, overlap_coeff_threshold=0.999)
    assert len(result) == 1


def test_three_way_chain_all_merge_via_the_largest_seed():
    # a and c don't directly overlap enough, but both overlap b heavily;
    # largest-first clustering should still fold everything into one cluster
    # since merging is against the growing cluster union, not the original seed.
    a = np.arange(0, 60)
    b = np.arange(0, 100)  # large, overlaps both a and c
    c = np.arange(40, 100)
    result = merge_by_3d_iou([a, b, c], iou_threshold=0.5)
    assert len(result) == 1
    assert set(result[0].tolist()) == set(range(0, 100))


def test_three_distinct_objects_stay_separate():
    a = np.arange(0, 50)
    b = np.arange(1000, 1050)
    c = np.arange(2000, 2050)
    result = merge_by_3d_iou([a, b, c], iou_threshold=0.5)
    assert len(result) == 3


# --- border-aware merging: fixes the real "pavement cut into 6 pieces by the
# sweep's view boundaries" problem found while smoke-testing auto-segment. ---

def test_border_touching_fragments_of_one_surface_merge_despite_low_iou():
    # Two ground-plane fragments: only a modest sliver in common relative to their
    # full extent (IoU well below 0.5), but each almost entirely explained by the
    # other once you account for the size difference (overlap coefficient ~0.9+).
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)  # overlap=100, iou=100/1050~=0.095, overlap_coeff=100/150~=0.67
    result = merge_by_3d_iou([a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6)
    assert len(result) == 1


def test_border_rule_does_not_apply_unless_both_sides_touch_border():
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)
    # Same overlap as above, but b is a fully-contained compact object (e.g. a
    # window), not a cut-off fragment - must NOT get swallowed into a.
    result = merge_by_3d_iou([a, b], iou_threshold=0.5, touches_border=[True, False], border_overlap_threshold=0.6)
    assert len(result) == 2


def test_border_rule_defaults_to_off_when_touches_border_not_provided():
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)
    result = merge_by_3d_iou([a, b], iou_threshold=0.5)
    assert len(result) == 2


def test_small_object_inside_a_large_border_touching_mask_is_not_absorbed():
    # A big wall mask (touches border) and a small window mask fully contained
    # in it (does NOT touch border, since it sits well inside the crop) - the
    # window must stay separate: this is the failure mode border-awareness must
    # NOT introduce.
    wall = np.arange(0, 10000)
    window = np.arange(500, 600)  # fully inside `wall`
    result = merge_by_3d_iou([wall, window], iou_threshold=0.5, touches_border=[True, False], border_overlap_threshold=0.6)
    assert len(result) == 2


def test_mask_touches_border_true_for_edge_pixel():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 5] = True
    assert mask_touches_border(mask) is True


def test_mask_touches_border_false_for_interior_only_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert mask_touches_border(mask) is False


def test_mask_touches_border_respects_margin():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 5] = True  # 1px from the top edge
    assert mask_touches_border(mask, margin=1) is False
    assert mask_touches_border(mask, margin=2) is True


# --- spatial (3D-distance) fallback: fixes the real "two views resample the same
# ground plane into near-disjoint point-index sets, so even overlap-coefficient
# never fires" problem found while smoke-testing a denser auto-segment sweep. ---

def _grid_points(x_range, y_range) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(*x_range), np.arange(*y_range))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1).astype(np.float32)
    return pts


def test_spatially_adjacent_but_index_disjoint_fragments_merge():
    # Two point sets with ZERO index overlap, but physically touching in 3D:
    # fragment a spans x in [0, 50), fragment b spans x in [50, 100) - a's
    # nearest point (x=49) is exactly 1.0 unit from b's nearest point (x=50).
    points_3d = np.vstack([_grid_points((0, 50), (0, 1)), _grid_points((50, 100), (0, 1))])
    a = np.arange(0, 50)
    b = np.arange(50, 100)

    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6,
        points_3d=points_3d, surface_merge_distance=1.5,
    )
    assert len(result) == 1


def test_spatially_distant_index_disjoint_fragments_stay_separate():
    points_3d = np.vstack([_grid_points((0, 50), (0, 1)), _grid_points((0, 50), (0, 1)) + [1000.0, 0, 0]])
    a = np.arange(0, 50)
    b = np.arange(50, 100)

    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6,
        points_3d=points_3d, surface_merge_distance=0.3,
    )
    assert len(result) == 2


def test_spatial_fallback_does_not_apply_unless_both_sides_touch_border():
    points_3d = np.vstack([_grid_points((0, 50), (0, 1)), _grid_points((50, 100), (0, 1)) + [0.1, 0, 0]])
    a = np.arange(0, 50)
    b = np.arange(50, 100)

    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, False], border_overlap_threshold=0.6,
        points_3d=points_3d, surface_merge_distance=0.3,
    )
    assert len(result) == 2


def test_spatial_fallback_disabled_when_no_distance_given():
    points_3d = np.vstack([_grid_points((0, 50), (0, 1)), _grid_points((50, 100), (0, 1)) + [0.1, 0, 0]])
    a = np.arange(0, 50)
    b = np.arange(50, 100)

    result = merge_by_3d_iou([a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6)
    assert len(result) == 2


# --- view-grid adjacency gating: fixes the real "ground near the tripod is
# close to nearly every view, so far-apart/unrelated sweep views' ground
# fragments transitively merge via the spatial fallback" problem found while
# smoke-testing a real scan. ---

def test_border_rules_apply_between_adjacent_views():
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)  # iou~=0.095, overlap_coeff~=0.67 -> would need the border rule
    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6,
        view_keys=[(0, 0), (1, 0)], n_theta=8,
    )
    assert len(result) == 1


def test_border_rules_skipped_between_non_adjacent_views():
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)
    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6,
        view_keys=[(0, 0), (4, 0)], n_theta=8,  # opposite side of an 8-view sweep
    )
    assert len(result) == 2


def test_border_rules_apply_across_the_theta_seam_wraparound():
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)
    # theta_idx 0 and 7 are adjacent in an 8-view sweep because of the 360/0 wrap.
    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6,
        view_keys=[(0, 0), (7, 0)], n_theta=8,
    )
    assert len(result) == 1


def test_view_gating_does_not_affect_plain_index_overlap_rule():
    # High IoU (real duplicate) must still merge even from "non-adjacent" view
    # keys - the base rule was never the problem and stays ungated.
    a = np.arange(0, 100)
    b = np.arange(0, 90)  # iou = 90/100 = 0.9
    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[False, False],
        view_keys=[(0, 0), (4, 0)], n_theta=8,
    )
    assert len(result) == 1


def test_spatial_fallback_gated_by_view_adjacency_too():
    points_3d = np.vstack([_grid_points((0, 50), (0, 1)), _grid_points((50, 100), (0, 1))])
    a = np.arange(0, 50)
    b = np.arange(50, 100)
    result = merge_by_3d_iou(
        [a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6,
        points_3d=points_3d, surface_merge_distance=1.5,
        view_keys=[(0, 0), (4, 0)], n_theta=8,  # not adjacent -> spatial rule must not apply either
    )
    assert len(result) == 2


def test_view_gating_off_by_default_matches_previous_behavior():
    a = np.arange(0, 1000)
    b = np.arange(900, 1050)
    result = merge_by_3d_iou([a, b], iou_threshold=0.5, touches_border=[True, True], border_overlap_threshold=0.6)
    assert len(result) == 1


# --- iterative "merge the merges" second phase: lets a bridging patch's growth
# in round 1 unlock a merge with a third patch that only became view-adjacent to
# the *grown* cluster, not to any single original view - requested after
# real-scan smoke-testing showed a handful of large fragments left over that a
# single greedy pass structurally cannot join (two already-formed clusters are
# never compared directly against each other within one pass). ---

def _line_points(x_start: int, count: int) -> np.ndarray:
    xs = np.arange(x_start, x_start + count, dtype=np.float32)
    return np.stack([xs, np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32)], axis=1)


def test_merge_by_3d_iou_iterative_stitches_a_three_patch_chain_across_rounds():
    # patch0 (theta0) touches patch1 (theta1) which touches patch2 (theta2); patch0
    # and patch2 are neither index-overlapping, spatially close, nor view-adjacent
    # to *each other* directly - only a second round (after patch0+patch1 have
    # already merged, growing that cluster's view footprint) can bridge them.
    points_3d = np.vstack([_line_points(0, 100), _line_points(100, 10), _line_points(111, 50)])
    patch0 = np.arange(0, 100)
    patch1 = np.arange(100, 110)
    patch2 = np.arange(110, 160)

    common_kwargs = dict(
        touches_border=[True, True, True],
        view_keys=[(0, 0), (1, 0), (2, 0)],
        n_theta=8,
        points_3d=points_3d,
        surface_merge_distance=2.5,
        iou_threshold=0.99,             # never let plain Jaccard fire in this test
        border_overlap_threshold=0.99,  # never let containment-overlap fire either
    )

    one_round = merge_by_3d_iou_iterative([patch0, patch1, patch2], max_rounds=1, **common_kwargs)
    assert len(one_round) == 2

    two_rounds = merge_by_3d_iou_iterative([patch0, patch1, patch2], max_rounds=2, **common_kwargs)
    assert len(two_rounds) == 1


def test_merge_by_3d_iou_iterative_leaves_unrelated_clusters_alone():
    points_3d = np.vstack([_line_points(0, 100), _line_points(10_000, 50)])
    a = np.arange(0, 100)
    b = np.arange(100, 150)
    result = merge_by_3d_iou_iterative(
        [a, b], max_rounds=5, touches_border=[True, True], view_keys=[(0, 0), (1, 0)], n_theta=8,
        points_3d=points_3d, surface_merge_distance=2.5,
    )
    assert len(result) == 2


def test_merge_by_3d_iou_iterative_stops_early_when_converged():
    a = np.arange(0, 100)
    b = np.arange(1000, 1050)
    result = merge_by_3d_iou_iterative([a, b], max_rounds=10, iou_threshold=0.5)
    assert len(result) == 2


def test_iterative_merge_never_absorbs_a_compact_non_border_object():
    """
    Regression test for a real bug found while smoke-testing on an actual scan: an
    earlier implementation forced every cluster to touches_border=True going into
    round 2+, so an already-closed object (a whole window-studded wall) became
    eligible for the loose merge rules against anything nearby, collapsing 244
    distinct objects into 40. Border-eligibility must instead propagate from what
    a cluster's original fragments actually were.
    """
    ground_a = np.arange(0, 100)
    ground_b = np.arange(100, 110)
    window = np.arange(200, 210)  # compact, never touched a crop edge

    points_3d = np.vstack([_line_points(0, 100), _line_points(100, 10), _line_points(111, 10)])

    result = merge_by_3d_iou_iterative(
        [ground_a, ground_b, window], max_rounds=3,
        touches_border=[True, True, False],
        view_keys=[(0, 0), (1, 0), (1, 0)],
        n_theta=8, points_3d=points_3d, surface_merge_distance=2.5,
        iou_threshold=0.99, border_overlap_threshold=0.99,
    )

    assert len(result) == 2
    assert sorted(len(r) for r in result) == [10, 110]


def test_merge_by_3d_iou_iterative_single_round_matches_plain_call():
    a = np.arange(0, 100)
    b = np.arange(50, 150)
    plain = merge_by_3d_iou([a, b], iou_threshold=0.3)
    iterative = merge_by_3d_iou_iterative([a, b], max_rounds=1, iou_threshold=0.3)
    assert len(plain) == len(iterative) == 1
