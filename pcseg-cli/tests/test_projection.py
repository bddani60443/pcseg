import numpy as np
import pytest

from core.mapping import IndexMapper
from core.projection import (
    compute_coverage_grid,
    compute_distance_panorama,
    compute_indexed_coverage_grid,
    compute_multiscale_coverage_grid,
    get_face_projection_map,
    pad_img,
    phi_extent_of_valid_data,
    real_scan_mask,
    reconstruct_panorama_for_channel,
    render_coverage_grid_preview,
)
from core.scan_node import ScanNode


def test_get_face_projection_map_shape_and_range():
    h, w, size = 180, 360, 64
    map_x, map_y = get_face_projection_map(h=h, w=w, theta=0.0, phi=0.0, size=size, fov=90)

    assert map_x.shape == (size, size)
    assert map_y.shape == (size, size)
    # Coordinates must stay within (or very near) the source panorama bounds.
    assert map_x.min() >= -1.0
    assert map_x.max() <= w + 1.0
    assert map_y.min() >= -1.0
    assert map_y.max() <= h + 1.0


def test_get_face_projection_map_center_pixel_looks_forward():
    # At theta=0, phi=0 the center of the crop should sample the panorama's
    # horizontal midpoint (longitude 0) and vertical midpoint (latitude 0).
    h, w, size = 180, 360, 65  # odd size so there's an exact center pixel
    map_x, map_y = get_face_projection_map(h=h, w=w, theta=0.0, phi=0.0, size=size, fov=90)

    center = size // 2
    assert abs(map_x[center, center] - w / 2) < 1.0
    assert abs(map_y[center, center] - h / 2) < 1.0


def test_pad_img_grayscale_pads_to_half_width():
    img = np.zeros((10, 100), dtype=np.uint8)
    padded = pad_img(img, fill_value=255)

    assert padded.shape == (50, 100)
    assert np.all(padded[10:, :] == 255)
    assert np.all(padded[:10, :] == 0)


def test_pad_img_color_no_op_when_already_tall_enough():
    img = np.zeros((60, 100, 3), dtype=np.uint8)
    padded = pad_img(img)

    assert padded.shape == img.shape


def test_pad_img_rejects_unsupported_ndim():
    import pytest

    # Short enough (h << w/2) that padding is actually attempted, so the
    # ndim check is reached instead of returning early as a no-op.
    bad = np.zeros((2, 100, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        pad_img(bad)


def test_compute_coverage_grid_covers_every_tile_when_everything_is_valid():
    valid_mask = np.ones((180, 360), dtype=bool)

    grid = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2)

    # 90*(1-0.2) = 72 deg step -> 5 theta steps (360/72) x ceil(180/72)=3 phi steps.
    assert len(grid) == 5 * 3
    thetas = sorted({round(t, 3) for t, _ in grid})
    phis = sorted({round(p, 3) for _, p in grid})
    assert len(thetas) == 5
    assert len(phis) == 3


def test_compute_coverage_grid_skips_tiles_with_no_valid_data():
    # Only a thin band around the horizon (the middle rows) has any data -
    # candidate tiles whose whole footprint is elsewhere must be dropped. Uses
    # an explicit full phi_range so this actually exercises the per-tile probe
    # pruning (phi_range=None would auto-narrow to the thin band itself first).
    valid_mask = np.zeros((180, 360), dtype=bool)
    valid_mask[85:95, :] = True

    grid = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2, phi_range=(-90.0, 90.0))

    assert len(grid) > 0
    # No tile should be centered far enough from the horizon (phi=0) to have
    # missed the thin valid band entirely at this fov.
    assert all(abs(phi) < 90.0 for _, phi in grid)


def test_phi_extent_of_valid_data_matches_the_populated_row_range():
    valid_mask = np.zeros((180, 360), dtype=bool)
    valid_mask[85:95, :] = True  # rows 85..94 populated

    phi_lo, phi_hi = phi_extent_of_valid_data(valid_mask)

    assert phi_lo == pytest.approx(180.0 * (85 / 180 - 0.5))
    assert phi_hi == pytest.approx(180.0 * (95 / 180 - 0.5))


def test_phi_extent_of_valid_data_spans_the_full_range_when_everything_is_valid():
    valid_mask = np.ones((180, 360), dtype=bool)
    assert phi_extent_of_valid_data(valid_mask) == pytest.approx((-90.0, 90.0))


def test_compute_coverage_grid_auto_derives_phi_range_from_valid_data_by_default():
    # No phi_range given: must narrow itself to the thin valid band rather
    # than sweeping (and then pruning) the full sphere.
    valid_mask = np.zeros((180, 360), dtype=bool)
    valid_mask[85:95, :] = True

    grid = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2)

    assert len(grid) > 0
    phi_lo, phi_hi = phi_extent_of_valid_data(valid_mask)
    assert all(phi_lo <= phi <= phi_hi for _, phi in grid)


def test_compute_coverage_grid_respects_a_restricted_phi_range():
    valid_mask = np.ones((180, 360), dtype=bool)

    grid = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2, phi_range=(-90.0, 10.0))

    assert len(grid) > 0
    assert all(-90.0 <= phi <= 10.0 for _, phi in grid)


def test_compute_coverage_grid_rejects_an_invalid_phi_range():
    with pytest.raises(ValueError):
        compute_coverage_grid(np.ones((10, 10), dtype=bool), phi_range=(10.0, -10.0))


def test_compute_coverage_grid_returns_nothing_for_an_entirely_empty_scan():
    valid_mask = np.zeros((180, 360), dtype=bool)
    assert compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2) == []


def test_compute_coverage_grid_rejects_overlap_of_one_or_more():
    with pytest.raises(ValueError):
        compute_coverage_grid(np.ones((10, 10), dtype=bool), fov=90.0, overlap=1.0)


def _make_node_with_distances(h: int, w: int, distances: np.ndarray) -> ScanNode:
    """A minimal ScanNode where point i sits at (distances[i], 0, 0) from the origin."""
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows, cols = rows.ravel().astype(np.int32), cols.ravel().astype(np.int32)
    n = h * w
    mapper = IndexMapper({"row": rows, "col": cols}, row_max=h - 1, col_max=w - 1)

    points = np.zeros((n, 3), dtype=np.int16)
    points[:, 0] = distances.astype(np.int16)
    return ScanNode(
        scan_id="test", scan_name="test", file_path="test.e57",
        points_quantized=points, quant_scale=1.0, quant_center=np.zeros(3, dtype=np.float32),
        normals=np.zeros((n, 3), dtype=np.int8), colors_rgb=None, colors_intensity=None,
        point_count=n, original_point_count=n, decimation_step=1,
        available_modes=["normals"], row_max=h, col_max=w,
        hidden_mask=np.zeros(n, dtype=bool), index_mapper=mapper,
    )


def test_real_scan_mask_excludes_degenerate_origin_points():
    # A scanner routinely fills a "no return" grid cell (e.g. open sky) with a
    # placeholder point sitting exactly at its own origin, rather than leaving
    # the cell empty - real_scan_mask must treat that as no data, unlike a
    # plain pano_map >= 0 check, which would report it as valid.
    h, w = 4, 4
    distances = np.zeros(h * w, dtype=np.float32)
    distances[:8] = 0.0   # first 8 points: degenerate "no return" placeholders
    distances[8:] = 50.0  # last 8 points: real points

    node = _make_node_with_distances(h, w, distances)
    assert (node.index_mapper.pano_map >= 0).all()  # every cell IS populated

    mask = real_scan_mask(node, min_distance=0.1)

    # IndexMapper flips row order internally (see IndexMapper.__init__), so look
    # up each point's actual panorama position instead of assuming a direction.
    p2p = node.index_mapper.point_to_pano
    degenerate_ys, degenerate_xs = p2p[:8, 0], p2p[:8, 1]
    real_ys, real_xs = p2p[8:, 0], p2p[8:, 1]

    assert not mask[degenerate_ys, degenerate_xs].any()
    assert mask[real_ys, real_xs].all()


def test_real_scan_mask_matches_padded_shape():
    h, w = 4, 20  # wide enough that reconstruct_panorama_for_channel would pad
    node = _make_node_with_distances(h, w, np.full(h * w, 50.0, dtype=np.float32))

    mask = real_scan_mask(node, min_distance=0.1)

    assert mask.shape == (pad_img(np.zeros((h, w), dtype=bool)).shape)


def test_compute_indexed_coverage_grid_matches_compute_coverage_grid():
    valid_mask = np.ones((180, 360), dtype=bool)

    plain = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2)
    indexed, n_theta = compute_indexed_coverage_grid(valid_mask, fov=90.0, overlap=0.2)

    assert [(theta, phi) for _, _, theta, phi in indexed] == plain
    assert n_theta == 5  # matches the other full-coverage test: 360/72deg step


def test_compute_indexed_coverage_grid_indices_survive_gaps():
    # Only the horizon row has data - the theta ring at that phi index should
    # keep the SAME theta_idx values it would have in the full (ungapped) grid,
    # not get renumbered 0..N after dropping every other phi ring.
    valid_mask = np.zeros((180, 360), dtype=bool)
    valid_mask[85:95, :] = True

    indexed, n_theta = compute_indexed_coverage_grid(valid_mask, fov=90.0, overlap=0.2, phi_range=(-90.0, 90.0))

    phi_indices = {phi_idx for _, phi_idx, _, _ in indexed}
    assert len(phi_indices) == 1  # only one phi ring actually has any data
    theta_indices = sorted(theta_idx for theta_idx, _, _, _ in indexed)
    assert theta_indices == list(range(n_theta))  # that ring's tiles keep 0..n_theta-1, no renumbering


def test_compute_indexed_coverage_grid_empty_mask_returns_zero_n_theta():
    assert compute_indexed_coverage_grid(np.zeros((10, 10), dtype=bool)) == ([], 0)


def test_compute_indexed_coverage_grid_theta_range_restricts_the_wedge():
    valid_mask = np.ones((180, 360), dtype=bool)

    indexed, wrap_n_theta = compute_indexed_coverage_grid(
        valid_mask, fov=90.0, overlap=0.2, phi_range=(-1.0, 1.0), theta_range=(-45.0, 45.0),
    )

    thetas = sorted({theta for _, _, theta, _ in indexed})
    assert all(-45.0 <= t <= 45.0 for t in thetas)
    # 90deg span / 72deg step -> ceil(90/72) = 2 tiles, not the full circle's 5.
    assert len(thetas) == 2
    # An open wedge must never report a closed-ring wraparound size.
    assert wrap_n_theta == 0


def test_compute_indexed_coverage_grid_theta_range_none_keeps_full_circle_wraparound():
    valid_mask = np.ones((180, 360), dtype=bool)

    _entries, wrap_n_theta = compute_indexed_coverage_grid(
        valid_mask, fov=90.0, overlap=0.2, phi_range=(-1.0, 1.0), theta_range=None,
    )

    assert wrap_n_theta == 5  # unrestricted: real step count, wraparound-eligible


def test_compute_indexed_coverage_grid_rejects_an_invalid_theta_range():
    with pytest.raises(ValueError):
        compute_indexed_coverage_grid(np.ones((10, 10), dtype=bool), theta_range=(45.0, -45.0))


def test_compute_coverage_grid_accepts_theta_range_too():
    valid_mask = np.ones((180, 360), dtype=bool)
    grid = compute_coverage_grid(valid_mask, fov=90.0, overlap=0.2, phi_range=(-1.0, 1.0), theta_range=(-45.0, 45.0))
    assert all(-45.0 <= theta <= 45.0 for theta, _ in grid)


def test_compute_distance_panorama_places_each_points_own_distance():
    h, w = 4, 4
    n = h * w
    distances = np.arange(n, dtype=np.float32) * 10.0  # a distinct value per point
    node = _make_node_with_distances(h, w, distances)

    dist_panorama = compute_distance_panorama(node)

    p2p = node.index_mapper.point_to_pano
    for i in range(n):
        y, x = p2p[i]
        assert dist_panorama[y, x] == pytest.approx(distances[i])


def test_compute_distance_panorama_zeroes_cells_excluded_by_valid_mask():
    h, w = 4, 4
    node = _make_node_with_distances(h, w, np.full(h * w, 50.0, dtype=np.float32))
    valid_mask = np.zeros((h, w), dtype=bool)  # exclude everything

    dist_panorama = compute_distance_panorama(node, valid_mask)

    assert (dist_panorama == 0.0).all()


def test_compute_distance_panorama_matches_real_scan_masks_padded_shape():
    """Real crash on a real scan (h=1194, w=3048 - a wide, vertically-narrow
    capture, h well under w/2): IndexMapper.map_to_panorama returns the RAW
    (mapper.h, mapper.w) shape, but real_scan_mask (and any valid_mask a
    caller reasonably passes) is padded to (w // 2, w) via pad_img - applying
    one against the other via np.where raised a broadcast ValueError. Needs
    h < w // 2 to actually exercise pad_img's padding branch, unlike every
    other test node in this file (all built at h >= w // 2, where padding is
    a no-op and this bug stayed invisible)."""
    h, w = 4, 20  # w // 2 = 10 > h=4 -> pad_img WILL pad
    node = _make_node_with_distances(h, w, np.full(h * w, 50.0, dtype=np.float32))
    valid_mask = real_scan_mask(node)

    dist_panorama = compute_distance_panorama(node, valid_mask)

    assert dist_panorama.shape == valid_mask.shape


def _make_three_band_node(h: int = 180, w: int = 360):
    """cols[0:w/3) get distance 1.0 (near third), [w/3, 2w/3) get 50.0 (middle,
    neither pass), [2w/3, w) get 200.0 (far third) - point index k's column is
    k % w (row-major ravel of the (h, w) meshgrid _make_node_with_distances
    builds), so this needs no per-point row/col bookkeeping beyond that."""
    n = h * w
    cols = np.tile(np.arange(w), h)
    distances = np.empty(n, dtype=np.float32)
    distances[cols < w // 3] = 1.0
    distances[(cols >= w // 3) & (cols < 2 * w // 3)] = 50.0
    distances[cols >= 2 * w // 3] = 200.0
    return _make_node_with_distances(h, w, distances)


def test_compute_multiscale_coverage_grid_adds_far_and_near_passes_at_scaled_fov():
    node = _make_three_band_node()

    grid = compute_multiscale_coverage_grid(
        node, fov=90.0, overlap=0.2, far_percentile=75.0, far_fov_scale=0.5,
        near_percentile=25.0, near_fov_scale=1.3,
    )

    fovs = sorted({round(fov, 3) for _, _, fov in grid})
    assert fovs == [45.0, 90.0, 117.0]  # middle, far (90*0.5), near (90*1.3)

    middle_thetas = [theta for theta, _phi, fov in grid if fov == 90.0]
    far_thetas = [theta for theta, _phi, fov in grid if fov == 45.0]
    near_thetas = [theta for theta, _phi, fov in grid if fov == 117.0]
    assert middle_thetas and far_thetas and near_thetas  # every tier actually contributed tiles


def test_compute_multiscale_coverage_grid_middle_tier_excludes_far_and_near_regions():
    """The middle tier must be a true partition remainder, not a scan-wide
    blanket pass - it should cover strictly LESS than an unrestricted
    full-scan pass at the same fov, since the far and near thirds of
    `_make_three_band_node` are carved out of it. Regression for the earlier
    "blanket base + far/near overlays" design, which swept the far/near
    regions twice."""
    node = _make_three_band_node()
    unrestricted_full_scan = compute_coverage_grid(real_scan_mask(node), fov=90.0, overlap=0.2)

    grid = compute_multiscale_coverage_grid(
        node, fov=90.0, overlap=0.2, far_percentile=75.0, far_fov_scale=0.5,
        near_percentile=25.0, near_fov_scale=1.3,
    )
    middle_tiles = [(theta, phi) for theta, phi, fov in grid if fov == 90.0]

    assert len(middle_tiles) < len(unrestricted_full_scan)
    assert set(middle_tiles).issubset(set(unrestricted_full_scan))


def test_compute_multiscale_coverage_grid_far_pass_matches_an_independently_built_far_mask():
    """Cross-check against the SAME primitives (compute_distance_panorama,
    np.percentile, compute_coverage_grid) built fresh here, rather than
    guessing which raw (theta, phi) values the far band's columns should
    produce - get_face_projection_map's theta convention (grid-step index *
    step size, starting at 0) isn't the same as the point-centered theta_deg
    formula used elsewhere in this module, so hand-deriving an expected theta
    range independently is easy to get wrong without re-deriving that
    rotation math too."""
    node = _make_three_band_node()
    far_fov = 90.0 * 0.5

    valid_mask = real_scan_mask(node)
    real_distances_all = np.linalg.norm(
        node.points_quantized.astype(np.float32) * node.quant_scale + node.quant_center, axis=1,
    )
    far_threshold = float(np.percentile(real_distances_all, 75.0))
    dist_panorama = compute_distance_panorama(node, valid_mask)
    expected_far_mask = valid_mask & (dist_panorama >= far_threshold)
    expected_far = compute_coverage_grid(expected_far_mask, fov=far_fov, overlap=0.2)

    grid = compute_multiscale_coverage_grid(
        node, fov=90.0, far_percentile=75.0, far_fov_scale=0.5,
        near_percentile=25.0, near_fov_scale=1.3,
    )
    actual_far = [(theta, phi) for theta, phi, fov in grid if fov == far_fov]

    assert actual_far == expected_far
    assert actual_far  # and it's actually non-empty, not a vacuous match


def test_compute_multiscale_coverage_grid_falls_back_to_base_with_no_real_points():
    h, w = 8, 8
    node = _make_node_with_distances(h, w, np.zeros(h * w, dtype=np.float32))  # all degenerate

    grid = compute_multiscale_coverage_grid(node, fov=90.0)

    assert grid == []  # base pass ALSO finds nothing - real_scan_mask excludes every point


def test_compute_multiscale_coverage_grid_does_not_crash_on_a_panorama_needing_padding():
    """End-to-end regression for the real crash: a wide, vertically-narrow
    panorama (h well under w // 2) needs real_scan_mask's own pad_img
    padding, which compute_distance_panorama previously didn't apply,
    breaking the np.where inside it with a shape mismatch."""
    h, w = 4, 20  # w // 2 = 10 > h=4 -> pad_img WILL pad
    node = _make_node_with_distances(h, w, np.full(h * w, 50.0, dtype=np.float32))

    grid = compute_multiscale_coverage_grid(node, fov=90.0, overlap=0.2)

    assert isinstance(grid, list)  # didn't raise - that's the whole point


def test_compute_multiscale_coverage_grid_respects_a_restricted_theta_range():
    node = _make_three_band_node()

    grid = compute_multiscale_coverage_grid(
        node, fov=90.0, far_percentile=75.0, far_fov_scale=0.5,
        near_percentile=25.0, near_fov_scale=1.3, theta_range=(-30.0, 30.0), phi_range=(-1.0, 1.0),
    )

    assert all(-30.0 <= theta <= 30.0 for theta, _phi, _fov in grid)


# ── render_coverage_grid_preview ────────────────────────────────────────

def test_render_coverage_grid_preview_returns_a_bgr_image_matching_the_panorama_shape():
    node = _make_three_band_node()
    expected_shape = reconstruct_panorama_for_channel(node, "normals").shape
    view_grid = compute_coverage_grid(real_scan_mask(node), fov=90.0, overlap=0.2)

    image = render_coverage_grid_preview(node, view_grid, default_fov=90.0, channel="normals")

    assert image.shape == expected_shape
    assert image.dtype == np.uint8


def test_render_coverage_grid_preview_accepts_mixed_theta_phi_and_theta_phi_fov_entries():
    node = _make_three_band_node()
    view_grid = [(0.0, 0.0), (45.0, 10.0, 30.0)]  # one plain, one with its own fov

    image = render_coverage_grid_preview(node, view_grid, default_fov=90.0, channel="normals")

    assert image.ndim == 3


def test_render_coverage_grid_preview_actually_draws_something():
    """The output must differ from the plain, unannotated panorama - an
    empty view_grid is the one case that should draw nothing at all."""
    node = _make_three_band_node()
    plain = render_coverage_grid_preview(node, [], default_fov=90.0, channel="normals")
    view_grid = [(0.0, 0.0, 90.0)]
    annotated = render_coverage_grid_preview(node, view_grid, default_fov=90.0, channel="normals")

    assert not np.array_equal(plain, annotated)


def test_render_coverage_grid_preview_raises_for_an_unavailable_channel():
    node = _make_three_band_node()  # no colors_rgb/colors_intensity set

    with pytest.raises(ValueError):
        render_coverage_grid_preview(node, [(0.0, 0.0, 90.0)], channel="rgb")


def test_render_coverage_grid_preview_handles_a_tile_that_wraps_the_0_360_seam():
    node = _make_three_band_node()
    view_grid = [(179.0, 0.0, 90.0)]  # centered right at the seam - definitely wraps

    image = render_coverage_grid_preview(node, view_grid, default_fov=90.0, channel="normals")

    assert image.ndim == 3  # didn't raise, produced a real image
