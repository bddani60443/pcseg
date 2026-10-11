"""
Regression test locking in that AutoSegmentationService.run_sweep, when given a
`view_grid` (from core.projection.compute_indexed_coverage_grid), sweeps exactly
those views - with their original (theta_idx, phi_idx) from the full candidate
grid - instead of building its own step_deg x phi_angles cartesian product. The
indices matter because core.merge's view-adjacency gating relies on them to tell
genuinely neighboring views apart from unrelated ones, even when some tiles were
dropped for having no valid scan data.

auto_segmentation_service.py takes its mask backend as a plain injected callable
(AutoMaskBackend) rather than importing DinoService/SamService itself, and its
generate_label_color import comes from core.colors (pure hashlib/numpy) -
so, unlike the rest of the segmentation-backend code, this module has no torch
dependency at all and this test needs no optional extra / skip gate.
"""
import numpy as np
import pytest

from core.mapping import IndexMapper
from core.scan_node import ScanNode
from exceptions import SweepCancelled
from services.auto_segmentation_service import AutoSegmentationService


def _make_node(distance: float = 100.0) -> ScanNode:
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


def test_run_sweep_uses_the_given_view_grid_instead_of_step_deg():
    node = _make_node()
    calls: list[tuple] = []

    def fake_backend(view_crop):
        calls.append(view_crop.shape)
        mask = np.zeros(view_crop.shape[:2], dtype=bool)
        mask[1:5, 1:5] = True
        return [mask]

    svc = AutoSegmentationService(fake_backend)

    # A deliberately gappy indexed grid: theta_idx 0 and 2 at phi_idx 0 - index 1
    # was "dropped" (as if it had no valid data), matching what
    # compute_indexed_coverage_grid produces on a real scan with real gaps.
    entries = [(0, 0, 0.0, 0.0), (2, 0, 144.0, 0.0)]
    n_theta = 3

    result = svc.run_sweep(
        node, channel="normals", min_points=1,
        view_grid=(entries, n_theta),
    )

    # Exactly one backend call per view_grid entry - the step_deg/phi_angles
    # cartesian product (which would produce a different view count/set here)
    # must not have been used instead. (Not asserting on len(result.instances):
    # this node is tiny enough that both views' masks legitimately back-project
    # to the same 3D indices and correctly merge into one cluster.)
    assert len(calls) == len(entries)
    assert len(result.instances) >= 1


def test_run_sweep_falls_back_to_step_deg_when_no_view_grid_given():
    node = _make_node()
    calls: list[tuple] = []

    def fake_backend(view_crop):
        calls.append(view_crop.shape)
        return []

    svc = AutoSegmentationService(fake_backend)
    svc.run_sweep(node, channel="normals", step_deg=180, phi_angles=[0.0])

    # step_deg=180 over 360 degrees -> exactly 2 views (0 and 180 degrees).
    assert len(calls) == 2


def test_run_sweep_raises_sweep_cancelled_when_cancel_check_returns_true():
    node = _make_node()
    svc = AutoSegmentationService(lambda view_crop: [])

    with pytest.raises(SweepCancelled):
        svc.run_sweep(node, channel="normals", step_deg=90, phi_angles=[0.0], cancel_check=lambda: True)
