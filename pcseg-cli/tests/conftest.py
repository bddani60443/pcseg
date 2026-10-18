# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
Shared fixtures for the end-to-end tests: a small synthetic structured E57
scan, and stand-ins for the GroundingDINO/SAM2 wrappers.

The fake model services replace `services.dino_service` and
`services.sam_service` in `sys.modules`, so the full CLI pipeline
(load -> sweep -> back-projection -> outlier filter -> merge -> export) runs
without torch, model weights or a GPU. They return simple, deterministic
rectangular masks, so the tests check the pipeline's plumbing, not model
quality.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from synthetic_scan import write_synthetic_e57


@pytest.fixture(scope="session")
def synthetic_scan_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("scan") / "synthetic.e57"
    write_synthetic_e57(path)
    return path


class FakeDinoService:
    """Returns one centered box per phrase; "door" gets a smaller box inside "wall"'s."""

    def __init__(self, config_path=None, checkpoint_path=None, device=None):
        self.device = device or "cpu"

    def _ensure_loaded(self):
        pass

    def run_inference(self, img, prompt, box_thresh=0.3, text_thresh=0.25):
        if prompt == "nothing":
            return np.array([]), np.array([]), []
        h, w = img.shape[:2]
        half = 0.125 if prompt == "door" else 0.25
        box = [int(w * (0.5 - half)), int(h * (0.5 - half)), int(w * (0.5 + half)), int(h * (0.5 + half))]
        return np.array([box]), np.array([0.8]), [prompt]


class FakeSamService:
    """Box prompt -> the box itself as the mask. Point prompt -> 3 nested squares, deliberately NOT in score order."""

    POINT_SCORES = np.array([0.5, 0.9, 0.7])
    calls: list[dict] = []

    def __init__(self, config_path=None, checkpoint_path=None, device=None):
        self.device = device or "cpu"

    def _ensure_loaded(self):
        pass

    def predict(self, img, point_coords=None, point_labels=None, box=None, multimask_output=True):
        FakeSamService.calls.append({
            "n_points": 0 if point_coords is None else len(point_coords),
            "multimask_output": multimask_output,
        })
        h, w = img.shape[:2]
        if box is not None:
            boxes = np.atleast_2d(box)
            masks = np.zeros((len(boxes), 1, h, w), dtype=bool)
            for i, (x0, y0, x1, y1) in enumerate(boxes):
                masks[i, 0, y0:y1, x0:x1] = True
            return masks, np.full(len(boxes), 0.95), None

        px, py = (int(v) for v in np.asarray(point_coords)[0])
        masks = np.zeros((3, h, w), dtype=bool)
        for i, half in enumerate((4, 8, 16)):
            masks[i, max(0, py - half):py + half, max(0, px - half):px + half] = True
        scores = self.POINT_SCORES.copy()
        if not multimask_output:
            return masks[1:2], scores[1:2], None
        return masks, scores, None

    def automatic_generate(self, image_rgb, points_per_side=32, pred_iou_thresh=0.86,
                           stability_score_thresh=0.92, min_mask_region_area=150):
        h, w = image_rgb.shape[:2]
        center = np.zeros((h, w), dtype=bool)
        center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        corner = np.zeros((h, w), dtype=bool)
        corner[:h // 3, :w // 3] = True
        return [center, corner]


_STUBBED = ("services.dino_service", "services.sam_service")
# Modules that bind DinoService/SamService at import time - re-imported
# against the fakes, then dropped again so later tests get the real ones.
_DEPENDENTS = (
    "services.segmentation_service",
    "services.auto_segmentation_service",
    "services.point_segmentation_service",
)


@pytest.fixture
def fake_model_backends(monkeypatch):
    FakeSamService.calls = []

    dino_module = types.ModuleType("services.dino_service")
    dino_module.DEFAULT_CONFIG_PATH = "fake-dino-config.py"
    dino_module.DEFAULT_CHECKPOINT_PATH = "fake-dino.pth"
    dino_module.DinoService = FakeDinoService

    sam_module = types.ModuleType("services.sam_service")
    sam_module.DEFAULT_CONFIG_PATH = "fake-sam-config.yaml"
    sam_module.DEFAULT_CHECKPOINT_PATH = "fake-sam.pt"
    sam_module.SamService = FakeSamService

    monkeypatch.setitem(sys.modules, "services.dino_service", dino_module)
    monkeypatch.setitem(sys.modules, "services.sam_service", sam_module)
    for name in _DEPENDENTS:
        monkeypatch.delitem(sys.modules, name, raising=False)

    yield FakeSamService

    for name in _DEPENDENTS:
        sys.modules.pop(name, None)
