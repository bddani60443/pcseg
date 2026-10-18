# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
benchmarks/evaluate.py (the accuracy evaluation behind the paper's numbers),
checked on the synthetic scan where the right answer is known: a ground truth
copied from the prediction must score perfectly, and removing part of one
class from the ground truth must turn exactly that part into false positives.
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cli import main

_spec = importlib.util.spec_from_file_location(
    "pcseg_benchmark_evaluate", Path(__file__).parents[1] / "benchmarks" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def test_score_labels_counts_tp_fp_fn_per_class_and_skips_ignored():
    gt = np.array([1, 1, 1, 2, 0, 255])
    pred = np.array([7, 7, 0, 7, 8, 7])  # pred ids 7 = "window", 8 = "building"
    valid = np.ones(6, dtype=bool)

    scores = evaluate.score_labels(gt, pred, {1: "window", 2: "building"}, {7: "window", 8: "building"}, valid)

    # window: TP=2 (idx 0,1), FN=1 (idx 2), FP=1 (idx 3); idx 5 is ignored.
    assert scores["window"]["iou"] == pytest.approx(2 / 4)
    assert scores["window"]["precision"] == pytest.approx(2 / 3)
    assert scores["window"]["recall"] == pytest.approx(2 / 3)
    # building: GT at idx 3 predicted as window (FN), pred at idx 4 is GT background (FP).
    assert scores["building"]["iou"] == 0.0


@pytest.fixture
def prediction(synthetic_scan_path, tmp_path, fake_model_backends):
    out = tmp_path / "result.e57"
    assert main(["run", "-i", str(synthetic_scan_path), "-p", "wall, door", "--decimation", "1",
                 "--view-size", "64", "--step-deg", "90", "-o", str(out)]) == 0
    return out


def _ground_truth_from_prediction(prediction_path, tmp_path, drop_every_other_door_row=False):
    grid, names = evaluate.predicted_label_grid(str(prediction_path), 1, (80, 160))
    gt = np.zeros_like(grid, dtype=np.uint8)
    gt_names = {}
    for new_id, (pred_id, name) in enumerate(sorted((k, v) for k, v in names.items() if k != 0), start=1):
        gt[grid == pred_id] = new_id
        gt_names[new_id] = name
    if drop_every_other_door_row:
        door_id = next(k for k, v in gt_names.items() if v == "door")
        rows = np.where((gt == door_id).any(axis=1))[0]
        for r in rows[::2]:
            gt[r][gt[r] == door_id] = 0
    gt_path, classes_path = tmp_path / "gt.png", tmp_path / "gt_classes.json"
    cv2.imwrite(str(gt_path), gt)
    classes_path.write_text(json.dumps({str(k): v for k, v in gt_names.items()}))
    return gt_path, classes_path, gt, gt_names


def test_ground_truth_equal_to_prediction_scores_perfectly(synthetic_scan_path, prediction, tmp_path):
    gt_path, classes_path, _, _ = _ground_truth_from_prediction(prediction, tmp_path)

    scores = evaluate.score(str(synthetic_scan_path), 1, str(gt_path), str(classes_path), str(prediction))

    assert set(scores) == {"wall", "door"}
    for s in scores.values():
        assert s["gt_points"] > 0
        assert s["iou"] == s["precision"] == s["recall"] == 1.0


def test_door_missing_from_ground_truth_rows_lowers_only_door_precision(synthetic_scan_path, prediction, tmp_path):
    gt_path, classes_path, gt, gt_names = _ground_truth_from_prediction(
        prediction, tmp_path, drop_every_other_door_row=True)

    scores = evaluate.score(str(synthetic_scan_path), 1, str(gt_path), str(classes_path), str(prediction))

    door = scores["door"]
    assert door["recall"] == 1.0  # every GT door point is still predicted as door
    assert door["precision"] == pytest.approx(door["gt_points"] / door["predicted_points"])
    assert door["precision"] < 1.0
    assert scores["wall"]["iou"] == 1.0


def test_panorama_export_matches_the_ground_truth_frame(synthetic_scan_path, tmp_path):
    out = tmp_path / "pano.png"

    assert evaluate.export_panorama(str(synthetic_scan_path), 1, str(out)) == (80, 160)
    assert cv2.imread(str(out)).shape == (80, 160, 3)
