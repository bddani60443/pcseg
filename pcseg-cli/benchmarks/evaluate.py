# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
Point-level accuracy of a pcseg result against a manually drawn ground truth.

Ground truth is drawn on the scan's own panorama instead of in 3D: a
structured scan has exactly one measured point per panorama pixel, so a label
image painted over the panorama IS a per-point 3D labelling - no rendering,
occlusion or nearest-neighbour matching involved.

1. Export the panorama to annotate (unpadded, one pixel per grid cell):

       python benchmarks/evaluate.py panorama --scan outdoor.e57 --decimation 4 --out pano.png

2. Paint a single-channel label image of the SAME size (e.g. in GIMP, CVAT or
   LabelMe): pixel value = class id, 0 = none of the evaluated classes,
   255 = ignore (ambiguous / not annotated). Where objects overlap, paint the
   smaller one (a window inside a building is "window", not "building") - the
   same rule pcseg uses when it classifies overlapping detections. Save the
   class names next to it as JSON, e.g. {"1": "window", "2": "building"}.

3. Run pcseg with an .e57 output at the SAME --decimation, then score it:

       pcseg run --input outdoor.e57 --decimation 4 ... --out result.e57
       python benchmarks/evaluate.py score --scan outdoor.e57 --decimation 4 \\
           --ground-truth gt.png --gt-classes gt_classes.json --prediction result.e57

   The prediction is read from pcseg's structured E57 export: its grid indices
   locate each point exactly, and its class id is stored in `intensity`, with
   the names in `result_classes.txt`. Classes are matched by name.

Reports per-class point IoU, precision and recall, and their mean over the
ground-truth classes. Only grid cells holding a real measurement are counted.

Tip: don't draw the ground truth with a SAM-assisted annotation tool - the
reference would then share the very model under evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pye57

from core.projection import real_point_mask, reconstruct_panorama_for_channel
from fileio.e57_loader import ScanData

IGNORE = 255


def export_panorama(scan: str, decimation: int, out: str, channel: str = "rgb") -> tuple[int, int]:
    node = ScanData(scan, decimation_step=decimation).load()
    pano = reconstruct_panorama_for_channel(node, channel, pad=False)
    if pano is None:
        raise ValueError(f"Channel {channel!r} is not available for this scan.")
    if channel == "rgb":
        pano = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out, pano)
    return pano.shape[:2]


def predicted_label_grid(prediction_e57: str, decimation: int, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """(h, w) grid of predicted class NAMES' ids (0 = unclassified), in the same pixel frame as the panorama."""
    e57 = pye57.E57(prediction_e57)
    try:
        raw = e57.read_scan_raw(0)
    finally:
        e57.close()

    h, w = shape
    grid = np.zeros(shape, dtype=np.int32)
    rows = raw["rowIndex"].astype(np.int64) // decimation
    cols = raw["columnIndex"].astype(np.int64) // decimation
    # IndexMapper flips rows: raw grid row 0 is the panorama's bottom row.
    grid[(h - 1) - rows, cols] = raw["intensity"].astype(np.int32)

    names = {}
    classes_txt = prediction_e57.rsplit(".", 1)[0] + "_classes.txt"
    for line in Path(classes_txt).read_text(encoding="utf-8").splitlines():
        class_id, name = line.split(" ", 1)
        names[int(class_id)] = name
    return grid, names


def score_labels(gt: np.ndarray, pred: np.ndarray, gt_names: dict[int, str], pred_names: dict[int, str],
                 valid: np.ndarray) -> dict[str, dict]:
    """
    Per-class point IoU / precision / recall. `gt`/`pred` are integer label
    arrays (any shape, same shape), `valid` marks which elements count as
    points to evaluate; ground-truth IGNORE elements are skipped as well.
    """
    evaluated = valid & (gt != IGNORE)
    scores = {}
    for gt_id, name in sorted(gt_names.items()):
        pred_ids = [pid for pid, pname in pred_names.items() if pname == name]
        is_gt = evaluated & (gt == gt_id)
        is_pred = evaluated & np.isin(pred, pred_ids)
        tp = int((is_gt & is_pred).sum())
        fp = int((~is_gt & is_pred).sum())
        fn = int((is_gt & ~is_pred).sum())
        scores[name] = {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "gt_points": int(is_gt.sum()),
            "predicted_points": int(is_pred.sum()),
        }
    return scores


def score(scan: str, decimation: int, ground_truth: str, gt_classes: str, prediction: str) -> dict[str, dict]:
    node = ScanData(scan, decimation_step=decimation).load()
    shape = (node.index_mapper.h, node.index_mapper.w)

    gt = cv2.imread(ground_truth, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(ground_truth)
    if gt.ndim == 3:
        raise ValueError("Ground truth must be a single-channel label image (pixel value = class id).")
    if gt.shape != shape:
        raise ValueError(f"Ground truth is {gt.shape}, but the panorama at decimation {decimation} is {shape}.")
    gt_names = {int(k): v for k, v in json.loads(Path(gt_classes).read_text(encoding="utf-8")).items()}

    pred, pred_names = predicted_label_grid(prediction, decimation, shape)
    # Only cells holding a real measurement count - "no return" placeholder points at
    # the scanner origin are never part of a pcseg result either (see real_point_mask).
    pano_map = node.index_mapper.pano_map
    valid = pano_map >= 0
    valid[valid] = real_point_mask(node)[pano_map[valid]]
    return score_labels(gt.astype(np.int32), pred, gt_names, pred_names, valid=valid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("panorama", help="Export the unpadded panorama to annotate.")
    p.add_argument("--scan", required=True)
    p.add_argument("--decimation", type=int, required=True)
    p.add_argument("--channel", default="rgb", choices=["rgb", "intensity", "normals"])
    p.add_argument("--out", required=True)

    s = sub.add_parser("score", help="Score a pcseg .e57 result against a ground-truth label image.")
    s.add_argument("--scan", required=True)
    s.add_argument("--decimation", type=int, required=True)
    s.add_argument("--ground-truth", required=True)
    s.add_argument("--gt-classes", required=True)
    s.add_argument("--prediction", required=True, help="pcseg output .e57 (with its _classes.txt next to it)")
    s.add_argument("--json", help="Also write the scores to this JSON file.")

    args = parser.parse_args()
    if args.command == "panorama":
        h, w = export_panorama(args.scan, args.decimation, args.out, args.channel)
        print(f"Wrote {args.out} ({w}x{h} px) - paint the label image at exactly this size.")
        return

    scores = score(args.scan, args.decimation, args.ground_truth, args.gt_classes, args.prediction)
    print(f"{'class':<16}{'IoU':>8}{'prec.':>8}{'recall':>8}{'GT pts':>12}{'pred pts':>12}")
    for name, s_ in scores.items():
        print(f"{name:<16}{s_['iou']:>8.3f}{s_['precision']:>8.3f}{s_['recall']:>8.3f}"
              f"{s_['gt_points']:>12,}{s_['predicted_points']:>12,}")
    print(f"{'mean':<16}{np.nanmean([s_['iou'] for s_ in scores.values()]):>8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
