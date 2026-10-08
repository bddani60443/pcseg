from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from pcseg.paths import find_project_root

# Hydra resolves this one relative to the installed sam2 package's own config store
# (see build_sam2's compose() call), not the filesystem - CWD-independent already.
DEFAULT_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"
# ...but the checkpoint is loaded via plain torch.load(), which *is* CWD-relative -
# anchor it to this repo's own root so it still resolves when pcseg is invoked from
# somewhere else (e.g. the pcseg-gui viewer, or `pcseg` on PATH run from any directory).
# See find_project_root for why this isn't just Path(__file__).resolve().parents[3].
_SAM_CKPT_REL = "external/sam2-external/checkpoints/sam2.1_hiera_tiny.pt"
_PROJECT_ROOT = find_project_root(_SAM_CKPT_REL, __file__)
DEFAULT_CHECKPOINT_PATH = str(_PROJECT_ROOT / _SAM_CKPT_REL)


class SamService:
    """Manages the SAM2 model state, lazy loading, and inference dispatching."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                 device: str | None = None) -> None:
        self.predictor = None
        self.auto_generator = None
        self._auto_generator_params: tuple | None = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def _ensure_loaded(self) -> None:
        if self.predictor is not None:
            return

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint_path!r}. Run "
                f"scripts/install_segmentation_backends.{{sh,ps1}} from the pcseg-cli "
                f"directory, or pass --sam-checkpoint explicitly if you downloaded the "
                f"weights somewhere else."
            )

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logging.info(f"Loading SAM2 model on {self.device}...")

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        sam2_model = build_sam2(self.config_path, self.checkpoint_path, device=self.device, apply_postprocessing=False)
        if self.device == "cuda":
            sam2_model.to(dtype=torch.bfloat16)

        self.predictor = SAM2ImagePredictor(sam2_model)
        logging.info("SAM2 loaded.")

    def predict(self, img: np.ndarray, point_coords: np.ndarray | None = None, point_labels: np.ndarray | None = None,
                box: np.ndarray | None = None, multimask_output: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Universal SAM2 inference - accepts points, boxes, or both."""
        if point_coords is None and box is None:
            raise ValueError("SAM2 needs either point_coords or box.")

        self._ensure_loaded()

        if point_coords is not None:
            point_coords = np.atleast_2d(np.asarray(point_coords))
        if point_labels is not None:
            point_labels = np.atleast_1d(np.asarray(point_labels))
        if box is not None:
            box = np.atleast_2d(np.asarray(box))

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.predictor.set_image(img)
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=multimask_output,
            )

        return masks, scores, logits

    def automatic_generate(self, image_rgb: np.ndarray, points_per_side: int = 32, pred_iou_thresh: float = 0.86,
                            stability_score_thresh: float = 0.92, min_mask_region_area: int = 150) -> list[np.ndarray]:
        """
        Class-agnostic 'segment everything' mode: proposes object masks on one image
        without any prompt (box/point/text). Used by the automatic-segmentation
        pipeline, as opposed to `predict()`'s prompted (box/point) mode.
        """
        self._ensure_loaded()

        params = (points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area)
        if self.auto_generator is None or self._auto_generator_params != params:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            self.auto_generator = SAM2AutomaticMaskGenerator(
                model=self.predictor.model,
                points_per_side=points_per_side,
                points_per_batch=128,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=0.6,
                min_mask_region_area=min_mask_region_area,
                crop_n_layers=0,
            )
            self._auto_generator_params = params

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks_dicts = self.auto_generator.generate(image_rgb)

        return [m["segmentation"].astype(bool) for m in masks_dicts]

    def cleanup(self) -> None:
        self.predictor = None
        self.auto_generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
