from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from paths import find_project_root

# GroundingDINO's text encoder is a plain HuggingFace `bert-base-uncased` (see
# external/GroundingDINO-external/groundingdino/config/*.py's text_encoder_type) -
# loading it normally makes `transformers` reach out to huggingface.co on every
# run (even once the weights are already cached locally) just to check for
# updates. That HEAD request has no bearing on whether inference works, but a
# flaky network or an SSL-inspecting corporate proxy turns it into repeated
# multi-second retries or an outright failure, on every single run, for a
# dependency that should only ever need the internet once. Forcing offline
# mode makes `transformers`/`huggingface_hub` use the local cache only, with
# no network call at all - install_segmentation_backends.{sh,ps1} pre-warms
# that cache, so by the time pcseg actually runs, it never needs to be online
# for this. setdefault (not direct assignment) so an explicit user override
# (e.g. HF_HUB_OFFLINE=0 to force a re-check) still wins. Must happen before
# `transformers`/`huggingface_hub` is imported anywhere in the process - both
# read this once at import time - which is why it's set here at module scope:
# this module's own `_ensure_loaded`/`run_inference` are the first (and only,
# in normal pcseg usage) place that import chain gets triggered.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Both loaded via plain filesystem calls (SLConfig.fromfile / torch.load), so unlike
# SAM2's Hydra-resolved config path these are CWD-relative - anchor them to this repo's
# own root so they still resolve when pcseg is invoked from somewhere else (e.g. the
# pcseg-gui viewer, or `pcseg` on PATH run from any directory). See find_project_root
# for why this isn't just Path(__file__).resolve().parents[1].
_DINO_CKPT_REL = "external/GroundingDINO-external/weights/groundingdino_swint_ogc.pth"
_PROJECT_ROOT = find_project_root(_DINO_CKPT_REL, __file__)
DEFAULT_CONFIG_PATH = str(_PROJECT_ROOT / "external/GroundingDINO-external/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DEFAULT_CHECKPOINT_PATH = str(_PROJECT_ROOT / _DINO_CKPT_REL)


class DinoService:
    """Manages the Grounding DINO model state, lazy loading, and inference dispatching."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                 device: str | None = None) -> None:
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return

        for path, kind in ((self.config_path, "config"), (self.checkpoint_path, "checkpoint")):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"GroundingDINO {kind} not found: {path!r}. Run "
                    f"scripts/install_segmentation_backends.{{sh,ps1}} from the pcseg-cli "
                    f"directory, or pass --dino-config/--dino-checkpoint explicitly if you "
                    f"downloaded the weights somewhere else."
                )

        from groundingdino.util.inference import load_model

        logging.info(f"Loading Grounding DINO model ({self.device})...")
        try:
            model = load_model(self.config_path, self.checkpoint_path, device=self.device)
        except OSError as e:
            raise OSError(
                "Couldn't load GroundingDINO's text encoder ('bert-base-uncased' from "
                "HuggingFace) from the local cache, and this process runs in offline mode "
                "(see dino_service.py) so it won't reach the network to fetch it. Run "
                "scripts/install_segmentation_backends.{sh,ps1} to pre-fetch it, or set "
                "HF_HUB_OFFLINE=0 to allow a one-off online fetch."
            ) from e
        self.model = model.to(self.device)
        logging.info("Grounding DINO loaded.")

    def run_inference(self, img: np.ndarray, prompt: str, box_thresh: float = 0.3,
                       text_thresh: float = 0.25) -> tuple[np.ndarray, np.ndarray, list[str]]:
        import groundingdino.datasets.transforms as gd_transforms
        from groundingdino.util.inference import predict
        from PIL import Image

        self._ensure_loaded()

        h, w, _ = img.shape
        image_pil = Image.fromarray(img)

        transform = gd_transforms.Compose([
            gd_transforms.RandomResize([800], max_size=1333),
            gd_transforms.ToTensor(),
            gd_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model, image=image_tensor, caption=prompt,
                box_threshold=box_thresh, text_threshold=text_thresh, device=self.device,
            )

        if len(boxes) == 0:
            return np.array([]), np.array([]), []

        areas = boxes[:, 2] * boxes[:, 3]
        sorted_indices = torch.argsort(areas, descending=True)
        boxes = boxes[sorted_indices]
        logits = logits[sorted_indices]
        phrases = [phrases[i] for i in sorted_indices.tolist()]

        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)

        return boxes.cpu().numpy().astype(int), logits.cpu().numpy(), phrases

    def cleanup(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
