from __future__ import annotations

import hashlib

import numpy as np


def generate_label_color(labels: list[str]) -> list[tuple]:
    """
    Deterministic (seeded by label text) color per class label, for preview overlays.

    Uses md5 rather than Python's builtin hash(): hash() is salted with a random
    seed per process (PYTHONHASHSEED) for security reasons, so the same label
    (e.g. "window") would get a different color on every run - md5 is stable
    across processes and interpreter runs, which is what a "same label, same
    color" guarantee actually requires.
    """
    colors = []
    for label in labels:
        clean = label.split("(")[0].strip().lower()
        digest = hashlib.md5(clean.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        colors.append(tuple(int(c) for c in rng.integers(50, 255, size=3)))
    return colors
