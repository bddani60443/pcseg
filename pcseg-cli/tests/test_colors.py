"""
Regression test for a real bug: generate_label_color() seeded its per-label RNG
with Python's builtin hash(), which is randomly salted per interpreter process
(PYTHONHASHSEED) for security reasons - so the same label (e.g. "window") got a
different color on every run, defeating the point of a "same label, same color"
preview overlay. Must use a hash that's stable across processes.

generate_label_color() lives in pcseg.core.colors (pure hashlib/numpy, no torch)
- it originally lived in dino_service.py, but that made pcseg_gui/loaders.py (a
plain LAS-classification-coloring caller with nothing to do with DINO) drag torch
in just by importing it, breaking pcseg-gui's plain-viewer-with-no-segmentation-
backends mode. Moved out so this test (and that caller) don't need the optional
`segmentation` extra at all.
"""
import os
import subprocess
import sys

from pcseg.core.colors import generate_label_color


def _color_in_subprocess(hash_seed: str) -> tuple:
    script = (
        "from pcseg.core.colors import generate_label_color; "
        "print(generate_label_color(['window'])[0])"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True, env=env,
    )
    return eval(out.stdout.strip())


def test_generate_label_color_is_stable_across_different_hash_seeds():
    # PYTHONHASHSEED changes what Python's builtin hash() returns for the same
    # string - generate_label_color must not be affected by it at all.
    assert _color_in_subprocess("1") == _color_in_subprocess("2") == generate_label_color(["window"])[0]


def test_generate_label_color_differs_between_distinct_labels():
    building_color, window_color = generate_label_color(["building", "window"])
    assert building_color != window_color
