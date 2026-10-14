from __future__ import annotations

from pathlib import Path


def find_project_root(marker_relative_path: str, file: str) -> Path:
    """
    Returns the pcseg-cli project root - the directory `external/` (the
    segmentation backends' downloaded weights) lives under.

    `Path(file).resolve().parents[1]` alone assumes an editable
    (`pip install -e .`) install, where `__file__` still points into the real
    source tree under `pcseg-cli/services/...`. A regular (non-editable)
    install copies this file into site-packages instead, silently pointing
    "project root" at the Python installation directory rather than wherever
    `external/` was actually downloaded to by install_segmentation_backends -
    this happened on a real Windows machine (from a stale non-editable
    install left over from before `-e` was used the first time) and produced
    a FileNotFoundError several layers deep inside model loading, far from
    the actual cause.

    Tries, in order: the file's own directory tree (correct for a genuine
    editable install), the current working directory (matches this
    project's own documented "run pcseg from pcseg-cli/" convention), and
    the working directory's sibling `pcseg-cli/` (matches pcseg-gui's
    documented "pcseg-cli and pcseg-gui live next to each other" layout) -
    returns the first candidate that actually contains
    `marker_relative_path`. Falls back to the `file`-based guess if none of
    them do (e.g. the weights simply haven't been downloaded yet), so
    callers still get a deterministic, previously-correct path to report in
    their own error messages rather than a hard crash here.
    """
    file_based = Path(file).resolve().parents[1]
    candidates = [file_based, Path.cwd(), Path.cwd().parent / "pcseg-cli"]
    for root in candidates:
        if (root / marker_relative_path).exists():
            return root
    return file_based
