from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable

# (config filename, checkpoint filename) per size - see install_segmentation_backends.sh.
DINO_SIZES: dict[str, tuple[str, str]] = {
    "tiny": ("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth"),
    "base": ("GroundingDINO_SwinB_cfg.py", "groundingdino_swinb_cogcoor.pth"),
}
SAM_SIZES: dict[str, tuple[str, str]] = {
    "tiny": ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}

_root: Path | None = None


def project_root() -> Path:
    """The pcseg-cli project root (where scripts/ and external/ live), found
    the same way the CLI's own model services find it (pcseg.paths.
    find_project_root: the source tree of an editable install, else the
    current directory or its sibling pcseg-cli/) - so the viewer and the CLI
    always agree on where the downloaded weights are. Anchored on the empty
    `pcseg.services` package, not dino_service.py/sam_service.py - see module
    docstring."""
    global _root
    if _root is None:
        from pcseg.paths import find_project_root

        import pcseg.services
        _root = find_project_root("scripts/install_segmentation_backends.sh", pcseg.services.__file__)
    return _root


def dino_paths(size: str) -> tuple[Path, Path]:
    cfg_name, ckpt_name = DINO_SIZES[size]
    root = project_root()
    return (
        root / "external/GroundingDINO-external/groundingdino/config" / cfg_name,
        root / "external/GroundingDINO-external/weights" / ckpt_name,
    )


def sam_paths(size: str) -> tuple[str, Path]:
    """Returns (config_path, checkpoint_path) - config_path is the Hydra-resolved
    relative string SamService expects (see sam_service.DEFAULT_CONFIG_PATH's own
    comment: CWD-independent, package-resolved, NOT a plain filesystem path), so
    only checkpoint_path is ever checked for on-disk existence."""
    cfg_name, ckpt_name = SAM_SIZES[size]
    root = project_root()
    return f"configs/sam2.1/{cfg_name}", root / "external/sam2-external/checkpoints" / ckpt_name


def dino_checkpoint_status(size: str) -> tuple[bool, int]:
    _cfg, ckpt = dino_paths(size)
    if ckpt.is_file():
        return True, ckpt.stat().st_size
    return False, 0


def sam_checkpoint_status(size: str) -> tuple[bool, int]:
    _cfg, ckpt = sam_paths(size)
    if ckpt.is_file():
        return True, ckpt.stat().st_size
    return False, 0


def human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def install_script_path() -> Path:
    name = "install_segmentation_backends.ps1" if platform.system() == "Windows" else "install_segmentation_backends.sh"
    return project_root() / "scripts" / name


def _windows_powershell_executable() -> str:
    """Prefers `pwsh` (PowerShell 7+) if it's actually on PATH, falling back
    to `powershell` (Windows PowerShell 5.1, built into every Windows install
    since Vista/7 - unlike `pwsh`, which is a separate, optional install).
    Real crash live: a machine without PowerShell 7 raised a bare
    FileNotFoundError ([WinError 2], from deep inside subprocess.Popen) the
    moment the install script tried to run at all, since `pwsh` was hardcoded
    with no fallback. install_segmentation_backends.ps1's own header already
    notes it's written against plain PS 5.1-compatible syntax specifically so
    this fallback is safe (confirmed: no PS7-only operators - ternary `?:`,
    null-coalescing `??`, pipeline chains `&&`/`||` - anywhere in it)."""
    for candidate in ("pwsh", "powershell"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "Neither 'pwsh' (PowerShell 7+) nor 'powershell' (Windows PowerShell) was found "
        "on PATH - one of them is needed to run the install script. Windows normally "
        "ships 'powershell' built in, so if it's missing here, this machine's PATH is "
        "likely misconfigured."
    )


def run_install_script(dino_size: str, sam_size: str,
                        progress_callback: Callable[[int, str], None] | None = None) -> bool:
    """
    Runs install_segmentation_backends.{sh,ps1} as a subprocess (never
    reimplements its fetch logic in Python - see the script's own header for
    why: pinned commits/URLs that would otherwise drift out of sync), streaming
    every stdout/stderr line to `progress_callback` as it arrives (percentage is
    always 0 - the script reports no fine-grained progress, only status lines).
    Meant to run inside workers.run_in_background on a background thread; raises
    RuntimeError if the script exits non-zero (workers.py turns that into the
    caller's on_error).
    """
    script = install_script_path()
    if not script.is_file():
        raise RuntimeError(f"Install script not found: {script}")

    if platform.system() == "Windows":
        # -ExecutionPolicy Bypass scopes ONLY to this one process - it does NOT
        # change the machine's/user's persistent policy. Needed because
        # Windows' default policy (Restricted, most machines' out-of-the-box
        # setting) refuses to run ANY local .ps1 file that isn't digitally
        # signed - reported live: "is not digitally signed. You cannot run
        # this script on the current system" (UnauthorizedAccess), for a
        # script that was never going to be signed (it's project source, not
        # a distributed/signed release artifact).
        cmd = [
            _windows_powershell_executable(), "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-DinoSize", dino_size, "-SamSize", sam_size,
        ]
    else:
        cmd = ["bash", str(script), "--dino-size", dino_size, "--sam-size", sam_size]

    proc = subprocess.Popen(
        cmd, cwd=str(project_root()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if progress_callback is not None and line:
            progress_callback(0, line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"install_segmentation_backends exited with code {returncode}")
    return True
