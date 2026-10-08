"""
Regression test for a real bug hit on a real Windows machine: a stale
non-editable pip install copied dino_service.py into site-packages, so
Path(__file__).resolve().parents[3] silently pointed "project root" at the
Python installation directory instead of the actual pcseg-cli checkout where
external/ (the downloaded model weights) actually lives - producing a
FileNotFoundError deep inside model loading with no clue what went wrong.
"""
import os

from pcseg.paths import find_project_root


def test_find_project_root_prefers_the_file_based_root_when_it_has_the_marker(tmp_path):
    real_root = tmp_path / "pcseg-cli"
    marker = "external/weights/model.pt"
    (real_root / "external" / "weights").mkdir(parents=True)
    (real_root / "external" / "weights" / "model.pt").touch()

    fake_file = real_root / "src" / "pcseg" / "services" / "dino_service.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()

    assert find_project_root(marker, str(fake_file)) == real_root


def test_find_project_root_falls_back_to_cwd_when_file_based_root_lacks_the_marker(tmp_path, monkeypatch):
    # Simulates the real bug: __file__ resolves to a site-packages copy (no
    # external/ next to it at all), but cwd is the actual project checkout -
    # matching this project's own documented "run from pcseg-cli/" convention.
    site_packages_copy = tmp_path / "site-packages" / "pcseg" / "services" / "dino_service.py"
    site_packages_copy.parent.mkdir(parents=True)
    site_packages_copy.touch()

    real_root = tmp_path / "pcseg-cli"
    marker = "external/weights/model.pt"
    (real_root / "external" / "weights").mkdir(parents=True)
    (real_root / "external" / "weights" / "model.pt").touch()

    monkeypatch.chdir(real_root)
    assert find_project_root(marker, str(site_packages_copy)) == real_root


def test_find_project_root_falls_back_to_cwds_sibling_pcseg_cli(tmp_path, monkeypatch):
    # Matches pcseg-gui's documented layout: pcseg-cli and pcseg-gui as sibling
    # directories, with pcseg-view run from inside pcseg-gui.
    site_packages_copy = tmp_path / "site-packages" / "pcseg" / "services" / "dino_service.py"
    site_packages_copy.parent.mkdir(parents=True)
    site_packages_copy.touch()

    pcseg_cli = tmp_path / "pcseg-cli"
    marker = "external/weights/model.pt"
    (pcseg_cli / "external" / "weights").mkdir(parents=True)
    (pcseg_cli / "external" / "weights" / "model.pt").touch()

    pcseg_gui = tmp_path / "pcseg-gui"
    pcseg_gui.mkdir()
    monkeypatch.chdir(pcseg_gui)

    assert find_project_root(marker, str(site_packages_copy)) == pcseg_cli


def test_find_project_root_falls_back_to_file_based_guess_when_nothing_has_the_marker(tmp_path, monkeypatch):
    fake_file = tmp_path / "pcseg-cli" / "src" / "pcseg" / "services" / "dino_service.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()

    empty_cwd = tmp_path / "somewhere_else"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    # Nothing has the marker (weights not downloaded yet) - must not raise,
    # must deterministically return the __file__-based guess.
    result = find_project_root("external/weights/model.pt", str(fake_file))
    assert result == fake_file.resolve().parents[3]
