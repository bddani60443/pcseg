"""
Regression test for a real bug hit on a real Windows machine: GroundingDINO's
text encoder (a plain HuggingFace `bert-base-uncased`) makes `transformers`
reach out to huggingface.co on every run - even once the weights are already
cached locally - just to check for updates. An SSL-inspecting proxy turned
that into repeated failed retries on every single `pcseg run`/`auto-segment`/
`pick-segment` invocation. dino_service.py now forces offline mode at import
time so that check never happens once the cache is warm (see
scripts/install_segmentation_backends.{sh,ps1}, which pre-warms it).
"""
import os

import pytest

pytest.importorskip("torch")


def test_dino_service_forces_offline_mode_by_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    import importlib

    import pcseg.services.dino_service as dino_service
    importlib.reload(dino_service)

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_dino_service_does_not_override_an_explicit_user_choice(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    import importlib

    import pcseg.services.dino_service as dino_service
    importlib.reload(dino_service)

    assert os.environ["HF_HUB_OFFLINE"] == "0"
