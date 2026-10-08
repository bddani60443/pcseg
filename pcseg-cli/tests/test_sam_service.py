"""
Unit tests for SamService.predict_batch's own wiring - the real SAM2 model is
never loaded (self.predictor is a lightweight fake object, injected directly
past _ensure_loaded's own early-return: `if self.predictor is not None:
return`), so these only verify the batching call shape/validation against
the SAM2ImagePredictor API (see external/sam2-external's own
sam2_image_predictor.py), not real segmentation output.
"""
import numpy as np
import pytest

pytest.importorskip("torch")

from pcseg.services.sam_service import SamService  # noqa: E402


class _FakePredictor:
    """Mimics SAM2ImagePredictor's set_image_batch()/predict_batch() pair
    just enough to verify SamService.predict_batch wires them correctly -
    not a real image encoder/decoder."""

    def __init__(self):
        self.set_image_batch_calls: list = []
        self.predict_batch_calls: list = []

    def set_image_batch(self, image_list):
        self.set_image_batch_calls.append(image_list)

    def predict_batch(self, box_batch=None, point_coords_batch=None, point_labels_batch=None,
                       mask_input_batch=None, multimask_output=True, return_logits=False, normalize_coords=True):
        self.predict_batch_calls.append((box_batch, multimask_output))
        masks = [np.zeros((len(b), 1, 4, 4), dtype=bool) for b in box_batch]
        scores = [np.ones(len(b)) for b in box_batch]
        logits = [np.zeros((len(b), 1, 4, 4)) for b in box_batch]
        return masks, scores, logits


def _make_service_with_fake_predictor() -> tuple[SamService, _FakePredictor]:
    svc = SamService()
    fake = _FakePredictor()
    svc.predictor = fake  # bypasses _ensure_loaded's real model-loading path entirely
    return svc, fake


def test_predict_batch_returns_empty_list_for_no_images_without_touching_the_predictor():
    svc, fake = _make_service_with_fake_predictor()

    result = svc.predict_batch([], [])

    assert result == []
    assert fake.set_image_batch_calls == []


def test_predict_batch_rejects_mismatched_image_and_box_list_lengths():
    svc, _fake = _make_service_with_fake_predictor()
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]
    boxes = [np.array([[0, 0, 2, 2]]), np.array([[0, 0, 2, 2]])]  # one too many

    with pytest.raises(ValueError):
        svc.predict_batch(images, boxes)


def test_predict_batch_calls_set_image_batch_then_predict_batch_with_the_given_boxes():
    svc, fake = _make_service_with_fake_predictor()
    images = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    boxes = [np.array([[0, 0, 2, 2]]), np.array([[1, 1, 3, 3], [0, 0, 4, 4]])]

    result = svc.predict_batch(images, boxes, multimask_output=False)

    assert len(fake.set_image_batch_calls) == 1
    assert fake.set_image_batch_calls[0] == images
    assert len(fake.predict_batch_calls) == 1
    called_boxes, called_multimask = fake.predict_batch_calls[0]
    assert called_multimask is False
    assert [b.tolist() for b in called_boxes] == [b.tolist() for b in boxes]

    # One (masks, scores, logits) tuple per image, in the SAME order, each
    # shaped for that image's OWN box count.
    assert len(result) == 2
    assert result[0][0].shape[0] == 1  # image 0 had 1 box
    assert result[1][0].shape[0] == 2  # image 1 had 2 boxes
