import numpy as np

from pcseg.core.mapping import IndexMapper


def _small_mapper():
    # 2x2 grid, one point per cell, identity row/col mapping before the
    # vertical flip IndexMapper applies internally.
    raw_indices = {
        "row": np.array([0, 0, 1, 1], dtype=np.int32),
        "col": np.array([0, 1, 0, 1], dtype=np.int32),
    }
    return IndexMapper(raw_indices, row_max=1, col_max=1)


def test_map_to_panorama_roundtrip_matches_map_from_panorama():
    mapper = _small_mapper()
    values = np.array([10, 20, 30, 40], dtype=np.uint8)

    pano = mapper.map_to_panorama(values, default_value=0)
    assert pano.shape == (2, 2)

    recovered = mapper.map_from_panorama(pano, default_value=0)
    assert np.array_equal(recovered, values)


def test_map_to_panorama_flags_unmapped_pixels_as_default():
    raw_indices = {"row": np.array([0], dtype=np.int32), "col": np.array([0], dtype=np.int32)}
    mapper = IndexMapper(raw_indices, row_max=1, col_max=1)

    values = np.array([99], dtype=np.uint8)
    pano = mapper.map_to_panorama(values, default_value=7)

    assert pano.shape == (2, 2)
    assert (pano == 7).sum() == 3  # 3 of the 4 cells were never populated


def test_apply_mask_returns_only_indices_under_the_mask():
    mapper = _small_mapper()
    # Identity map_x/map_y: perspective pixel (x, y) samples panorama pixel (x, y).
    map_x, map_y = np.meshgrid(np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32))

    mask_2d = np.zeros((2, 2), dtype=bool)
    mask_2d[0, 0] = True  # selects exactly one perspective pixel

    result = mapper.apply_mask(mask_2d, map_x, map_y, fov=90.0)

    assert len(result) >= 1
    assert result.dtype.kind in ("i", "u")


def test_apply_mask_empty_mask_returns_empty_array():
    mapper = _small_mapper()
    map_x, map_y = np.meshgrid(np.arange(2, dtype=np.float32), np.arange(2, dtype=np.float32))
    mask_2d = np.zeros((2, 2), dtype=bool)

    result = mapper.apply_mask(mask_2d, map_x, map_y, fov=90.0)
    assert len(result) == 0
