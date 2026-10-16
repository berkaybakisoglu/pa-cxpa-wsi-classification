from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from src.preprocess import (detect_tissue, extract_patch_coords, find_target_level, get_patient_list,
                            load_config, open_wsi, process_patient, save_patch_data)


class FakeSlide:
    def __init__(self, size=(4096, 2048), tissue_box=(1024, 512, 2048, 1536)):
        self.dimensions = size
        self.level_downsamples = (1.0, 2.0, 4.0)
        self.level_dimensions = tuple((size[0] // int(d), size[1] // int(d)) for d in self.level_downsamples)
        self.properties = {"openslide.mpp-x": "0.2435"}
        self.tissue_box = tissue_box

    def get_thumbnail(self, size):
        image = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
        sx, sy = size[0] / self.dimensions[0], size[1] / self.dimensions[1]
        x0, y0, x1, y1 = self.tissue_box
        image[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)] = (190, 90, 160)
        return Image.fromarray(image)


def test_target_level_is_closest_to_20x():
    level, mpp = find_target_level(FakeSlide(), target_mpp=0.5)
    assert level == 1 and mpp == pytest.approx(0.487)


def test_tissue_mask_marks_stained_region_only():
    mask, thumbnail, scale = detect_tissue(FakeSlide(), thumbnail_downsample=64)
    assert mask.shape == thumbnail.shape[:2] == (32, 64)
    assert scale == (64, 64)
    assert mask[16, 24] == 1 and mask[2, 2] == 0


def test_patch_coords_cover_only_tissue():
    slide = FakeSlide()
    mask, _, scale = detect_tissue(slide, thumbnail_downsample=64)
    coords = extract_patch_coords(slide, mask, 1, patch_size=256, tissue_threshold=0.5, scale_factors=scale)
    x0, y0, x1, y1 = slide.tissue_box
    assert len(coords) == ((x1 - x0) // 512) * ((y1 - y0) // 512)
    assert ((coords[:, 0] >= x0) & (coords[:, 0] < x1) & (coords[:, 1] >= y0) & (coords[:, 1] < y1)).all()


def test_patch_data_roundtrip(tmp_path):
    coords = np.array([[0, 0], [512, 1024]], dtype=np.int64)
    path = tmp_path / "P01_coords.h5"
    save_patch_data(path, coords, 1, 256, 2.0, 0.487, "P01", 1)
    with h5py.File(path, "r") as f:
        assert (f["coords"][:] == coords).all()
        assert (f.attrs["target_level"], f.attrs["num_patches"], f.attrs["label"]) == (1, 2, 1)


@pytest.mark.data
@pytest.mark.skipif(not (Path("Bening").is_dir() and Path("Malign").is_dir()), reason="private slides not available")
def test_real_slide_patches(tmp_path):
    cfg = load_config()
    patient_id, label, patient_dir = get_patient_list(cfg)[0]
    n = process_patient(patient_id, label, patient_dir, cfg, str(tmp_path))
    with h5py.File(tmp_path / f"{patient_id}_coords.h5", "r") as f:
        coords, level, size = f["coords"][:], int(f.attrs["target_level"]), int(f.attrs["patch_size"])
    assert n == len(coords) > 50
    with open_wsi(patient_dir) as slide:
        assert slide.level_count > 1
        patch = np.array(slide.read_region(tuple(map(int, coords[len(coords) // 2])), level, (size, size)).convert("RGB"))
    assert patch.std() > 5
