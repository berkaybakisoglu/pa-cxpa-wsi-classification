import argparse
import multiprocessing
import os
from functools import partial

import cv2
import h5py
import numpy as np
import torchstain
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.extract_features import run_extraction
from src.models.feature_extractors import EXTRACTORS
from src.preprocess import get_patient_list, load_config, open_wsi


def reference_slide(cfg):
    return next(p for p in get_patient_list(cfg) if p[1] == 0)


def mean_saturation(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)[:, :, 1].mean()


def pick_target_patch(cfg):
    patient_id, _, patient_dir = reference_slide(cfg)
    h5_path = os.path.join(cfg["data"]["output_dir"], "patches", f"{patient_id}_coords.h5")
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        level = int(f.attrs["target_level"])
        size = int(f.attrs["patch_size"])

    stain_cfg = cfg["stain_normalization"]
    rng = np.random.default_rng(stain_cfg["reference_seed"])
    idx = rng.choice(len(coords), size=min(stain_cfg["reference_candidates"], len(coords)),
                     replace=False)
    with open_wsi(patient_dir) as slide:
        candidates = [slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
                      for x, y in coords[idx]]
    return max(candidates, key=mean_saturation)


def to_tensor_255(img):
    return to_tensor(img) * 255.0


def fit_normalizer(cfg):
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(to_tensor_255(pick_target_patch(cfg)))
    return normalizer


def normalize_or_none(normalizer, img):
    try:
        normalized = normalizer.normalize(I=to_tensor_255(img), stains=False)[0]
        return Image.fromarray(np.clip(normalized.numpy(), 0, 255).astype(np.uint8))
    except Exception:
        return None


class MacenkoTransform:
    def __init__(self, normalizer, fallbacks, total, base_transform):
        self.normalizer = normalizer
        self.fallbacks = fallbacks
        self.total = total
        self.base_transform = base_transform

    def __call__(self, img):
        normalized = normalize_or_none(self.normalizer, img)
        with self.total.get_lock():
            self.total.value += 1
        if normalized is None:
            with self.fallbacks.get_lock():
                self.fallbacks.value += 1
            normalized = img
        return self.base_transform(normalized)


def main():
    parser = argparse.ArgumentParser(description="Extract features from Macenko-normalised patches")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--extractor", default=None, help=f"one of {', '.join(EXTRACTORS)}")
    parser.add_argument("--patient_id", default=None, help="process a single patient")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patients = get_patient_list(cfg)
    normalizer = fit_normalizer(cfg)
    extractors = [args.extractor] if args.extractor else cfg["feature_extraction"]["extractors"]

    for extractor_name in extractors:
        fallbacks, total = multiprocessing.Value("q", 0), multiprocessing.Value("q", 0)
        run_extraction(cfg, extractor_name, patients, patient_filter=args.patient_id,
                       wrap_transform=partial(MacenkoTransform, normalizer, fallbacks, total),
                       output_name=f"{extractor_name}_stainnorm")
        print(f"  Macenko fallbacks: {fallbacks.value}/{total.value} patches "
              f"({100 * fallbacks.value / max(total.value, 1):.2f}%)")


if __name__ == "__main__":
    main()
