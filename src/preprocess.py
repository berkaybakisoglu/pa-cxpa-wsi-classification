"""Tissue detection and patch-coordinate extraction for MIRAX whole-slide images.

Usage:
    python -m src.preprocess [--patient_id <patient_id>]
"""

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import yaml
from tqdm import tqdm

MORPHOLOGY_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
FALLBACK_LEVEL0_MPP = 0.25


def load_config(config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_patient_list(cfg):
    patients = []
    for data_dir, label in [(cfg["data"]["benign_dir"], 0), (cfg["data"]["malign_dir"], 1)]:
        for d in sorted(Path(data_dir).iterdir()):
            if d.is_dir():
                patients.append((d.name, label, str(d)))
    return patients


def open_wsi(patient_dir):
    # OpenSlide's MIRAX driver opens a slide only through a <name>.mrxs file
    # next to the <name>/ directory; an empty placeholder is enough, so one is
    # created in the data directory if missing.
    mrxs_path = f"{os.path.normpath(patient_dir)}.mrxs"
    if not os.path.exists(mrxs_path):
        open(mrxs_path, "w").close()
    return openslide.OpenSlide(mrxs_path)


def level0_mpp(slide):
    for key in ("openslide.mpp-x", "mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X"):
        if slide.properties.get(key) is not None:
            return float(slide.properties[key])
    print(f"  WARNING: Cannot determine MPP, assuming {FALLBACK_LEVEL0_MPP} µm/px")
    return FALLBACK_LEVEL0_MPP


def find_target_level(slide, target_mpp=0.5):
    base_mpp = level0_mpp(slide)
    downsamples = slide.level_downsamples
    best_level = min(range(len(downsamples)),
                     key=lambda i: abs(downsamples[i] - target_mpp / base_mpp))
    actual_mpp = base_mpp * downsamples[best_level]
    print(f"  Target MPP: {target_mpp:.3f}, Selected level {best_level} "
          f"(downsample {downsamples[best_level]:.2f}x, actual MPP {actual_mpp:.3f})")
    return best_level, actual_mpp


def detect_tissue(slide, thumbnail_downsample=64, saturation_threshold=15):
    width, height = slide.dimensions
    thumbnail = slide.get_thumbnail((max(width // thumbnail_downsample, 1),
                                     max(height // thumbnail_downsample, 1)))
    thumbnail = np.array(thumbnail.convert("RGB"))

    saturation = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:, :, 1]
    tissue_mask = (saturation > saturation_threshold).astype(np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, MORPHOLOGY_KERNEL)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, MORPHOLOGY_KERNEL)

    thumb_h, thumb_w = thumbnail.shape[:2]
    level0_per_thumb_px = (width / thumb_w, height / thumb_h)
    return tissue_mask, thumbnail, level0_per_thumb_px


def extract_patch_coords(slide, tissue_mask, target_level, patch_size=256,
                         tissue_threshold=0.5, *, scale_factors):
    level_w, level_h = slide.level_dimensions[target_level]
    downsample = slide.level_downsamples[target_level]
    scale_x, scale_y = scale_factors
    patch_size_l0 = int(patch_size * downsample)
    mask_h, mask_w = tissue_mask.shape

    coords = []
    for j in range(level_h // patch_size):
        for i in range(level_w // patch_size):
            x_l0 = int(i * patch_size * downsample)
            y_l0 = int(j * patch_size * downsample)

            tx_start = max(0, min(int(x_l0 / scale_x), mask_w - 1))
            tx_end = max(tx_start + 1, min(int((x_l0 + patch_size_l0) / scale_x), mask_w))
            ty_start = max(0, min(int(y_l0 / scale_y), mask_h - 1))
            ty_end = max(ty_start + 1, min(int((y_l0 + patch_size_l0) / scale_y), mask_h))

            if tissue_mask[ty_start:ty_end, tx_start:tx_end].mean() >= tissue_threshold:
                coords.append((x_l0, y_l0))

    return np.array(coords, dtype=np.int64)


def save_patch_data(output_path, coords, target_level, patch_size, level_downsample,
                    actual_mpp, patient_id, label):
    with h5py.File(output_path, "w") as f:
        f.create_dataset("coords", data=coords)
        f.attrs["target_level"] = target_level
        f.attrs["patch_size"] = patch_size
        f.attrs["level_downsample"] = level_downsample
        f.attrs["actual_mpp"] = actual_mpp
        f.attrs["patient_id"] = patient_id
        f.attrs["label"] = label
        f.attrs["num_patches"] = len(coords)


def process_patient(patient_id, label, patient_dir, cfg, output_dir):
    output_path = os.path.join(output_dir, f"{patient_id}_coords.h5")
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            n = f.attrs["num_patches"]
        print(f"  [{patient_id}] Already processed ({n} patches), skipping.")
        return n

    prep_cfg = cfg["preprocessing"]
    with open_wsi(patient_dir) as slide:
        target_level, actual_mpp = find_target_level(slide, prep_cfg["target_mpp"])
        tissue_mask, _, scale_factors = detect_tissue(
            slide,
            thumbnail_downsample=prep_cfg["thumbnail_downsample"],
            saturation_threshold=prep_cfg["saturation_threshold"],
        )
        print(f"  [{patient_id}] Tissue coverage: {tissue_mask.mean() * 100:.1f}%")
        coords = extract_patch_coords(
            slide, tissue_mask, target_level,
            patch_size=prep_cfg["patch_size"],
            tissue_threshold=prep_cfg["tissue_threshold"],
            scale_factors=scale_factors,
        )
        save_patch_data(
            output_path, coords, target_level, prep_cfg["patch_size"],
            slide.level_downsamples[target_level], actual_mpp, patient_id, label,
        )
    print(f"  [{patient_id}] Saved {len(coords)} patches")
    return len(coords)


def main():
    parser = argparse.ArgumentParser(description="Preprocess WSIs: tissue detection + patch extraction")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--patient_id", default=None, help="process a single patient")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = os.path.join(cfg["data"]["output_dir"], "patches")
    os.makedirs(output_dir, exist_ok=True)

    patients = get_patient_list(cfg)
    n_malignant = sum(label for _, label, _ in patients)
    print(f"Found {len(patients)} patients ({len(patients) - n_malignant} benign, {n_malignant} malignant)")

    if args.patient_id:
        patients = [p for p in patients if p[0] == args.patient_id]
        if not patients:
            print(f"Patient {args.patient_id} not found!")
            return

    results = []
    for patient_id, label, patient_dir in tqdm(patients, desc="Processing patients"):
        n_patches = process_patient(patient_id, label, patient_dir, cfg, output_dir)
        results.append((patient_id, "malignant" if label == 1 else "benign", n_patches))

    print(f"\n{'Patient ID':<12} {'Label':<10} {'Patches':>8}")
    for patient_id, label_name, n in results:
        print(f"{patient_id:<12} {label_name:<10} {n:>8}")
    print(f"{'TOTAL':<22} {sum(n for *_, n in results):>8}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
