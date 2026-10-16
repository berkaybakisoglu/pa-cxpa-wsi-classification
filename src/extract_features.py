import argparse
import os
import time

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.feature_extractors import EXTRACTORS, default_weights_path, get_extractor
from src.preprocess import get_patient_list, load_config, open_wsi


class PatchDataset(Dataset):
    # OpenSlide handles are not picklable; each DataLoader worker opens its own on first access.
    def __init__(self, patient_dir, coords, target_level, patch_size, transform):
        self.patient_dir = patient_dir
        self.coords = coords
        self.target_level = target_level
        self.patch_size = patch_size
        self.transform = transform
        self.slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.slide is None:
            self.slide = open_wsi(self.patient_dir)
        x, y = map(int, self.coords[idx])
        patch = self.slide.read_region((x, y), self.target_level,
                                       (self.patch_size, self.patch_size)).convert("RGB")
        return self.transform(patch) if self.transform else patch


def extract_features_for_patient(patient_id, label, patient_dir, h5_path,
                                 model, transform, feat_dim, device,
                                 batch_size=64, num_workers=0):
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        target_level = int(f.attrs["target_level"])
        patch_size = int(f.attrs["patch_size"])

    dataset = PatchDataset(patient_dir, coords, target_level, patch_size, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"    {patient_id}", leave=False):
            all_features.append(model(batch.to(device)).cpu())

    features = torch.cat(all_features, dim=0)
    assert features.shape == (len(coords), feat_dim), \
        f"Shape mismatch: {features.shape} vs ({len(coords)}, {feat_dim})"

    return {
        "features": features,
        "coords": coords,
        "label": label,
        "patient_id": patient_id,
    }


def run_extraction(cfg, extractor_name, patients, patient_filter=None,
                   wrap_transform=None, output_name=None):
    output_name = output_name or extractor_name
    feat_cfg = cfg["feature_extraction"]
    output_dir = os.path.join(cfg["data"]["output_dir"], "features", output_name)
    patches_dir = os.path.join(cfg["data"]["output_dir"], "patches")
    os.makedirs(output_dir, exist_ok=True)

    if patient_filter:
        patients = [p for p in patients if p[0] == patient_filter]
        if not patients:
            print(f"Patient {patient_filter} not found!")
            return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}\nExtractor: {output_name} (device={device})\n{'=' * 60}")
    model, transform, feat_dim = get_extractor(
        extractor_name, weights_path=default_weights_path(cfg, extractor_name), device=device)
    if wrap_transform:
        transform = wrap_transform(transform)
    print(f"  Feature dim: {feat_dim}")

    results = []
    for patient_id, label, patient_dir in patients:
        output_path = os.path.join(output_dir, f"{patient_id}.pt")
        h5_path = os.path.join(patches_dir, f"{patient_id}_coords.h5")

        if os.path.exists(output_path):
            n = torch.load(output_path, map_location="cpu", weights_only=False)["features"].shape[0]
            print(f"  [{patient_id}] Already extracted ({n} patches), skipping.")
            results.append((patient_id, n))
            continue

        if not os.path.exists(h5_path):
            print(f"  [{patient_id}] No patch coords found at {h5_path}, skipping.")
            continue

        t0 = time.time()
        data = extract_features_for_patient(
            patient_id, label, patient_dir, h5_path,
            model, transform, feat_dim, device,
            batch_size=feat_cfg["batch_size"],
            num_workers=feat_cfg.get("num_workers", 0),
        )
        torch.save(data, output_path)
        n = data["features"].shape[0]
        print(f"  [{patient_id}] {n} patches, shape ({n}, {feat_dim}), {time.time() - t0:.1f}s")
        results.append((patient_id, n))

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract features from WSI patches")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--extractor", default=None, help=f"one of {', '.join(EXTRACTORS)}")
    parser.add_argument("--patient_id", default=None, help="process a single patient")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patients = get_patient_list(cfg)
    extractors = [args.extractor] if args.extractor else cfg["feature_extraction"]["extractors"]
    print(f"Patients: {len(patients)}, Extractors: {extractors}")

    for extractor_name in extractors:
        results = run_extraction(cfg, extractor_name, patients, patient_filter=args.patient_id)
        print(f"\n  Summary for {extractor_name}:")
        print(f"  {len(results)} patients, {sum(n for _, n in results)} total patches")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
