import argparse
import gc
import os
import random
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from torch.optim import Adam

from src.models.aggregators import AGGREGATORS, build_aggregator
from src.preprocess import get_patient_list, load_config

STRATEGIES = ("none", "oversample", "smote")
FOLD_COLUMNS = ["fold", "patient_id", "true_label", "predicted_label",
                "prob_malignant", "prob_benign", "epochs_trained", "final_train_loss"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_feature_cache = {"extractor": None, "patients": {}}


def load_features(patient_id, extractor_name, cfg):
    if _feature_cache["extractor"] != extractor_name:
        _feature_cache.update(extractor=extractor_name, patients={})
    cache = _feature_cache["patients"]
    if patient_id not in cache:
        path = os.path.join(cfg["data"]["output_dir"], "features", extractor_name, f"{patient_id}.pt")
        data = torch.load(path, map_location="cpu", weights_only=False)
        cache[patient_id] = (data["features"], data["label"])
    return cache[patient_id]


def apply_oversample(features, labels):
    counts = np.bincount(labels, minlength=2)
    if counts[0] == counts[1]:
        return features, labels

    minority = int(counts.argmin())
    minority_idx = [i for i, label in enumerate(labels) if label == minority]
    n_extra = counts.max() - counts.min()
    extra = [minority_idx[i % len(minority_idx)] for i in range(n_extra)]
    print(f"    Oversample: {len(minority_idx)} -> {counts.max()} minority bags (+{n_extra} copies)")
    return list(features) + [features[i] for i in extra], list(labels) + [labels[i] for i in extra]


def apply_smote(features, labels, seed, k_neighbors):
    k_neighbors = min(k_neighbors, np.bincount(labels, minlength=2).min() - 1)
    if k_neighbors < 1:
        print("    SMOTE: not enough minority samples, skipping")
        return features, labels, [False] * len(features)

    bag_means = torch.stack([f.mean(dim=0) for f in features]).numpy()
    X, y = SMOTE(k_neighbors=k_neighbors, random_state=seed).fit_resample(bag_means, np.array(labels))
    synthetic = [torch.tensor(x).unsqueeze(0).float() for x in X[len(features):]]
    print(f"    SMOTE: added {len(synthetic)} synthetic bags (total: {len(X)})")
    return (list(features) + synthetic,
            list(labels) + [int(label) for label in y[len(features):]],
            [False] * len(features) + [True] * len(synthetic))


def apply_imbalance(strategy, features, labels, seed, cfg):
    if strategy == "none":
        return features, labels, [False] * len(features)
    if strategy == "oversample":
        features, labels = apply_oversample(features, labels)
        return features, labels, [False] * len(features)
    if strategy == "smote":
        return apply_smote(features, labels, seed, cfg["imbalance"]["smote_k_neighbors"])
    raise ValueError(f"Unknown imbalance strategy: {strategy}")


def lopocv_folds(patients):
    for fold, (test_id, test_label) in enumerate(patients):
        train = [(pid, label) for pid, label in patients if pid != test_id]
        yield fold, test_id, test_label, [pid for pid, _ in train], [label for _, label in train]


def train_one_fold(extractor_name, aggregator, strategy, cfg, device,
                   test_id, test_label, train_ids, train_labels, seed, on_epoch_end=None):
    train_cfg = cfg["training"]
    train_features = [load_features(pid, extractor_name, cfg)[0] for pid in train_ids]
    test_features = load_features(test_id, extractor_name, cfg)[0]
    train_features, train_labels, is_synthetic = apply_imbalance(
        strategy, train_features, train_labels, seed, cfg)

    model = build_aggregator(aggregator, train_features[0].shape[1], cfg["model"]).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    train_curve = []
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        order = list(range(len(train_features)))
        random.shuffle(order)
        for idx in order:
            label = train_labels[idx]
            optimizer.zero_grad()
            # SMOTE bags are a single synthetic vector, so there are no patches to rank for the instance loss.
            results = model(train_features[idx].to(device), label=None if is_synthetic[idx] else label)
            loss = criterion(results["logits"], torch.tensor([label], dtype=torch.long, device=device))
            loss = loss + results["instance_loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_curve.append(epoch_loss / len(train_features))
        if on_epoch_end:
            on_epoch_end(epoch, model)

    model.eval()
    with torch.no_grad():
        results = model(test_features.to(device), return_attention=True)
        prob = F.softmax(results["logits"], dim=1)[0]

    return {
        "patient_id": test_id,
        "true_label": test_label,
        "predicted_label": prob.argmax().item(),
        "prob_malignant": prob[1].item(),
        "prob_benign": prob[0].item(),
        "attention_scores": results["attention"].cpu().numpy(),
        "epochs_trained": train_cfg["epochs"],
        "final_train_loss": train_curve[-1],
        "train_curve": train_curve,
    }


def results_dir(cfg):
    path = os.path.join(cfg["data"]["output_dir"], "results")
    os.makedirs(path, exist_ok=True)
    return path


def run_lopocv(config_name, train_fold, patients, cfg):
    partial_path = os.path.join(results_dir(cfg), f"{config_name}_partial.pt")
    fold_results = torch.load(partial_path, weights_only=False) if os.path.exists(partial_path) else []
    done = {r["patient_id"] for r in fold_results}
    if done:
        print(f"  Resuming: {len(done)}/{len(patients)} folds already done")

    for fold, test_id, test_label, train_ids, train_labels in lopocv_folds(patients):
        if test_id in done:
            continue
        seed = cfg["seed"] + fold
        set_seed(seed)
        result = train_fold(test_id, test_label, train_ids, train_labels, seed)
        result["fold"] = fold
        print(f"    Fold {fold + 1:2d}/{len(patients)} | {test_id:>6s} "
              f"({'malignant' if test_label else 'benign':>9s}) | "
              f"P(malig)={result['prob_malignant']:.3f} | pred={result['predicted_label']} | "
              f"{'OK' if result['predicted_label'] == test_label else 'WRONG'} | "
              f"loss {result['final_train_loss']:.3f}")
        fold_results.append(result)
        torch.save(fold_results, partial_path)
        gc.collect()
        torch.cuda.empty_cache()

    if os.path.exists(partial_path):
        os.remove(partial_path)
    return fold_results


def save_results(fold_results, config_name, cfg):
    prefix = os.path.join(results_dir(cfg), config_name)
    pd.DataFrame(fold_results, columns=FOLD_COLUMNS).to_csv(f"{prefix}_folds.csv", index=False)
    curves = [{"fold": r["fold"], "patient_id": r["patient_id"], "epoch": epoch, "train_loss": loss}
              for r in fold_results for epoch, loss in enumerate(r["train_curve"])]
    pd.DataFrame(curves).to_csv(f"{prefix}_curves.csv", index=False)
    if "attention_scores" in fold_results[0]:
        torch.save({r["patient_id"]: r["attention_scores"] for r in fold_results},
                   f"{prefix}_attention.pt")
    print(f"  Saved: {prefix}_*")


def run_configs(configs, patients, cfg):
    for config_name, train_fold in configs:
        if os.path.exists(os.path.join(results_dir(cfg), f"{config_name}_folds.csv")):
            print(f"\n[{config_name}] Already completed, skipping.")
            continue
        print(f"\n{'=' * 60}\nConfig: {config_name}\n{'=' * 60}")
        t0 = time.time()
        fold_results = run_lopocv(config_name, train_fold, patients, cfg)
        save_results(fold_results, config_name, cfg)
        correct = sum(r["predicted_label"] == r["true_label"] for r in fold_results)
        print(f"\n  {config_name}: {correct}/{len(fold_results)} correct, {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="MIL training with leave-one-patient-out CV")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--aggregator", default="clam_sb", choices=AGGREGATORS)
    parser.add_argument("--extractor", default=None)
    parser.add_argument("--strategy", default=None, choices=STRATEGIES)
    args = parser.parse_args()

    cfg = load_config(args.config)
    patients = [(pid, label) for pid, label, _ in get_patient_list(cfg)]
    extractors = [args.extractor] if args.extractor else cfg["feature_extraction"]["extractors"]
    strategies = [args.strategy] if args.strategy else cfg["imbalance"]["strategies"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = "" if args.aggregator == "clam_sb" else f"{args.aggregator}_"
    print(f"Patients: {len(patients)}, aggregator: {args.aggregator}, "
          f"extractors: {extractors}, strategies: {strategies}, device: {device}")

    configs = [(f"{prefix}{extractor}_{strategy}",
                partial(train_one_fold, extractor, args.aggregator, strategy, cfg, device))
               for extractor in extractors for strategy in strategies]
    run_configs(configs, patients, cfg)


if __name__ == "__main__":
    main()
