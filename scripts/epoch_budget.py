import argparse
import os
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from scripts.fig_lang import agg_styles
from src.models.aggregators import AGGREGATORS
from src.patch_baseline import predict_slide, train_patch_classifier
from src.preprocess import get_patient_list, load_config
from src.train import load_features, lopocv_folds, set_seed, train_one_fold

EXTRACTOR = "ctranspath"
MODELS = ("patch_level", *AGGREGATORS)
CHECKPOINTS = (3, 5, 10, 15, 20, 30, 40, 50)
TRAINLOSS_EPOCHS = 60
TRAINLOSS_FOLDS = (0, 7, 14, 16)
LABELS = {
    "tr": {"epochs": "Epoch sayısı (N)", "auroc": "AUROC (LOPOCV)",
           "epoch": "Epoch", "train_loss": "Eğitim kaybı (fold ortalaması)"},
    "en": {"epochs": "Number of epochs (N)", "auroc": "AUROC (LOPOCV)",
           "epoch": "Epoch", "train_loss": "Training loss (fold mean)"},
}


def train_fold(model_name, cfg, device, fold, test_id, test_label, train_ids, train_labels,
               on_epoch_end=None):
    seed = cfg["seed"] + fold
    set_seed(seed)
    if model_name == "patch_level":
        features = [load_features(pid, EXTRACTOR, cfg)[0] for pid in train_ids]
        _, curve = train_patch_classifier(features, train_labels, cfg, device, on_epoch_end=on_epoch_end)
        return curve
    return train_one_fold(EXTRACTOR, model_name, "none", cfg, device, test_id, test_label,
                          train_ids, train_labels, seed, on_epoch_end)["train_curve"]


def record_checkpoint(probs, model_name, test_features, device, epoch, model):
    if epoch not in CHECKPOINTS:
        return
    model.eval()
    with torch.no_grad():
        if model_name == "patch_level":
            probs[epoch] = predict_slide(model, test_features, device)
        else:
            probs[epoch] = F.softmax(model(test_features.to(device))["logits"], dim=1)[0, 1].item()


def epoch_sweep(cfg, device, patients):
    cfg["training"]["epochs"] = max(CHECKPOINTS)
    probs = {m: [] for m in MODELS}
    for fold, *split in lopocv_folds(patients):
        print(f"fold {fold + 1}/{len(patients)}", flush=True)
        test_features = load_features(split[0], EXTRACTOR, cfg)[0]
        for m in MODELS:
            fold_probs = {}
            train_fold(m, cfg, device, fold, *split,
                       on_epoch_end=partial(record_checkpoint, fold_probs, m, test_features, device))
            probs[m].append(fold_probs)
    y_true = [label for _, label in patients]
    return pd.DataFrame([{"config": m, "epochs": n,
                          "auroc": round(roc_auc_score(y_true, [p[n] for p in probs[m]]), 4)}
                         for m in MODELS for n in CHECKPOINTS])


def trainloss_curve(cfg, device, patients):
    cfg["training"]["epochs"] = TRAINLOSS_EPOCHS
    curves = {m: [] for m in MODELS}
    for fold, *split in lopocv_folds(patients):
        if fold not in TRAINLOSS_FOLDS:
            continue
        print(f"fold {fold + 1}/{len(patients)}", flush=True)
        for m in MODELS:
            curves[m].append(train_fold(m, cfg, device, fold, *split))
    return pd.DataFrame([{"config": m, "epoch": epoch, "train_loss": round(float(loss), 5)}
                         for m in MODELS
                         for epoch, loss in enumerate(np.mean(curves[m], axis=0), start=1)])


def plot(df, x, y, path, lang, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, (label, colour) in agg_styles(lang).items():
        rows = df[df["config"] == m]
        ax.plot(rows[x], rows[y], marker=None if log_y else "o", color=colour, label=label)
    ax.set_xlabel(LABELS[lang][x])
    ax.set_ylabel(LABELS[lang][y])
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evidence for the fixed epoch budget (CTransPath, no imbalance correction)")
    parser.add_argument("mode", choices=("sweep", "trainloss"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=os.path.join("outputs", "figures"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    patients = [(pid, label) for pid, label, _ in get_patient_list(cfg)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "sweep":
        name, df, x, y, log_y = "epoch_sweep", epoch_sweep(cfg, device, patients), "epochs", "auroc", False
    else:
        name, df, x, y, log_y = "trainloss_curve", trainloss_curve(cfg, device, patients), "epoch", "train_loss", True
    df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
    for lang in LABELS:
        plot(df, x, y, os.path.join(args.out, f"{name}_{lang}.png"), lang, log_y)
    print(df.pivot(index=x, columns="config", values=y).to_string())


if __name__ == "__main__":
    main()
