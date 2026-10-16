import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.models.aggregators import PatchMLP
from src.preprocess import get_patient_list, load_config
from src.train import apply_imbalance, load_features, run_configs

STRATEGIES = ("none", "oversample")


def train_patch_classifier(train_features, train_labels, cfg, device, batch_size=256,
                           on_epoch_end=None):
    model_cfg, train_cfg = cfg["model"], cfg["training"]
    X = torch.cat(train_features).to(device)
    y = torch.cat([torch.full((len(feats),), label, dtype=torch.long)
                   for feats, label in zip(train_features, train_labels)]).to(device)

    model = PatchMLP(X.shape[1], model_cfg["hidden_dim"], model_cfg["dropout"]).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    train_curve = []
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        batches = torch.randperm(len(X), device=device).split(batch_size)
        for idx in batches:
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_curve.append(epoch_loss / len(batches))
        if on_epoch_end:
            on_epoch_end(epoch, model)
    return model.eval(), train_curve


def predict_slide(model, test_features, device, batch_size=512):
    with torch.no_grad():
        probs = [F.softmax(model(batch.to(device)), dim=1)[:, 1].cpu()
                 for batch in test_features.split(batch_size)]
    return torch.cat(probs).mean().item()


def patch_fold(extractor_name, strategy, cfg, device,
               test_id, test_label, train_ids, train_labels, seed):
    train_features = [load_features(pid, extractor_name, cfg)[0] for pid in train_ids]
    train_features, train_labels, _ = apply_imbalance(strategy, train_features, train_labels, seed, cfg)
    model, train_curve = train_patch_classifier(train_features, train_labels, cfg, device)
    prob = predict_slide(model, load_features(test_id, extractor_name, cfg)[0], device)
    return {
        "patient_id": test_id,
        "true_label": test_label,
        "predicted_label": int(prob >= 0.5),
        "prob_malignant": prob,
        "prob_benign": 1 - prob,
        "epochs_trained": cfg["training"]["epochs"],
        "final_train_loss": train_curve[-1],
        "train_curve": train_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Patch-level baseline with leave-one-patient-out CV")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--extractor", default=None)
    parser.add_argument("--strategy", default=None, choices=STRATEGIES)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    patients = [(pid, label) for pid, label, _ in get_patient_list(cfg)]
    extractors = [args.extractor] if args.extractor else cfg["feature_extraction"]["extractors"]
    strategies = [args.strategy] if args.strategy else STRATEGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Patients: {len(patients)}, patch baseline, extractors: {extractors}, "
          f"strategies: {strategies}, epochs: {cfg['training']['epochs']}, device: {device}")

    configs = [(f"patch_{extractor}_{strategy}", partial(patch_fold, extractor, strategy, cfg, device))
               for extractor in extractors for strategy in strategies]
    run_configs(configs, patients, cfg)


if __name__ == "__main__":
    main()
