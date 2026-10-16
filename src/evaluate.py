import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

from src.preprocess import load_config

AGGREGATOR_PREFIXES = (("mean_pool_", "mean_pool"), ("abmil_", "abmil"), ("patch_", "patch_level"))
YOUDEN_METRICS = ("sensitivity", "specificity", "f1", "balanced_accuracy")
PAIRED_METRICS = ("auroc", "ap", "sensitivity", "specificity", "f1", "balanced_accuracy")


def brier(y_true, y_prob):
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def ece(y_true, y_prob, n_bins=5):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    confidence = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
    correct = (y_prob >= 0.5).astype(int) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # The last bin is closed on the right so that confidence == 1.0 is counted.
        mask = (confidence >= lo) & ((confidence <= hi) if i == n_bins - 1 else (confidence < hi))
        if mask.any():
            error += (mask.sum() / len(y_true)) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(error)


METRIC_FNS = {
    "auroc": lambda y, p, d: roc_auc_score(y, p),
    "ap": lambda y, p, d: average_precision_score(y, p),
    "balanced_accuracy": lambda y, p, d: balanced_accuracy_score(y, d),
    "f1": lambda y, p, d: f1_score(y, d, zero_division=0),
    "sensitivity": lambda y, p, d: d[y == 1].sum() / (y == 1).sum(),
    "specificity": lambda y, p, d: (1 - d[y == 0]).sum() / (y == 0).sum(),
    "brier": lambda y, p, d: brier(y, p),
    "ece": lambda y, p, d: ece(y, p),
}


def parse_config_name(config_name):
    aggregator, rest = next(((agg, config_name[len(prefix):]) for prefix, agg in AGGREGATOR_PREFIXES
                             if config_name.startswith(prefix)), ("clam_sb", config_name))
    extractor, strategy = rest.split("_", 1)
    return aggregator, extractor, strategy


def compute_metrics(y_true, y_prob, y_pred):
    y_true, y_prob, y_pred = np.asarray(y_true), np.asarray(y_prob), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {name: fn_(y_true, y_prob, y_pred) for name, fn_ in METRIC_FNS.items()}
    metrics.update(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
    return metrics


def stratified_resample(rng, y_true):
    # Resampling within each class keeps the class ratio fixed, so no replicate is single-class.
    neg_idx, pos_idx = np.flatnonzero(y_true == 0), np.flatnonzero(y_true == 1)
    return np.concatenate([rng.choice(neg_idx, size=len(neg_idx), replace=True),
                           rng.choice(pos_idx, size=len(pos_idx), replace=True)])


def bootstrap_samples(y_true, statistic, names, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in names}
    for _ in range(n_bootstrap):
        idx = stratified_resample(rng, y_true)
        for name in names:
            try:
                value = statistic(name, idx)
            except ValueError:
                continue
            if not np.isnan(value):
                samples[name].append(value)
    return samples


def percentile_ci(samples, confidence):
    if not samples:
        return float("nan"), float("nan")
    alpha = 1 - confidence
    return np.percentile(samples, 100 * alpha / 2), np.percentile(samples, 100 * (1 - alpha / 2))


def bootstrap_cis(y_true, y_prob, y_pred, names, n_bootstrap=1000, confidence=0.95, seed=42):
    y_true, y_prob, y_pred = np.asarray(y_true), np.asarray(y_prob), np.asarray(y_pred)
    samples = bootstrap_samples(
        y_true, lambda name, idx: METRIC_FNS[name](y_true[idx], y_prob[idx], y_pred[idx]),
        names, n_bootstrap, seed)
    return {name: percentile_ci(s, confidence) for name, s in samples.items()}


def paired_bootstrap_ci(df_a, df_b, metric_names=PAIRED_METRICS, n_bootstrap=1000,
                        confidence=0.95, seed=42):
    a = df_a.set_index("patient_id").sort_index()
    b = df_b.set_index("patient_id").sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    y_true = a["true_label"].values
    if not np.array_equal(y_true, b["true_label"].values):
        raise ValueError("true labels differ between the two result sets")

    def delta(name, idx):
        fn_ = METRIC_FNS[name]
        return (fn_(y_true[idx], a["prob_malignant"].values[idx], a["predicted_label"].values[idx])
                - fn_(y_true[idx], b["prob_malignant"].values[idx], b["predicted_label"].values[idx]))

    everyone = np.arange(len(y_true))
    samples = bootstrap_samples(y_true, delta, metric_names, n_bootstrap, seed)
    out = {}
    for name, s in samples.items():
        lo, hi = percentile_ci(s, confidence)
        out[name] = {"delta": delta(name, everyone), "ci_lower": float(lo), "ci_upper": float(hi), "n": len(s)}
    return out


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # roc_curve sorts thresholds in decreasing order, so ties in J resolve to the highest threshold.
    return float(thresholds[np.argmax(tpr - fpr)])


def loo_youden_predictions(df_folds):
    df = df_folds.reset_index(drop=True).copy()
    df["youden_threshold"] = [youden_threshold(df["true_label"].drop(i).values,
                                               df["prob_malignant"].drop(i).values)
                              for i in range(len(df))]
    df["predicted_label_youden"] = (df["prob_malignant"] >= df["youden_threshold"]).astype(int)
    return df


def evaluate_config(df, n_bootstrap=1000, confidence=0.95):
    y_true = df["true_label"].values
    y_prob = df["prob_malignant"].values
    metrics = compute_metrics(y_true, y_prob, df["predicted_label"].values)
    for name, (lo, hi) in bootstrap_cis(y_true, y_prob, df["predicted_label"].values, METRIC_FNS,
                                        n_bootstrap, confidence).items():
        metrics[f"{name}_ci_lower"], metrics[f"{name}_ci_upper"] = lo, hi

    df_youden = loo_youden_predictions(df)
    y_pred_youden = df_youden["predicted_label_youden"].values
    youden = compute_metrics(y_true, y_prob, y_pred_youden)
    metrics["youden_threshold_median"] = float(np.median(df_youden["youden_threshold"].values))
    for key in (*YOUDEN_METRICS, "tp", "tn", "fp", "fn"):
        metrics[f"{key}_youden"] = youden[key]
    # Each patient's LOO decision is held fixed, so these CIs are conditional on the threshold rule.
    for name, (lo, hi) in bootstrap_cis(y_true, y_prob, y_pred_youden, YOUDEN_METRICS,
                                        n_bootstrap, confidence).items():
        metrics[f"{name}_youden_ci_lower"], metrics[f"{name}_youden_ci_upper"] = lo, hi
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Metrics and bootstrap CIs for LOPOCV results")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    results_dir = os.path.join(cfg["data"]["output_dir"], "results")
    csv_files = sorted(f for f in os.listdir(results_dir) if f.endswith("_folds.csv"))
    if not csv_files:
        print("No fold results found. Run training first.")
        return

    rows = []
    for csv_file in csv_files:
        config_name = csv_file.removesuffix("_folds.csv")
        aggregator, extractor, strategy = parse_config_name(config_name)
        m = evaluate_config(pd.read_csv(os.path.join(results_dir, csv_file)),
                            eval_cfg["n_bootstrap"], eval_cfg["confidence_level"])
        rows.append({"config": config_name, "aggregator": aggregator,
                     "extractor": extractor, "strategy": strategy, **m})
        print(f"  {config_name:<40s} AUROC {m['auroc']:.3f} [{m['auroc_ci_lower']:.3f}, "
              f"{m['auroc_ci_upper']:.3f}]  AP {m['ap']:.3f}  "
              f"BalAcc@0.5 {m['balanced_accuracy']:.3f}  BalAcc@Youden {m['balanced_accuracy_youden']:.3f}")

    summary_path = os.path.join(results_dir, "summary_metrics.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    best = max(rows, key=lambda r: r["auroc"])
    print(f"\n{len(rows)} configurations -> {summary_path}")
    print(f"Best by AUROC: {best['config']} ({best['auroc']:.3f})")


if __name__ == "__main__":
    main()
