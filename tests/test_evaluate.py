import numpy as np
import pandas as pd
import pytest

from src.evaluate import (bootstrap_cis, compute_metrics, ece, evaluate_config, loo_youden_predictions,
                          paired_bootstrap_ci, parse_config_name, percentile_ci)

Y_TRUE = np.array([0] * 14 + [1] * 5)


def folds(prob, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Y_TRUE))
    return pd.DataFrame({"patient_id": [f"P{i:02d}" for i in order], "true_label": Y_TRUE[order],
                         "prob_malignant": prob[order], "predicted_label": (prob[order] >= 0.5).astype(int)})


def noisy_prob(seed=42):
    return np.clip(np.random.default_rng(seed).random(19) * 0.5 + Y_TRUE * 0.4, 0, 1)


def test_metrics_of_perfect_classifier():
    m = compute_metrics(Y_TRUE, Y_TRUE * 0.8 + 0.1, Y_TRUE)
    assert m["auroc"] == m["ap"] == m["balanced_accuracy"] == m["sensitivity"] == m["specificity"] == 1
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (5, 14, 0, 0)


def test_metrics_of_all_benign_classifier():
    m = compute_metrics(Y_TRUE, np.full(19, 0.3), np.zeros(19, dtype=int))
    assert (m["sensitivity"], m["specificity"], m["f1"]) == (0, 1, 0)


def test_ece_counts_full_confidence():
    assert ece([0, 1], [0.0, 1.0]) == 0
    assert ece([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate():
    prob = noisy_prob()
    pred = (prob >= 0.5).astype(int)
    point = compute_metrics(Y_TRUE, prob, pred)
    cis = bootstrap_cis(Y_TRUE, prob, pred, ["auroc", "sensitivity"], n_bootstrap=300)
    for name, (lo, hi) in cis.items():
        assert 0 <= lo < hi <= 1
        assert lo <= point[name] <= hi


def test_percentile_ci_of_no_samples_is_nan():
    assert all(np.isnan(percentile_ci([], 0.95)))


def test_summary_row_has_all_columns():
    m = evaluate_config(folds(noisy_prob()), n_bootstrap=100)
    assert len(m) == 45
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == m["tp_youden"] + m["tn_youden"] + m["fp_youden"] + m["fn_youden"] == 19


def test_loo_youden_threshold_ignores_own_patient():
    df = folds(noisy_prob())
    thresholds = loo_youden_predictions(df)["youden_threshold"]
    moved = df.copy()
    moved.loc[0, "prob_malignant"] = 1 - moved.loc[0, "prob_malignant"]
    assert loo_youden_predictions(moved)["youden_threshold"][0] == thresholds[0]


def test_paired_bootstrap_of_identical_models_is_zero():
    df = folds(noisy_prob())
    result = paired_bootstrap_ci(df, df.sample(frac=1, random_state=1), n_bootstrap=100)
    assert all(r["delta"] == r["ci_lower"] == r["ci_upper"] == 0 for r in result.values())


def test_paired_bootstrap_rejects_mismatched_labels():
    a = folds(noisy_prob())
    b = a.copy()
    b["true_label"] = 1 - b["true_label"]
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b, n_bootstrap=10)


@pytest.mark.parametrize("name, expected", [
    ("ctranspath_none", ("clam_sb", "ctranspath", "none")),
    ("mean_pool_uni_smote", ("mean_pool", "uni", "smote")),
    ("abmil_resnet50_stainnorm_oversample", ("abmil", "resnet50", "stainnorm_oversample")),
    ("patch_uni_stainnorm_none", ("patch_level", "uni", "stainnorm_none")),
])
def test_parse_config_name(name, expected):
    assert parse_config_name(name) == expected
