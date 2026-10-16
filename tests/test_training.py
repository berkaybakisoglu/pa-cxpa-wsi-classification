import os

import pandas as pd
import pytest
import torch

import src.evaluate as evaluate
import src.patch_baseline as patch_baseline
import src.train as train
from src.models.aggregators import AGGREGATORS
from src.preprocess import get_patient_list


def bags(labels, dim=4):
    return [torch.full((3 + i, dim), float(label)) for i, label in enumerate(labels)]


def test_oversample_balances_with_minority_copies():
    labels = [0, 0, 0, 0, 1]
    features, new_labels = train.apply_oversample(bags(labels), labels)
    assert new_labels.count(0) == new_labels.count(1) == 4
    assert all(torch.equal(f, features[4]) for f in features[5:])


def test_oversample_is_noop_when_balanced():
    labels = [0, 1]
    features = bags(labels)
    assert train.apply_oversample(features, labels) == (features, labels)


def test_smote_adds_single_vector_bags_flagged_synthetic():
    labels = [0, 0, 0, 0, 0, 1, 1, 1]
    features, new_labels, synthetic = train.apply_smote(bags(labels), labels, seed=0, k_neighbors=3)
    assert new_labels.count(0) == new_labels.count(1)
    assert synthetic == [False] * len(labels) + [True] * (len(features) - len(labels))
    assert all(f.shape == (1, 4) for f in features[len(labels):])


def test_smote_skips_when_minority_too_small():
    labels = [0, 0, 0, 1]
    features = bags(labels)
    assert train.apply_smote(features, labels, seed=0, k_neighbors=3) == (features, labels, [False] * 4)


def test_unknown_imbalance_strategy_raises(toy_project):
    with pytest.raises(ValueError):
        train.apply_imbalance("weighted", bags([0, 1]), [0, 1], 0, toy_project[0])


def test_lopocv_folds_hold_out_each_patient_once():
    patients = [(f"P{i}", i % 2) for i in range(5)]
    folds = list(train.lopocv_folds(patients))
    assert [test_id for _, test_id, *_ in folds] == [pid for pid, _ in patients]
    for fold, test_id, test_label, train_ids, train_labels in folds:
        assert test_id not in train_ids and len(train_ids) == len(patients) - 1
        assert dict(patients)[test_id] == test_label
        assert train_labels == [dict(patients)[pid] for pid in train_ids]


@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_train_writes_one_row_per_patient(toy_project, run_main, aggregator):
    cfg, cfg_path = toy_project
    run_main(train, "--config", cfg_path, "--aggregator", aggregator, "--strategy", "smote")
    name = "toy_smote" if aggregator == "clam_sb" else f"{aggregator}_toy_smote"
    results = os.path.join(cfg["data"]["output_dir"], "results")
    labels = sorted(label for _, label, _ in get_patient_list(cfg))
    n_patients = len(labels)
    folds = pd.read_csv(os.path.join(results, f"{name}_folds.csv"))
    assert list(folds.columns) == train.FOLD_COLUMNS
    assert sorted(folds["true_label"]) == labels
    assert folds["prob_malignant"].between(0, 1).all()
    assert len(pd.read_csv(os.path.join(results, f"{name}_curves.csv"))) == n_patients * 2
    assert len(torch.load(os.path.join(results, f"{name}_attention.pt"), weights_only=False)) == n_patients
    assert not os.path.exists(os.path.join(results, f"{name}_partial.pt"))


def test_training_is_deterministic(toy_project):
    cfg, _ = toy_project
    patients = [("B0", 0), ("B1", 0), ("B2", 0), ("M0", 1), ("M1", 1)]

    def run():
        train.set_seed(0)
        return train.train_one_fold("toy", "clam_sb", "none", cfg, "cpu", "B0", 0,
                                    [p for p, _ in patients[1:]], [l for _, l in patients[1:]], 0)

    first, second = run(), run()
    assert first["train_curve"] == second["train_curve"]
    assert first["prob_malignant"] == second["prob_malignant"]


def test_run_lopocv_resumes_from_partial(toy_project):
    cfg, _ = toy_project
    patients = [("B0", 0), ("B1", 0), ("M0", 1)]
    calls = []

    def fake_fold(test_id, test_label, train_ids, train_labels, seed):
        calls.append((test_id, seed))
        return {"patient_id": test_id, "true_label": test_label, "predicted_label": 0,
                "prob_malignant": 0.1, "prob_benign": 0.9, "epochs_trained": 1,
                "final_train_loss": 0.0, "train_curve": [0.0]}

    partial = os.path.join(train.results_dir(cfg), "demo_partial.pt")
    torch.save([dict(fake_fold("B0", 0, [], [], 0), fold=0)], partial)
    calls.clear()
    results = train.run_lopocv("demo", fake_fold, patients, cfg)
    assert calls == [("B1", cfg["seed"] + 1), ("M0", cfg["seed"] + 2)]
    assert [r["fold"] for r in results] == [0, 1, 2]
    assert not os.path.exists(partial)


def test_existing_results_are_skipped(toy_project, run_main):
    cfg, cfg_path = toy_project
    folds_csv = os.path.join(train.results_dir(cfg), "toy_none_folds.csv")
    pd.DataFrame(columns=train.FOLD_COLUMNS).to_csv(folds_csv, index=False)
    run_main(train, "--config", cfg_path, "--strategy", "none")
    assert pd.read_csv(folds_csv).empty


@pytest.mark.parametrize("strategy", patch_baseline.STRATEGIES)
def test_patch_baseline_then_evaluate(toy_project, run_main, strategy):
    cfg, cfg_path = toy_project
    run_main(patch_baseline, "--config", cfg_path, "--strategy", strategy)
    results = os.path.join(cfg["data"]["output_dir"], "results")
    folds = pd.read_csv(os.path.join(results, f"patch_toy_{strategy}_folds.csv"))
    assert len(folds) == len(get_patient_list(cfg))
    assert ((folds["prob_malignant"] >= 0.5).astype(int) == folds["predicted_label"]).all()

    run_main(evaluate, "--config", cfg_path)
    summary = pd.read_csv(os.path.join(results, "summary_metrics.csv"))
    assert summary[["config", "aggregator", "extractor", "strategy"]].values.tolist() == \
        [[f"patch_toy_{strategy}", "patch_level", "toy", strategy]]
