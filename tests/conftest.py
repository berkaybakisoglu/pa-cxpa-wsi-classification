import sys

import pytest
import torch
import yaml

import src.train as train

FEATURE_DIM = 16


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(train, "_feature_cache", {"extractor": None, "patients": {}})


@pytest.fixture
def toy_project(tmp_path):
    cfg = {
        "seed": 0,
        "data": {"benign_dir": str(tmp_path / "Bening"), "malign_dir": str(tmp_path / "Malign"),
                 "output_dir": str(tmp_path / "outputs"), "pretrained_dir": "pretrained"},
        "feature_extraction": {"extractors": ["toy"]},
        "model": {"hidden_dim": 8, "attention_dim": 4, "dropout": 0.0, "k_sample": 2,
                  "bag_weight": 0.7, "instance_loss_weight": 0.3},
        "training": {"lr": 1e-2, "weight_decay": 0.0, "epochs": 2},
        "imbalance": {"strategies": ["none", "oversample", "smote"], "smote_k_neighbors": 3},
        "evaluation": {"n_bootstrap": 50, "confidence_level": 0.95},
    }
    features_dir = tmp_path / "outputs" / "features" / "toy"
    features_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    for label, folder, n in ((0, "Bening", 4), (1, "Malign", 3)):
        for i in range(n):
            pid = f"{folder[0]}{i}"
            (tmp_path / folder / pid).mkdir(parents=True)
            n_patches = 10 + i
            torch.save({"features": torch.randn(n_patches, FEATURE_DIM, generator=generator) + label,
                        "coords": torch.zeros(n_patches, 2), "label": label, "patient_id": pid},
                       features_dir / f"{pid}.pt")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg, cfg_path


@pytest.fixture
def run_main(monkeypatch):
    def run(module, *args):
        monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
        module.main()
    return run

