import pytest
import torch

from src.models.aggregators import AGGREGATORS, PatchMLP, build_aggregator
from src.models.clam import CLAM_SB

MODEL_CFG = {"hidden_dim": 32, "attention_dim": 16, "dropout": 0.25, "k_sample": 4, "instance_loss_weight": 0.3}


@pytest.mark.parametrize("aggregator", AGGREGATORS)
@pytest.mark.parametrize("n_patches", [1, 50])
def test_aggregator_output_contract(aggregator, n_patches):
    model = build_aggregator(aggregator, 64, MODEL_CFG).eval()
    with torch.no_grad():
        out = model(torch.randn(n_patches, 64), return_attention=True)
    assert out["logits"].shape == (1, 2)
    assert out["attention"].shape == (n_patches,)
    assert torch.isclose(out["attention"].sum(), torch.tensor(1.0))
    assert out["instance_loss"].item() == 0


def test_unknown_aggregator_raises():
    with pytest.raises(ValueError):
        build_aggregator("transmil", 64, MODEL_CFG)


def test_clam_instance_loss_only_in_training_with_label():
    model = build_aggregator("clam_sb", 64, MODEL_CFG)
    features = torch.randn(20, 64)
    assert model.train()(features, label=1)["instance_loss"].item() > 0
    assert model.train()(features, label=None)["instance_loss"].item() == 0
    assert model.eval()(features, label=1)["instance_loss"].item() == 0


def test_clam_handles_bag_smaller_than_k():
    model = CLAM_SB(64, 32, 16, k_sample=8).train()
    assert torch.isfinite(model(torch.randn(3, 64), label=0)["instance_loss"])


def test_clam_backward_reaches_shared_layers():
    model = build_aggregator("clam_sb", 64, MODEL_CFG).train()
    out = model(torch.randn(20, 64), label=1)
    (torch.nn.functional.cross_entropy(out["logits"], torch.tensor([1])) + out["instance_loss"]).backward()
    for module in (model.compress, model.attention, model.classifier, model.instance_classifiers[1]):
        assert all(p.grad.abs().sum() > 0 for name, p in module.named_parameters() if name.endswith("weight"))
    assert all(p.grad is None for p in model.instance_classifiers[0].parameters())


def test_patch_mlp_scores_every_patch():
    assert PatchMLP(64, 32)(torch.randn(10, 64)).shape == (10, 2)
