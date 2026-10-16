import os

import pytest
import torch
from PIL import Image

from src.models.feature_extractors import (_remap_timm05_downsample, default_weights_path, get_extractor)

EXPECTED_DIMS = {"resnet50": 2048, "ctranspath": 768, "uni": 1024}
CFG = {"data": {"pretrained_dir": "pretrained"}}


def check_extractor(name, weights_path):
    model, transform, dim = get_extractor(name, weights_path=weights_path, device="cpu")
    assert dim == EXPECTED_DIMS[name]
    assert not any(p.requires_grad for p in model.parameters())
    image = transform(Image.new("RGB", (256, 256), (200, 120, 180)))
    assert image.shape == (3, 224, 224)
    with torch.no_grad():
        single = model(image.unsqueeze(0))
        batch = model(torch.stack([image, image]))
    assert single.shape == (1, dim) and torch.isfinite(single).all() and single.abs().sum() > 0
    assert torch.allclose(batch[0], single[0], atol=1e-4)


def test_ctranspath_architecture_without_weights():
    check_extractor("ctranspath", None)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["resnet50", "ctranspath", "uni"])
def test_pretrained_extractor(name):
    weights = default_weights_path(CFG, name)
    if weights and not os.path.exists(weights):
        pytest.skip(f"{weights} not available")
    check_extractor(name, weights)


def test_unknown_extractor_raises():
    with pytest.raises(ValueError):
        get_extractor("vit_huge", device="cpu")


def test_default_weights_paths():
    assert default_weights_path(CFG, "resnet50") is None
    assert default_weights_path(CFG, "uni") == os.path.join("pretrained", "uni.bin")


def test_timm05_downsample_keys_are_remapped():
    remapped = _remap_timm05_downsample({"layers.0.downsample.reduction.weight": 1, "head.weight": 2})
    assert remapped == {"layers.1.downsample.reduction.weight": 1, "head.weight": 2}
