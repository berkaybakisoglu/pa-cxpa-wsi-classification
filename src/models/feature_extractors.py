import os

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHT_FILES = {"ctranspath": "ctranspath.pth", "uni": "uni.bin"}
IGNORED_KEY_PARTS = ("attn_mask", "relative_position")


def default_weights_path(cfg, name):
    if name not in WEIGHT_FILES:
        return None
    return os.path.join(cfg["data"]["pretrained_dir"], WEIGHT_FILES[name])


def _imagenet_transform(resize):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_state_dict(path, weights_only):
    state_dict = torch.load(path, map_location="cpu", weights_only=weights_only)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def _build_resnet50(weights_path=None):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    return model, _imagenet_transform(256), 2048


class _ConvStem(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, embed_dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


def _remap_timm05_downsample(state_dict):
    # timm 0.5.4 attaches downsample to the end of stage i; current timm to the start of stage i+1.
    remapped = {}
    for k, v in state_dict.items():
        for old_idx in (0, 1, 2):
            old_prefix = f"layers.{old_idx}.downsample."
            if k.startswith(old_prefix):
                k = k.replace(old_prefix, f"layers.{old_idx + 1}.downsample.", 1)
                break
        remapped[k] = v
    return remapped


def _report_keys(kind, keys):
    keys = [k for k in keys if not any(part in k for part in IGNORED_KEY_PARTS)]
    if keys:
        print(f"  CTransPath: {kind} keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")


def _build_ctranspath(weights_path=None):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
    model.patch_embed = _ConvStem(embed_dim=96)

    if weights_path is not None:
        state_dict = {k: v for k, v in _load_state_dict(weights_path, weights_only=True).items()
                      if not k.startswith(("head.", "classifier.")) and isinstance(v, torch.Tensor)}
        missing, unexpected = model.load_state_dict(_remap_timm05_downsample(state_dict), strict=False)
        _report_keys("missing", missing)
        _report_keys("unexpected", unexpected)
        print(f"  CTransPath: loaded weights from {weights_path}")
    else:
        print("  WARNING: CTransPath loaded without pretrained weights (random init)")

    return model, _imagenet_transform(224), 768


def _build_uni(weights_path=None):
    vit_kwargs = dict(num_classes=0, init_values=1e-5, dynamic_img_size=True)

    if weights_path is not None and os.path.exists(weights_path):
        model = timm.create_model("vit_large_patch16_224", pretrained=False, **vit_kwargs)
        missing, unexpected = model.load_state_dict(
            _load_state_dict(weights_path, weights_only=False), strict=False)
        if len(missing) > 5 or len(unexpected) > 5:
            raise RuntimeError(
                f"UNI local weights mismatch ({len(missing)} missing, "
                f"{len(unexpected)} unexpected): {weights_path}")
        print(f"  UNI: loaded from local weights: {weights_path}")
    else:
        try:
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **vit_kwargs)
            print("  UNI: loaded from HuggingFace hub")
        except Exception as e:
            raise RuntimeError(
                f"UNI weights unavailable: no local file at {weights_path} and HuggingFace "
                f"load failed ({e}). Refusing random init (features would be meaningless).") from e

    return model, _imagenet_transform(224), 1024


EXTRACTORS = {
    "resnet50": _build_resnet50,
    "ctranspath": _build_ctranspath,
    "uni": _build_uni,
}


def get_extractor(name, weights_path=None, device="cuda"):
    if name not in EXTRACTORS:
        raise ValueError(f"Unknown extractor: {name}. Choose from {list(EXTRACTORS)}")
    model, transform, feature_dim = EXTRACTORS[name](weights_path)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, transform, feature_dim
