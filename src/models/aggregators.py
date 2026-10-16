import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.clam import CLAM_SB


class MeanPoolClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.25, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features, label=None, return_attention=False):
        logits = self.classifier(features.mean(dim=0, keepdim=True))
        results = {
            "logits": logits,
            "instance_loss": torch.tensor(0.0, device=features.device),
        }
        if return_attention:
            N = features.shape[0]
            results["attention"] = torch.ones(N, device=features.device) / N
        return results


class ABMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, attention_dim=256,
                 dropout=0.25, n_classes=2):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features, label=None, return_attention=False):
        h = self.compress(features)
        A = F.softmax(self.attention(h), dim=0)
        logits = self.classifier((A * h).sum(dim=0, keepdim=True))
        results = {
            "logits": logits,
            "instance_loss": torch.tensor(0.0, device=features.device),
        }
        if return_attention:
            results["attention"] = A.squeeze(1).detach()
        return results


class PatchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.25, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


AGGREGATORS = ("mean_pool", "abmil", "clam_sb")


def build_aggregator(name, input_dim, model_cfg):
    if name == "mean_pool":
        return MeanPoolClassifier(input_dim, model_cfg["hidden_dim"], model_cfg["dropout"])
    if name == "abmil":
        return ABMIL(input_dim, model_cfg["hidden_dim"], model_cfg["attention_dim"],
                     model_cfg["dropout"])
    if name == "clam_sb":
        return CLAM_SB(input_dim, model_cfg["hidden_dim"], model_cfg["attention_dim"],
                       model_cfg["dropout"], k_sample=model_cfg["k_sample"],
                       instance_loss_weight=model_cfg["instance_loss_weight"])
    raise ValueError(f"Unknown aggregator: {name}")
