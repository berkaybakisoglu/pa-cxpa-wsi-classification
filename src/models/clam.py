import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


class CLAM_SB(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, attention_dim=256,
                 dropout=0.25, n_classes=2, k_sample=8,
                 instance_loss_weight=0.3):
        super().__init__()
        self.k_sample = k_sample
        self.instance_loss_weight = instance_loss_weight
        self.compress = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden_dim, attention_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(n_classes)
        ])

    def forward(self, features, label=None, return_attention=False):
        h = self.compress(features)
        A = F.softmax(self.attention(h), dim=0)
        logits = self.classifier((A * h).sum(dim=0, keepdim=True))
        attention = A.squeeze(1)

        instance_loss = torch.tensor(0.0, device=features.device)
        if label is not None and self.training:
            instance_loss = self._instance_loss(h, attention, label)

        results = {"logits": logits, "instance_loss": instance_loss}
        if return_attention:
            results["attention"] = attention.detach()
        return results

    def _instance_loss(self, h, attention, label):
        # subtyping=False path of the original CLAM: only the true class's
        # instance classifier, top-k attended patches -> 1, bottom-k -> 0.
        k = min(self.k_sample, h.shape[0])
        top_idx = torch.topk(attention, k).indices
        bottom_idx = torch.topk(attention, k, largest=False).indices
        classifier = self.instance_classifiers[label]
        logits = torch.cat([classifier(h[top_idx]), classifier(h[bottom_idx])])
        targets = torch.cat([torch.ones(k, dtype=torch.long, device=h.device),
                             torch.zeros(k, dtype=torch.long, device=h.device)])
        return F.cross_entropy(logits, targets) * self.instance_loss_weight
