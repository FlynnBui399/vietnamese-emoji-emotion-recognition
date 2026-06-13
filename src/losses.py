"""Loss functions for ViGoEmotions multi-label classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Implements the formulation requested from Ridnik et al. (2021):
    positive loss uses p = sigmoid(logits), while negative loss uses
    p_m = max(p - clip, 0) to down-weight easy negatives more aggressively.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"logits shape {tuple(x.shape)} != targets shape {tuple(y.shape)}")

        y = y.type_as(x)
        p = torch.sigmoid(x)

        p_m = torch.clamp(p - self.clip, min=0.0)

        log_pos = torch.log(torch.clamp(p, min=self.eps))
        log_neg = torch.log(torch.clamp(1.0 - p_m, min=self.eps))
        pos_loss = -y * torch.pow(1.0 - p, self.gamma_pos) * log_pos
        neg_loss = -(1.0 - y) * torch.pow(p_m, self.gamma_neg) * log_neg
        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_bce_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Standard `BCEWithLogitsLoss`. Pass `pos_weight` (per-class) to handle imbalance.

    pos_weight should have shape `(num_labels,)`. The Module-form loss reduces
    over the batch with `mean`, matching the ViGoEmotions paper's setup.
    """
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    if pos_weight.dim() != 1:
        raise ValueError(f"pos_weight must be 1-D, got shape {tuple(pos_weight.shape)}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
