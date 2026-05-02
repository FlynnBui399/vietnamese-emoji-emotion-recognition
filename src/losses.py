"""Loss factories. Phase 1: BCEWithLogits + per-class pos_weight."""
from __future__ import annotations

import torch
import torch.nn as nn


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
