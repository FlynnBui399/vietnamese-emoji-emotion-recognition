"""Losses and effective-number positive-class weighting for C3."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def effective_number_class_weights(
    train_targets: np.ndarray,
    *,
    beta: float = 0.999,
) -> np.ndarray:
    if train_targets.ndim != 2:
        raise ValueError("train_targets must be a two-dimensional target matrix")
    counts = train_targets.sum(axis=0).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("Every class must have at least one positive training example")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights /= weights.mean()
    if not np.isfinite(weights).all():
        raise ValueError("Effective-number class weights are not finite")
    return weights.astype(np.float32)


class AsymmetricLoss(nn.Module):
    """Multi-label ASL with optional weights applied only to positive terms."""

    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 0.0,
        probability_clip: float = 0.05,
        positive_class_weights: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_negative = float(gamma_negative)
        self.gamma_positive = float(gamma_positive)
        self.probability_clip = float(probability_clip)
        self.eps = float(eps)
        if positive_class_weights is None:
            self.register_buffer("positive_class_weights", None)
        else:
            if positive_class_weights.ndim != 1:
                raise ValueError("positive_class_weights must be one-dimensional")
            self.register_buffer(
                "positive_class_weights", positive_class_weights.detach().float()
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"logits shape {logits.shape} != targets shape {targets.shape}")
        targets = targets.to(dtype=logits.dtype)
        probabilities = torch.sigmoid(logits)
        clipped_negative_probabilities = torch.clamp(
            probabilities - self.probability_clip, min=0.0
        )
        positive = -targets * torch.pow(
            1.0 - probabilities, self.gamma_positive
        ) * torch.log(probabilities.clamp_min(self.eps))
        if self.positive_class_weights is not None:
            positive = positive * self.positive_class_weights.view(1, -1)
        negative = -(1.0 - targets) * torch.pow(
            clipped_negative_probabilities, self.gamma_negative
        ) * torch.log((1.0 - clipped_negative_probabilities).clamp_min(self.eps))
        return (positive + negative).mean()


def build_loss(
    experiment_name: str,
    train_targets: np.ndarray,
    loss_config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if experiment_name == "A0_controlled_text_BCE":
        return nn.BCEWithLogitsLoss()
    use_class_balancing = experiment_name in {
        "A3_controlled_ASL_Emoji_CB",
        "Emoji-random-control",
        "Emoji-shuffle-control",
        "Emoji-zero-control",
        "C3-RDrop",
        "C3-extended-matched",
    }
    weights = None
    if use_class_balancing:
        array = effective_number_class_weights(
            train_targets, beta=float(loss_config["effective_number_beta"])
        )
        weights = torch.tensor(array, dtype=torch.float32, device=device)
    return AsymmetricLoss(
        gamma_negative=float(loss_config["gamma_negative"]),
        gamma_positive=float(loss_config["gamma_positive"]),
        probability_clip=float(loss_config["probability_clip"]),
        positive_class_weights=weights,
    ).to(device)


__all__ = ["AsymmetricLoss", "build_loss", "effective_number_class_weights"]
