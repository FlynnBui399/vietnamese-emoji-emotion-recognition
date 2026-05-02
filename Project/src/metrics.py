"""Multi-label evaluation metrics for ViGoEmotions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score, hamming_loss


@dataclass
class EvalMetrics:
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    hamming: float
    per_class_f1: list[float]
    threshold: float
    macro_f1_tuned: float = 0.0
    threshold_tuned: float = 0.5
    per_class_f1_tuned: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "micro_f1": self.micro_f1,
            "hamming": self.hamming,
            "threshold": self.threshold,
            "macro_f1_tuned": self.macro_f1_tuned,
            "threshold_tuned": self.threshold_tuned,
            "per_class_f1": list(map(float, self.per_class_f1)),
            "per_class_f1_tuned": list(map(float, self.per_class_f1_tuned)),
        }


def _binarize(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.int8)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    sweep_thresholds: bool = True,
) -> EvalMetrics:
    """Compute the metrics required by the ViGoEmotions paper plus a tuned threshold.

    `probs`: shape (N, C), sigmoid probabilities.
    `targets`: shape (N, C), {0, 1}.
    """
    if probs.shape != targets.shape:
        raise ValueError(f"probs {probs.shape} != targets {targets.shape}")

    preds = _binarize(probs, threshold)
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    weighted = f1_score(targets, preds, average="weighted", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    per_class = f1_score(targets, preds, average=None, zero_division=0).tolist()
    hl = hamming_loss(targets, preds)

    metrics = EvalMetrics(
        macro_f1=float(macro),
        weighted_f1=float(weighted),
        micro_f1=float(micro),
        hamming=float(hl),
        per_class_f1=per_class,
        threshold=float(threshold),
    )

    if sweep_thresholds:
        best_t, best_macro, best_per_class = tune_threshold(probs, targets)
        metrics.macro_f1_tuned = float(best_macro)
        metrics.threshold_tuned = float(best_t)
        metrics.per_class_f1_tuned = best_per_class

    return metrics


def tune_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float, list[float]]:
    """Sweep a global threshold to maximize Macro F1. Returns (best_t, best_macro, per_class)."""
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.05)
    best_t = 0.5
    best_macro = -1.0
    best_per_class: list[float] = []
    for t in grid:
        preds = _binarize(probs, float(t))
        macro = f1_score(targets, preds, average="macro", zero_division=0)
        if macro > best_macro:
            best_macro = float(macro)
            best_t = float(t)
            best_per_class = f1_score(
                targets, preds, average=None, zero_division=0
            ).tolist()
    return best_t, best_macro, best_per_class
