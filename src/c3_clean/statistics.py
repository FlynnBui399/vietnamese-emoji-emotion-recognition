"""Paired comment-level bootstrap and multiplicity-corrected per-class tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _macro_f1_fast(targets: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = np.logical_and(targets == 1, predictions == 1).sum(axis=0)
    false_positive = np.logical_and(targets == 0, predictions == 1).sum(axis=0)
    false_negative = np.logical_and(targets == 1, predictions == 0).sum(axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    per_class = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    return float(per_class.mean())


def paired_bootstrap_macro_f1(
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    c3_predictions: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resample test comments as paired units; labels are never sampled alone."""
    if not (targets.shape == baseline_predictions.shape == c3_predictions.shape):
        raise ValueError("Targets and both prediction matrices must have identical shapes")
    generator = np.random.default_rng(seed)
    sample_count = targets.shape[0]
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        indices = generator.integers(0, sample_count, size=sample_count)
        sampled_targets = targets[indices]
        baseline_score = _macro_f1_fast(sampled_targets, baseline_predictions[indices])
        c3_score = _macro_f1_fast(sampled_targets, c3_predictions[indices])
        rows.append(
            {
                "iteration": iteration,
                "baseline_macro_f1": baseline_score,
                "c3_macro_f1": c3_score,
                "delta_c3_minus_baseline": c3_score - baseline_score,
            }
        )
    distribution = pd.DataFrame(rows)
    ci_rows = []
    for system, column in (
        ("baseline", "baseline_macro_f1"),
        ("C3 Ensemble", "c3_macro_f1"),
    ):
        values = distribution[column].to_numpy()
        ci_rows.append(
            {
                "system": system,
                "metric": "macro_f1",
                "point_estimate": _macro_f1_fast(
                    targets,
                    baseline_predictions if system == "baseline" else c3_predictions,
                ),
                "ci_lower_95": float(np.percentile(values, 2.5)),
                "ci_upper_95": float(np.percentile(values, 97.5)),
                "iterations": iterations,
                "seed": seed,
                "resampling_unit": "test_comment",
            }
        )
    delta = distribution["delta_c3_minus_baseline"].to_numpy()
    probability_le_zero = float((np.count_nonzero(delta <= 0) + 1) / (iterations + 1))
    probability_ge_zero = float((np.count_nonzero(delta >= 0) + 1) / (iterations + 1))
    comparison = pd.DataFrame(
        [
            {
                "comparison": "C3 Ensemble minus baseline",
                "point_delta": _macro_f1_fast(targets, c3_predictions)
                - _macro_f1_fast(targets, baseline_predictions),
                "ci_lower_95": float(np.percentile(delta, 2.5)),
                "ci_upper_95": float(np.percentile(delta, 97.5)),
                "two_sided_bootstrap_p_value": min(
                    1.0, 2.0 * min(probability_le_zero, probability_ge_zero)
                ),
                "probability_delta_less_than_or_equal_to_zero": probability_le_zero,
                "iterations": iterations,
                "seed": seed,
                "resampling_unit": "test_comment",
            }
        ]
    )
    return pd.DataFrame(ci_rows), distribution, comparison


def holm_bonferroni(raw_p_values: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_p_values, dtype=np.float64)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running_max = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * raw[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def per_class_mcnemar_tests(
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    c3_predictions: np.ndarray,
    label_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (targets.shape == baseline_predictions.shape == c3_predictions.shape):
        raise ValueError("Targets and both prediction matrices must have identical shapes")
    rows: list[dict[str, Any]] = []
    for label_id, label_name in enumerate(label_names):
        baseline_correct = baseline_predictions[:, label_id] == targets[:, label_id]
        c3_correct = c3_predictions[:, label_id] == targets[:, label_id]
        baseline_only = int(np.logical_and(baseline_correct, ~c3_correct).sum())
        c3_only = int(np.logical_and(~baseline_correct, c3_correct).sum())
        discordant = baseline_only + c3_only
        p_value = (
            float(binomtest(c3_only, discordant, p=0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "label_id": label_id,
                "label": label_name,
                "baseline_correct_c3_incorrect": baseline_only,
                "baseline_incorrect_c3_correct": c3_only,
                "discordant": discordant,
                "raw_p_value": p_value,
                "test": "exact_mcnemar_binomial",
            }
        )
    raw = pd.DataFrame(rows)
    adjusted = raw[["label_id", "label", "raw_p_value"]].copy()
    adjusted["holm_adjusted_p_value"] = holm_bonferroni(
        adjusted["raw_p_value"].to_numpy()
    )
    adjusted["reject_at_alpha_0_05"] = adjusted["holm_adjusted_p_value"] <= 0.05
    return raw, adjusted


def write_statistical_artifacts(
    output_dir: str | Path,
    *,
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    c3_predictions: np.ndarray,
    label_names: list[str],
    iterations: int = 10_000,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ci, distribution, paired = paired_bootstrap_macro_f1(
        targets,
        baseline_predictions,
        c3_predictions,
        iterations=iterations,
        seed=seed,
    )
    per_class, corrected = per_class_mcnemar_tests(
        targets, baseline_predictions, c3_predictions, label_names
    )
    ci.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    distribution.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    per_class.to_csv(output_dir / "per_class_tests.csv", index=False)
    corrected.to_csv(output_dir / "holm_corrected_pvalues.csv", index=False)
    paired.to_csv(output_dir / "paired_bootstrap_summary.csv", index=False)


__all__ = [
    "holm_bonferroni",
    "paired_bootstrap_macro_f1",
    "per_class_mcnemar_tests",
    "write_statistical_artifacts",
]
