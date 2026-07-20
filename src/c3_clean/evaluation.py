"""Leakage-safe threshold fitting, exact metrics, and qualitative exports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .data_audit import LABEL_NAMES, NUM_LABELS


def validate_targets(targets: np.ndarray, *, expected_rows: int | None = None) -> None:
    if targets.ndim != 2 or targets.shape[1] != NUM_LABELS:
        raise ValueError(f"Expected target shape (N, {NUM_LABELS}), got {targets.shape}")
    if expected_rows is not None and targets.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} target rows, got {targets.shape[0]}")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Targets must be binary")


def validate_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    expected_rows: int | None = None,
) -> None:
    validate_targets(targets, expected_rows=expected_rows)
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"Probability shape {probabilities.shape} != target shape {targets.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinity")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must be in [0, 1]")


def fit_per_class_thresholds(
    validation_probabilities: np.ndarray,
    validation_targets: np.ndarray,
    *,
    source_split: str,
    minimum_validation_positives: int = 10,
) -> np.ndarray:
    """Fit one threshold per class; refuses every source other than validation."""
    if source_split != "validation":
        raise ValueError(
            "Threshold optimization is restricted to source_split='validation'; "
            f"received {source_split!r}"
        )
    validate_probabilities(validation_probabilities, validation_targets)
    grid = np.arange(0.05, 0.95, 0.01)
    thresholds = np.full(NUM_LABELS, 0.5, dtype=np.float64)
    for label_id in range(NUM_LABELS):
        if int(validation_targets[:, label_id].sum()) < minimum_validation_positives:
            continue
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in grid:
            predictions = validation_probabilities[:, label_id] >= threshold
            score = f1_score(
                validation_targets[:, label_id], predictions, zero_division=0
            )
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[label_id] = np.clip(best_threshold, 0.15, 0.85)
    return thresholds


def assert_threshold_leakage_guard() -> None:
    probabilities = np.full((2, NUM_LABELS), 0.5, dtype=np.float32)
    targets = np.zeros((2, NUM_LABELS), dtype=np.int8)
    try:
        fit_per_class_thresholds(
            probabilities, targets, source_split="test"
        )
    except ValueError:
        return
    raise AssertionError("Threshold optimizer accepted test targets")


def binarize(probabilities: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    if np.isscalar(thresholds):
        threshold_array = np.full(NUM_LABELS, float(thresholds), dtype=np.float64)
    else:
        threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.shape != (NUM_LABELS,):
        raise ValueError(f"Threshold shape {threshold_array.shape} != ({NUM_LABELS},)")
    return (probabilities >= threshold_array.reshape(1, -1)).astype(np.int8)


def exact_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: float | np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    validate_probabilities(probabilities, targets)
    predictions = binarize(probabilities, thresholds)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    support = support.astype(int)
    metrics = {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "subset_accuracy_exact_match": float(accuracy_score(targets, predictions)),
        "macro_average_precision": float(
            average_precision_score(targets, probabilities, average="macro")
        ),
        "micro_average_precision": float(
            average_precision_score(targets, probabilities, average="micro")
        ),
        "label_ranking_average_precision": float(
            label_ranking_average_precision_score(targets, probabilities)
        ),
        "positive_label_support": int(support.sum()),
        "comments": int(targets.shape[0]),
    }
    per_class = pd.DataFrame(
        {
            "label_id": np.arange(NUM_LABELS),
            "label": LABEL_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": per_f1,
            "support": support,
        }
    )
    return metrics, per_class, predictions


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    stable_ids: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    require_test_support: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(stable_ids) != targets.shape[0]:
        raise ValueError("Stable ID count does not match target rows")
    fixed, fixed_pc, fixed_predictions = exact_metrics(targets, probabilities, 0.5)
    tuned, tuned_pc, tuned_predictions = exact_metrics(targets, probabilities, thresholds)
    if require_test_support:
        assert targets.shape == (2067, 28)
        assert int(targets.sum()) == 3942
        assert int(tuned_pc["support"].sum()) == 3942

    payload = {
        "fixed_threshold_0_5": fixed,
        "validation_tuned_per_class_thresholds": tuned,
        "thresholds": thresholds.astype(float).tolist(),
        "threshold_source": "validation_only",
    }
    _write_json(output_dir / "metrics_exact.json", payload)
    _write_json(
        output_dir / "thresholds.json",
        {LABEL_NAMES[i]: float(thresholds[i]) for i in range(NUM_LABELS)},
    )

    paper_rows: list[dict[str, Any]] = []
    for mode, values in (("fixed_0.5", fixed), ("validation_tuned", tuned)):
        for metric, value in values.items():
            paper_rows.append(
                {
                    "threshold_mode": mode,
                    "metric": metric,
                    "paper_value": round(value, 4) if isinstance(value, float) else value,
                }
            )
    pd.DataFrame(paper_rows).to_csv(output_dir / "metrics_paper.csv", index=False)

    per_class = fixed_pc.rename(
        columns={
            "precision": "fixed_precision",
            "recall": "fixed_recall",
            "f1": "fixed_f1",
        }
    ).merge(
        tuned_pc.rename(
            columns={
                "precision": "tuned_precision",
                "recall": "tuned_recall",
                "f1": "tuned_f1",
                "support": "tuned_support",
            }
        ),
        on=["label_id", "label"],
        validate="one_to_one",
    )
    if not np.array_equal(per_class["support"], per_class["tuned_support"]):
        raise AssertionError("Fixed and tuned per-class supports differ")
    per_class = per_class.drop(columns="tuned_support")
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    report = classification_report(
        targets,
        tuned_predictions,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    prediction_rows = []
    for row_index, stable_id in enumerate(stable_ids):
        prediction_rows.append(
            {
                "id": str(stable_id),
                "ground_truth_labels": json.dumps(
                    np.flatnonzero(targets[row_index]).astype(int).tolist()
                ),
                "fixed_0_5_labels": json.dumps(
                    np.flatnonzero(fixed_predictions[row_index]).astype(int).tolist()
                ),
                "validation_tuned_labels": json.dumps(
                    np.flatnonzero(tuned_predictions[row_index]).astype(int).tolist()
                ),
                "probabilities": json.dumps(probabilities[row_index].astype(float).tolist()),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    return payload


def subset_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    mask: np.ndarray,
    *,
    system: str,
    subset: str,
) -> list[dict[str, Any]]:
    if mask.dtype != bool or mask.shape != (targets.shape[0],):
        raise ValueError("Subset mask must be a one-dimensional boolean array")
    rows: list[dict[str, Any]] = []
    for threshold_mode, value in (("fixed_0.5", 0.5), ("validation_tuned", thresholds)):
        metrics, _, _ = exact_metrics(targets[mask], probabilities[mask], value)
        for metric, score in metrics.items():
            rows.append(
                {
                    "system": system,
                    "subset": subset,
                    "threshold_mode": threshold_mode,
                    "metric": metric,
                    "value": score,
                }
            )
    return rows


def recover_macro_f1_from_classification_report(
    report_csv: str | Path,
) -> float:
    """Recompute Macro-F1 from the 28 exact per-class F1 values in a report CSV."""
    frame = pd.read_csv(report_csv)
    if "f1-score" not in frame:
        raise ValueError(f"{report_csv} does not contain an f1-score column")
    class_rows = frame.iloc[:NUM_LABELS]
    if len(class_rows) != NUM_LABELS:
        raise ValueError("Classification report does not contain 28 class rows")
    return float(class_rows["f1-score"].astype(float).mean())


def write_qualitative_candidates(
    output_dir: str | Path,
    *,
    stable_ids: Sequence[str],
    original_texts: Sequence[str],
    targets: np.ndarray,
    baseline_probabilities: np.ndarray,
    baseline_thresholds: np.ndarray,
    c3_probabilities: np.ndarray,
    c3_thresholds: np.ndarray,
    emoji_sequences: Sequence[Sequence[str]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_predictions = binarize(baseline_probabilities, baseline_thresholds)
    c3_predictions = binarize(c3_probabilities, c3_thresholds)
    baseline_exact = np.all(baseline_predictions == targets, axis=1)
    c3_exact = np.all(c3_predictions == targets, axis=1)
    records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    confusion_group = {24, 25, 26}
    cognitive_group = {12, 15, 27}
    for i, stable_id in enumerate(stable_ids):
        truth = set(np.flatnonzero(targets[i]).astype(int).tolist())
        baseline = set(np.flatnonzero(baseline_predictions[i]).astype(int).tolist())
        c3 = set(np.flatnonzero(c3_predictions[i]).astype(int).tolist())
        categories: list[str] = []
        if not baseline_exact[i] and c3_exact[i]:
            categories.append("baseline_incorrect_c3_correct")
        if baseline_exact[i] and not c3_exact[i]:
            categories.append("baseline_correct_c3_incorrect")
        if emoji_sequences[i] and len(truth ^ c3) < len(truth ^ baseline):
            categories.append("c3_improves_emoji_sample")
        if (truth | baseline | c3) & confusion_group:
            categories.append("disapproval_anger_annoyance")
        if (truth | baseline | c3) & cognitive_group:
            categories.append("neutral_confusion_realization")
        record = {
                "candidate_sets": "|".join(categories),
                "id": str(stable_id),
                "original_text": str(original_texts[i]),
                "ground_truth_labels": json.dumps(sorted(truth)),
                "baseline_predictions": json.dumps(sorted(baseline)),
                "c3_predictions": json.dumps(sorted(c3)),
                "baseline_probabilities": json.dumps(
                    baseline_probabilities[i].astype(float).tolist()
                ),
                "c3_probabilities": json.dumps(c3_probabilities[i].astype(float).tolist()),
                "emoji_presence": bool(emoji_sequences[i]),
                "extracted_emoji_sequence": json.dumps(list(emoji_sequences[i]), ensure_ascii=False),
                "baseline_exact": bool(baseline_exact[i]),
                "c3_exact": bool(c3_exact[i]),
                "baseline_false_positives": json.dumps(sorted(baseline - truth)),
                "baseline_false_negatives": json.dumps(sorted(truth - baseline)),
                "c3_false_positives": json.dumps(sorted(c3 - truth)),
                "c3_false_negatives": json.dumps(sorted(truth - c3)),
                "privacy_review": "required_before_paper_inclusion",
            }
        all_records.append(record)
        if categories:
            records.append(record)
    private_columns = [
        "candidate_sets", "id", "original_text", "ground_truth_labels",
        "baseline_predictions", "c3_predictions", "baseline_probabilities",
        "c3_probabilities", "emoji_presence", "extracted_emoji_sequence",
        "baseline_exact", "c3_exact", "baseline_false_positives",
        "baseline_false_negatives", "c3_false_positives", "c3_false_negatives",
        "privacy_review",
    ]
    private = pd.DataFrame(records, columns=private_columns)
    private.to_csv(output_dir / "qualitative_candidates_private.csv", index=False)
    public_columns = [
        "candidate_sets",
        "id",
        "ground_truth_labels",
        "baseline_predictions",
        "c3_predictions",
        "emoji_presence",
        "baseline_exact",
        "c3_exact",
        "baseline_false_positives",
        "baseline_false_negatives",
        "c3_false_positives",
        "c3_false_negatives",
        "privacy_review",
    ]
    private.reindex(columns=public_columns).to_csv(
        output_dir / "qualitative_candidates_public.csv", index=False
    )
    all_private = pd.DataFrame(all_records, columns=private_columns)
    all_private.to_csv(output_dir / "per_sample_evaluation_private.csv", index=False)
    all_private.reindex(columns=public_columns).to_csv(
        output_dir / "per_sample_evaluation_public.csv", index=False
    )


__all__ = [
    "assert_threshold_leakage_guard",
    "binarize",
    "exact_metrics",
    "fit_per_class_thresholds",
    "recover_macro_f1_from_classification_report",
    "subset_metrics",
    "validate_probabilities",
    "validate_targets",
    "write_evaluation_artifacts",
    "write_qualitative_candidates",
]
