"""Strict source-data audit for the canonical C3 experiments.

This module intentionally does not repair malformed labels. It writes an audit
and discrepancy report first, then raises ``AuditFailure`` if any invariant is
violated. The raw CSV files are never modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NUM_LABELS = 28
LABEL_NAMES = [
    "amusement", "excitement", "joy", "love", "desire", "optimism",
    "caring", "pride", "admiration", "gratitude", "relief", "approval",
    "realization", "surprise", "curiosity", "confusion", "fear",
    "nervousness", "remorse", "embarrassment", "disappointment", "sadness",
    "grief", "disgust", "anger", "annoyance", "disapproval", "neutral",
]
EXPECTED = {
    "train": {"rows": 16531, "positive_label_assignments": 31545},
    "validation": {"rows": 2066, "positive_label_assignments": 3958},
    "test": {"rows": 2067, "positive_label_assignments": 3942},
}
FILE_BY_SPLIT = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}


class AuditFailure(RuntimeError):
    """Raised after audit artifacts have been written for a fatal discrepancy."""


@dataclass(frozen=True)
class AuditedSplit:
    name: str
    path: Path
    frame: pd.DataFrame
    label_ids: tuple[tuple[int, ...], ...]
    targets: np.ndarray
    sha256: str
    id_order_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_order_hash(ids: list[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_label_cell(cell: object, *, row_index: int, split: str) -> tuple[int, ...]:
    """Parse one stringified label list using only ``ast.literal_eval``."""
    try:
        parsed = ast.literal_eval(str(cell))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{split} row {row_index}: invalid label literal {cell!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise TypeError(f"{split} row {row_index}: labels must be a list or tuple")
    values: list[int] = []
    for value in parsed:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{split} row {row_index}: label {value!r} is not an integer"
            )
        if not 0 <= value < NUM_LABELS:
            raise ValueError(
                f"{split} row {row_index}: label {value} is outside [0, 27]"
            )
        values.append(value)
    return tuple(values)


def labels_to_targets(labels: tuple[tuple[int, ...], ...]) -> np.ndarray:
    targets = np.zeros((len(labels), NUM_LABELS), dtype=np.int8)
    for row_index, row_labels in enumerate(labels):
        for label_id in row_labels:
            targets[row_index, label_id] = 1
    return targets


def load_split(path: str | Path, split: str) -> AuditedSplit:
    path = Path(path)
    frame = pd.read_csv(
        path,
        dtype={"id": str, "text": str, "labels": str},
        keep_default_na=False,
    )
    required = ["id", "text", "labels"]
    if frame.columns.tolist() != required:
        raise ValueError(f"{path}: expected columns {required}, got {frame.columns.tolist()}")
    label_ids = tuple(
        parse_label_cell(cell, row_index=i, split=split)
        for i, cell in enumerate(frame["labels"].tolist())
    )
    targets = labels_to_targets(label_ids)
    return AuditedSplit(
        name=split,
        path=path,
        frame=frame,
        label_ids=label_ids,
        targets=targets,
        sha256=_sha256(path),
        id_order_sha256=_id_order_hash(frame["id"].astype(str).tolist()),
    )


def load_all_splits(data_dir: str | Path) -> dict[str, AuditedSplit]:
    data_dir = Path(data_dir)
    return {
        split: load_split(data_dir / filename, split)
        for split, filename in FILE_BY_SPLIT.items()
    }


def _duplicate_label_rows(split: AuditedSplit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, values in enumerate(split.label_ids):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            rows.append(
                {
                    "row_index": row_index,
                    "id": str(split.frame.iloc[row_index]["id"]),
                    "labels": list(values),
                    "duplicate_label_ids": duplicates,
                }
            )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def audit_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    raise_on_failure: bool = True,
) -> tuple[dict[str, AuditedSplit], dict[str, Any]]:
    """Audit all raw splits, write reports, and optionally raise on failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []

    try:
        splits = load_all_splits(data_dir)
    except Exception as exc:
        payload = {
            "status": "failed",
            "fatal": True,
            "issues": [{"code": "load_failure", "message": str(exc)}],
        }
        _write_json(output_dir / "data_audit.json", payload)
        (output_dir / "data_discrepancies.md").write_text(
            "# Data discrepancies\n\nPipeline stopped while loading raw CSV files.\n\n"
            f"- {exc}\n",
            encoding="utf-8",
        )
        raise AuditFailure(str(exc)) from exc

    split_records: list[dict[str, Any]] = []
    duplicate_records: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, Any] = {}
    label_rows: list[dict[str, Any]] = []

    for split_name, split in splits.items():
        expected = EXPECTED[split_name]
        rows = len(split.frame)
        raw_assignments = int(sum(len(values) for values in split.label_ids))
        matrix_assignments = int(split.targets.sum())
        duplicates = _duplicate_label_rows(split)
        duplicate_records[split_name] = duplicates

        if rows != expected["rows"]:
            issues.append(
                {
                    "code": "row_count_mismatch",
                    "split": split_name,
                    "expected": expected["rows"],
                    "actual": rows,
                }
            )
        if raw_assignments != expected["positive_label_assignments"]:
            issues.append(
                {
                    "code": "raw_positive_assignment_mismatch",
                    "split": split_name,
                    "expected": expected["positive_label_assignments"],
                    "actual": raw_assignments,
                }
            )
        if matrix_assignments != expected["positive_label_assignments"]:
            issues.append(
                {
                    "code": "target_matrix_support_mismatch",
                    "split": split_name,
                    "expected": expected["positive_label_assignments"],
                    "actual": matrix_assignments,
                }
            )
        if duplicates:
            issues.append(
                {
                    "code": "duplicate_label_ids",
                    "split": split_name,
                    "count": len(duplicates),
                    "rows": duplicates,
                }
            )
        if split.frame["id"].duplicated().any():
            duplicate_ids = sorted(
                split.frame.loc[split.frame["id"].duplicated(keep=False), "id"]
                .astype(str)
                .unique()
                .tolist()
            )
            issues.append(
                {"code": "duplicate_example_ids", "split": split_name, "ids": duplicate_ids}
            )

        split_records.append(
            {
                "split": split_name,
                "comments": rows,
                "expected_comments": expected["rows"],
                "raw_label_entries": raw_assignments,
                "target_matrix_positive_labels": matrix_assignments,
                "expected_positive_labels": expected["positive_label_assignments"],
                "labels_per_comment": matrix_assignments / rows,
                "duplicate_label_rows": len(duplicates),
                "target_shape": f"{split.targets.shape[0]}x{split.targets.shape[1]}",
            }
        )
        hashes[split_name] = {
            "path": split.path.as_posix(),
            "sha256": split.sha256,
            "bytes": split.path.stat().st_size,
            "id_order_sha256": split.id_order_sha256,
        }

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(
            set(splits[left].frame["id"].astype(str))
            & set(splits[right].frame["id"].astype(str))
        )
        if overlap:
            issues.append(
                {
                    "code": "cross_split_id_overlap",
                    "splits": [left, right],
                    "count": len(overlap),
                    "ids": overlap,
                }
            )

    test = splits["test"]
    per_class_support = test.targets.sum(axis=0).astype(int)
    try:
        assert len(test.frame) == 2067
        assert test.targets.shape == (2067, 28)
        assert int(test.targets.sum()) == 3942
        assert int(per_class_support.sum()) == 3942
    except AssertionError as exc:
        issues.append(
            {
                "code": "mandatory_test_assertion_failed",
                "message": str(exc) or "One or more mandatory test assertions failed",
            }
        )

    total_comments = sum(len(split.frame) for split in splits.values())
    total_matrix_labels = sum(int(split.targets.sum()) for split in splits.values())
    split_records.append(
        {
            "split": "all",
            "comments": total_comments,
            "expected_comments": 20664,
            "raw_label_entries": sum(
                sum(len(values) for values in split.label_ids) for split in splits.values()
            ),
            "target_matrix_positive_labels": total_matrix_labels,
            "expected_positive_labels": 39445,
            "labels_per_comment": total_matrix_labels / total_comments,
            "duplicate_label_rows": sum(len(rows) for rows in duplicate_records.values()),
            "target_shape": "not_applicable",
        }
    )

    for label_id, label_name in enumerate(LABEL_NAMES):
        counts = {
            split_name: int(split.targets[:, label_id].sum())
            for split_name, split in splits.items()
        }
        label_rows.append(
            {
                "label_id": label_id,
                "label": label_name,
                "train": counts["train"],
                "validation": counts["validation"],
                "test": counts["test"],
                "all": sum(counts.values()),
            }
        )

    audit = {
        "status": "failed" if issues else "passed",
        "fatal": bool(issues),
        "num_labels": NUM_LABELS,
        "source_precedence": "raw_csv_and_parsed_target_matrices",
        "splits": {row["split"]: row for row in split_records if row["split"] != "all"},
        "totals": next(row for row in split_records if row["split"] == "all"),
        "test_per_class_support": per_class_support.tolist(),
        "test_per_class_support_sum": int(per_class_support.sum()),
        "duplicate_label_rows": duplicate_records,
        "issues": issues,
        "raw_files_modified": False,
    }
    _write_json(output_dir / "data_audit.json", audit)
    _write_json(output_dir / "data_hashes.json", hashes)
    pd.DataFrame(split_records).to_csv(output_dir / "split_statistics.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_dir / "label_counts.csv", index=False)

    discrepancy_lines = [
        "# Data discrepancies",
        "",
        f"Status: **{'FATAL - pipeline stopped' if issues else 'PASS'}**",
        "",
        "The audit never edits or silently deduplicates a raw label list.",
        "",
    ]
    if issues:
        discrepancy_lines.extend(["## Findings", ""])
        for issue in issues:
            code = issue["code"]
            if code == "duplicate_label_ids":
                for row in issue["rows"]:
                    discrepancy_lines.append(
                        f"- `{issue['split']}` row {row['row_index']} (ID `{row['id']}`) "
                        f"contains labels `{row['labels']}`; duplicated IDs: "
                        f"`{row['duplicate_label_ids']}`."
                    )
            else:
                discrepancy_lines.append(f"- `{code}`: `{json.dumps(issue, ensure_ascii=False)}`")
        discrepancy_lines.extend(
            [
                "",
                "## Consequence",
                "",
                "P0 metric reconstruction, threshold fitting, P1-P3 training, and packaging "
                "are blocked until the source CSV invariants pass. The parsed multi-hot test "
                "matrix may still sum to 3,942 because repeated indices collapse to one cell; "
                "that does not make the raw duplicate valid.",
            ]
        )
    else:
        discrepancy_lines.append("No discrepancies were found.")
    (output_dir / "data_discrepancies.md").write_text(
        "\n".join(discrepancy_lines) + "\n", encoding="utf-8"
    )

    if issues and raise_on_failure:
        raise AuditFailure(
            f"Data audit failed with {len(issues)} issue(s); see "
            f"{output_dir / 'data_discrepancies.md'}"
        )
    return splits, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-raise", action="store_true")
    args = parser.parse_args(argv)
    try:
        _, audit = audit_dataset(
            args.data_dir, args.output_dir, raise_on_failure=not args.no_raise
        )
    except AuditFailure as exc:
        print(f"FATAL: {exc}")
        return 2
    if audit["fatal"]:
        print(
            f"FATAL: data audit recorded {len(audit['issues'])} issue(s); "
            f"see {args.output_dir / 'data_discrepancies.md'}"
        )
        return 2
    print("Data audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
