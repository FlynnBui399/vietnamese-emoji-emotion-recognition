"""Command-line orchestrator for P0-P3 C3 experiments and Kaggle packaging."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data_audit import (
    AuditFailure,
    AuditedSplit,
    LABEL_NAMES,
    audit_dataset,
    load_split,
)
from .evaluation import (
    assert_threshold_leakage_guard,
    binarize,
    fit_per_class_thresholds,
    recover_macro_f1_from_classification_report,
    subset_metrics,
    validate_probabilities,
    write_evaluation_artifacts,
    write_qualitative_candidates,
)
from .model import verify_canonical_state_dict
from .preprocessing import (
    ImmutablePreprocessor,
    emoji2vec_coverage,
    emoji_package_version,
    emoji_presence_statistics,
    extract_emoji_sequence,
    prepare_text_columns,
    token_length_statistics,
)
from .statistics import paired_bootstrap_macro_f1, write_statistical_artifacts
from .training import load_emoji2vec, train_one_seed

HISTORICAL = {
    "c3_tuned_macro_f1": 0.6329413412542984,
    "c3_fixed_0_5_macro_f1_approx": 0.6309,
    "baseline_macro_f1": 0.6140661767682877,
}
SEED_BUNDLE_ALIASES = {
    42: ("ASL_Emoji_CB__seed42", "ASL_Emoji_CB_seed42", "seed42"),
    1: ("ASL_Emoji_CB_ensemble_seed1", "ASL_Emoji_CB_seed1", "seed1"),
    7: ("ASL_Emoji_CB_ensemble_seed7", "ASL_Emoji_CB_seed7", "seed7"),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(repository_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_environment(repository_root: Path, dataset_hashes: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu_models = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        torch_version = None
        cuda_version = None
        gpu_models = []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch_version,
        "cuda": cuda_version,
        "transformers": _package_version("transformers"),
        "scikit_learn": _package_version("scikit-learn"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "emoji": _package_version("emoji"),
        "gensim": _package_version("gensim"),
        "gpu_models": gpu_models,
        "visible_gpu_count": len(gpu_models),
        "git_commit_sha": _git_commit(repository_root),
        "dataset_hashes": dataset_hashes,
    }


def _find_dataset_dir(
    repository_root: Path,
    candidates: Sequence[str],
    override: str | None = None,
) -> Path:
    local = repository_root / "data" / "vigoemotions"
    search: list[Path] = []
    if override:
        search.append(Path(override))
    search.extend(
        Path(candidate) if Path(candidate).is_absolute() else repository_root / candidate
        for candidate in candidates
    )
    search.append(local)
    if Path("/kaggle/input").is_dir():
        search.extend(path.parent for path in Path("/kaggle/input").rglob("train.csv"))
    for candidate in search:
        if all((candidate / name).is_file() for name in ("train.csv", "val.csv", "test.csv")):
            return candidate.resolve()
    raise FileNotFoundError("Could not locate a directory containing train.csv, val.csv, and test.csv")


def _find_file(repository_root: Path, configured: str, filename: str) -> Path:
    direct = Path(configured)
    if direct.is_file():
        return direct.resolve()
    local = repository_root / configured
    if local.is_file():
        return local.resolve()
    if Path("/kaggle/input").is_dir():
        matches = list(Path("/kaggle/input").rglob(filename))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"Could not locate {filename}")


def resolve_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    configured_root = config.get("paths", {}).get("repository_root")
    repository_root = (
        Path(configured_root).resolve() if configured_root else config_path.parent.parent
    )
    paths = dict(config["paths"])
    data_dir = _find_dataset_dir(
        repository_root,
        paths.get("data_candidates", []),
        override=paths.get("data_dir_override") or os.environ.get("C3_DATA_DIR"),
    )
    paths["data_dir"] = data_dir.as_posix()
    paths["docs_dir"] = (repository_root / "docs").resolve().as_posix()
    paths["emoji2vec"] = _find_file(
        repository_root, paths.get("emoji2vec", "data/emoji2vec.bin"), "emoji2vec.bin"
    ).as_posix()
    paths["artifact_roots"] = [
        str((repository_root / path).resolve())
        if not Path(path).is_absolute()
        else str(Path(path))
        for path in paths.get("artifact_roots", [])
    ]
    paths["extended_train_candidates"] = [
        str((repository_root / path).resolve())
        if not Path(path).is_absolute()
        else str(Path(path))
        for path in paths.get("extended_train_candidates", [])
    ]
    if Path("/kaggle/working").is_dir():
        output_dir = Path("/kaggle/working/c3_clean_artifacts")
    else:
        output_dir = repository_root / "outputs" / "c3_clean"
    paths["output_dir"] = output_dir.resolve().as_posix()
    config["paths"] = paths
    return config, repository_root, output_dir.resolve()


def _load_hashes(output_dir: Path) -> dict[str, Any]:
    with (output_dir / "data_hashes.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _inventory_legacy_artifacts(repository_root: Path, configured_roots: Iterable[str]) -> dict[str, Any]:
    roots = [repository_root, *(Path(root) for root in configured_roots)]
    if Path("/kaggle/input").is_dir():
        roots.append(Path("/kaggle/input"))
    names = {
        "ASL_Emoji_CB",
        "ASL_Emoji_CB__seed42",
        "ASL_Emoji_CB_ensemble_seed1",
        "ASL_Emoji_CB_ensemble_seed7",
        "ASL_Emoji_CB_3seed_ensemble",
        "final_ensemble",
    }
    found: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            for match in root.rglob(name):
                found.append(match.resolve().as_posix())
    return {"searched_roots": [path.as_posix() for path in roots], "matches": sorted(set(found))}


def write_configuration_audit(
    output_dir: Path,
    *,
    inventory: Mapping[str, Any],
    artifact_reconstruction: Mapping[str, Any] | None = None,
) -> None:
    lines = [
        "# C3 configuration audit",
        "",
        "Canonical paper name: **C3 Ensemble**  ",
        "Technical artifact name: `ASL_Emoji_CB_3seed_ensemble`",
        "",
        "## Evidence recovered from this repository",
        "",
        "- The executed legacy Kaggle notebook displays tuned test Macro-F1 "
        "`0.6329413412542984` and fixed-0.5 Macro-F1 rounded to `0.6309`.",
        "- The executed source references three bundles: `ASL_Emoji_CB__seed42`, "
        "`ASL_Emoji_CB_ensemble_seed1`, and `ASL_Emoji_CB_ensemble_seed7`.",
        "- The current notebook source defines a simple `EmojiAwareViSoBERT` with masked "
        "mean pooling, a 300-to-768 Emoji2Vec projection, concatenation, and a "
        "1536-to-768-to-28 fusion head.",
        "- The current source/output pair has drifted: the source no longer contains the "
        "`ASL_Emoji_CB` experiment definition that generated the displayed output.",
        "",
        "## Checkpoint verification",
        "",
    ]
    if artifact_reconstruction:
        lines.append(
            f"Artifact reconstruction status: `{artifact_reconstruction.get('status')}`."
        )
        for seed, details in artifact_reconstruction.get("seeds", {}).items():
            lines.append(
                f"- Seed {seed}: checkpoint architecture verification "
                f"`{details.get('checkpoint_verification', {}).get('passed', False)}`."
            )
    else:
        lines.append(
            "No local checkpoint or probability bundle was available, so checkpoint "
            "state-dict verification could not be performed."
        )
    lines.extend(
        [
            "",
            "## Unresolved assumptions",
            "",
            "Until the three historical checkpoints are attached, the model class is an "
            "evidence-based inference rather than an exactly recovered configuration. The "
            "clean rerun uses: ViSoBERT, max length 128, masked mean pooling, raw-text "
            "Emoji2Vec mean vectors, the simple dual branch, ASL (4, 0, 0.05), effective-"
            "number positive loss weights (beta 0.999), dropout 0.2, AdamW at 2e-5, one "
            "warmup epoch, at most 10 epochs, and seeds 42/1/7.",
            "",
            "R-Drop, label graphs, cross-attention, gloss embeddings, C4, and extended data "
            "are not part of the canonical C3 configuration unless checkpoint metadata "
            "later proves otherwise.",
            "",
            "## Artifact search inventory",
            "",
            f"Searched roots: `{json.dumps(inventory.get('searched_roots', []))}`",
            "",
            f"Matching paths: `{json.dumps(inventory.get('matches', []))}`",
        ]
    )
    (output_dir / "c3_configuration_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_annotation_audit(output_dir: Path, repository_root: Path) -> None:
    extended = repository_root / "data" / "vigoemotions_extended"
    filenames = sorted(path.name for path in extended.glob("*")) if extended.is_dir() else []
    text = f"""# Annotation agreement and extended-data provenance audit

## Annotator agreement

No independent annotator-level label file was found. The extended CSV files and the
`dataset_V1.xlsx` workbook expose only `id`, `text`, and final `labels` columns. A
consensus label cannot be used to recompute Cohen's Kappa.

Therefore, **Kappa = 0.67 is unverifiable from repository artifacts**. Do not
reverse-engineer agreement. Remove the numeric claim from the main paper unless the
original independent annotator decisions are recovered.

## Provenance and permissions

Repository files found under `data/vigoemotions_extended`: `{json.dumps(filenames)}`.

These files do not verify collection dates, source-platform terms, collection method,
annotator provenance, or redistribution permission for the extended examples. The
original ViGoEmotions README also states that the dataset must not be redistributed;
that notice does not establish permission for the separately extended material.

Status: provenance, collection dates, platform terms, and redistribution permissions
for the extended data are **not verifiable from repository files**.
"""
    (output_dir / "annotation_agreement_audit.md").write_text(text, encoding="utf-8")


def _find_seed_bundle(seed: int, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for alias in SEED_BUNDLE_ALIASES[seed]:
            direct = root / alias
            if (direct / "test_probs.npy").is_file():
                return direct
            for candidate in root.rglob(alias):
                if (candidate / "test_probs.npy").is_file():
                    return candidate
    return None


def _load_checkpoint_metadata(bundle: Path) -> tuple[dict[str, Any], Path | None]:
    candidates = [
        bundle / "best_checkpoint.pt",
        bundle / "best_ckpt.pt",
        bundle / "best.pt",
        *bundle.glob("*.pth"),
        *bundle.glob("*.pt"),
    ]
    checkpoint_path = next((path for path in candidates if path.is_file()), None)
    if checkpoint_path is None:
        return {"passed": False, "reason": "checkpoint_absent"}, None
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        state_dict = next(
            (
                payload[key]
                for key in ("model_state_dict", "state_dict", "model")
                if key in payload and isinstance(payload[key], Mapping)
            ),
            payload,
        )
    else:
        state_dict = payload.state_dict()
    verification = verify_canonical_state_dict(state_dict)
    if isinstance(payload, Mapping):
        verification["checkpoint_config"] = _json_safe(
            payload.get("config", payload.get("cfg", payload.get("configuration")))
        )
        verification["checkpoint_epoch"] = _json_safe(
            payload.get("epoch", payload.get("best_epoch"))
        )
        verification["checkpoint_model_class"] = _json_safe(
            payload.get("model_class", payload.get("model_cls"))
        )
    return verification, checkpoint_path


def _read_ids(bundle: Path, split: str) -> list[str] | None:
    for filename in (f"{split}_ids.json", f"{split}_ids.npy"):
        path = bundle / filename
        if path.suffix == ".json" and path.is_file():
            return list(map(str, json.loads(path.read_text(encoding="utf-8"))))
        if path.suffix == ".npy" and path.is_file():
            return list(map(str, np.load(path, allow_pickle=False).tolist()))
    return None


def reconstruct_historical_artifacts(
    *,
    audited_splits: Mapping[str, AuditedSplit],
    output_dir: Path,
    artifact_roots: Sequence[Path],
) -> dict[str, Any]:
    seed_data: dict[int, dict[str, Any]] = {}
    details: dict[str, Any] = {"status": "started", "seeds": {}}
    for seed in (42, 1, 7):
        bundle = _find_seed_bundle(seed, artifact_roots)
        if bundle is None:
            raise FileNotFoundError(
                f"Historical probability bundle for seed {seed} was not found"
            )
        arrays = {
            name: np.load(bundle / f"{name}.npy", allow_pickle=False)
            for name in ("val_probs", "val_targets", "test_probs", "test_targets")
        }
        validate_probabilities(arrays["val_probs"], arrays["val_targets"], expected_rows=2066)
        validate_probabilities(arrays["test_probs"], arrays["test_targets"], expected_rows=2067)
        if not np.array_equal(arrays["val_targets"], audited_splits["validation"].targets):
            raise AssertionError(f"Seed {seed} validation targets do not match raw CSV order")
        if not np.array_equal(arrays["test_targets"], audited_splits["test"].targets):
            raise AssertionError(f"Seed {seed} test targets do not match raw CSV order")
        assert int(arrays["test_targets"].sum()) == 3942
        expected_val_ids = audited_splits["validation"].frame["id"].astype(str).tolist()
        expected_test_ids = audited_splits["test"].frame["id"].astype(str).tolist()
        supplied_val_ids = _read_ids(bundle, "val")
        supplied_test_ids = _read_ids(bundle, "test")
        if supplied_val_ids is not None and supplied_val_ids != expected_val_ids:
            raise AssertionError(f"Seed {seed} validation IDs are not in canonical order")
        if supplied_test_ids is not None and supplied_test_ids != expected_test_ids:
            raise AssertionError(f"Seed {seed} test IDs are not in canonical order")
        verification, checkpoint_path = _load_checkpoint_metadata(bundle)
        destination = output_dir / f"seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            np.save(destination / f"{name}.npy", array)
        _write_json(destination / "val_ids.json", expected_val_ids)
        _write_json(destination / "test_ids.json", expected_test_ids)
        seed_thresholds = fit_per_class_thresholds(
            arrays["val_probs"], arrays["val_targets"], source_split="validation"
        )
        write_evaluation_artifacts(
            destination,
            stable_ids=expected_test_ids,
            targets=arrays["test_targets"],
            probabilities=arrays["test_probs"],
            thresholds=seed_thresholds,
            require_test_support=True,
        )
        config_payload = {
            "recovery": "historical_probability_bundle",
            "source_bundle": bundle.resolve().as_posix(),
            "ordered_id_evidence": (
                "explicit_ids_and_exact_target_match"
                if supplied_val_ids is not None and supplied_test_ids is not None
                else "exact_rowwise_target_match_to_raw_csv;_ids_not_stored_in_legacy_bundle"
            ),
            "checkpoint_architecture_verification": verification,
        }
        _write_json(destination / "config.json", config_payload)
        if checkpoint_path is not None:
            shutil.copy2(checkpoint_path, destination / "best_checkpoint.pt")
        for history_name in ("training_history.csv", "history.csv"):
            history_path = bundle / history_name
            if history_path.is_file():
                shutil.copy2(history_path, destination / "training_history.csv")
                break
        seed_data[seed] = arrays
        details["seeds"][str(seed)] = {
            "bundle": bundle.resolve().as_posix(),
            "checkpoint_verification": verification,
            "explicit_ids_present": supplied_val_ids is not None and supplied_test_ids is not None,
        }

    reference_val_targets = seed_data[42]["val_targets"]
    reference_test_targets = seed_data[42]["test_targets"]
    for seed in (1, 7):
        if not np.array_equal(seed_data[seed]["val_targets"], reference_val_targets):
            raise AssertionError("Seeds do not share identical ordered validation targets")
        if not np.array_equal(seed_data[seed]["test_targets"], reference_test_targets):
            raise AssertionError("Seeds do not share identical ordered test targets")

    ensemble_val = np.mean(
        [seed_data[seed]["val_probs"] for seed in (42, 1, 7)], axis=0
    )
    ensemble_test = np.mean(
        [seed_data[seed]["test_probs"] for seed in (42, 1, 7)], axis=0
    )
    thresholds = fit_per_class_thresholds(
        ensemble_val, reference_val_targets, source_split="validation"
    )
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    np.save(ensemble_dir / "val_probs.npy", ensemble_val)
    np.save(ensemble_dir / "val_targets.npy", reference_val_targets)
    np.save(ensemble_dir / "test_probs.npy", ensemble_test)
    np.save(ensemble_dir / "test_targets.npy", reference_test_targets)
    metrics = write_evaluation_artifacts(
        ensemble_dir,
        stable_ids=audited_splits["test"].frame["id"].astype(str).tolist(),
        targets=reference_test_targets,
        probabilities=ensemble_test,
        thresholds=thresholds,
        require_test_support=True,
    )
    tuned = metrics["validation_tuned_per_class_thresholds"]["macro_f1"]
    fixed = metrics["fixed_threshold_0_5"]["macro_f1"]
    details.update(
        {
            "status": "recomputed",
            "artifact_recomputed_tuned_macro_f1": tuned,
            "artifact_recomputed_fixed_0_5_macro_f1": fixed,
            "historical_displayed_tuned_macro_f1": HISTORICAL["c3_tuned_macro_f1"],
            "historical_tuned_exact_match": bool(
                np.isclose(tuned, HISTORICAL["c3_tuned_macro_f1"], rtol=0.0, atol=1e-15)
            ),
            "threshold_source": "averaged_validation_probabilities",
        }
    )
    _write_json(output_dir / "artifact_reconstruction.json", details)
    return details


def run_preprocessing_audit(
    config: Mapping[str, Any],
    audited_splits: Mapping[str, AuditedSplit],
    output_dir: Path,
) -> None:
    from transformers import AutoTokenizer

    actual_emoji_version = emoji_package_version()
    expected_emoji_version = str(config["preprocessing"]["emoji_package_version"])
    if actual_emoji_version != expected_emoji_version:
        raise RuntimeError(
            f"emoji package version {actual_emoji_version} != pinned {expected_emoji_version}"
        )
    preprocessor = ImmutablePreprocessor.from_docs(config["paths"]["docs_dir"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"], use_fast=False)
    keyed_vectors = load_emoji2vec(config["paths"]["emoji2vec"])
    preprocessing: dict[str, Any] = {
        "preprocessing_version": preprocessor.version,
        "emoji_package_version": actual_emoji_version,
        "pyvi_applied": False,
        "tokenizer": config["model"]["model_name"],
        "max_length": config["model"]["max_length"],
        "splits": {},
    }
    coverage_rows: list[dict[str, Any]] = []
    oov_frames: list[pd.DataFrame] = []
    historical_emoji_comment_checks = {"train": 4054, "validation": 499, "test": 529}
    for split_name, audited in audited_splits.items():
        prepared = prepare_text_columns(audited.frame, preprocessor)
        texts = prepared["original_text"].astype(str).tolist()
        presence = emoji_presence_statistics(texts)
        observed = presence["comments_with_unicode_emoji"]
        expected_check = historical_emoji_comment_checks[split_name]
        preprocessing["splits"][split_name] = {
            **presence,
            "historical_notebook_check_approximate": expected_check,
            "difference_from_historical_check": observed - expected_check,
            "token_lengths": token_length_statistics(
                prepared["model_text"].astype(str).tolist(), tokenizer
            ),
        }
        coverage, oov = emoji2vec_coverage(texts, keyed_vectors, split=split_name)
        coverage_rows.append(coverage)
        oov_frames.append(oov)
    _write_json(output_dir / "preprocessing_audit.json", preprocessing)
    emoji_dir = output_dir / "emoji_analysis"
    emoji_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(
        emoji_dir / "emoji_coverage_by_split.csv", index=False
    )
    pd.concat(oov_frames, ignore_index=True).to_csv(
        emoji_dir / "emoji_oov_frequency.csv", index=False
    )


def _ensemble_from_seed_dirs(
    seed_dirs: Sequence[Path],
    output_dir: Path,
    stable_ids: Sequence[str],
) -> dict[str, Any]:
    val_probabilities = [np.load(path / "val_probs.npy") for path in seed_dirs]
    test_probabilities = [np.load(path / "test_probs.npy") for path in seed_dirs]
    val_targets = [np.load(path / "val_targets.npy") for path in seed_dirs]
    test_targets = [np.load(path / "test_targets.npy") for path in seed_dirs]
    val_ids = [
        json.loads((path / "val_ids.json").read_text(encoding="utf-8"))
        for path in seed_dirs
    ]
    test_ids = [
        json.loads((path / "test_ids.json").read_text(encoding="utf-8"))
        for path in seed_dirs
    ]
    if any(not np.array_equal(val_targets[0], item) for item in val_targets[1:]):
        raise AssertionError("Seed validation targets differ")
    if any(not np.array_equal(test_targets[0], item) for item in test_targets[1:]):
        raise AssertionError("Seed test targets differ")
    if any(val_ids[0] != item for item in val_ids[1:]):
        raise AssertionError("Seeds do not share the exact ordered validation IDs")
    if any(test_ids[0] != item for item in test_ids[1:]):
        raise AssertionError("Seeds do not share the exact ordered test IDs")
    if list(map(str, stable_ids)) != list(map(str, test_ids[0])):
        raise AssertionError("Seed test IDs do not match canonical raw test order")
    val_probs = np.mean(val_probabilities, axis=0)
    test_probs = np.mean(test_probabilities, axis=0)
    thresholds = fit_per_class_thresholds(
        val_probs, val_targets[0], source_split="validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "val_probs.npy", val_probs)
    np.save(output_dir / "val_targets.npy", val_targets[0])
    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "test_targets.npy", test_targets[0])
    return write_evaluation_artifacts(
        output_dir,
        stable_ids=stable_ids,
        targets=test_targets[0],
        probabilities=test_probs,
        thresholds=thresholds,
        require_test_support=True,
    )


def _load_thresholds(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float64)
    return np.asarray([payload[label] for label in LABEL_NAMES], dtype=np.float64)


def run_primary_analyses(
    output_dir: Path,
    audited_splits: Mapping[str, AuditedSplit],
    *,
    a0_ensemble_dir: Path,
    c3_ensemble_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    targets = np.load(c3_ensemble_dir / "test_targets.npy")
    c3_probs = np.load(c3_ensemble_dir / "test_probs.npy")
    a0_probs = np.load(a0_ensemble_dir / "test_probs.npy")
    c3_thresholds = _load_thresholds(c3_ensemble_dir / "thresholds.json")
    a0_thresholds = _load_thresholds(a0_ensemble_dir / "thresholds.json")
    c3_predictions = binarize(c3_probs, c3_thresholds)
    a0_predictions = binarize(a0_probs, a0_thresholds)
    write_statistical_artifacts(
        output_dir / "statistics",
        targets=targets,
        baseline_predictions=a0_predictions,
        c3_predictions=c3_predictions,
        label_names=LABEL_NAMES,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    test_frame = audited_splits["test"].frame
    emoji_sequences = [extract_emoji_sequence(text) for text in test_frame["text"].astype(str)]
    emoji_mask = np.asarray([bool(sequence) for sequence in emoji_sequences], dtype=bool)
    subset_rows: list[dict[str, Any]] = []
    for system, probabilities, thresholds in (
        ("A0 probability ensemble", a0_probs, a0_thresholds),
        ("C3 Ensemble", c3_probs, c3_thresholds),
    ):
        for subset, mask in (
            ("complete_test", np.ones(len(targets), dtype=bool)),
            ("emoji_containing", emoji_mask),
            ("non_emoji", ~emoji_mask),
        ):
            subset_rows.extend(
                subset_metrics(
                    targets,
                    probabilities,
                    thresholds,
                    mask,
                    system=system,
                    subset=subset,
                )
            )
    emoji_dir = output_dir / "emoji_analysis"
    pd.DataFrame(subset_rows).to_csv(emoji_dir / "emoji_subset_metrics.csv", index=False)
    bootstrap_rows: list[pd.DataFrame] = []
    for subset, mask in (("emoji_containing", emoji_mask), ("non_emoji", ~emoji_mask)):
        _, distribution, summary = paired_bootstrap_macro_f1(
            targets[mask],
            a0_predictions[mask],
            c3_predictions[mask],
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        summary.insert(0, "subset", subset)
        bootstrap_rows.append(summary)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(
        emoji_dir / "emoji_subset_bootstrap.csv", index=False
    )

    qualitative_dir = output_dir / "qualitative"
    write_qualitative_candidates(
        qualitative_dir,
        stable_ids=test_frame["id"].astype(str).tolist(),
        original_texts=test_frame["text"].astype(str).tolist(),
        targets=targets,
        baseline_probabilities=a0_probs,
        baseline_thresholds=a0_thresholds,
        c3_probabilities=c3_probs,
        c3_thresholds=c3_thresholds,
        emoji_sequences=emoji_sequences,
    )


def _read_metrics(path: Path) -> dict[str, Any]:
    return json.loads((path / "metrics_exact.json").read_text(encoding="utf-8"))


def write_paper_outputs(output_dir: Path) -> None:
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    dataset_table = r"""\begin{tabular}{lrrr}
\toprule
Split & Comments & Positive labels & Labels/comment \\
\midrule
Train & 16,531 & 31,545 & 1.908 \\
Validation & 2,066 & 3,958 & 1.916 \\
Test & 2,067 & 3,942 & 1.907 \\
All & 20,664 & 39,445 & 1.909 \\
\bottomrule
\end{tabular}
"""
    (paper_dir / "dataset_table.tex").write_text(dataset_table, encoding="utf-8")
    experiment_roots = {
        "A0 controlled text BCE": output_dir / "experiments" / "A0_controlled_text_BCE",
        "A1 controlled text ASL": output_dir / "experiments" / "A1_controlled_text_ASL",
        "A2 controlled ASL Emoji": output_dir / "experiments" / "A2_controlled_ASL_Emoji",
        "C3 single model": output_dir,
    }
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for system, root in experiment_roots.items():
        system_rows: list[dict[str, Any]] = []
        for seed in (42, 1, 7):
            metrics_path = root / f"seed{seed}" / "metrics_exact.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "system": system,
                "seed": seed,
                "macro_f1_fixed_0_5": metrics["fixed_threshold_0_5"]["macro_f1"],
                "macro_f1_validation_tuned": metrics[
                    "validation_tuned_per_class_thresholds"
                ]["macro_f1"],
            }
            seed_rows.append(row)
            system_rows.append(row)
        if len(system_rows) == 3:
            fixed_values = np.asarray(
                [row["macro_f1_fixed_0_5"] for row in system_rows], dtype=float
            )
            tuned_values = np.asarray(
                [row["macro_f1_validation_tuned"] for row in system_rows], dtype=float
            )
            aggregate_rows.append(
                {
                    "system": system,
                    "seed_count": 3,
                    "fixed_mean": float(fixed_values.mean()),
                    "fixed_sample_std": float(fixed_values.std(ddof=1)),
                    "tuned_mean": float(tuned_values.mean()),
                    "tuned_sample_std": float(tuned_values.std(ddof=1)),
                }
            )
    seed_summary = pd.DataFrame(seed_rows)
    aggregate_summary = pd.DataFrame(aggregate_rows)
    seed_summary.to_csv(output_dir / "seed_results.csv", index=False)
    aggregate_summary.to_csv(output_dir / "seed_results_summary.csv", index=False)

    ensemble_metrics_path = output_dir / "ensemble" / "metrics_exact.json"
    c3_ensemble_metrics = (
        json.loads(ensemble_metrics_path.read_text(encoding="utf-8"))
        if ensemble_metrics_path.is_file()
        else None
    )
    if c3_ensemble_metrics:
        fixed = c3_ensemble_metrics["fixed_threshold_0_5"]["macro_f1"]
        tuned = c3_ensemble_metrics["validation_tuned_per_class_thresholds"]["macro_f1"]
        result_line = f"C3 Ensemble exact Macro-F1: fixed={fixed!r}, tuned={tuned!r}."
    else:
        result_line = "C3 Ensemble metrics pending successful artifact recovery or clean training."
    summary = f"""# Paper results summary

Use **C3 Ensemble** as the final system name and
`ASL_Emoji_CB_3seed_ensemble` as the technical artifact name.

Dataset reporting is fixed at 2,067 test comments and 3,942 positive test
labels. All-split totals are 20,664 comments and 39,445 positive labels.

{result_line}

The historical reproduced baseline is separately labeled at exact report-derived
Macro-F1 `{HISTORICAL['baseline_macro_f1']}`; it is not an ensemble-matched
controlled A0 result.

Historical executed-notebook output, artifact reconstruction, and newly trained
clean reruns must remain separately labeled. C4 and extended-data results are
excluded from the headline table.
"""
    (paper_dir / "paper_results_summary.md").write_text(summary, encoding="utf-8")

    def latex_name(value: str) -> str:
        return value.replace("_", r"\_")

    main_rows: list[tuple[str, float, float]] = []
    a0_aggregate = next(
        (row for row in aggregate_rows if row["system"] == "A0 controlled text BCE"), None
    )
    c3_aggregate = next(
        (row for row in aggregate_rows if row["system"] == "C3 single model"), None
    )
    if a0_aggregate:
        main_rows.append(
            (
                "A0 single-model mean",
                a0_aggregate["fixed_mean"],
                a0_aggregate["tuned_mean"],
            )
        )
    if c3_aggregate:
        main_rows.append(
            (
                "C3 single-model mean",
                c3_aggregate["fixed_mean"],
                c3_aggregate["tuned_mean"],
            )
        )
    a0_ensemble_path = (
        output_dir
        / "experiments"
        / "A0_controlled_text_BCE"
        / "ensemble"
        / "metrics_exact.json"
    )
    if a0_ensemble_path.is_file():
        metrics = json.loads(a0_ensemble_path.read_text(encoding="utf-8"))
        main_rows.append(
            (
                "A0 probability ensemble",
                metrics["fixed_threshold_0_5"]["macro_f1"],
                metrics["validation_tuned_per_class_thresholds"]["macro_f1"],
            )
        )
    if c3_ensemble_metrics:
        main_rows.append(
            (
                "C3 Ensemble",
                c3_ensemble_metrics["fixed_threshold_0_5"]["macro_f1"],
                c3_ensemble_metrics["validation_tuned_per_class_thresholds"]["macro_f1"],
            )
        )
    if main_rows:
        lines = [
            r"\begin{tabular}{lrr}", r"\toprule",
            r"System & Macro-F1 @0.5 & Macro-F1 tuned \\", r"\midrule",
        ]
        lines.extend(
            f"{latex_name(name)} & {fixed_value:.4f} & {tuned_value:.4f} \\\\" 
            for name, fixed_value, tuned_value in main_rows
        )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        main_table = "\n".join(lines) + "\n"
    else:
        main_table = "% Main results require completed A0 and C3 ensembles.\n"
    (paper_dir / "main_results_table.tex").write_text(main_table, encoding="utf-8")

    if seed_rows:
        lines = [
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"System & Seed & Macro-F1 @0.5 & Macro-F1 tuned \\", r"\midrule",
        ]
        lines.extend(
            f"{latex_name(row['system'])} & {row['seed']} & "
            f"{row['macro_f1_fixed_0_5']:.4f} & "
            f"{row['macro_f1_validation_tuned']:.4f} \\\\" for row in seed_rows
        )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        seed_table = "\n".join(lines) + "\n"
    else:
        seed_table = "% Seed results require all requested runs.\n"
    (paper_dir / "seed_results_table.tex").write_text(seed_table, encoding="utf-8")

    per_class_path = output_dir / "ensemble" / "per_class_metrics.csv"
    if per_class_path.is_file():
        per_class = pd.read_csv(per_class_path)
        lines = [
            r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Label & Precision & Recall & F1 & Support \\", r"\midrule",
        ]
        lines.extend(
            f"{latex_name(str(row.label))} & {row.tuned_precision:.4f} & "
            f"{row.tuned_recall:.4f} & {row.tuned_f1:.4f} & {int(row.support)} \\\\"
            for row in per_class.itertuples(index=False)
        )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        per_class_table = "\n".join(lines) + "\n"
    else:
        per_class_table = "% Per-class results require completed C3 ensemble evaluation.\n"
    (paper_dir / "per_class_table.tex").write_text(per_class_table, encoding="utf-8")

    paired_path = output_dir / "statistics" / "paired_bootstrap_summary.csv"
    if paired_path.is_file():
        paired = pd.read_csv(paired_path).iloc[0]
        statistical_table = (
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Comparison & Delta & 95\\% CI low & 95\\% CI high & $p$ \\\\\n"
            "\\midrule\n"
            f"C3 Ensemble $-$ A0 ensemble & {paired.point_delta:.4f} & "
            f"{paired.ci_lower_95:.4f} & {paired.ci_upper_95:.4f} & "
            f"{paired.two_sided_bootstrap_p_value:.4g} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
    else:
        statistical_table = "% Statistical results require paired model predictions.\n"
    (paper_dir / "statistical_results_table.tex").write_text(
        statistical_table, encoding="utf-8"
    )


def _write_readme(output_dir: Path) -> None:
    text = """# C3 clean artifacts

This directory is generated by `src/c3_clean/run_experiments.py`.

The pipeline is strict: source CSV discrepancies are written to
`data_discrepancies.md` and stop all reconstruction, tuning, training, and test
evaluation. Historical notebook display values are never substituted for
recomputed array-based metrics.

Priority order:

1. P0: data and artifact audit, probability reconstruction, validation-only thresholds.
2. P1: canonical C3 seeds 42, 1, and 7 plus probability ensemble.
3. P2: controlled A0-A3 three-seed ablations and ensemble-matched comparison.
4. P3: optional diagnostics kept separate from the canonical final model.

Private qualitative text is written only to
`qualitative/qualitative_candidates_private.csv`, which is gitignored and must
undergo manual privacy review before paper use.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _append_manifest(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = output_dir / "experiment_manifest.csv"
    existing = pd.read_csv(path).to_dict(orient="records") if path.is_file() else []
    pd.DataFrame([*existing, *rows]).to_csv(path, index=False)


def _extended_training_split(
    config: Mapping[str, Any],
    canonical_splits: Mapping[str, AuditedSplit],
) -> AuditedSplit:
    path = next(
        (
            Path(candidate)
            for candidate in config["paths"].get("extended_train_candidates", [])
            if Path(candidate).is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError("C3-extended-matched requested but extended train CSV is absent")
    extended = load_split(path, "train")
    duplicate_rows = [
        index
        for index, labels in enumerate(extended.label_ids)
        if len(labels) != len(set(labels))
    ]
    if duplicate_rows:
        raise ValueError(
            f"Extended training split contains duplicate label IDs in rows {duplicate_rows[:10]}"
        )
    if extended.frame["id"].duplicated().any():
        raise ValueError("Extended training split contains duplicate stable IDs")
    held_out_ids = set(
        canonical_splits["validation"].frame["id"].astype(str).tolist()
        + canonical_splits["test"].frame["id"].astype(str).tolist()
    )
    overlap = sorted(set(extended.frame["id"].astype(str)) & held_out_ids)
    if overlap:
        raise ValueError(
            f"Extended training split overlaps canonical validation/test IDs: {overlap[:10]}"
        )
    return extended


def _run_training_group(
    names: Sequence[str],
    *,
    config: Mapping[str, Any],
    audited_splits: Mapping[str, AuditedSplit],
    output_dir: Path,
    dataset_hashes: Mapping[str, Any],
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    selected_seeds = list(seeds or config["training"]["seeds"])
    for name in names:
        run_splits = dict(audited_splits)
        run_hashes = dict(dataset_hashes)
        if name == "C3-extended-matched":
            extended = _extended_training_split(config, audited_splits)
            run_splits["train"] = extended
            run_hashes["extended_train"] = {
                "path": extended.path.as_posix(),
                "sha256": extended.sha256,
                "id_order_sha256": extended.id_order_sha256,
            }
        for seed in selected_seeds:
            if name == "A3_controlled_ASL_Emoji_CB":
                seed_dir = output_dir / f"seed{seed}"
            else:
                seed_dir = output_dir / "experiments" / name / f"seed{seed}"
            completed_files = (
                seed_dir / "metrics_exact.json",
                seed_dir / "best_checkpoint.pt",
                seed_dir / "training_history.csv",
                seed_dir / "val_probs.npy",
                seed_dir / "test_probs.npy",
            )
            if all(path.is_file() for path in completed_files):
                result = {
                    "experiment": name,
                    "seed": seed,
                    "status": "skipped_completed",
                    "output_dir": seed_dir.as_posix(),
                }
            else:
                result = train_one_seed(
                    experiment_name=name,
                    seed=int(seed),
                    audited_splits=run_splits,
                    config=config,
                    output_dir=seed_dir,
                    dataset_hashes=run_hashes,
                    resume=True,
                )
            results.append(result)
            _append_manifest(output_dir, [result])
    return results


def _validate_selected_seeds(
    requested: Sequence[int] | None, configured: Sequence[int]
) -> list[int]:
    """Return a stable, validated subset of configured training seeds."""
    configured_seeds = [int(seed) for seed in configured]
    if requested is None:
        return configured_seeds
    selected = [int(seed) for seed in requested]
    unknown = sorted(set(selected) - set(configured_seeds))
    if unknown:
        raise ValueError(
            f"Requested seeds {unknown} are not configured; expected a subset of {configured_seeds}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("Each --seeds value must be specified at most once")
    return selected


def _seed_arrays_ready(seed_dirs: Sequence[Path]) -> bool:
    """Whether a complete set of arrays and ordering IDs is available for ensembling."""
    required_names = (
        "val_probs.npy",
        "val_targets.npy",
        "test_probs.npy",
        "test_targets.npy",
        "val_ids.json",
        "test_ids.json",
    )
    return all(
        all((seed_dir / name).is_file() for name in required_names)
        for seed_dir in seed_dirs
    )


def _record_pending_assembly(
    output_dir: Path,
    *,
    priority: str,
    experiment: str,
    seed_dirs: Sequence[Path],
) -> None:
    required_names = (
        "val_probs.npy",
        "val_targets.npy",
        "test_probs.npy",
        "test_targets.npy",
        "val_ids.json",
        "test_ids.json",
    )
    missing: list[str] = []
    for seed_dir in seed_dirs:
        absent = [name for name in required_names if not (seed_dir / name).is_file()]
        if absent:
            missing.append(f"{seed_dir.as_posix()} [{', '.join(absent)}]")
    _append_manifest(
        output_dir,
        [
            {
                "priority": priority,
                "experiment": experiment,
                "seed": "",
                "status": "awaiting_seed_artifacts",
                "missing_seed_artifacts": ";".join(missing),
            }
        ],
    )


def package_kaggle_artifacts(output_dir: Path) -> Path:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("The final Kaggle ZIP is only created under /kaggle/working")
    required = [
        output_dir / "data_audit.json",
        output_dir / "ensemble" / "metrics_exact.json",
        output_dir / "ensemble" / "per_class_metrics.csv",
        output_dir / "statistics" / "bootstrap_ci.csv",
        output_dir / "statistics" / "paired_bootstrap.csv",
        output_dir / "statistics" / "per_class_tests.csv",
        output_dir / "statistics" / "holm_corrected_pvalues.csv",
        output_dir / "emoji_analysis" / "emoji_coverage_by_split.csv",
        output_dir / "emoji_analysis" / "emoji_oov_frequency.csv",
        output_dir / "emoji_analysis" / "emoji_subset_metrics.csv",
    ]
    for seed in (42, 1, 7):
        seed_dir = output_dir / f"seed{seed}"
        required.extend(
            [
                seed_dir / "best_checkpoint.pt",
                seed_dir / "training_history.csv",
                seed_dir / "val_probs.npy",
                seed_dir / "val_targets.npy",
                seed_dir / "test_probs.npy",
                seed_dir / "test_targets.npy",
                seed_dir / "metrics_exact.json",
            ]
        )
    missing = [path.as_posix() for path in required if not path.is_file()]
    audit = json.loads((output_dir / "data_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("Refusing to package: data audit did not pass")
    if missing:
        raise RuntimeError(
            "Refusing to package incomplete artifacts; missing:\n" + "\n".join(missing)
        )
    archive = Path("/kaggle/working/C3_clean_artifacts.zip")
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=output_dir)
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/c3_clean.yaml")
    parser.add_argument("--priority", choices=("P0", "P1", "P2", "P3", "ALL"), default="P0")
    parser.add_argument("--run-experiments", nargs="*", default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Train only this subset of configured seeds. Useful for independent "
            "Kaggle seed-worker jobs; ensemble assembly waits for all configured seeds."
        ),
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help=(
            "Do not train. Assemble completed seed artifacts into ensembles and fail "
            "if the required arrays/ordered IDs are not available."
        ),
    )
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args(argv)

    config, repository_root, output_dir = resolve_config(args.config)
    selected_seeds = _validate_selected_seeds(args.seeds, config["training"]["seeds"])
    if args.assemble_only and args.priority not in {"P1", "P2", "ALL"}:
        parser.error("--assemble-only is valid only for P1, P2, or ALL")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(output_dir)
    inventory = _inventory_legacy_artifacts(
        repository_root, config["paths"].get("artifact_roots", [])
    )
    _write_json(output_dir / "artifact_inventory.json", inventory)
    write_configuration_audit(output_dir, inventory=inventory)
    write_annotation_audit(output_dir, repository_root)
    assert_threshold_leakage_guard()

    baseline_report = (
        repository_root
        / "outputs"
        / "c1-visobert_reproduce"
        / "baseline_classification_report.csv"
    )
    if baseline_report.is_file():
        baseline_recomputed = recover_macro_f1_from_classification_report(baseline_report)
        _write_json(
            output_dir / "historical_baseline_reconstruction.json",
            {
                "source": baseline_report.as_posix(),
                "method": "mean_of_28_exact_per_class_f1_values",
                "recomputed_macro_f1": baseline_recomputed,
                "historical_value": HISTORICAL["baseline_macro_f1"],
                "exact_match": bool(
                    np.isclose(
                        baseline_recomputed,
                        HISTORICAL["baseline_macro_f1"],
                        rtol=0.0,
                        atol=1e-15,
                    )
                ),
                "limitation": "Predictions and probabilities were not present; other metrics cannot be independently reconstructed.",
            },
        )

    try:
        audited_splits, _ = audit_dataset(
            config["paths"]["data_dir"], output_dir, raise_on_failure=True
        )
    except AuditFailure as exc:
        hashes = _load_hashes(output_dir) if (output_dir / "data_hashes.json").is_file() else {}
        _write_json(
            output_dir / "environment.json",
            collect_environment(repository_root, hashes),
        )
        _append_manifest(
            output_dir,
            [
                {
                    "priority": args.priority,
                    "experiment": "pipeline",
                    "seed": "",
                    "status": "blocked_data_invariant",
                    "reason": str(exc),
                }
            ],
        )
        write_paper_outputs(output_dir)
        print(f"FATAL: {exc}")
        return 2

    dataset_hashes = _load_hashes(output_dir)
    _write_json(
        output_dir / "environment.json",
        collect_environment(repository_root, dataset_hashes),
    )
    run_preprocessing_audit(config, audited_splits, output_dir)

    reconstruction: dict[str, Any] | None = None
    if args.priority in {"P0", "ALL"}:
        roots = [Path(path) for path in config["paths"].get("artifact_roots", [])]
        if Path("/kaggle/input").is_dir():
            roots.append(Path("/kaggle/input"))
        try:
            reconstruction = reconstruct_historical_artifacts(
                audited_splits=audited_splits,
                output_dir=output_dir,
                artifact_roots=roots,
            )
            _append_manifest(
                output_dir,
                [{"priority": "P0", "experiment": "artifact_reconstruction", **reconstruction}],
            )
        except FileNotFoundError as exc:
            reconstruction = {"status": "not_recomputed", "reason": str(exc)}
            _write_json(output_dir / "artifact_reconstruction.json", reconstruction)
            _append_manifest(
                output_dir,
                [
                    {
                        "priority": "P0",
                        "experiment": "artifact_reconstruction",
                        "status": "blocked_missing_artifacts",
                        "reason": str(exc),
                    }
                ],
            )
        write_configuration_audit(
            output_dir, inventory=inventory, artifact_reconstruction=reconstruction
        )

    selected = args.run_experiments
    if args.priority in {"P1", "ALL"}:
        names = selected or ["A3_controlled_ASL_Emoji_CB"]
        if not args.assemble_only:
            _run_training_group(
                names,
                config=config,
                audited_splits=audited_splits,
                output_dir=output_dir,
                dataset_hashes=dataset_hashes,
                seeds=selected_seeds,
            )
        seed_dirs = [output_dir / f"seed{seed}" for seed in config["training"]["seeds"]]
        if _seed_arrays_ready(seed_dirs):
            _ensemble_from_seed_dirs(
                seed_dirs,
                output_dir / "ensemble",
                audited_splits["test"].frame["id"].astype(str).tolist(),
            )
        else:
            _record_pending_assembly(
                output_dir,
                priority="P1",
                experiment="C3_ensemble_assembly",
                seed_dirs=seed_dirs,
            )
            if args.assemble_only:
                raise RuntimeError(
                    "Cannot assemble C3 Ensemble: attach/import all three seed artifacts first"
                )

    if args.priority in {"P2", "ALL"}:
        names = selected or [
            "A0_controlled_text_BCE",
            "A1_controlled_text_ASL",
            "A2_controlled_ASL_Emoji",
            "A3_controlled_ASL_Emoji_CB",
        ]
        if not args.assemble_only:
            _run_training_group(
                names,
                config=config,
                audited_splits=audited_splits,
                output_dir=output_dir,
                dataset_hashes=dataset_hashes,
                seeds=selected_seeds,
            )
        stable_ids = audited_splits["test"].frame["id"].astype(str).tolist()
        a0_seed_dirs = [
            output_dir / "experiments" / "A0_controlled_text_BCE" / f"seed{seed}"
            for seed in config["training"]["seeds"]
        ]
        a0_ensemble_dir = output_dir / "experiments" / "A0_controlled_text_BCE" / "ensemble"
        c3_seed_dirs = [output_dir / f"seed{seed}" for seed in config["training"]["seeds"]]
        c3_ensemble_dir = output_dir / "ensemble"
        if _seed_arrays_ready(a0_seed_dirs) and _seed_arrays_ready(c3_seed_dirs):
            _ensemble_from_seed_dirs(a0_seed_dirs, a0_ensemble_dir, stable_ids)
            _ensemble_from_seed_dirs(c3_seed_dirs, c3_ensemble_dir, stable_ids)
            run_primary_analyses(
                output_dir,
                audited_splits,
                a0_ensemble_dir=a0_ensemble_dir,
                c3_ensemble_dir=c3_ensemble_dir,
                bootstrap_iterations=int(config["statistics"]["bootstrap_iterations"]),
                bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
            )
        else:
            _record_pending_assembly(
                output_dir,
                priority="P2",
                experiment="A0_and_C3_ensemble_assembly",
                seed_dirs=[*a0_seed_dirs, *c3_seed_dirs],
            )
            if args.assemble_only:
                raise RuntimeError(
                    "Cannot assemble P2 analyses: attach/import all A0 and C3 seed artifacts first"
                )

    if args.priority in {"P3", "ALL"}:
        names = selected or config["optional_experiments"]
        if not args.assemble_only:
            _run_training_group(
                names,
                config=config,
                audited_splits=audited_splits,
                output_dir=output_dir,
                dataset_hashes=dataset_hashes,
                seeds=selected_seeds,
            )

    write_paper_outputs(output_dir)
    if args.package:
        archive = package_kaggle_artifacts(output_dir)
        print(f"Created {archive}")
    print(f"Completed {args.priority}; artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
