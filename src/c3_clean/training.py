"""Deterministic, resumable training for the controlled C3 experiment family."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data_audit import AuditedSplit
from .evaluation import (
    binarize,
    exact_metrics,
    fit_per_class_thresholds,
    validate_probabilities,
    write_evaluation_artifacts,
)
from .losses import build_loss
from .model import build_model
from .preprocessing import (
    ImmutablePreprocessor,
    emoji_vector_for_text,
    prepare_text_columns,
)


def set_deterministic_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def load_emoji2vec(path: str | Path):
    from gensim.models import KeyedVectors

    return KeyedVectors.load_word2vec_format(str(path), binary=True)


def _fixed_random_vector(token: str, *, dimension: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    token_seed = int.from_bytes(digest[:8], "little", signed=False)
    generator = np.random.default_rng(token_seed)
    return generator.standard_normal(dimension).astype(np.float32)


def build_emoji_matrix(
    original_texts: Sequence[str],
    keyed_vectors: Any,
    *,
    dimension: int,
    control: str,
    seed: int,
) -> np.ndarray:
    from .preprocessing import extract_emoji_sequence, resolve_emoji2vec_key

    vectors: list[np.ndarray] = []
    for text in original_texts:
        if control == "random":
            token_vectors = [
                _fixed_random_vector(token, dimension=dimension, seed=seed)
                for token in extract_emoji_sequence(text)
            ]
            vector = (
                np.stack(token_vectors).mean(axis=0).astype(np.float32)
                if token_vectors
                else np.zeros(dimension, dtype=np.float32)
            )
        elif control == "zero":
            vector = np.zeros(dimension, dtype=np.float32)
        else:
            vector = emoji_vector_for_text(
                text, keyed_vectors, dimension=dimension
            )
        vectors.append(vector)
    matrix = np.stack(vectors).astype(np.float32)
    if control == "shuffle":
        generator = np.random.default_rng(seed)
        matrix = matrix[generator.permutation(len(matrix))]
    return matrix


class C3Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        targets: np.ndarray,
        emoji_matrix: np.ndarray,
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if not (len(frame) == targets.shape[0] == emoji_matrix.shape[0]):
            raise ValueError("Frame, targets, and emoji matrix lengths differ")
        self.ids = frame["id"].astype(str).tolist()
        self.texts = frame["model_text"].astype(str).tolist()
        self.targets = targets.astype(np.float32, copy=True)
        self.emoji_matrix = emoji_matrix.astype(np.float32, copy=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "id": self.ids[index],
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "emoji_vectors": torch.from_numpy(self.emoji_matrix[index]),
            "labels": torch.from_numpy(self.targets[index]),
        }


def _device_and_precision(runtime: Mapping[str, Any]) -> tuple[torch.device, Any, int]:
    requested_gpus = int(runtime["num_gpus"])
    if requested_gpus not in (1, 2):
        raise ValueError("runtime.num_gpus must be exactly 1 or 2")
    available_gpus = torch.cuda.device_count()
    if available_gpus < requested_gpus:
        raise RuntimeError(
            f"Requested {requested_gpus} GPU(s), but only {available_gpus} are visible"
        )
    device = torch.device("cuda:0")
    precision = str(runtime["precision"])
    if precision == "fp32":
        autocast_dtype = None
    elif precision == "fp16":
        autocast_dtype = torch.float16
    elif precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is not supported by the active GPU")
        autocast_dtype = torch.bfloat16
    else:
        raise ValueError("runtime.precision must be fp32, fp16, or bf16")
    return device, autocast_dtype, requested_gpus


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _symmetric_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first = torch.stack((first_logits, -first_logits), dim=-1)
    second = torch.stack((second_logits, -second_logits), dim=-1)
    first_log = F.log_softmax(first, dim=-1)
    second_log = F.log_softmax(second, dim=-1)
    first_prob = first_log.exp()
    second_prob = second_log.exp()
    return 0.5 * (
        F.kl_div(first_log, second_prob, reduction="batchmean")
        + F.kl_div(second_log, first_prob, reduction="batchmean")
    )


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ordered_ids: list[str] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        emoji_vectors = batch["emoji_vectors"].to(device, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with context:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emoji_vectors=emoji_vectors,
            )
        probabilities.append(torch.sigmoid(output.logits.float()).cpu().numpy())
        targets.append(batch["labels"].numpy().astype(np.int8))
        ordered_ids.extend(map(str, batch["id"]))
    return (
        np.concatenate(probabilities),
        np.concatenate(targets),
        ordered_ids,
    )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_epoch: int,
    best_validation_macro_f1: float,
    patience_used: int,
    resolved_config: dict[str, Any],
    dataset_hashes: dict[str, Any],
    data_loader_generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "patience_used": patience_used,
            "config": resolved_config,
            "dataset_hashes": dataset_hashes,
            "model_class": type(_unwrap(model)).__name__,
            "data_loader_generator_state": data_loader_generator.get_state(),
        },
        path,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def train_one_seed(
    *,
    experiment_name: str,
    seed: int,
    audited_splits: Mapping[str, AuditedSplit],
    config: Mapping[str, Any],
    output_dir: str | Path,
    dataset_hashes: dict[str, Any],
    resume: bool = True,
) -> dict[str, Any]:
    """Train one seed, selecting checkpoints and thresholds on validation only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(seed)
    device, autocast_dtype, num_gpus = _device_and_precision(config["runtime"])

    experiment_cfg = dict(config["experiments"][experiment_name])
    resolved_config = {
        "canonical_name": "C3 Ensemble" if experiment_name == "A3_controlled_ASL_Emoji_CB" else None,
        "technical_name": "ASL_Emoji_CB_3seed_ensemble"
        if experiment_name == "A3_controlled_ASL_Emoji_CB"
        else experiment_name,
        "experiment_name": experiment_name,
        "seed": seed,
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "loss": dict(config["loss"]),
        "runtime": dict(config["runtime"]),
        "experiment_flags": experiment_cfg,
        "resolved_num_gpus": num_gpus,
        "resolved_precision": config["runtime"]["precision"],
        "resolved_gradient_accumulation": int(config["training"]["gradient_accumulation"]),
        "resolved_batch_size": int(config["training"]["batch_size"]),
        "resolved_max_length": int(config["model"]["max_length"]),
        "host": platform.node(),
    }
    _write_json(output_dir / "config.json", resolved_config)

    preprocessor = ImmutablePreprocessor.from_docs(config["paths"]["docs_dir"])
    from .preprocessing import emoji_package_version

    actual_emoji_version = emoji_package_version()
    expected_emoji_version = str(config["preprocessing"]["emoji_package_version"])
    if actual_emoji_version != expected_emoji_version:
        raise RuntimeError(
            f"emoji package version {actual_emoji_version} != pinned {expected_emoji_version}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_name"], use_fast=False
    )
    emoji2vec = load_emoji2vec(config["paths"]["emoji2vec"])
    control = str(experiment_cfg.get("emoji_control", "normal"))
    datasets: dict[str, C3Dataset] = {}
    prepared_frames: dict[str, pd.DataFrame] = {}
    for split_name, audited in audited_splits.items():
        prepared = prepare_text_columns(audited.frame, preprocessor)
        prepared_frames[split_name] = prepared
        emoji_matrix = build_emoji_matrix(
            prepared["original_text"].tolist(),
            emoji2vec,
            dimension=int(config["model"]["emoji_dim"]),
            control=control,
            seed=seed,
        )
        datasets[split_name] = C3Dataset(
            prepared,
            audited.targets,
            emoji_matrix,
            tokenizer,
            max_length=int(config["model"]["max_length"]),
        )

    batch_size = int(config["training"]["batch_size"])
    workers = int(config["runtime"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    model = build_model(experiment_name, config["model"]).to(device)
    if num_gpus == 2:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    criterion = build_loss(
        experiment_name, audited_splits["train"].targets, dict(config["loss"]), device
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    grad_accum = int(config["training"]["gradient_accumulation"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    maximum_epochs = int(config["training"]["max_epochs"])
    total_steps = optimizer_steps_per_epoch * maximum_epochs
    warmup_steps = optimizer_steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_dtype == torch.float16)

    last_checkpoint = output_dir / "last_checkpoint.pt"
    best_checkpoint = output_dir / "best_checkpoint.pt"
    start_epoch = 1
    best_epoch = 0
    best_validation_macro_f1 = -1.0
    patience_used = 0
    history: list[dict[str, Any]] = []
    history_path = output_dir / "training_history.csv"
    if resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if checkpoint["dataset_hashes"] != dataset_hashes:
            raise RuntimeError("Resume checkpoint dataset hashes do not match current CSV files")
        _unwrap(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint["config"] != resolved_config:
            raise RuntimeError("Resume checkpoint resolved config does not match current config")
        generator.set_state(checkpoint["data_loader_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_macro_f1 = float(checkpoint["best_validation_macro_f1"])
        patience_used = int(checkpoint["patience_used"])
        if history_path.is_file():
            history = pd.read_csv(history_path).to_dict(orient="records")

    use_rdrop = bool(experiment_cfg.get("use_rdrop", False))
    rdrop_alpha = float(experiment_cfg.get("rdrop_alpha", 1.0))
    epoch_dir = output_dir / "epochs"
    epoch_dir.mkdir(exist_ok=True)
    training_start = time.time()
    for epoch in range(start_epoch, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            emoji_vectors = batch["emoji_vectors"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else torch.autocast(device_type="cuda", enabled=False)
            )
            with context:
                first = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    emoji_vectors=emoji_vectors,
                )
                if use_rdrop:
                    second = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        emoji_vectors=emoji_vectors,
                    )
                    loss = 0.5 * (
                        criterion(first.logits.float(), labels)
                        + criterion(second.logits.float(), labels)
                    ) + rdrop_alpha * _symmetric_kl(
                        first.logits.float(), second.logits.float()
                    )
                else:
                    loss = criterion(first.logits.float(), labels)
                loss = loss / grad_accum
            running_loss += float(loss.detach().cpu()) * grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % grad_accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_probs, val_targets, val_ids = predict_probabilities(
            model, validation_loader, device, autocast_dtype
        )
        expected_val_ids = audited_splits["validation"].frame["id"].astype(str).tolist()
        if val_ids != expected_val_ids:
            raise AssertionError("Validation ID order changed during inference")
        thresholds = fit_per_class_thresholds(
            val_probs, val_targets, source_split="validation"
        )
        fixed_metrics, _, _ = exact_metrics(val_targets, val_probs, 0.5)
        tuned_metrics, _, _ = exact_metrics(val_targets, val_probs, thresholds)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "validation_macro_f1_fixed_0_5": fixed_metrics["macro_f1"],
            "validation_macro_f1_tuned": tuned_metrics["macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.time() - training_start,
        }
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        np.save(epoch_dir / f"epoch_{epoch:02d}_val_probs.npy", val_probs)
        np.save(epoch_dir / f"epoch_{epoch:02d}_val_targets.npy", val_targets)
        _write_json(
            epoch_dir / f"epoch_{epoch:02d}_thresholds.json",
            thresholds.astype(float).tolist(),
        )
        if tuned_metrics["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = float(tuned_metrics["macro_f1"])
            best_epoch = epoch
            patience_used = 0
            _save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_macro_f1=best_validation_macro_f1,
                patience_used=patience_used,
                resolved_config=resolved_config,
                dataset_hashes=dataset_hashes,
                data_loader_generator=generator,
            )
        else:
            patience_used += 1
        _save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_macro_f1=best_validation_macro_f1,
            patience_used=patience_used,
            resolved_config=resolved_config,
            dataset_hashes=dataset_hashes,
            data_loader_generator=generator,
        )
        if patience_used >= int(config["training"]["early_stopping_patience"]):
            break

    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    _unwrap(model).load_state_dict(best["model_state_dict"], strict=True)
    val_probs, val_targets, val_ids = predict_probabilities(
        model, validation_loader, device, autocast_dtype
    )
    thresholds = fit_per_class_thresholds(
        val_probs, val_targets, source_split="validation"
    )
    test_probs, test_targets, test_ids = predict_probabilities(
        model, test_loader, device, autocast_dtype
    )
    expected_test_ids = audited_splits["test"].frame["id"].astype(str).tolist()
    if test_ids != expected_test_ids:
        raise AssertionError("Test ID order changed during inference")
    validate_probabilities(test_probs, test_targets, expected_rows=2067)
    assert int(test_targets.sum()) == 3942

    np.save(output_dir / "val_probs.npy", val_probs)
    np.save(output_dir / "val_targets.npy", val_targets)
    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "test_targets.npy", test_targets)
    np.save(output_dir / "fixed_0_5_predictions.npy", binarize(test_probs, 0.5))
    np.save(output_dir / "tuned_threshold_predictions.npy", binarize(test_probs, thresholds))
    _write_json(output_dir / "val_ids.json", val_ids)
    _write_json(output_dir / "test_ids.json", test_ids)
    metrics = write_evaluation_artifacts(
        output_dir,
        stable_ids=test_ids,
        targets=test_targets,
        probabilities=test_probs,
        thresholds=thresholds,
        require_test_support=True,
    )
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_validation_macro_f1"] = float(best_validation_macro_f1)
    _write_json(output_dir / "metrics_exact.json", metrics)
    return {
        "experiment": experiment_name,
        "seed": seed,
        "output_dir": output_dir.as_posix(),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "test_macro_f1_fixed_0_5": metrics["fixed_threshold_0_5"]["macro_f1"],
        "test_macro_f1_tuned": metrics["validation_tuned_per_class_thresholds"][
            "macro_f1"
        ],
        "status": "completed",
    }


__all__ = [
    "C3Dataset",
    "build_emoji_matrix",
    "load_emoji2vec",
    "predict_probabilities",
    "set_deterministic_seed",
    "train_one_seed",
]
