"""Training loop for ViGoEmotions multi-label baseline."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import CleanTextFn, build_dataloaders, build_tokenizer
from .losses import build_bce_loss
from .metrics import EvalMetrics, compute_metrics
from .model import ViSoBertMultiLabel
from .preprocess import get_pyvi_segmenter, load_resources, make_clean_text
from .utils import EMOTION_LABELS, NUM_LABELS, device_info, get_logger, set_seed

LOGGER = get_logger(__name__)


def _autocast_dtype(use_amp: bool) -> torch.dtype | None:
    if not use_amp or not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(
    model: ViSoBertMultiLabel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    autocast_dtype: torch.dtype | None,
    sweep_thresholds: bool = True,
) -> tuple[EvalMetrics, float]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits.float()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits

        loss = bce(logits, labels)
        total_loss += float(loss.item())
        total_examples += labels.numel()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.cpu().numpy().astype(np.int8))

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(probs_arr, targets_arr, threshold=threshold, sweep_thresholds=sweep_thresholds)
    avg_loss = total_loss / max(total_examples, 1)
    return metrics, avg_loss


def _save_checkpoint(
    path: Path,
    model: ViSoBertMultiLabel,
    config: TrainConfig,
    epoch: int,
    metrics: EvalMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "val_metrics": metrics.to_dict(),
    }
    torch.save(payload, path)


def _maybe_init_wandb(use_wandb: bool, run_name: str, cfg: TrainConfig) -> Any:
    if not use_wandb:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        LOGGER.warning("use_wandb=True but WANDB_API_KEY is not set; skipping W&B init.")
        return None
    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb not installed; skipping W&B init.")
        return None
    project = os.environ.get("WANDB_PROJECT", "vigoemotions")
    run = wandb.init(project=project, name=run_name, config=cfg.to_dict(), reinit=True)
    return run


def run_training(
    config_path: str,
    run_name: str,
    use_wandb: bool = False,
    seed_override: int | None = None,
) -> dict[str, Any]:
    cfg = TrainConfig.from_yaml(config_path)
    if seed_override is not None:
        cfg.seed = int(seed_override)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = _autocast_dtype(cfg.use_amp)
    LOGGER.info(
        "Device: %s | autocast: %s (use_amp=%s)",
        device_info(),
        autocast_dtype,
        cfg.use_amp,
    )
    LOGGER.info("Config: %s", json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))

    runs_dir = Path(cfg.runs_dir) / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = runs_dir / "tb"
    tb_writer = SummaryWriter(tb_dir.as_posix())

    with (runs_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    LOGGER.info("Building tokenizer (%s, use_fast=%s)", cfg.model_name, cfg.use_fast_tokenizer)
    tokenizer = build_tokenizer(cfg.model_name, use_fast=cfg.use_fast_tokenizer)

    text_steps: list[CleanTextFn] = []
    if cfg.apply_clean_text:
        resources = load_resources(cfg.docs_dir)
        text_steps.append(make_clean_text(resources))
        LOGGER.info("clean_text enabled (docs_dir=%s)", cfg.docs_dir)
    else:
        LOGGER.info("clean_text disabled (apply_clean_text=False)")

    if cfg.apply_pyvi:
        pyvi_seg = get_pyvi_segmenter()
        if pyvi_seg is not None:
            text_steps.append(pyvi_seg)
            LOGGER.info("pyvi word segmentation enabled (after clean_text)")
    else:
        LOGGER.info("pyvi disabled (apply_pyvi=False)")

    if text_steps:
        def clean_text_fn(text: str) -> str:
            for step in text_steps:
                text = step(text)
            return text
    else:
        clean_text_fn = None

    LOGGER.info("Building dataloaders from %s", cfg.data_dir)
    train_loader, val_loader, test_loader, pos_weight, _raw = build_dataloaders(
        data_dir=cfg.data_dir,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        num_labels=cfg.num_labels,
        clean_text_fn=clean_text_fn,
    )
    LOGGER.info(
        "Loader sizes -- train: %d batches, val: %d batches, test: %d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    LOGGER.info("Building model (%s)", cfg.model_name)
    model = ViSoBertMultiLabel(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        dropout=cfg.dropout,
    ).to(device)
    LOGGER.info("Trainable params: %s", f"{model.num_trainable_parameters():,}")

    pos_weight_dev = pos_weight.to(device) if cfg.use_pos_weight else None
    criterion = build_bce_loss(pos_weight_dev)

    # Single AdamW group with default weight_decay applied to all parameters,
    # matching the ViGoEmotions baseline (`AdamW(model.parameters(), lr=5e-5)`).
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(cfg.grad_accum, 1))
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.warmup_epochs is not None:
        # Baseline: warmup_steps == len(train_loader) (≈ 1 epoch of optimizer steps).
        warmup_steps = max(1, int(round(cfg.warmup_epochs * steps_per_epoch)))
    else:
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    LOGGER.info(
        "Scheduler: linear warmup %d / total %d (steps_per_epoch=%d, warmup_epochs=%s, warmup_ratio=%s)",
        warmup_steps, total_steps, steps_per_epoch, cfg.warmup_epochs, cfg.warmup_ratio,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16_scaler = autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    wandb_run = _maybe_init_wandb(use_wandb, run_name, cfg)

    best_macro = -1.0
    best_epoch = -1
    best_ckpt = runs_dir / "best.pt"
    history: list[dict[str, Any]] = []

    global_step = 0
    train_start = time.time()

    if cfg.apply_clean_text:
        _docs = Path(cfg.docs_dir)
        assert _docs.is_dir(), f"docs_dir must be an existing directory: {_docs}"
        assert (_docs / "patterns.json").is_file(), (
            f"Missing required preprocessing file: {_docs / 'patterns.json'}"
        )
        assert (_docs / "teencode4.txt").is_file(), (
            f"Missing required preprocessing file: {_docs / 'teencode4.txt'}"
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{cfg.epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(out.logits.float(), labels)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(out.logits, labels)

            loss_value = loss.detach().float().item()
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            loss = loss / max(cfg.grad_accum, 1)

            if use_fp16_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.grad_accum == 0 or step == len(train_loader):
                if use_fp16_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % cfg.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    tb_writer.add_scalar("train/loss", loss_value, global_step)
                    tb_writer.add_scalar("train/lr", lr, global_step)
                    if wandb_run is not None:
                        wandb_run.log(
                            {"train/loss": loss_value, "train/lr": lr, "step": global_step}
                        )
                    progress.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{lr:.2e}"})

        avg_train_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        LOGGER.info("[epoch %d] avg train loss = %.4f", epoch, avg_train_loss)

        val_metrics, val_loss = evaluate(
            model, val_loader, device, cfg.threshold, autocast_dtype, sweep_thresholds=True
        )
        LOGGER.info(
            "[epoch %d] val | loss=%.4f | macroF1=%.4f | weightedF1=%.4f | microF1=%.4f | "
            "hamming=%.4f | tunedMacroF1=%.4f @ t=%.2f",
            epoch, val_loss, val_metrics.macro_f1, val_metrics.weighted_f1,
            val_metrics.micro_f1, val_metrics.hamming,
            val_metrics.macro_f1_tuned, val_metrics.threshold_tuned,
        )

        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/macro_f1", val_metrics.macro_f1, epoch)
        tb_writer.add_scalar("val/weighted_f1", val_metrics.weighted_f1, epoch)
        tb_writer.add_scalar("val/micro_f1", val_metrics.micro_f1, epoch)
        tb_writer.add_scalar("val/hamming", val_metrics.hamming, epoch)
        tb_writer.add_scalar("val/macro_f1_tuned", val_metrics.macro_f1_tuned, epoch)
        tb_writer.add_scalar("val/threshold_tuned", val_metrics.threshold_tuned, epoch)
        tb_writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "val/loss": val_loss,
                    "val/macro_f1": val_metrics.macro_f1,
                    "val/weighted_f1": val_metrics.weighted_f1,
                    "val/micro_f1": val_metrics.micro_f1,
                    "val/hamming": val_metrics.hamming,
                    "val/macro_f1_tuned": val_metrics.macro_f1_tuned,
                    "val/threshold_tuned": val_metrics.threshold_tuned,
                    "train/epoch_loss": avg_train_loss,
                }
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                **val_metrics.to_dict(),
            }
        )

        if val_metrics.macro_f1 > best_macro:
            best_macro = val_metrics.macro_f1
            best_epoch = epoch
            _save_checkpoint(best_ckpt, model, cfg, epoch, val_metrics)
            LOGGER.info("[epoch %d] new best macro F1 %.4f -> saved %s", epoch, best_macro, best_ckpt)

    train_secs = time.time() - train_start
    LOGGER.info("Training done in %.1f s. Best epoch=%d (macroF1=%.4f).",
                train_secs, best_epoch, best_macro)

    LOGGER.info("Loading best checkpoint for final test eval: %s", best_ckpt)
    payload = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])

    test_metrics, test_loss = evaluate(
        model, test_loader, device, cfg.threshold, autocast_dtype, sweep_thresholds=True
    )
    LOGGER.info(
        "[TEST] loss=%.4f | macroF1=%.4f | weightedF1=%.4f | microF1=%.4f | hamming=%.4f | "
        "tunedMacroF1=%.4f @ t=%.2f",
        test_loss, test_metrics.macro_f1, test_metrics.weighted_f1,
        test_metrics.micro_f1, test_metrics.hamming,
        test_metrics.macro_f1_tuned, test_metrics.threshold_tuned,
    )

    tb_writer.add_scalar("test/loss", test_loss, best_epoch)
    tb_writer.add_scalar("test/macro_f1", test_metrics.macro_f1, best_epoch)
    tb_writer.add_scalar("test/weighted_f1", test_metrics.weighted_f1, best_epoch)
    tb_writer.add_scalar("test/micro_f1", test_metrics.micro_f1, best_epoch)
    tb_writer.add_scalar("test/hamming", test_metrics.hamming, best_epoch)
    tb_writer.add_scalar("test/macro_f1_tuned", test_metrics.macro_f1_tuned, best_epoch)

    for i, name in enumerate(EMOTION_LABELS):
        tb_writer.add_scalar(f"test/per_class_f1/{i:02d}_{name}", test_metrics.per_class_f1[i], best_epoch)

    if wandb_run is not None:
        try:
            import wandb
            data = [[name, f] for name, f in zip(EMOTION_LABELS, test_metrics.per_class_f1)]
            table = wandb.Table(data=data, columns=["emotion", "f1"])
            wandb_run.log({"test/per_class_f1": wandb.plot.bar(table, "emotion", "f1", title="Test Per-Class F1")})
            wandb_run.log(
                {
                    "test/loss": test_loss,
                    "test/macro_f1": test_metrics.macro_f1,
                    "test/weighted_f1": test_metrics.weighted_f1,
                    "test/micro_f1": test_metrics.micro_f1,
                    "test/hamming": test_metrics.hamming,
                    "test/macro_f1_tuned": test_metrics.macro_f1_tuned,
                    "test/threshold_tuned": test_metrics.threshold_tuned,
                }
            )
        except Exception as e:  # pragma: no cover
            LOGGER.warning("W&B test logging failed: %s", e)

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro,
        "test": test_metrics.to_dict(),
        "test_loss": test_loss,
        "train_seconds": train_secs,
        "history": history,
        "config": cfg.to_dict(),
        "device": device_info(),
        "label_map": {i: name for i, name in enumerate(EMOTION_LABELS)},
        "num_labels": NUM_LABELS,
    }
    with (runs_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tb_writer.flush()
    tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    return summary
