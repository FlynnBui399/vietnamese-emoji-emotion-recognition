#!/usr/bin/env python3
"""
EmoViS Decisive Ablation Experiments
=====================================
Two publication-grade ablation experiments for Vietnamese multi-label emotion
recognition (ViGoEmotions benchmark, 28 classes).

Experiment 1: A0 + Explicit Emoji2Vec Branch + Weighted BCE
Experiment 2: ViSoBERT with Raw Unicode Emoji Preserved (no emoji-to-text conversion)

Usage on Kaggle:
    !git clone https://github.com/FlynnBui399/vietnamese-emoji-emotion-recognition.git repo
    !cd repo && python scripts/run_decisive_ablations.py \
        --data_dir /kaggle/input/vigoemotions \
        --docs_dir docs \
        --emoji2vec_path /kaggle/input/emoji2vec/emoji2vec.bin \
        --output_dir /kaggle/working/emovis_decisive_experiments \
        --model_name uitnlp/visobert

Resume after timeout: re-run the same command; completed seeds are skipped.

Author: EmoViS Team
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Version pinning / logging
# ---------------------------------------------------------------------------

def print_environment():
    """Print versions for reproducibility log."""
    import transformers
    import sklearn
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"Python:        {sys.version}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Transformers:  {transformers.__version__}")
    print(f"scikit-learn:  {sklearn.__version__}")
    print(f"NumPy:         {np.__version__}")
    print(f"Pandas:        {pd.__version__}")
    try:
        import gensim
        print(f"Gensim:        {gensim.__version__}")
    except ImportError:
        print("Gensim:        NOT INSTALLED (needed for Exp1 emoji branch)")
    print(f"CUDA avail:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:  {torch.version.cuda}")
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"GPU count:     {torch.cuda.device_count()}")
    print(f"Platform:      {platform.platform()}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("decisive_ablation")

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Emotion labels (28 classes from ViGoEmotions)
# ---------------------------------------------------------------------------

EMOTION_LABELS: Tuple[str, ...] = (
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
    "pride", "admiration", "gratitude", "relief", "approval", "realization",
    "surprise", "curiosity", "confusion", "fear", "nervousness", "remorse",
    "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger",
    "annoyance", "disapproval", "neutral",
)
NUM_LABELS = len(EMOTION_LABELS)

# ---------------------------------------------------------------------------
# Preprocessing (from src/preprocess.py, self-contained here)
# ---------------------------------------------------------------------------

def load_preprocessing_resources(docs_dir: str) -> Tuple[dict, dict, dict]:
    """Load patterns.json, teencode4.txt, emojis.json from docs_dir."""
    docs = Path(docs_dir)
    if not docs.is_dir():
        raise FileNotFoundError(
            f"docs_dir does not exist or is not a directory: {docs.resolve()}. "
            f"Expected to find patterns.json, teencode4.txt, and emojis.json inside."
        )
    patterns_path = docs / "patterns.json"
    if not patterns_path.is_file():
        raise FileNotFoundError(f"Missing required preprocessing file: {patterns_path}")
    teen_path = docs / "teencode4.txt"
    if not teen_path.is_file():
        raise FileNotFoundError(f"Missing required preprocessing file: {teen_path}")

    with patterns_path.open("r", encoding="utf-8") as f:
        pattern_dict = json.load(f) or {}
    teen_dict = {}
    with teen_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                teen_dict[parts[0]] = parts[1]
    emoji_dict = {}
    emojis_path = docs / "emojis.json"
    if emojis_path.is_file():
        with emojis_path.open("r", encoding="utf-8") as f:
            emoji_dict = json.load(f) or {}

    logger.info("Loaded preprocessing resources: %d patterns, %d emoji entries, %d teencode pairs",
                len(pattern_dict), len(emoji_dict), len(teen_dict))
    return pattern_dict, emoji_dict, teen_dict


def _normalize_pattern(text: str, pattern_dict: Mapping[str, str]) -> str:
    for pattern, replacement in pattern_dict.items():
        text = re.sub(pattern=pattern, repl=replacement, string=text)
    return text


def _remove_duplicate_chars(text: str) -> str:
    prev_char = None
    result = []
    for char in text:
        if char.isalpha() and prev_char == char:
            continue
        prev_char = char
        result.append(char)
    return "".join(result)


def _remove_duplicate_emoji(text: str) -> str:
    try:
        import emoji as _emoji_lib
    except ImportError:
        return text
    out = []
    prev = None
    for ch in text:
        if ch in _emoji_lib.EMOJI_DATA:
            if ch == prev:
                continue
            prev = ch
        else:
            prev = None
        out.append(ch)
    return "".join(out)


def _replace_teencode(text: str, teen_dict: Mapping[str, str]) -> str:
    for old, new in teen_dict.items():
        pattern = re.compile(r"\b{}\b".format(re.escape(old)))
        text = pattern.sub(new, text)
    return text


def _replacing_emojis(text: str, emoji_dict: Mapping[str, str]) -> str:
    """Replace emoji glyphs with Vietnamese text descriptions."""
    for em, word in emoji_dict.items():
        text = text.replace(em, " " + word + " ")
    return text


_PUNCT_RE = re.compile(r"([.,!?;:])")
_NEWLINE_NOPUNCT_RE = re.compile(r"(?<![.,!?;:])\n")
_NEWLINE_PUNCT_RE = re.compile(r"\n([.,!?;:])?")
_WS_RE = re.compile(r"\s+")


def build_clean_text_fn(
    pattern_dict: dict,
    teen_dict: dict,
    emoji_dict: dict,
    replace_emoji_with_text: bool = False,
) -> Callable[[str], str]:
    """Return a composed clean_text function.

    If replace_emoji_with_text=True, emoji are converted to Vietnamese
    text descriptions (used for A0 baseline and Exp1 text branch).
    If False, raw Unicode emoji are kept (used for Exp2).
    """
    def clean_text(text: str) -> str:
        text = text.lower()
        text = _normalize_pattern(text, pattern_dict)
        text = _remove_duplicate_chars(text)
        text = _remove_duplicate_emoji(text)
        text = _replace_teencode(text, teen_dict)
        if replace_emoji_with_text:
            text = _replacing_emojis(text, emoji_dict)
        # Newline handling
        text = _NEWLINE_NOPUNCT_RE.sub(". ", text)
        text = _NEWLINE_PUNCT_RE.sub(r" \1", text)
        text = _PUNCT_RE.sub(r" \1 ", text)
        text = _WS_RE.sub(" ", text).strip()
        return text
    return clean_text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_label_cell(cell) -> List[int]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        value = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        s2 = s.strip("[]")
        if not s2:
            return []
        try:
            value = [int(x.strip()) for x in s2.split(",") if x.strip()]
        except ValueError:
            return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if hasattr(value, "__iter__"):
        return [int(v) for v in value]
    return []


def to_multi_hot(label_ids: List[int], num_labels: int = NUM_LABELS) -> np.ndarray:
    vec = np.zeros(num_labels, dtype=np.float32)
    for idx in label_ids:
        if 0 <= idx < num_labels:
            vec[idx] = 1.0
    return vec


def load_split_csv(
    csv_path: Path,
    clean_text_fn: Callable[[str], str],
    num_labels: int = NUM_LABELS,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'text' and 'labels' columns; got {df.columns.tolist()}")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    # Keep original text for emoji extraction before any preprocessing
    df["original_text"] = df["text"].astype(str)
    # Apply preprocessing to get the normalized text
    df["text"] = df["original_text"].apply(clean_text_fn)
    df["labels"] = df["labels"].apply(parse_label_cell)
    df["multi_hot"] = df["labels"].apply(lambda ids: to_multi_hot(ids, num_labels))
    return df


def compute_pos_weight(multi_hot_matrix: np.ndarray, eps: float = 1.0) -> torch.Tensor:
    """pos_weight[c] = (N - n_pos[c]) / max(n_pos[c], eps)."""
    n_pos = multi_hot_matrix.sum(axis=0)
    n_neg = multi_hot_matrix.shape[0] - n_pos
    pos_weight = n_neg / np.maximum(n_pos, eps)
    return torch.tensor(pos_weight, dtype=torch.float32)


def extract_emojis(text: str) -> List[str]:
    """Extract all emoji characters from text."""
    try:
        import emoji as _emoji_lib
    except ImportError:
        return []
    return [ch for ch in str(text) if ch in _emoji_lib.EMOJI_DATA]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AblationDataset(Dataset):
    """Dataset that returns tokenized text, labels, and optionally emoji vectors."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 160,
        e2v=None,
        emoji_dim: int = 300,
        use_emoji_vectors: bool = False,
    ):
        self.texts = df["text"].tolist()
        self.original_texts = df["original_text"].tolist()
        self.labels = np.stack(df["multi_hot"].to_list(), axis=0)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.e2v = e2v
        self.emoji_dim = emoji_dim
        self.use_emoji_vectors = use_emoji_vectors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.from_numpy(self.labels[idx]),
        }
        if self.use_emoji_vectors:
            emojis = extract_emojis(self.original_texts[idx])
            vec = self._get_emoji_vector(emojis)
            item["emoji_vectors"] = torch.tensor(vec, dtype=torch.float32)
        return item

    def _get_emoji_vector(self, emojis: List[str]) -> np.ndarray:
        """Mean-pool emoji2vec vectors for all emoji found in the sentence."""
        if not emojis or self.e2v is None:
            return np.zeros(self.emoji_dim, dtype=np.float32)
        vectors = []
        for em in emojis:
            if em in self.e2v:
                vectors.append(np.asarray(self.e2v[em], dtype=np.float32))
            else:
                vectors.append(np.zeros(self.emoji_dim, dtype=np.float32))
        if not vectors:
            return np.zeros(self.emoji_dim, dtype=np.float32)
        return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MaskedMeanPooling(nn.Module):
    """Average all non-padding token embeddings using the attention mask."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, seq_len, hidden_size)
        # attention_mask: (B, seq_len) with 1 for real tokens, 0 for padding
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)  # (B, hidden_size)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return sum_embeddings / sum_mask


class BaselineModel(nn.Module):
    """A0 Baseline: ViSoBERT + masked mean pooling + dropout + linear classifier.

    Used for both A0 (normalized text with emoji->text) and Exp2 (raw emoji text).
    """

    def __init__(self, model_name: str, num_labels: int = 28, dropout: float = 0.2):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.pooling = MaskedMeanPooling()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = self.pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(h))
        return logits, h


class EmojiBranchModel(nn.Module):
    """Exp1: ViSoBERT text branch + Emoji2Vec branch + fusion.

    Text branch: ViSoBERT -> masked mean pooling -> h_text (768-d)
    Emoji branch: mean-pooled emoji2vec (300-d) -> Linear(300->768) -> GELU -> LayerNorm -> h_emoji (768-d)
    Fusion: concat [h_text; h_emoji] (1536) -> Linear(1536->768) -> GELU -> Dropout -> Linear(768->28)

    Note: emoji_projection has bias, so all-zero input does NOT necessarily map to
    all-zero output. This is intentional per the spec -- do NOT gate or zero-out.
    """

    def __init__(self, model_name: str, num_labels: int = 28, dropout: float = 0.2, emoji_dim: int = 300):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.pooling = MaskedMeanPooling()

        # Emoji projection: 300 -> 768
        self.emoji_projection = nn.Sequential(
            nn.Linear(emoji_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Fusion: 1536 -> 768 -> 28
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, emoji_vectors=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h_text = self.pooling(outputs.last_hidden_state, attention_mask)

        if emoji_vectors is None:
            emoji_vectors = torch.zeros(h_text.size(0), 300, device=h_text.device)
        h_emoji = self.emoji_projection(emoji_vectors.float())

        fused = torch.cat([h_text, h_emoji], dim=1)
        logits = self.fusion(fused)
        return logits, h_text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_macro_f1_at_thresholds(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> float:
    """Compute Macro-F1 using per-class thresholds."""
    from sklearn.metrics import f1_score
    preds = (probs >= thresholds[np.newaxis, :]).astype(np.int8)
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def compute_micro_f1_at_thresholds(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    preds = (probs >= thresholds[np.newaxis, :]).astype(np.int8)
    return float(f1_score(targets, preds, average="micro", zero_division=0))


def compute_macro_f1_fixed(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score
    preds = (probs >= threshold).astype(np.int8)
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def compute_micro_f1_fixed(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score
    preds = (probs >= threshold).astype(np.int8)
    return float(f1_score(targets, preds, average="micro", zero_division=0))


def compute_map(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean Average Precision over 28 classes, computed from probabilities (threshold-independent)."""
    from sklearn.metrics import average_precision_score
    aps = []
    for c in range(probs.shape[1]):
        if targets[:, c].sum() == 0:
            aps.append(0.0)
        else:
            aps.append(float(average_precision_score(targets[:, c], probs[:, c])))
    return float(np.mean(aps))


def optimize_per_class_thresholds(
    probs_val: np.ndarray,
    targets_val: np.ndarray,
    min_positive_samples: int = 10,
    default_threshold: float = 0.5,
) -> np.ndarray:
    """Per-class threshold optimization on validation set.

    For each class:
    - Sweep thresholds from 0.05 to 0.94 in steps of 0.01
    - Select threshold that maximizes per-class F1
    - Clip selected threshold to [0.15, 0.85]
    - If fewer than min_positive_samples positive val samples, force 0.5
    """
    from sklearn.metrics import f1_score as _f1
    num_classes = probs_val.shape[1]
    thresholds = np.full(num_classes, default_threshold, dtype=np.float64)
    candidates = np.arange(0.05, 0.95, 0.01)  # 0.05 to 0.94 inclusive

    for c in range(num_classes):
        n_pos = int(targets_val[:, c].sum())
        if n_pos < min_positive_samples:
            thresholds[c] = default_threshold
            continue

        best_f1 = -1.0
        best_t = default_threshold
        y_true_c = targets_val[:, c]
        p_c = probs_val[:, c]

        for t in candidates:
            preds_c = (p_c >= t).astype(np.int8)
            f1_c = float(_f1(y_true_c, preds_c, zero_division=0))
            if f1_c > best_f1:
                best_f1 = f1_c
                best_t = float(t)

        # Clip to [0.15, 0.85]
        best_t = max(0.15, min(0.85, best_t))
        thresholds[c] = best_t

    return thresholds.astype(np.float32)


def paired_bootstrap_test(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    targets: np.ndarray,
    thresholds_a: np.ndarray,
    thresholds_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap significance test.

    Computes one-sided p-value (proportion of resamples where delta <= 0)
    and 95% percentile CI (2.5th-97.5th percentile) of
    delta = MF1(system_B) - MF1(system_A).

    Uses fixed thresholds (derived from validation, not re-optimized per bootstrap).
    """
    rng = np.random.RandomState(seed)
    n_samples = targets.shape[0]
    deltas = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        p_a_boot = probs_a[idx]
        p_b_boot = probs_b[idx]
        t_boot = targets[idx]

        mf1_a = compute_macro_f1_at_thresholds(p_a_boot, t_boot, thresholds_a)
        mf1_b = compute_macro_f1_at_thresholds(p_b_boot, t_boot, thresholds_b)
        deltas[i] = mf1_b - mf1_a

    p_value = float(np.mean(deltas <= 0))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    mean_delta = float(np.mean(deltas))

    return {
        "mean_delta": mean_delta,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (probs, targets) arrays."""
    model.eval()
    all_probs = []
    all_targets = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        kwargs = {}
        if "emoji_vectors" in batch:
            kwargs["emoji_vectors"] = batch["emoji_vectors"].to(device, non_blocking=True)

        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.cpu().numpy().astype(np.int8))

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def train_single_seed(
    experiment_name: str,
    seed: int,
    model_cls,
    model_kwargs: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    pos_weight: torch.Tensor,
    output_dir: Path,
    max_length: int = 160,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 2,
    epochs: int = 12,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_epochs: float = 1.0,
    patience: int = 3,
    grad_clip: float = 1.0,
    e2v=None,
    use_emoji_vectors: bool = False,
) -> dict:
    """Train one model for one seed. Supports resume from checkpoint."""
    set_seed(seed)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Check if this seed is already complete
    metrics_path = seed_dir / "metrics.json"
    if metrics_path.exists():
        logger.info("[%s/seed_%d] Already complete, loading existing results.", experiment_name, seed)
        with open(metrics_path, "r") as f:
            return json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s/seed_%d] Starting training on %s", experiment_name, seed, device)

    # Build datasets
    train_ds = AblationDataset(train_df, tokenizer, max_length, e2v=e2v, use_emoji_vectors=use_emoji_vectors)
    val_ds = AblationDataset(val_df, tokenizer, max_length, e2v=e2v, use_emoji_vectors=use_emoji_vectors)
    test_ds = AblationDataset(test_df, tokenizer, max_length, e2v=e2v, use_emoji_vectors=use_emoji_vectors)

    pin_memory = torch.cuda.is_available()
    # Use a generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Build model
    model = model_cls(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[%s/seed_%d] Trainable parameters: %s", experiment_name, seed, f"{n_params:,}")

    # Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: linear warmup for 1 epoch, then linear decay
    from transformers import get_linear_schedule_with_warmup
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(round(warmup_epochs * steps_per_epoch)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info("[%s/seed_%d] Scheduler: warmup %d / total %d steps", experiment_name, seed, warmup_steps, total_steps)

    # Training loop with early stopping
    best_val_macro = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_checkpoint_path = seed_dir / "best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            kwargs = {}
            if "emoji_vectors" in batch:
                kwargs["emoji_vectors"] = batch["emoji_vectors"].to(device, non_blocking=True)

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_probs, val_targets = predict_probs(model, val_loader, device)
        val_macro_fixed = compute_macro_f1_fixed(val_probs, val_targets, threshold=0.5)

        logger.info(
            "[%s/seed_%d] epoch %d/%d | train_loss=%.4f | val_MF1_fix=%.4f",
            experiment_name, seed, epoch, epochs, avg_loss, val_macro_fixed,
        )

        if val_macro_fixed > best_val_macro:
            best_val_macro = val_macro_fixed
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_val_macro,
                "seed": seed,
                "experiment": experiment_name,
            }, best_checkpoint_path)
            logger.info("[%s/seed_%d] New best MF1=%.4f at epoch %d", experiment_name, seed, best_val_macro, epoch)
        else:
            epochs_without_improvement += 1
            logger.info("[%s/seed_%d] No improvement for %d epoch(s) (patience=%d)",
                       experiment_name, seed, epochs_without_improvement, patience)
            if epochs_without_improvement >= patience:
                logger.info("[%s/seed_%d] Early stopping triggered at epoch %d", experiment_name, seed, epoch)
                break

    # Load best checkpoint
    logger.info("[%s/seed_%d] Loading best checkpoint from epoch %d", experiment_name, seed, best_epoch)
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Final evaluation
    val_probs, val_targets = predict_probs(model, val_loader, device)
    test_probs, test_targets = predict_probs(model, test_loader, device)

    # Per-class threshold optimization on validation
    opt_thresholds = optimize_per_class_thresholds(val_probs, val_targets)

    # Compute all metrics
    val_mf1_opt = compute_macro_f1_at_thresholds(val_probs, val_targets, opt_thresholds)
    val_mf1_fix = compute_macro_f1_fixed(val_probs, val_targets, 0.5)
    val_micro = compute_micro_f1_at_thresholds(val_probs, val_targets, opt_thresholds)
    val_map = compute_map(val_probs, val_targets)

    test_mf1_opt = compute_macro_f1_at_thresholds(test_probs, test_targets, opt_thresholds)
    test_mf1_fix = compute_macro_f1_fixed(test_probs, test_targets, 0.5)
    test_micro = compute_micro_f1_at_thresholds(test_probs, test_targets, opt_thresholds)
    test_map = compute_map(test_probs, test_targets)

    logger.info(
        "[%s/seed_%d] TEST | MF1-opt=%.4f | MF1-fix=%.4f | MicroF1=%.4f | mAP=%.4f",
        experiment_name, seed, test_mf1_opt, test_mf1_fix, test_micro, test_map,
    )

    # Save artifacts
    np.save(seed_dir / "probs_val.npy", val_probs.astype(np.float32))
    np.save(seed_dir / "probs_test.npy", test_probs.astype(np.float32))
    np.save(seed_dir / "targets_val.npy", val_targets.astype(np.int8))
    np.save(seed_dir / "targets_test.npy", test_targets.astype(np.int8))
    np.save(seed_dir / "thresholds.npy", opt_thresholds)

    config_info = {
        "experiment": experiment_name,
        "seed": seed,
        "model_cls": model_cls.__name__,
        "model_kwargs": {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in model_kwargs.items()},
        "max_length": max_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "grad_clip": grad_clip,
        "use_emoji_vectors": use_emoji_vectors,
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    metrics = {
        "experiment": experiment_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "val": {
            "mf1_opt": val_mf1_opt, "mf1_fix": val_mf1_fix,
            "micro_f1": val_micro, "map": val_map,
        },
        "test": {
            "mf1_opt": test_mf1_opt, "mf1_fix": test_mf1_fix,
            "micro_f1": test_micro, "map": test_map,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup GPU memory
    del model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Emoji tokenizer diagnostic (Exp2)
# ---------------------------------------------------------------------------

def run_emoji_tokenizer_diagnostic(tokenizer, df: pd.DataFrame, n_samples: int = 200):
    """Check how ViSoBERT tokenizer handles raw Unicode emoji.

    Reports: average subword tokens per emoji, UNK rate.
    """
    try:
        import emoji as _emoji_lib
    except ImportError:
        logger.warning("emoji library not installed, skipping diagnostic")
        return {}

    has_emoji_mask = df["original_text"].apply(lambda x: any(c in _emoji_lib.EMOJI_DATA for c in str(x)))
    emoji_df = df[has_emoji_mask].head(n_samples)

    if len(emoji_df) == 0:
        logger.info("No emoji-containing samples found for diagnostic")
        return {"n_samples": 0}

    total_emojis = 0
    total_tokens_for_emojis = 0
    total_unks = 0
    unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.convert_tokens_to_ids("[UNK]")

    for _, row in emoji_df.iterrows():
        text = str(row["original_text"])
        emojis_in_text = [c for c in text if c in _emoji_lib.EMOJI_DATA]
        if not emojis_in_text:
            continue

        for em in emojis_in_text:
            total_emojis += 1
            tokens = tokenizer.encode(em, add_special_tokens=False)
            total_tokens_for_emojis += len(tokens)
            total_unks += sum(1 for t in tokens if t == unk_token_id)

    avg_tokens_per_emoji = total_tokens_for_emojis / max(total_emojis, 1)
    unk_rate = total_unks / max(total_tokens_for_emojis, 1)

    result = {
        "n_samples_checked": len(emoji_df),
        "total_emojis_found": total_emojis,
        "total_subword_tokens": total_tokens_for_emojis,
        "avg_subword_tokens_per_emoji": round(avg_tokens_per_emoji, 3),
        "total_unk_tokens": total_unks,
        "unk_rate": round(unk_rate, 5),
    }
    logger.info("Emoji tokenizer diagnostic: %s", json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_experiment(
    experiment_name: str,
    model_cls,
    model_kwargs: dict,
    clean_text_fn: Callable,
    data_dir: Path,
    output_dir: Path,
    tokenizer,
    seeds: List[int],
    e2v=None,
    use_emoji_vectors: bool = False,
    **train_kwargs,
) -> dict:
    """Run all seeds for one experiment, return aggregated results."""
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info("=" * 60)

    # Load data with the appropriate clean_text function
    train_df = load_split_csv(data_dir / "train.csv", clean_text_fn)
    val_df = load_split_csv(data_dir / "val.csv", clean_text_fn)
    test_df = load_split_csv(data_dir / "test.csv", clean_text_fn)

    # Verify split sizes
    expected = {"train": 16531, "val": 2066, "test": 2067}
    actual = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    if actual != expected:
        raise ValueError(f"Unexpected split sizes: expected {expected}, got {actual}")
    logger.info("Split sizes verified: %s", actual)

    # Compute pos_weight from training labels
    train_labels = np.stack(train_df["multi_hot"].to_list(), axis=0)
    pos_weight = compute_pos_weight(train_labels)

    all_seed_results = {}
    for seed in seeds:
        result = train_single_seed(
            experiment_name=experiment_name,
            seed=seed,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tokenizer=tokenizer,
            pos_weight=pos_weight,
            output_dir=exp_dir,
            e2v=e2v,
            use_emoji_vectors=use_emoji_vectors,
            **train_kwargs,
        )
        all_seed_results[seed] = result

    # Compute ensemble (3-seed probability-averaged) metrics
    logger.info("[%s] Computing 3-seed ensemble metrics...", experiment_name)
    test_probs_list = []
    val_probs_list = []
    for seed in seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        test_probs_list.append(np.load(seed_dir / "probs_test.npy"))
        val_probs_list.append(np.load(seed_dir / "probs_val.npy"))

    avg_test_probs = np.mean(np.stack(test_probs_list), axis=0)
    avg_val_probs = np.mean(np.stack(val_probs_list), axis=0)

    # Load targets (same across seeds)
    test_targets = np.load(exp_dir / f"seed_{seeds[0]}" / "targets_test.npy")
    val_targets = np.load(exp_dir / f"seed_{seeds[0]}" / "targets_val.npy")

    # Optimize thresholds on averaged val probs
    ens_thresholds = optimize_per_class_thresholds(avg_val_probs, val_targets)

    ens_mf1_opt = compute_macro_f1_at_thresholds(avg_test_probs, test_targets, ens_thresholds)
    ens_mf1_fix = compute_macro_f1_fixed(avg_test_probs, test_targets, 0.5)
    ens_micro = compute_micro_f1_at_thresholds(avg_test_probs, test_targets, ens_thresholds)
    ens_map = compute_map(avg_test_probs, test_targets)

    logger.info(
        "[%s] ENSEMBLE | MF1-opt=%.4f | MF1-fix=%.4f | MicroF1=%.4f | mAP=%.4f",
        experiment_name, ens_mf1_opt, ens_mf1_fix, ens_micro, ens_map,
    )

    # Save ensemble artifacts
    np.save(exp_dir / "ensemble_probs_test.npy", avg_test_probs.astype(np.float32))
    np.save(exp_dir / "ensemble_probs_val.npy", avg_val_probs.astype(np.float32))
    np.save(exp_dir / "ensemble_thresholds.npy", ens_thresholds)

    ensemble_metrics = {
        "mf1_opt": ens_mf1_opt, "mf1_fix": ens_mf1_fix,
        "micro_f1": ens_micro, "map": ens_map,
    }

    return {
        "experiment": experiment_name,
        "seeds": all_seed_results,
        "ensemble": ensemble_metrics,
        "ensemble_thresholds": ens_thresholds.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="EmoViS Decisive Ablation Experiments")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--docs_dir", type=str, required=True,
                        help="Path to docs/ directory with patterns.json, teencode4.txt, emojis.json")
    parser.add_argument("--emoji2vec_path", type=str, default=None,
                        help="Path to emoji2vec.bin (required for Exp1)")
    parser.add_argument("--output_dir", type=str, default="emovis_decisive_experiments",
                        help="Output directory for all artifacts")
    parser.add_argument("--model_name", type=str, default="uitnlp/visobert")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="42,1,7",
                        help="Comma-separated list of seeds")
    parser.add_argument("--skip_a0", action="store_true", help="Skip A0 baseline if already run")
    parser.add_argument("--skip_exp1", action="store_true", help="Skip experiment 1")
    parser.add_argument("--skip_exp2", action="store_true", help="Skip experiment 2")
    args = parser.parse_args()

    # Print environment
    print_environment()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    logger.info("Seeds: %s", seeds)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate data directory
    for split in ["train.csv", "val.csv", "test.csv"]:
        if not (data_dir / split).exists():
            raise FileNotFoundError(
                f"Missing {split} at {data_dir / split}. "
                f"Please ensure the ViGoEmotions CSV files are at: {data_dir}"
            )

    # Load preprocessing resources
    pattern_dict, emoji_dict, teen_dict = load_preprocessing_resources(args.docs_dir)

    # Build tokenizer
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # Load emoji2vec if needed
    e2v = None
    if not args.skip_exp1:
        if args.emoji2vec_path is None:
            raise ValueError(
                "Experiment 1 requires emoji2vec.bin. Pass --emoji2vec_path /path/to/emoji2vec.bin. "
                "Download from https://github.com/uclnlp/emoji2vec or run: python scripts/download_emoji2vec.py"
            )
        if not Path(args.emoji2vec_path).exists():
            raise FileNotFoundError(
                f"emoji2vec.bin not found at {args.emoji2vec_path}. "
                f"Download from https://github.com/uclnlp/emoji2vec or run: python scripts/download_emoji2vec.py"
            )
        from gensim.models import KeyedVectors
        logger.info("Loading emoji2vec from %s ...", args.emoji2vec_path)
        e2v = KeyedVectors.load_word2vec_format(str(args.emoji2vec_path), binary=True)
        logger.info("Loaded emoji2vec: %d emoji, %d dims", len(e2v), e2v.vector_size)

    # Common training kwargs
    train_kwargs = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "grad_clip": 1.0,
    }

    # Clean text functions
    # A0 + Exp1: normalize text WITH emoji -> Vietnamese text replacement
    clean_text_with_emoji_replacement = build_clean_text_fn(
        pattern_dict, teen_dict, emoji_dict, replace_emoji_with_text=True,
    )
    # Exp2: normalize text WITHOUT emoji replacement (keep raw Unicode emoji)
    clean_text_keep_raw_emoji = build_clean_text_fn(
        pattern_dict, teen_dict, emoji_dict, replace_emoji_with_text=False,
    )

    all_results = {}

    # ======================================================================
    # A0 BASELINE: ViSoBERT + Weighted BCE + normalized text (emoji->text)
    # ======================================================================
    if not args.skip_a0:
        a0_results = run_experiment(
            experiment_name="a0_baseline_wbce",
            model_cls=BaselineModel,
            model_kwargs={"model_name": args.model_name, "num_labels": NUM_LABELS, "dropout": args.dropout},
            clean_text_fn=clean_text_with_emoji_replacement,
            data_dir=data_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            seeds=seeds,
            **train_kwargs,
        )
        all_results["a0"] = a0_results

    # ======================================================================
    # EXPERIMENT 1: A0 + Emoji2Vec Branch + Weighted BCE
    # ======================================================================
    if not args.skip_exp1:
        exp1_results = run_experiment(
            experiment_name="exp1_emoji_branch_wbce",
            model_cls=EmojiBranchModel,
            model_kwargs={
                "model_name": args.model_name,
                "num_labels": NUM_LABELS,
                "dropout": args.dropout,
                "emoji_dim": 300,
            },
            clean_text_fn=clean_text_with_emoji_replacement,
            data_dir=data_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            seeds=seeds,
            e2v=e2v,
            use_emoji_vectors=True,
            **train_kwargs,
        )
        all_results["exp1"] = exp1_results

    # ======================================================================
    # EXPERIMENT 2: ViSoBERT with raw Unicode emoji (no emoji->text conversion)
    # ======================================================================
    if not args.skip_exp2:
        # Run tokenizer diagnostic first
        logger.info("Running emoji tokenizer diagnostic for Exp2...")
        diag_df = load_split_csv(data_dir / "train.csv", clean_text_keep_raw_emoji)
        diag_result = run_emoji_tokenizer_diagnostic(tokenizer, diag_df, n_samples=200)
        diag_path = output_dir / "exp2_raw_emoji_no_branch" / "emoji_tokenizer_diagnostic.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(diag_result, f, indent=2)
        del diag_df
        gc.collect()

        exp2_results = run_experiment(
            experiment_name="exp2_raw_emoji_no_branch",
            model_cls=BaselineModel,
            model_kwargs={"model_name": args.model_name, "num_labels": NUM_LABELS, "dropout": args.dropout},
            clean_text_fn=clean_text_keep_raw_emoji,
            data_dir=data_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            seeds=seeds,
            **train_kwargs,
        )
        all_results["exp2"] = exp2_results

    # ======================================================================
    # BOOTSTRAP SIGNIFICANCE TESTING
    # ======================================================================
    logger.info("=" * 60)
    logger.info("BOOTSTRAP SIGNIFICANCE TESTING (B=10,000)")
    logger.info("=" * 60)

    bootstrap_results = {}

    # Load A0 ensemble probs + thresholds
    a0_dir = output_dir / "a0_baseline_wbce"
    if a0_dir.exists():
        a0_ens_probs = np.load(a0_dir / "ensemble_probs_test.npy")
        a0_ens_thresholds = np.load(a0_dir / "ensemble_thresholds.npy")
        test_targets = np.load(a0_dir / f"seed_{seeds[0]}" / "targets_test.npy")

        # Exp1 vs A0
        exp1_dir = output_dir / "exp1_emoji_branch_wbce"
        if exp1_dir.exists() and (exp1_dir / "ensemble_probs_test.npy").exists():
            exp1_ens_probs = np.load(exp1_dir / "ensemble_probs_test.npy")
            exp1_ens_thresholds = np.load(exp1_dir / "ensemble_thresholds.npy")
            bs_exp1 = paired_bootstrap_test(
                a0_ens_probs, exp1_ens_probs, test_targets,
                a0_ens_thresholds, exp1_ens_thresholds,
                n_bootstrap=10000, seed=42,
            )
            bootstrap_results["exp1_vs_a0"] = bs_exp1
            logger.info("Exp1 vs A0 bootstrap: delta=%.4f, p=%.4f, CI=[%.4f, %.4f]",
                       bs_exp1["mean_delta"], bs_exp1["p_value"],
                       bs_exp1["ci_95_low"], bs_exp1["ci_95_high"])

        # Exp2 vs A0
        exp2_dir = output_dir / "exp2_raw_emoji_no_branch"
        if exp2_dir.exists() and (exp2_dir / "ensemble_probs_test.npy").exists():
            exp2_ens_probs = np.load(exp2_dir / "ensemble_probs_test.npy")
            exp2_ens_thresholds = np.load(exp2_dir / "ensemble_thresholds.npy")
            bs_exp2 = paired_bootstrap_test(
                a0_ens_probs, exp2_ens_probs, test_targets,
                a0_ens_thresholds, exp2_ens_thresholds,
                n_bootstrap=10000, seed=42,
            )
            bootstrap_results["exp2_vs_a0"] = bs_exp2
            logger.info("Exp2 vs A0 bootstrap: delta=%.4f, p=%.4f, CI=[%.4f, %.4f]",
                       bs_exp2["mean_delta"], bs_exp2["p_value"],
                       bs_exp2["ci_95_low"], bs_exp2["ci_95_high"])

    with open(output_dir / "bootstrap_results.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    # ======================================================================
    # CONSOLIDATED COMPARISON TABLE
    # ======================================================================
    logger.info("=" * 60)
    logger.info("CONSOLIDATED COMPARISON TABLE")
    logger.info("=" * 60)

    def _get_seed_mf1(results: dict, seed: int) -> float:
        return results["seeds"][seed]["test"]["mf1_opt"]

    def _get_ens_metrics(results: dict) -> dict:
        return results["ensemble"]

    rows = []

    # A0 row
    if "a0" in all_results:
        r = all_results["a0"]
        mf1s = [_get_seed_mf1(r, s) for s in seeds]
        row = {
            "System": "A0 (WBCE, norm text, no emoji branch)",
            **{f"Seed {s} MF1-opt": _get_seed_mf1(r, s) for s in seeds},
            "Mean MF1-opt": np.mean(mf1s),
            "Std MF1-opt": np.std(mf1s),
            "Ens MF1-opt": r["ensemble"]["mf1_opt"],
            "Ens MF1-fix": r["ensemble"]["mf1_fix"],
            "Ens MicroF1": r["ensemble"]["micro_f1"],
            "Ens mAP": r["ensemble"]["map"],
            "Bootstrap delta vs A0": "-",
            "p-value": "-",
            "95% CI": "-",
        }
        rows.append(row)

    # Exp1 row
    if "exp1" in all_results:
        r = all_results["exp1"]
        mf1s = [_get_seed_mf1(r, s) for s in seeds]
        bs = bootstrap_results.get("exp1_vs_a0", {})
        row = {
            "System": "Exp1 (WBCE + emoji branch)",
            **{f"Seed {s} MF1-opt": _get_seed_mf1(r, s) for s in seeds},
            "Mean MF1-opt": np.mean(mf1s),
            "Std MF1-opt": np.std(mf1s),
            "Ens MF1-opt": r["ensemble"]["mf1_opt"],
            "Ens MF1-fix": r["ensemble"]["mf1_fix"],
            "Ens MicroF1": r["ensemble"]["micro_f1"],
            "Ens mAP": r["ensemble"]["map"],
            "Bootstrap delta vs A0": bs.get("mean_delta", "-"),
            "p-value": bs.get("p_value", "-"),
            "95% CI": f"[{bs.get('ci_95_low', '-')}, {bs.get('ci_95_high', '-')}]" if bs else "-",
        }
        rows.append(row)

    # Exp2 row
    if "exp2" in all_results:
        r = all_results["exp2"]
        mf1s = [_get_seed_mf1(r, s) for s in seeds]
        bs = bootstrap_results.get("exp2_vs_a0", {})
        row = {
            "System": "Exp2 (WBCE + raw emoji, no branch)",
            **{f"Seed {s} MF1-opt": _get_seed_mf1(r, s) for s in seeds},
            "Mean MF1-opt": np.mean(mf1s),
            "Std MF1-opt": np.std(mf1s),
            "Ens MF1-opt": r["ensemble"]["mf1_opt"],
            "Ens MF1-fix": r["ensemble"]["mf1_fix"],
            "Ens MicroF1": r["ensemble"]["micro_f1"],
            "Ens mAP": r["ensemble"]["map"],
            "Bootstrap delta vs A0": bs.get("mean_delta", "-"),
            "p-value": bs.get("p_value", "-"),
            "95% CI": f"[{bs.get('ci_95_low', '-')}, {bs.get('ci_95_high', '-')}]" if bs else "-",
        }
        rows.append(row)

    # Historical reference
    rows.append({
        "System": "EmoViS-Ens (historical, NOT reproduced)",
        **{f"Seed {s} MF1-opt": "-" for s in seeds},
        "Mean MF1-opt": 0.6329,
        "Std MF1-opt": "-",
        "Ens MF1-opt": 0.6329,
        "Ens MF1-fix": "-",
        "Ens MicroF1": "-",
        "Ens mAP": "-",
        "Bootstrap delta vs A0": "-",
        "p-value": "-",
        "95% CI": "-",
    })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "final_comparison_table.csv", index=False)

    # Print the table
    print("\n" + "=" * 120)
    print("FINAL CROSS-EXPERIMENT COMPARISON TABLE")
    print("=" * 120)
    print(comparison_df.to_string(index=False))
    print("=" * 120)

    # ======================================================================
    # EXPLICIT DELTA REPORTING
    # ======================================================================
    print("\n" + "=" * 80)
    print("DETAILED DELTA ANALYSIS")
    print("=" * 80)

    if "a0" in all_results:
        a0_mean_mf1 = np.mean([_get_seed_mf1(all_results["a0"], s) for s in seeds])

        if "exp1" in all_results:
            exp1_mean_mf1 = np.mean([_get_seed_mf1(all_results["exp1"], s) for s in seeds])
            mean_delta = exp1_mean_mf1 - a0_mean_mf1
            bs = bootstrap_results.get("exp1_vs_a0", {})
            print(f"\nExp1 vs A0:")
            print(f"  (a) Difference between mean-of-3-seeds MF1-opt:  {mean_delta:+.4f}")
            print(f"      A0 mean={a0_mean_mf1:.4f}, Exp1 mean={exp1_mean_mf1:.4f}")
            if bs:
                print(f"  (b) Bootstrap delta (from 3-seed-averaged probs): {bs['mean_delta']:+.4f}")
                print(f"      p-value={bs['p_value']:.4f}, 95% CI=[{bs['ci_95_low']:+.4f}, {bs['ci_95_high']:+.4f}]")
            print(f"  NOTE: (a) and (b) are DIFFERENT quantities.")
            print(f"        (a) averages per-seed metrics; (b) averages probabilities then computes metric.")

        if "exp2" in all_results:
            exp2_mean_mf1 = np.mean([_get_seed_mf1(all_results["exp2"], s) for s in seeds])
            mean_delta = exp2_mean_mf1 - a0_mean_mf1
            bs = bootstrap_results.get("exp2_vs_a0", {})
            print(f"\nExp2 vs A0:")
            print(f"  (a) Difference between mean-of-3-seeds MF1-opt:  {mean_delta:+.4f}")
            print(f"      A0 mean={a0_mean_mf1:.4f}, Exp2 mean={exp2_mean_mf1:.4f}")
            if bs:
                print(f"  (b) Bootstrap delta (from 3-seed-averaged probs): {bs['mean_delta']:+.4f}")
                print(f"      p-value={bs['p_value']:.4f}, 95% CI=[{bs['ci_95_low']:+.4f}, {bs['ci_95_high']:+.4f}]")
            print(f"  NOTE: (a) and (b) are DIFFERENT quantities.")
            print(f"        (a) averages per-seed metrics; (b) averages probabilities then computes metric.")

    # ======================================================================
    # STATISTICAL SIGNIFICANCE CONCLUSIONS
    # ======================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE CONCLUSIONS")
    print("=" * 80)

    sig_results = []
    for exp_name, bs_key in [("Exp1", "exp1_vs_a0"), ("Exp2", "exp2_vs_a0")]:
        bs = bootstrap_results.get(bs_key, {})
        if bs:
            sig = bs["p_value"] < 0.05
            sig_results.append((exp_name, bs["mean_delta"], bs["p_value"], sig))
            status = "SIGNIFICANT (p < 0.05)" if sig else "NOT significant (p >= 0.05)"
            direction = "improvement" if bs["mean_delta"] > 0 else "degradation"
            print(f"  {exp_name} vs A0: {status}, delta={bs['mean_delta']:+.4f} ({direction})")
            print(f"    p-value={bs['p_value']:.4f}, 95% CI=[{bs['ci_95_low']:+.4f}, {bs['ci_95_high']:+.4f}]")

    # Compare the two if both significant improvements
    sig_improvements = [(n, d, p) for n, d, p, s in sig_results if s and d > 0]
    if len(sig_improvements) == 2:
        better = max(sig_improvements, key=lambda x: x[1])
        worse = min(sig_improvements, key=lambda x: x[1])
        print(f"\n  BOTH experiments show significant improvement over A0.")
        print(f"  {better[0]} improves MORE by {better[1] - worse[1]:+.4f} delta")
    elif len(sig_improvements) == 1:
        print(f"\n  Only {sig_improvements[0][0]} shows significant improvement over A0.")
    else:
        print(f"\n  Neither experiment shows a statistically significant improvement over A0.")

    # ======================================================================
    # ZIP ARTIFACTS
    # ======================================================================
    zip_path = output_dir / "emovis_decisive_experiments.zip"
    logger.info("Zipping artifacts to %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                filepath = Path(root) / file
                if filepath == zip_path:
                    continue  # Don't zip the zip itself
                arcname = filepath.relative_to(output_dir)
                zf.write(filepath, arcname)
    logger.info("Artifacts zipped: %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    # Save pip freeze
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(output_dir / "pip_freeze.txt", "w") as f:
            f.write(pip_freeze)
    except Exception:
        logger.warning("Could not save pip freeze output")

    logger.info("All experiments complete. Artifacts saved to: %s", output_dir)
    return all_results


if __name__ == "__main__":
    main()
