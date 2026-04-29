"""Data loading for ViGoEmotions multi-label classification.

CSV format: id,text,labels   where `labels` is a stringified list of int indices,
e.g. "[2, 8, 3]". Each comment can carry multiple of the 28 labels (27 emotions +
neutral at index 27, per data/vigoemotions/README.md).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import NUM_LABELS, get_logger

LOGGER = get_logger(__name__)


def _parse_label_cell(cell: object) -> list[int]:
    """Parse the `labels` column. Robust to NaN, whitespace, and missing brackets."""
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
    if isinstance(value, Iterable):
        return [int(v) for v in value]
    return []


def _to_multi_hot(label_ids: list[int], num_labels: int) -> np.ndarray:
    vec = np.zeros(num_labels, dtype=np.float32)
    for idx in label_ids:
        if 0 <= idx < num_labels:
            vec[idx] = 1.0
    return vec


def load_split(csv_path: str | Path, num_labels: int = NUM_LABELS) -> pd.DataFrame:
    """Load a CSV split into a DataFrame with `text` (str) and `labels` (np.ndarray)."""
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'text' and 'labels' columns; got {df.columns.tolist()}")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["labels"] = df["labels"].apply(_parse_label_cell)
    df["multi_hot"] = df["labels"].apply(lambda ids: _to_multi_hot(ids, num_labels))
    LOGGER.info("Loaded %s rows from %s", len(df), csv_path)
    return df


@dataclass
class TokenizedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ViGoEmotionsDataset(Dataset):
    """Tokenizes on the fly so we can keep memory low and respect dynamic padding later."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.texts: list[str] = df["text"].tolist()
        self.labels: np.ndarray = np.stack(df["multi_hot"].to_list(), axis=0)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.from_numpy(self.labels[idx]),
        }


def compute_pos_weight(multi_hot_matrix: np.ndarray, eps: float = 1.0) -> torch.Tensor:
    """pos_weight[c] = (N - n_pos[c]) / max(n_pos[c], eps).

    This is the standard formula used by `BCEWithLogitsLoss` for class imbalance,
    matching the ViGoEmotions baseline setup.
    """
    n_pos = multi_hot_matrix.sum(axis=0)
    n_neg = multi_hot_matrix.shape[0] - n_pos
    pos_weight = n_neg / np.maximum(n_pos, eps)
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_dataloaders(
    data_dir: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    num_labels: int = NUM_LABELS,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, dict[str, pd.DataFrame]]:
    data_dir = Path(data_dir)
    train_df = load_split(data_dir / "train.csv", num_labels)
    val_df = load_split(data_dir / "val.csv", num_labels)
    test_df = load_split(data_dir / "test.csv", num_labels)

    train_ds = ViGoEmotionsDataset(train_df, tokenizer, max_length)
    val_ds = ViGoEmotionsDataset(val_df, tokenizer, max_length)
    test_ds = ViGoEmotionsDataset(test_df, tokenizer, max_length)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    pos_weight = compute_pos_weight(train_ds.labels)

    raw = {"train": train_df, "val": val_df, "test": test_df}
    return train_loader, val_loader, test_loader, pos_weight, raw
