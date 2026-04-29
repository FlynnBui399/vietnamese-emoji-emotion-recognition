"""Shared utilities: seeding, label maps, lightweight logger setup."""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


EMOTION_LABELS: tuple[str, ...] = (
    "amusement",
    "excitement",
    "joy",
    "love",
    "desire",
    "optimism",
    "caring",
    "pride",
    "admiration",
    "gratitude",
    "relief",
    "approval",
    "realization",
    "surprise",
    "curiosity",
    "confusion",
    "fear",
    "nervousness",
    "remorse",
    "embarrassment",
    "disappointment",
    "sadness",
    "grief",
    "disgust",
    "anger",
    "annoyance",
    "disapproval",
    "neutral",
)
NUM_LABELS = len(EMOTION_LABELS)
LABEL2ID = {name: i for i, name in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(EMOTION_LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "vigoemotions", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def device_info() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    return f"cuda ({name}, sm_{cap[0]}{cap[1]})"
