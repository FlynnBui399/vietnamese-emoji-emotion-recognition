"""YAML-backed training configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_name: str = "uitnlp/visobert"
    num_labels: int = 28
    max_length: int = 128
    batch_size: int = 48
    eval_batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 1
    seed: int = 42
    use_pos_weight: bool = True
    threshold: float = 0.5
    num_workers: int = 2
    log_every: int = 50

    data_dir: str = "/data"
    runs_dir: str = "/runs"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
