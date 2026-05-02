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
    max_length: int = 200
    batch_size: int = 32    
    eval_batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # When set, overrides `warmup_ratio`: warmup_steps = warmup_epochs * steps_per_epoch.
    # The ViGoEmotions baseline uses warmup_epochs=1.0 (≈ len(train_loader) steps).
    warmup_epochs: float | None = None
    dropout: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 1
    seed: int = 42
    use_pos_weight: bool = True
    threshold: float = 0.5
    num_workers: int = 2
    log_every: int = 50

    # Mixed precision on CUDA (bf16 if supported else fp16). Set false for full fp32 (notebook-like).
    use_amp: bool = True

    # Tokenizer / preprocessing.
    use_fast_tokenizer: bool = False
    apply_clean_text: bool = True
    apply_pyvi: bool = True  # baseline applies pyvi.ViTokenizer.tokenize before SP tokenizer
    docs_dir: str = "docs"

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
