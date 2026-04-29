"""Multi-label classifier on top of a HuggingFace encoder (default: ViSoBERT)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pooled: torch.Tensor


class ViSoBertMultiLabel(nn.Module):
    """ViSoBERT (or any HF encoder) + dropout + linear head for multi-label.

    Phase 2 can subclass this and add an emoji branch + fusion before the
    classifier head without touching the rest of the training code.
    """

    def __init__(
        self,
        model_name: str = "uitnlp/visobert",
        num_labels: int = 28,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size from config of {model_name}")
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = outputs.last_hidden_state[:, 0]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ModelOutput:
        pooled = self.encode(input_ids, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return ModelOutput(logits=logits, pooled=pooled)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
