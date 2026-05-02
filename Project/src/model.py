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
        # Use nn.Linear's default initialization (Kaiming-uniform for weight,
        # uniform for bias) to match the ViGoEmotions baseline notebook, which
        # does not override the classifier head init.
        self.classifier = nn.Linear(hidden_size, num_labels)

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
        # Match ViGoEmotions baseline: feed the encoder's pooler_output
        # (Linear+Tanh on [CLS]) into the classifier head, not last_hidden_state[:, 0].
        pooled = outputs.pooler_output
        if pooled is None:
            raise RuntimeError(
                f"{self.model_name} did not return a pooler_output. "
                "Reload the backbone with add_pooling_layer=True or update the head."
            )
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
