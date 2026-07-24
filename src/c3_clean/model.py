"""Canonical mean-pooled ViSoBERT models for controlled C3 experiments."""
from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class C3ModelOutput(NamedTuple):
    logits: torch.Tensor
    pooled_text: torch.Tensor
    projected_emoji: torch.Tensor | None


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * mask).sum(dim=1) / denominator


class MeanPooledTextViSoBERT(nn.Module):
    """Text-only controlled baseline with non-padding masked mean pooling."""

    def __init__(
        self,
        model_name: str = "uitnlp/visobert",
        num_labels: int = 28,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = int(config.hidden_size)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> C3ModelOutput:
        encoded = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = masked_mean_pool(encoded.last_hidden_state, attention_mask)
        return C3ModelOutput(self.classifier(pooled), pooled, None)


class EmojiAwareViSoBERT(nn.Module):
    """The simple dual-branch architecture expected for the canonical C3 model."""

    def __init__(
        self,
        model_name: str = "uitnlp/visobert",
        num_labels: int = 28,
        emoji_dim: int = 300,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.text_encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = int(config.hidden_size)
        self.emoji_projection = nn.Sequential(
            nn.Linear(emoji_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emoji_vectors: torch.Tensor,
        **_: Any,
    ) -> C3ModelOutput:
        encoded = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled_text = masked_mean_pool(encoded.last_hidden_state, attention_mask)
        projected_emoji = self.emoji_projection(emoji_vectors.float())
        logits = self.fusion(torch.cat((pooled_text, projected_emoji), dim=1))
        return C3ModelOutput(logits, pooled_text, projected_emoji)


def build_model(experiment_name: str, model_config: Mapping[str, Any]) -> nn.Module:
    common = {
        "model_name": model_config["model_name"],
        "num_labels": int(model_config["num_labels"]),
        "dropout": float(model_config["dropout"]),
    }
    if experiment_name in {"A0_controlled_text_BCE", "A1_controlled_text_ASL"}:
        return MeanPooledTextViSoBERT(**common)
    if experiment_name in {
        "A2_controlled_ASL_Emoji",
        "A3_controlled_ASL_Emoji_CB",
        "Emoji-random-control",
        "Emoji-shuffle-control",
        "Emoji-zero-control",
        "C3-RDrop",
        "C3-extended-matched",
        "A2_RDrop_03",
        "A2_RDrop_10",
    }:
        return EmojiAwareViSoBERT(
            **common,
            emoji_dim=int(model_config.get("emoji_dim", 300)),
        )
    raise ValueError(f"Unknown experiment: {experiment_name}")


def _clean_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        (key.removeprefix("module.").removeprefix("model.")): value
        for key, value in state_dict.items()
    }


def verify_canonical_state_dict(
    state_dict: Mapping[str, Any],
    *,
    hidden_size: int = 768,
    emoji_dim: int = 300,
    num_labels: int = 28,
) -> dict[str, Any]:
    """Check head keys/shapes that distinguish the simple C3 architecture."""
    cleaned = _clean_state_dict(state_dict)
    expected_shapes = {
        "emoji_projection.0.weight": (hidden_size, emoji_dim),
        "emoji_projection.0.bias": (hidden_size,),
        "emoji_projection.2.weight": (hidden_size,),
        "emoji_projection.2.bias": (hidden_size,),
        "fusion.0.weight": (hidden_size, hidden_size * 2),
        "fusion.0.bias": (hidden_size,),
        "fusion.3.weight": (num_labels, hidden_size),
        "fusion.3.bias": (num_labels,),
    }
    actual_shapes = {
        key: tuple(value.shape) for key, value in cleaned.items() if hasattr(value, "shape")
    }
    missing = [key for key in expected_shapes if key not in actual_shapes]
    mismatched = {
        key: {"expected": list(shape), "actual": list(actual_shapes[key])}
        for key, shape in expected_shapes.items()
        if key in actual_shapes and actual_shapes[key] != shape
    }
    forbidden_fragments = ("label_graph", "labelgraph", "cross_attention", "emoji_attn", "gat")
    forbidden_keys = sorted(
        key for key in cleaned if any(fragment in key.lower() for fragment in forbidden_fragments)
    )
    encoder_prefixes = sorted(
        {key.split(".", 1)[0] for key in cleaned if key.endswith("embeddings.word_embeddings.weight")}
    )
    passed = not missing and not mismatched and not forbidden_keys and bool(encoder_prefixes)
    return {
        "passed": passed,
        "expected_head_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "missing_keys": missing,
        "mismatched_shapes": mismatched,
        "forbidden_graph_or_attention_keys": forbidden_keys,
        "detected_encoder_prefixes": encoder_prefixes,
        "state_dict_key_count": len(cleaned),
    }


__all__ = [
    "C3ModelOutput",
    "EmojiAwareViSoBERT",
    "MeanPooledTextViSoBERT",
    "build_model",
    "masked_mean_pool",
    "verify_canonical_state_dict",
]
