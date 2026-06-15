"""Emoji-aware ViSoBERT model and Emoji2Vec helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import emoji
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class EmojiModelOutput:
    logits: torch.Tensor
    pooled_text: torch.Tensor
    pooled_emoji: torch.Tensor


def extract_emojis(text: str) -> list[str]:
    return [ch for ch in str(text) if ch in emoji.EMOJI_DATA]


def load_emoji2vec(path: str | Path = "emoji2vec.bin"):
    from gensim.models import KeyedVectors

    return KeyedVectors.load_word2vec_format(str(path), binary=True)


def get_emoji_vector(emojis: list[str], e2v, dim: int = 300) -> np.ndarray:
    vectors = []
    for item in emojis:
        if item in e2v:
            vectors.append(np.asarray(e2v[item], dtype=np.float32))
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


class EmojiAwareViSoBERT(nn.Module):
    def __init__(
        self,
        model_name: str = "uitnlp/visobert",
        num_labels: int = 28,
        emoji_dim: int = 300,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.emoji_dim = emoji_dim

        config = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size from config of {model_name}")
        self.hidden_size = hidden_size

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

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emoji_vectors: torch.Tensor,
    ) -> EmojiModelOutput:
        h_text = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        h_emoji = self.emoji_projection(emoji_vectors.float())
        # Gate: zero out projected emoji for samples with no emoji input.
        # ~75% of samples have no emoji; without gating, the projection
        # layers (Linear bias + LayerNorm) turn zero vectors into non-zero
        # noise that pollutes the fusion for non-emoji samples.
        has_emoji = (emoji_vectors.abs().sum(dim=-1, keepdim=True) > 0).float()
        h_emoji_gated = h_emoji * has_emoji
        logits = self.fusion(torch.cat([h_text, h_emoji_gated], dim=1))
        return logits
