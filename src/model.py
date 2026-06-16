"""Multi-label classifier on top of a HuggingFace encoder (default: ViSoBERT) with optional Emoji2Vec dual-encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pooled: torch.Tensor


class EmojiEncoder(nn.Module):
    """Encodes a sequence of emojis using pretrained emoji2vec embeddings."""

    def __init__(self, e2v, dim: int = 300) -> None:
        super().__init__()
        self.dim = dim
        self.emoji_dict = {}
        if e2v is not None:
            # e2v is a Gensim KeyedVectors object
            for word in e2v.index_to_key:
                self.emoji_dict[word] = torch.tensor(e2v[word], dtype=torch.float32)

    def forward(self, emoji_ids: list[list[str]]) -> torch.Tensor:
        # Determine device dynamically from model parameters or fallback
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_vectors = []
        for sample_emojis in emoji_ids:
            sample_vectors = []
            for emo in sample_emojis:
                if emo == "":  # Skip padding
                    continue
                if emo in self.emoji_dict:
                    sample_vectors.append(self.emoji_dict[emo].to(device))
                else:
                    sample_vectors.append(torch.zeros(self.dim, device=device))
            if not sample_vectors:
                batch_vectors.append(torch.zeros(self.dim, device=device))
            else:
                batch_vectors.append(torch.stack(sample_vectors).mean(dim=0))
        return torch.stack(batch_vectors)


class ViSoBertMultiLabel(nn.Module):
    """ViSoBERT (or any HF encoder) + optional Emoji2Vec branch + dropout + linear head for multi-label."""

    def __init__(
        self,
        model_name: str = "uitnlp/visobert",
        num_labels: int = 28,
        dropout: float = 0.1,
        use_emoji_branch: bool = False,
        e2v: any = None,
        emoji_dim: int = 300,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_emoji_branch = use_emoji_branch

        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size from config of {model_name}")
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)

        if self.use_emoji_branch:
            self.emoji_encoder = EmojiEncoder(e2v, dim=emoji_dim)
            self.classifier = nn.Linear(hidden_size + emoji_dim, num_labels)
        else:
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
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emoji_ids: list[list[str]] | None = None,
    ) -> ModelOutput:
        h_cls = self.encode(input_ids, attention_mask)
        
        if self.use_emoji_branch:
            if emoji_ids is None:
                # Fallback to zero vectors if not provided
                device = h_cls.device
                h_emoji = torch.zeros(h_cls.size(0), self.emoji_encoder.dim, device=device)
            else:
                h_emoji = self.emoji_encoder(emoji_ids)
            feat = torch.cat([h_cls, h_emoji], dim=1)
        else:
            feat = h_cls
            
        logits = self.classifier(self.dropout(feat))
        return ModelOutput(logits=logits, pooled=h_cls)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
