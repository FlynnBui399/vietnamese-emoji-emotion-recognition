"""Loss functions for ViGoEmotions multi-label classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Implements the formulation requested from Ridnik et al. (2021):
    positive loss uses p = sigmoid(logits), while negative loss uses
    p_m = max(p - clip, 0) to down-weight easy negatives more aggressively.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"logits shape {tuple(x.shape)} != targets shape {tuple(y.shape)}")

        y = y.type_as(x)
        p = torch.sigmoid(x)

        p_m = torch.clamp(p - self.clip, min=0.0)

        log_pos = torch.log(torch.clamp(p, min=self.eps))
        log_neg = torch.log(torch.clamp(1.0 - p_m, min=self.eps))
        pos_loss = -y * torch.pow(1.0 - p, self.gamma_pos) * log_pos
        neg_loss = -(1.0 - y) * torch.pow(p_m, self.gamma_neg) * log_neg
        loss = pos_loss + neg_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_bce_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Standard `BCEWithLogitsLoss`. Pass `pos_weight` (per-class) to handle imbalance.

    pos_weight should have shape `(num_labels,)`. The Module-form loss reduces
    over the batch with `mean`, matching the ViGoEmotions paper's setup.
    """
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    if pos_weight.dim() != 1:
        raise ValueError(f"pos_weight must be 1-D, got shape {tuple(pos_weight.shape)}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class ClusteringContrastiveLoss(nn.Module):
    def __init__(self, tau: float = 0.07, margin: float = 0.3):
        super().__init__()
        self.tau = tau
        self.margin = margin

        # Clusters mapping
        self.label_to_cluster = {
            8: 0, 0: 0, 1: 0, 2: 0, 3: 0, 7: 0, 9: 0, 10: 0, 5: 0, 6: 0, # positive_high
            11: 1, 4: 1, # positive_low
            24: 2, 25: 2, 23: 2, 26: 2, 19: 2, 16: 2, # negative_high
            20: 3, 21: 3, 18: 3, 22: 3, # negative_low
            15: 4, 14: 4, 12: 4, 13: 4, # cognitive
            27: 5 # neutral
        }

        # Create mapping matrix M of shape [28, 6]
        M = torch.zeros(28, 6)
        for label_idx, cluster_idx in self.label_to_cluster.items():
            M[label_idx, cluster_idx] = 1.0
        self.register_buffer("M", M)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: L2-normalized CLS embeddings [B, 768]
        # y: targets [B, 28]
        B = z.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=z.device)

        # Sample-to-cluster matrix: [B, 28] @ [28, 6] -> [B, 6]
        S = torch.matmul(y, self.M)
        S_bin = (S > 0).float()

        # Shared cluster matrix: [B, B]
        shared = torch.matmul(S_bin, S_bin.T)
        shared_mask = (shared > 0).float()

        # Identity mask to exclude self-comparison
        identity_mask = torch.eye(B, device=z.device)

        # Positive pairs mask: i != j and sharing at least one cluster
        pos_mask = shared_mask * (1.0 - identity_mask)

        # Negative pairs mask: i != j and sharing no cluster
        neg_mask = (1.0 - shared_mask) * (1.0 - identity_mask)

        # Cosine similarity matrix: [B, B]
        similarity_matrix = torch.matmul(z, z.T)

        # L_pos computation
        logits_sim = similarity_matrix / self.tau
        log_sum_k = torch.logsumexp(logits_sim, dim=-1, keepdim=True)
        L_pos_matrix = log_sum_k - logits_sim

        pos_losses = L_pos_matrix[pos_mask.bool()]

        # L_neg computation
        L_neg_matrix = torch.clamp(similarity_matrix - self.margin, min=0.0)
        neg_losses = L_neg_matrix[neg_mask.bool()]

        mean_pos = torch.mean(pos_losses) if pos_losses.numel() > 0 else torch.tensor(0.0, device=z.device)
        mean_neg = torch.mean(neg_losses) if neg_losses.numel() > 0 else torch.tensor(0.0, device=z.device)

        return mean_pos + mean_neg


class LabelDescriptionLoss(nn.Module):
    def __init__(self, backbone: nn.Module, tokenizer: any, device: torch.device, tau: float = 0.07):
        super().__init__()
        self.tau = tau

        # Descriptions in the correct order of EMOTION_LABELS
        descriptions = [
            "amusement: cảm xúc vui vẻ, giải trí và gây cười nhẹ nhàng",
            "excitement: cảm giác hào hứng, phấn khích trước điều gì đó thú vị",
            "joy: niềm vui sướng, hạnh phúc tràn ngập năng lượng tích cực",
            "love: tình yêu thương và sự gắn bó sâu sắc dành cho người khác",
            "desire: khao khát hoặc thèm muốn có được một điều gì đó",
            "optimism: sự lạc quan và niềm tin vào tương lai tươi sáng",
            "caring: sự quan tâm, chăm sóc và lo lắng cho người khác",
            "pride: lòng tự hào và sự tự tôn về bản thân hoặc thành tựu",
            "admiration: sự ngưỡng mộ, kính trọng và đánh giá cao người khác",
            "gratitude: sự biết ơn sâu sắc trước sự giúp đỡ hay lòng tốt",
            "relief: cảm giác nhẹ nhõm khi trút bỏ được lo lắng căng thẳng",
            "approval: sự tán thành, đồng ý và ủng hộ ý kiến hành động",
            "realization: sự nhận ra, thấu hiểu hoặc phát hiện mới mẻ",
            "surprise: sự ngạc nhiên, bất ngờ trước việc không ngờ tới",
            "curiosity: sự tò mò, ham học hỏi và tìm hiểu mọi thứ",
            "confusion: sự bối rối, hoang mang chưa hiểu rõ vấn đề",
            "fear: nỗi sợ hãi, lo lắng trước mối nguy hiểm đe dọa",
            "nervousness: sự lo lắng, bồn chồn trước một sự kiện sắp diễn ra",
            "remorse: sự hối hận, ăn năn tự trách về lỗi lầm đã qua",
            "embarrassment: sự ngượng ngùng, xấu hổ trước mặt người khác",
            "disappointment: sự thất vọng khi kết quả không như ý muốn",
            "sadness: nỗi buồn bã, u sầu và chán nản trong lòng",
            "grief: sự đau buồn sâu sắc trước mất mát lớn lao",
            "disgust: sự ghê tởm, khó chịu trước điều bẩn thỉu xấu xa",
            "anger: sự tức giận, giận dữ phẫn nộ mạnh mẽ",
            "annoyance: sự khó chịu, bực mình vì phiền toái nhỏ nhặt",
            "disapproval: sự phản đối, không chấp nhận một hành vi nào đó",
            "neutral: trạng thái trung tính bình thường không có cảm xúc đặc biệt"
        ]

        backbone.eval()
        with torch.no_grad():
            inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = backbone(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            if getattr(outputs, "pooler_output", None) is not None:
                d_k = outputs.pooler_output
            else:
                d_k = outputs.last_hidden_state[:, 0]
            d_k = torch.nn.functional.normalize(d_k, p=2, dim=-1)
            self.register_buffer("d_k", d_k)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: L2-normalized CLS embeddings [B, 768]
        # y: targets [B, 28]
        sim_matrix = torch.matmul(z, self.d_k.T)
        scores = sim_matrix / self.tau
        log_sum_j = torch.logsumexp(scores, dim=-1, keepdim=True)
        log_probs = scores - log_sum_j

        ldl_i = -torch.sum(y * log_probs, dim=-1)
        return torch.mean(ldl_i)
