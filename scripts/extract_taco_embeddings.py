"""
Standalone TACO Cosine Stats extraction script.
No imports from src/ -- fully self-contained for Kaggle execution.
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ---- Model definition (copied from src/c3_clean/model.py) ----
def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * mask).sum(dim=1) / denominator


class EmojiAwareViSoBERT(nn.Module):
    def __init__(self, model_name="uitnlp/visobert", num_labels=28, emoji_dim=300, dropout=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        self.text_encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = int(config.hidden_size)
        self.emoji_projection = nn.Sequential(
            nn.Linear(emoji_dim, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, emoji_vectors, **_):
        encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_text = masked_mean_pool(encoded.last_hidden_state, attention_mask)
        projected_emoji = self.emoji_projection(emoji_vectors.float())
        logits = self.fusion(torch.cat((pooled_text, projected_emoji), dim=1))
        return logits, pooled_text, projected_emoji


# ---- Main ----
def main():
    repo_root = Path(__file__).resolve().parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Find checkpoint
    print("Searching for A2 seed42 checkpoint...")
    ckpt_path = None
    search_roots = [Path('/kaggle/input'), Path('/kaggle/working'), repo_root / 'outputs']
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob('best_checkpoint.pt'):
            pstr = str(p)
            if 'A2_controlled_ASL_Emoji' in pstr and 'seed42' in pstr:
                ckpt_path = p
                break
        if ckpt_path:
            break

    if not ckpt_path:
        print("ERROR: best_checkpoint.pt for A2_controlled_ASL_Emoji/seed42 not found!")
        print("Searched in:", [str(r) for r in search_roots])
        sys.exit(1)
    print(f"Found checkpoint: {ckpt_path}")

    # 2. Load model
    model = EmojiAwareViSoBERT(model_name="uitnlp/visobert", num_labels=28, emoji_dim=300, dropout=0.2)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Clean state dict (remove "module." prefix if DataParallel was used)
    state_dict = {k.removeprefix("module.").removeprefix("model."): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # 3. Load test data
    data_candidates = [
        Path('/kaggle/input/c3-clean-inputs/data/vigoemotions'),
        Path('/kaggle/input/vigoemotions/data/vigoemotions'),
        Path('/kaggle/input/vigoemotions'),
        repo_root / 'data' / 'vigoemotions',
    ]
    test_csv = None
    for d in data_candidates:
        candidate = d / 'test.csv'
        if candidate.exists():
            test_csv = candidate
            break
    if not test_csv:
        print("ERROR: test.csv not found!")
        sys.exit(1)
    print(f"Loading test data from: {test_csv}")

    df_test = pd.read_csv(test_csv)
    label_cols = [c for c in df_test.columns if c not in ('id', 'text')]
    targets = df_test[label_cols].values.astype(np.float32)
    texts = df_test['text'].tolist()

    # 4. Tokenize and create dummy emoji vectors (we only care about h_text)
    tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert", use_fast=False)
    BATCH_SIZE = 32
    all_h_text = []

    print(f"Extracting embeddings from {len(texts)} test samples...")
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts, truncation=True, padding="max_length",
                max_length=128, return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            # Zero emoji vectors (300d) -- we only need h_text, not emoji projection
            emoji_vectors = torch.zeros(len(batch_texts), 300, device=device)

            _, h_text, _ = model(input_ids, attention_mask, emoji_vectors)
            all_h_text.append(h_text.cpu().numpy())

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  Processed {start + len(batch_texts)}/{len(texts)}")

    embeddings = np.concatenate(all_h_text, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # 5. TACO Clusters
    taco_clusters = {
        0: [8, 0, 1, 2, 3, 7, 9, 10, 5, 6],
        1: [11, 4],
        2: [24, 25, 23, 26, 19, 16],
        3: [20, 21, 18, 22],
        4: [15, 14, 12, 13],
        5: [27]
    }

    # Find active clusters per sample
    sample_clusters = []
    for i in range(len(targets)):
        active = set()
        for label_idx in np.where(targets[i] == 1)[0]:
            for cid, labels in taco_clusters.items():
                if label_idx in labels:
                    active.add(cid)
        sample_clusters.append(active)

    # 6. Compute cosine distance matrix
    print("Computing cosine distance matrix...")
    dist_matrix = squareform(pdist(embeddings, metric='cosine'))

    # 7. Collect positive pair distances (sharing >= 1 cluster)
    n = len(targets)
    positive_distances = []
    for i in range(n):
        if not sample_clusters[i]:
            continue
        for j in range(i + 1, n):
            if sample_clusters[i].intersection(sample_clusters[j]):
                positive_distances.append(dist_matrix[i, j])

    d_arr = np.array(positive_distances)

    stats = {
        "n_positive_pairs": len(positive_distances),
        "mean_dist": float(np.mean(d_arr)),
        "median_dist": float(np.median(d_arr)),
        "std_dist": float(np.std(d_arr)),
        "pct_dist_gt_0.3": float(np.mean(d_arr > 0.3) * 100),
        "pct_dist_gt_0.5": float(np.mean(d_arr > 0.5) * 100),
        "checkpoint": str(ckpt_path),
        "embedding_layer": "h_text (masked_mean_pool before fusion)"
    }

    out_file = repo_root / 'outputs' / 'c3_clean' / 'taco_cosine_stats.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n===== TACO Cosine Stats =====")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved to: {out_file}")


if __name__ == '__main__':
    main()
