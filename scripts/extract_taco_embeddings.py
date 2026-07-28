import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# Add repo root to sys.path so 'src' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.c3_clean.data_audit import load_split
from src.c3_clean.preprocessing import prepare_text_columns, ImmutablePreprocessor
from src.c3_clean.training import C3Dataset, collate_fn
from src.c3_clean.model import build_model
from transformers import AutoTokenizer

def main():
    repo_root = Path.cwd()
    config_path = repo_root / 'configs' / 'c3_clean.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Kaggle override
    kaggle_data = Path('/kaggle/input/vigoemotions/data/vigoemotions')
    if kaggle_data.exists(): cfg['paths']['data_dir'] = str(kaggle_data)
    else: cfg['paths']['data_dir'] = str(repo_root / 'data' / 'vigoemotions')
        
    kaggle_emoji = Path('/kaggle/input/emoji2vec-data/emoji2vec.bin')
    if kaggle_emoji.exists(): cfg['paths']['emoji2vec'] = str(kaggle_emoji)
    else: cfg['paths']['emoji2vec'] = str(repo_root / 'data' / 'vigoemotions_extended' / 'emoji2vec.bin')
    
    # 1. Load test data
    print("Loading test data...")
    test_frame = pd.read_csv(Path(cfg['paths']['data_dir']) / 'test.csv')
    preprocessor = ImmutablePreprocessor.from_docs(cfg['paths']['docs_dir'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_name'], use_fast=False)
    prepared_test = prepare_text_columns(test_frame, preprocessor)
    
    # Prepare emoji2vec dict if needed
    emoji_dict = {}
    # We can load emoji2vec if required by dataset, but C3Dataset can do it internally or we pass it
    from gensim.models import KeyedVectors
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w2v = KeyedVectors.load_word2vec_format(cfg['paths']['emoji2vec'], binary=True)
    emoji_dict = {word: w2v[word] for word in w2v.index_to_key}
    
    test_ds = C3Dataset(prepared_test, tokenizer, emoji_dict, max_length=128)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, collate_fn=collate_fn, shuffle=False)
    
    # 2. Load model
    print("Loading model A2_controlled_ASL_Emoji seed 42...")
    model = build_model('A2_controlled_ASL_Emoji', cfg['model'])
    
    # Find checkpoint in Kaggle structure
    ckpt_path = None
    if Path('/kaggle/input').exists():
        print("Searching for best_checkpoint.pt in /kaggle/input...")
        for p in Path('/kaggle/input').rglob('best_checkpoint.pt'):
            if 'A2_controlled_ASL_Emoji' in str(p) and 'seed42' in str(p):
                ckpt_path = p
                break
                
    if not ckpt_path:
        ckpt_candidates = [
            Path('/kaggle/working/c3_clean_artifacts/experiments/A2_controlled_ASL_Emoji/seed42/best_checkpoint.pt'),
            repo_root / 'outputs/c3_clean/experiments/A2_controlled_ASL_Emoji/seed42/best_checkpoint.pt'
        ]
        for p in ckpt_candidates:
            if p.exists():
                ckpt_path = p
                break
            
    if not ckpt_path:
        print("ERROR: Checkpoint for A2 seed 42 not found!")
        print("Please ensure you are running this on Kaggle with the worker-seed-42 dataset attached.")
        return
        
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 3. Extract embeddings (h_text)
    print("Extracting embeddings...")
    all_h_text = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl):
            for k in ['input_ids', 'attention_mask', 'emoji_vectors']:
                batch[k] = batch[k].to(device)
            out = model(**batch)
            all_h_text.append(out.h_text.cpu().numpy()) # h_text is 768d pooled text before fusion
            targets.append(batch['labels'].cpu().numpy())
            
    embeddings = np.concatenate(all_h_text, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # 4. Define TACO Clusters & compute stats
    taco_clusters = {
        0: [8, 0, 1, 2, 3, 7, 9, 10, 5, 6], 
        1: [11, 4], 
        2: [24, 25, 23, 26, 19, 16], 
        3: [20, 21, 18, 22], 
        4: [15, 14, 12, 13], 
        5: [27] 
    }
    
    print("Finding positive pairs based on TACO...")
    # Find active clusters for each sample
    sample_clusters = []
    for i in range(len(targets)):
        active_c = set()
        active_labels = np.where(targets[i] == 1)[0]
        for l in active_labels:
            for cid, cls_labels in taco_clusters.items():
                if l in cls_labels:
                    active_c.add(cid)
        sample_clusters.append(active_c)
        
    # Get all positive pairs (sharing >= 1 cluster)
    n = len(targets)
    positive_distances = []
    
    # Compute full distance matrix (N=2067 is small enough)
    dist_matrix = squareform(pdist(embeddings, metric='cosine'))
    
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(sample_clusters[i].intersection(sample_clusters[j])) > 0:
                positive_distances.append(dist_matrix[i, j])
                pair_count += 1
                
    d_arr = np.array(positive_distances)
    
    stats = {
        "n_positive_pairs": pair_count,
        "mean_dist": float(np.mean(d_arr)),
        "median_dist": float(np.median(d_arr)),
        "std_dist": float(np.std(d_arr)),
        "pct_dist_gt_0.3": float(np.mean(d_arr > 0.3) * 100),
        "pct_dist_gt_0.5": float(np.mean(d_arr > 0.5) * 100),
        "checkpoint": "A2_controlled_ASL_Emoji_seed42",
        "embedding_layer": "h_text_before_fusion"
    }
    
    out_file = repo_root / 'outputs/c3_clean/taco_cosine_stats.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(json.dumps(stats, indent=2))
    print(f"Done! Stats saved to {out_file}")

if __name__ == '__main__':
    main()
