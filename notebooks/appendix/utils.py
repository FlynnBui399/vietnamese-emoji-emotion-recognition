import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, accuracy_score

def optimize_thresholds(probs_val, y_val, grid=(0.05, 0.94, 0.01), clip=(0.15, 0.85), min_pos=10):
    """
    Optimizes per-class thresholds for Macro-F1 on the validation set.
    Classes with < min_pos positive samples default to 0.5.
    """
    num_classes = y_val.shape[1]
    best_thresholds = np.full(num_classes, 0.5)
    
    for c in range(num_classes):
        if y_val[:, c].sum() < min_pos:
            continue
            
        best_f1 = -1
        best_th = 0.5
        for th in np.arange(grid[0], grid[1] + grid[2]/2, grid[2]):
            if th < clip[0] or th > clip[1]:
                continue
            preds = (probs_val[:, c] > th).astype(int)
            
            tp = np.sum(y_val[:, c] * preds)
            fp = np.sum((1 - y_val[:, c]) * preds)
            fn = np.sum(y_val[:, c] * (1 - preds))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[c] = best_th
        
    return best_thresholds

def fast_macro_f1(y_true, y_pred):
    tp = np.sum(y_true * y_pred, axis=0)
    fp = np.sum((1 - y_true) * y_pred, axis=0)
    fn = np.sum(y_true * (1 - y_pred), axis=0)
    
    precision = np.zeros_like(tp, dtype=float)
    recall = np.zeros_like(tp, dtype=float)
    
    p_mask = (tp + fp) > 0
    precision[p_mask] = tp[p_mask] / (tp[p_mask] + fp[p_mask])
    
    r_mask = (tp + fn) > 0
    recall[r_mask] = tp[r_mask] / (tp[r_mask] + fn[r_mask])
    
    f1 = np.zeros_like(tp, dtype=float)
    f_mask = (precision + recall) > 0
    f1[f_mask] = 2 * (precision[f_mask] * recall[f_mask]) / (precision[f_mask] + recall[f_mask])
    
    return np.mean(f1)

def paired_bootstrap(y_true, probs_a, probs_b, B=10000, threshold_a=0.5, threshold_b=0.5):
    """
    Computes p-value via paired bootstrap for Macro-F1.
    """
    preds_a = (probs_a > threshold_a).astype(int)
    preds_b = (probs_b > threshold_b).astype(int)
    
    n = len(y_true)
    diffs = np.zeros(B)
    
    np.random.seed(42)
    for i in range(B):
        idx = np.random.randint(0, n, n)
        f1_a = fast_macro_f1(y_true[idx], preds_a[idx])
        f1_b = fast_macro_f1(y_true[idx], preds_b[idx])
        diffs[i] = f1_b - f1_a
        
    # one-sided p-value: probability that B is not better than A
    p_val = np.mean(diffs <= 0)
    return p_val

def compute_all_metrics(probs, y_true, thresholds):
    """
    Computes Macro-F1, Micro-F1, Weighted-F1, mAP, Hamming Loss, Exact Match.
    """
    preds = (probs > thresholds).astype(int)
    
    # Macro F1
    macro_f1 = fast_macro_f1(y_true, preds)
    
    # Micro F1
    tp = np.sum(y_true * preds)
    fp = np.sum((1 - y_true) * preds)
    fn = np.sum(y_true * (1 - preds))
    mi_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    mi_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) > 0 else 0
    
    # Weighted F1
    class_support = np.sum(y_true, axis=0)
    
    c_tp = np.sum(y_true * preds, axis=0)
    c_fp = np.sum((1 - y_true) * preds, axis=0)
    c_fn = np.sum(y_true * (1 - preds), axis=0)
    
    c_p = np.zeros_like(c_tp, dtype=float)
    p_mask = (c_tp + c_fp) > 0
    c_p[p_mask] = c_tp[p_mask] / (c_tp[p_mask] + c_fp[p_mask])
    
    c_r = np.zeros_like(c_tp, dtype=float)
    r_mask = (c_tp + c_fn) > 0
    c_r[r_mask] = c_tp[r_mask] / (c_tp[r_mask] + c_fn[r_mask])
    
    c_f1 = np.zeros_like(c_tp, dtype=float)
    f_mask = (c_p + c_r) > 0
    c_f1[f_mask] = 2 * (c_p[f_mask] * c_r[f_mask]) / (c_p[f_mask] + c_r[f_mask])
    
    weighted_f1 = np.average(c_f1, weights=class_support)
    
    # mAP
    map_score = average_precision_score(y_true, probs, average='macro')
    
    # Hamming
    hamming = hamming_loss(y_true, preds)
    
    # Exact Match
    exact_match = accuracy_score(y_true, preds)
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "map": map_score,
        "hamming_loss": hamming,
        "exact_match": exact_match
    }

def emoji_coverage(texts, emoji2vec_vocab_set):
    import emoji
    total_emojis = 0
    unique_emojis = set()
    found_occurrences = 0
    found_unique = set()
    sentences_with_oov = 0
    oov_counts = {}
    
    for text in texts:
        emojis_in_text = [item["emoji"] for item in emoji.emoji_list(str(text))]
        has_oov = False
        for emj in emojis_in_text:
            total_emojis += 1
            unique_emojis.add(emj)
            if emj in emoji2vec_vocab_set:
                found_occurrences += 1
                found_unique.add(emj)
            else:
                has_oov = True
                oov_counts[emj] = oov_counts.get(emj, 0) + 1
        if has_oov:
            sentences_with_oov += 1
            
    pct_occ_found = (found_occurrences / total_emojis) * 100 if total_emojis > 0 else 0
    pct_unique_found = (len(found_unique) / len(unique_emojis)) * 100 if len(unique_emojis) > 0 else 0
    pct_sent_oov = (sentences_with_oov / len(texts)) * 100 if len(texts) > 0 else 0
    
    import unicodedata
    top_10_oov = sorted(oov_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_formatted = []
    for emj, count in top_10_oov:
        try:
            name = unicodedata.name(emj[0])
        except:
            name = "UNKNOWN"
        top_10_formatted.append({"emoji": emj, "name": name, "count": count})
        
    return {
        "total_occurrences": total_emojis,
        "unique_types": len(unique_emojis),
        "coverage_by_occurrence": pct_occ_found,
        "coverage_by_type": pct_unique_found,
        "pct_sentences_with_oov": pct_sent_oov,
        "top_10_oov": top_10_formatted
    }

def to_latex_table(df, caption, label):
    """
    Converts a pandas DataFrame to a simple LaTeX table.
    """
    latex = "\\begin{table}[htpb]\n\\centering\n"
    latex += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    
    latex += "\\begin{tabular}{" + "c" * len(df.columns) + "}\n\\hline\n"
    latex += " & ".join([str(c).replace('_', '\\_') for c in df.columns]) + " \\\\\n\\hline\n"
    
    for _, row in df.iterrows():
        latex += " & ".join([str(v).replace('_', '\\_') for v in row.values]) + " \\\\\n"
        
    latex += "\\hline\n\\end{tabular}\n\\end{table}"
    return latex
