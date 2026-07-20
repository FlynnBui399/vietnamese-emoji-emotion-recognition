# C3 configuration audit

Canonical paper name: **C3 Ensemble**  
Technical artifact name: `ASL_Emoji_CB_3seed_ensemble`

## Artifact inventory

No checkpoint, `val_probs.npy`, `test_probs.npy`, `val_targets.npy`,
`test_targets.npy`, or `thresholds.json` for the three historical C3 seeds is
present in this repository.

The executed legacy notebook references these external Kaggle bundles:

- `ASL_Emoji_CB__seed42`
- `ASL_Emoji_CB_ensemble_seed1`
- `ASL_Emoji_CB_ensemble_seed7`
- `/kaggle/working/artifacts/final_ensemble`

The executed output displays an exact tuned test Macro-F1 of
`0.6329413412542984` and a fixed-0.5 result rounded to `0.6309`. These are
historical executed-notebook outputs, not an independent local recomputation.

## Architecture evidence

The current notebook source defines a simple `EmojiAwareViSoBERT` with:

- ViSoBERT and masked mean pooling over non-padding tokens;
- raw-text Unicode emoji extraction using `emoji.emoji_list`;
- mean Emoji2Vec aggregation in 300 dimensions, with zeros when uncovered;
- `Linear(300, 768)`, GELU, and LayerNorm;
- concatenation of 768-dimensional text and emoji representations;
- `Linear(1536, 768)`, GELU, Dropout(0.2), and `Linear(768, 28)`.

However, the current notebook source no longer contains the `ASL_Emoji_CB`
experiment definition that produced the saved output. This source/output drift,
combined with absent checkpoints, means the exact historical class and full
configuration cannot be verified from a state dict.

## Configuration used by the clean rerun

Pending successful source-data audit, the clean implementation resolves to
ViSoBERT, max length 128, no PyVi, batch size 32, AdamW at `2e-5`, weight decay
0.01, one warmup epoch followed by linear decay, at most 10 epochs, patience 3,
ASL `(gamma_negative=4, gamma_positive=0, clip=0.05)`, and effective-number
positive-term loss weights with beta 0.999 for seeds 42, 1, and 7.

This configuration is explicitly labeled an inference where historical
checkpoint metadata is absent. R-Drop, label graphs, cross-attention, Vietnamese
gloss embeddings, C4, and extended data are excluded from canonical C3.

