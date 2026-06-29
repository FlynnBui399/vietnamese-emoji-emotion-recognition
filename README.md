# ViGoEmotions Multi-Label Classification (Phase 1 & Phase 2)

Reproduce and extend the **ViSoBERT multi-label baseline** and **Emoji-Aware Fusion** models from
[ViGoEmotions (EACL 2026)](https://aclanthology.org/2026.eacl-long.129.pdf)
on the local CSV split at `data/vigoemotions/`, training on a Modal A100 GPU or Kaggle.

This project implements:
- **Phase 1**: Baseline ViSoBERT with Asymmetric Loss (ASL).
- **Phase 2**: Emoji2Vec dual-encoder fusion, R-Drop regularization, TACO contrastive losses, and per-class decision threshold optimization.

**Reproduction target** (Scenario 1 = keep raw emoji, paper Table):
- Baseline Macro F1 ≈ **61.50%**
- Baseline Weighted F1 ≈ **63.26%**
- C3 / C3_extended Macro F1 (with optimized thresholds) ≈ **61.25%+**

Protocol details:
- Decision threshold is fixed at `0.5`.
- The provided `train.csv`, `val.csv`, and `test.csv` files are used directly.
- Scenario 1 keeps raw emoji and applies rule-based normalization resources from `docs/`.
- ViSoBERT uses its own tokenizer directly; PyVi tokenization is disabled.

---

## Project layout

```
NLP/
├── modal_app.py                 # Modal app: image, volumes, A100 train function, local CLI
├── requirements.txt
├── configs/
│   └── visobert_baseline.yaml   # all hyperparameters
├── src/
│   ├── config.py                # YAML -> TrainConfig dataclass
│   ├── data.py                  # CSV parse, 28-class multi-hot, pos_weight, DataLoaders
│   ├── model.py                 # ViSoBertMultiLabel (backbone + dropout + linear head)
│   ├── losses.py                # BCEWithLogitsLoss + pos_weight
│   ├── metrics.py               # macro/weighted/micro F1, per-class F1, Hamming
│   ├── train.py                 # training loop: bf16 AMP, eval/epoch, best-ckpt, TB + W&B
│   └── utils.py                 # seeding, label maps
├── scripts/
│   ├── upload_data.py           # push data/vigoemotions/ to Modal volume `vigoemotions-data`
│   └── download_ckpt.py         # pull tb logs + best.pt + metrics.json back locally
├── data/
│   └── vigoemotions/            # train.csv / val.csv / test.csv (already downloaded)
└── nlp.py                       # original starter (unused by training)
```

---

## Prerequisites

- Modal CLI installed and authenticated (`modal token set` already done).
- Python 3.10+ locally for running the upload / download helper scripts.
- Local `data/vigoemotions/` containing `train.csv`, `val.csv`, `test.csv`, `README.md`.

Install local Python deps (only the Modal SDK is strictly needed for the helper
scripts; the training image installs everything else inside Modal):

```bash
pip install -r requirements.txt
```

---

## End-to-end workflow

### 1. Upload the dataset to a Modal Volume (one time)

```bash
python scripts/upload_data.py
```

This creates the `vigoemotions-data` volume (if it doesn't exist) and pushes
your local `data/vigoemotions/{train,val,test}.csv` to it. Inside the training
container these are visible at `/data/{train,val,test}.csv`.

Re-run with `--force` if you ever change the local CSVs.

### 2. Launch a training run on an A100

```bash
modal run modal_app.py::train \
  --config-path configs/visobert_baseline.yaml \
  --run-name visobert-baseline-v1
```

What happens:

- Modal builds the image (PyTorch + transformers + tensorboard + wandb + ...).
- An A100 container starts.
- ViSoBERT (`uitnlp/visobert`) weights download into the persistent `hf-cache`
  Volume, so subsequent runs don't re-download.
- Training runs for 10 epochs, ~30–60 min on A100.
- Best checkpoint (by val Macro F1), TensorBoard event files, and `metrics.json`
  are written to the `vigoemotions-runs` Volume under `/runs/<run-name>/`.

### 3. Optional: enable W&B logging

```bash
export WANDB_API_KEY=...           # your Weights & Biases key
export WANDB_PROJECT=vigoemotions  # optional, defaults to "vigoemotions"
modal run modal_app.py::train \
  --config-path configs/visobert_baseline.yaml \
  --run-name visobert-baseline-v1-wandb \
  --use-wandb
```

The local `WANDB_API_KEY` is forwarded to the container via a Modal secret.
TensorBoard logging is always on regardless of W&B; the two are independent.

### 4. List runs in the volume

```bash
modal run modal_app.py::runs
```

### 5. Pull artifacts back locally

```bash
python scripts/download_ckpt.py --run-name visobert-baseline-v1
# or just the TB logs + metrics:
python scripts/download_ckpt.py --run-name visobert-baseline-v1 --tb-only
```

Files land in `artifacts/<run-name>/`:

```
artifacts/visobert-baseline-v1/
├── best.pt          # state dict + config + val metrics at best epoch
├── config.json      # frozen training config
├── metrics.json     # full history + final test metrics (macro/weighted/micro F1, Hamming, per-class)
└── tb/              # TensorBoard event files
```

View TensorBoard:

```bash
tensorboard --logdir artifacts/visobert-baseline-v1/tb
```

---

## What's logged

Every step (`log_every` controls cadence):
- `train/loss`, `train/lr`

Every epoch (val + train summary):
- `train/epoch_loss`
- `val/loss`, `val/macro_f1`, `val/weighted_f1`, `val/micro_f1`, `val/hamming`

After training (loaded best checkpoint, eval on test):
- `test/loss`, `test/macro_f1`, `test/weighted_f1`, `test/micro_f1`, `test/hamming`
- `test/per_class_f1/{idx}_{emotion_name}` for all 28 labels

Same scalars are logged to W&B if enabled.

---

## Configuration

All hyperparameters live in [configs/visobert_baseline.yaml](configs/visobert_baseline.yaml).
Key knobs:

| Field             | Default          | Notes                                                |
|-------------------|------------------|------------------------------------------------------|
| `model_name`      | `uitnlp/visobert`| Any HF encoder works (CafeBERT/PhoBERT later).       |
| `num_labels`      | `28`             | 27 emotions + neutral, per the dataset README.       |
| `max_length`      | `200`            | Matches the baseline notebook setting.               |
| `batch_size`      | `32`             | Fits comfortably in 40GB A100.                       |
| `epochs`          | `12`             | Paper baseline trains until convergence.             |
| `learning_rate`   | `5.0e-5`         | Standard BERT fine-tune LR.                          |
| `use_pos_weight`  | `true`           | Per-class `pos_weight` for BCE — handles imbalance.  |
| `threshold`       | `0.5`            | Fixed decision threshold for validation and test metrics.  |
  uses `dim`. The model file already handles both, but if you swap to a more
  exotic backbone, add another fallback.
- **`OSError: ... uitnlp/visobert ... gated`**: ViSoBERT is public; if HF rate-
  limits you, retry — the `hf-cache` volume preserves successful downloads.
- **Out-of-memory**: bump down `batch_size` to 16 or set `grad_accum: 2`. On
  A100-40GB this should not happen at the defaults.
- **`modal: command not found`**: `pip install modal` and re-run `modal token set`.
