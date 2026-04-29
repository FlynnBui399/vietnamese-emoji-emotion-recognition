# ViGoEmotions Baseline (Phase 1) — ViSoBERT on Modal A100

Reproduce the **ViSoBERT multi-label baseline** from
[ViGoEmotions (EACL 2026)](https://aclanthology.org/2026.eacl-long.129.pdf)
on the local CSV split at `data/vigoemotions/`, training on a Modal A100 GPU.

This is **Phase 1** of the research plan in
[`Kế Hoạch Nghiên Cứu Chi Tiết...md`](Kế%20Hoạch%20Nghiên%20Cứu%20Chi%20Tiết%20%20Emoji-Aware%20Fine-Grained%20Emotion%20Recognition%20Cho%20Tiếng%20Việt.md).
Phase 2 (Emoji2Vec dual-encoder fusion + TACO contrastive losses) reuses the
same data / model / training scaffolding.

**Reproduction target** (Scenario 1 = keep raw emoji, paper Table):
- Macro F1 ≈ **61.50%**
- Weighted F1 ≈ **63.26%**

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
│   ├── metrics.py               # macro/weighted/micro F1, per-class F1, Hamming, threshold sweep
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
- `val/macro_f1_tuned`, `val/threshold_tuned` (best global threshold sweep)

After training (loaded best checkpoint, eval on test):
- `test/loss`, `test/macro_f1`, `test/weighted_f1`, `test/micro_f1`, `test/hamming`,
  `test/macro_f1_tuned`
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
| `max_length`      | `128`            | ~99% of comments fit.                                |
| `batch_size`      | `32`             | Fits comfortably in 40GB A100.                       |
| `epochs`          | `10`             | Paper baseline trains until convergence.             |
| `learning_rate`   | `2.0e-5`         | Standard BERT fine-tune LR.                          |
| `use_pos_weight`  | `true`           | Per-class `pos_weight` for BCE — handles imbalance.  |
| `threshold`       | `0.5`            | Default decision threshold; tuned threshold also reported. |

Make a new YAML next to it for ablations later (e.g. `configs/visobert_no_pos_weight.yaml`).

---

## Modal volumes used

| Volume name           | Mount path                       | Purpose                                  |
|-----------------------|----------------------------------|------------------------------------------|
| `vigoemotions-data`   | `/data`                          | Train/val/test CSVs (uploaded once)      |
| `vigoemotions-runs`   | `/runs`                          | TB logs + checkpoints + metrics.json     |
| `hf-cache`            | `/root/.cache/huggingface`       | HF model + tokenizer cache               |

---

## Phase 2 hooks (not implemented yet)

These pieces are deliberately structured so Phase 2 just plugs in:

- **Emoji2Vec dual-encoder + intermediate fusion**: subclass
  `ViSoBertMultiLabel` in [src/model.py](src/model.py); add a second branch and
  a fusion module before the classifier head.
- **TACO contrastive losses (CCL + LDL)**: extend [src/losses.py](src/losses.py)
  with new modules; combine in `src/train.py` via a `loss_weights` config block.
- **CafeBERT / PhoBERT comparison**: just change `model_name` in a new YAML.
- **LoRA**: wrap `model.backbone` with `peft.get_peft_model(...)` after
  `ViSoBertMultiLabel.__init__`; toggle via a `lora` config section.

---

## Troubleshooting

- **`AttributeError: ... config has no 'hidden_size'`**: model loaded but config
  uses `dim`. The model file already handles both, but if you swap to a more
  exotic backbone, add another fallback.
- **`OSError: ... uitnlp/visobert ... gated`**: ViSoBERT is public; if HF rate-
  limits you, retry — the `hf-cache` volume preserves successful downloads.
- **Out-of-memory**: bump down `batch_size` to 16 or set `grad_accum: 2`. On
  A100-40GB this should not happen at the defaults.
- **`modal: command not found`**: `pip install modal` and re-run `modal token set`.
