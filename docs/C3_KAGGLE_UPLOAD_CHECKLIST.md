# C3 clean Kaggle upload checklist

Use **private** Kaggle datasets for the raw social-media data, Emoji2Vec vectors,
historical predictions, and checkpoints. The repository documentation does not
authorize public redistribution of the source data.

## 1. Code

Preferred setup: enable Internet for the Kaggle notebook and let
`notebooks/C3_ensemble_clean_Kaggle.ipynb` clone:

```text
https://github.com/FlynnBui399/vietnamese-emoji-emotion-recognition.git
```

The clean pipeline files must be committed and pushed before this works. A clone
of the current remote branch cannot include local uncommitted files. If Internet
must remain disabled, upload an updated repository snapshot as a private Kaggle
dataset instead.

## 2. Required private input dataset

Recommended Kaggle dataset slug and layout:

```text
c3-clean-inputs/
├── data/
│   ├── emoji2vec.bin
│   └── vigoemotions/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
```

The notebook automatically prefers:

```text
/kaggle/input/c3-clean-inputs/data/vigoemotions
```

Set `DATA_DIR_OVERRIDE` and `EMOJI2VEC_OVERRIDE` in the runtime cell if the
attached dataset uses another path.

Before upload, the canonical CSV bundle must satisfy all clean-pipeline
assertions. In particular, `test.csv` must contain 2,067 rows and its parsed
28-dimensional target matrix must contain exactly 3,942 positive assignments.
It must also contain no repeated label ID within a row.

The legacy repository copy currently has one repeated label in stable test ID
`you002641`. Do not silently edit or overwrite that legacy file. Supply the
authoritatively corrected CSV as a separate private input and include a small
correction manifest recording the original and corrected SHA-256 hashes, stable
ID, old and new label list, date, author, and rationale.

## 3. Historical C3 artifacts for P0

To independently reconstruct the historical C3 result without retraining,
attach a private artifact dataset containing the three seed bundles. Preserve
their original directory names when possible:

```text
ASL_Emoji_CB__seed42/
ASL_Emoji_CB_ensemble_seed1/
ASL_Emoji_CB_ensemble_seed7/
```

Minimum required arrays for every seed:

- `val_probs.npy`, shape `(2066, 28)`
- `val_targets.npy`, shape `(2066, 28)`
- `test_probs.npy`, shape `(2067, 28)`
- `test_targets.npy`, shape `(2067, 28)` with support sum 3,942

Strongly recommended for a configuration and ordering audit:

- best checkpoint/state dict;
- frozen config and resume manifest;
- training history and best epoch;
- ordered validation and test ID arrays;
- thresholds and classification reports.

Do not upload only the three threshold files: ensemble thresholds must be
refitted from the averaged validation probabilities.

## 4. Optional offline dependencies

With Kaggle Internet enabled, the notebook can clone GitHub and download
`uitnlp/visobert`. For a fully offline run, also attach:

- a complete Hugging Face snapshot of `uitnlp/visobert` and set
  `VISOBERT_MODEL_OVERRIDE`;
- wheels for any dependency missing from the Kaggle image, including the pinned
  `emoji` package version in `requirements.txt`.

## 5. Resume between Kaggle sessions

At the end of each session, save `/kaggle/working/c3_clean_artifacts/` as a
private Kaggle dataset. In the next session, attach it and set
`RESUME_ARTIFACTS_SOURCE` in the runtime cell. The notebook restores completed
runs before execution, and the pipeline validates configs and hashes before
skipping them.

## Not needed for P0-P2

Do not attach the extended/C4 dataset, graph or cross-attention artifacts,
R-Drop artifacts, or the course report for the canonical C3 run. Those are
outside the P0-P2 source of truth and model definition.
