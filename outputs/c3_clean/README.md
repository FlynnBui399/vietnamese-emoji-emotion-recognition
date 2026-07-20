# C3 clean pipeline status

Canonical system name: **C3 Ensemble**  
Technical artifact name: `ASL_Emoji_CB_3seed_ensemble`

This directory contains the locally executable P0 audit for the clean Kaggle
pipeline. The raw source audit is fatal: `test.csv` contains one repeated label
ID. The raw list therefore has one excess repeated entry even though the parsed
28-dimensional target matrix and per-class support both sum to exactly 3,942.

No source CSV or legacy notebook was modified. In accordance with the protocol,
artifact reconstruction, threshold fitting, training, test evaluation,
statistics, emoji analysis, and ZIP packaging were stopped.

The clean implementation is in `src/c3_clean/`, the frozen configuration is
`configs/c3_clean.yaml`, and the Kaggle launcher is
`notebooks/C3_ensemble_clean_Kaggle.ipynb`.
