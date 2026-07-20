# Final discrepancy report

## Historical artifact result

The executed legacy Kaggle notebook displays tuned test Macro-F1
`0.6329413412542984` and fixed-0.5 Macro-F1 rounded to `0.6309` for
`ASL_Emoji_CB_3seed_ensemble`. These values are retained only as historical
executed output.

## Artifact reconstruction result

Not available. The three referenced validation/test probability bundles and
checkpoints are absent locally. The displayed C3 value was therefore not
independently recomputed. The historical baseline report was available, and
its 28 exact per-class F1 values recompute to Macro-F1
`0.6140661767682877` with support 3,942.

## Newly trained clean rerun result

Not run. The strict raw-source audit stopped the pipeline because test row 1665
(ID `you002641`) repeats label ID 12. The raw label list has one excess entry; the
parsed target-matrix and per-class support sums are both exactly 3,942. The raw
CSV was not modified or silently deduplicated.

## Resolution required

Obtain an authoritative corrected `test.csv` whose raw lists contain no
duplicate label IDs while preserving 2,067 rows and 3,942 positive target
assignments. Then attach the three historical C3 bundles to Kaggle and rerun
P0 before any training.
