# Data discrepancies

Status: **FATAL - pipeline stopped**

The audit never edits or silently deduplicates a raw label list.

## Findings

- `raw_positive_assignment_mismatch`: `{"code": "raw_positive_assignment_mismatch", "split": "test", "expected": 3942, "actual": 3943}`
- `test` row 1665 (ID `you002641`) contains labels `[11, 8, 12, 9, 12]`; duplicated IDs: `[12]`.

## Consequence

P0 metric reconstruction, threshold fitting, P1-P3 training, and packaging are blocked until the source CSV invariants pass. The parsed multi-hot test matrix may still sum to 3,942 because repeated indices collapse to one cell; that does not make the raw duplicate valid.
