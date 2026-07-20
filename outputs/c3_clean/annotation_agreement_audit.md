# Annotation agreement and extended-data provenance audit

## Annotator agreement

No independent annotator-level labels were found. The extended CSV files and
the `dataset_V1.xlsx` workbook expose only `id`, `text`, and final `labels`
columns. A final consensus label is insufficient for recomputing Cohen's Kappa.

Therefore, **Kappa = 0.67 is unverifiable from repository artifacts**. Do not
fabricate or reverse-engineer agreement. The numeric Kappa claim should be
removed from the main paper unless independent annotator decisions are
recovered.

## Extended-data provenance and permissions

The repository includes `dataset_V1.xlsx`, `train.csv`,
`train - extended.csv`, `val.csv`, `test.csv`, and `label_dict.json` under the
extended-data directory. These files do not verify collection dates,
source-platform terms, collection method, annotator provenance, or
redistribution permission for the extended examples.

The original ViGoEmotions README states that the dataset must not be
redistributed. That notice does not establish permission for the separately
extended material. Provenance, collection dates, platform terms, and
redistribution permissions for extended data are **not verifiable from
repository files**.

