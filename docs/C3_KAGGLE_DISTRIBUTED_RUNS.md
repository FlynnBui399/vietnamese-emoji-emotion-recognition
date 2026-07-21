# C3 clean Kaggle: three-worker workflow

Use one copy of `notebooks/C3_ensemble_clean_Kaggle.ipynb` per person. This
does not change the C3 definition: workers train the fixed seeds 42, 1, and 7,
then a separate finalizer averages their probabilities and calibrates thresholds
from the averaged validation probabilities.

## Seed workers (three people)

Every worker attaches the same **private** corrected `c3-clean-inputs` dataset,
containing `data/vigoemotions/{train,val,test}.csv` and
`data/emoji2vec.bin`. The test target support must be exactly 3,942.

In the notebook runtime cell, each worker sets:

```python
JOB_MODE = 'seed_worker'
WORK_PRIORITY = 'P1'
ASSIGNED_SEED = 42  # person 1; use 1 for person 2 and 7 for person 3
ARTIFACT_IMPORT_SOURCES = []
IMPORT_CHECKPOINTS = False
CREATE_FINAL_ZIP = False
QUIET_EXECUTION = True
```

Each worker runs the bootstrap cell, runtime cell, P0 audit cell, and P1 cell.
P1 exits successfully after its assigned seed finishes; it records that ensemble
assembly is waiting for the other two seeds. The worker then publishes its
`/kaggle/working/c3_clean_artifacts/` output as a separate private Kaggle
dataset, for example `c3-canonical-seed42`.

Keep `QUIET_EXECUTION = True`. It redirects model warnings and training output
to `c3_clean_artifacts/logs/`, preventing Papermill from storing a large stream
of output inside the executed notebook.

## Ensemble finalizer

Attach the three private worker-output datasets. In a new Kaggle session, set:

```python
JOB_MODE = 'ensemble_finalizer'
WORK_PRIORITY = 'P1'
ARTIFACT_IMPORT_SOURCES = [
    '/kaggle/input/c3-canonical-seed42/c3_clean_artifacts',
    '/kaggle/input/c3-canonical-seed1/c3_clean_artifacts',
    '/kaggle/input/c3-canonical-seed7/c3_clean_artifacts',
]
IMPORT_CHECKPOINTS = False
CREATE_FINAL_ZIP = False
QUIET_EXECUTION = True
```

Run the bootstrap, runtime, P0, and P1 cells. The finalizer imports only the
probability arrays, targets, ordered IDs, metrics, configurations, and history
by default. It does not retrain. It verifies that all seeds use the exact same
ordered validation/test IDs, averages their probabilities, fits fresh
per-class thresholds only on averaged validation probabilities, and produces
the C3 Ensemble metrics and evaluation artifacts.

## Optional archival/package session

The required final ZIP contains all best checkpoints and can temporarily need
space for both the imported checkpoints and the ZIP. Run this only in a session
with sufficient free disk:

```python
IMPORT_CHECKPOINTS = True
CREATE_FINAL_ZIP = True
```

Attach all three worker datasets again, rerun the finalizer, then run the Final
package cell. If disk is tight, leave both settings false: the workers retain
their individual checkpoints in their published private datasets, and the
finalizer still produces the ensemble metrics safely.

## Distributed P2 ablations

To parallelize P2, set `WORK_PRIORITY = 'P2'` for every worker. Each person
then runs the assigned seed for all selected A0-A3 experiments. The finalizer
requires all three A0 and all three C3 seed artifacts before producing matched
ensemble statistics. This is substantially more expensive than canonical P1,
so complete P1 first.

## GitHub clone prerequisite

The notebook can clone the repository into `/kaggle/working`, but the clean
pipeline changes must be committed and pushed before Kaggle can clone them.
Until then, attach an updated private repository snapshot instead.
