# machine_learning_project — LAEI 2019 data and EDA

A subset of the LAEI 2019 atmospheric emissions modelling project: the
exploratory analysis notebook, the modeller handover guide, the Task 1 & 2
report, and the prepared modelling tables.

## Contents

| Path | What it is |
|---|---|
| `notebooks/02_eda_and_profiling.ipynb` | Exploratory analysis and data preparation, executed with outputs |
| `reports/Task1_Task2_Report.md` | Algorithm selection and expected predictive insights |
| `reports/MODELLER_HANDOVER.md` | Detailed guide to the data and the modelling approach |
| `data/processed/` | Prepared modelling tables plus the intermediate exports |
| `requirements.txt` | Pinned dependencies (Python 3.12) |

## Note on `grid_all_years.csv.gz`

That file is stored gzipped because the raw CSV is 123 MB, over GitHub's 100 MB
limit. It needs no special handling — pandas reads it directly:

```python
import pandas as pd
df = pd.read_csv("data/processed/grid_all_years.csv.gz", low_memory=False)   # 699,120 rows
```

Or decompress it with `gunzip -k data/processed/grid_all_years.csv.gz`.

## Which table to use

`table_A_links.csv` (road links), `table_B_grid_source_mix.csv` (1 km cells for
clustering) and `table_C_grid_nonleaky.csv` (1 km cells for classification) are
the prepared modelling tables. `manifest.json` describes all of them.

`table_C_LEAKY_reference.csv` is a deliberate data-leakage demonstration — its
features sum to its target. It is not a model input.

See `reports/MODELLER_HANDOVER.md` for full detail.

## Attribution

Contains data from the Greater London Authority, *London Atmospheric Emissions
Inventory (LAEI) 2019*, London Datastore, licensed under the UK Open Government
Licence v3.0.
