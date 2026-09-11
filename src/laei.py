"""
LAEI 2019 -- shared loading and feature-preparation logic.

Everything the EDA notebook and the modelling notebooks need lives here, so
that the analysis and the handover package cannot drift apart. Import it
rather than copying code:

    import sys; sys.path.append("../src")
    import laei

The three prepared tables map one-to-one onto the three selected algorithms:

    build_link_table()      -> Table A  : Linear Regression + Random Forest (regression)
    build_grid_source_mix() -> Table B  : K-Means (segmentation)
    build_grid_nonleaky()   -> Table C  : Random Forest (classification) + Linear Regression

All figures quoted in the docstrings were verified against the 2019 release.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASE_YEAR = 2019
FORECAST_YEARS = (2025, 2030)

# The 19 pollutant columns in the grid workbook, in file order.
POLLUTANTS = [
    "bap", "cd", "c4h6", "c6h6", "ch4", "co", "co2", "hc", "hcl", "hg",
    "n2o", "nh3", "nmvoc", "nox", "pb", "pcb", "pm10", "pm2.5", "so2",
]

# Only these four are dense enough to support supervised learning; see the
# coverage table in the EDA notebook (section 3).
MODELLABLE = ["nox", "pm10", "pm2.5", "co2"]

# Zone labels differ between the two workbooks. Normalise to the grid form.
ZONE_CANON = {
    "Non-GLA": "Non GLA",
    "Non GLA": "Non GLA",
    "Central London": "Central",
    "Inner London": "Inner",
    "Outer London": "Outer",
    "Central": "Central",
    "Inner": "Inner",
    "Outer": "Outer",
}


# --------------------------------------------------------------------------
# raw loaders
# --------------------------------------------------------------------------

def load_grid(year: int | None = BASE_YEAR) -> pd.DataFrame:
    """
    Load the 1 km grid emissions table.

    699,120 rows across 2013 / 2016 / 2019 / 2025 / 2030. Passing a year
    filters to it; 2019 gives 143,976 rows over 3,460 unique cells. Pass
    year=None to keep every year (needed only for the forecast-validation
    extension, never for the 2019 baseline models).
    """
    df = pd.read_csv(PROCESSED / "grid_all_years.csv", low_memory=False)
    if year is not None:
        df = df[df["Year"] == year].copy()
    df["Zone"] = df["Zone"].map(lambda z: ZONE_CANON.get(z, z))
    return df


def load_link_features() -> pd.DataFrame:
    """Major-road traffic table: 79,437 links x 48 columns, 2019 only."""
    df = pd.read_csv(PROCESSED / "link_features.csv", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_link_targets() -> pd.DataFrame:
    """Per-link 2019 emissions, one row per TOID (79,439), PM summed over source."""
    return pd.read_csv(PROCESSED / "link_targets.csv", low_memory=False)


# --------------------------------------------------------------------------
# Table A -- road-link level (Linear Regression + Random Forest regression)
# --------------------------------------------------------------------------

def link_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("AADT", "VKM", "Speed", "Link"))]


LINK_CATEGORICAL = ["LAEI Zone", "Borough", "Road Classification"]


def build_link_table(drop_null_target: bool = True) -> pd.DataFrame:
    """
    Table A: join traffic features to emission targets on TOID.

    Shapes: features 79,437 x 48; targets 79,439 x 5; inner join 79,388.

    Two coercions happen here, and they are NOT the same coercion:

    * Every AADT/VKM column arrives from Excel as text, and uses '-' as a
      placeholder. '-' means ZERO VEHICLES OF THAT CLASS, not missing data.
      This was verified directly: on the 33,705 links that have a numeric
      total and at least one dashed class, treating '-' as zero makes
      sum(classes) == total to within the same rounding tolerance (max 3
      vehicles) as the 45,043 links with no dashes at all. These columns are
      therefore coerced and ZERO-FILLED. Median-imputing them instead --
      the obvious default -- would invent traffic on 42% of the network.

    * The two Speed columns also use '-', but there it does mean missing.
      26,998 links lack a buses-only speed while 15,889 of those do carry
      bus traffic, so zero would be a false reading. Speeds are left as NaN
      for median imputation inside the modelling Pipeline, and a
      `bus_speed_missing` indicator is added so the model can respond to
      the missingness itself.
    """
    X = load_link_features()
    y = load_link_targets()
    df = X.merge(y, on="TOID", how="inner")

    count_cols = [c for c in df.columns if c.startswith(("AADT", "VKM"))]
    other_num = [c for c in link_numeric_columns(df) if c not in count_cols]

    df["no_traffic_data"] = (
        df["AADT 2019 - Total"].astype(str).str.strip() == "-"
    ).astype(int)
    df["bus_speed_missing"] = (
        df["Speed (km/hr) - Buses Only"].astype(str).str.strip() == "-"
    ).astype(int)

    for c in count_cols:                       # '-' == zero vehicles
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in other_num:                        # '-' == genuinely missing
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["zone_canon"] = df["LAEI Zone"].map(lambda z: ZONE_CANON.get(z, z))

    # Emission intensity: total emissions scale with link length, so a long
    # quiet road out-emits a short busy one. Per-metre intensity is the
    # quantity that actually proxies roadside exposure.
    length_m = df["Link Length (m)"].where(df["Link Length (m)"] > 0)
    df["nox_per_m"] = df["nox"] / length_m
    df["is_motorway"] = (
        df["Road Classification"]
        .astype(str)
        .str.match(r"^(M\d+|A\d+M)$")
        .astype(int)
    )

    if drop_null_target:
        df = df.dropna(subset=["nox"])
    return df


# --------------------------------------------------------------------------
# Table B -- grid source mix (K-Means)
# --------------------------------------------------------------------------

GRID_KEYS = ["Grid ID 2019", "Easting", "Northing", "Borough", "Zone"]


def build_grid_source_mix(pollutant: str = "nox", year: int = BASE_YEAR):
    """
    Table B: one row per 1 km cell, one column per sector, plus cell totals.

    Returns (table, sector_columns). Verified shape for NOx/2019: (3460, 25)
    -- 5 keys + 16 sectors + 4 pollutant totals.

    A null pollutant value means 'not estimated for this source', not
    'measured as zero', so fill_value=0 is correct when aggregating to a
    cell total but mean-imputation would be indefensible.
    """
    d = load_grid(year)
    wide = d.pivot_table(
        index=GRID_KEYS, columns="Sector", values=pollutant,
        aggfunc="sum", fill_value=0,
    ).reset_index()
    sectors = [c for c in wide.columns if c not in GRID_KEYS]

    totals = d.groupby("Grid ID 2019")[MODELLABLE].sum().reset_index()
    table = wide.merge(totals, on="Grid ID 2019")
    return table, sectors


def to_shares(table: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    """
    Convert sector emissions to within-cell shares.

    K-Means on raw tonnes would simply rank cells by size -- the loudest
    cell would dominate every centroid. Shares ask the question we actually
    want ('what kind of place is this?') rather than 'how big is it?'.
    """
    raw = table[sectors].sum(axis=1)
    denom = raw.where(raw > 0)
    return table[sectors].div(denom, axis=0).fillna(0)


# --------------------------------------------------------------------------
# Table C -- grid, non-leaky predictors (RF classification / LR regression)
# --------------------------------------------------------------------------

def build_grid_nonleaky(year: int = BASE_YEAR, hotspot_quantile: float = 0.90):
    """
    Table C: predict NOx from CO2-by-sector plus location.

    Why not NOx-by-sector? Because those 16 columns SUM TO the target -- a
    model on them scores ROC-AUC 0.998 while learning nothing but addition.
    CO2 by sector proxies activity and fuel throughput; NOx additionally
    depends on combustion technology and abatement, so the residual carries
    real signal.

    Returns (table, feature_columns). 23 columns with Grid ID as the index;
    25 after reset_index() and the hotspot flag. 18 feature columns. Hotspot threshold at the 90th percentile is
    27.52 t/yr, giving 346 positives of 3,460.
    """
    d = load_grid(year)

    co2 = d.pivot_table(
        index="Grid ID 2019", columns="Sector", values="co2",
        aggfunc="sum", fill_value=0,
    )
    co2.columns = ["co2_" + c for c in co2.columns]

    meta = d.groupby("Grid ID 2019").agg(
        Easting=("Easting", "first"),
        Northing=("Northing", "first"),
        Zone=("Zone", "first"),
        Borough=("Borough", "first"),
    )
    targets = d.groupby("Grid ID 2019")[["nox", "pm10", "pm2.5"]].sum()

    table = meta.join(co2).join(targets).reset_index()

    features = [c for c in table.columns if c.startswith("co2_")] + ["Easting", "Northing"]

    thr = table["nox"].quantile(hotspot_quantile)
    table["nox_hotspot"] = (table["nox"] >= thr).astype(int)
    table.attrs["hotspot_threshold"] = float(thr)
    return table, features


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool = True):
    """
    Standard ColumnTransformer for this project.

    Median imputation applies only to the two Speed columns -- the traffic
    counts are already zero-filled in build_link_table(), because '-' there
    means zero vehicles rather than missing. Median rather than mean because
    both features and targets are heavily right-skewed. Then optional
    scaling, and one-hot encoding with
    handle_unknown='ignore' so an unseen borough at predict time cannot
    raise. Always used INSIDE a Pipeline so the split is never leaked
    through a scaler fitted on the full dataset.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


# --------------------------------------------------------------------------
# feature sets -- the contract handed to the modeller
# --------------------------------------------------------------------------

def link_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Two feature sets over Table A, and the contrast between them is the
    point of the whole exercise.

    'full' includes VKM. LAEI computes link emissions as
    sum(VKM_class x emission_factor_class), so VKM is the inventory's own
    input: a model given it reproduces arithmetic and scores R2 ~ 0.99
    regardless of algorithm. Report it as a LEAKAGE DEMONSTRATION.

    'realistic' withholds VKM, leaving the traffic counts and geometry a
    borough monitoring unit actually holds. The target is then multiplicative
    (AADT x length), which trees represent natively and a linear model
    cannot -- and the model ranking flips.
    """
    numeric = link_numeric_columns(df)
    return {
        "full": numeric,
        "realistic": [c for c in numeric if not c.startswith("VKM")],
    }


# --------------------------------------------------------------------------
# evaluation -- shared so every notebook reports the same way
# --------------------------------------------------------------------------

RESULTS = ROOT / "reports" / "results"
RANDOM_STATE = 42


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """R2, MAE and RMSE together.

    Never report R2 alone on this data. The targets are strongly right-skewed,
    so RMSE (which squares errors) is dominated by a handful of motorway links
    and the Heathrow cell, while MAE weights every observation equally. The two
    can rank models differently, and when they do that divergence is a finding:
    MAE describes the typical road, RMSE the extremes.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """ROC-AUC, average precision, and per-class precision/recall/F1.

    Accuracy is deliberately reported last and should not be headlined: the
    hotspot classes are roughly 1:9, so predicting 'never a hotspot' scores 90%.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
        out["avg_precision"] = float(average_precision_score(y_true, y_proba))
    return out


def save_results(name: str, payload: dict) -> Path:
    """Persist one notebook's results so notebook 06 can consolidate them."""
    import json

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    return path


def load_results(name: str) -> dict:
    import json

    with open(RESULTS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)
