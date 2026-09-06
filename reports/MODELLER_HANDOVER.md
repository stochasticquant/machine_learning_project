# Modeller Handover — LAEI 2019

### A working guide for building the models, written for someone new to ML

**From:** EDA / data preparation (Tasks 1 & 2)
**To:** Modelling (Tasks 3 & 4)
**Environment:** conda env `dba` — Python 3.12.13, pandas 3.0.5, scikit-learn 1.9.0

---

## How to read this document

You do **not** need to read it end to end before starting.

| If you want to… | Go to |
|---|---|
| Get something running in ten minutes | §1 |
| Understand what the data actually *is* | §2 |
| Learn the ML concepts this project relies on | §3 |
| Understand **why** I chose these three algorithms | §4 |
| Know exactly what is in each prepared file | §5 |
| Copy working implementations | §6 |
| Avoid the mistakes that cost me time | §7 |
| Check your results are correct | §8 |
| Look up a term | §9 |

Sections 3 and 4 are the theory. If you are already comfortable with train/test
splits, cross-validation and the difference between MAE and RMSE, skim §3 and
read §4 properly — §4 is where the reasoning specific to *this* project lives.

Everything you need is already prepared. **You should not open a single `.xlsx`
file**, and you should not have to re-derive any feature engineering.

---

## 1 · Your first ten minutes

```bash
conda activate dba
cd notebooks
jupyter lab
```

In a new notebook:

```python
import sys
sys.path.append("../src")
import laei                      # the project's shared helper module — short, worth reading
import pandas as pd, json

links    = pd.read_csv("../data/processed/table_A_links.csv", low_memory=False)
manifest = json.load(open("../data/processed/manifest.json"))

print(links.shape)                # (79388, 56)
print(manifest["tables"].keys())  # what has been prepared for you
```

Now fit something, just to see the pipeline work end to end:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

fs  = laei.link_feature_sets(links)     # {'full': 43, 'realistic': 23}
num = fs["realistic"]
cat = laei.LINK_CATEGORICAL             # ['LAEI Zone','Borough','Road Classification']

X, y = links[num + cat], links["nox"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("pre",   laei.make_preprocessor(num, cat)),
    ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
]).fit(X_tr, y_tr)

print(laei.regression_metrics(y_te, model.predict(X_te)))
# {'R2': 0.988, 'MAE': 0.019, 'RMSE': 0.119}
```

If you got roughly those three numbers, your environment is correct and you can
trust everything downstream. **This takes about 40 seconds to fit.**

> **Windows note.** Put this in the first cell of every notebook, *before*
> importing scikit-learn, or `joblib` prints a noisy core-count warning on every
> parallel fit:
> ```python
> import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
> ```

---

## 2 · What the data actually is

Before any modelling, understand what you are predicting. Getting this wrong is
the most expensive kind of mistake, because the code still runs.

### 2.1 What is an emissions inventory?

The **London Atmospheric Emissions Inventory (LAEI)** is the Greater London
Authority's estimate of how much pollution is released across London in a year.
It covers **NOx** (nitrogen oxides, from combustion — the main traffic
pollutant), **PM10 / PM2.5** (particulate matter — soot plus brake and tyre
wear), and **CO2**.

Everything is in **tonnes per year**.

### 2.2 The single most important thing to understand

**LAEI is a *model*, not a set of measurements.**

Nobody stood at 79,000 roadsides with an instrument. The GLA *calculated* these
numbers, roughly as:

```
emissions from a road = traffic volume × distance travelled × emission factor
```

where the emission factor is a lab-derived constant such as "an average diesel
car emits X grams of NOx per kilometre."

Three consequences shape the whole project:

1. **A very high accuracy score does not mean you predicted reality.** It means
   you reproduced the GLA's arithmetic. This is why §7.1 matters so much.
2. **Some columns are the *inputs* to that calculation.** Feed them to a model
   and you have handed it the answer.
3. **The "truth" you are scored against has its own uncertainty**, which no
   metric here captures.

### 2.3 The two levels of data

Two different scales, and it matters not to confuse them:

**Road links** — a "link" is one segment of road, roughly between junctions.
There are **79,388**. Each has traffic counts by vehicle type, speeds, and
length. A **big, detailed** table.

**Grid cells** — London divided into **3,460** squares of 1 km × 1 km. Each has
emissions split across 16 *sectors* (road transport, heating, aviation, river
traffic, industry, …). A **small, aggregated** table.

Different questions live at each level, which is why different models are
applied to each.

---

## 3 · The ML concepts you need

Skip to §4 if these are already familiar.

### 3.1 Supervised vs unsupervised

**Supervised learning** — you have labelled examples: for each road link you know
both the inputs (traffic, length) and the correct answer (its emissions). The
model learns the mapping. Two of the three models are supervised.

**Unsupervised learning** — there is no correct answer to learn; you are looking
for structure. The clustering model groups grid cells by similarity without ever
being told which grouping is "right."

### 3.2 Regression vs classification

**Regression** predicts a number: *how many tonnes does this road emit?*

**Classification** predicts a category: *is this cell a pollution hotspot?*

I do both deliberately, because they answer different client questions.
Regression gives magnitude; classification tells you where to send an inspector.

### 3.3 Train/test split, and why it is non-negotiable

Fit a model and score it on the same data and you are measuring memory, not
learning. A flexible model can memorise every row, score perfectly, and be
useless on anything new.

So you split: fit on ~80%, score on the held-out 20% the model has never seen.

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
```

`random_state=42` fixes the shuffling so the split is identical every run.
**Always set it.** Without it your numbers move on every run and you cannot tell
a real improvement from noise.

### 3.4 Overfitting and underfitting

**Overfitting** — the model learns noise specific to the training data. Symptom:
excellent training score, poor test score.

**Underfitting** — the model is too simple to capture the real pattern. Symptom:
poor score on both.

This is the **bias–variance trade-off**. A very simple model (linear regression)
has high bias — strong assumptions, misses real structure. A very flexible model
(a deep decision tree) has high variance — it chases noise. The skill is landing
between them, and a random forest is specifically a device for cutting variance
without adding much bias (§4.2).

### 3.5 Cross-validation

A single split can be lucky. **K-fold cross-validation** splits the data into k
parts and fits k times, each part serving as the test set once. You get k scores
and report mean and standard deviation.

```python
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
print(f"{scores.mean():.4f} ± {scores.std():.4f}")
```

**The standard deviation matters.** If model A scores 0.75 ± 0.01 and model B
scores 0.76 ± 0.05, B is not reliably better.

> `shuffle=True` is important here. The table is ordered roughly by geography.
> Without shuffling, each fold becomes a different region and scores drop for
> reasons that have nothing to do with the model. I was caught by exactly this
> — see §7.4.

### 3.6 Grouped cross-validation — a subtlety that matters here

Ordinary k-fold assumes rows are independent. These are not: **links in the same
borough share fleet mix, road design and traffic patterns.**

With a random split the model sees the road next door during training and is
then tested on this one — a much easier task than the real one, which is
predicting in a borough you have never modelled.

`GroupKFold` prevents it by keeping all of a borough's links in the same fold:

```python
from sklearn.model_selection import GroupKFold
scores = cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5),
                         groups=links["Borough"], scoring="r2")
```

**The gap between the two numbers is real information** — it tells the client how
much optimism is baked into the headline figure. Report both.

### 3.7 Data leakage — the concept this project is built around

**Leakage** is information reaching the model that would not be available at
prediction time, or that trivially encodes the answer.

The textbook example: predicting whether a patient has a disease using a feature
called "was prescribed the disease medication." Perfect accuracy, zero value.

**This dataset contains exactly that trap** (§7.1). The short version: some
columns are the GLA's own inputs to computing the target. A model given them
scores ~0.99 and has learned nothing.

Leakage raises no error. It looks like success. That is what makes it dangerous.

### 3.8 Metrics for regression

| Metric | What it means | Range |
|---|---|---|
| **R²** | Fraction of variance explained. 1.0 = perfect, 0 = no better than predicting the mean, **negative = worse than the mean** | (−∞, 1] |
| **MAE** | Mean Absolute Error — average error size, in the target's own units | [0, ∞) |
| **RMSE** | Root Mean Squared Error — errors squared before averaging, so **large errors are punished disproportionately** | [0, ∞) |

**Why I insist on all three.** The target is severely skewed: the median link
emits **0.078 t/yr**, the maximum **43.8**. RMSE squares errors, so it is
dominated by a handful of motorway links. MAE weights every link equally, so it
describes the typical street.

**A model can be better on the typical case and worse on the extremes at the same
time.** In my results, on the leaked feature set, linear regression wins on R²
and RMSE while random forest more than halves the MAE. Neither is "the winner" —
they are better at different things, and which you want depends on the question.
Reporting R² alone hides this completely.

### 3.9 Metrics for classification

The hotspot classes are imbalanced: **346 of 3,460 cells (10%)**. That makes
accuracy actively misleading.

Proof from the baseline: a `DummyClassifier` that always predicts "not a
hotspot" scores **90.1% accuracy** and finds **zero** hotspots. Its recall and F1
are both **0.0**.

Use these instead:

- **Precision** — of the cells you flagged, what fraction really were hotspots?
  *(How much do I waste?)*
- **Recall** — of the real hotspots, what fraction did you find?
  *(How much do I miss?)*
- **F1** — harmonic mean of the two; one balanced summary number.
- **ROC-AUC** — probability the model ranks a random hotspot above a random
  non-hotspot. 0.5 = guessing, 1.0 = perfect.

Precision and recall **trade off**. Lower the decision threshold and you catch
more hotspots but raise more false alarms. §6.3 shows how to make that trade
explicitly instead of accepting scikit-learn's default of 0.5.

### 3.10 Why preprocessing must live inside a Pipeline

Scaling rescales features to comparable ranges. If you scale using statistics
from the *whole* dataset, information from the test set leaks into training — a
subtle leak that inflates your score.

A `Pipeline` prevents it by refitting the preprocessing **within each fold**:

```python
pipe = Pipeline([("pre", preprocessor), ("model", RandomForestRegressor())])
```

**Rule: never call `.fit_transform()` on your full dataset before splitting.**
Put preprocessing in the Pipeline — `laei.make_preprocessor(...)` returns one
already configured correctly.

---

## 4 · Why I chose these three algorithms

The heart of the project, and the section to read most carefully. Task 1 asks
*why*, and "it's popular" is not an answer.

### 4.0 The finding that drives every choice

During EDA I established one fact that determined everything else:

> **Emissions are a *product* of traffic and length, not a sum.**

The evidence:

| Predictor | Correlation with NOx |
|---|---:|
| Traffic volume (AADT) alone | 0.487 |
| Link length alone | 0.772 |
| **Traffic × length** | **0.990** |

This makes physical sense — emissions accumulate along a road. Double the
traffic *or* double the length and you roughly double the output:
`traffic × length × emission factor`.

**This single fact is why the project needs more than one kind of model**, and it
is the mechanism behind the headline result.

### 4.1 Multiple Linear Regression

#### The theory

Linear regression fits:

```
y = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ
```

choosing the coefficients β that minimise the sum of squared errors. There is a
closed-form solution — no iteration, no hyperparameters in the basic form. It is
the fastest thing you will fit (~**0.5 seconds** here).

**Key property: it can only *add* terms.** It cannot represent `x₁ × x₂` unless
you explicitly create that product as a new column.

#### Why I chose it for this data

**1. It matches how the inventory was actually built.** LAEI computes
`emissions = Σ(VKM_class × emission_factor_class)` — a weighted sum with roughly
constant weights, which is *exactly* the form linear regression fits. When the
VKM columns are supplied, linear regression is not a weak baseline; it is the
mathematically correct model of the underlying arithmetic.

**2. Interpretability.** A random forest can tell you diesel vans matter. Only a
linear model attaches a *number* to each vehicle class — the form a policy
appraisal needs.

**3. It is the reference point that makes everything else meaningful.** Without a
simple baseline, an ensemble scoring 0.99 is uninterpretable: you cannot tell
whether the problem is easy or the model is good. The entire headline result
depends on having this comparison.

#### Its weakness here — and this is the point

It cannot represent the product. Remove the VKM columns and linear regression
collapses from **0.9907 to 0.7503**, while random forest holds at **0.9880**. It
loses 0.24 of R²; the forest loses 0.002. **A 140× difference in damage from
removing one feature group.**

#### An honest caveat about the coefficients

I recommended linear regression partly for interpretable coefficients. When you
fit it you will find some are **physically impossible** — the coefficient on
articulated HGVs comes out *negative*, implying lorries reduce pollution.

That is not a bug in your code. It is **multicollinearity**: the vehicle-class
columns are strongly correlated with each other and with the total. When
predictors move together, the fit cannot cleanly attribute effect between them,
and individual coefficients become unstable and can flip sign.

**What to do:**

- Fit `Ridge` alongside plain `LinearRegression`. Ridge penalises large
  coefficients, which stabilises them. (In my run Ridge and OLS scored almost
  identically — the benefit is coefficient stability, not accuracy.)
- **Report coefficients as correlational, not causal.** "Associated with," never
  "causes."
- Cross-check the ranking against the forest's permutation importance. Where
  they agree — diesel cars, diesel LGVs and petrol cars are top-ranked in both —
  you can speak with more confidence.
- **Mention the collinearity explicitly in your write-up.** Spotting it earns
  more credit than quietly reporting an impossible number.

### 4.2 Random Forest

#### The theory

Start with a **decision tree**: a flowchart of yes/no questions. *"Is the road
longer than 100 m? If yes, is traffic above 20,000/day? …"* Each leaf holds a
prediction. Trees are easy to interpret but **unstable** — change the data
slightly and you get a very different tree. High variance.

A **random forest** fixes this with two ideas:

1. **Bagging** — train many trees, each on a random resample of the rows (drawn
   with replacement), then average their predictions.
2. **Feature randomness** — at each split consider only a random subset of
   features. This *decorrelates* the trees so their errors cancel rather than
   compound.

Averaging many noisy-but-unbiased predictors reduces variance without adding
much bias. That is the whole trick.

#### Why I chose it for this data — three concrete reasons

**1. It captures the product natively.** Decisive. A tree splits on length, then
splits on traffic *within each length band*. Nesting one split inside another is
how a tree represents an interaction — it gets `traffic × length` for free, with
no feature engineering. Precisely what linear regression cannot do.

**2. It is robust to the extreme skew.** Per-cell NOx spans 0.003 to 1,012
tonnes — five orders of magnitude. Trees split on **rank order**, not magnitude,
so one Heathrow-sized cell cannot drag the model the way it drags a least-squares
line.

**3. Minimal preprocessing, and it tolerates the collinearity.** No scaling
needed, mixed numeric and categorical inputs are fine, and the AADT/VKM
correlation that destabilises linear coefficients does not trouble a forest.

Plus `feature_importances_` gives an immediately presentable ranking.

#### Weaknesses you must state

**It is slow** — ~35–60 s per fit here against 0.5 s for linear regression.
Cross-validating 5 folds means 5 fits. Budget for it.

**It cannot extrapolate.** A forest predicts by averaging training values, so it
can never predict outside the range it has seen. As London's fleet electrifies,
future emissions fall *below* anything in the 2019 data and the forest floors
out. **For long-horizon forecasting I would recommend the linear model
instead** — the exact reverse of my 2019 recommendation. Saying this shows you
understand the tool rather than just the score.

**Impurity importance is biased** toward high-cardinality continuous features —
and `Link Length (m)` is exactly that. Always cross-check with
`permutation_importance` (§6.1).

### 4.3 K-Means clustering

#### The theory

K-means partitions observations into *k* groups. It:

1. places *k* centroids at random,
2. assigns every point to its nearest centroid,
3. moves each centroid to the mean of its assigned points,
4. repeats 2–3 until nothing moves.

It minimises within-cluster sum of squares. Because step 1 is random, results
vary between runs — so always set `n_init=10` (run 10 times, keep the best) and
`random_state=42`.

Distance is central to the algorithm, so **features must be on comparable
scales** or the largest-numbered feature dominates. These are pre-standardised in
`table_B_kmeans_matrix.npy`.

#### Why I chose it for this data

**It answers a question the other two cannot.** Regression tells you *how much*.
Clustering tells you **what kind of place this is** — and that determines which
policy lever applies. Two cells emitting identical NOx may need completely
different interventions if one is road-dominated and the other sits under a
flight path.

**The data has exactly the right shape.** Each cell carries a 16-dimensional
breakdown across emission sectors — a natural profile to cluster on.

**I clustered on *shares*, not tonnes.** A deliberate design decision. Clustering
raw magnitudes would just re-discover "big cell / small cell," which the Lorenz
curve already showed. Dividing each sector by the cell total asks the question
that actually matters: *what is the mix here?*

**I deliberately withheld the coordinates.** `Easting`/`Northing` are in the CSV
for plotting, but they are **not** in the clustering features. This buys a
validation that would otherwise be impossible: if geography emerges anyway, the
structure is real. It did — see §8.

#### Weaknesses — and how to report them honestly

**You must choose *k*, and this data does not choose it for you.** The silhouette
scores across k = 2…10 run 0.250, 0.272, 0.272, 0.271, 0.284, 0.255, 0.278,
0.299, 0.321 — all low (above ~0.5 would indicate strong separation) and
**rising at the top end**.

> **A trap for beginners.** Silhouette rising as k grows does *not* mean k=10 is
> best. More clusters mean smaller, tighter groups almost mechanically. Pick k by
> maximising silhouette and you will keep raising k for no real gain.

I chose **k = 4** for interpretability, and I say so plainly rather than
implying a metric decided it. The honest reading is that London's cells form a
**continuum** of source mixes rather than four separate islands — DBSCAN finds no
dense clusters at any setting, which supports this. The clustering is a useful
*summary* of that continuum, not a discovery of natural kinds.

**Other assumptions:** k-means assumes roughly spherical, similarly-sized
clusters. These are wildly unequal (69 / 1,253 / 2,071 / 67 cells). Cross-check
with `AgglomerativeClustering` — in my run it agreed only partially (ARI ≈ 0.39),
concurring on the small distinctive clusters and dividing the big mass
differently, which is consistent with the continuum reading.

### 4.4 The fourth model — a benchmark, not a headline

I also fit **HistGradientBoosting**. Where a forest builds trees *in parallel*
and averages, **boosting** builds them *sequentially*, each new tree correcting
the previous ensemble's errors. It is usually the strongest tabular learner and
is far faster here (0.8 s vs 32 s).

**It lost on this data**, scoring 0.903 against the forest's 0.988. The reason
matters: it **bins** each continuous feature into ~255 buckets before splitting.
`Link Length (m)` has a median of 43 m over a range to 3,770 m and is the
dominant predictor — binning a long-tailed feature that dominates the target
costs exactly the resolution the model needs on short links.

**Being able to explain why the fashionable model lost is a better result than
having it win.** Include it.

---

## 5 · The prepared data

### 5.1 What is in `data/processed/`

| File | Unit | Rows | For |
|---|---|---:|---|
| `table_A_links.csv` | one road link | 79,388 | Linear Regression, Random Forest (regression) |
| `table_B_grid_source_mix.csv` | one 1 km cell | 3,460 | K-Means |
| `table_B_kmeans_matrix.npy` | one 1 km cell | 3,460 | K-Means — pre-standardised, fit-ready |
| `table_C_grid_nonleaky.csv` | one 1 km cell | 3,460 | Classification + regression |
| `table_C_LEAKY_reference.csv` | one 1 km cell | 3,460 | **leakage demonstration only** |
| `manifest.json` | — | — | machine-readable description of all of the above |

`grid_all_years.csv`, `link_features.csv` and `link_targets.csv` are intermediate
exports. You want the `table_*` files.

### 5.2 Table A — road links (56 columns)

**Identifier**

| Column | Meaning |
|---|---|
| `TOID` | Ordnance Survey ID for the segment — a unique key, **not a feature** |

**Categorical features** (`laei.LINK_CATEGORICAL`)

| Column | Meaning |
|---|---|
| `LAEI Zone` | Central / Inner / Outer London / Non-GLA |
| `Borough` | 33 London boroughs plus "Non-GLA" |
| `Road Classification` | A Road, B Road, C/Unclassified, M25, M1, M4, … |

**Numeric — traffic counts (20 columns).** `AADT` = *Annual Average Daily
Traffic*, vehicles per day, one column per vehicle class:
`AADT Motorcycle`, `AADT Taxi`, `AADT Petrol Car`, `AADT Diesel Car`,
`AADT Electric Car`, the same petrol/diesel/electric split for `PHV` (private
hire — minicabs and Ubers) and `LGV` (light goods — vans), six HGV classes split
by rigid/articulated and axle count, then `AADT 2019 - Buses`,
`AADT 2019 - Coaches` and `AADT 2019 - Total`.

**Numeric — road conditions (3 columns).**
`Speed (km/hr) - Except Buses`, `Speed (km/hr) - Buses Only`, `Link Length (m)`.

**Numeric — VKM (20 columns) ⚠️** `VKM` = *Vehicle Kilometres*, one mirroring
each AADT column. **`VKM ≈ AADT × Link Length × 365`.** These are the leaked
columns — see §7.1.

**Derived flags I added**

| Column | Meaning |
|---|---|
| `zone_canon` | `LAEI Zone` normalised to match the grid tables' spelling |
| `is_motorway` | 1 if the road class matches M-something |
| `no_traffic_data` | 1 for the 688 links whose AADT total was a placeholder |
| `bus_speed_missing` | 1 for the 26,979 links with no buses-only speed |

**Targets**

| Column | Meaning |
|---|---|
| `nox` | **primary target** — NOx, tonnes/year |
| `pm10`, `pm25`, `co2` | alternative targets |
| `nox_per_m` | NOx per metre — the *intensity* target (§6.4) |

### 5.3 Table B — grid cells for clustering (25 columns)

`Grid ID 2019` (key), `Easting` / `Northing` (**plotting and validation only — do
not use as features**), `Borough`, `Zone`, then 16 `share_<sector>` columns that
sum to 1.0 per row, then `nox` / `pm10` / `pm2.5` / `co2` totals.

Fit on `table_B_kmeans_matrix.npy` — the 16 share columns, already standardised:

```python
import numpy as np
X = np.load("../data/processed/table_B_kmeans_matrix.npy")   # (3460, 16)
```

> Five sectors — Biomass, Commercial Cooking, Forestry, Gas Leakage,
> Resuspension — contribute **zero NOx everywhere**. They are retained so the
> feature space matches the published sector list, but they cannot influence the
> partition. Mention this in your write-up.

### 5.4 Table C — grid cells for classification (25 columns)

Features are the 16 `co2_<sector>` columns plus `Easting` and `Northing` (18
total — here coordinates **are** legitimate features).

Targets: `nox` (regression) and `nox_hotspot` (binary — 1 if the cell is in the
top decile of NOx; threshold **27.52 t/yr**, giving 346 positives).

**Why CO2 features to predict NOx?** Because the obvious construction is leaky.
Predicting total NOx from NOx-by-sector is just addition — those 16 columns *sum
to* the target.

CO2 by sector instead proxies **activity and fuel throughput**. NOx additionally
depends on **combustion technology and abatement** — how cleanly that fuel is
burned. So the model answers a genuine question: *given how much energy is burned
here and where here is, how dirty is the burning?*

**And the imperfection is the deliverable.** This model reaches only R² ≈ 0.80.
That shortfall *is* the technology signal: cells whose actual NOx most exceeds
the CO2-based prediction are burning dirtier than their energy use implies. That
ranked residual list is arguably the most actionable output of the project. **Do
not treat the middling R² as a failure to fix.**

### 5.5 Reading the manifest

```python
manifest = json.load(open("../data/processed/manifest.json"))
manifest["tables"]["table_C_grid_nonleaky.csv"]["features"]   # exact feature list
manifest["modelling_protocol"]                                # the agreed protocol
```

Use it rather than hardcoding column lists — if the preparation is re-run, the
manifest updates and your code follows.

---

## 6 · Implementation walkthroughs

### 6.1 Model 1 — link-level regression

**Run this twice**, once per feature set. The comparison *is* the result.

```python
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import sys, time; sys.path.append("../src")
import pandas as pd, numpy as np, laei

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline

links = pd.read_csv("../data/processed/table_A_links.csv", low_memory=False)
fs    = laei.link_feature_sets(links)
CAT   = laei.LINK_CATEGORICAL
y     = links["nox"]

def models():
    return {
        "LinearRegression":     LinearRegression(),
        "Ridge":                Ridge(alpha=1.0, random_state=42),
        "RandomForest":         RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                      random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300,
                                                              random_state=42),
    }

rows = []
for fs_name, num in fs.items():                     # 'full' then 'realistic'
    X = links[num + CAT]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # ALWAYS fit a baseline first — it gives your R2 a reference point
    dummy = DummyRegressor(strategy="median").fit(X_tr, y_tr)
    print(fs_name, "baseline:", laei.regression_metrics(y_te, dummy.predict(X_te)))

    for name, model in models().items():
        t0 = time.time()
        pipe = Pipeline([("pre", laei.make_preprocessor(num, CAT)),
                         ("model", model)]).fit(X_tr, y_tr)
        met = laei.regression_metrics(y_te, pipe.predict(X_te))
        rows.append({"feature_set": fs_name, "model": name, **met,
                     "fit_s": round(time.time() - t0, 1)})
        print(f"  {name:<21} R2={met['R2']:.4f} MAE={met['MAE']:.4f} "
              f"RMSE={met['RMSE']:.4f}")

results = pd.DataFrame(rows)
```

**Then cross-validate**, because a single split is not evidence:

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for fs_name, num in fs.items():
    for name, model in models().items():
        pipe = Pipeline([("pre", laei.make_preprocessor(num, CAT)), ("model", model)])
        s = cross_val_score(pipe, links[num + CAT], y, cv=cv, scoring="r2")
        print(f"{fs_name:>10} {name:<21} {s.mean():.4f} ± {s.std():.4f}")
```

**Then check spatial optimism:**

```python
from sklearn.model_selection import GroupKFold
num  = fs["realistic"]
pipe = Pipeline([("pre", laei.make_preprocessor(num, CAT)),
                 ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                                 random_state=42))])
s = cross_val_score(pipe, links[num + CAT], y, cv=GroupKFold(n_splits=5),
                    groups=links["Borough"], scoring="r2")
print(f"by borough: {s.mean():.4f}")
```

**Then feature importance — both kinds:**

```python
from sklearn.inspection import permutation_importance

X_tr, X_te, y_tr, y_te = train_test_split(links[num + CAT], y,
                                          test_size=0.2, random_state=42)
rf = Pipeline([("pre", laei.make_preprocessor(num, CAT)),
               ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                               random_state=42))]).fit(X_tr, y_tr)

ohe      = rf.named_steps["pre"].named_transformers_["cat"]
names    = num + list(ohe.get_feature_names_out(CAT))
impurity = pd.Series(rf.named_steps["model"].feature_importances_,
                     index=names).sort_values(ascending=False)

# permutation importance is slower — subsample the test set
sample = X_te.sample(6000, random_state=42)
perm   = permutation_importance(rf, sample, y_te.loc[sample.index],
                                n_repeats=5, random_state=42, scoring="r2")
perm_s = pd.Series(perm.importances_mean,
                   index=num + CAT).sort_values(ascending=False)
```

**Timing.** Forest fits are ~35 s each. Hold-out + 5-fold CV + GroupKFold +
permutation importance runs about **35–45 minutes** total. Start it and do
something else.

### 6.2 Model 2 — clustering

```python
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

table_b = pd.read_csv("../data/processed/table_B_grid_source_mix.csv")
X = np.load("../data/processed/table_B_kmeans_matrix.npy")   # already scaled

# 1. scan k — report the scan, do not just pick the max
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"k={k}  inertia={km.inertia_:>8.0f}  "
          f"silhouette={silhouette_score(X, km.labels_):.3f}")

# 2. fit the chosen k
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
table_b["cluster"] = km.labels_

# 3. INTERPRET the clusters — this is the actual deliverable
share_cols = [c for c in table_b.columns if c.startswith("share_")]
profile = table_b.groupby("cluster")[share_cols].mean() * 100
print(profile.round(1).T.sort_values(0, ascending=False).head(8))

# 4. validate spatially — coordinates were NOT features
from sklearn.neighbors import NearestNeighbors
coords = table_b[["Easting", "Northing"]].to_numpy()
_, idx = NearestNeighbors(n_neighbors=9).fit(coords).kneighbors(coords)
neigh  = idx[:, 1:]                       # drop self
lab    = table_b["cluster"].to_numpy()
observed = (lab[neigh] == lab[:, None]).mean()

rng  = np.random.default_rng(42)
null = np.array([(lambda s: (s[neigh] == s[:, None]).mean())(rng.permutation(lab))
                 for _ in range(200)])
print(f"neighbours agree {observed:.3f} vs {null.mean():.3f} shuffled, "
      f"z={(observed - null.mean()) / null.std():.1f}")

# 5. robustness
agg = AgglomerativeClustering(n_clusters=4).fit(X)
print("ARI vs KMeans:", round(adjusted_rand_score(km.labels_, agg.labels_), 3))
```

**The interpretation step is not optional.** A cluster ID is worthless to a
stakeholder; *"these 69 cells are aviation-dominated, so ULEZ will not touch
them"* is the deliverable.

### 6.3 Model 3 — hotspot classification

```python
import json, pandas as pd, numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

table_c  = pd.read_csv("../data/processed/table_C_grid_nonleaky.csv")
manifest = json.load(open("../data/processed/manifest.json"))
FEATS    = manifest["tables"]["table_C_grid_nonleaky.csv"]["features"]

X, y = table_c[FEATS], table_c["nox_hotspot"]

# stratify=y keeps the 10% positive rate in BOTH halves — essential when imbalanced
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

# the baseline that proves accuracy is the wrong metric
dum = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print("dummy:", laei.classification_metrics(y_te, dum.predict(X_te)))
# ~90% accuracy, 0.0 recall

models = {
    "LogisticRegression": Pipeline([
        ("sc", StandardScaler()),
        ("m", LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=42))]),
    "RandomForest": RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                           class_weight="balanced",
                                           random_state=42),
}
probas = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    probas[name] = m.predict_proba(X_te)[:, 1]
    print(name, laei.classification_metrics(y_te, m.predict(X_te), probas[name]))
```

**Then tune the threshold** — which matters more here than the choice of model:

```python
COST_MISS, COST_FALSE_ALARM = 5.0, 1.0   # a missed hotspot costs 5 wasted visits

for name, p in probas.items():
    best = None
    for t in np.arange(0.05, 0.96, 0.05):
        tn, fp, fn, tp = confusion_matrix(y_te, (p >= t).astype(int),
                                          labels=[0, 1]).ravel()
        cost = COST_MISS * fn + COST_FALSE_ALARM * fp
        if best is None or cost < best[1]:
            best = (t, cost, fn, fp)
    print(f"{name}: best threshold {best[0]:.2f}, cost {best[1]:.0f} "
          f"(missed {best[2]}, false alarms {best[3]})")
```

In my run the cost-optimal thresholds were **0.65** and **0.35** — neither is
the default 0.5. **Do not declare a winner on ROC-AUC without naming the decision
it feeds.**

### 6.4 Optional but valuable — the intensity target

```python
mask   = links["nox_per_m"].notna()
X2, y2 = links.loc[mask, num + CAT], links.loc[mask, "nox_per_m"]
```

Total emissions measure *quantity* — a long quiet road out-emits a short busy
one, which is why `Link Length` dominates every importance ranking. If the
client's real interest is roadside **exposure**, emissions **per metre** is the
right target.

I found the forest predicts it just as well (**R² 0.990**), and linear
regression *improves* to **0.912**, because dividing by length removes the
product it could not represent. **A modelling choice, not a trade-off** — and a
genuinely useful thing to tell the client.

---

## 7 · Pitfalls — the mistakes I actually made

### 7.1 The VKM leakage trap ⚠️ most important

`VKM = AADT × Link Length × 365`, and LAEI computes the target *from* VKM.

Give a model the VKM columns and it scores ~0.99 while doing arithmetic, not
prediction. **I include this deliberately, clearly labelled, as a leakage
demonstration** — showing you can detect leakage is a stronger result than
quietly avoiding it.

**Rules:**

- Headline the `realistic` feature set. That is the honest evaluation.
- Report `full` **only** as a labelled demonstration.
- Never present a `full`-set score as your model's performance.

The same trap exists in `table_C_LEAKY_reference.csv`, whose 16 features sum to
its target. Run it, show it scores ~0.99, and explain why that is worthless.

### 7.2 `-` means zero, not missing

41 of 43 numeric traffic columns use `-` as a placeholder. The instinctive
response — coerce to `NaN`, median-impute — **would invent traffic on 42% of the
road network.**

I verified the semantics directly: on 33,705 links with a numeric total and at
least one dashed class, reading `-` as **zero** makes the classes sum to the
stated total within 3 vehicles — the same rounding tolerance as the 45,043 links
with no dashes at all.

**This is already handled.** Counts are zero-filled; the two Speed columns (where
`-` genuinely does mean missing) are left `NaN` for median imputation, with
`bus_speed_missing` as an indicator. **Do not add imputation on top.**

### 7.3 Accuracy on imbalanced classes

Covered in §3.9, but it bears repeating because it is the easiest way to produce
a confidently wrong report. **90.1% accuracy, zero hotspots found.** Quote
precision, recall and F1.

### 7.4 Unshuffled cross-validation

I hit this one. `GridSearchCV` with `cv=3` uses an **unshuffled** KFold by
default. Because the table is ordered roughly by geography, each fold became a
different region — and tuning appeared to make the model *worse*.

**Always pass a shuffled splitter explicitly:**

```python
GridSearchCV(pipe, grid, cv=KFold(3, shuffle=True, random_state=42))
```

And when you compare two numbers, **make sure they came from the same CV
scheme.** Comparing a 3-fold unshuffled score against a 5-fold shuffled one is
meaningless.

### 7.5 Do not re-derive what is already prepared

Four decisions are deliberate. Undoing them will silently change your results:

1. **Do not add VKM to the `realistic` set** (§7.1).
2. **Do not add coordinates to the K-Means features** — they are in the CSV for
   plotting. Geography emerging without them *is* the validation.
3. **Do not report the leaky table as a result.**
4. **Do not mean-impute the pollutant columns.** A null in the grid data means
   "not estimated for this source," not "measured as zero."

### 7.6 Budget your compute

A forest fit is ~35 s here. A 3×3×3 grid search with 5-fold CV is 135 fits —
about 80 minutes. **Keep grids small.** I found tuning moved the score by less
than 0.002 because the defaults were already near the ceiling; effort is better
spent on features than on hyperparameters.

### 7.7 Watch the library versions

`dba` has pandas 3.0 and scikit-learn 1.9, newer than most tutorials assume. Two
differences you will meet: `matplotlib`'s `boxplot(labels=...)` is now
`tick_labels=`, and string columns load as `str` dtype rather than `object`.

---

## 8 · Checking your work

### 8.1 Numbers you should reproduce

If your figures differ materially, something is wrong — investigate before
writing anything up.

**Regression, per-link NOx (test_size=0.2, random_state=42):**

| Model | With VKM | Without VKM |
|---|---:|---:|
| LinearRegression | R² 0.9907 | R² 0.7503 |
| Ridge | R² 0.9907 | R² 0.7503 |
| RandomForest | R² 0.9897 | R² 0.9880 |
| HistGradientBoosting | R² 0.9494 | R² 0.9033 |

Baseline `DummyRegressor(median)`: R² **−0.026**, MAE **0.2215**.
(A negative R² is correct — the median predictor is worse than the mean.)

GroupKFold by borough: LinearRegression **0.756 → 0.580**,
RandomForest **0.987 → 0.973**.

**Classification, hotspot:**

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.982 | 0.599 | 0.953 | 0.735 |
| RandomForest | 0.989 | 0.766 | 0.837 | 0.800 |

Leaky variant: ROC-AUC ≈ **0.995–0.998**. Dummy: **90.1%** accuracy, **0.0**
recall.

**Clustering, k=4:** sizes **69 / 1,253 / 2,071 / 67**, silhouette **0.272**,
neighbour agreement **0.588** vs **0.508** shuffled (z ≈ 23).

### 8.2 Before you report anything

- [ ] Every metric comes from cross-validation, reported as mean ± std
- [ ] Dummy baselines included for both regression and classification
- [ ] `random_state=42` on every split, model and clustering run
- [ ] Preprocessing is **inside** the Pipeline
- [ ] R², MAE **and** RMSE reported together for regression
- [ ] Precision, recall, F1 — not accuracy — for classification
- [ ] The leaky model is labelled as a demonstration
- [ ] `GroupKFold` gap reported as the honest deployment estimate
- [ ] Clusters **interpreted**, not just numbered
- [ ] Threshold choice justified by cost, not left at 0.5

### 8.3 What Task 4 is actually asking

*"Explain in detail any differences in model performance."* Not *which model
won* — **why they differ**. The five mechanisms available to you:

1. **Feature structure** — VKM linearises a multiplicative problem, which is why
   the ranking flips.
2. **Target skew** — MAE and RMSE can disagree, and each answers a different
   question.
3. **Spatial correlation** — random splits flatter every model.
4. **Cost asymmetry** — two classifiers with the same AUC behave completely
   differently.
5. **Binning** — why the fastest, most fashionable learner lost.

---

## 9 · Glossary

| Term | Meaning |
|---|---|
| **AADT** | Annual Average Daily Traffic — vehicles per day |
| **VKM** | Vehicle Kilometres per year ≈ AADT × length × 365 ⚠️ leaked |
| **NOx** | Nitrogen oxides — the main traffic-related pollutant |
| **PM10 / PM2.5** | Particulate matter under 10 / 2.5 micrometres |
| **LGV / HGV / PHV** | Light goods (van) / Heavy goods (lorry) / Private hire vehicle |
| **Bagging** | Training many models on resampled data and averaging them |
| **Boosting** | Training models sequentially, each fixing the last one's errors |
| **Leakage** | Information reaching the model that trivially encodes the answer |
| **Overfitting** | Learning noise — good on training data, poor on new data |
| **Stratified split** | Splitting so each half keeps the same class proportions |
| **Silhouette** | Cluster-quality score; >0.5 indicates strong separation |
| **ARI** | Adjusted Rand Index — agreement between two clusterings |
| **Precision** | Of what you flagged, how much was right |
| **Recall** | Of what was really there, how much you found |
| **ROC-AUC** | Probability a random positive ranks above a random negative |

---

## 10 · Where to look next

- `reports/Task1_Task2_Report.md` — the full justification for the algorithm
  choices and the eight predictive insights the models should confirm or refute.
- `reports/Task3_Task4_Report.md` — the completed performance assessment,
  generated from `reports/results/*.json`.
- `notebooks/02_eda_and_profiling.ipynb` — the EDA, including all five
  data-quality findings.
- `notebooks/03`–`06` — reference implementations of everything above.
- `src/laei.py` — short, and worth reading in full.
- `RUNBOOK.md` — end-to-end project runbook.

**Optional extension.** The grid workbook holds GLA forecasts for 2025 and 2030,
and the link workbook holds `Road-Total-2025` / `-2030`. A model trained on 2019
can be scored against the GLA's own 2025 figures — a genuine out-of-sample test
needing no extra data. `laei.load_grid(year=None)` returns all five years.

---

*Ask about anything unclear rather than guessing — a wrong assumption about what
a column means is far more expensive than a question.*
