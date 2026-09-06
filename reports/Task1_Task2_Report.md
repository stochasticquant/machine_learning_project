# Machine Learning Approach and Expected Predictive Insights

## LAEI 2019 Atmospheric Emissions Modelling

**Prepared for:** Department of Environment
**Prepared by:** Athana Data Science Services — Module End Project group
**Covers:** Task 1 (algorithm selection and justification) and Task 2 (expected predictive insights)
**Dataset:** London Atmospheric Emissions Inventory (LAEI) 2019, London Datastore (`e758q`)
**Supporting analysis:** `notebooks/01_eda_and_data_preparation.ipynb`

---

## Executive summary

We have profiled the LAEI 2019 release in full and selected **three machine
learning techniques**: **Multiple Linear Regression**, **Random Forest** (used
for both regression and classification), and **K-Means clustering**. Together
they cover the three questions the Department actually needs answered — *how
much is emitted, where should we intervene,* and *what kind of place is this* —
and they are chosen against measured properties of these specific files rather
than by general reputation.

Three findings from the exploratory analysis drive the selection:

1. **Emissions are a multiplicative function of traffic flow and road length.**
   LAEI computes link emissions as `Σ(VKM × emission factor)`, and
   `VKM = AADT × length × 365`. A purely additive model cannot represent this;
   a tree ensemble can. This is the central methodological point of the project.
2. **Every target is severely right-skewed.** Per-cell NOx has a median of
   7.49 t/yr against a maximum of 1,012 t/yr. This rules out mean imputation,
   makes R² a misleading single metric, and favours rank-based learners.
3. **Emissions are highly concentrated.** A small minority of 1 km cells carry
   a disproportionate share of London's NOx, which makes *targeting* — a
   classification problem — more operationally valuable than precise
   quantity prediction.

We also identify a **target-leakage trap** in the obvious modelling setup and
recommend running it deliberately, clearly labelled, as a demonstration.

---

# Task 1 — Machine learning techniques selected

## Context: what kind of learning problem is this?

The LAEI is a **modelled inventory, not a set of measurements**. Emissions are
reported in tonnes/year on two grains:

- **79,437 major road links**, each with traffic counts by 20 vehicle classes,
  speeds and geometry — a natural **supervised regression** problem.
- **3,460 one-kilometre grid cells**, each decomposed across 16 emission
  sectors and 39 sources — suitable for both **segmentation** and
  **hotspot classification**.

There is no time dimension below one year, so time-series methods are not
applicable to the 2019 baseline. Neural networks were considered and rejected:
with 3,460 cells at grid level, and a tabular problem with strong monotone
structure at link level, they would add opacity and training cost without
accuracy benefit — and stakeholder interpretability is an explicit project
requirement.

---

## 1. Multiple Linear Regression

### What it does

Fits a linear combination of predictors, `y = β₀ + Σβᵢxᵢ`, minimising the sum
of squared residuals. It has a closed-form solution, no hyperparameters in its
basic form, and produces one coefficient per predictor.

### Why we selected it for this data

**It matches how the inventory is actually constructed.** LAEI derives link
emissions as `Σ (VKM_class × emission_factor_class)` — a weighted sum with
approximately constant weights. When vehicle-kilometre columns are supplied as
features, the target is *by construction* close to linear in them. Linear
regression is therefore not a strawman baseline here; it is the functional form
that mirrors the data-generating process.

**Its coefficients are the deliverable, not a by-product.** A tree ensemble can
tell the Department that diesel LGV traffic matters; only a linear model states
*by how much*, in units a policy team can use — "one additional million diesel
LGV vehicle-kilometres corresponds to X tonnes of NOx per year." For appraising
interventions such as fleet renewal or a low-emission zone, that marginal
effect is precisely the quantity required.

**It is the reference point that makes the other models interpretable.** Without
a linear baseline, an R² of 0.99 from an ensemble is uninterpretable — we cannot
tell whether the problem is hard or the model is good. Our analysis shows the
answer differs entirely depending on which features are supplied.

### Known weaknesses in this application

- **It cannot represent products of features.** Emissions depend on
  `AADT × length`; a linear model in those two terms cannot express their
  product without an explicit interaction being engineered.
- **Least squares is sensitive to the extreme right tail.** A handful of
  motorway links and the Heathrow cell exert leverage far beyond their number.
- **Residuals are heteroscedastic** — error variance grows with fitted value —
  so the standard errors on coefficients should be treated with caution.

We will fit **Ridge** alongside ordinary least squares, since the AADT and VKM
columns are strongly collinear by construction (VKM is a deterministic function
of AADT and length), which inflates coefficient variance.

---

## 2. Random Forest

### What it does

Fits many decision trees, each on a bootstrap resample of the rows and
considering a random subset of features at each split, then averages their
predictions (or takes a majority vote for classification). It is
non-parametric and makes no distributional assumption about the target.

### Why we selected it for this data

**It represents the multiplicative structure natively.** This is the decisive
reason. Because emissions scale with `AADT × length`, a model must capture an
interaction. A decision tree does this by construction: it splits on link
length, then splits on AADT *within* each length band, approximating the
product through recursive partitioning. No feature engineering is required.
Our exploratory analysis quantifies the effect — the correlation of `AADT`
alone with NOx is modest, but the correlation of `AADT × length` is far
stronger.

**It is robust to the severe skew we measured.** Trees split on rank order, not
magnitude, so the 1,012 t/yr Heathrow cell cannot distort the model the way it
distorts a least-squares fit. Given that per-cell NOx spans roughly five orders
of magnitude, this robustness is not a nicety but a requirement.

**It handles the practical shape of the data with minimal preprocessing** —
mixed numeric and categorical predictors, no scaling requirement, and tolerance
of the collinearity between the AADT and VKM blocks that would destabilise a
linear model.

**It serves both supervised tasks.** The same algorithm family provides
`RandomForestRegressor` for tonnage prediction and `RandomForestClassifier` for
hotspot identification, which keeps the methodology coherent across the two
supervised deliverables.

**`feature_importances_` gives a directly presentable output** — a ranked
attribution over vehicle classes and road attributes that answers the
Department's "what should we target first?" question in one chart.

### Known weaknesses in this application

- **Computationally heavy.** Fitting is roughly two orders of magnitude slower
  than linear regression on the 79,000-link table. This matters when
  cross-validating or grid-searching, and must be budgeted for.
- **It cannot extrapolate.** Predictions are averages of observed training
  values, so the model cannot predict below the lowest emission level it has
  seen. As London's fleet electrifies, future emissions will fall outside the
  2019 range and the model will floor out — a serious caveat for the client's
  eventual multi-decade ambition, and one we flag now rather than later.
- **Impurity-based importance is biased** toward high-cardinality continuous
  features. We will cross-check with `permutation_importance`.

We will also fit **Histogram Gradient Boosting** as a benchmark within this
family. It is typically the strongest tabular learner and is far faster, but we
expect its feature binning to lose resolution on link length, whose
distribution is long-tailed with a median of only 43 m.

---

## 3. K-Means clustering

### What it does

Partitions observations into *k* groups, each represented by a centroid,
iteratively minimising within-cluster sum of squares. Unsupervised: no target
variable.

### Why we selected it for this data

**It answers a different and complementary question.** The two supervised models
answer *"how much is emitted here?"*. K-Means answers **"what kind of place is
this?"** — segmenting the 3,460 grid cells by their **emission source profile**.
This matters operationally: two cells with identical NOx totals may require
completely different interventions if one is road-dominated and the other is
dominated by aviation or river traffic. A regression model cannot express that
distinction; a segmentation can.

**The data has exactly the structure clustering needs.** Each cell carries a
16-dimensional sector decomposition. Expressed as *within-cell shares* rather
than tonnes, these form a compositional profile — a natural clustering space.
Using shares rather than magnitudes is a deliberate design choice: clustering on
raw tonnes would simply rediscover "large cell / small cell", which we already
know.

**It provides an independent validation opportunity.** We deliberately withhold
Easting and Northing from the clustering features. If the resulting clusters
nonetheless align with known geography — airports, the Thames corridor, the
central road network — that emergent coherence is strong evidence the
segmentation reflects real structure rather than arbitrary partitioning. This
is a more persuasive validation for a non-technical audience than any internal
metric.

**It scales to the client's eventual 50-year brief.** K-Means is O(nkd) per
iteration and supports `MiniBatchKMeans` for far larger inputs, so the approach
transfers directly to the full inventory series.

### Known weaknesses in this application

- ***k* must be chosen, and the data does not choose it cleanly.** Our
  exploratory scan across k = 2…8 shows a **flat silhouette profile** with no
  clear elbow. We will therefore select *k* on **interpretability and policy
  usefulness**, and state that openly rather than implying a metric determined
  it.
- **It assumes roughly spherical, comparably sized clusters** in the scaled
  space. Emission archetypes are unlikely to be perfectly spherical, and some
  archetypes are genuinely rare. We will report cluster sizes honestly and
  compare against **Agglomerative Clustering** and **DBSCAN** as robustness
  checks.
- **It is sensitive to initialisation**, so `n_init=10` and a fixed
  `random_state` are used throughout.

---

## Why these three together

| Question the Department has | Technique | Output |
|---|---|---|
| How much does this road emit, and what drives it? | Linear Regression | Coefficients as marginal effects per vehicle class |
| Can we predict emissions where the relationship is non-linear? | Random Forest (regression) | Accurate prediction + feature importance ranking |
| Which cells should we inspect first? | Random Forest (classification) | Ranked hotspot probabilities |
| What kind of place is this, and which lever applies? | K-Means | Interpretable emission archetypes |

The three span **parametric and non-parametric**, **supervised and
unsupervised**, and **prediction and segmentation** — which also satisfies the
methodological breadth the brief asks for. Critically, the Linear Regression /
Random Forest pairing is not redundant: the *comparison between them* is our
principal analytical result, because it isolates exactly which structural
property of the data each can and cannot represent.

---

# Task 2 — Predictive insights expected from the data

Every claim below is grounded in inspection of the actual 2019 files and is
stated so that it can be **confirmed or refuted** by the modelling stage. Each
names the deliverable it produces for the Department.

## Insight 1 — A quantified emissions model of London's road network

**Expected.** A model predicting per-link NOx (tonnes/year) from traffic counts,
speed, road class and geometry, with per-vehicle-class marginal effects
attached.

**Basis.** 79,388 usable road links after joining traffic to emissions on the OS
`TOID`, with 23 traffic predictors plus geometry available in the realistic
(non-leaked) feature set.

**Deliverable.** A ranked table of NOx tonnes attributable per unit of traffic
for each of 20 vehicle classes, allowing the Department to compare, for example,
diesel LGV against diesel car interventions on a common basis.

## Insight 2 — Which vehicle classes deserve policy attention first

**Expected.** Diesel light goods vehicles and diesel cars will dominate the NOx
attribution, and — this is the specific, testable part — **diesel LGVs will rank
above diesel cars despite far lower traffic volumes**, indicating a
disproportionate per-vehicle contribution.

**Basis.** The vehicle-class composition of the AADT columns combined with the
correlation structure against the NOx target observed in the exploratory
analysis.

**Deliverable.** A "which vehicle first" chart for the video, and an estimate of
NOx avoided per percentage point of fleet electrification — validated against the
GLA's own 2025 and 2030 forecast columns, which are already present in the files.

## Insight 3 — Emissions measure *quantity*, not *intensity* — and the client may want the other one

**Expected.** Link length will be the single strongest predictor of total link
emissions, well ahead of traffic volume.

**Basis.** Total emissions are an extensive quantity: a long, quiet road
accumulates more tonnes per year than a short, busy one. Our analysis compares
the top-ranked links under `nox` against `nox_per_metre` and finds the two
rankings materially different.

**Why this matters — and it is the most important caveat we will raise.** If the
Department's underlying interest is **roadside exposure**, total emissions is the
wrong target and **emissions per metre** is the right one. We will present both
and recommend explicitly which to use for which decision. Identifying this kind
of specification mismatch early is exactly what the consultancy is for.

## Insight 4 — Emissions are concentrated, so triage is tractable

**Expected.** A small minority of 1 km cells carry a large share of London's
NOx, and cells in the top decile can be identified from *non-NOx* predictors
(sector CO2 plus location) with high discrimination.

**Basis.** Per-cell NOx has median 7.49 t/yr, mean 13.12 t/yr and maximum
1,012 t/yr — a mean/median ratio near 1.8 and a maximum roughly 135× the median.
The 90th-percentile threshold sits at 27.52 t/yr, giving 346 hotspot cells of
3,460.

**Deliverable.** A ranked inspection list. Operationally, this means a borough
holding only energy and activity data can prioritise its monitoring budget
*before* commissioning NOx measurement.

## Insight 5 — Four distinct emission archetypes, recoverable without geography

**Expected.** Clustering on sector shares will separate cells into
interpretable profiles — a road-transport-dominated majority, a
heating-dominated suburban group, and small but distinct aviation-dominated and
river/shipping-dominated groups.

**Basis.** The grid table decomposes every cell across 16 sectors, and the
London-wide sector shares differ markedly by pollutant. Aviation and river
sectors are present in the data with sufficient mass to form separable groups.

**Deliverable — and its policy payload.** A map of London by emission archetype,
accompanied by the statement that **road-focused interventions such as ULEZ
cannot move the aviation- and river-dominated cells at all.** For those cells a
different instrument is required. Because coordinates are excluded from the
clustering, any geographic coherence in the result is emergent and therefore
strong evidence.

## Insight 6 — The gap between CO2 and NOx is the technology signal

**Expected.** Predicting cell NOx from sector CO2 and location will achieve
moderate but clearly imperfect accuracy — we anticipate R² in the region of
0.75–0.80, not above 0.95.

**Basis.** CO2 tracks *how much* fuel is burned; NOx additionally depends on
*how cleanly* it is burned — combustion technology, abatement equipment, and
fuel type. The two therefore decouple by design.

**Why the imperfection is the point.** The unexplained variance is not model
failure — it *is* the abatement signal. Cells where actual NOx substantially
exceeds what their CO2 predicts are running dirtier combustion than their energy
use implies.

**Deliverable.** A ranked "investigate these cells" table built from the largest
positive residuals. This converts a middling R² into arguably the most
actionable output of the project.

## Insight 7 — A methodological finding: which model wins depends on which features you supply

**Expected.** With vehicle-kilometre columns included, linear regression will
match or beat the ensemble. With them withheld, the ensemble will retain its
accuracy while linear regression degrades sharply.

**Basis.** VKM is definitionally `AADT × length × 365` and is the input LAEI
uses to compute the target. Supplying it makes the problem linear; withholding
it makes it multiplicative.

**Why we will report it prominently.** This is a clean, demonstrable result
about **feature leakage** — and it carries a governance message the Department
should hear: *a model that scores 0.99 against an inventory may simply be
reproducing the inventory's own arithmetic.* Any future contractor's accuracy
claims should be interrogated on exactly this point.

## Insight 8 — Boundaries of what this data can support

Stated deliberately, because a consultancy that omits these is not doing its job:

- **LAEI is modelled, not measured.** All targets are outputs of the GLA's
  emission-factor model. High accuracy demonstrates that we reproduced *their
  model*, not that we predicted physical reality. Any validation against
  monitoring-station measurements would require a separate dataset.
- **Annual resolution only.** No diurnal, weekly or seasonal structure is
  present, so this work cannot support short-term air-quality forecasting.
- **Emissions, not concentrations.** We deliberately scope to emissions
  (tonnes/year), which boroughs can influence, rather than modelled
  ground-level concentrations (µg/m³), which additionally depend on meteorology
  and dispersion physics. The concentration grids exist in the release but at
  20 m resolution are disproportionate to this phase.
- **No population exposure linkage** unless the separate schools, hospitals and
  population-exceedance workbooks are brought in — a recommended next phase.
- **The 2019 vintage has been superseded** by LAEI 2022 (August 2025). Our
  pipeline is built to be re-pointed at the newer release, but any *multi-year*
  model must account for methodology changes between vintages, or it will learn
  changes in GLA's own method as though they were real emission trends.

---

## Recommended next steps for the modelling stage

1. Fit all three techniques on the prepared tables in `data/processed/`,
   following the protocol in `manifest.json`.
2. Run the link regression under **both** feature sets and report the contrast
   as the headline methodological result.
3. Report **MAE alongside RMSE and R²**. Given the skew, these can rank models
   differently, and that divergence is itself a finding worth explaining.
4. Include **dummy baselines** and **5-fold cross-validation** on every reported
   figure; add a `GroupKFold` by borough as a spatial-correlation sanity check.
5. Frame the hotspot classifier's operating point around **the cost of a missed
   hotspot versus a wasted inspection**, rather than declaring a winner on
   ROC-AUC alone.

---

## Attribution

Contains data from the Greater London Authority, *London Atmospheric Emissions
Inventory (LAEI) 2019*, London Datastore, licensed under the UK Open Government
Licence v3.0. Dataset superseded by LAEI 2022 in August 2025; LAEI 2019 used
here as specified by the project brief.
