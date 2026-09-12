# Modelling Results and Statistical Summary

## LAEI 2019 Atmospheric Emissions Modelling

**Prepared for:** Department of Environment  
**Prepared by:** Athana Data Science Services — Module End Project group  
**Covers:** Tasks 3 (model implementation)  
**Dataset:** London Atmospheric Emissions Inventory (LAEI) 2019, London Datastore (`e758q`)  
**Supporting analysis:** `notebooks/modelling.ipynb` and `manifest.json`

---

## Executive summary

The three techniques were fitted to the prepared tables in `data/processed/` using the stated protocol: fixed random state 42, preprocessing inside pipelines, stratified classification splitting, shuffled five-fold cross-validation, and borough-grouped cross-validation as a spatial sanity check.

The headline methodological result is the contrast between the two link-level feature sets. With VKM included, linear regression achieves R2 = 0.9907; without VKM, it falls to R2 = 0.7503. Random forest remains strong, falling only from R2 = 0.9888 to 0.9871. VKM therefore exposes the leakage trap: it makes the inventory calculation almost directly available to the model.

The regression target is strongly right-skewed, so R2, MAE, and RMSE describe different aspects of performance. RMSE is more affected by large motorway-link errors, whereas MAE describes the typical link. Classification results likewise depend on the operating threshold and the relative cost of missed hotspots and wasted inspections, not ROC-AUC alone.

An additional intensity analysis replaces the total-emissions target with emissions per metre of road. This better proxies roadside exposure. The same models and protocol are reused. Random forest again dominates, while linear models improve markedly relative to the total-emissions case, consistent with the removal of the multiplicative length effect.

---

## Regression results (total NOx)

The first table reports the held-out 20% split. The second reports five-fold cross-validation as mean ± standard deviation of R2. The baseline is a median DummyRegressor and is identical for both feature sets because it does not use predictors.

### Hold-out metrics (total NOx)

| Feature set | Model | R2 | MAE | RMSE |
|---|---|---:|---:|---:|
| Full, with VKM | DummyRegressor | -0.026 | 0.221 | 1.101 |
| Full, with VKM | LinearRegression | 0.9907 | 0.0334 | 0.1048 |
| Full, with VKM | Ridge | 0.9907 | 0.0332 | 0.1047 |
| Full, with VKM | RandomForest | 0.9888 | 0.0150 | 0.1152 |
| Full, with VKM | HistGradientBoosting | 0.9502 | 0.0244 | 0.2426 |
| Realistic, without VKM | DummyRegressor | -0.026 | 0.221 | 1.101 |
| Realistic, without VKM | LinearRegression | 0.7503 | 0.2060 | 0.5432 |
| Realistic, without VKM | Ridge | 0.7503 | 0.2060 | 0.5432 |
| Realistic, without VKM | RandomForest | 0.9871 | 0.0191 | 0.1233 |
| Realistic, without VKM | HistGradientBoosting | 0.9114 | 0.0445 | 0.3236 |

The negative baseline R2 is expected: predicting the median is worse than predicting the mean for this skewed target. With VKM, linear regression and Ridge reproduce the near-linear inventory arithmetic. Without VKM, linear models cannot express traffic × length, while the forest captures the interaction through tree splits. HistGradientBoosting is weaker here, consistent with loss of resolution from continuous-feature binning.

### Five-fold cross-validation (total NOx)

| Feature set | Model | Mean R2 | Standard deviation |
|---|---|---:|---:|
| Full, with VKM | LinearRegression | 0.9867 | 0.0026 |
| Full, with VKM | Ridge | 0.9867 | 0.0026 |
| Full, with VKM | RandomForest | 0.9904 | 0.0017 |
| Full, with VKM | HistGradientBoosting | 0.9626 | 0.0113 |
| Realistic, without VKM | LinearRegression | 0.7561 | 0.0124 |
| Realistic, without VKM | Ridge | 0.7561 | 0.0124 |
| Realistic, without VKM | RandomForest | 0.9870 | 0.0013 |
| Realistic, without VKM | HistGradientBoosting | 0.9030 | 0.0162 |

The realistic random forest's ordinary cross-validation R2 is 0.9870 ± 0.0013. Borough-grouped cross-validation gives R2 = 0.9727, quantifying the optimism caused by spatially related links appearing in both randomly assigned training and test folds.

### Feature importance (total NOx)

| Importance method | Feature | Importance statistic |
|---|---|---:|
| Impurity | Link Length (m) | 0.686884 |
| Impurity | AADT 2019 - Total | 0.129500 |
| Impurity | AADT Diesel LGV | 0.039415 |
| Impurity | AADT Petrol Car | 0.033166 |
| Impurity | AADT Diesel Car | 0.020921 |
| Impurity | AADT Diesel PHV | 0.012562 |
| Impurity | AADT 2019 - HGVs - Articulated - 3 to 4 Axles | 0.010470 |
| Impurity | AADT Petrol PHV | 0.009643 |
| Impurity | AADT Electric Car | 0.008139 |
| Impurity | AADT Petrol LGV | 0.006586 |
| Permutation | Link Length (m) | 1.015516 |
| Permutation | AADT 2019 - Total | 0.103179 |
| Permutation | AADT Petrol Car | 0.013130 |
| Permutation | AADT Diesel LGV | 0.011984 |
| Permutation | AADT Diesel Car | 0.006403 |
| Permutation | Speed (km/hr) - Except Buses | 0.005983 |
| Permutation | AADT 2019 - HGVs - Articulated - 5 Axles | 0.005885 |
| Permutation | AADT Petrol PHV | 0.004283 |
| Permutation | AADT Diesel PHV | 0.003732 |
| Permutation | AADT 2019 - HGVs - Articulated - 3 to 4 Axles | 0.003259 |

Impurity importance sums to one, whereas permutation importance is the mean reduction in test-set R2 after shuffling a feature. The two methods agree that link length and total traffic dominate, but their lower rankings differ because correlated predictors share information and impurity importance favours high-cardinality continuous variables.

---

## Intensity regression results (NOx per metre)

This section repeats the link-level regression with the alternative target `nox_per_m`, which divides total NOx by link length. This changes the question from "which links emit the most in total?" to "which links are most emission-intensive per metre?", which is more relevant to roadside exposure.

### Hold-out metrics (NOx per metre)

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| LinearRegression | 0.9124 | 0.0003 | 0.0006 |
| Ridge | 0.9125 | 0.0003 | 0.0006 |
| RandomForest | 0.9899 | 0.0001 | 0.0002 |
| HistGradientBoosting | 0.9692 | 0.0002 | 0.0004 |

Compared with the realistic total-NOx results, linear models improve from R2 ≈ 0.75 to R2 ≈ 0.91. This is expected: dividing by length removes much of the multiplicative structure that linear models could not represent, making the relationship closer to linear in the remaining predictors. Random forest remains very strong, with R2 ≈ 0.99 in both cases.

### Five-fold cross-validation (NOx per metre)

| Model | Mean R2 | Standard deviation |
|---|---:|---:|
| LinearRegression | 0.9096 | 0.0023 |
| Ridge | 0.9096 | 0.0023 |
| RandomForest | 0.9894 | 0.0009 |
| HistGradientBoosting | 0.9698 | 0.0022 |

The cross-validated results mirror the hold-out pattern: linear models achieve R2 around 0.91, while the forest remains near 0.99. This reinforces the interpretation that intensity is a more linearly predictable quantity once the length scaling is removed.

---

## Hotspot classification

There are 346 hotspot cells out of 3,460, a positive rate of 10%. The dummy classifier illustrates why accuracy alone is inadequate.

### Default threshold of 0.5

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 0.901 | 0.000 | 0.000 | 0.000 | — |
| LogisticRegression | 0.932 | 0.599 | 0.953 | 0.735 | 0.982 |
| RandomForest | 0.963 | 0.966 | 0.651 | 0.778 | 0.989 |

Logistic regression finds 95.3% of hotspots but creates more false alarms. The forest has higher precision at the default threshold but lower recall. ROC-AUC measures ranking quality, not the final inspection decision.

### Cost-based operating point

The decision rule assigns cost 5 to each missed hotspot and cost 1 to each false alarm.

| Model | Selected threshold | Missed hotspots (FN) | False alarms (FP) | Total cost |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.65 | 5 | 39 | 64 |
| RandomForest | 0.25 | 5 | 38 | 63 |

The optimal thresholds differ because the models produce differently calibrated probability scores. Under this cost assumption, the forest is marginally cheaper, but the models are practically close; the preferred model should also reflect inspection capacity, the consequences of missed hotspots, interpretability, and probability calibration.

---

## Clustering results

K-Means was fitted to the 16 standardised source-share variables, excluding Easting and Northing from the features. The silhouette scan is:

| k | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 33,808 | 0.250 |
| 3 | 30,482 | 0.272 |
| 4 | 27,331 | 0.272 |
| 5 | 24,545 | 0.271 |
| 6 | 21,343 | 0.284 |
| 7 | 19,395 | 0.255 |
| 8 | 16,416 | 0.278 |
| 9 | 13,940 | 0.299 |
| 10 | 11,973 | 0.321 |

Silhouette values are low and do not identify a uniquely separated solution. Four clusters were retained for interpretability. Their sizes are 69, 1,253, 2,071, and 67 cells.

| Cluster | Dominant profile | Key mean shares |
|---:|---|---|
| 0 | Aviation-dominated | Aviation 63.9%; Road Transport 18.0%; Heat and Power 13.4% |
| 1 | Heat and power-dominated | Heat and Power 56.6%; Road Transport 26.9%; Agriculture 6.9% |
| 2 | Road-transport-dominated | Road Transport 68.0%; Heat and Power 25.0%; Construction 3.0% |
| 3 | River-dominated | River 71.5%; Road Transport 8.8%; Heat and Power 12.7% |

Spatial validation gives neighbour agreement of 0.588, compared with 0.508 after shuffling labels, with z = 22.7. This indicates strong spatial coherence despite coordinates being withheld from clustering. Agglomerative clustering agrees only partially, with ARI = 0.385, supporting the interpretation of a continuum of source mixes rather than four natural, sharply separated kinds.

---

## Interpretation and limitations

The central methodological result is the full-versus-realistic contrast. A high score with VKM is not evidence of real-world prediction: VKM is closely related to the arithmetic used to construct the target. The realistic feature set is therefore the appropriate headline evaluation, while the full set should be reported only as a labelled leakage demonstration.

MAE, RMSE, and R2 must be read together. MAE = 0.019 and RMSE = 0.123 for the realistic forest show that the typical error is small, while the larger RMSE reflects a minority of large errors. This skew means that a model can rank differently depending on whether the priority is average performance or protection against extreme errors.

The dummy baselines, five-fold validation, and borough-grouped validation prevent an apparently high score from being interpreted without context. The classification threshold analysis similarly connects statistical output to the actual decision: whether to inspect a cell, and how costly a missed hotspot is relative to a wasted visit.

The intensity analysis shows that, when the target is emissions per metre rather than total emissions, linear models perform much better (R2 ≈ 0.91 vs ≈ 0.75), while random forest remains near ceiling (R2 ≈ 0.99). This supports the interpretation that total emissions are dominated by the multiplicative effect of link length, whereas intensity is more directly related to traffic and other predictors in an approximately linear way. For questions about roadside exposure, intensity is therefore the more appropriate target.

---

## Protocol compliance

- Three techniques were fitted on the prepared tables in `data/processed/`.
- Link regression was run with both full and realistic feature sets.
- R2, MAE, and RMSE were reported for the hold-out regression results.
- Dummy regression and classification baselines were included.
- Five-fold shuffled cross-validation used `random_state=42`.
- GroupKFold by `Borough` was used as a spatial-correlation sanity check.
- Classification used a stratified split and reported precision, recall, F1, and ROC-AUC.
- Hotspot thresholds were selected using explicit miss and false-alarm costs.
- K-Means used standardised source-share inputs and excluded coordinates from the clustering features.
- An intensity-target regression was run using the same protocol and models, and its results were reported alongside the total-emissions results.

---

## Attribution

Contains data from the Greater London Authority, *London Atmospheric Emissions Inventory (LAEI) 2019*, London Datastore, licensed under the UK Open Government Licence v3.0. LAEI 2019 was used as specified by the project brief.