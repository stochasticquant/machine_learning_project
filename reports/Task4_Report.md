# Model Assessment and Differences in Performance

## LAEI 2019 Atmospheric Emissions Modelling

**Prepared for:** Department of Environment  
**Prepared by:** Athana Data Science Services — Module End Project group  
**Covers:** Task 4 — assessment of model performance  
**Dataset:** London Atmospheric Emissions Inventory (LAEI) 2019  
**Supporting analysis:** Stage 3 modelling outputs and Stage 4 assessment notebook

## 1. Assessment Scope and Evaluation Framework

The models created in tasks 1-2 and implemented in 3 are taken here in task 4 for evaluation.

The three principal techniques are **Multiple Linear Regression**, **Random Forest**, and **K-Means clustering**. 

Ridge Regression, Histogram Gradient Boosting and Logistic Regression are kept as they can help explain some model behaviour.

Regression models were assessed using an 80/20 hold-out split with `random_state=42`, together with shuffled five-fold cross-validation. `GroupKFold` by borough was also used for checking spatial parameters. 

The main regression metrics are **R²**, **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

Classification models were evaluated using a stratified hold-out split and assessed using **precision, recall, F1 and ROC-AUC**. 

K-Means was assessed using internal clustering metrics. I tested it to see whether the resulting clusters showed spatial coherence.

A further limitation is fundamental to the dataset: LAEI 2019 is a **modelled emissions inventory**, not a set of direct roadside measurements. 

---

## 2. Regression Performance and Model Comparison

The main supervised task predicts annual NOx emissions for road links. Held-out results shown below -

| Model | R² | MAE (t/yr) | RMSE (t/yr) |
|---|---:|---:|---:|
| Dummy Regressor | -0.026 | 0.221 | 1.101 |
| Linear Regression | 0.750 | 0.206 | 0.543 |
| Ridge Regression | 0.750 | 0.206 | 0.543 |
| Histogram Gradient Boosting | 0.911 | 0.045 | 0.324 |
| Random Forest | **0.987** | **0.019** | **0.123** |

Shows that Random Forest is the strongest but some margin, and indicates that its R² of 0.987 reproduces most of the variance.

In contrast MAE and RMSE are a lot lower. The main reason for this difference is that total emissions depend stringly on interactions between **traffic volume and road length**.

Linear regression can only add the features if an interaction term is specifically created. In contrast tree-based models can approximate this through splits.

This explains why Linear Regression is a lot lower than Random Forest on the realistic dataset.

Also, five-fold cross-validation shows that the results are not because of a favourable split.

| Model | Mean CV R² | Standard deviation |
|---|---:|---:|
| Linear / Ridge | 0.756 | 0.012 |
| Histogram Gradient Boosting | 0.903 | 0.016 |
| Random Forest | **0.987** | **0.001** |

---

## 3. Generalisation, Leakage and Error Behaviour

### Spatial generalisation

To make sure that links from the same borough were not split between folds, Random Forest was also tested using borough-grouped cross-validation.

This resulted in a mean R² of **0.973** as opposed to the **0.987** under ordinary shuffled validation.

Albeit a small difference it does show that random splitting slightly overstates generalisation because nearby or similar roads can appear on both sides of a random split.

### Effect of target-derived features

A leakage experiment was run to demonstrate why model inputs must be interpreted carefully.

| Model | Realistic R² | Full + VKM R² | Change |
|---|---:|---:|---:|
| Linear / Ridge | 0.750 | 0.991 | +0.241 |
| Histogram Gradient Boosting | 0.911 | 0.950 | +0.039 |
| Random Forest | 0.987 | 0.989 | +0.002 |

This shows that VKM artificially boosts linear-model performance by exposing the inventory’s underlying calculation, so only the realistic non-VKM results should be treated as genuine predictive performance.

### Performance on the busiest links

Overall metrics can hide poor performance in the high-emission tail. The top 10% of road links were therefore assessed separately.

| Model | MAE all | MAE top 10% | RMSE all | RMSE top 10% |
|---|---:|---:|---:|---:|
| Linear Regression | 0.206 | 0.632 | 0.543 | 1.480 |
| Random Forest | 0.019 | 0.127 | 0.123 | 0.388 |

This showed that both models deterioated on the busiest links.

## 4. Effect of Target Definition: Total NOx vs Intensity

As this task was focusing on which roads had the most emissions a second test was added to determine which roads are most emission-intensive per metre.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear / Ridge | 0.912 | 0.0003 | 0.0006 |
| Histogram Gradient Boosting | 0.969 | 0.0002 | 0.0004 |
| Random Forest | **0.990** | **0.0001** | **0.0002** |

This showed that Linear Regression improves markedly from approximately R² 0.75 on total emissions to approximately 0.91 on intensity.

Random forest stayed the strongest but the differences between linear and non-linear grew much smaller.

---

## 5. Hotspot Classification Assessment

At the default probability threshold of 0.5:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.599 | **0.953** | 0.735 | 0.982 |
| Random Forest | **0.966** | 0.651 | **0.778** | **0.989** |

Cells in the top 10% of NOx emissions were considered/treated as hotspots.

Logistic Regression prioritises recall and finds nearly all hotspots, howeevr thiss did produce more false alarms. 

Random Forest achieved high precision, however at default threshold did moss some hotspots.

A cost based threshold analysis was therefore sued, this meant that a cost of 5 was assigned to a missed hotspot and 1 to a false alarm.

| Model | Selected threshold | Missed hotspots | False alarms | Total cost |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.65 | 5 | 39 | 64 |
| Random Forest | 0.25 | 5 | 38 | **63** |

Once the threshold is matched to the same decision cost, the models were very similar.

---

## 6. K-Means Cluster Assessment

A four-cluster soloution was used for interpretability -

| Cluster | Cells | Dominant profile |
|---:|---:|---|
| 0 | 69 | Aviation-dominated |
| 1 | 1,253 | Heat-and-power-dominated |
| 2 | 2,071 | Road-transport-dominated |
| 3 | 67 | River-dominated |

It was determined that the groups could not be determined as four seperate classes because at k = 4, silhouette is approximately **0.272**.

There were some findings that were useful however, cells did share labels more that expected under shuffled labels - spatial-coherence z-score of **22.7**.

It was determined that K-Means is actually a useful **policy typology**, but not that London contains 4 distinct emission classes.

---

## 7. Overall Assessment and Recommendation

All models perform different jobs and do not need to be ranked singularly.

- **Random Forest** is the preferred predictor.
- **Linear Regression** remains useful as a simpler explanatory benchmark.
- **Histogram Gradient Boosting** provides a strong and computationally cheaper benchmark.
- **Hotspot classification** is highly effective for both Logistic Regression and Random Forest.
- **K-Means** Its clusters are useful for policy targeting.
- **VKM-Target-derived results should not be considered as genuine predictive performance.**

---

### Limitations and Validity

It should be noted that this assesment was restricted to LAEI 2019 and thus repressents only one years inventory.

It cannot support seasonal air quality forecasting.

Where results agreed with LAEI, these should not be considered set and validated because LAEI is a modelled inventory itself.

---

## References

Greater London Authority, *London Atmospheric Emissions Inventory (LAEI) 2019*, London Datastore. Data used under the UK Open Government Licence v3.0.

Scikit-learn documentation, model implementations and evaluation metrics used throughout the project.