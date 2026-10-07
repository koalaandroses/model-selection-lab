# Experiment results (synthetic data)

Seed: 42; training rows: 120; untouched holdout rows: 60.
Enumerated models: 15 (nonempty subsets of four predictors).
Valid bootstrap samples: 1000; skipped rank-deficient samples: 0.
AICc winner on original training sample: x1+x3.
AIC-weighted prediction holdout RMSE: 1.9443.

The simulation was generated with x1 and x3 as true signals. x2 is correlated
with x1; x4 is noise. Inspect the CSV to compare IC rankings and stability.
RMSE is diagnostic here, not used to tune models. Model averaging is weighted
prediction averaging, not averaging raw coefficients across different models.
These results do not establish general performance on real datasets.
