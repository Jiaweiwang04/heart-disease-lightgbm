# SHAP Interpretation Summary

This report explains the tuned LightGBM model using SHAP values on the held-out test set.

Positive SHAP values push predictions toward higher heart disease risk, while negative SHAP values push predictions toward lower risk.

## Top SHAP Features

| Feature | MeanAbsSHAP |
| --- | ---: |
| cat__cp_asymptomatic | 0.573 |
| num__oldpeak | 0.460 |
| cat__exang_False | 0.309 |
| cat__dataset_Switzerland | 0.298 |
| num__chol | 0.270 |
| cat__ca_0.0 | 0.268 |
| cat__sex_Female | 0.266 |
| num__age | 0.258 |
| cat__cp_atypical angina | 0.246 |
| cat__dataset_Cleveland | 0.206 |

## Interpretation Notes

The strongest SHAP signals include chest pain category, ST depression (`oldpeak`), exercise-induced angina, cholesterol, ca, sex, age, and dataset indicators.

These features are clinically plausible for a heart disease risk model, but SHAP values explain model behavior rather than proving causal effects.
