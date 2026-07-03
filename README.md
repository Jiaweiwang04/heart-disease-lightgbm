# Heart Disease Prediction with LightGBM

## Project Status

This project is complete.

It builds an end-to-end binary classification workflow for the UCI Heart Disease dataset, from exploratory analysis and preprocessing through baseline modeling, LightGBM training, threshold tuning, hyperparameter tuning, and SHAP interpretation.

The project is a machine learning study, not a clinical decision system. In medical risk prediction, false negatives are especially important because they represent patients with heart disease who are incorrectly classified as low-risk.

## Objective

Predict whether a patient has heart disease using the UCI Heart Disease dataset.

The original label column is `num`, which indicates disease severity. It is converted into a binary target:

```python
target = 1 if num > 0 else 0
```

The preprocessing step drops:

- `id`, because it is only an identifier
- `num`, because it is the original disease severity label and would cause label leakage

## Final Recommendation

Two LightGBM configurations are worth keeping, depending on the goal:

| Use Case | Model | Threshold | Recall | False Negatives | False Positives |
| --- | --- | ---: | ---: | ---: | ---: |
| Balanced performance | LightGBM tuned params | 0.50 | 0.912 | 9 | 14 |
| Screening-focused | LightGBM tuned params + threshold | 0.20 | 0.990 | 1 | 35 |

For a heart disease screening-style task, the threshold-tuned model is the safer option because it misses only `1` positive case on the held-out test set. The tradeoff is more false positives, meaning more patients would be flagged for follow-up.

## Model Results

Held-out test set results:

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | False Negatives | False Positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.50 | 0.842 | 0.835 | 0.892 | 0.863 | 0.926 | 11 | 18 |
| Random Forest | 0.50 | 0.848 | 0.849 | 0.882 | 0.865 | 0.928 | 12 | 16 |
| XGBoost | 0.50 | 0.859 | 0.852 | 0.902 | 0.876 | 0.901 | 10 | 16 |
| LightGBM | 0.50 | 0.842 | 0.848 | 0.873 | 0.860 | 0.928 | 13 | 16 |
| LightGBM tuned threshold | 0.11 | 0.766 | 0.706 | 0.990 | 0.824 | 0.928 | 1 | 42 |
| LightGBM tuned params | 0.50 | 0.875 | 0.869 | 0.912 | 0.890 | 0.927 | 9 | 14 |
| LightGBM tuned params + threshold | 0.20 | 0.804 | 0.743 | 0.990 | 0.849 | 0.927 | 1 | 35 |

Hyperparameter tuning improved the default-threshold LightGBM model, reducing false negatives from `13` to `9`.

Combining tuned hyperparameters with a tuned threshold kept false negatives at `1` while reducing false positives from `42` to `35` compared with the earlier threshold-only LightGBM approach.

## Completed Workflow

### 1. Exploratory Data Analysis

Notebook:

- `notebooks/01_eda.ipynb`

Explores the raw dataset structure, missing values, target distribution, and feature patterns.

### 2. Preprocessing

Notebook:

- `notebooks/02_preprocessing.ipynb`

Outputs:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

Preprocessing includes binary target creation, leakage-safe feature selection, stratified train/test splitting, median imputation for numerical columns, most-frequent imputation for categorical columns, and one-hot encoding.

### 3. Baseline Models

Notebook:

- `notebooks/03_baseline_models.ipynb`

Models:

- Logistic Regression
- Random Forest
- XGBoost

### 4. First LightGBM Model

Notebook:

- `notebooks/04_lightgbm_baseline.ipynb`

Initial LightGBM parameters:

- `objective="binary"`
- `n_estimators=100`
- `learning_rate=0.05`
- `num_leaves=31`
- `random_state=42`

### 5. LightGBM Threshold Tuning

Notebook:

- `notebooks/05_lightgbm_threshold_tuning.ipynb`

This step keeps the initial LightGBM parameters fixed and changes only the probability cutoff. The selected threshold is chosen using an internal validation split from the training data.

Selected threshold:

- `0.11`

### 6. LightGBM Hyperparameter Tuning

Notebook:

- `notebooks/06_lightgbm_hyperparameter_tuning.ipynb`

The tuning notebook uses a small randomized search with cross-validated ROC-AUC on an internal training split. The held-out test set is used only for final evaluation.

Best tuned parameters:

```python
{
    "n_estimators": 100,
    "learning_rate": 0.08,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 50,
    "subsample": 1.0,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.0,
}
```

Selected threshold after tuning:

- `0.20`

### 7. SHAP Interpretation

Notebook:

- `notebooks/07_shap_interpretation.ipynb`

Report:

- `reports/shap_summary.md`

Top SHAP features:

| Feature | MeanAbsSHAP |
| --- | ---: |
| `cat__cp_asymptomatic` | 0.573 |
| `num__oldpeak` | 0.460 |
| `cat__exang_False` | 0.309 |
| `cat__dataset_Switzerland` | 0.298 |
| `num__chol` | 0.270 |
| `cat__ca_0.0` | 0.268 |
| `cat__sex_Female` | 0.266 |
| `num__age` | 0.258 |
| `cat__cp_atypical angina` | 0.246 |
| `cat__dataset_Cleveland` | 0.206 |

The strongest SHAP signals include chest pain category, ST depression (`oldpeak`), exercise-induced angina, cholesterol, ca, sex, age, and dataset indicators. These are model explanations, not causal claims.

## Project Files

### Notebooks

- `notebooks/01_eda.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `notebooks/03_baseline_models.ipynb`
- `notebooks/04_lightgbm_baseline.ipynb`
- `notebooks/05_lightgbm_threshold_tuning.ipynb`
- `notebooks/06_lightgbm_hyperparameter_tuning.ipynb`
- `notebooks/07_shap_interpretation.ipynb`

### Source Code

- `src/evaluate.py` - reusable evaluation, plotting, and report helpers

### Reports and Figures

- `reports/results.md`
- `reports/shap_summary.md`
- `reports/figures/baseline_confusion_matrices.png`
- `reports/figures/baseline_roc_curve.png`
- `reports/figures/lightgbm_confusion_matrix.png`
- `reports/figures/lightgbm_roc_curve.png`
- `reports/figures/lightgbm_feature_importance.png`
- `reports/figures/lightgbm_threshold_tradeoff.png`
- `reports/figures/lightgbm_tuned_confusion_matrix.png`
- `reports/figures/lightgbm_hyperparameter_search.png`
- `reports/figures/lightgbm_tuned_params_confusion_matrices.png`
- `reports/figures/lightgbm_tuned_params_roc_curve.png`
- `reports/figures/lightgbm_tuned_params_feature_importance.png`
- `reports/figures/lightgbm_tuned_params_threshold_tradeoff.png`
- `reports/figures/shap_feature_importance_bar.png`
- `reports/figures/shap_summary_beeswarm.png`
- `reports/figures/shap_summary_bar.png`
- `reports/figures/shap_high_risk_waterfall.png`

## Repository Structure

- `data/raw/` - original dataset
- `data/processed/` - processed train/test datasets
- `notebooks/` - EDA, preprocessing, modeling, tuning, and interpretation notebooks
- `src/` - reusable project code
- `reports/` - generated reports and figures
- `README.md` - final project overview

## Environment

The notebooks were validated with the conda environment:

```powershell
conda activate ml
```

Main dependencies:

- pandas
- scikit-learn
- xgboost
- lightgbm
- shap
- matplotlib

## Reproducible Run Order

Run the notebooks in this order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_baseline_models.ipynb`
4. `notebooks/04_lightgbm_baseline.ipynb`
5. `notebooks/05_lightgbm_threshold_tuning.ipynb`
6. `notebooks/06_lightgbm_hyperparameter_tuning.ipynb`
7. `notebooks/07_shap_interpretation.ipynb`

## Limitations

- The dataset is small, so results may vary with different splits or external validation data.
- The tuned threshold is optimized for recall and may create many false positives.
- SHAP explains model behavior, not causal medical relationships.
- A real clinical workflow would require calibration, external validation, and domain review before use.

## Future Work

- Validate on an external dataset
- Calibrate predicted probabilities
- Review the operating threshold with clinical stakeholders
- Package the final workflow into a reproducible pipeline or app
