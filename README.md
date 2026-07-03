# Heart Disease Prediction with LightGBM

## Project Overview

This project uses the UCI Heart Disease dataset to build a binary classification workflow for predicting whether a patient has heart disease.

The current work has completed exploratory analysis, preprocessing, baseline model training and evaluation, the first untuned LightGBM model, LightGBM threshold tuning, and LightGBM hyperparameter tuning. The next stage is to add interpretability with SHAP.

In medical risk prediction, false negatives are particularly important because they represent patients with heart disease who are incorrectly classified as low-risk.

## Current Progress

- [x] Data loading
- [x] Exploratory Data Analysis
- [x] Data preprocessing
- [x] Baseline model training
- [x] Baseline model evaluation
- [x] LightGBM model training
- [x] LightGBM evaluation
- [x] LightGBM feature importance
- [x] LightGBM threshold tuning
- [x] LightGBM hyperparameter tuning
- [ ] SHAP analysis

## Dataset

The original label column is `num`, which indicates heart disease severity.

For binary classification, the target is defined as:

```python
target = 1 if num > 0 else 0
```

The preprocessing step drops:

- `id`, because it is only an identifier
- `num`, because it is the original disease severity label and would cause label leakage

## Completed Workflow

### 1. Exploratory Data Analysis

Notebook:

- `notebooks/01_eda.ipynb`

This notebook explores the raw UCI Heart Disease dataset and checks the structure, missing values, target distribution, and feature patterns.

### 2. Preprocessing

Notebook:

- `notebooks/02_preprocessing.ipynb`

Outputs:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

Preprocessing includes:

- binary target creation
- leakage-safe feature selection
- stratified train/test split
- median imputation for numerical columns
- most-frequent imputation for categorical columns
- one-hot encoding for categorical columns

### 3. Baseline Models

Notebook:

- `notebooks/03_baseline_models.ipynb`

Evaluation utilities:

- `src/evaluate.py`

Baseline models trained:

- Logistic Regression
- Random Forest
- XGBoost

### 4. First LightGBM Model

Notebook:

- `notebooks/04_lightgbm_baseline.ipynb`

The LightGBM notebook reuses the existing processed train/test split. It does not redo preprocessing or create a new train/test split, so the comparison remains aligned with the baseline models.

Initial LightGBM settings:

- `objective="binary"`
- `n_estimators=100`
- `learning_rate=0.05`
- `num_leaves=31`
- `random_state=42`

No hyperparameter tuning has been performed yet.

### 5. LightGBM Threshold Tuning

Notebook:

- `notebooks/05_lightgbm_threshold_tuning.ipynb`

The threshold tuning notebook keeps the same LightGBM hyperparameters and changes only the probability cutoff used to classify patients as positive or negative.

To avoid selecting a threshold directly on the test set, the notebook creates an internal validation split from the existing training data. The original test set is used only for final evaluation.

Selected threshold:

- `0.11`

This threshold greatly reduces false negatives, but it also increases false positives. That tradeoff is expected when recall is prioritized for a screening-style medical task.

### 6. LightGBM Hyperparameter Tuning

Notebook:

- `notebooks/06_lightgbm_hyperparameter_tuning.ipynb`

The hyperparameter tuning notebook uses a small randomized search with cross-validated ROC-AUC on an internal training split. The original held-out test set remains untouched until final evaluation.

Best tuned parameters:

- `n_estimators=100`
- `learning_rate=0.08`
- `num_leaves=7`
- `max_depth=3`
- `min_child_samples=50`
- `subsample=1.0`
- `colsample_bytree=0.9`
- `reg_alpha=0.1`
- `reg_lambda=0.0`

The tuned model is evaluated with both the default threshold `0.50` and a validation-selected threshold `0.20`.

## Model Comparison Results

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

Hyperparameter tuning improves the default-threshold LightGBM result, reducing false negatives from `13` to `9` and improving recall from `0.873` to `0.912`.

Combining tuned hyperparameters with a tuned threshold keeps false negatives at `1`, while reducing false positives from `42` to `35` compared with the earlier threshold-only LightGBM approach.

Generated evaluation outputs:

- `reports/results.md`
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

## Repository Structure

- `data/raw/` - original dataset
- `data/processed/` - processed train/test datasets
- `notebooks/` - EDA, preprocessing, and modeling notebooks
- `src/` - reusable project code
- `reports/` - generated evaluation reports and figures
- `README.md` - project overview and progress

## Environment

The notebooks have been validated with the conda environment:

```powershell
conda activate ml
```

The current workflow depends on:

- pandas
- scikit-learn
- xgboost
- lightgbm
- matplotlib

## Next Steps

- Add SHAP interpretation as a future interpretability step
- Review whether the tuned threshold tradeoff is clinically acceptable
- Consider probability calibration before presenting risk probabilities
