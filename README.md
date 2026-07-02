# Heart Disease Prediction with LightGBM

## Project Overview

This project uses the UCI Heart Disease dataset to build a binary classification workflow for predicting whether a patient has heart disease.

The current work has completed exploratory analysis, preprocessing, baseline model training and evaluation, and the first untuned LightGBM model. The next stage is to improve model development and add interpretability with SHAP.

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

## Model Comparison Results

Held-out test set results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False Negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.842 | 0.835 | 0.892 | 0.863 | 0.926 | 11 |
| Random Forest | 0.848 | 0.849 | 0.882 | 0.865 | 0.928 | 12 |
| XGBoost | 0.859 | 0.852 | 0.902 | 0.876 | 0.901 | 10 |
| LightGBM | 0.842 | 0.848 | 0.873 | 0.860 | 0.928 | 13 |

The first untuned LightGBM model has competitive ROC-AUC, but it does not improve the medical-risk-focused metrics. XGBoost currently has the highest recall and the fewest false negatives.

Generated evaluation outputs:

- `reports/results.md`
- `reports/figures/baseline_confusion_matrices.png`
- `reports/figures/baseline_roc_curve.png`
- `reports/figures/lightgbm_confusion_matrix.png`
- `reports/figures/lightgbm_roc_curve.png`
- `reports/figures/lightgbm_feature_importance.png`

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

- Investigate whether threshold tuning can reduce false negatives
- Tune LightGBM hyperparameters after establishing the baseline
- Compare tuned LightGBM against XGBoost and Logistic Regression
- Add SHAP interpretation as a future interpretability step
