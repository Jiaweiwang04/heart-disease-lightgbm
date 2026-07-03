# Model Comparison Results

In medical risk prediction, false negatives are particularly important because they represent patients with heart disease who are incorrectly classified as low-risk.

The table below compares the baseline models, LightGBM threshold tuning, and LightGBM hyperparameter tuning on the held-out test set.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | True Negatives | False Positives | False Negatives | True Positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.500 | 0.842 | 0.835 | 0.892 | 0.863 | 0.926 | 64 | 18 | 11 | 91 |
| Random Forest | 0.500 | 0.848 | 0.849 | 0.882 | 0.865 | 0.928 | 66 | 16 | 12 | 90 |
| XGBoost | 0.500 | 0.859 | 0.852 | 0.902 | 0.876 | 0.901 | 66 | 16 | 10 | 92 |
| LightGBM | 0.500 | 0.842 | 0.848 | 0.873 | 0.860 | 0.928 | 66 | 16 | 13 | 89 |
| LightGBM tuned threshold | 0.110 | 0.766 | 0.706 | 0.990 | 0.824 | 0.928 | 40 | 42 | 1 | 101 |
| LightGBM tuned params | 0.500 | 0.875 | 0.869 | 0.912 | 0.890 | 0.927 | 68 | 14 | 9 | 93 |
| LightGBM tuned params + threshold | 0.200 | 0.804 | 0.743 | 0.990 | 0.849 | 0.927 | 47 | 35 | 1 | 101 |

## Medical-Risk Focus

The highest-recall model is **LightGBM tuned params + threshold** with recall **0.990** and **1** false negatives.

Recall is emphasized because it measures how many true heart-disease cases are detected. False negatives are emphasized because they are the missed disease cases.
