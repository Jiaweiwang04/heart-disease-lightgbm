# Baseline Model Results

In medical risk prediction, false negatives are particularly important because they represent patients with heart disease who are incorrectly classified as low-risk.

The table below compares the baseline models on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | True Negatives | False Positives | False Negatives | True Positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.842 | 0.835 | 0.892 | 0.863 | 0.926 | 64 | 18 | 11 | 91 |
| Random Forest | 0.848 | 0.849 | 0.882 | 0.865 | 0.928 | 66 | 16 | 12 | 90 |
| XGBoost | 0.859 | 0.852 | 0.902 | 0.876 | 0.901 | 66 | 16 | 10 | 92 |

## Medical-Risk Focus

The highest-recall baseline is **XGBoost** with recall **0.902** and **10** false negatives.

Recall is emphasized because it measures how many true heart-disease cases are detected. False negatives are emphasized because they are the missed disease cases.
