"""Evaluation helpers for binary heart disease classification models."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _positive_class_scores(model, X):
    """Return scores for the positive class for ROC-AUC/ROC plotting."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    """Fit a binary classifier and return standard test-set metrics."""
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = _positive_class_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }


def plot_confusion_matrix(
    models,
    X_test,
    y_test,
    save_path=None,
    class_labels=("No Heart Disease", "Heart Disease"),
):
    """Plot binary confusion matrices for a mapping of fitted models."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = recall_score(y_test, y_pred, zero_division=0)

        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{model_name}\nRecall={recall:.3f}, FN={fn}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels, rotation=30, ha="right")
        ax.set_yticklabels(class_labels)

        labels = [["TN", "FP"], ["FN", "TP"]]
        threshold = cm.max() / 2
        for row in range(2):
            for col in range(2):
                color = "white" if cm[row, col] > threshold else "black"
                ax.text(
                    col,
                    row,
                    f"{labels[row][col]}\n{cm[row, col]}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_roc_curve(models, X_test, y_test, save_path=None, title="ROC Curves"):
    """Plot ROC curves for a mapping of fitted binary classifiers."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, model in models.items():
        y_score = _positive_class_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def _dataframe_to_markdown(df):
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    columns = list(df.columns)

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    rows = [[format_value(value) for value in row] for row in df.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_results_markdown(
    results_df,
    output_path,
    title="Baseline Model Results",
    model_group_name="baseline models",
    best_model_label="baseline",
):
    """Write model metrics and medical-risk notes to a Markdown report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_df = results_df.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    display_df[metric_columns] = display_df[metric_columns].round(3)
    best_recall = display_df.sort_values(
        ["Recall", "False Negatives", "ROC-AUC"],
        ascending=[False, True, False],
    ).iloc[0]

    content = [
        f"# {title}",
        "",
        "In medical risk prediction, false negatives are particularly important because they represent patients with heart disease who are incorrectly classified as low-risk.",
        "",
        f"The table below compares the {model_group_name} on the held-out test set.",
        "",
        _dataframe_to_markdown(display_df),
        "",
        "## Medical-Risk Focus",
        "",
        f"The highest-recall {best_model_label} is **{best_recall['Model']}** with recall **{best_recall['Recall']:.3f}** and **{int(best_recall['False Negatives'])}** false negatives.",
        "",
        "Recall is emphasized because it measures how many true heart-disease cases are detected. False negatives are emphasized because they are the missed disease cases.",
        "",
    ]

    output_path.write_text("\n".join(content), encoding="utf-8")
