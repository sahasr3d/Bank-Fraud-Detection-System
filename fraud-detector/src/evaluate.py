"""Model evaluation and visualization utilities.

Fraud detection evaluation should focus on finding fraudulent transactions
without creating too many false alarms. Accuracy alone is misleading because a
model can score high by predicting every transaction as legitimate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


FRAUD_LABEL = 1


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a classifier using fraud-focused metrics."""

    probabilities = _predict_fraud_probabilities(model, x_test)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
        if y_test.nunique() > 1
        else 0.0,
        "threshold": float(threshold),
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["legitimate", "fraud"],
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def save_evaluation_report(
    model_name: str,
    evaluation: dict[str, Any],
    reports_dir: Path,
) -> Path:
    """Save model metrics and detailed classification report as JSON."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{model_name}_evaluation.json"

    serializable = {
        "metrics": evaluation["metrics"],
        "classification_report": evaluation["classification_report"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)

    return report_path


def plot_confusion_matrix(
    y_true: pd.Series,
    probabilities: list[float],
    model_name: str,
    output_dir: Path,
    threshold: float = 0.5,
) -> Path:
    """Save a labeled confusion matrix chart."""

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (pd.Series(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted legitimate", "Predicted fraud"],
        yticklabels=["Actual legitimate", "Actual fraud"],
        ax=axis,
    )
    axis.set_title(f"{model_name} Confusion Matrix")
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Actual")
    path = output_dir / f"{model_name}_confusion_matrix.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_roc_curve(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save the ROC curve for a model."""

    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = _predict_fraud_probabilities(model, x_test)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    roc_score = auc(false_positive_rate, true_positive_rate)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {roc_score:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    axis.set_title(f"{model_name} ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    path = output_dir / f"{model_name}_roc_curve.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_precision_recall_curve(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save the precision-recall curve for a model."""

    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = _predict_fraud_probabilities(model, x_test)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label="Model")
    axis.set_title(f"{model_name} Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    path = output_dir / f"{model_name}_precision_recall_curve.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_prediction_distribution(
    probabilities: list[float],
    y_true: pd.Series,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save fraud probability distributions by actual class."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_frame = pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "actual_class": y_true.map({0: "Legitimate", 1: "Fraud"}).values,
        }
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    sns.histplot(
        data=plot_frame,
        x="fraud_probability",
        hue="actual_class",
        bins=20,
        kde=True,
        element="step",
        ax=axis,
    )
    axis.set_title(f"{model_name} Fraud Probability Distribution")
    axis.set_xlabel("Predicted Fraud Probability")
    axis.set_ylabel("Transaction Count")
    path = output_dir / f"{model_name}_prediction_distribution.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_feature_importance(
    model: Any,
    feature_columns: list[str],
    model_name: str,
    output_dir: Path,
    top_n: int = 15,
) -> Path | None:
    """Save feature importance for tree models or coefficient magnitude for linear models."""

    output_dir.mkdir(parents=True, exist_ok=True)
    values = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])

    if values is None:
        return None

    importance_frame = (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=importance_frame,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=axis,
    )
    axis.set_title(f"{model_name} Top Feature Signals")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    path = output_dir / f"{model_name}_feature_importance.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_model_comparison(metrics_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Save a side-by-side model comparison chart."""

    output_dir.mkdir(parents=True, exist_ok=True)
    melted = metrics_frame.melt(
        id_vars="model",
        value_vars=["precision", "recall", "f1_score", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title("Fraud Model Performance Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.legend(title="Metric", loc="lower right")
    path = output_dir / "model_comparison.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _predict_fraud_probabilities(model: Any, features: pd.DataFrame) -> pd.Series:
    """Return fraud-class probabilities for models with or without predict_proba."""

    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(features)[:, FRAUD_LABEL])

    if hasattr(model, "decision_function"):
        raw_scores = pd.Series(model.decision_function(features))
        scaled_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        return scaled_scores.fillna(0.0)

    return pd.Series(model.predict(features), dtype=float)
