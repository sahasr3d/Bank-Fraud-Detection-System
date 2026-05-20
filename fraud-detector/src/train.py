"""Training pipeline for fraud detection models.

This module trains multiple models on the processed Phase 1 datasets and keeps
all resampling inside the training split to avoid data leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.balance import BalanceConfig, plot_balance_comparison, resample_training_data
from src.config import PATHS, SETTINGS
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_prediction_distribution,
    plot_roc_curve,
    save_evaluation_report,
)
from src.logging_utils import get_logger
from src.preprocess import clean_data, resolve_target_column, split_features_and_labels


LOGGER = get_logger(__name__, PATHS.reports_dir / "training.log")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a complete training run."""

    processed_dir: Path = PATHS.processed_dir
    raw_data_path: Path = PATHS.raw_data_path
    model_dir: Path = PATHS.models_dir
    visualizations_dir: Path = PATHS.visualizations_dir
    reports_dir: Path = PATHS.reports_dir
    target_column: str | None = None
    test_size: float = SETTINGS.test_size
    random_state: int = SETTINGS.random_state
    threshold: float = SETTINGS.fraud_threshold
    balance_method: str = "smote"
    # Kept for backward compatibility with the Streamlit app's earlier call.
    model_path: Path | None = None


def load_training_data(
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], str]:
    """Load processed train/test data, or create it from raw data if needed."""

    x_train_path = config.processed_dir / "X_train.csv"
    x_test_path = config.processed_dir / "X_test.csv"
    y_train_path = config.processed_dir / "y_train.csv"
    y_test_path = config.processed_dir / "y_test.csv"

    if all(path.exists() for path in [x_train_path, x_test_path, y_train_path, y_test_path]):
        LOGGER.info("Loading processed training data from %s", config.processed_dir)
        x_train = pd.read_csv(x_train_path)
        x_test = pd.read_csv(x_test_path)
        y_train = pd.read_csv(y_train_path).iloc[:, 0].astype(int)
        y_test = pd.read_csv(y_test_path).iloc[:, 0].astype(int)
        return x_train, x_test, y_train, y_test, list(x_train.columns), "target"

    LOGGER.info("Processed files not found. Building training data from raw CSV.")
    raw_dataframe = pd.read_csv(config.raw_data_path)
    target_column = resolve_target_column(raw_dataframe, config.target_column)
    cleaned_dataframe = clean_data(raw_dataframe, target_column=target_column)
    x_train, x_test, y_train, y_test = split_features_and_labels(
        cleaned_dataframe,
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    return x_train, x_test, y_train, y_test, list(x_train.columns), target_column


def build_model_registry(random_state: int) -> dict[str, Any]:
    """Create the supported model set with reproducible hyperparameters."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_all_models(config: TrainingConfig | None = None) -> dict[str, Any]:
    """Train, evaluate, visualize, and save all fraud models."""

    config = config or TrainingConfig()
    _ensure_output_dirs(config)

    x_train, x_test, y_train, y_test, feature_columns, target_column = load_training_data(config)
    x_train_balanced, y_train_balanced, balance_metadata = resample_training_data(
        x_train,
        y_train,
        BalanceConfig(
            method=config.balance_method,  # type: ignore[arg-type]
            random_state=config.random_state,
        ),
    )

    LOGGER.info("Training with balance method: %s", balance_metadata["method"])
    plot_balance_comparison(balance_metadata, config.visualizations_dir)
    models = build_model_registry(config.random_state)
    results: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        LOGGER.info("Training model: %s", model_name)
        model.fit(x_train_balanced, y_train_balanced)
        evaluation = evaluate_model(model, x_test, y_test, threshold=config.threshold)
        metrics = evaluation["metrics"]
        metric_rows.append({"model": model_name, **metrics})

        bundle = {
            "model": model,
            "model_name": model_name,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "threshold": config.threshold,
            "metrics": metrics,
            "balance_metadata": balance_metadata,
            "training_config": _serializable_config(config),
        }

        model_path = config.model_dir / f"{model_name}.joblib"
        joblib.dump(bundle, model_path)
        save_evaluation_report(model_name, evaluation, config.reports_dir)
        _save_model_visualizations(
            model=model,
            model_name=model_name,
            x_test=x_test,
            y_test=y_test,
            probabilities=evaluation["probabilities"],
            feature_columns=feature_columns,
            output_dir=config.visualizations_dir,
            threshold=config.threshold,
        )

        results[model_name] = {
            "model_path": str(model_path),
            "metrics": metrics,
            "evaluation": evaluation,
        }

    metrics_frame = pd.DataFrame(metric_rows).sort_values("f1_score", ascending=False)
    metrics_path = config.reports_dir / "model_comparison.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    plot_model_comparison(metrics_frame, config.visualizations_dir)

    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_model_path = config.model_dir / f"{best_model_name}.joblib"
    default_model_path = config.model_path or config.model_dir / "fraud_model.joblib"
    joblib.dump(joblib.load(best_model_path), default_model_path)

    summary = {
        "best_model": best_model_name,
        "best_model_path": str(best_model_path),
        "default_model_path": str(default_model_path),
        "metrics_path": str(metrics_path),
        "balance_metadata": balance_metadata,
        "models": {name: result["metrics"] for name, result in results.items()},
    }
    with (config.reports_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    LOGGER.info("Training complete. Best model: %s", best_model_name)
    return summary


def train_fraud_model(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Backward-compatible helper used by the Streamlit app.

    It trains the full model set from an in-memory dataframe, saves all models,
    and returns the selected default model bundle.
    """

    config = _normalize_config(config)
    _ensure_output_dirs(config)

    target_column = resolve_target_column(dataframe, config.target_column)
    cleaned_dataframe = clean_data(dataframe, target_column=target_column)
    x_train, x_test, y_train, y_test = split_features_and_labels(
        cleaned_dataframe,
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    x_train_balanced, y_train_balanced, balance_metadata = resample_training_data(
        x_train,
        y_train,
        BalanceConfig(method=config.balance_method, random_state=config.random_state),  # type: ignore[arg-type]
    )
    plot_balance_comparison(balance_metadata, config.visualizations_dir)

    models = build_model_registry(config.random_state)
    metric_rows: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        model.fit(x_train_balanced, y_train_balanced)
        evaluation = evaluate_model(model, x_test, y_test, threshold=config.threshold)
        metrics = evaluation["metrics"]
        metric_rows.append({"model": model_name, **metrics})

        bundle = {
            "model": model,
            "model_name": model_name,
            "target_column": target_column,
            "feature_columns": list(x_train.columns),
            "threshold": config.threshold,
            "metrics": metrics,
            "balance_metadata": balance_metadata,
            "training_config": _serializable_config(config),
        }
        bundles[model_name] = bundle
        joblib.dump(bundle, config.model_dir / f"{model_name}.joblib")
        save_evaluation_report(model_name, evaluation, config.reports_dir)
        _save_model_visualizations(
            model=model,
            model_name=model_name,
            x_test=x_test,
            y_test=y_test,
            probabilities=evaluation["probabilities"],
            feature_columns=list(x_train.columns),
            output_dir=config.visualizations_dir,
            threshold=config.threshold,
        )

    metrics_frame = pd.DataFrame(metric_rows).sort_values("f1_score", ascending=False)
    metrics_frame.to_csv(config.reports_dir / "model_comparison.csv", index=False)
    plot_model_comparison(metrics_frame, config.visualizations_dir)

    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_bundle = bundles[best_model_name]
    joblib.dump(best_bundle, config.model_path or config.model_dir / "fraud_model.joblib")
    return best_bundle


def _save_model_visualizations(
    model: Any,
    model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: list[float],
    feature_columns: list[str],
    output_dir: Path,
    threshold: float,
) -> None:
    """Generate standard model performance charts."""

    plot_confusion_matrix(y_test, probabilities, model_name, output_dir, threshold=threshold)
    plot_roc_curve(model, x_test, y_test, model_name, output_dir)
    plot_precision_recall_curve(model, x_test, y_test, model_name, output_dir)
    plot_prediction_distribution(probabilities, y_test, model_name, output_dir)
    plot_feature_importance(model, feature_columns, model_name, output_dir)


def _ensure_output_dirs(config: TrainingConfig) -> None:
    """Create output folders required by Phase 2."""

    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.visualizations_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)


def _serializable_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert dataclass config to JSON-safe values."""

    config_dict = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config_dict.items()
    }


def _normalize_config(config: TrainingConfig) -> TrainingConfig:
    """Fill derived paths when a legacy caller passes only model_path."""

    if config.model_path is None:
        return config

    return TrainingConfig(
        processed_dir=config.processed_dir,
        raw_data_path=config.raw_data_path,
        model_dir=config.model_path.parent,
        visualizations_dir=config.visualizations_dir,
        reports_dir=config.reports_dir,
        target_column=config.target_column,
        test_size=config.test_size,
        random_state=config.random_state,
        threshold=config.threshold,
        balance_method=config.balance_method,
        model_path=config.model_path,
    )


if __name__ == "__main__":
    train_all_models()
