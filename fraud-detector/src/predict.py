"""Prediction pipeline for saved fraud models.

The prediction layer loads a persisted model bundle, aligns incoming transaction
features to the training feature schema, and returns both fraud probabilities
and a decision label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import PATHS, SETTINGS
from src.preprocess import DEFAULT_IDENTIFIER_COLUMNS, add_datetime_features


def load_model_bundle(model_path: Path = PATHS.models_dir / "fraud_model.joblib") -> dict[str, Any]:
    """Load a saved fraud model bundle from disk."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_columns", "threshold"}
    missing_keys = required_keys.difference(bundle)
    if missing_keys:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing_keys)}")

    return bundle


def predict_transaction(
    transaction: dict[str, Any],
    model_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict whether one transaction is fraudulent."""

    prediction_frame = predict_batch_transactions(
        pd.DataFrame([transaction]),
        model_bundle=model_bundle,
    )
    return prediction_frame.iloc[0].to_dict()


def predict_single_transaction(
    transaction: dict[str, Any],
    model_path: Path = PATHS.models_dir / "fraud_model.joblib",
) -> dict[str, Any]:
    """Load a model and predict fraud risk for one transaction."""

    bundle = load_model_bundle(model_path)
    return predict_transaction(transaction, bundle)


def predict_batch_transactions(
    transactions: pd.DataFrame | Path | str,
    model_path: Path = PATHS.models_dir / "fraud_model.joblib",
    model_bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict fraud risk for many transactions.

    Args:
        transactions: Raw transaction dataframe or CSV path.
        model_path: Saved model path used when model_bundle is not provided.
        model_bundle: Optional already-loaded model bundle.

    Returns:
        DataFrame containing prediction, probability, confidence, and label.
    """

    bundle = model_bundle or load_model_bundle(model_path)
    dataframe = _load_prediction_input(transactions)
    features = prepare_transaction_features(dataframe, bundle["feature_columns"])
    probabilities = pd.Series(bundle["model"].predict_proba(features)[:, 1])
    threshold = float(bundle.get("threshold", SETTINGS.fraud_threshold))
    predictions = (probabilities >= threshold).astype(int)

    result = pd.DataFrame(
        {
            "prediction": predictions,
            "probability": probabilities.round(6),
            "confidence_score": probabilities.where(
                predictions.eq(1),
                1 - probabilities,
            ).round(6),
            "label": predictions.map({0: "Not Fraud", 1: "Fraud"}),
        }
    )
    return result


def prepare_transaction_features(
    transactions: dict[str, Any] | pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform raw or partially processed transactions into model features."""

    dataframe = pd.DataFrame([transactions]) if isinstance(transactions, dict) else transactions.copy()

    # Remove target columns if a labeled batch is passed for scoring.
    possible_targets = {"Class", "isFraud", "fraud", "Fraud", "fraud_label", "label", "target"}
    dataframe = dataframe.drop(columns=[col for col in possible_targets if col in dataframe.columns])

    # Already processed data can be aligned directly to the training schema.
    if set(feature_columns).issubset(set(dataframe.columns)):
        return dataframe.reindex(columns=feature_columns, fill_value=0)

    drop_columns = [column for column in DEFAULT_IDENTIFIER_COLUMNS if column in dataframe.columns]
    if drop_columns:
        dataframe = dataframe.drop(columns=drop_columns)

    dataframe = add_datetime_features(dataframe, target_column="__missing_target__")
    dataframe = pd.get_dummies(dataframe, dtype=int)
    dataframe = dataframe.reindex(columns=feature_columns, fill_value=0)
    return dataframe


def _load_prediction_input(transactions: pd.DataFrame | Path | str) -> pd.DataFrame:
    """Load prediction input from a dataframe or CSV path."""

    if isinstance(transactions, pd.DataFrame):
        return transactions.copy()

    path = Path(transactions)
    if not path.exists():
        raise FileNotFoundError(f"Prediction input not found: {path}")

    return pd.read_csv(path)

