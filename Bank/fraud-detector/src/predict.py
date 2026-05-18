"""Prediction helpers for the Streamlit fraud dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import DEFAULT_IDENTIFIER_COLUMNS, add_datetime_features


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load a saved fraud model bundle from disk."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return joblib.load(model_path)


def predict_transaction(
    transaction: dict[str, Any],
    model_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Predict whether a single transaction is fraudulent."""

    feature_columns = model_bundle["feature_columns"]
    model = model_bundle["model"]
    features = prepare_transaction_features(transaction, feature_columns)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "probability": probability,
        "label": "Fraud" if prediction == 1 else "Not Fraud",
    }


def prepare_transaction_features(
    transaction: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform manual transaction input into the model feature columns."""

    dataframe = pd.DataFrame([transaction])
    drop_columns = [column for column in DEFAULT_IDENTIFIER_COLUMNS if column in dataframe.columns]
    if drop_columns:
        dataframe = dataframe.drop(columns=drop_columns)

    dataframe = add_datetime_features(dataframe, target_column="__missing_target__")
    dataframe = pd.get_dummies(dataframe, dtype=int)
    dataframe = dataframe.reindex(columns=feature_columns, fill_value=0)
    return dataframe

