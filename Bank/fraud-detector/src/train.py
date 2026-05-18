"""Model training utilities for fraud prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.preprocess import clean_data, resolve_target_column, split_features_and_labels


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training the prediction model."""

    model_path: Path = Path("models/fraud_model.joblib")
    target_column: str | None = None
    test_size: float = 0.2
    random_state: int = 42


def train_fraud_model(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Train and save a Random Forest fraud model.

    SMOTE is applied only to the training split so the test set remains a fair
    representation of the original fraud imbalance.
    """

    target_column = resolve_target_column(dataframe, config.target_column)
    cleaned_dataframe = clean_data(dataframe, target_column=target_column)
    x_train, x_test, y_train, y_test = split_features_and_labels(
        cleaned_dataframe,
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    x_train_balanced, y_train_balanced = _apply_smote_if_possible(
        x_train,
        y_train,
        random_state=config.random_state,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train_balanced, y_train_balanced)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
        if y_test.nunique() > 1
        else 0.0,
    }

    model_bundle = {
        "model": model,
        "target_column": target_column,
        "feature_columns": list(x_train.columns),
        "metrics": metrics,
    }

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, config.model_path)
    return model_bundle


def _apply_smote_if_possible(
    features: pd.DataFrame,
    labels: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Balance classes with SMOTE when the minority class has enough samples."""

    class_counts = labels.value_counts()
    if labels.nunique() < 2 or class_counts.min() < 2:
        return features, labels

    k_neighbors = min(5, int(class_counts.min()) - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    balanced_features, balanced_labels = smote.fit_resample(features, labels)
    return pd.DataFrame(balanced_features, columns=features.columns), pd.Series(balanced_labels)

