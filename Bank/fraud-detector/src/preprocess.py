"""Data loading, cleaning, statistics, and train/test splitting utilities.

This module is intentionally focused on the first ML engineering phase:
turning the raw Kaggle fraud dataset into clean, reproducible processed files.

Expected default dataset:
    data/raw/creditcard.csv

The common Kaggle credit card fraud dataset uses `Class` as the target column,
where 1 means fraudulent and 0 means legitimate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TARGET_CANDIDATES = (
    "Class",
    "isFraud",
    "fraud",
    "Fraud",
    "fraud_label",
    "label",
    "target",
)
DEFAULT_IDENTIFIER_COLUMNS = ("transaction_id", "card_number", "account_number", "customer_id")


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for preprocessing raw transaction data."""

    input_path: Path = Path("data/raw/creditcard.csv")
    output_dir: Path = Path("data/processed")
    target_column: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    drop_duplicates: bool = True


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a CSV dataset from disk.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file exists but is empty.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "Download the Kaggle fraud CSV and place it in data/raw/."
        )

    try:
        dataframe = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {file_path}") from exc

    if dataframe.empty:
        raise ValueError(f"Dataset has no rows: {file_path}")

    return dataframe


def resolve_target_column(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
    candidates: tuple[str, ...] = DEFAULT_TARGET_CANDIDATES,
) -> str:
    """Find the fraud label column.

    Args:
        dataframe: Input dataset.
        target_column: Explicit target column name, if provided.
        candidates: Common fraud label column names to try.

    Returns:
        Name of the resolved target column.

    Raises:
        ValueError: If no valid target column can be found.
    """

    if target_column:
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(dataframe.columns)}"
            )
        return target_column

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "Could not infer target column. Pass --target-column explicitly. "
        f"Tried: {list(candidates)}"
    )


def clean_data(
    dataframe: pd.DataFrame,
    target_column: str,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Clean raw transaction data.

    Cleaning steps:
    - Remove exact duplicate rows.
    - Fill missing numeric values with the median.
    - Fill missing categorical values with the most frequent value.
    - One-hot encode categorical feature columns.

    Args:
        dataframe: Raw input DataFrame.
        target_column: Fraud label column.
        drop_duplicates: Whether to remove exact duplicate rows.

    Returns:
        Cleaned DataFrame containing encoded features and the target column.
    """

    cleaned = dataframe.copy()

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Normalize column names by stripping accidental whitespace from CSV headers.
    cleaned.columns = [column.strip() for column in cleaned.columns]
    target_column = target_column.strip()

    if target_column not in cleaned.columns:
        raise ValueError(f"Target column '{target_column}' is missing after cleaning.")

    identifier_columns = [
        column
        for column in DEFAULT_IDENTIFIER_COLUMNS
        if column in cleaned.columns and column != target_column
    ]
    if identifier_columns:
        cleaned = cleaned.drop(columns=identifier_columns)

    cleaned = add_datetime_features(cleaned, target_column=target_column)

    feature_columns = [column for column in cleaned.columns if column != target_column]
    numeric_columns = cleaned[feature_columns].select_dtypes(include=[np.number]).columns
    categorical_columns = [
        column for column in feature_columns if column not in set(numeric_columns)
    ]

    for column in numeric_columns:
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    for column in categorical_columns:
        mode_values = cleaned[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned[column] = cleaned[column].fillna(fill_value)

    if categorical_columns:
        cleaned = pd.get_dummies(
            cleaned,
            columns=categorical_columns,
            drop_first=False,
            dtype=int,
        )

    # Keep the fraud label numeric for scikit-learn metrics and models.
    cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="raise")

    return cleaned


def add_datetime_features(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Convert datetime-like feature columns into model-friendly numeric features.

    Raw timestamp strings usually have too many unique values for useful one-hot
    encoding. Extracting hour/day/month features gives the model useful behavior
    signals while keeping the feature matrix compact.
    """

    transformed = dataframe.copy()
    feature_columns = [column for column in transformed.columns if column != target_column]

    for column in feature_columns:
        column_lower = column.lower()
        looks_like_datetime = "time" in column_lower or "date" in column_lower

        if not looks_like_datetime or pd.api.types.is_numeric_dtype(transformed[column]):
            continue

        parsed = pd.to_datetime(transformed[column], errors="coerce")
        parsed_ratio = parsed.notna().mean()

        if parsed_ratio < 0.5:
            continue

        transformed[f"{column}_hour"] = parsed.dt.hour
        transformed[f"{column}_dayofweek"] = parsed.dt.dayofweek
        transformed[f"{column}_day"] = parsed.dt.day
        transformed[f"{column}_month"] = parsed.dt.month
        transformed[f"{column}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
        transformed = transformed.drop(columns=[column])

    return transformed


def build_dataset_statistics(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Build simple dataset statistics for logging and metadata."""

    target_counts = dataframe[target_column].value_counts(dropna=False).to_dict()
    target_percentages = (
        dataframe[target_column].value_counts(normalize=True, dropna=False).mul(100).round(4)
    ).to_dict()

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values_total": int(dataframe.isna().sum().sum()),
        "target_column": target_column,
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "target_percentages": {
            str(key): float(value) for key, value in target_percentages.items()
        },
    }


def print_dataset_statistics(dataframe: pd.DataFrame, target_column: str) -> None:
    """Print beginner-friendly dataset statistics to the console."""

    stats = build_dataset_statistics(dataframe, target_column)
    print("\nDataset statistics")
    print("------------------")
    print(f"Rows: {stats['rows']}")
    print(f"Columns: {stats['columns']}")
    print(f"Duplicate rows: {stats['duplicate_rows']}")
    print(f"Missing values: {stats['missing_values_total']}")
    print(f"Target column: {stats['target_column']}")
    print(f"Target counts: {stats['target_counts']}")
    print(f"Target percentages: {stats['target_percentages']}")

    numeric_summary = dataframe.describe().transpose()
    print("\nNumeric summary")
    print("---------------")
    print(numeric_summary[["mean", "std", "min", "max"]].round(4))


def split_features_and_labels(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cleaned data into train/test features and labels."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    features = dataframe.drop(columns=[target_column])
    labels = dataframe[target_column].astype(int)

    class_counts = labels.value_counts()
    can_stratify = labels.nunique() > 1 and class_counts.min() >= 2
    stratify_labels = labels if can_stratify else None

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )


def save_processed_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Save processed train/test datasets and metadata to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(output_dir / "X_train.csv", index=False)
    x_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(name="target").to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(name="target").to_csv(output_dir / "y_test.csv", index=False)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def run_preprocessing(config: PreprocessConfig) -> dict[str, Any]:
    """Run the full preprocessing workflow.

    Args:
        config: Preprocessing configuration.

    Returns:
        Metadata describing the processed dataset.
    """

    raw_dataframe = load_dataset(config.input_path)
    target_column = resolve_target_column(raw_dataframe, config.target_column)

    print_dataset_statistics(raw_dataframe, target_column)
    cleaned_dataframe = clean_data(
        raw_dataframe,
        target_column=target_column,
        drop_duplicates=config.drop_duplicates,
    )

    print("\nAfter cleaning")
    print("--------------")
    print_dataset_statistics(cleaned_dataframe, target_column)

    x_train, x_test, y_train, y_test = split_features_and_labels(
        cleaned_dataframe,
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    metadata = {
        "config": {
            **asdict(config),
            "input_path": str(config.input_path),
            "output_dir": str(config.output_dir),
        },
        "raw_statistics": build_dataset_statistics(raw_dataframe, target_column),
        "cleaned_statistics": build_dataset_statistics(cleaned_dataframe, target_column),
        "feature_columns": list(x_train.columns),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    save_processed_data(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=config.output_dir,
        metadata=metadata,
    )

    print(f"\nProcessed files saved to: {config.output_dir.resolve()}")
    return metadata
