"""Class balancing utilities for fraud detection.

Fraud datasets are usually highly imbalanced: legitimate transactions dominate,
while fraudulent transactions are rare. If we train directly on the raw class
distribution, a model can look accurate by mostly predicting "not fraud" while
missing the cases that matter most.

This module applies resampling only to the training split. The test split must
stay untouched so evaluation reflects real-world class imbalance and avoids
data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

BalanceMethod = Literal["none", "smote", "undersample"]


@dataclass(frozen=True)
class BalanceConfig:
    """Configuration for class balancing."""

    method: BalanceMethod = "smote"
    random_state: int = 42
    sampling_strategy: float | str = "auto"


def get_class_distribution(labels: pd.Series) -> pd.DataFrame:
    """Return counts and percentages for each target class."""

    counts = labels.value_counts().sort_index()
    percentages = labels.value_counts(normalize=True).sort_index().mul(100)
    return pd.DataFrame(
        {
            "class": counts.index.astype(int),
            "count": counts.values.astype(int),
            "percentage": percentages.round(4).values,
        }
    )


def resample_training_data(
    features: pd.DataFrame,
    labels: pd.Series,
    config: BalanceConfig,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Resample training features and labels.

    Args:
        features: Training feature matrix only.
        labels: Training labels only.
        config: Balancing configuration.

    Returns:
        Balanced features, balanced labels, and metadata with before/after
        distributions.
    """

    before = get_class_distribution(labels)

    if config.method == "none":
        return features, labels, _build_metadata("none", before, before)

    if labels.nunique() < 2:
        return features, labels, _build_metadata("skipped_single_class", before, before)

    class_counts = labels.value_counts()
    minority_count = int(class_counts.min())

    if config.method == "smote":
        if minority_count < 2:
            return features, labels, _build_metadata("skipped_too_few_minority", before, before)

        # SMOTE needs k_neighbors < minority class count.
        k_neighbors = min(5, minority_count - 1)
        sampler = SMOTE(
            random_state=config.random_state,
            sampling_strategy=config.sampling_strategy,
            k_neighbors=k_neighbors,
        )
    elif config.method == "undersample":
        sampler = RandomUnderSampler(
            random_state=config.random_state,
            sampling_strategy=config.sampling_strategy,
        )
    else:
        raise ValueError(f"Unsupported balance method: {config.method}")

    balanced_features, balanced_labels = sampler.fit_resample(features, labels)
    balanced_features = pd.DataFrame(balanced_features, columns=features.columns)
    balanced_labels = pd.Series(balanced_labels, name=labels.name)
    after = get_class_distribution(balanced_labels)

    return balanced_features, balanced_labels, _build_metadata(config.method, before, after)


def plot_balance_comparison(
    balance_metadata: dict[str, object],
    output_dir: Path,
) -> Path:
    """Visualize class distribution before and after train-only resampling."""

    output_dir.mkdir(parents=True, exist_ok=True)
    before = pd.DataFrame(balance_metadata["before"]).assign(stage="Before resampling")
    after = pd.DataFrame(balance_metadata["after"]).assign(stage="After resampling")
    plot_frame = pd.concat([before, after], ignore_index=True)
    plot_frame["class"] = plot_frame["class"].map({0: "Legitimate", 1: "Fraud"})

    figure, axis = plt.subplots(figsize=(7, 5))
    sns.barplot(data=plot_frame, x="stage", y="count", hue="class", palette="Set2", ax=axis)
    axis.set_title("Training Class Distribution Before vs After Balancing")
    axis.set_xlabel("")
    axis.set_ylabel("Training Samples")
    axis.legend(title="Class")
    path = output_dir / "class_balance_before_after.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _build_metadata(
    method: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict[str, object]:
    """Create serializable resampling metadata."""

    return {
        "method": method,
        "before": before.to_dict(orient="records"),
        "after": after.to_dict(orient="records"),
    }
