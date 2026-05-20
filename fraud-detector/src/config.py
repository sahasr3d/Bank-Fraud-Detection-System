"""Central project configuration.

Keeping paths and global settings in one place makes the ML pipeline easier to
run from scripts, notebooks, and the Streamlit app without hard-coded values
spreading across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the fraud detection pipeline."""

    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    visualizations_dir: Path = PROJECT_ROOT / "visualizations"
    reports_dir: Path = PROJECT_ROOT / "reports"
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"


@dataclass(frozen=True)
class MLSettings:
    """Global machine learning settings."""

    random_state: int = 42
    test_size: float = 0.2
    fraud_threshold: float = 0.5


PATHS = ProjectPaths()
SETTINGS = MLSettings()

