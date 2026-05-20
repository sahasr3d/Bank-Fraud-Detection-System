"""Command-line entry point for the fraud detection project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocess import PreprocessConfig, run_preprocessing


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Bank Transaction Fraud Detection System"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/creditcard.csv"),
        help="Path to the raw Kaggle fraud CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed train/test files will be saved.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Fraud label column. Defaults to auto-detecting common names.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test splitting.",
    )

    return parser.parse_args()


def main() -> None:
    """Run preprocessing from the command line."""

    args = parse_args()
    config = PreprocessConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    run_preprocessing(config)


if __name__ == "__main__":
    main()

