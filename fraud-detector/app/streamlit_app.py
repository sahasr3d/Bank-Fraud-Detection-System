"""Streamlit interface for the fraud detection project.

This dashboard lets a user inspect the raw transaction dataset and run the
preprocessing phase without using the command line.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocess import (  # noqa: E402
    PreprocessConfig,
    build_dataset_statistics,
    clean_data,
    load_dataset,
    resolve_target_column,
    save_processed_data,
    split_features_and_labels,
)
from src.predict import load_model_bundle, predict_transaction  # noqa: E402
from src.train import TrainingConfig, train_fraud_model  # noqa: E402


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
UPLOADED_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "uploaded_transactions.csv"


st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply small visual refinements for a cleaner dashboard."""

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
        }
        div[data-testid="stTabs"] button {
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_expected_credentials() -> tuple[str, str]:
    """Return dashboard credentials from environment variables or local defaults."""

    username = os.getenv("FRAUD_APP_USERNAME", "admin")
    password = os.getenv("FRAUD_APP_PASSWORD", "admin123")
    return username, password


def render_login() -> bool:
    """Render login form and return whether the user is authenticated."""

    if st.session_state.get("authenticated"):
        return True

    expected_username, expected_password = get_expected_credentials()
    _, center, _ = st.columns([1, 1.2, 1])

    with center:
        st.title("Bank Fraud Detection")
        st.caption("Secure dashboard login")
        with st.form("login_form"):
            username = st.text_input("User ID")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if username == expected_username and password == expected_password:
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("Invalid user ID or password.")

        with st.expander("Demo credentials"):
            st.write("User ID: `admin`")
            st.write("Password: set via `FRAUD_APP_PASSWORD` environment variable (demo only)")

    return False


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    """Load CSV data from a file path for Streamlit caching."""

    return load_dataset(Path(path))


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Save an uploaded CSV to the raw data folder and return its path."""

    UPLOADED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOADED_DATASET_PATH.write_bytes(uploaded_file.getbuffer())
    return UPLOADED_DATASET_PATH


def show_metric_row(stats: dict[str, Any]) -> None:
    """Display high-level dataset statistics."""

    target_counts = stats.get("target_counts", {})
    target_percentages = stats.get("target_percentages", {})
    legitimate_count = target_counts.get("0", 0)
    fraud_count = target_counts.get("1", 0)
    fraud_rate = target_percentages.get("1", 0.0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{stats['rows']:,}")
    col2.metric("Columns", f"{stats['columns']:,}")
    col3.metric("Missing Values", f"{stats['missing_values_total']:,}")
    col4.metric("Legitimate", f"{legitimate_count:,}")
    col5.metric("Fraud Rate", f"{fraud_rate:.2f}% ({fraud_count:,})")


def plot_class_distribution(dataframe: pd.DataFrame, target_column: str) -> plt.Figure:
    """Create a class distribution chart."""

    figure, axis = plt.subplots(figsize=(7, 4))
    sns.countplot(data=dataframe, x=target_column, hue=target_column, palette="Set2", ax=axis)
    axis.set_title("Fraud Class Distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Transaction Count")
    axis.legend_.remove()
    return figure


def plot_amount_distribution(dataframe: pd.DataFrame, target_column: str) -> plt.Figure:
    """Create a transaction amount distribution chart when the column exists."""

    figure, axis = plt.subplots(figsize=(7, 4))
    if "transaction_amount" not in dataframe.columns:
        axis.text(0.5, 0.5, "transaction_amount column not found", ha="center")
        axis.axis("off")
        return figure

    sns.histplot(
        data=dataframe,
        x="transaction_amount",
        hue=target_column,
        bins=25,
        kde=True,
        palette="Set1",
        ax=axis,
    )
    axis.set_title("Transaction Amount Distribution")
    axis.set_xlabel("Transaction Amount")
    axis.set_ylabel("Count")
    return figure


def plot_correlation_heatmap(dataframe: pd.DataFrame, target_column: str) -> plt.Figure:
    """Create a compact heatmap of correlations with the fraud label."""

    numeric_dataframe = dataframe.select_dtypes(include="number")
    figure, axis = plt.subplots(figsize=(8, 5))

    if target_column not in numeric_dataframe.columns or numeric_dataframe.shape[1] < 2:
        axis.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center")
        axis.axis("off")
        return figure

    correlations = (
        numeric_dataframe.corr(numeric_only=True)[[target_column]]
        .drop(index=target_column, errors="ignore")
        .sort_values(by=target_column, key=lambda series: series.abs(), ascending=False)
        .head(12)
    )
    sns.heatmap(correlations, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=axis)
    axis.set_title("Top Numeric Correlations With Fraud")
    return figure


def run_preprocessing_from_dataframe(
    dataframe: pd.DataFrame,
    config: PreprocessConfig,
) -> dict[str, Any]:
    """Run preprocessing for the dashboard and save processed outputs."""

    target_column = resolve_target_column(dataframe, config.target_column)
    cleaned_dataframe = clean_data(
        dataframe,
        target_column=target_column,
        drop_duplicates=config.drop_duplicates,
    )
    x_train, x_test, y_train, y_test = split_features_and_labels(
        cleaned_dataframe,
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    metadata = {
        "config": {
            "input_path": str(config.input_path),
            "output_dir": str(config.output_dir),
            "target_column": config.target_column,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "drop_duplicates": config.drop_duplicates,
        },
        "raw_statistics": build_dataset_statistics(dataframe, target_column),
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
    return metadata


def read_metadata(output_dir: Path) -> dict[str, Any] | None:
    """Read processed metadata if it exists."""

    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_prediction_tab(dataframe: pd.DataFrame, target_column: str) -> None:
    """Render the credit-card transaction prediction interface."""

    st.subheader("Credit Card Fraud Check")
    st.write("Enter transaction details and run the saved fraud model.")

    model_exists = DEFAULT_MODEL_PATH.exists()
    status_col, action_col = st.columns([2, 1])
    with status_col:
        if model_exists:
            st.success(f"Model ready: {DEFAULT_MODEL_PATH.name}")
        else:
            st.warning("No trained model found yet.")
    with action_col:
        if st.button("Train model", type="primary", use_container_width=True):
            try:
                with st.spinner("Training Random Forest model with SMOTE balancing..."):
                    bundle = train_fraud_model(
                        dataframe,
                        TrainingConfig(
                            model_path=DEFAULT_MODEL_PATH,
                            target_column=target_column,
                        ),
                    )
                st.success("Model trained and saved.")
                st.json(bundle["metrics"])
            except Exception as exc:
                st.error(str(exc))

    st.divider()

    with st.form("transaction_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            card_number = st.text_input("Card Number", value="4111111111111111")
            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=1200.0,
                step=50.0,
            )
            failed_attempts = st.number_input(
                "Failed Attempts",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
            )
        with col2:
            merchant = st.text_input("Merchant", value="Online Store")
            location = st.text_input("Location", value="Riyadh")
            device_type = st.selectbox(
                "Device Type",
                ["Mobile", "Desktop", "ATM", "POS"],
            )
        with col3:
            is_international = st.selectbox("International Transaction", [0, 1])
            transaction_date = st.date_input("Transaction Date", value=datetime.now().date())
            transaction_time = st.time_input("Transaction Time", value=datetime.now().time())

        submitted = st.form_submit_button("Check Fraud Risk", type="primary", use_container_width=True)

    if not submitted:
        return

    if not DEFAULT_MODEL_PATH.exists():
        st.error("Train the model first, then run prediction.")
        return

    timestamp = datetime.combine(transaction_date, transaction_time).isoformat()
    transaction = {
        "transaction_id": "MANUAL_INPUT",
        "timestamp": timestamp,
        "card_number": card_number,
        "merchant": merchant,
        "location": location,
        "transaction_amount": float(transaction_amount),
        "device_type": device_type,
        "is_international": int(is_international),
        "failed_attempts": int(failed_attempts),
    }

    try:
        model_bundle = load_model_bundle(DEFAULT_MODEL_PATH)
        result = predict_transaction(transaction, model_bundle)
    except Exception as exc:
        st.error(str(exc))
        return

    probability_percent = result["probability"] * 100
    if result["prediction"] == 1:
        st.error(f"Prediction: {result['label']} ({probability_percent:.2f}% fraud probability)")
    else:
        st.success(f"Prediction: {result['label']} ({probability_percent:.2f}% fraud probability)")

    st.progress(min(max(result["probability"], 0.0), 1.0))


def main() -> None:
    """Render the Streamlit dashboard."""

    apply_page_styles()

    if not render_login():
        return

    st.title("Bank Transaction Fraud Detection")
    st.caption("Dataset inspection and preprocessing dashboard")

    with st.sidebar:
        st.header("Session")
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

        st.header("Data Source")
        source = st.radio(
            "Choose dataset",
            ["Project CSV", "Upload CSV"],
            help="Use the existing raw CSV or upload another transaction dataset.",
        )

        uploaded_file = None
        dataset_path = DEFAULT_DATASET_PATH
        if source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
            if uploaded_file is not None:
                dataset_path = save_uploaded_file(uploaded_file)

        st.header("Preprocessing")
        target_choice = st.text_input(
            "Target column",
            value="",
            placeholder="Auto-detect, e.g. fraud_label",
        )
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)
        drop_duplicates = st.checkbox("Drop duplicate rows", value=True)

    if source == "Upload CSV" and uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        dataframe = load_csv_from_path(str(dataset_path))
        target_column = resolve_target_column(
            dataframe,
            target_choice.strip() or None,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    stats = build_dataset_statistics(dataframe, target_column)
    show_metric_row(stats)

    tab_overview, tab_charts, tab_predict, tab_preprocess, tab_outputs = st.tabs(
        ["Overview", "Charts", "Predict Fraud", "Preprocess", "Processed Files"]
    )

    with tab_overview:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Dataset Preview")
            st.dataframe(dataframe.head(20), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Schema")
            schema = pd.DataFrame(
                {
                    "column": dataframe.columns,
                    "type": [str(dtype) for dtype in dataframe.dtypes],
                    "missing": dataframe.isna().sum().values,
                }
            )
            st.dataframe(schema, use_container_width=True, hide_index=True)

    with tab_charts:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(plot_class_distribution(dataframe, target_column))
        with chart_col2:
            st.pyplot(plot_amount_distribution(dataframe, target_column))
        st.pyplot(plot_correlation_heatmap(dataframe, target_column))

    with tab_predict:
        render_prediction_tab(dataframe, target_column)

    with tab_preprocess:
        st.subheader("Run Preprocessing")
        st.write(
            "This creates clean train/test files for the next ML phase. "
            "SMOTE and model training will use these outputs."
        )

        if st.button("Run preprocessing", type="primary", use_container_width=True):
            config = PreprocessConfig(
                input_path=dataset_path,
                output_dir=DEFAULT_OUTPUT_DIR,
                target_column=target_column,
                test_size=float(test_size),
                random_state=int(random_state),
                drop_duplicates=drop_duplicates,
            )
            try:
                with st.spinner("Cleaning, encoding, splitting, and saving data..."):
                    metadata = run_preprocessing_from_dataframe(dataframe, config)
                st.success("Preprocessing complete.")
                st.json(
                    {
                        "train_rows": metadata["train_rows"],
                        "test_rows": metadata["test_rows"],
                        "features": len(metadata["feature_columns"]),
                        "output_dir": str(DEFAULT_OUTPUT_DIR),
                    }
                )
            except Exception as exc:
                st.error(str(exc))

    with tab_outputs:
        st.subheader("Saved Processed Data")
        metadata = read_metadata(DEFAULT_OUTPUT_DIR)
        if metadata is None:
            st.warning("No processed metadata found yet. Run preprocessing first.")
            return

        output_files = sorted(DEFAULT_OUTPUT_DIR.glob("*"))
        file_table = pd.DataFrame(
            {
                "file": [path.name for path in output_files if path.is_file()],
                "size_kb": [round(path.stat().st_size / 1024, 2) for path in output_files if path.is_file()],
            }
        )
        st.dataframe(file_table, use_container_width=True, hide_index=True)
        st.json(
            {
                "target_column": metadata["cleaned_statistics"]["target_column"],
                "train_rows": metadata["train_rows"],
                "test_rows": metadata["test_rows"],
                "feature_count": len(metadata["feature_columns"]),
            }
        )


if __name__ == "__main__":
    main()
