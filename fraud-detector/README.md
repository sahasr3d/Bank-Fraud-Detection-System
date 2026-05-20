# Bank Transaction Fraud Detection System

Production-oriented machine learning project for detecting fraudulent banking
and credit-card transactions using pandas, scikit-learn, XGBoost, SMOTE,
Streamlit, joblib, and MySQL.

## Project Structure

```text
fraud-detector/
|-- app/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|-- visualizations/
|-- requirements.txt
|-- README.md
`-- main.py
```

## Run Phase 1: Preprocessing

Place the raw dataset at:

```text
data/raw/creditcard.csv
```

Important: do NOT commit raw datasets to the repository. Add the raw files
to your local `.gitignore` (or follow the project's `.gitignore`) and store
datasets separately (e.g., a secure data storage or an internal data
registry). See `SECURITY.md` for more guidance.

Run:

```bash
python main.py
```

Processed outputs:

```text
data/processed/
|-- X_train.csv
|-- X_test.csv
|-- y_train.csv
|-- y_test.csv
`-- metadata.json
```

## Run Phase 2: EDA, Balancing, Training, Evaluation

Run the full model training and reporting pipeline:

```bash
python -m src.train
```

This trains:

- Logistic Regression
- Random Forest
- XGBoost

SMOTE is applied only to the training split. This avoids data leakage because
the test split keeps the original fraud imbalance.

## Streamlit Interface

If port `8501` is free:

```bash
streamlit run app/streamlit_app.py
```

If another Streamlit app is using `8501`:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

Default dashboard login:

```text
User ID: admin
Password: (demo-only; configurable via the `FRAUD_APP_PASSWORD` environment variable)
```

## Phase 2 Modules

### `src/balance.py`

Concept: Handles class imbalance with SMOTE oversampling and optional random
undersampling.

Why it matters: Fraud is rare. Without balancing, a model may learn to predict
almost everything as legitimate.

Best practice: Resampling happens after the train/test split and only on the
training data.

### `src/train.py`

Concept: Trains multiple fraud classification models in a reusable pipeline.

Why it matters: Different model families have different strengths. A comparison
framework helps choose the best tradeoff between recall, precision, and overall
risk ranking.

### `src/evaluate.py`

Concept: Evaluates fraud models with precision, recall, F1-score, ROC-AUC,
confusion matrix, ROC curve, and precision-recall curve.

Why it matters: Accuracy alone is misleading on imbalanced fraud datasets. A
high-accuracy model can still miss most fraud.

### `src/predict.py`

Concept: Loads saved model bundles and predicts fraud risk for single or batch
transactions.

Why it matters: A production model must return probability and confidence, not
only a hard class label.

### `notebooks/EDA.ipynb`

Concept: Analyst-facing exploratory data analysis notebook.

Includes:

- Fraud vs legitimate transaction analysis
- Class distribution visualization
- Transaction amount analysis
- Correlation heatmaps
- Feature signal exploration
- Outlier analysis
- Time-based transaction analysis

## Phase 2 Outputs

Models:

```text
models/
|-- logistic_regression.joblib
|-- random_forest.joblib
|-- xgboost.joblib
`-- fraud_model.joblib
```

Reports:

```text
reports/
|-- model_comparison.csv
|-- training_summary.json
|-- logistic_regression_evaluation.json
|-- random_forest_evaluation.json
`-- xgboost_evaluation.json
```

Visualizations:

```text
visualizations/
|-- class_balance_before_after.png
|-- model_comparison.png
|-- *_confusion_matrix.png
|-- *_roc_curve.png
|-- *_precision_recall_curve.png
|-- *_prediction_distribution.png
`-- *_feature_importance.png
```

## Model Tradeoffs

- Logistic Regression is fast and interpretable. It is useful as a baseline,
  but may miss nonlinear fraud behavior.
- Random Forest handles nonlinear tabular patterns well and provides feature
  importance. It can be heavier than linear models.
- XGBoost is often strong for structured fraud data, but it needs tuning and
  threshold calibration, especially with small datasets.

## Current Sample Run

The included sample dataset has only 100 rows. These metrics confirm the
pipeline works, but they are not production-quality model evidence.

```text
model,precision,recall,f1_score,roc_auc
random_forest,0.3333,0.3333,0.3333,0.6078
logistic_regression,0.1579,1.0000,0.2727,0.6078
xgboost,0.0000,0.0000,0.0000,0.6863
```

SMOTE example from the latest run:

```text
Before SMOTE: 69 legitimate, 11 fraud
After SMOTE:  69 legitimate, 69 fraud
```

## Example Prediction

```python
from src.predict import predict_single_transaction

transaction = {
    "transaction_id": "TXN-DEMO",
    "timestamp": "2026-05-19T12:30:00",
    "card_number": "4111111111111111",
    "merchant": "Online Store",
    "location": "Riyadh",
    "transaction_amount": 1200.0,
    "device_type": "Mobile",
    "is_international": 0,
    "failed_attempts": 1,
}

print(predict_single_transaction(transaction))
```

Example output:

```text
{'prediction': 0, 'probability': 0.236667, 'confidence_score': 0.763333, 'label': 'Not Fraud'}
```

