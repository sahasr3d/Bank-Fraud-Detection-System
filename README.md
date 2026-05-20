# Bank Transaction Fraud Detection System

A production-oriented machine learning project for detecting fraudulent banking
and credit-card transactions. The system includes data preprocessing, exploratory
data analysis, class imbalance handling with SMOTE, model training, model
evaluation, prediction utilities, and a Streamlit user interface for interactive
fraud checks.

This project is designed as a practical fintech ML portfolio project. The code
is modular, beginner-friendly, and structured so the pipeline can be extended
toward real-time monitoring, explainable AI, API deployment, and cloud
deployment.

> Note: The current sample dataset is small. The generated metrics are useful
> for validating the pipeline, but they should not be treated as production
> model performance.

## Problem Statement

Financial fraud detection is a highly imbalanced classification problem. In
real transaction systems, fraudulent transactions are much rarer than legitimate
transactions, which makes traditional model evaluation difficult.

The goal of this project is to build a machine learning pipeline that can:

- Clean and prepare transaction data
- Analyze fraud patterns
- Handle class imbalance correctly
- Train and compare multiple ML models
- Evaluate models using fraud-specific metrics
- Predict whether new transactions are likely to be fraudulent

## Key Features

- Load raw transaction datasets from CSV
- Handle missing values and duplicate records
- Perform feature engineering and one-hot encoding
- Split data into train and test sets
- Run advanced EDA in a Jupyter notebook
- Apply SMOTE only to training data to avoid data leakage
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate models using precision, recall, F1-score, ROC-AUC, confusion matrix,
  ROC curves, and precision-recall curves
- Save trained models and metadata using joblib
- Predict single transactions and batch transactions
- Return fraud probability and confidence score
- Use a Streamlit dashboard with login and a fraud prediction form

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn, SMOTE |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |
| Model Persistence | joblib |
| Database | MySQL |

## Project Architecture

```text
fraud-detector/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|   `-- EDA.ipynb
|-- reports/
|-- src/
|   |-- preprocess.py
|   |-- balance.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- config.py
|   `-- logging_utils.py
|-- visualizations/
|-- requirements.txt
|-- README.md
`-- main.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detector.git
cd fraud-detector
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the raw transaction dataset here:

```text
data/raw/creditcard.csv
```

The project can automatically detect common target column names such as:

- `Class`
- `fraud_label`
- `isFraud`
- `target`
- `label`

For the included sample schema, the fraud label column is:

```text
fraud_label
```

where:

- `0` = legitimate transaction
- `1` = fraudulent transaction

## Run Preprocessing

Run Phase 1 preprocessing:

```bash
python main.py
```

This step:

- Loads the raw dataset
- Removes duplicates
- Handles missing values
- Encodes categorical variables
- Splits features and labels
- Saves train/test datasets

Processed files are saved to:

```text
data/processed/
|-- X_train.csv
|-- X_test.csv
|-- y_train.csv
|-- y_test.csv
`-- metadata.json
```

## Run Exploratory Data Analysis

Open the EDA notebook:

```text
notebooks/EDA.ipynb
```

The notebook includes:

- Fraud vs legitimate transaction analysis
- Class distribution visualization
- Transaction amount analysis
- Correlation heatmaps
- Feature signal exploration
- Outlier analysis
- Time-based transaction analysis

## Run Model Training

Run the full training pipeline:

```bash
python -m src.train
```

This trains and compares:

1. Logistic Regression
2. Random Forest
3. XGBoost

The best model is saved as:

```text
models/fraud_model.joblib
```

Individual model files are also saved:

```text
models/
|-- logistic_regression.joblib
|-- random_forest.joblib
|-- xgboost.joblib
`-- fraud_model.joblib
```

## Run the Streamlit Dashboard

Start the dashboard:

```bash
streamlit run app/streamlit_app.py
```

If port `8501` is already in use:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

Then open the local URL shown in the terminal, for example:

```text
http://localhost:8502
```

### Default Login

```text
User ID: admin
Password: admin123
```

The dashboard includes:

- Dataset overview
- Fraud distribution charts
- Preprocessing controls
- Model training status
- Manual transaction fraud prediction form

## Model Evaluation Metrics

The project evaluates models using fraud-focused metrics:

| Metric | Meaning |
| --- | --- |
| Precision | Of transactions predicted as fraud, how many were actually fraud |
| Recall | Of actual fraud transactions, how many the model detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to rank fraud higher than legitimate transactions |
| Confusion Matrix | Counts true positives, false positives, true negatives, and false negatives |
| Precision-Recall Curve | Shows precision and recall tradeoffs across thresholds |

## Why Accuracy Alone Is Misleading

Fraud datasets are usually highly imbalanced. If only 1% of transactions are
fraudulent, a model can reach 99% accuracy by predicting every transaction as
legitimate.

That model would look accurate but would detect no fraud.

For this reason, this project prioritizes:

- Recall, to catch more fraud
- Precision, to reduce false alarms
- F1-score, to balance precision and recall
- ROC-AUC and precision-recall curves, to evaluate probability ranking quality

## SMOTE and Data Leakage

SMOTE creates synthetic minority-class samples to help the model learn fraud
patterns. However, applying SMOTE before the train/test split can leak synthetic
information into the test set.

This project follows the correct approach:

1. Split the data into training and test sets
2. Apply SMOTE only to the training set
3. Keep the test set untouched
4. Evaluate on the original imbalanced test set

This gives a more realistic estimate of model behavior.

## Example Prediction Usage

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

result = predict_single_transaction(transaction)
print(result)
```

Example output:

```text
{
  'prediction': 0,
  'probability': 0.236667,
  'confidence_score': 0.763333,
  'label': 'Not Fraud'
}
```

## Output Artifacts

### Reports

```text
reports/
|-- model_comparison.csv
|-- training_summary.json
|-- logistic_regression_evaluation.json
|-- random_forest_evaluation.json
`-- xgboost_evaluation.json
```

### Visualizations

```text
visualizations/
|-- class_balance_before_after.png
|-- model_comparison.png
|-- logistic_regression_confusion_matrix.png
|-- random_forest_confusion_matrix.png
|-- xgboost_confusion_matrix.png
|-- *_roc_curve.png
|-- *_precision_recall_curve.png
|-- *_prediction_distribution.png
`-- *_feature_importance.png
```

## Current Sample Model Results

The current sample dataset is small, so these results should be interpreted as
pipeline validation rather than production performance.

```text
model,precision,recall,f1_score,roc_auc
random_forest,0.3333,0.3333,0.3333,0.6078
logistic_regression,0.1579,1.0000,0.2727,0.6078
xgboost,0.0000,0.0000,0.0000,0.6863
```

## Future Improvements

- Add SHAP or LIME for explainable fraud predictions
- Add FastAPI endpoint for real-time transaction scoring
- Add MySQL transaction logging and dashboard history
- Add Docker support
- Add model versioning and experiment tracking
- Add threshold tuning for business-specific fraud risk tolerance
- Add anomaly detection models
- Add cloud deployment
- Add real-time fraud monitoring alerts

## Author

Developed by Sahasrad S Nair.

