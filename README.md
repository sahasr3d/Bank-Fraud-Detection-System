# Bank Transaction Fraud Detection System

A production-oriented fraud detection system designed to identify suspicious financial transactions using machine learning and real-world banking datasets. The project is built using Python, pandas, scikit-learn, XGBoost, SMOTE, Streamlit, and MySQL, with a focus on scalable preprocessing pipelines, model evaluation, and interactive fraud analysis.

The system is designed to simulate practical fraud detection workflows used in modern financial systems by combining data preprocessing, imbalance handling, machine learning classification, and real-time prediction interfaces within a modular architecture.

The project aims to support experimentation with multiple fraud detection models while providing an expandable foundation for future enterprise-grade fraud monitoring systems and intelligent financial analytics.

> ⚠️ **Project Status:** The Bank Transaction Fraud Detection System is currently under active development and is not yet in its final production stage. Core preprocessing pipelines, dataset handling, dashboard foundations, and initial fraud prediction workflows are implemented, while advanced model optimization, real-time transaction monitoring, database integrations, and intelligent fraud analysis features are still being expanded and refined.

## Technologies Used

* Python
* pandas
* scikit-learn
* XGBoost
* SMOTE
* Streamlit
* MySQL

## Phase 1: Preprocessing

Place the Kaggle fraud dataset CSV at:

```text id="ztbx6f"
data/raw/creditcard.csv
```

The default Kaggle Credit Card Fraud dataset uses `Class` as the target column:

* `0` → Legitimate transaction
* `1` → Fraudulent transaction

## Run Preprocessing

```bash id="q8rjlwm"
python main.py
```

## Run the Streamlit Dashboard

```bash id="8tr4fo"
streamlit run app/streamlit_app.py
```

If port `8501` is already in use:

```bash id="a9x7uv"
streamlit run app/streamlit_app.py --server.port 8502
```

## Default Dashboard Login

```text id="t4gexs"
User ID: admin
Password: admin123
```

The dashboard includes a **Predict Fraud** section where users can enter transaction details and check whether the trained model predicts fraudulent activity.

You can also provide a custom dataset path and target column:

```bash id="m7rq8l"
python main.py --input-path data/raw/your_dataset.csv --target-column Class
```

## Processed Outputs

Processed datasets and metadata are saved to:

```text id="4kpvru"
data/processed/
|-- X_train.csv
|-- X_test.csv
|-- y_train.csv
|-- y_test.csv
`-- metadata.json
```

## Preprocessing Pipeline

The preprocessing phase currently performs:

* Loading the raw transaction dataset
* Detecting and validating the fraud target column
* Printing dataset statistics and fraud distribution
* Removing duplicate records
* Filling missing numeric values using median imputation
* Filling missing categorical values using mode imputation
* One-hot encoding categorical feature columns
* Stratified train-test dataset splitting
* Saving processed datasets and metadata for training workflows

## Planned Next Phases

1. Exploratory Data Analysis (EDA) using Jupyter notebooks
2. SMOTE balancing applied only on training data
3. Training models using:

   * Logistic Regression
   * Random Forest
   * XGBoost
4. Model evaluation using:

   * Precision
   * Recall
   * F1-Score
   * ROC-AUC
   * Confusion Matrix
5. Model persistence using joblib
6. Advanced Streamlit fraud monitoring dashboard
7. Real-time fraud scoring and transaction analysis
8. Database integration and intelligent fraud reporting
