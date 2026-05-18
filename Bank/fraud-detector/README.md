# Bank Transaction Fraud Detection System

Production-oriented fraud detection project using Python, pandas, scikit-learn,
XGBoost, SMOTE, Streamlit, and MySQL.

## Phase 1: Preprocessing

Place the Kaggle fraud dataset CSV at:

```text
data/raw/creditcard.csv
```

The default Kaggle credit card fraud dataset uses `Class` as the target column:

- `0`: legitimate transaction
- `1`: fraudulent transaction

Run preprocessing:

```bash
python main.py
```

Run the Streamlit interface:

```bash
streamlit run app/streamlit_app.py
```

If port `8501` is already being used by another Streamlit app:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

Default dashboard login:

```text
User ID: admin
Password: admin123
```

The dashboard includes a **Predict Fraud** tab where you can enter credit-card
transaction details and check whether the saved model predicts fraud.

Or pass a custom dataset path and target column:

```bash
python main.py --input-path data/raw/your_dataset.csv --target-column Class
```

Processed outputs are saved to:

```text
data/processed/
|-- X_train.csv
|-- X_test.csv
|-- y_train.csv
|-- y_test.csv
`-- metadata.json
```

## What preprocessing does

1. Loads the raw CSV.
2. Detects or validates the fraud target column.
3. Prints dataset statistics and target class distribution.
4. Removes duplicate rows.
5. Fills missing numeric values with the median.
6. Fills missing categorical values with the mode.
7. One-hot encodes categorical feature columns.
8. Splits features and labels into stratified train/test files.
9. Saves processed data and metadata for the training phase.

## Next Phases

- Exploratory data analysis in `notebooks/`.
- SMOTE balancing on the training set only.
- Model training for Logistic Regression, Random Forest, and XGBoost.
- Evaluation with precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Model persistence with joblib.
- Streamlit fraud prediction dashboard.
