# 💳 Bank Transaction Fraud Detection System

Production-oriented Machine Learning project for detecting fraudulent banking and credit-card transactions using advanced classification models, imbalance handling techniques, and real-time analytics dashboards.

Built using Python, pandas, scikit-learn, XGBoost, SMOTE, Streamlit, and MySQL, this system analyzes transaction behavior to identify suspicious activity with high precision and recall.

---

# 🚀 Features

## 📊 Data Processing & Analysis
- Automated preprocessing pipeline
- Missing value handling
- Duplicate transaction removal
- Feature engineering
- One-hot encoding
- Dataset statistics generation
- Fraud distribution analysis

## 🤖 Machine Learning
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Model comparison framework

## ⚖️ Imbalanced Dataset Handling
- SMOTE oversampling
- Stratified train-test splitting
- Class distribution monitoring
- Precision-focused optimization

## 📈 Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Reports

## 🌐 Dashboard & Deployment
- Interactive Streamlit dashboard
- Real-time fraud prediction
- Transaction analysis interface
- Secure login system
- Custom dataset support

---

# 🧠 Problem Statement

Financial fraud is one of the largest challenges faced by digital banking platforms today.

Fraudulent transactions are extremely rare compared to legitimate transactions, making fraud detection a highly imbalanced classification problem.

The goal of this project is to build an intelligent system capable of:
- learning transaction patterns
- identifying anomalies
- predicting fraudulent activities accurately

---

# 🛠️ Tech Stack

## Languages
- Python
- SQL

## Libraries & Frameworks
- pandas
- numpy
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- matplotlib
- seaborn
- Streamlit
- joblib

## Database
- MySQL

---

# 📂 Project Structure

fraud-detector/
│
├── app/
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── main.py

---

# ⚙️ Installation Guide

## Clone Repository

git clone https://github.com/your-username/fraud-detector.git

## Create Virtual Environment

python -m venv venv

## Install Dependencies

pip install -r requirements.txt

---

# ▶️ Running the Project

## Run Preprocessing

python main.py

## Run Streamlit Dashboard

streamlit run app/streamlit_app.py

---

# ⚠️ Why Accuracy Alone Is Misleading

Fraud detection datasets are highly imbalanced.

A model predicting every transaction as legitimate may still achieve very high accuracy.

Therefore, this project prioritizes:
- Recall
- Precision
- F1-score
- ROC-AUC

---

# 📈 Model Evaluation

Metrics used:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

# 📌 Future Improvements

- Deep learning fraud detection
- Graph-based anomaly detection
- Docker deployment
- Cloud deployment
- Real-time fraud monitoring

---

# 👨‍💻 Author

Developed by Sahasrad S Nair
