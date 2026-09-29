Absolutely. Below is a **complete, polished GitHub `README.md`** for your UPI FraudGuard AI project, including the full pipeline, installation, execution steps, architecture, results, project structure, and limitations.

````markdown
# 🛡️ UPI FraudGuard AI

### AI-Powered UPI Transaction Fraud Detection & Risk Analytics

UPI FraudGuard AI is a Machine Learning-based fraud detection system designed to identify anomalous and potentially fraudulent Unified Payments Interface (UPI) transactions. The system analyzes transaction behavior, transaction velocity, amount deviations, device switching, transaction timing, bank combinations, merchant categories, and regional patterns to detect suspicious activity.

The project combines **behavioral feature engineering, imbalanced-learning techniques, multiple ML models, threshold optimization, and an interactive Streamlit dashboard** to provide transaction-level fraud detection along with regional and merchant-category risk analysis.

---

## 🚀 Key Features

- 🔍 UPI transaction fraud detection
- 🤖 Machine Learning-based classification
- ⚡ Behavioral anomaly detection
- 📊 Transaction velocity analysis
- 💰 Transaction amount deviation detection
- 🌙 Odd-hour transaction detection
- 📱 Device-switching detection
- 🏦 Cross-bank transaction analysis
- 🌍 State-wise fraud-risk aggregation
- 🏪 Merchant-category risk analysis
- 📈 Model performance comparison
- 🎯 Fraud probability scoring
- ⚙️ Tuned classification threshold
- 📊 Interactive Streamlit dashboard
- 🧪 Synthetic dataset generation
- 🔄 Time-aware train/test splitting
- ⚖️ SMOTE for class imbalance handling

---

# ⚠️ About the Dataset

Real transaction-level UPI fraud data is not publicly available because banks, payment providers, NPCI, and financial institutions treat transaction and fraud records as confidential.

Therefore, this project uses a **synthetic dataset** generated using:

```text
src/generate_data.py
````

The dataset is designed to mimic realistic UPI transaction behavior, including:

* Indian states
* Indian banks
* Merchant categories
* Transaction amounts
* Transaction timestamps
* Sender behavior
* Device information
* Fraud patterns

The synthetic dataset contains approximately **120,000 transactions**, with around **0.6% fraudulent transactions**.

### Simulated Fraud Typologies

The dataset contains four primary fraud patterns:

1. **Odd-hour high-value transactions**
2. **Device switching**
3. **Transaction-velocity bursts**
4. **Unfamiliar bank combinations**

Noise is also introduced so that fraud cannot be detected using a single obvious feature.

> **Important:** The dataset is synthetic and should not be presented as real banking or NPCI transaction data.

---

# 📊 Model Results

| Model               |  Accuracy | Precision |    Recall |  F1 Score |   ROC-AUC |    PR-AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression |     80.0% |      2.1% |     66.7% |      4.2% |     0.795 |     0.333 |
| Decision Tree       |     88.3% |      3.5% |     64.7% |      6.7% |     0.798 |     0.468 |
| Random Forest       |     97.1% |     10.3% |     44.9% |     16.8% |     0.841 |     0.378 |
| **XGBoost**         | **99.6%** | **85.7%** | **50.0%** | **63.2%** | **0.881** | **0.560** |

### ⚠️ Why Accuracy Is Not Enough

Fraud detection is a highly imbalanced classification problem.

Approximately **99.4% of transactions are legitimate**, meaning a model that predicts every transaction as legitimate could still achieve approximately 99.4% accuracy.

Therefore, this project focuses on:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

### XGBoost Performance

The tuned XGBoost model achieved:

* **Accuracy:** 99.6%
* **Precision:** 85.7%
* **Recall:** 50.0%
* **F1 Score:** 63.2%
* **ROC-AUC:** 0.881
* **PR-AUC:** 0.560

The tuned threshold prioritizes a stronger precision/recall balance than simply using the default 0.5 probability threshold.

---

# 🧠 Machine Learning Pipeline

```text
Synthetic UPI Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Time-Aware Train/Test Split
        ↓
SMOTE on Training Data
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Threshold Optimization
        ↓
XGBoost Fraud Prediction
        ↓
Risk Aggregation
        ↓
Streamlit Dashboard
```

---

# 🔬 Feature Engineering

The project derives behavioral features from raw transaction information.

### Transaction Velocity

Measures the number of transactions performed by a sender within a specific time period.

```text
Transaction Velocity =
Number of transactions by sender within time window
```

A sudden burst of transactions can indicate suspicious activity.

### Amount Deviation

Measures how different the current transaction amount is from the sender's historical behavior.

```text
Amount Deviation =
Current Transaction Amount - Historical Average Amount
```

Large deviations can indicate abnormal transactions.

### Odd-Hour Flag

Identifies transactions performed during unusual hours.

### Device Switching

Detects when a sender suddenly uses a different device from their previous transactions.

### Cross-Bank Transactions

Identifies transactions involving unfamiliar bank combinations.

---

# 🤖 Machine Learning Models

The project compares multiple classification algorithms.

## 1. Logistic Regression

Used as the baseline model.

It provides a simple and interpretable benchmark for fraud classification.

## 2. Decision Tree

A tree-based model capable of capturing non-linear relationships between transaction features.

## 3. Random Forest

An ensemble of decision trees that improves robustness and reduces overfitting.

## 4. XGBoost

The primary model used for fraud detection.

XGBoost is a gradient-boosting algorithm that performs well on structured/tabular datasets and can capture complex relationships between behavioral features.

---

# ⚖️ Handling Class Imbalance

Fraud transactions represent only a small fraction of the dataset.

To address this imbalance, the project uses:

### SMOTE

**Synthetic Minority Over-sampling Technique**

SMOTE generates synthetic examples of the minority class to provide the model with more fraud examples during training.

SMOTE is applied **only to the training dataset**.

The test dataset remains unchanged to provide a more realistic evaluation.

---

# ⏱️ Time-Aware Train/Test Split

Instead of randomly splitting the dataset, transactions are divided chronologically.

```text
First 80%
    ↓
Training Data

Last 20%
    ↓
Testing Data
```

This better represents a real-world fraud detection scenario where historical transactions are used to predict future transactions.

---

# 🎯 Threshold Optimization

The default classification threshold of:

```text
0.50
```

is not always optimal for highly imbalanced fraud detection.

The project evaluates different probability thresholds and selects a threshold based on the precision-recall trade-off.

This allows the system to control the balance between:

* False positives
* False negatives
* Precision
* Recall

The selected threshold is stored in:

```text
models/best_threshold.json
```

---

# 🌍 Regional & Merchant Risk Analysis

One of the key features of this project is aggregation of fraud-risk information.

Instead of only detecting individual suspicious transactions, the system aggregates flagged transactions by:

### Indian State

Examples:

```text
Telangana
Maharashtra
Karnataka
Tamil Nadu
Delhi
```

### Merchant Category

Examples:

```text
Food
Shopping
Travel
Utilities
Entertainment
Healthcare
```

This allows the dashboard to highlight areas and merchant categories that may require additional monitoring.

> These are model-generated risk indicators from synthetic data and should not be interpreted as evidence that any real Indian state, bank, or merchant category has higher fraud.

---

# 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Run:

```bash
streamlit run src/app.py
```

The dashboard contains three primary sections.

### 1. Model Performance

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix
* Model comparison

### 2. Regional & Category Risk

Displays:

* State-wise flagged transaction rate
* Merchant-category flagged transaction rate
* Risk aggregation
* Visual charts

### 3. Transaction Scoring

Users can enter transaction information and receive:

```text
Fraud Probability
        ↓
Risk Classification
        ↓
Potentially Suspicious / Legitimate
```

---

# 📁 Project Structure

```text
upi_fraud_project/
│
├── data/
│   ├── upi_transactions.csv
│   └── upi_transactions_features.csv
│
├── models/
│   ├── xgb_fraud_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── best_threshold.json
│
├── outputs/
│   ├── model_metrics.json
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── regional_risk.csv
│   └── category_risk.csv
│
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── app.py
│
├── README.md
└── requirements.txt
```

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn
* SMOTE

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Dashboard

* Streamlit

### Model Persistence

* Joblib

---

# 💻 System Requirements

Recommended:

```text
Python 3.9+
RAM: 8 GB+
Storage: 1 GB+
Operating System: Windows / Linux / macOS
```

No GPU is required.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Example:

```bash
git clone https://github.com/yourusername/upi-fraudguard-ai.git
```

---

## 2. Navigate to the Project

```bash
cd upi-fraudguard-ai
```

---

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

Install all required Python packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn streamlit plotly joblib
```

Alternatively, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

The project must be executed in the following order.

## Step 1 — Generate Dataset

```bash
python src/generate_data.py
```

This generates the synthetic UPI transaction dataset.

Output:

```text
data/upi_transactions.csv
```

---

## Step 2 — Feature Engineering

```bash
python src/feature_engineering.py
```

This processes the raw dataset and creates behavioral features.

Output:

```text
data/upi_transactions_features.csv
```

---

## Step 3 — Train the Models

```bash
python src/train_model.py
```

This step:

* Loads the engineered dataset
* Performs the time-aware train/test split
* Applies SMOTE to training data
* Trains multiple ML models
* Evaluates model performance
* Tunes the classification threshold
* Saves the XGBoost model
* Generates evaluation metrics
* Generates charts
* Creates regional risk analysis
* Creates merchant-category risk analysis

Generated files include:

```text
models/xgb_fraud_model.pkl
models/scaler.pkl
models/label_encoders.pkl
models/best_threshold.json
```

and:

```text
outputs/model_metrics.json
outputs/model_comparison.csv
outputs/model_comparison.png
outputs/confusion_matrix.png
outputs/feature_importance.png
outputs/regional_risk.csv
outputs/category_risk.csv
```

---

# 🚀 Step 4 — Launch the Dashboard

Run:

```bash
streamlit run src/app.py
```

After starting Streamlit, open:

```text
http://localhost:8501
```

in your browser.

---

# 🔄 Complete One-Command Workflow

After cloning and installing dependencies, the complete workflow is:

```bash
python src/generate_data.py
python src/feature_engineering.py
python src/train_model.py
streamlit run src/app.py
```

Run the commands in this exact order.

---

# 📈 Generated Outputs

The training pipeline generates several analytical outputs.

### Model Comparison

```text
outputs/model_comparison.csv
outputs/model_comparison.png
```

Compares the performance of all trained models.

### Confusion Matrix

```text
outputs/confusion_matrix.png
```

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Feature Importance

```text
outputs/feature_importance.png
```

Shows which behavioral features contribute most to the XGBoost model.

### Regional Risk

```text
outputs/regional_risk.csv
```

Contains state-level flagged transaction rates.

### Category Risk

```text
outputs/category_risk.csv
```

Contains merchant-category flagged transaction rates.

---

# 🔐 Fraud Detection Logic

The system considers multiple behavioral signals rather than relying on transaction amount alone.

```text
Transaction
     │
     ├── Amount Analysis
     │
     ├── Transaction Velocity
     │
     ├── Time Analysis
     │
     ├── Device Behavior
     │
     ├── Bank Combination
     │
     └── Historical Sender Behavior
             │
             ↓
       Feature Engineering
             │
             ↓
          XGBoost
             │
             ↓
       Fraud Probability
             │
             ↓
      Tuned Threshold
             │
       ┌─────┴─────┐
       ↓           ↓
   Legitimate   Suspicious
```

---

# ⚠️ Limitations

This project is an academic/prototype implementation and has several limitations.

### Synthetic Data

The dataset does not represent actual bank or NPCI transactions.

### Limited Features

Real-world fraud detection systems can use additional signals such as:

* Device fingerprints
* IP addresses
* GPS/geolocation
* Account age
* SIM information
* Network information
* Historical account behavior
* Beneficiary history

These are not included because real transaction-level datasets are unavailable.

### Recall

The current model detects approximately 50% of fraud cases at the selected threshold.

Increasing recall may increase false positives.

Real fraud detection systems typically use multiple risk tiers and human review rather than relying on a single binary prediction.

---

# 🔮 Future Improvements

Future versions could include:

* Device fingerprinting
* IP and geolocation analysis
* Account-age features
* Beneficiary relationship analysis
* Real-time transaction streaming
* Autoencoder-based anomaly detection
* Graph-based fraud detection
* XGBoost + neural network ensemble
* Real-time fraud alerts
* Two-stage fraud review system
* Explainable AI using SHAP
* Model monitoring and drift detection
* Production API using FastAPI
* Cloud deployment
* Database integration

---

# 🧪 Future Two-Tier Detection System

A production-oriented implementation could use two risk levels:

```text
Transaction
     ↓
Risk Model
     ↓
 ┌───────────────┐
 │ Fraud Score   │
 └───────┬───────┘
         │
    ┌────┴────┐
    ↓         ↓
High Risk   Medium Risk
    ↓         ↓
Auto Action  Human Review
```

This approach can help balance fraud detection with false-positive control.

---

# 🎯 Project Objective

The primary objective of UPI FraudGuard AI is to demonstrate how Machine Learning and behavioral analytics can be applied to digital payment fraud detection.

The project focuses not only on identifying suspicious transactions but also on understanding **where and in which merchant categories suspicious activity is concentrated**.

---

# 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Data generation
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Imbalanced classification
* SMOTE
* Time-series-aware splitting
* Logistic Regression
* Decision Trees
* Random Forest
* XGBoost
* Hyperparameter/threshold optimization
* Precision-recall analysis
* ROC-AUC
* PR-AUC
* Model persistence
* Data visualization
* Streamlit dashboard development
* Risk aggregation

---

# ⚖️ Disclaimer

This project is created for **educational, research, and demonstration purposes**.

The dataset is synthetic and does not contain real UPI, bank, NPCI, customer, or payment information.

The model should not be used to make real financial decisions or automatically block transactions without proper validation, monitoring, security testing, regulatory compliance, and human oversight.

Regional and merchant-category risk results are generated from synthetic data and should not be interpreted as real-world fraud statistics.

---

# 👨‍💻 Author

**Your Name**

Machine Learning | Data Science | AI

---

## ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub.

```text
UPI FraudGuard AI
Machine Learning + Behavioral Analytics + Fraud Risk Detection
```

```
```
