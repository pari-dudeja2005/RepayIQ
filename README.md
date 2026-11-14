# RepayIQ

### Intelligent Repayment Risk Analytics

RepayIQ is a machine learning-based repayment risk analysis system that estimates the probability of credit default using historical repayment behaviour and financial indicators.

The application provides account-level risk scores, early warning indicators, explainable predictions, and batch analysis through an interactive Streamlit dashboard.

## Key Features

- **Repayment Risk Prediction:** Estimates default probabilities using a trained Gradient Boosting model.
- **Risk Classification:** Categorizes accounts into low, medium, and high-risk segments.
- **Early Warning Indicators:** Identifies delayed repayments, low payment-to-bill ratios, and high credit utilization.
- **Explainable AI:** Uses SHAP to interpret individual predictions.
- **Risk Monitoring:** Provides filtering, prioritization, and account-level analysis.
- **Batch Analysis:** Supports CSV uploads, bulk predictions, and downloadable reports.
- **Interactive Dashboard:** Four-page Streamlit interface for repayment risk analysis.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Explainable AI | SHAP |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Model Serialization | Joblib |

## Dataset

**Source:** UCI Machine Learning Repository — Default of Credit Card Clients

Dataset link: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The original dataset contains 30,000 observations with demographic, credit limit, repayment status, billing, and payment information.

RepayIQ uses financial and repayment-related variables to predict default in the following month.

After preprocessing, the development dataset contains 29,268 records and 31 columns, including engineered features and the target variable.

### Feature Engineering

The pipeline generates additional repayment behaviour indicators, including:

- Late payment count
- Maximum repayment delay
- Recent repayment delay
- Average bill amount
- Average payment amount
- Payment-to-bill ratio
- Credit utilization
- Billing changes
- Payment volatility
- Months with recorded payments

## Machine Learning Pipeline

1. Load and preprocess the dataset.
2. Engineer repayment behaviour features.
3. Split the data into training, validation, and test sets.
4. Train Logistic Regression, Random Forest, and Gradient Boosting models.
5. Compare models using validation PR-AUC.
6. Optimize the decision threshold using validation data.
7. Evaluate the selected model on the test set.
8. Generate account-level predictions and risk indicators.
9. Provide SHAP explanations through the dashboard.

### Data Split

| Dataset | Records |
|---|---:|
| Training | 17,560 |
| Validation | 5,854 |
| Test | 5,854 |

## Model Performance

**Selected Model:** Gradient Boosting

**Decision Threshold:** 0.24

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 77.18% |
| Precision | 47.99% |
| Recall | 58.99% |
| F1 Score | 52.92% |
| ROC-AUC | 0.7756 |
| PR-AUC | 0.5363 |

The threshold was selected on the validation set to balance accuracy with a minimum recall target of 60%. The final test recall was 58.99%.

### Confusion Matrix

| | Predicted No Default | Predicted Default |
|---|---:|---:|
| Actual No Default | 3,767 | 814 |
| Actual Default | 522 | 751 |

## Dashboard

The Streamlit application contains four pages.

### 1. Overview

Displays portfolio-level risk metrics, risk distribution, default probability distribution, and priority accounts.

### 2. Risk Monitoring

Provides account-level risk scores, risk-category filtering, and CSV report exports.

### 3. Customer Analysis

Displays repayment history, warning indicators, and SHAP-based explanations for individual predictions.

### 4. Batch Analysis

Supports CSV imports, batch risk scoring, risk distribution analysis, and downloadable prediction results.

## Project Structure

```text
RepayIQ/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── repayiq_model.joblib
├── reports/
│   ├── demo_customers.csv
│   ├── model_comparison.csv
│   ├── model_metrics.json
│   ├── threshold_comparison.csv
│   └── validation_predictions.csv
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    ├── explain.py
    └── threshold_analysis.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RepayIQ.git
cd RepayIQ
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The trained model and demonstration data must be present in their configured locations.

## Model Training

To retrain the model using the prepared dataset:

```bash
python -m src.train
```

To review validation threshold performance:

```bash
python -m src.threshold_analysis
```

## Limitations

RepayIQ is a proof-of-concept developed using a public historical credit-card dataset. Its performance has not been validated on actual lending portfolios.

Risk scores are statistical estimates, not guarantees of default. The system is intended for demonstration and analysis rather than automated lending decisions.

## Future Enhancements

- Model monitoring and drift detection
- Portfolio-level risk trend analysis
- API-based scoring integration
- Model calibration and additional validation
- Secure integration with lending platforms
