
import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH


def load_model():
    """Load the trained RepayIQ model."""
    return joblib.load(MODEL_PATH)


def assign_risk_level(probability, threshold):
    """
    Demo risk bands based on the validation-selected
    decision threshold.

    These bands are illustrative and are not
    production-calibrated business thresholds.
    """
    if probability >= max(0.50, threshold):
        return "High Risk"
    elif probability >= threshold:
        return "Medium Risk"
    else:
        return "Low Risk"


def generate_warning_indicators(row):
    """
    Rule-based indicators based on repayment behaviour.
    These are NOT SHAP explanations or causal claims.
    """
    warnings = []

    if row["late_payment_count"] >= 3:
        warnings.append(
            "Multiple months with delayed repayments"
        )

    if row["recent_delay"] > 0:
        warnings.append(
            "Recent repayment delay recorded"
        )

    if row["payment_to_bill_ratio"] < 0.10:
        warnings.append(
            "Low payments relative to billed amounts"
        )

    if row["credit_utilization"] > 0.80:
        warnings.append(
            "High recent bill relative to credit limit"
        )

    if row["months_with_payment"] <= 2:
        warnings.append(
            "Few months with positive payments"
        )

    if not warnings:
        warnings.append(
            "No predefined repayment warning indicators triggered"
        )

    return warnings


def predict_customers(df):
    """
    Predict next-month default risk for a DataFrame
    containing engineered features.
    """

    bundle = load_model()

    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle["threshold"]

    missing_features = [
        col for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[features].copy()

    probabilities = model.predict_proba(X)[:, 1]

    results = df.copy()

    results["default_probability"] = probabilities

    results["risk_percentage"] = (
        probabilities * 100
    ).round(2)

    results["predicted_default"] = (
        probabilities >= threshold
    ).astype(int)

    results["risk_level"] = [
        assign_risk_level(p, threshold)
        for p in probabilities
    ]

    results["warning_indicators"] = [
        generate_warning_indicators(row)
        for _, row in results.iterrows()
    ]

    results = results.sort_values(
        "default_probability",
        ascending=False
    )

    return results


if __name__ == "__main__":

    from config import REPORT_DIR

    print("\nLoading demo customers...")

    df = pd.read_csv(
        REPORT_DIR / "demo_customers.csv"
    )

    # Select 10 example customers
    sample = df.head(10).copy()

    predictions = predict_customers(sample)

    print("\nRepayIQ Prediction Results:\n")

    print(
        predictions[
            [
                "default_probability",
                "risk_percentage",
                "risk_level",
                "predicted_default"
            ]
        ].to_string(index=False)
    )

    print("\nExample Warning Indicators:")

    first_customer = predictions.iloc[0]

    for warning in first_customer["warning_indicators"]:
        print("-", warning)

    print("\nPrediction engine working successfully!")
