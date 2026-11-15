
import os
import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, MODEL_PATH
from src.feature_engineering import create_features
from src.advanced_features import add_advanced_features


FINAL_MODEL_PATH = (
    MODEL_DIR / "repayiq_catboost_final.joblib"
)


def load_model():
    choice = os.getenv(
        "REPAYIQ_MODEL", "original"
    ).lower()

    if choice == "catboost":
        path = FINAL_MODEL_PATH
    else:
        path = MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return joblib.load(path)


def prepare_model_features(df, bundle=None):
    if bundle is None:
        bundle = load_model()

    data = df.copy()

    raw_columns = (
        ["credit_limit"]
        + [
            f"repay_status_{i}"
            for i in range(1, 7)
        ]
        + [
            f"bill_{i}"
            for i in range(1, 7)
        ]
        + [
            f"payment_{i}"
            for i in range(1, 7)
        ]
    )

    missing_raw = [
        col for col in raw_columns
        if col not in data.columns
    ]

    if missing_raw:
        raise ValueError(
            "Missing raw columns: "
            + ", ".join(missing_raw)
        )

    for col in raw_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    if not np.isfinite(
        data[raw_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Raw data contains invalid numeric values."
        )

    if (data["credit_limit"] <= 0).any():
        raise ValueError(
            "Credit limit must be positive."
        )

    # Recompute original features consistently.
    data = create_features(data)

    if bundle.get("feature_set") == "advanced":
        data = add_advanced_features(data)

    features = bundle["features"]

    missing = [
        col for col in features
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    if not np.isfinite(
        data[features].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Engineered features contain invalid values."
        )

    return data


def assign_risk_level(probability, threshold):
    # Illustrative risk bands, not calibrated
    # lending policy thresholds.
    if probability >= max(0.50, threshold):
        return "High Risk"
    elif probability >= threshold:
        return "Medium Risk"
    return "Low Risk"


def generate_warning_indicators(row):
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
    bundle = load_model()

    data = prepare_model_features(
        df, bundle
    )

    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle["threshold"])

    probabilities = model.predict_proba(
        data[features]
    )[:, 1]

    results = data.copy()

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

    return results.sort_values(
        "default_probability",
        ascending=False
    )


if __name__ == "__main__":
    from config import REPORT_DIR

    demo = pd.read_csv(
        REPORT_DIR / "demo_customers.csv"
    )

    predictions = predict_customers(
        demo.head(10)
    )

    print(
        predictions[
            [
                "risk_percentage",
                "risk_level",
                "predicted_default"
            ]
        ].to_string(index=False)
    )

    print("\nPrediction test passed.")

