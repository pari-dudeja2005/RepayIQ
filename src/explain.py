
import numpy as np
import pandas as pd
import shap

from src.predict import load_model


def explain_customer(customer_df):
    """
    Calculate SHAP values for one customer.

    Uses a model-agnostic explainer so that it works
    with our sklearn Gradient Boosting pipeline.
    """

    bundle = load_model()

    model = bundle["model"]
    features = bundle["features"]

    X = customer_df[features].copy()

    # Background samples represent typical customers.
    background = pd.read_csv(
        "reports/demo_customers.csv"
    )[features].sample(
        n=50,
        random_state=42
    )

    def predict_default(data):
        data_df = pd.DataFrame(
            data,
            columns=features
        )
        return model.predict_proba(data_df)[:, 1]

    explainer = shap.Explainer(
        predict_default,
        background,
        algorithm="permutation"
    )

    shap_values = explainer(
        X,
        max_evals=2 * len(features) + 1
    )

    contributions = pd.DataFrame({
        "Feature": features,
        "SHAP Value": shap_values.values[0]
    })

    contributions["Absolute Impact"] = (
        contributions["SHAP Value"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return contributions


if __name__ == "__main__":

    from config import REPORT_DIR

    df = pd.read_csv(
        REPORT_DIR / "demo_customers.csv"
    )

    customer = df.iloc[[0]]

    result = explain_customer(customer)

    print("\nTop Model Contributions:\n")
    print(result.head(10).to_string(index=False))
