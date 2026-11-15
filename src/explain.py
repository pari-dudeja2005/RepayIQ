
import numpy as np
import pandas as pd
import shap

from catboost import Pool

from config import REPORT_DIR
from src.predict import (
    load_model,
    prepare_model_features,
)


def explain_customer(customer_df):
    bundle = load_model()

    model = bundle["model"]
    features = bundle["features"]

    prepared = prepare_model_features(
        customer_df, bundle
    )

    X = prepared[features].head(1).copy()

    if bundle.get("model_name") == "Advanced CatBoost":
        pool = Pool(X)

        # Native CatBoost SHAP values are expressed
        # on the model's raw-score (log-odds) scale.
        values = model.get_feature_importance(
            pool,
            type="ShapValues"
        )

        contributions_array = values[0, :-1]

    else:
        background_df = pd.read_csv(
            REPORT_DIR / "demo_customers.csv"
        )

        background = prepare_model_features(
            background_df, bundle
        )[features]

        background = background.sample(
            n=min(50, len(background)),
            random_state=42
        )

        def predict_default(data):
            frame = pd.DataFrame(
                data,
                columns=features
            )

            return model.predict_proba(
                frame
            )[:, 1]

        explainer = shap.Explainer(
            predict_default,
            background,
            algorithm="permutation"
        )

        values = explainer(
            X,
            max_evals=2 * len(features) + 1
        )

        contributions_array = values.values[0]

    contributions = pd.DataFrame({
        "Feature": features,
        "SHAP Value": contributions_array
    })

    contributions["Absolute Impact"] = (
        contributions["SHAP Value"].abs()
    )

    return contributions.sort_values(
        "Absolute Impact",
        ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    demo = pd.read_csv(
        REPORT_DIR / "demo_customers.csv"
    )

    result = explain_customer(
        demo.head(1)
    )

    print(result.head(10))
    print("\nSHAP explanation test passed.")

