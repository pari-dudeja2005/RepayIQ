
import json
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from config import (
    PROCESSED_DATA_PATH,
    MODEL_DIR,
    REPORT_DIR,
    RANDOM_STATE,
)
from src.advanced_features import add_advanced_features


TARGET = "default_next_month"
OUTPUT_PATH = MODEL_DIR / "repayiq_catboost_final.joblib"


def choose_threshold(y_true, probabilities):
    rows = []

    for threshold in np.round(
        np.arange(0.05, 0.951, 0.01), 2
    ):
        predicted = (
            probabilities >= threshold
        ).astype(int)

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(
                y_true, predicted
            ),
            "precision": precision_score(
                y_true, predicted, zero_division=0
            ),
            "recall": recall_score(
                y_true, predicted, zero_division=0
            ),
            "f1": f1_score(
                y_true, predicted, zero_division=0
            ),
        })

    table = pd.DataFrame(rows)

    eligible = table[
        table["recall"] >= 0.60
    ]

    if eligible.empty:
        eligible = table.sort_values(
            ["recall", "f1"],
            ascending=False
        )
    else:
        eligible = eligible.sort_values(
            ["f1", "precision"],
            ascending=False
        )

    return (
        float(eligible.iloc[0]["threshold"]),
        table
    )


def main():
    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )
    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = add_advanced_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    indices = np.arange(len(df))

    development_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=2026
    )

    train_idx, validation_idx = train_test_split(
        development_idx,
        test_size=0.25,
        stratify=y.iloc[development_idx],
        random_state=2026
    )

    estimator = CatBoostClassifier(
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE,
        thread_count=2,
        allow_writing_files=False
    )

    parameters = {
        "iterations": [200, 350, 500],
        "depth": [4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.08],
        "l2_leaf_reg": [3, 5, 7],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=2026
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=2026,
        n_jobs=1,
        error_score="raise"
    )

    print("Training final CatBoost candidate...")

    search.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    model = search.best_estimator_

    validation_probabilities = (
        model.predict_proba(
            X.iloc[validation_idx]
        )[:, 1]
    )

    threshold, threshold_table = choose_threshold(
        y.iloc[validation_idx],
        validation_probabilities
    )

    threshold_table.to_csv(
        REPORT_DIR / "final_catboost_thresholds.csv",
        index=False
    )

    bundle = {
        "model": model,
        "features": X.columns.tolist(),
        "threshold": threshold,
        "model_name": "Advanced CatBoost",
        "feature_set": "advanced",
        "feature_engineering_version": "advanced_v2",
        "training_rows": len(train_idx),
        "validation_rows": len(validation_idx),
        "best_params": search.best_params_,
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print("\nMODEL SAVED")
    print("Path:", OUTPUT_PATH)
    print("Features:", len(bundle["features"]))
    print("Threshold:", threshold)
    print("Best parameters:", search.best_params_)
    print("CV PR-AUC:", round(search.best_score_, 4))
    print(
        "Validation PR-AUC:",
        round(
            average_precision_score(
                y.iloc[validation_idx],
                validation_probabilities
            ),
            4
        )
    )

    print(
        "\nOriginal production model remains unchanged."
    )


if __name__ == "__main__":
    main()
