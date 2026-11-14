
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    REPORT_DIR,
    RANDOM_STATE,
    TEST_SIZE
)


TARGET = "default_next_month"
MIN_RECALL = 0.60

# Candidate thresholds
THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2
)


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        )
    }


def analyze_thresholds(y_true, probabilities):

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            probabilities >= threshold,
            labels=[0, 1]
        ).ravel()

        rows.append({
            "Threshold": float(threshold),
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "False Positives": int(fp),
            "False Negatives": int(fn)
        })

    return pd.DataFrame(rows)


def choose_threshold(results):

    eligible = results[
        results["Recall"] >= MIN_RECALL
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No threshold meets the minimum "
            f"recall requirement of {MIN_RECALL:.0%}."
        )

    best = eligible.sort_values(
        ["Accuracy", "F1 Score", "Threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    return float(best["Threshold"])


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =============================================
    # LOAD DATA
    # =============================================

    df = pd.read_csv(PROCESSED_DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    features = X.columns.tolist()

    print("\nDataset shape:", df.shape)

    # =============================================
    # TRAIN / VALIDATION / TEST SPLIT
    # =============================================

    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    # With TEST_SIZE=0.20, this creates
    # 60% training, 20% validation, 20% test.
    validation_fraction = (
        TEST_SIZE / (1 - TEST_SIZE)
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=RANDOM_STATE,
            stratify=y_train_val
        )
    )

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Test records:", len(X_test))

    # =============================================
    # DEFINE MODELS
    # =============================================

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_STATE
            )
        )
    }

    # =============================================
    # MODEL COMPARISON ON VALIDATION SET
    # =============================================

    comparison_rows = []
    trained_models = {}

    print("\nMODEL COMPARISON\n")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        metrics = calculate_metrics(
            y_val,
            probabilities,
            threshold=0.50
        )

        comparison_rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"]
        })

        trained_models[name] = model

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "PR-AUC",
        ascending=False
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    comparison_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    # =============================================
    # SELECT BEST MODEL
    # =============================================

    best_model_name = comparison_df.iloc[0][
        "Model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    print(
        "\nBest model:",
        best_model_name
    )

    # =============================================
    # OPTIMIZE THRESHOLD ON VALIDATION DATA
    # =============================================

    val_probabilities = (
        best_model.predict_proba(X_val)[:, 1]
    )

    threshold_df = analyze_thresholds(
        y_val,
        val_probabilities
    )

    threshold_df.to_csv(
        REPORT_DIR / "threshold_comparison.csv",
        index=False
    )

    best_threshold = choose_threshold(
        threshold_df
    )

    print(
        "\nSelected validation threshold:",
        best_threshold
    )

    selected_row = threshold_df.loc[
        threshold_df["Threshold"] == best_threshold
    ].iloc[0]

    print("\nVALIDATION METRICS AT SELECTED THRESHOLD\n")

    print(selected_row.to_string())

    # =============================================
    # FINAL TEST EVALUATION
    # =============================================

    test_probabilities = (
        best_model.predict_proba(X_test)[:, 1]
    )

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )

    print("\nFINAL TEST METRICS\n")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # =============================================
    # SAVE TRAINED MODEL
    # =============================================

    model_bundle = {
        "model": best_model,
        "features": features,
        "threshold": best_threshold,
        "model_name": best_model_name
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    # =============================================
    # SAVE REPORTS
    # =============================================

    report = {
        "best_model": best_model_name,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "validation_min_recall": MIN_RECALL,
        "validation_metrics": {
            "accuracy": float(
                selected_row["Accuracy"]
            ),
            "precision": float(
                selected_row["Precision"]
            ),
            "recall": float(
                selected_row["Recall"]
            ),
            "f1_score": float(
                selected_row["F1 Score"]
            )
        }
    }

    with open(
        REPORT_DIR / "model_metrics.json",
        "w"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    # Save held-out demonstration customers.
    demo_customers = X_test.copy()

    demo_customers[TARGET] = (
        y_test.to_numpy()
    )

    demo_customers.to_csv(
        REPORT_DIR / "demo_customers.csv",
        index=False
    )

    # Save validation predictions for analysis.
    validation_predictions = pd.DataFrame({
        "actual_default": y_val.to_numpy(),
        "predicted_probability": (
            val_probabilities
        )
    })

    validation_predictions.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False
    )

    print("\nFiles saved successfully:")
    print("models/repayiq_model.joblib")
    print("reports/model_comparison.csv")
    print("reports/model_metrics.json")
    print("reports/threshold_comparison.csv")
    print("reports/demo_customers.csv")
    print("reports/validation_predictions.csv")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

