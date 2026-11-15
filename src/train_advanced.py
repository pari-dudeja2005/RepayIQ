
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    PROCESSED_DATA_PATH,
    MODEL_DIR,
    REPORT_DIR,
    RANDOM_STATE,
)


TARGET = "default_next_month"
SEED = RANDOM_STATE
MIN_RECALL = 0.60
MIN_ACCURACY_RECALL = 0.30

THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01), 2
)


# =====================================================
# 1. FEATURE ENGINEERING
# =====================================================

def add_advanced_features(df):
    data = df.copy()

    status_cols = [
        f"repay_status_{i}" for i in range(1, 7)
    ]
    bill_cols = [
        f"bill_{i}" for i in range(1, 7)
    ]
    payment_cols = [
        f"payment_{i}" for i in range(1, 7)
    ]

    required = (
        status_cols + bill_cols + payment_cols
        + ["credit_limit"]
    )

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    statuses = data[status_cols].to_numpy(
        dtype=float
    )
    bills = data[bill_cols].to_numpy(
        dtype=float
    )
    payments = data[payment_cols].to_numpy(
        dtype=float
    )

    safe_limit = np.maximum(
        data["credit_limit"].to_numpy(
            dtype=float
        ),
        1.0
    )

    positive_bills = np.maximum(bills, 0)

    # Repayment behavior
    late = statuses > 0

    consecutive = np.zeros(
        len(data), dtype=int
    )
    active = np.ones(
        len(data), dtype=bool
    )

    for month in range(6):
        active &= late[:, month]
        consecutive += active.astype(int)

    data["consecutive_late_payments"] = consecutive

    data["late_payment_count_3m"] = (
        late[:, :3].sum(axis=1)
    )

    data["late_payment_count_6m"] = (
        late.sum(axis=1)
    )

    data["max_delinquency_severity"] = (
        statuses.max(axis=1)
    )

    data["recent_delinquency_severity"] = (
        np.maximum(statuses[:, :3], 0)
        .mean(axis=1)
    )

    data["repayment_status_trend"] = (
        statuses[:, 0] - statuses[:, 5]
    )

    # Outstanding balance
    data["total_outstanding_balance"] = (
        positive_bills.sum(axis=1)
    )

    data["average_credit_utilization"] = (
        positive_bills.mean(axis=1)
        / safe_limit
    )

    data["latest_credit_utilization"] = (
        positive_bills[:, 0] / safe_limit
    )

    data["bill_balance_trend"] = (
        bills[:, 0] - bills[:, 5]
    ) / (np.abs(bills[:, 5]) + 1.0)

    data["bill_volatility"] = (
        bills.std(axis=1)
        / (np.abs(bills.mean(axis=1)) + 1.0)
    )

    # Payment patterns
    payment_mean = payments.mean(axis=1)

    data["repayment_consistency"] = (
        payments.std(axis=1)
        / (np.abs(payment_mean) + 1.0)
    )

    data["recent_payment_trend"] = (
        payments[:, 0] - payments[:, 5]
    ) / (np.abs(payments[:, 5]) + 1.0)

    data["payment_to_bill_ratio"] = (
        payments.sum(axis=1)
        / (positive_bills.sum(axis=1) + 1.0)
    )

    data["latest_payment_to_bill_ratio"] = (
        payments[:, 0]
        / (positive_bills[:, 0] + 1.0)
    )

    data["recent_payment_ratio"] = (
        payments[:, :3].sum(axis=1)
        / (
            positive_bills[:, :3].sum(axis=1)
            + 1.0
        )
    )

    data["zero_payment_months"] = (
        (payments <= 0).sum(axis=1)
    )

    # Replace invalid values defensively
    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    if data.isna().any().any():
        raise ValueError(
            "Missing or infinite values detected. "
            "Check preprocessing before training."
        )

    return data


# =====================================================
# 2. EVALUATION
# =====================================================

def evaluate(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true, probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
    }


def analyze_thresholds(y_true, probabilities):
    rows = []

    for threshold in THRESHOLDS:
        scores = evaluate(
            y_true,
            probabilities,
            threshold
        )

        rows.append({
            "threshold": float(threshold),
            **scores,
        })

    table = pd.DataFrame(rows)

    # Scenario A:
    # Maximize F1 while maintaining recall >= 60%
    eligible_risk = table[
        table["recall"] >= MIN_RECALL
    ]

    if eligible_risk.empty:
        risk = table.sort_values(
            ["recall", "f1_score"],
            ascending=False
        ).iloc[0]
    else:
        risk = eligible_risk.sort_values(
            ["f1_score", "precision"],
            ascending=False
        ).iloc[0]

    # Scenario B:
    # Maximize accuracy subject to recall >= 30%
    eligible_accuracy = table[
        table["recall"] >= MIN_ACCURACY_RECALL
    ]

    if eligible_accuracy.empty:
        accuracy = table.sort_values(
            ["accuracy", "recall"],
            ascending=False
        ).iloc[0]
    else:
        accuracy = eligible_accuracy.sort_values(
            ["accuracy", "recall"],
            ascending=False
        ).iloc[0]

    return (
        float(risk["threshold"]),
        float(accuracy["threshold"]),
        table
    )


# =====================================================
# 3. MODEL CONFIGURATIONS
# =====================================================

def get_models(imbalance_ratio):

    return {
        "Gradient Boosting": (
            HistGradientBoostingClassifier(
                random_state=SEED
            ),
            {
                "max_iter": [150, 250, 350, 500],
                "learning_rate": [
                    0.02, 0.03, 0.05, 0.08
                ],
                "max_leaf_nodes": [
                    7, 15, 31
                ],
                "l2_regularization": [
                    0.0, 1.0, 5.0, 10.0
                ],
                "min_samples_leaf": [
                    20, 40, 60
                ],
            }
        ),

        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=2
            ),
            {
                "n_estimators": [
                    200, 350, 500, 700
                ],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [
                    0.02, 0.03, 0.05, 0.08
                ],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [
                    0.7, 0.85, 1.0
                ],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [1, 3, 5, 10],
                "scale_pos_weight": [
                    1.0, imbalance_ratio
                ],
            }
        ),

        "LightGBM": (
            LGBMClassifier(
                objective="binary",
                random_state=SEED,
                n_jobs=2,
                verbosity=-1
            ),
            {
                "n_estimators": [
                    200, 350, 500, 700
                ],
                "learning_rate": [
                    0.02, 0.03, 0.05, 0.08
                ],
                "num_leaves": [
                    7, 15, 31
                ],
                "max_depth": [-1, 4, 6],
                "min_child_samples": [
                    20, 40, 60
                ],
                "reg_lambda": [0, 1, 5, 10],
                "scale_pos_weight": [
                    1.0, imbalance_ratio
                ],
            }
        ),

        "CatBoost": (
            CatBoostClassifier(
                loss_function="Logloss",
                verbose=False,
                thread_count=2,
                random_seed=SEED,
                allow_writing_files=False
            ),
            {
                "iterations": [
                    200, 350, 500, 700
                ],
                "depth": [4, 5, 6],
                "learning_rate": [
                    0.02, 0.03, 0.05, 0.08
                ],
                "l2_leaf_reg": [
                    3, 5, 7, 10
                ],
                "auto_class_weights": [
                    "Balanced",
                    "SqrtBalanced"
                ],
            }
        ),
    }


# =====================================================
# 4. TRAINING
# =====================================================

def main():

    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )
    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    df = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' missing."
        )

    advanced_df = add_advanced_features(df)

    original_features = [
        col for col in df.columns
        if col != TARGET
    ]

    advanced_features = [
        col for col in advanced_df.columns
        if col != TARGET
    ]

    y = df[TARGET].astype(int)

    indices = np.arange(len(df))

    # Identical splits for every experiment
    dev_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    train_idx, val_idx = train_test_split(
        dev_idx,
        test_size=0.25,
        random_state=SEED,
        stratify=y.iloc[dev_idx]
    )

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    imbalance_ratio = (
        (y_train == 0).sum()
        / (y_train == 1).sum()
    )

    print("\nDATASET INFORMATION")
    print("Rows:", len(df))
    print("Original features:", len(original_features))
    print("Advanced features:", len(advanced_features))
    print("Training:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    experiments = {
        "Original Features": (
            df, original_features
        ),
        "Advanced Features": (
            advanced_df, advanced_features
        ),
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    )

    comparison = []
    trained = {}

    for feature_set, (
        dataset, features
    ) in experiments.items():

        X_train = dataset.iloc[
            train_idx
        ][features]

        X_val = dataset.iloc[
            val_idx
        ][features]

        for name, (
            estimator, params
        ) in get_models(
            imbalance_ratio
        ).items():

            print(
                f"\nTraining {name} "
                f"({feature_set})..."
            )

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                n_iter=15,
                scoring="average_precision",
                cv=cv,
                random_state=SEED,
                n_jobs=1,
                refit=True,
                error_score="raise",
                verbose=0
            )

            search.fit(
                X_train, y_train
            )

            best_model = search.best_estimator_

            probabilities = (
                best_model.predict_proba(
                    X_val
                )[:, 1]
            )

            metrics = evaluate(
                y_val,
                probabilities,
                0.5
            )

            key = (feature_set, name)

            trained[key] = {
                "model": best_model,
                "features": features,
                "dataset": dataset,
                "params": search.best_params_,
            }

            comparison.append({
                "Feature Set": feature_set,
                "Model": name,
                "CV PR-AUC": float(
                    search.best_score_
                ),
                "Validation PR-AUC":
                    metrics["pr_auc"],
                "Validation ROC-AUC":
                    metrics["roc_auc"],
                "Validation Accuracy":
                    metrics["accuracy"],
                "Best Parameters":
                    str(search.best_params_),
            })

            print(
                "CV PR-AUC:",
                round(search.best_score_, 4)
            )
            print(
                "Validation PR-AUC:",
                round(metrics["pr_auc"], 4)
            )

    comparison_df = pd.DataFrame(
        comparison
    ).sort_values(
        "CV PR-AUC",
        ascending=False
    )

    comparison_df.to_csv(
        REPORT_DIR / "advanced_model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(
        comparison_df[
            [
                "Feature Set",
                "Model",
                "CV PR-AUC",
                "Validation PR-AUC",
                "Validation ROC-AUC",
                "Validation Accuracy",
            ]
        ].to_string(index=False)
    )

    # Model selection based only on training CV
    winner = comparison_df.iloc[0]

    best_feature_set = winner["Feature Set"]
    best_name = winner["Model"]

    best_info = trained[
        (best_feature_set, best_name)
    ]

    best_model = best_info["model"]
    best_features = best_info["features"]
    best_dataset = best_info["dataset"]

    X_val = best_dataset.iloc[
        val_idx
    ][best_features]

    X_test = best_dataset.iloc[
        test_idx
    ][best_features]

    val_probabilities = (
        best_model.predict_proba(
            X_val
        )[:, 1]
    )

    (
        risk_threshold,
        accuracy_threshold,
        threshold_table
    ) = analyze_thresholds(
        y_val,
        val_probabilities
    )

    threshold_table.to_csv(
        REPORT_DIR / "advanced_thresholds.csv",
        index=False
    )

    test_probabilities = (
        best_model.predict_proba(
            X_test
        )[:, 1]
    )

    # Both scenarios evaluated on same test set
    scenarios = {
        "Risk Detection": risk_threshold,
        "Accuracy Focused": accuracy_threshold,
        "Default 0.50": 0.50,
    }

    scenario_results = []
    confusion_matrices = {}

    for scenario, threshold in scenarios.items():

        scores = evaluate(
            y_test,
            test_probabilities,
            threshold
        )

        predictions = (
            test_probabilities >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).tolist()

        confusion_matrices[scenario] = cm

        scenario_results.append({
            "Scenario": scenario,
            "Threshold": threshold,
            **scores,
        })

    scenario_df = pd.DataFrame(
        scenario_results
    )

    scenario_df.to_csv(
        REPORT_DIR / "advanced_scenario_comparison.csv",
        index=False
    )

    print("\nSELECTED MODEL")
    print("Model:", best_name)
    print("Feature set:", best_feature_set)

    print("\nTHRESHOLD SCENARIO COMPARISON")
    print(
        scenario_df.to_string(
            index=False
        )
    )

    # Save risk-focused candidate as primary
    candidate_path = (
        MODEL_DIR
        / "repayiq_advanced_candidate.joblib"
    )

    joblib.dump(
        {
            "model": best_model,
            "features": best_features,
            "threshold": risk_threshold,
            "model_name": best_name,
            "feature_set": best_feature_set,
            "feature_engineering_version":
                "advanced_v2",
        },
        candidate_path
    )

    report = {
        "best_model": best_name,
        "feature_set": best_feature_set,
        "feature_count": len(best_features),
        "risk_threshold": risk_threshold,
        "accuracy_threshold": accuracy_threshold,
        "scenarios": scenario_results,
        "confusion_matrices": confusion_matrices,
        "best_parameters": best_info["params"],
        "evaluation_note": (
            "All scenarios use the same held-out "
            "split. This dataset has been examined "
            "in earlier experiments, so results "
            "are exploratory rather than an "
            "independent external validation."
        ),
    }

    with open(
        REPORT_DIR / "advanced_model_metrics.json",
        "w"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print("\nCandidate saved:")
    print(candidate_path)

    print(
        "\nExisting production model "
        "was not modified."
    )


if __name__ == "__main__":
    main()

