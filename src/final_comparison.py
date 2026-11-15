
"""
RepayIQ - Final Model Comparison

Compares:
1. Original Gradient Boosting
2. Advanced CatBoost

Uses identical train/validation/test splits.
Does not overwrite the deployed model.
"""

import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.ensemble import (
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

from catboost import CatBoostClassifier

from config import (
    PROCESSED_DATA_PATH,
    REPORT_DIR,
    RANDOM_STATE
)

from src.advanced_features import (
    add_advanced_features
)


TARGET = "default_next_month"

THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2
)


# =====================================================
# EVALUATION METRICS
# =====================================================

def metrics(y, proba, threshold):

    pred = (
        proba >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, proba)
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                proba
            )
        )
    }


# =====================================================
# THRESHOLD OPTIMIZATION
# =====================================================

def thresholds(y, proba):

    rows = []

    for t in THRESHOLDS:

        scores = metrics(
            y,
            proba,
            t
        )

        rows.append({
            "threshold": float(t),
            **scores
        })

    table = pd.DataFrame(rows)

    # Risk-focused scenario:
    # Maximize F1 with recall >= 60%.

    risk = table[
        table["recall"] >= 0.60
    ]

    if risk.empty:
        risk = table.sort_values(
            ["recall", "f1"],
            ascending=False
        )
    else:
        risk = risk.sort_values(
            ["f1", "precision"],
            ascending=False
        )

    # Accuracy-focused scenario:
    # Maximize accuracy with recall >= 30%.

    accuracy = table[
        table["recall"] >= 0.30
    ]

    if accuracy.empty:
        accuracy = table.sort_values(
            ["accuracy", "recall"],
            ascending=False
        )
    else:
        accuracy = accuracy.sort_values(
            ["accuracy", "recall"],
            ascending=False
        )

    return {
        "Risk Focused": float(
            risk.iloc[0]["threshold"]
        ),
        "Accuracy Focused": float(
            accuracy.iloc[0]["threshold"]
        ),
        "Default 0.50": 0.50
    }, table


# =====================================================
# MAIN TRAINING EXPERIMENT
# =====================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    base = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    if TARGET not in base.columns:
        raise ValueError(
            f"Missing target: {TARGET}"
        )

    advanced = add_advanced_features(
        base
    )

    y = base[TARGET].astype(int)

    indices = np.arange(
        len(base)
    )

    # Identical splits for both models

    dev, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=2026,
        stratify=y
    )

    train, val = train_test_split(
        dev,
        test_size=0.25,
        random_state=2026,
        stratify=y.iloc[dev]
    )

    print("\nDATASET INFORMATION")
    print("Rows:", len(base))
    print("Training:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    print(
        "Majority-class test accuracy:",
        round(
            float(
                (y.iloc[test] == 0).mean()
            ),
            4
        )
    )

    experiments = [
        (
            "Original Gradient Boosting",
            base,
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
            {
                "max_iter": [
                    150, 250, 350
                ],
                "learning_rate": [
                    0.03, 0.05, 0.08
                ],
                "max_leaf_nodes": [
                    7, 15, 31
                ],
                "l2_regularization": [
                    0.0, 1.0, 5.0
                ]
            }
        ),
        (
            "Advanced CatBoost",
            advanced,
            CatBoostClassifier(
                loss_function="Logloss",
                verbose=False,
                random_seed=RANDOM_STATE,
                thread_count=2,
                allow_writing_files=False
            ),
            {
                "iterations": [
                    200, 350, 500
                ],
                "depth": [
                    4, 5, 6
                ],
                "learning_rate": [
                    0.02, 0.05, 0.08
                ],
                "l2_leaf_reg": [
                    3, 5, 7
                ]
            }
        )
    ]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=2026
    )

    results = []
    selected = []

    # =================================================
    # TRAIN BOTH MODELS
    # =================================================

    for (
        name,
        frame,
        estimator,
        param_grid
    ) in experiments:

        print(
            f"\nTraining: {name}"
        )

        X = frame.drop(
            columns=[TARGET]
        )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Invalid features in {name}"
            )

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=10,
            scoring="average_precision",
            cv=cv,
            random_state=2026,
            n_jobs=1,
            error_score="raise"
        )

        search.fit(
            X.iloc[train],
            y.iloc[train]
        )

        model = search.best_estimator_

        # Validation predictions
        val_proba = (
            model.predict_proba(
                X.iloc[val]
            )[:, 1]
        )

        chosen, table = thresholds(
            y.iloc[val],
            val_proba
        )

        filename = (
            "thresholds_"
            + name.lower().replace(
                " ", "_"
            )
            + ".csv"
        )

        table.to_csv(
            REPORT_DIR / filename,
            index=False
        )

        # Test predictions
        test_proba = (
            model.predict_proba(
                X.iloc[test]
            )[:, 1]
        )

        cv_pr_auc = float(
            search.best_score_
        )

        validation_pr_auc = float(
            average_precision_score(
                y.iloc[val],
                val_proba
            )
        )

        print(
            "CV PR-AUC:",
            round(cv_pr_auc, 4)
        )

        print(
            "Validation PR-AUC:",
            round(validation_pr_auc, 4)
        )

        selected.append({
            "model": name,
            "cv_pr_auc": cv_pr_auc,
            "validation_pr_auc":
                validation_pr_auc,
            "best_params":
                search.best_params_
        })

        # Evaluate each threshold scenario

        for scenario, t in chosen.items():

            pred = (
                test_proba >= t
            ).astype(int)

            tn, fp, fn, tp = (
                confusion_matrix(
                    y.iloc[test],
                    pred,
                    labels=[0, 1]
                ).ravel()
            )

            results.append({
                "Model": name,
                "Scenario": scenario,
                "Threshold": t,
                **metrics(
                    y.iloc[test],
                    test_proba,
                    t
                ),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            })

    # =================================================
    # SAVE REPORTS
    # =================================================

    comparison = pd.DataFrame(
        results
    )

    comparison.to_csv(
        REPORT_DIR
        / "final_fair_comparison.csv",
        index=False
    )

    metadata = {
        "rows": len(base),
        "split_seed": 2026,
        "note": (
            "Exploratory matched-split comparison. "
            "The dataset was examined in earlier "
            "experiments, so this is not an "
            "independent external validation."
        ),
        "models": selected
    }

    with open(
        REPORT_DIR
        / "final_comparison_metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print(
        "\nFINAL MODEL COMPARISON"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        REPORT_DIR
        / "final_fair_comparison.csv"
    )

    print(
        "\nExisting dashboard model "
        "was not modified."
    )


if __name__ == "__main__":
    main()
