
"""
Additional repayment-risk features.

Use after:
src.feature_engineering.create_features()
"""

import numpy as np


def add_advanced_features(df):
    data = df.copy()

    status_cols = [
        f"repay_status_{i}"
        for i in range(1, 7)
    ]

    bill_cols = [
        f"bill_{i}"
        for i in range(1, 7)
    ]

    payment_cols = [
        f"payment_{i}"
        for i in range(1, 7)
    ]

    required = (
        status_cols
        + bill_cols
        + payment_cols
        + ["credit_limit"]
    )

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for advanced features: {missing}"
        )

    statuses = data[
        status_cols
    ].to_numpy(dtype=float)

    bills = data[
        bill_cols
    ].to_numpy(dtype=float)

    payments = data[
        payment_cols
    ].to_numpy(dtype=float)

    positive_bills = np.maximum(bills, 0)

    late = statuses > 0

    safe_limit = np.maximum(
        data["credit_limit"].to_numpy(dtype=float),
        1.0
    )

    # -----------------------------------------
    # Repayment behaviour
    # -----------------------------------------

    consecutive = np.zeros(
        len(data),
        dtype=int
    )

    active = np.ones(
        len(data),
        dtype=bool
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
        np.maximum(statuses, 0).max(axis=1)
    )

    data["recent_delinquency_severity"] = (
        np.maximum(statuses[:, :3], 0)
        .mean(axis=1)
    )

    data["repayment_status_trend"] = (
        statuses[:, 0] - statuses[:, 5]
    )

    # -----------------------------------------
    # Outstanding balances
    # -----------------------------------------

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
    ) / (
        np.abs(bills[:, 5]) + 1.0
    )

    data["bill_volatility"] = (
        bills.std(axis=1)
        / (
            np.abs(bills.mean(axis=1)) + 1.0
        )
    )

    # -----------------------------------------
    # Payment patterns
    # -----------------------------------------

    data["repayment_consistency"] = (
        payments.std(axis=1)
        / (
            np.abs(payments.mean(axis=1)) + 1.0
        )
    )

    data["recent_payment_trend"] = (
        payments[:, 0] - payments[:, 5]
    ) / (
        np.abs(payments[:, 5]) + 1.0
    )

    # Preserve the original payment_to_bill_ratio.
    # Give the new calculation a different name.
    data["payment_to_positive_bill_ratio_v2"] = (
        payments.sum(axis=1)
        / (
            positive_bills.sum(axis=1) + 1.0
        )
    )

    data["latest_payment_to_bill_ratio"] = (
        payments[:, 0]
        / (
            positive_bills[:, 0] + 1.0
        )
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

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    added = [
        col for col in data.columns
        if col not in df.columns
    ]

    if not np.isfinite(
        data[added].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite advanced features detected."
        )

    return data
