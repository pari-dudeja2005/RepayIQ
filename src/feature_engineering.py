
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_PATH, PROCESSED_DATA_DIR
from src.preprocessing import preprocess_data


def create_features(df):

    df = df.copy()

    status_cols = [
        f"repay_status_{i}" for i in range(1, 7)
    ]

    bill_cols = [
        f"bill_{i}" for i in range(1, 7)
    ]

    payment_cols = [
        f"payment_{i}" for i in range(1, 7)
    ]

    # 1. Number of months with repayment delays
    df["late_payment_count"] = (
        df[status_cols] > 0
    ).sum(axis=1)

    # 2. Maximum recorded repayment delay
    df["max_delay"] = (
        df[status_cols].clip(lower=0).max(axis=1)
    )

    # 3. Most recent repayment status
    df["recent_delay"] = df["repay_status_1"]

    # 4. Average bill amount
    df["avg_bill_amount"] = (
        df[bill_cols].mean(axis=1)
    )

    # 5. Average payment amount
    df["avg_payment_amount"] = (
        df[payment_cols].mean(axis=1)
    )

    # 6. Total payment-to-bill ratio
    total_bills = (
        df[bill_cols].clip(lower=0).sum(axis=1)
    )

    total_payments = (
        df[payment_cols].sum(axis=1)
    )

    df["payment_to_bill_ratio"] = (
        total_payments /
        np.maximum(total_bills, 1)
    )

    # 7. Credit utilization proxy
    df["credit_utilization"] = (
        df["bill_1"].clip(lower=0) /
        df["credit_limit"].clip(lower=1)
    )

    # 8. Change in billing amount
    df["bill_change"] = (
        df["bill_1"] - df["bill_2"]
    )

    # 9. Change in payment amount
    df["payment_change"] = (
        df["payment_1"] - df["payment_2"]
    )

    # 10. Payment consistency
    df["payment_volatility"] = (
        df[payment_cols].std(axis=1)
    )

    # 11. Number of months with positive payments
    df["months_with_payment"] = (
        df[payment_cols] > 0
    ).sum(axis=1)

    return df


if __name__ == "__main__":

    df = preprocess_data()

    df = create_features(df)

    PROCESSED_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("\nFeature Engineering Completed!")

    print("Final Dataset Shape:", df.shape)

    print("\nEngineered Features:")
    engineered_features = [
        "late_payment_count",
        "max_delay",
        "recent_delay",
        "avg_bill_amount",
        "avg_payment_amount",
        "payment_to_bill_ratio",
        "credit_utilization",
        "bill_change",
        "payment_change",
        "payment_volatility",
        "months_with_payment"
    ]

    print(df[engineered_features].head())

    print("\nSaved to:", PROCESSED_DATA_PATH)
