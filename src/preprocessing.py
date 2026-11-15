
import pandas as pd
from config import RAW_DATA_PATH


COLUMN_MAPPING = {
    "X1": "credit_limit",
    "X2": "sex",
    "X3": "education",
    "X4": "marital_status",
    "X5": "age",
    **{
        f"X{i+5}": f"repay_status_{i}"
        for i in range(1, 7)
    },
    **{
        f"X{i+11}": f"bill_{i}"
        for i in range(1, 7)
    },
    **{
        f"X{i+17}": f"payment_{i}"
        for i in range(1, 7)
    },
    "Y": "default_next_month",
}


def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset:", df.shape)

    df = df.rename(columns=COLUMN_MAPPING)

    # Check duplicate customer IDs before removing ID.
    if "ID" in df.columns:
        print(
            "Duplicate IDs:",
            int(df["ID"].duplicated().sum())
        )
        df = df.drop(columns=["ID"])

    # Audit exact duplicates before removing demographics.
    print(
        "Exact duplicate rows (before demographic removal):",
        int(df.duplicated().sum())
    )

    # Exclude demographic fields from model inputs.
    # Do not remove records merely because their
    # financial attributes are identical.
    df = df.drop(
        columns=[
            "sex",
            "education",
            "marital_status",
            "age"
        ]
    )

    print(
        "Matching rows after demographic removal (retained):",
        int(df.duplicated().sum())
    )

    print("Preprocessed dataset:", df.shape)

    return df


if __name__ == "__main__":
    preprocess_data()


