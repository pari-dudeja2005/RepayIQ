
import pandas as pd

from config import RAW_DATA_PATH


COLUMN_MAPPING = {
    "X1": "credit_limit",
    "X2": "sex",
    "X3": "education",
    "X4": "marital_status",
    "X5": "age",

    "X6": "repay_status_1",
    "X7": "repay_status_2",
    "X8": "repay_status_3",
    "X9": "repay_status_4",
    "X10": "repay_status_5",
    "X11": "repay_status_6",

    "X12": "bill_1",
    "X13": "bill_2",
    "X14": "bill_3",
    "X15": "bill_4",
    "X16": "bill_5",
    "X17": "bill_6",

    "X18": "payment_1",
    "X19": "payment_2",
    "X20": "payment_3",
    "X21": "payment_4",
    "X22": "payment_5",
    "X23": "payment_6",

    "Y": "default_next_month"
}


def preprocess_data():

    df = pd.read_csv(RAW_DATA_PATH)

    print("Original Dataset:", df.shape)

    df = df.rename(columns=COLUMN_MAPPING)

    # Remove demographic variables from the model inputs.
    df = df.drop(
        columns=["sex", "education", "marital_status", "age"]
    )

    # Remove duplicate records if any.
    df = df.drop_duplicates()

    print("Preprocessed Dataset:", df.shape)

    return df


if __name__ == "__main__":

    df = preprocess_data()

    print("\nUpdated Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())
