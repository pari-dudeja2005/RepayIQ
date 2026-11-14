
import pandas as pd
from ucimlrepo import fetch_ucirepo

from config import RAW_DATA_PATH, RAW_DATA_DIR


def load_dataset():

    print("Downloading UCI Credit Default Dataset...")

    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    df.columns = [
        str(col).strip().replace(" ", "_")
        for col in df.columns
    ]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_DATA_PATH, index=False)

    print("\nDataset downloaded successfully!")
    print("Saved to:", RAW_DATA_PATH)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(y.value_counts())

    return df


if __name__ == "__main__":
    load_dataset()
