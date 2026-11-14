
import pandas as pd

from config import REPORT_DIR


MIN_RECALL = 0.60


def main():

    path = (
        REPORT_DIR / "threshold_comparison.csv"
    )

    df = pd.read_csv(path)

    print("\nVALIDATION THRESHOLD COMPARISON\n")

    display = df[
        df["Threshold"].round(2).isin(
            [
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60
            ]
        )
    ].copy()

    print(
        display.round(4).to_string(
            index=False
        )
    )

    eligible = df[
        df["Recall"] >= MIN_RECALL
    ].copy()

    if eligible.empty:
        print(
            "\nNo threshold satisfies "
            "the minimum recall requirement."
        )
        return

    best = eligible.sort_values(
        ["Accuracy", "F1 Score", "Threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    print("\nBEST VALIDATION THRESHOLD\n")

    print(best.round(4).to_string())


if __name__ == "__main__":
    main()

