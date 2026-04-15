from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_natural"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def main():
    df = pd.read_csv(RESULTS_DIR / "baseline_natural_openclip.csv")

    # overall confusion
    overall = pd.crosstab(
        df["true_label_name"],
        df["pred_label_name"],
        rownames=["true"],
        colnames=["pred"],
        dropna=False
    ).reindex(index=CIFAR10_CLASSES, columns=CIFAR10_CLASSES, fill_value=0)

    overall.to_csv(RESULTS_DIR / "confusion_baseline_natural_openclip_overall.csv")
    print("Overall confusion:")
    print(overall)

    # confusion by subtype
    for subtype in sorted(df["subtype"].unique()):
        sub = df[df["subtype"] == subtype]
        cm = pd.crosstab(
            sub["true_label_name"],
            sub["pred_label_name"],
            rownames=["true"],
            colnames=["pred"],
            dropna=False
        ).reindex(index=CIFAR10_CLASSES, columns=CIFAR10_CLASSES, fill_value=0)

        out_path = RESULTS_DIR / f"confusion_baseline_natural_openclip_{subtype}.csv"
        cm.to_csv(out_path)
        print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()