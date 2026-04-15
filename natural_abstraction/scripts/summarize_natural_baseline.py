from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_natural"

def summarize_openclip():
    df = pd.read_csv(RESULTS_DIR / "baseline_natural_openclip.csv")

    by_class = (
        df.groupby(["true_label_name"])
        .agg(
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            n=("correct", "count"),
        )
        .reset_index()
        .sort_values("true_label_name")
    )

    by_subtype = (
        df.groupby(["subtype"])
        .agg(
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            n=("correct", "count"),
        )
        .reset_index()
        .sort_values("subtype")
    )

    by_class_subtype = (
        df.groupby(["true_label_name", "subtype"])
        .agg(
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            n=("correct", "count"),
        )
        .reset_index()
        .sort_values(["true_label_name", "subtype"])
    )

    by_class.to_csv(RESULTS_DIR / "summary_baseline_natural_openclip_by_class.csv", index=False)
    by_subtype.to_csv(RESULTS_DIR / "summary_baseline_natural_openclip_by_subtype.csv", index=False)
    by_class_subtype.to_csv(RESULTS_DIR / "summary_baseline_natural_openclip_by_class_subtype.csv", index=False)

    print("\nOpenCLIP by class:")
    print(by_class)
    print("\nOpenCLIP by subtype:")
    print(by_subtype)

def summarize_efficientnet():
    df = pd.read_csv(RESULTS_DIR / "baseline_natural_efficientnet.csv")

    by_class = (
        df.groupby(["true_label_name"])
        .agg(
            mean_confidence=("confidence", "mean"),
            n=("confidence", "count"),
        )
        .reset_index()
        .sort_values("true_label_name")
    )

    by_subtype = (
        df.groupby(["subtype"])
        .agg(
            mean_confidence=("confidence", "mean"),
            n=("confidence", "count"),
        )
        .reset_index()
        .sort_values("subtype")
    )

    by_class_subtype = (
        df.groupby(["true_label_name", "subtype"])
        .agg(
            mean_confidence=("confidence", "mean"),
            n=("confidence", "count"),
        )
        .reset_index()
        .sort_values(["true_label_name", "subtype"])
    )

    by_class.to_csv(RESULTS_DIR / "summary_baseline_natural_efficientnet_by_class.csv", index=False)
    by_subtype.to_csv(RESULTS_DIR / "summary_baseline_natural_efficientnet_by_subtype.csv", index=False)
    by_class_subtype.to_csv(RESULTS_DIR / "summary_baseline_natural_efficientnet_by_class_subtype.csv", index=False)

    print("\nEfficientNet by class:")
    print(by_class)
    print("\nEfficientNet by subtype:")
    print(by_subtype)

if __name__ == "__main__":
    summarize_openclip()
    summarize_efficientnet()