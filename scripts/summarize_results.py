from __future__ import annotations

from pathlib import Path
import pandas as pd


def summarize(path_in: str, path_out: str) -> None:
    df = pd.read_csv(path_in)

    has_correct = "correct" in df.columns
    group_cols = [c for c in ["model", "dataset", "axis", "transform", "level"] if c in df.columns]

    if not group_cols:
        raise ValueError(f"No grouping columns found in {path_in}")

    if has_correct:
        out = (
            df.groupby(group_cols)
              .agg(
                  accuracy=("correct", "mean"),
                  mean_confidence=("confidence", "mean"),
                  n=("confidence", "count"),
              )
              .reset_index()
        )
    else:
        out = (
            df.groupby(group_cols)
              .agg(
                  mean_confidence=("confidence", "mean"),
                  n=("confidence", "count"),
              )
              .reset_index()
        )

    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path_out, index=False)
    print(f"Wrote {path_out}")
    

def compute_failure_levels(path_in: str):
    import pandas as pd
    df = pd.read_csv(path_in)

    results = []

    for (model, transform), group in df.groupby(["model", "transform"]):
        group = group.sort_values("level")

        baseline = group[group["level"] == 0]["mean_confidence"].values[0]
        threshold = 0.5 * baseline

        failure_level = None
        for _, row in group.iterrows():
            if row["level"] > 0 and row["mean_confidence"] <= threshold:
                failure_level = int(row["level"])
                break

        results.append({
            "model": model,
            "transform": transform,
            "failure_level_50pct": failure_level
        })

    out = pd.DataFrame(results)
    print("\nFailure levels (50% drop rule):")
    print(out)
if __name__ == "__main__":

    # Baseline
    summarize(
        "results/baseline_openclip.csv",
        "results/summary_baseline_openclip.csv",
    )

    # EfficientNet corruptions
    summarize(
        "results/corruptions_appearance_efficientnet.csv",
        "results/summary_corruptions_appearance_efficientnet.csv",
    )

    # OpenCLIP corruptions  <-- THIS WAS MISSING
    summarize(
        "results/corruptions_appearance_openclip.csv",
        "results/summary_corruptions_appearance_openclip.csv",
    )

    # Failure threshold analysis (optional but useful)
    compute_failure_levels(
        "results/summary_corruptions_appearance_efficientnet.csv"
    )

    compute_failure_levels(
        "results/summary_corruptions_appearance_openclip.csv"
    )


