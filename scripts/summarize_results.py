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
    

def compute_failure_levels(path_in: str, path_out: str | None = None):
    import pandas as pd
    df = pd.read_csv(path_in)

    metric = "accuracy" if "accuracy" in df.columns else "mean_confidence"
    group_keys = [k for k in ["model", "axis", "transform"] if k in df.columns]

    results = []
    for keys, group in df.groupby(group_keys):
        group = group.sort_values("level")
        baseline_rows = group[group["level"] == 0][metric].values
        if len(baseline_rows) == 0:
            continue

        baseline = float(baseline_rows[0])
        threshold = 0.5 * baseline

        failure_level = None
        for _, row in group.iterrows():
            if int(row["level"]) > 0 and float(row[metric]) <= threshold:
                failure_level = int(row["level"])
                break

        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        out_row = dict(zip(group_keys, keys_tuple))
        out_row.update({
            "metric": metric,
            "baseline_level0": baseline,
            "threshold_50pct": threshold,
            "failure_level_50pct": failure_level,
        })
        results.append(out_row)

    out = pd.DataFrame(results)
    print("\nFailure levels (50% drop rule):")
    print(out)

    if path_out:
        out.to_csv(path_out, index=False)
        print(f"Wrote {path_out}")

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

    summarize(
        "results/corruptions_geometry_efficientnet.csv",
        "results/summary_corruptions_geometry_efficientnet.csv",
    )

    summarize(
        "results/corruptions_geometry_openclip.csv",
        "results/summary_corruptions_geometry_openclip.csv",
    )

    # Failure threshold analysis (optional but useful)
    compute_failure_levels(
        "results/summary_corruptions_appearance_efficientnet.csv",
        "results/failure_corruptions_appearance_efficientnet.csv",
    )
    compute_failure_levels(
        "results/summary_corruptions_geometry_efficientnet.csv",
        "results/failure_corruptions_geometry_efficientnet.csv",
    )
    compute_failure_levels(
        "results/summary_corruptions_appearance_openclip.csv",
        "results/failure_corruptions_appearance_openclip.csv",
    )
    compute_failure_levels(
        "results/summary_corruptions_geometry_openclip.csv",
        "results/failure_corruptions_geometry_openclip.csv",
    )

