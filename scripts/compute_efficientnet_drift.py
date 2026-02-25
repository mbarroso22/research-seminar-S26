from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def compute_drift(axis: str):
    root = Path(f"results_n2000/embeddings_efficientnet_{axis}")
    if not root.exists():
        raise FileNotFoundError(f"Missing folder: {root}")

    transforms = {}
    for file in root.glob("*.npy"):
        stem = file.stem  # e.g., blur_L2
        transform, level_str = stem.split("_L")
        level = int(level_str)
        transforms.setdefault(transform, {})[level] = file

    rows = []
    for transform, levels_dict in transforms.items():
        e0 = np.load(levels_dict[0])

        for level, path in sorted(levels_dict.items()):
            eL = np.load(path)
            cos = np.sum(e0 * eL, axis=1)  # both normalized
            mean_sim = float(np.mean(cos))
            rows.append(
                {
                    "model": "efficientnet_v2_s",
                    "axis": axis,
                    "transform": transform,
                    "level": level,
                    "mean_cosine_similarity": mean_sim,
                    "mean_drift": 1.0 - mean_sim,
                    "n": int(len(cos)),
                }
            )

    out = pd.DataFrame(rows).sort_values(["transform", "level"])
    out_path = Path(f"results_n2000/embedding_drift_efficientnet_{axis}.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    compute_drift("appearance")
    compute_drift("geometry")