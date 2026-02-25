from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def compute_drift(axis="appearance"):

    root = Path(f"results_n2000/embeddings_openclip_{axis}")
    transforms = {}

    for file in root.glob("*.npy"):
        name = file.stem
        transform, level_str = name.split("_L")
        level = int(level_str)
        transforms.setdefault(transform, {})[level] = file

    rows = []

    for transform, levels_dict in transforms.items():

        e0 = np.load(levels_dict[0])

        for level, path in sorted(levels_dict.items()):
            eL = np.load(path)

            cosine_sim = np.sum(e0 * eL, axis=1)  # normalized vectors
            mean_sim = float(np.mean(cosine_sim))

            rows.append({
                "axis": axis,
                "transform": transform,
                "level": level,
                "mean_cosine_similarity": mean_sim,
                "mean_drift": 1 - mean_sim,
                "n": len(cosine_sim),
            })

    out = pd.DataFrame(rows)
    out_path = Path(f"results_n2000/embedding_drift_openclip_{axis}.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    compute_drift("appearance")
    compute_drift("geometry")