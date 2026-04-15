from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

NA_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = NA_ROOT / "results_natural"

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
    natural_emb = np.load(RESULTS_DIR / "natural_openclip_embeddings.npy")
    natural_meta = pd.read_csv(RESULTS_DIR / "natural_openclip_metadata.csv")

    cifar_emb = np.load(RESULTS_DIR / "cifar_baseline_openclip_embeddings.npy")
    cifar_meta = pd.read_csv(RESULTS_DIR / "cifar_baseline_openclip_metadata.csv")

    # Build class prototypes from CIFAR baseline
    prototype_rows = []
    prototypes = {}

    for class_name in CIFAR10_CLASSES:
        mask = cifar_meta["true_label_name"] == class_name
        class_vecs = cifar_emb[mask.values]

        proto = class_vecs.mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        prototypes[class_name] = proto

        prototype_rows.append({
            "class_name": class_name,
            "n": len(class_vecs),
        })

    pd.DataFrame(prototype_rows).to_csv(
        RESULTS_DIR / "cifar_openclip_prototype_counts.csv", index=False
    )

    # Per-image drift
    rows = []
    for i in range(len(natural_meta)):
        class_name = natural_meta.loc[i, "true_label_name"]
        subtype = natural_meta.loc[i, "subtype"]
        vec = natural_emb[i]
        proto = prototypes[class_name]

        sim = float(np.dot(vec, proto))
        drift = 1.0 - sim

        rows.append({
            "image_id": int(natural_meta.loc[i, "image_id"]),
            "image_path": natural_meta.loc[i, "image_path"],
            "true_label": int(natural_meta.loc[i, "true_label"]),
            "true_label_name": class_name,
            "subtype": subtype,
            "cosine_similarity_to_cifar_prototype": sim,
            "drift_from_cifar_prototype": drift,
        })

    per_image = pd.DataFrame(rows)
    per_image_path = RESULTS_DIR / "openclip_natural_drift_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    # Summary by class
    by_class = (
        per_image.groupby("true_label_name")
        .agg(
            mean_cosine_similarity=("cosine_similarity_to_cifar_prototype", "mean"),
            mean_drift=("drift_from_cifar_prototype", "mean"),
            n=("drift_from_cifar_prototype", "count"),
        )
        .reset_index()
        .sort_values("true_label_name")
    )
    by_class_path = RESULTS_DIR / "openclip_natural_drift_by_class.csv"
    by_class.to_csv(by_class_path, index=False)

    # Summary by subtype
    by_subtype = (
        per_image.groupby("subtype")
        .agg(
            mean_cosine_similarity=("cosine_similarity_to_cifar_prototype", "mean"),
            mean_drift=("drift_from_cifar_prototype", "mean"),
            n=("drift_from_cifar_prototype", "count"),
        )
        .reset_index()
        .sort_values("subtype")
    )
    by_subtype_path = RESULTS_DIR / "openclip_natural_drift_by_subtype.csv"
    by_subtype.to_csv(by_subtype_path, index=False)

    # Summary by class x subtype
    by_class_subtype = (
        per_image.groupby(["true_label_name", "subtype"])
        .agg(
            mean_cosine_similarity=("cosine_similarity_to_cifar_prototype", "mean"),
            mean_drift=("drift_from_cifar_prototype", "mean"),
            n=("drift_from_cifar_prototype", "count"),
        )
        .reset_index()
        .sort_values(["true_label_name", "subtype"])
    )
    by_class_subtype_path = RESULTS_DIR / "openclip_natural_drift_by_class_subtype.csv"
    by_class_subtype.to_csv(by_class_subtype_path, index=False)

    print(f"Wrote {per_image_path}")
    print(f"Wrote {by_class_path}")
    print(f"Wrote {by_subtype_path}")
    print(f"Wrote {by_class_subtype_path}")

    print("\nOpenCLIP drift by class:")
    print(by_class)

    print("\nOpenCLIP drift by subtype:")
    print(by_subtype)

if __name__ == "__main__":
    main()