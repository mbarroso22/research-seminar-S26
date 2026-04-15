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


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def main():
    natural_emb_path = RESULTS_DIR / "natural_dino_embeddings.npy"
    natural_meta_path = RESULTS_DIR / "natural_dino_metadata.csv"

    cifar_emb_path = RESULTS_DIR / "cifar_baseline_dino_embeddings.npy"
    cifar_meta_path = RESULTS_DIR / "cifar_baseline_dino_metadata.csv"

    if not natural_emb_path.exists():
        raise FileNotFoundError(f"Missing {natural_emb_path}")
    if not natural_meta_path.exists():
        raise FileNotFoundError(f"Missing {natural_meta_path}")
    if not cifar_emb_path.exists():
        raise FileNotFoundError(
            f"Missing {cifar_emb_path}. You need CIFAR baseline DINO embeddings first."
        )
    if not cifar_meta_path.exists():
        raise FileNotFoundError(
            f"Missing {cifar_meta_path}. You need CIFAR baseline DINO metadata first."
        )

    natural_emb = np.load(natural_emb_path)
    natural_meta = pd.read_csv(natural_meta_path)

    cifar_emb = np.load(cifar_emb_path)
    cifar_meta = pd.read_csv(cifar_meta_path)

    # ---- build CIFAR class prototypes ----
    prototypes = {}
    proto_rows = []

    for class_name in CIFAR10_CLASSES:
        mask = cifar_meta["true_label_name"] == class_name
        class_vecs = cifar_emb[mask.values]

        if len(class_vecs) == 0:
            raise ValueError(f"No CIFAR DINO embeddings found for class '{class_name}'")

        proto = class_vecs.mean(axis=0)
        proto = normalize(proto)
        prototypes[class_name] = proto

        proto_rows.append({
            "class_name": class_name,
            "n": len(class_vecs),
        })

    pd.DataFrame(proto_rows).to_csv(
        RESULTS_DIR / "cifar_dino_prototype_counts.csv", index=False
    )

    # ---- per-image drift ----
    rows = []
    for i in range(len(natural_meta)):
        class_name = natural_meta.loc[i, "true_label_name"]
        subtype = natural_meta.loc[i, "subtype"] if "subtype" in natural_meta.columns else "unknown"

        vec = normalize(natural_emb[i])
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
    per_image_path = RESULTS_DIR / "dino_natural_drift_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    # ---- by class ----
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
    by_class_path = RESULTS_DIR / "dino_natural_drift_by_class.csv"
    by_class.to_csv(by_class_path, index=False)

    # ---- by subtype ----
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
    by_subtype_path = RESULTS_DIR / "dino_natural_drift_by_subtype.csv"
    by_subtype.to_csv(by_subtype_path, index=False)

    # ---- by class x subtype ----
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
    by_class_subtype_path = RESULTS_DIR / "dino_natural_drift_by_class_subtype.csv"
    by_class_subtype.to_csv(by_class_subtype_path, index=False)

    print(f"Wrote {per_image_path}")
    print(f"Wrote {by_class_path}")
    print(f"Wrote {by_subtype_path}")
    print(f"Wrote {by_class_subtype_path}")

    print("\nDINO drift by class:")
    print(by_class)

    print("\nDINO drift by subtype:")
    print(by_subtype)


if __name__ == "__main__":
    main()