from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "natural_abstraction" / "results_natural"

# -----------------------------
# Load embeddings + metadata
# -----------------------------
nat_emb = np.load(RESULTS / "natural_dino_embeddings.npy")
nat_meta = pd.read_csv(RESULTS / "natural_dino_metadata.csv")

cifar_emb = np.load(RESULTS / "cifar_baseline_dino_embeddings.npy")
cifar_meta = pd.read_csv(RESULTS / "cifar_baseline_dino_metadata.csv")

print("Natural embeddings:", nat_emb.shape)
print("CIFAR embeddings:", cifar_emb.shape)

# -----------------------------
# Build CIFAR prototypes
# -----------------------------
prototypes = {}

for label in sorted(cifar_meta["true_label"].unique()):
    mask = cifar_meta["true_label"] == label
    class_embs = cifar_emb[mask]
    prototypes[label] = class_embs.mean(axis=0)

# Convert to matrix
proto_labels = sorted(prototypes.keys())
proto_matrix = np.stack([prototypes[l] for l in proto_labels])

# -----------------------------
# Normalize embeddings
# -----------------------------
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

nat_emb_norm = normalize(nat_emb)
proto_matrix_norm = normalize(proto_matrix)

# -----------------------------
# Compute similarities
# -----------------------------
sim = cosine_similarity(nat_emb_norm, proto_matrix_norm)

# -----------------------------
# Predict classes
# -----------------------------
pred_indices = sim.argmax(axis=1)
pred_labels = [proto_labels[i] for i in pred_indices]

nat_meta["pred_label"] = pred_labels
nat_meta["correct"] = (nat_meta["pred_label"] == nat_meta["true_label"]).astype(int)

# -----------------------------
# Accuracy
# -----------------------------
overall_acc = nat_meta["correct"].mean()
print("\nDINO Natural Accuracy:", overall_acc)

# -----------------------------
# Accuracy by subtype
# -----------------------------
acc_by_subtype = (
    nat_meta
    .groupby("subtype")["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy", "count": "n"})
    .reset_index()
)

print("\nAccuracy by subtype:")
print(acc_by_subtype)

# -----------------------------
# Accuracy by class
# -----------------------------
acc_by_class = (
    nat_meta
    .groupby("true_label_name")["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy", "count": "n"})
    .reset_index()
)

print("\nAccuracy by class:")
print(acc_by_class)

# -----------------------------
# Confusion matrix
# -----------------------------
conf = confusion_matrix(
    nat_meta["true_label"],
    nat_meta["pred_label"]
)

conf_df = pd.DataFrame(conf)

# -----------------------------
# Save outputs
# -----------------------------
nat_meta.to_csv(RESULTS / "dino_natural_predictions.csv", index=False)
acc_by_subtype.to_csv(RESULTS / "dino_natural_accuracy_by_subtype.csv", index=False)
acc_by_class.to_csv(RESULTS / "dino_natural_accuracy_by_class.csv", index=False)
conf_df.to_csv(RESULTS / "dino_natural_confusion_matrix.csv", index=False)

print("\nSaved outputs to:", RESULTS)