from __future__ import annotations

from pathlib import Path
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "natural_abstraction" / "yolo_cls_dataset"
META_PATH = ROOT / "natural_abstraction" / "results_natural" / "yolo_cls_dataset_metadata.csv"
RESULTS_DIR = ROOT / "natural_abstraction" / "results_natural"

CLASS_ORDER = [
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

MODEL_PATH = ROOT / "natural_abstraction" / "yolo_runs" / "yolo26n_cls_mirrored" / "weights" / "best.pt"


def main():
    model = YOLO(str(MODEL_PATH))
    meta = pd.read_csv(META_PATH)
    test_df = meta[meta["split"] == "test"].copy()

    rows = []

    for _, row in test_df.iterrows():
        img_path = row["image_path"]
        results = model.predict(source=img_path, verbose=False)

        probs = results[0].probs
        pred_idx = int(probs.top1)
        conf = float(probs.top1conf.item())

        true_idx = int(row["true_label"])

        rows.append({
            "model": "yolo26n_cls",
            "dataset": "natural_gallery",
            "image_path": img_path,
            "true_label": true_idx,
            "true_label_name": row["true_label_name"],
            "subtype": row["subtype"],
            "pred_label": pred_idx,
            "pred_label_name": CLASS_ORDER[pred_idx],
            "confidence": conf,
            "correct": int(pred_idx == true_idx),
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(RESULTS_DIR / "baseline_natural_yolo26.csv", index=False)

    # overall metrics
    acc = accuracy_score(pred_df["true_label"], pred_df["pred_label"])
    macro_f1 = f1_score(pred_df["true_label"], pred_df["pred_label"], average="macro")

    overall = pd.DataFrame([{
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n": len(pred_df),
    }])
    overall.to_csv(RESULTS_DIR / "summary_baseline_natural_yolo26_overall.csv", index=False)

    # by subtype
    by_subtype = (
        pred_df.groupby("subtype")
        .apply(lambda d: pd.Series({
            "accuracy": accuracy_score(d["true_label"], d["pred_label"]),
            "macro_f1": f1_score(d["true_label"], d["pred_label"], average="macro"),
            "mean_confidence": d["confidence"].mean(),
            "n": len(d),
        }))
        .reset_index()
    )
    by_subtype.to_csv(RESULTS_DIR / "summary_baseline_natural_yolo26_by_subtype.csv", index=False)

    # by class
    by_class = (
        pred_df.groupby("true_label_name")
        .apply(lambda d: pd.Series({
            "accuracy": accuracy_score(d["true_label"], d["pred_label"]),
            "mean_confidence": d["confidence"].mean(),
            "n": len(d),
        }))
        .reset_index()
    )
    by_class.to_csv(RESULTS_DIR / "summary_baseline_natural_yolo26_by_class.csv", index=False)

    # by class x subtype
    by_class_subtype = (
        pred_df.groupby(["true_label_name", "subtype"])
        .apply(lambda d: pd.Series({
            "accuracy": accuracy_score(d["true_label"], d["pred_label"]),
            "mean_confidence": d["confidence"].mean(),
            "n": len(d),
        }))
        .reset_index()
    )
    by_class_subtype.to_csv(RESULTS_DIR / "summary_baseline_natural_yolo26_by_class_subtype.csv", index=False)

    # confusion
    conf = confusion_matrix(pred_df["true_label"], pred_df["pred_label"], labels=list(range(10)))
    conf_df = pd.DataFrame(conf, index=CLASS_ORDER, columns=CLASS_ORDER)
    conf_df.to_csv(RESULTS_DIR / "confusion_baseline_natural_yolo26_overall.csv")

    print("Overall accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Wrote YOLO clean evaluation outputs.")

if __name__ == "__main__":
    main()