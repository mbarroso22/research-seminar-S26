from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score

from src.data import cifar10_loader
from src.transforms import pixelate, blur_pil, rotate_pil, shear_x_pil, translate_pil

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "natural_abstraction" / "results_natural"
MODEL_PATH = ROOT / "natural_abstraction" / "yolo_runs" / "yolo26n_cls_mirrored" / "weights" / "best.pt"

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

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def run_axis(axis_name, transforms_dict, levels=(0, 1, 2, 3), n=2000):
    model = YOLO(str(MODEL_PATH))
    dl = cifar10_loader(n=n)
    rows = []

    for transform_name, transform_fn in transforms_dict.items():
        for level in levels:
            global_idx = 0

            for _, (x, y) in enumerate(dl):
                for j in range(x.size(0)):
                    image_id = global_idx
                    global_idx += 1

                    pil = tensor_to_pil(x[j])
                    pil2 = transform_fn(pil, level)

                    results = model.predict(source=pil2, verbose=False)
                    probs = results[0].probs

                    pred = int(probs.top1)
                    conf = float(probs.top1conf.item())
                    true = int(y[j].item())

                    rows.append({
                        "model": "yolo26n_cls",
                        "dataset": "cifar10",
                        "axis": axis_name,
                        "transform": transform_name,
                        "level": level,
                        "image_id": image_id,
                        "true_label": true,
                        "true_label_name": CLASS_ORDER[true],
                        "pred_label": pred,
                        "pred_label_name": CLASS_ORDER[pred],
                        "confidence": conf,
                        "correct": int(pred == true),
                    })

    out_path = RESULTS_DIR / f"corruptions_{axis_name}_yolo26.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

def summarize_axis(axis_name):
    df = pd.read_csv(RESULTS_DIR / f"corruptions_{axis_name}_yolo26.csv")

    summary = (
        df.groupby(["transform", "level"])
        .apply(lambda d: pd.Series({
            "accuracy": accuracy_score(d["true_label"], d["pred_label"]),
            "macro_f1": f1_score(d["true_label"], d["pred_label"], average="macro"),
            "mean_confidence": d["confidence"].mean(),
            "n": len(d),
        }))
        .reset_index()
    )

    out_path = RESULTS_DIR / f"summary_corruptions_{axis_name}_yolo26.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

def main():
    run_axis("appearance", {
        "pixelate": pixelate,
        "blur": blur_pil,
    })

    run_axis("geometry", {
        "rotate": rotate_pil,
        "shear_x": shear_x_pil,
        "translate": translate_pil,
    })

    summarize_axis("appearance")
    summarize_axis("geometry")

if __name__ == "__main__":
    main()