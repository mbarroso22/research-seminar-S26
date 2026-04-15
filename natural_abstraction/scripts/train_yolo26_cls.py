from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "natural_abstraction" / "yolo_cls_dataset"

def main():
    model = YOLO("yolo26n-cls.pt")

    results = model.train(
        data=str(DATASET_DIR),
        epochs=30,
        imgsz=224,
        batch=32,
        device="cpu",      # change to "cpu" if needed
        project=str(ROOT / "natural_abstraction" / "yolo_runs"),
        name="yolo26n_cls_mirrored",
        pretrained=True,
    )

    print(results)

if __name__ == "__main__":
    main()