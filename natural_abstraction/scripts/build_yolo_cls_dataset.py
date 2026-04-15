from __future__ import annotations

from pathlib import Path
import shutil
import random
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILTERED_DIR = ROOT / "filtered"
OUT_DIR = ROOT / "natural_abstraction" / "yolo_cls_dataset"
META_PATH = ROOT / "natural_abstraction" / "results_natural" / "yolo_cls_dataset_metadata.csv"

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

SUBTYPE_ORDER = ["real", "painting", "cartoon", "sketch", "abstract"]

random.seed(42)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    rows = []

    for split in ["train", "val", "test"]:
        for cls in CLASS_ORDER:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for label_idx, cls in enumerate(CLASS_ORDER):
        class_dir = FILTERED_DIR / cls
        if not class_dir.exists():
            print(f"Missing class dir: {class_dir}")
            continue

        all_images = []

        for subtype in SUBTYPE_ORDER:
            subtype_dir = class_dir / subtype
            if not subtype_dir.exists():
                continue

            files = [p for p in subtype_dir.iterdir() if p.is_file() and is_image_file(p)]
            files.sort()

            for p in files:
                all_images.append((p, subtype))

        random.shuffle(all_images)

        n = len(all_images)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val

        split_assignments = (
            [("train", x) for x in all_images[:n_train]]
            + [("val", x) for x in all_images[n_train:n_train + n_val]]
            + [("test", x) for x in all_images[n_train + n_val:]]
        )

        for idx_in_class, (split, (src, subtype)) in enumerate(split_assignments):
            # Keep destination names short and unique
            dst_name = f"{cls}_{subtype}_{idx_in_class:05d}{src.suffix.lower()}"
            dst = OUT_DIR / split / cls / dst_name

            try:
                shutil.copyfile(src, dst)
            except Exception as e:
                print(f"Failed copying:\n  src={src}\n  dst={dst}\n  error={e}")
                raise

            rows.append({
                "split": split,
                "image_path": str(dst),
                "true_label": label_idx,
                "true_label_name": cls,
                "subtype": subtype,
                "source_path": str(src),
            })

        print(f"{cls}: total={n}, train={n_train}, val={n_val}, test={n_test}")

    df = pd.DataFrame(rows)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META_PATH, index=False)

    print(f"\nWrote metadata to: {META_PATH}")
    print(f"Wrote dataset to: {OUT_DIR}")


if __name__ == "__main__":
    main()