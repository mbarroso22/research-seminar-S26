from __future__ import annotations

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from category_config import CATEGORY_CONFIG

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class GalleryCategoryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for category, cfg in CATEGORY_CONFIG.items():
            cat_dir = self.root_dir / category
            print(f"Checking category folder: {cat_dir}")

            if not cat_dir.exists():
                print(f"  missing: {cat_dir}")
                continue

            cat_total = 0

            for subtype_dir in sorted(cat_dir.iterdir()):
                if not subtype_dir.is_dir():
                    continue

                subtype = subtype_dir.name
                subtype_count = 0

                for img_path in sorted(subtype_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED:
                        self.samples.append({
                            "img_path": img_path,
                            "label": cfg["cifar_label"],
                            "category": category,
                            "subtype": subtype,
                        })
                        subtype_count += 1
                        cat_total += 1

                print(f"  subtype {subtype}: {subtype_count} images")

            print(f"  total for {category}: {cat_total}")

        print(f"\nTotal dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return (
            img,
            sample["label"],
            sample["category"],
            sample["subtype"],
            str(sample["img_path"]),
        )