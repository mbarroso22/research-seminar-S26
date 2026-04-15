from __future__ import annotations

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from category_config import CATEGORY_CONFIG

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class CIFARFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for category, cfg in CATEGORY_CONFIG.items():
            class_dir = self.root_dir / category
            print(f"Checking CIFAR baseline folder: {class_dir}")

            if not class_dir.exists():
                print(f"  missing: {class_dir}")
                continue

            count = 0
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED:
                    self.samples.append((img_path, cfg["cifar_label"]))
                    count += 1

            print(f"  found {count} images")

        print(f"\nTotal CIFAR baseline dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(img_path)