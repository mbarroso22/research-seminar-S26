from __future__ import annotations

from pathlib import Path
from torchvision.datasets import CIFAR10
from category_config import CATEGORY_CONFIG

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "cifar_baseline"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

label_to_name = {v["cifar_label"]: k for k, v in CATEGORY_CONFIG.items()}

ds = CIFAR10(root=str(ROOT.parent / "data"), train=False, download=True)

counts = {k: 0 for k in CATEGORY_CONFIG}

for i, (img, y) in enumerate(ds):
    category = label_to_name[int(y)]
    out_dir = OUT_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"{category}_{i:05d}.png")
    counts[category] += 1

for k, v in counts.items():
    print(f"{k}: {v}")