from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

OUT_DIR = Path("../cifar_baseline/car")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 label index: 1 = automobile
TARGET_LABEL = 1

ds = CIFAR10(root="../../data", train=False, download=True)

count = 0
for i, (img, y) in enumerate(ds):
    if y != TARGET_LABEL:
        continue
    # img is PIL.Image
    img.save(OUT_DIR / f"cifar_auto_{i:05d}.png")
    count += 1

print("Saved", count, "CIFAR automobile images to", OUT_DIR.resolve())