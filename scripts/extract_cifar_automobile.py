from pathlib import Path
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "cifar_baseline" / "automobile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LABEL = 1  # CIFAR-10 automobile
ds = CIFAR10(root=str(ROOT / "data"), train=False, download=True)

count = 0
for i, (img, y) in enumerate(ds):
    if y == TARGET_LABEL:
        img.save(OUT_DIR / f"cifar_auto_{i:05d}.png")
        count += 1

print("Saved", count, "CIFAR automobile images to", OUT_DIR)