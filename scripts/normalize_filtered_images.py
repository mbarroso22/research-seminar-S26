from __future__ import annotations

from pathlib import Path
from PIL import Image
from category_config import CATEGORY_CONFIG

ROOT = Path(__file__).resolve().parent.parent
FILTERED_DIR = ROOT / "filtered"

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for category in CATEGORY_CONFIG:
    src_dir = FILTERED_DIR / category
    out_dir = FILTERED_DIR / f"{category}_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"Missing folder: {src_dir}")
        continue

    count = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in SUPPORTED:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            out_path = out_dir / f"{img_path.stem}.jpg"
            img.save(out_path, "JPEG", quality=95)
            count += 1
        except Exception as e:
            print(f"skip {img_path.name}: {e}")

    print(f"{category}: normalized {count} images -> {out_dir}")