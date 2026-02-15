from __future__ import annotations
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np

from src.data import cifar10_loader
from src.models import load_efficientnet
from src.log import write_rows_csv
from src.transforms import pixelate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR10_CLASSES = [
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


def tensor_to_pil(x):
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def run_pixelate(levels=(0, 1, 2, 3)):
    model, preprocess, categories = load_efficientnet(DEVICE)
    dl = cifar10_loader(n=32)
    rows = []
    img_counter = 0
    for level in levels:
        for batch_i, (x, y) in enumerate(tqdm(dl, desc=f"Pixelate level {level}")):
            for j in range(x.size(0)):
                pil = tensor_to_pil(x[j])
                pil2 = pixelate(pil, level)
                inp = preprocess(pil2).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(inp)
                    probs = F.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())
                conf = float(torch.max(probs).item())
                true = int(y[j].item())
                rows.append(
                    {
                        "model": "efficientnet_v2_s",
                        "dataset": "cifar10",
                        "transform": "pixelate",
                        "level": level,
                        "image_id": img_counter,
                        "true_label": true,
                        "true_label_name": CIFAR10_CLASSES[true],
                        "pred_label": pred,
                        "pred_label_name": categories[pred],
                        "confidence": conf,
                    }
                )
            img_counter += 1
    write_rows_csv(Path("results/corruptions_pixelate_efficientnet.csv"), rows)


if __name__ == "__main__":
    run_pixelate()
    print("Wrote results/corruptions_pixelate_efficientnet.csv")
