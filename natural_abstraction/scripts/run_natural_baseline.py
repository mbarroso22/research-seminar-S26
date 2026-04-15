from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from src.models import load_efficientnet, load_openclip
from gallery_dataset import GalleryCategoryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NA_ROOT = Path(__file__).resolve().parent.parent
FILTERED_DIR = PROJECT_ROOT / "filtered"
RESULTS_DIR = NA_ROOT / "results_natural"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

def run_efficientnet():
    model, preprocess, categories = load_efficientnet(DEVICE)
    ds = GalleryCategoryDataset(FILTERED_DIR, transform=None)

    rows = []

    for idx in tqdm(range(len(ds)), desc="EfficientNet natural baseline"):
        img, label, category, subtype, img_path = ds[idx]
        inp = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)[0]

        pred = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())

        rows.append({
            "model": "efficientnet_v2_s",
            "dataset": "natural_gallery",
            "image_id": idx,
            "image_path": img_path,
            "true_label": int(label),
            "true_label_name": category,
            "subtype": subtype,
            "pred_label": pred,
            "pred_label_name": categories[pred],
            "confidence": conf,
        })

    out_path = RESULTS_DIR / "baseline_natural_efficientnet.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

def run_openclip():
    model, preprocess, tokenizer = load_openclip(DEVICE)
    ds = GalleryCategoryDataset(FILTERED_DIR, transform=None)

    texts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    text_tokens = tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []

    for idx in tqdm(range(len(ds)), desc="OpenCLIP natural baseline"):
        img, label, category, subtype, img_path = ds[idx]
        inp = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(inp)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T)[0]
            probs = torch.softmax(sims, dim=0)

        pred = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())

        rows.append({
            "model": "openclip_vit_b_32",
            "dataset": "natural_gallery",
            "image_id": idx,
            "image_path": img_path,
            "true_label": int(label),
            "true_label_name": category,
            "subtype": subtype,
            "pred_label": pred,
            "pred_label_name": CIFAR10_CLASSES[pred],
            "confidence": conf,
            "correct": int(pred == int(label)),
        })

    out_path = RESULTS_DIR / "baseline_natural_openclip.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    run_efficientnet()
    run_openclip()