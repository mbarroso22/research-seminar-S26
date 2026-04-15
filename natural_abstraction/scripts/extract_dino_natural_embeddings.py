from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from gallery_dataset import GalleryCategoryDataset
from cifar_folder_dataset import CIFARFolderDataset
from src.models_dino import load_dinov2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NA_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = NA_ROOT / "results_natural"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_DIR = PROJECT_ROOT / "filtered"
CIFAR_BASELINE_DIR = NA_ROOT / "cifar_baseline"

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

# DINOv2 preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def unpack_dino_output(feat):
    """
    Make DINO output handling robust across possible return formats.
    """
    if isinstance(feat, dict):
        if "x_norm_clstoken" in feat:
            feat = feat["x_norm_clstoken"]
        elif "x_norm_patchtokens" in feat:
            feat = feat["x_norm_patchtokens"].mean(dim=1)
        else:
            raise ValueError(f"Unexpected DINO output keys: {feat.keys()}")
    elif isinstance(feat, (tuple, list)):
        feat = feat[0]
    return feat


@torch.no_grad()
def extract_natural_embeddings(dataset, preprocess, model, out_prefix: str):
    rows = []
    embeddings = []

    for idx in tqdm(range(len(dataset)), desc=f"Extract DINO embeddings: {out_prefix}"):
        img, label, category, subtype, img_path = dataset[idx]
        inp = preprocess(img).unsqueeze(0).to(DEVICE)

        feat = model(inp)
        feat = unpack_dino_output(feat)

        vec = feat[0].detach().cpu().numpy()

        embeddings.append(vec)
        rows.append({
            "image_id": idx,
            "image_path": img_path,
            "true_label": int(label),
            "true_label_name": category,
            "subtype": subtype,
        })

    emb = np.stack(embeddings, axis=0).astype(np.float32)

    emb_path = RESULTS_DIR / f"{out_prefix}_dino_embeddings.npy"
    meta_path = RESULTS_DIR / f"{out_prefix}_dino_metadata.csv"

    np.save(emb_path, emb)
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    print(f"Wrote {emb_path}")
    print(f"Wrote {meta_path}")
    print(f"Shape: {emb.shape}")


@torch.no_grad()
def extract_cifar_embeddings(dataset, preprocess, model, out_prefix: str):
    rows = []
    embeddings = []

    for idx in tqdm(range(len(dataset)), desc=f"Extract DINO embeddings: {out_prefix}"):
        img, label, img_path = dataset[idx]
        inp = preprocess(img).unsqueeze(0).to(DEVICE)

        feat = model(inp)
        feat = unpack_dino_output(feat)

        vec = feat[0].detach().cpu().numpy()

        embeddings.append(vec)
        rows.append({
            "image_id": idx,
            "image_path": img_path,
            "true_label": int(label),
            "true_label_name": CIFAR10_CLASSES[int(label)],
        })

    emb = np.stack(embeddings, axis=0).astype(np.float32)

    emb_path = RESULTS_DIR / f"{out_prefix}_dino_embeddings.npy"
    meta_path = RESULTS_DIR / f"{out_prefix}_dino_metadata.csv"

    np.save(emb_path, emb)
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    print(f"Wrote {emb_path}")
    print(f"Wrote {meta_path}")
    print(f"Shape: {emb.shape}")


def main():
    model = load_dinov2(DEVICE)
    model.eval()

    natural_ds = GalleryCategoryDataset(FILTERED_DIR, transform=None)
    cifar_ds = CIFARFolderDataset(CIFAR_BASELINE_DIR, transform=None)

    extract_natural_embeddings(natural_ds, preprocess, model, "natural")
    extract_cifar_embeddings(cifar_ds, preprocess, model, "cifar_baseline")


if __name__ == "__main__":
    main()