from __future__ import annotations
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np

from src.data import cifar10_loader
from src.models import load_efficientnet, load_openclip
from src.log import write_rows_csv
from src.transforms import pixelate, blur_pil

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


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# ==========================================================
# EfficientNet Appearance Corruptions
# ==========================================================

def run_appearance_corruptions(levels=(0, 1, 2, 3), n=32):
    model, preprocess, categories = load_efficientnet(DEVICE)
    dl = cifar10_loader(n=n)

    transforms = {
        "pixelate": pixelate,
        "blur": blur_pil,
    }

    rows = []

    for transform_name, transform_fn in transforms.items():
        for level in levels:
            for batch_i, (x, y) in enumerate(
                tqdm(dl, desc=f"EfficientNet {transform_name} level {level}")
            ):
                batch_size = x.size(0)

                for j in range(batch_size):
                    # Stable image_id across all transforms/levels
                    image_id = batch_i * batch_size + j

                    pil = tensor_to_pil(x[j])
                    pil2 = transform_fn(pil, level)

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
                            "axis": "appearance",
                            "transform": transform_name,
                            "level": level,
                            "image_id": image_id,
                            "true_label": true,
                            "true_label_name": CIFAR10_CLASSES[true],
                            "pred_label": pred,
                            "pred_label_name": categories[pred],
                            "confidence": conf,
                        }
                    )

    write_rows_csv(
        Path("results/corruptions_appearance_efficientnet.csv"), rows
    )


# ==========================================================
# OpenCLIP Appearance Corruptions
# ==========================================================

def run_appearance_corruptions_openclip(levels=(0, 1, 2, 3), n=32):
    model, preprocess, tokenizer = load_openclip(DEVICE)
    dl = cifar10_loader(n=n)

    transforms = {
        "pixelate": pixelate,
        "blur": blur_pil,
    }

    texts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    text_tokens = tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []

    for transform_name, transform_fn in transforms.items():
        for level in levels:
            for batch_i, (x, y) in enumerate(
                tqdm(dl, desc=f"OpenCLIP {transform_name} level {level}")
            ):
                batch_size = x.size(0)

                for j in range(batch_size):
                    # Same stable ID logic
                    image_id = batch_i * batch_size + j

                    pil = tensor_to_pil(x[j])
                    pil2 = transform_fn(pil, level)

                    image_inp = preprocess(pil2).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        image_features = model.encode_image(image_inp)
                        image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )
                        sims = (image_features @ text_features.T)[0]
                        probs = torch.softmax(sims, dim=0)

                    pred = int(torch.argmax(probs).item())
                    conf = float(torch.max(probs).item())
                    true = int(y[j].item())

                    rows.append(
                        {
                            "model": "openclip_vit_b_32",
                            "dataset": "cifar10",
                            "axis": "appearance",
                            "transform": transform_name,
                            "level": level,
                            "image_id": image_id,
                            "true_label": true,
                            "true_label_name": CIFAR10_CLASSES[true],
                            "pred_label": pred,
                            "pred_label_name": CIFAR10_CLASSES[pred],
                            "confidence": conf,
                            "correct": int(pred == true),
                        }
                    )

    write_rows_csv(
        Path("results/corruptions_appearance_openclip.csv"), rows
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    run_appearance_corruptions()
    run_appearance_corruptions_openclip()
    print("Wrote appearance corruption CSVs (EfficientNet + OpenCLIP)")
