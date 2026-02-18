from __future__ import annotations
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.data import cifar10_loader
from src.models import load_efficientnet, load_openclip
from src.log import write_rows_csv
from pathlib import Path
from PIL import Image
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 class names
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
    # x: (3,H,W), in [0,1]
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def run_efficientnet():
    model, preprocess_fn, categories = load_efficientnet(DEVICE)
    dl = cifar10_loader(n=32)
    rows = []
    img_counter = 0

    for batch_i, (x, y) in enumerate(tqdm(dl, desc="EfficientNetV2 baseline")):
        for j in range(x.size(0)):
            pil = tensor_to_pil(x[j])
            inp = preprocess_fn(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1)[0]

            pred = int(torch.argmax(probs).item())
            conf = float(torch.max(probs).item())

            rows.append({
                "model": "efficientnet_v2_s",
                "dataset": "cifar10",
                "transform": "none",
                "level": 0,
                "image_id": img_counter,
                "true_label": int(y[j].item()),
                "pred_label": pred,
                "confidence": conf,
            })

            img_counter += 1   # <-- move here

    write_rows_csv(Path("results/baseline_efficientnet.csv"), rows)


def run_openclip():
    model, preprocess, tokenizer = load_openclip(DEVICE)
    dl = cifar10_loader(n=32)

    texts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    text_tokens = tokenizer(texts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []
    img_counter = 0

    for _, (x, y) in enumerate(tqdm(dl, desc="OpenCLIP baseline")):
        for j in range(x.size(0)):
            pil = tensor_to_pil(x[j])
            image_inp = preprocess(pil).unsqueeze(0).to(DEVICE)  # fixed name

            with torch.no_grad():
                image_features = model.encode_image(image_inp)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                sims = (image_features @ text_features.T)[0]
                probs = torch.softmax(sims, dim=0)

            pred = int(torch.argmax(probs).item())
            conf = float(torch.max(probs).item())
            true = int(y[j].item())

            rows.append({
                "model": "openclip_vit_b_32",
                "dataset": "cifar10",
                "transform": "none",
                "level": 0,
                "image_id": img_counter,
                "true_label": true,
                "true_label_name": CIFAR10_CLASSES[true],
                "pred_label": pred,
                "pred_label_name": CIFAR10_CLASSES[pred],
                "confidence": conf,
                "correct": int(pred == true),
            })

            img_counter += 1  # moved inside inner loop

    write_rows_csv(Path("results/baseline_openclip.csv"), rows)



if __name__ == "__main__":
    run_efficientnet()
    run_openclip()
    print("Wrote results/ baseline CSVs")
