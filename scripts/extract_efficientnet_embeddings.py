from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.data import cifar10_loader
from src.models import load_efficientnet
from src.transforms import pixelate, blur_pil, rotate_pil, shear_x_pil, translate_pil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def extract_efficientnet_embeddings(axis: str, levels=(0, 1, 2, 3), n=2000):
    """
    Saves penultimate-layer embeddings (before classifier) for EfficientNetV2-S.
    Output: .npy arrays shaped (n, d).
    """
    model, preprocess, _ = load_efficientnet(DEVICE)
    dl = cifar10_loader(n=n)

    if axis == "appearance":
        transforms = {"blur": blur_pil, "pixelate": pixelate}
    elif axis == "geometry":
        transforms = {"rotate": rotate_pil, "shear_x": shear_x_pil, "translate": translate_pil}
    else:
        raise ValueError("axis must be 'appearance' or 'geometry'")

    out_root = Path(f"results_n2000/embeddings_efficientnet_{axis}")
    out_root.mkdir(parents=True, exist_ok=True)

    # EfficientNetV2 forward_features returns conv features; we pool to embedding vector.
    # This matches how torchvision builds logits: features -> avgpool -> flatten -> classifier
    for transform_name, transform_fn in transforms.items():
        for level in levels:
            embeddings = []
            for _, (x, _) in enumerate(tqdm(dl, desc=f"EffNet {axis} | {transform_name} | L{level}")):
                for j in range(x.size(0)):
                    pil = tensor_to_pil(x[j])
                    pil2 = transform_fn(pil, level)
                    inp = preprocess(pil2).unsqueeze(0).to(DEVICE)

                    feats = model.features(inp)            # [1, C, H, W]
                    pooled = model.avgpool(feats)          # [1, C, 1, 1]
                    vec = torch.flatten(pooled, 1)         # [1, C]
                    vec = vec / vec.norm(dim=1, keepdim=True)  # normalize so cosine is meaningful

                    embeddings.append(vec.cpu().numpy()[0])

            emb = np.stack(embeddings, axis=0).astype(np.float32)
            out_path = out_root / f"{transform_name}_L{level}.npy"
            np.save(out_path, emb)
            print(f"Saved {out_path} | shape={emb.shape} | dtype={emb.dtype}")


if __name__ == "__main__":
    extract_efficientnet_embeddings("appearance")
    extract_efficientnet_embeddings("geometry")