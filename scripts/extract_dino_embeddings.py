from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.data import cifar10_loader
from src.transforms import pixelate, blur_pil, rotate_pil, shear_x_pil, translate_pil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchvision import transforms as T

DINO_PREPROCESS = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def tensor_to_pil(x):
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def load_dinov2(device: str, model_name: str = "dinov2_vits14"):
    # downloads the repo + checkpoint into torch hub cache on first run
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()
    return model

@torch.no_grad()
def extract_dino(axis: str, levels=(0, 1, 2, 3), n=2000, model_name="dinov2_vits14"):
    model = load_dinov2(DEVICE, model_name=model_name)
    dl = cifar10_loader(n=n)

    if axis == "appearance":
        transforms = {"blur": blur_pil, "pixelate": pixelate}
    elif axis == "geometry":
        transforms = {"rotate": rotate_pil, "shear_x": shear_x_pil, "translate": translate_pil}
    else:
        raise ValueError("axis must be appearance or geometry")

    out_root = Path(f"results_n2000/embeddings_dino_{axis}")
    out_root.mkdir(parents=True, exist_ok=True)

    for tname, tfn in transforms.items():
        for level in levels:
            out_path = out_root / f"{tname}_L{level}.npy"
            if out_path.exists():
                print(f"Skip existing {out_path}")
                continue

            embs = []
            try:
                for _, (x, _) in enumerate(tqdm(dl, desc=f"DINO {axis} | {tname} | L{level}")):
                    for j in range(x.size(0)):
                        pil = tensor_to_pil(x[j])
                        pil2 = tfn(pil, level)
                        inp = DINO_PREPROCESS(pil2).unsqueeze(0).to(DEVICE)

                        feats = model.forward_features(inp)

                        # dinov2 forward_features returns dict with x_norm_clstoken
                        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                            vec = feats["x_norm_clstoken"]  # [1, d]
                        else:
                            # fallback: if forward_features returns a tensor [1, tokens, d], take CLS token
                            if torch.is_tensor(feats) and feats.ndim == 3:
                                vec = feats[:, 0, :]
                            else:
                                raise RuntimeError(f"Unexpected forward_features output type/shape: {type(feats)}")

                        vec = vec / vec.norm(dim=1, keepdim=True)
                        embs.append(vec.cpu().numpy()[0]) 

                embs = np.stack(embs, axis=0).astype(np.float32)
                np.save(out_path, embs)
                print(f"Saved {out_path} | shape={embs.shape}")

            except KeyboardInterrupt:
                print("\nInterrupted. No file written for this (transform, level). Re-run to resume.")
                return

if __name__ == "__main__":
    extract_dino("appearance")
    extract_dino("geometry")