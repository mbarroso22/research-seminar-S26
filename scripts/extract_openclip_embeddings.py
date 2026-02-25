from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.data import cifar10_loader
from src.models import load_openclip
from src.transforms import pixelate, blur_pil, rotate_pil, shear_x_pil, translate_pil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]

def tensor_to_pil(x):
    arr = (x.permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
    return Image.fromarray(arr)


def extract_embeddings(axis="appearance", levels=(0,1,2,3), n=2000):

    model, preprocess, tokenizer = load_openclip(DEVICE)
    dl = cifar10_loader(n=n)

    if axis == "appearance":
        transforms = {
            "blur": blur_pil,
            "pixelate": pixelate,
        }
    elif axis == "geometry":
        transforms = {
            "rotate": rotate_pil,
            "shear_x": shear_x_pil,
            "translate": translate_pil,
        }
    else:
        raise ValueError("axis must be appearance or geometry")

    out_root = Path(f"results_n2000/embeddings_openclip_{axis}")
    out_root.mkdir(parents=True, exist_ok=True)

    for transform_name, transform_fn in transforms.items():
        for level in levels:

            embeddings = []
            global_idx = 0

            for _, (x, _) in enumerate(
                tqdm(dl, desc=f"{axis} | {transform_name} | L{level}")
            ):
                for j in range(x.size(0)):
                    pil = tensor_to_pil(x[j])
                    pil2 = transform_fn(pil, level)

                    image_inp = preprocess(pil2).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        image_features = model.encode_image(image_inp)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    embeddings.append(image_features.cpu().numpy()[0])
                    global_idx += 1

            embeddings = np.stack(embeddings, axis=0)

            out_path = out_root / f"{transform_name}_L{level}.npy"
            np.save(out_path, embeddings)

            print(f"Saved {out_path} | shape={embeddings.shape}")


if __name__ == "__main__":
    extract_embeddings(axis="appearance")
    extract_embeddings(axis="geometry")