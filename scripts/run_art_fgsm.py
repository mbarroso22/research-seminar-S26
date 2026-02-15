from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from src.data import cifar10_loader
from src.models import load_efficientnet
from src.log import write_rows_csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tensor_to_pil(x):
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def main(eps_list=(0.0, 0.005, 0.01, 0.02), n=32):
    model, preprocess, _ = load_efficientnet(DEVICE)

    # ART expects numpy arrays; we'll feed already-preprocessed tensors converted to numpy.
    # We'll wrap EfficientNet in ART's PyTorchClassifier.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0
    )  # dummy optimizer (not training)

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
        device_type="gpu" if DEVICE == "cuda" else "cpu",
    )

    dl = cifar10_loader(n=n)
    # Build a small batch dataset in model-input space
    xs = []
    ys = []
    for x, y in dl:
        for j in range(x.size(0)):
            pil = tensor_to_pil(x[j])
            inp = preprocess(
                pil
            )  # tensor 3x224x224 in [0,1] normalized appropriately by weights.transforms()
            # weights.transforms() returns normalized float tensor; ART clip_values expects [0,1].
            # For strict ART usage, you may want a non-normalized input pipeline.
            # For tomorrow: use this to demonstrate workflow; later we can align normalization properly.
            xs.append(inp.numpy())
            ys.append(int(y[j].item()))
    x_np = np.stack(xs, axis=0).astype(np.float32)
    y_np = np.array(ys, dtype=np.int64)

    rows = []
    for eps in eps_list:
        if eps == 0.0:
            x_adv = x_np
        else:
            attack = FastGradientMethod(estimator=classifier, eps=eps)
            x_adv = attack.generate(x=x_np)

        preds = classifier.predict(x_adv)
        pred_labels = np.argmax(preds, axis=1)
        acc = float(np.mean(pred_labels == y_np))
        rows.append(
            {
                "model": "efficientnet_v2_s",
                "dataset": "cifar10",
                "attack": "fgsm",
                "epsilon": eps,
                "n": len(y_np),
                "accuracy": acc,
            }
        )

    write_rows_csv(Path("results/art_fgsm_summary.csv"), rows)
    print("Wrote results/art_fgsm_summary.csv")


if __name__ == "__main__":
    main()
