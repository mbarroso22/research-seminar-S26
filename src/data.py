from __future__ import annotations
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


def cifar10_loader(root="data", train=False, n=32):
    # NOTE: CIFAR images are 32x32; we'll resize later per model.
    ds = CIFAR10(root=root, train=train, download=True, transform=transforms.ToTensor())
    idx = list(range(min(n, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=8, shuffle=False)
