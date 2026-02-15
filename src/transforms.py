# src/transforms.py
from __future__ import annotations
from PIL import Image, ImageFilter


def blur_pil(img: Image.Image, level: int) -> Image.Image:
    # level 0 = no blur; higher = more blur
    if level <= 0:
        return img
    # GaussianBlur radius in pixels
    return img.filter(ImageFilter.GaussianBlur(radius=float(level)))


def pixelate(img: Image.Image, level: int) -> Image.Image:
    if level <= 0:
        return img
    w, h = img.size
    factor = max(1, 2 ** level)
    resample = Image.Resampling.NEAREST
    small = img.resize((max(1, w // factor), max(1, h // factor)), resample=resample)
    return small.resize((w, h), resample=resample)
