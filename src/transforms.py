from __future__ import annotations
from PIL import Image, ImageFilter

def blur_pil(img: Image.Image, level: int) -> Image.Image:
    if level <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(level)))

def pixelate(img: Image.Image, level: int) -> Image.Image:
    if level <= 0:
        return img
    w, h = img.size
    factor = max(1, 2 ** level)
    resample = Image.Resampling.NEAREST
    small = img.resize((max(1, w // factor), max(1, h // factor)), resample=resample)
    return small.resize((w, h), resample=resample)

# ---------------- Geometry axis ----------------

def rotate_pil(img: Image.Image, level: int) -> Image.Image:
    """
    level 0: no rotation
    level 1: 10 degrees
    level 2: 20 degrees
    level 3: 30 degrees
    """
    if level <= 0:
        return img
    angles = {0: 0, 1: 30, 2: 60, 3: 90}
    angle = angles.get(level, 90)
    # expand=False keeps image size constant (important for preprocess)
    # fillcolor avoids black corners if supported by your Pillow version
    try:
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    except TypeError:
        # older Pillow without fillcolor
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

def shear_x_pil(img: Image.Image, level: int) -> Image.Image:
    """
    Simple X-shear using an affine transform.
    level 0: none
    levels map to shear factors: 0.0, 0.1, 0.2, 0.3
    """
    if level <= 0:
        return img
    shear = 0.1 * level
    w, h = img.size
    # affine matrix for x-shear: x' = x + shear*y
    matrix = (1, shear, 0,
              0, 1,     0)
    return img.transform((w, h), Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR)

def translate_pil(img: Image.Image, level: int) -> Image.Image:
    """
    Optional: translation as another geometry corruption.
    level 1..3 shift by 2,4,6 pixels.
    """
    if level <= 0:
        return img
    dx = 2 * level
    dy = 2 * level
    w, h = img.size
    matrix = (1, 0, dx,
              0, 1, dy)
    return img.transform((w, h), Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR)