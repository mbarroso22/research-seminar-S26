from __future__ import annotations
import torch

def load_dinov2(device: str, model_name: str = "dinov2_vits14"):
    # Common names: dinov2_vits14, dinov2_vitb14, dinov2_vitl14
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()
    return model