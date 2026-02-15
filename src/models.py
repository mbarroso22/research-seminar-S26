from __future__ import annotations
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import open_clip


def load_efficientnet(device: str):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights).to(device).eval()
    preprocess = weights.transforms()  # callable
    categories = weights.meta["categories"]
    return model, preprocess, categories


def load_openclip(device: str, model_name="ViT-B-32", pretrained="laion400m_e32"):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess_val, tokenizer
