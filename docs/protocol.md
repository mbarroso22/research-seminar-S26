# Standard Test Protocol (Draft)

## Models
- EfficientNetV2 (torchvision, pretrained)
- OpenCLIP (open_clip_torch, pretrained)
- DINOv3 (planned; feature extractor; evaluated via kNN/linear probe)

## Input preprocessing
- Resize: 224x224 (for EfficientNet / CLIP)
- Normalization:
  - EfficientNet: ImageNet weights transforms (torchvision weights transforms)
  - CLIP: open_clip preprocess pipeline

## Datasets (for pipeline testing)
- CIFAR-10 (debug + baseline pipeline)

## Corruption suite (initial)
- Blur: levels [0, 1, 2, 3]
- Pixelation: levels [0, 1, 2, 3]
(extend later)

## Adversarial suite (initial, ART)
- FGSM: eps [0.0, 0.005, 0.01, 0.02]
(extend to PGD later)

## Metrics logged
- top-1 prediction
- confidence (max softmax / similarity)
- correct (if label available)
- model, dataset, transform, level/epsilon, seed
