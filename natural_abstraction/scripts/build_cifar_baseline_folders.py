from pathlib import Path
import torchvision

OUT_DIR = Path("natural_abstraction/cifar_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

for c in CIFAR10_CLASSES:
    (OUT_DIR / c).mkdir(parents=True, exist_ok=True)

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)

print("Building CIFAR baseline folders...")

counts = {c: 0 for c in CIFAR10_CLASSES}

for i, (img, label) in enumerate(dataset):
    class_name = CIFAR10_CLASSES[label]
    save_path = OUT_DIR / class_name / f"{class_name}_{i}.png"
    img.save(save_path)
    counts[class_name] += 1

print("Done.\n")
for k, v in counts.items():
    print(f"{k}: {v}")