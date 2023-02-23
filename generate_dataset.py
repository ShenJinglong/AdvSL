
import os
from PIL import Image
import random
import torch
import torchvision

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.Lambda(lambda img: torchvision.transforms.Grayscale(num_output_channels=3)(img) if img.mode == 'L' else img),
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.ConvertImageDtype(torch.float),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# domain = "clipart"
# domain = "real"
# domain = "sketch"
# domain = "quickdraw"
# domain = "infograph"
domain = "painting"
root_path = os.environ.get("ADVSL_DATASET_PATH") + f"domain-net/{domain}/"

with open(os.environ.get("ADVSL_DATASET_PATH") + f"domain-net/{domain}/{domain}_train.txt", "r") as f:
    lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:4000]

    imgs = []
    labels = []
    for i, line in enumerate(lines):
        print(i)
        path = line.strip().split(' ')[0]
        label = int(line.strip().split(' ')[1])
        with Image.open(root_path + path) as img:
            imgs.append(transforms(img))
        labels.append(label)
    torch.save({
        "imgs": torch.stack(imgs),
        "labels": torch.tensor(labels)
    }, f"domain-net_{domain}_train.pth")

with open(os.environ.get("ADVSL_DATASET_PATH") + f"domain-net/{domain}/{domain}_test.txt", "r") as f:
    lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:500]

    imgs = []
    labels = []
    for i, line in enumerate(lines):
        print(i)
        path = line.strip().split(' ')[0]
        label = int(line.strip().split(' ')[1])
        with Image.open(root_path + path) as img:
            imgs.append(transforms(img))
        labels.append(label)
    torch.save({
        "imgs": torch.stack(imgs),
        "labels": torch.tensor(labels)
    }, f"domain-net_{domain}_test.pth")