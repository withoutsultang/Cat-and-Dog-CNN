import argparse
from pathlib import Path

import torch
from PIL import Image

from model import ResNet18Binary
from utils import transform


def main():
    parser = argparse.ArgumentParser(description="Classify an image as a cat or dog")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="Checkpoint created by train.py")
    args = parser.parse_args()

    model = ResNet18Binary(pretrained=False)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        dog_score = model(tensor).item()
    label = "dog" if dog_score > 0.5 else "cat"
    print(f"{label} (dog score: {dog_score:.4f})")


if __name__ == "__main__":
    main()
