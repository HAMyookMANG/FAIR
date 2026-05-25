import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_image_files(root_dir: str | Path, exts: tuple[str, ...] = IMAGE_EXTS):
    root_dir = Path(root_dir)
    items, labels = [], []
    for class_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        count = 0
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                items.append(str(path))
                labels.append(class_dir.name)
                count += 1
        print(f"[INFO] {class_dir.name}: {count} images")
    return items, labels


def pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def np_rgb(path: str | Path, size: int = 256) -> np.ndarray:
    img = pil_rgb(path).resize((size, size))
    return np.array(img)
