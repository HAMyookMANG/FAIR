import argparse
import json
from pathlib import Path

from fair_attributor.embeddings import build_dino_transform, load_clip_model, load_dino_model
from fair_attributor.model import predict_image
from fair_attributor.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Predict generator attribution for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--bundle", default="outputs/hybrid_attributor_artifacts.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    device = get_device()
    dino_transform = build_dino_transform(224)
    dino_model = load_dino_model(device)
    clip_model, clip_preprocess = load_clip_model(device)
    result = predict_image(Path(args.image), Path(args.bundle), dino_model, clip_model, clip_preprocess, dino_transform, device)
    print(json.dumps(result, indent=2, ensure_ascii=False))
