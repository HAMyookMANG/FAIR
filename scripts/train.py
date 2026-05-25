import argparse
from pathlib import Path

from fair_attributor.config import FAIRConfig
from fair_attributor.pipeline import train_from_raw_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train FAIR hybrid attributor from raw_features.pkl.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root. Used for config bookkeeping.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--unknown-threshold", type=float, default=0.45)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = FAIRConfig(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        unknown_threshold=args.unknown_threshold,
    )
    train_from_raw_features(cfg)
