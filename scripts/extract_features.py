import argparse
from pathlib import Path

from fair_attributor.config import FAIRConfig
from fair_attributor.pipeline import extract_and_save_raw_features


def parse_args():
    parser = argparse.ArgumentParser(description="Extract artifact, DINOv2, and OpenCLIP features.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root. Each class should be a subdirectory.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for raw_features.pkl and model artifacts.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip TTA for embeddings.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = FAIRConfig(dataset_dir=Path(args.dataset_dir), output_dir=Path(args.output_dir), batch_size=args.batch_size)
    extract_and_save_raw_features(cfg, n_jobs=args.n_jobs, tta=not args.no_tta)
