import argparse
from pathlib import Path

from fair_attributor.config import FAIRConfig
from fair_attributor.pipeline import extract_and_save_raw_features, train_from_raw_features


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature extraction and training end-to-end.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = FAIRConfig(dataset_dir=Path(args.dataset_dir), output_dir=Path(args.output_dir), batch_size=args.batch_size)
    extract_and_save_raw_features(cfg, n_jobs=args.n_jobs)
    train_from_raw_features(cfg)
