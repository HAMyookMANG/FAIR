from dataclasses import dataclass
from pathlib import Path


@dataclass
class FAIRConfig:
    """Project-wide configuration."""

    dataset_dir: Path
    output_dir: Path
    artifact_save: Path | None = None
    raw_features_path: Path | None = None

    img_size_artifact: int = 256
    img_size_dino: int = 224
    img_size_clip: int = 224
    batch_size: int = 16
    unknown_threshold: float = 0.45
    test_size: float = 0.15
    val_size_from_remain: float = 0.1765
    seed: int = 42

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.artifact_save is None:
            self.artifact_save = self.output_dir / "hybrid_attributor_artifacts.pkl"
        else:
            self.artifact_save = Path(self.artifact_save)
        if self.raw_features_path is None:
            self.raw_features_path = self.output_dir / "raw_features.pkl"
        else:
            self.raw_features_path = Path(self.raw_features_path)
