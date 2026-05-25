# FAIR: Hybrid AI Image Attributor

FAIR is a hybrid image attribution pipeline for classifying the source model or tool of AI-generated images. This repository reorganizes the original research notebook into a cleaner Python package structure with reusable modules and command-line scripts.

## Overview

The pipeline combines handcrafted image artifact features with deep visual embeddings and an ensemble classifier.

- **Artifact features**: RGB statistics, histograms, FFT frequency features, edge features, GLCM texture features, blockiness, and noise residual features
- **DINOv2 embeddings**: visual representation features extracted from a pretrained DINOv2 model
- **OpenCLIP embeddings**: visual features extracted from a pretrained OpenCLIP model
- **Hybrid classifier**: probability-averaged ensemble of Logistic Regression and XGBoost
- **Unknown handling**: predictions below a confidence threshold are returned as `unknown`
- **Additional experiments**: feature ablation, family-level attribution, confidence-threshold sweep, post-processing robustness, and open-set/OOD detection

## Repository Structure

```text
FAIR/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── FAIR4_0_4.ipynb
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   ├── predict.py
│   └── run_all.py
└── src/
    └── fair_attributor/
        ├── __init__.py
        ├── config.py
        ├── utils.py
        ├── features.py
        ├── embeddings.py
        ├── model.py
        ├── pipeline.py
        └── experiments.py
```

## Installation

A GPU environment is recommended. The code can be run on Google Colab or on a local machine with CUDA configured.

```bash
git clone <REPOSITORY_URL>
cd FAIR

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

If your environment requires a specific PyTorch CUDA wheel, install PyTorch first and then install the remaining requirements.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Dataset Format

The dataset should be organized by class name. Each class folder contains image files for that generator or tool.

```text
finalDataset/
├── biggan/
│   ├── image_001.png
│   └── ...
├── progan/
├── stylegan3/
├── sd2.0/
├── dalle3/
├── midjourney/
└── firefly/
```

Supported image extensions include `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, and `.tiff`.

## Usage

### 1. Extract Features

```bash
python scripts/extract_features.py \
  --dataset-dir /path/to/finalDataset \
  --output-dir outputs \
  --batch-size 16 \
  --n-jobs 4
```

This creates:

```text
outputs/raw_features.pkl
```

### 2. Train and Evaluate

```bash
python scripts/train.py \
  --dataset-dir /path/to/finalDataset \
  --output-dir outputs \
  --unknown-threshold 0.45
```

This creates:

```text
outputs/hybrid_attributor_artifacts.pkl
```

### 3. Run the Full Pipeline

```bash
python scripts/run_all.py \
  --dataset-dir /path/to/finalDataset \
  --output-dir outputs \
  --batch-size 16 \
  --n-jobs 4
```

### 4. Predict a Single Image

```bash
python scripts/predict.py \
  --image /path/to/test_image.png \
  --bundle outputs/hybrid_attributor_artifacts.pkl
```

Example output:

```json
{
  "image_path": "/path/to/test_image.png",
  "predicted_label": "midjourney",
  "raw_top_label": "midjourney",
  "confidence": 0.82,
  "probabilities": {
    "midjourney": 0.82,
    "dalle3": 0.09
  }
}
```

## Using on Google Colab

The original notebook is preserved in `notebooks/FAIR4_0_4.ipynb`.

When using Colab, mount Google Drive and pass your dataset path to `--dataset-dir`.

```bash
python scripts/run_all.py \
  --dataset-dir /content/drive/MyDrive/<DATASET_FOLDER>/finalDataset \
  --output-dir /content/drive/MyDrive/<OUTPUT_FOLDER>/outputs
```

## Notes

- DINOv2 is loaded with `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`, so the first run requires an internet connection.
- OpenCLIP `ViT-H-14` is a large model. If GPU memory is limited, reduce `--batch-size`.
- Generated files such as `outputs/`, `*.pkl`, and `*.csv` are excluded by `.gitignore`.
- Do not upload private datasets, model checkpoints, or large generated result files unless you have confirmed storage, licensing, and privacy requirements.
- Paths in the examples are placeholders. Replace them with your own local or Colab paths.

## Citation / Acknowledgement

This project uses DINOv2, OpenCLIP, scikit-learn, XGBoost, and scikit-image for hybrid feature extraction and classification.
