# FAIR: Hybrid AI Image Attributor

FAIR는 AI 생성 이미지의 생성 모델/도구를 분류하기 위한 하이브리드 이미지 attribution 파이프라인입니다. 기존 Colab 노트북을 GitHub에서 보기 쉽게 실행할 수 있도록 Python 모듈과 CLI 스크립트로 정리했습니다.

## 핵심 아이디어

- **Artifact features**: RGB 통계, 히스토그램, FFT, edge, GLCM, blockiness, noise residual 등 handcrafted feature 추출
- **DINOv2 embedding**: 이미지의 시각적 표현 학습 feature 추출
- **OpenCLIP embedding**: 이미지-텍스트 사전학습 모델 기반 feature 추출
- **Hybrid classifier**: Logistic Regression + XGBoost 확률 평균 ensemble
- **Unknown 처리**: confidence threshold 이하 예측은 `unknown`으로 처리
- **추가 실험**: feature ablation, family-level attribution, confidence threshold sweep, post-processing robustness, open-set/OOD 실험

## Repository 구조

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

## 설치

GPU 환경을 권장합니다. Colab 또는 CUDA가 설정된 로컬 환경에서 실행하세요.

```bash
git clone https://github.com/HAMyookMANG/FAIR.git
cd FAIR

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

PyTorch CUDA wheel을 직접 지정해야 하는 환경이면 아래처럼 먼저 설치한 뒤 requirements를 설치하세요.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Dataset 형식

데이터셋 폴더는 class별 하위 폴더 구조를 사용합니다.

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

## 실행 방법

### 1. Feature 추출

```bash
python scripts/extract_features.py \
  --dataset-dir /path/to/finalDataset \
  --output-dir outputs \
  --batch-size 16 \
  --n-jobs 4
```

생성 파일:

```text
outputs/raw_features.pkl
```

### 2. 학습 및 평가

```bash
python scripts/train.py \
  --dataset-dir /path/to/finalDataset \
  --output-dir outputs \
  --unknown-threshold 0.45
```

생성 파일:

```text
outputs/hybrid_attributor_artifacts.pkl
```

### 3. 전체 파이프라인 한 번에 실행

```bash
python scripts/run_all.py \
  --dataset-dir /path/to/finalDataset \
  --output-dir outputs \
  --batch-size 16 \
  --n-jobs 4
```

### 4. 단일 이미지 예측

```bash
python scripts/predict.py \
  --image /path/to/test_image.png \
  --bundle outputs/hybrid_attributor_artifacts.pkl
```

출력 예시:

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

## Colab에서 사용하기

기존 노트북은 `notebooks/FAIR4_0_4.ipynb`에 보관되어 있습니다. Colab에서는 Google Drive를 mount한 뒤 `dataset-dir`만 Drive 경로로 지정하면 됩니다.

```bash
python scripts/run_all.py \
  --dataset-dir /content/drive/MyDrive/window_AIDetector/finalDataset \
  --output-dir /content/drive/MyDrive/window_AIDetector/outputs
```

## 주의사항

- DINOv2는 `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`를 사용하므로 첫 실행 시 인터넷 연결이 필요합니다.
- OpenCLIP `ViT-H-14` 모델은 크기가 크므로 GPU 메모리가 부족하면 batch size를 줄이세요.
- `outputs/`, `*.pkl`, `*.csv`는 `.gitignore`에 포함되어 있어 GitHub에 올라가지 않습니다.
- 데이터셋 이미지 원본은 용량과 라이선스 문제를 확인한 뒤 업로드 여부를 결정하세요.

## Citation / Acknowledgement

This project uses DINOv2, OpenCLIP, scikit-learn, XGBoost, and scikit-image for hybrid feature extraction and classification.
