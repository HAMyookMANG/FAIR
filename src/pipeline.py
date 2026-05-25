import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .embeddings import build_dino_transform, extract_embeddings_unified, load_clip_model, load_dino_model
from .features import extract_artifact_features
from .model import evaluate_split, get_ensemble_proba, preprocess_features, save_bundle, train_ensemble
from .utils import get_device, list_image_files, np_rgb, set_seed


def extract_and_save_raw_features(config, n_jobs: int = 4, tta: bool = True):
    set_seed(config.seed)
    device = get_device()
    print(f"[INFO] Device: {device}")

    image_paths, labels = list_image_files(config.dataset_dir)
    if not image_paths:
        raise RuntimeError(f"No image files found in: {config.dataset_dir}")

    def process_single_artifact(path):
        try:
            rgb = np_rgb(path, config.img_size_artifact)
            return extract_artifact_features(rgb)
        except Exception as exc:
            print(f"[WARN] Artifact extraction failed: {path} ({exc})")
            return None

    print("[INFO] Extracting artifact features...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_artifact)(p) for p in tqdm(image_paths, desc="Artifacts")
    )

    artifact_rows, valid_paths, valid_labels = [], [], []
    for p, label, feats in zip(image_paths, labels, results):
        if feats is not None:
            artifact_rows.append(feats)
            valid_paths.append(p)
            valid_labels.append(label)

    artifact_df = pd.DataFrame(artifact_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    artifact_feature_names = artifact_df.columns.tolist()
    X_art = artifact_df.values.astype(np.float32)

    print("[INFO] Loading DINOv2 and OpenCLIP...")
    dino_transform = build_dino_transform(config.img_size_dino)
    dino_model = load_dino_model(device)
    clip_model, clip_preprocess = load_clip_model(device)

    X_dino, X_clip = extract_embeddings_unified(
        valid_paths,
        dino_model=dino_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        dino_transform=dino_transform,
        device=device,
        batch_size=config.batch_size,
        tta=tta,
    )

    raw = {
        "X_art": X_art,
        "X_dino": X_dino,
        "X_clip": X_clip,
        "labels": valid_labels,
        "image_paths": valid_paths,
        "artifact_feature_names": artifact_feature_names,
    }
    joblib.dump(raw, config.raw_features_path)
    print(f"[SAVED] {config.raw_features_path}")
    return raw


def train_from_raw_features(config):
    set_seed(config.seed)
    raw = joblib.load(config.raw_features_path)
    le = LabelEncoder()
    y = le.fit_transform(raw["labels"])

    arrays, preprocessors = preprocess_features(raw["X_art"], raw["X_dino"], raw["X_clip"], y, config)
    print("[INFO] Final train shape:", arrays["X_train"].shape)

    clf_lr, clf_xgb = train_ensemble(
        arrays["X_train"], arrays["y_train"], arrays["X_val"], arrays["y_val"], len(le.classes_), config.seed
    )

    proba_val = get_ensemble_proba(arrays["X_val"], clf_lr, clf_xgb)
    val_result = evaluate_split(arrays["y_val"], proba_val, le, "VAL", config.unknown_threshold)

    proba_test = get_ensemble_proba(arrays["X_test"], clf_lr, clf_xgb)
    test_result = evaluate_split(arrays["y_test"], proba_test, le, "TEST", config.unknown_threshold)

    class_names = list(le.classes_)
    save_bundle(
        config.artifact_save,
        le,
        raw["artifact_feature_names"],
        preprocessors,
        (clf_lr, clf_xgb),
        config,
        class_names,
    )
    return {"val": val_result, "test": test_result, "arrays": arrays, "preprocessors": preprocessors}
