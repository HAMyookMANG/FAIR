import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier

from .embeddings import extract_single_embeddings
from .features import extract_artifact_features
from .utils import np_rgb


def preprocess_features(X_art, X_dino, X_clip, y, config):
    indices = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=config.test_size, random_state=config.seed, stratify=y
    )
    y_trainval = y[trainval_idx]
    train_idx_rel, val_idx_rel = train_test_split(
        np.arange(len(trainval_idx)),
        test_size=config.val_size_from_remain,
        random_state=config.seed,
        stratify=y_trainval,
    )
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    X_art_train, X_art_val, X_art_test = X_art[train_idx], X_art[val_idx], X_art[test_idx]
    X_dino_train, X_dino_val, X_dino_test = X_dino[train_idx], X_dino[val_idx], X_dino[test_idx]
    X_clip_train, X_clip_val, X_clip_test = X_clip[train_idx], X_clip[val_idx], X_clip[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    art_scaler = StandardScaler()
    X_art_train_s = art_scaler.fit_transform(X_art_train)
    X_art_val_s = art_scaler.transform(X_art_val)
    X_art_test_s = art_scaler.transform(X_art_test)

    dino_scaler = StandardScaler()
    X_dino_train_s = dino_scaler.fit_transform(X_dino_train)
    X_dino_val_s = dino_scaler.transform(X_dino_val)
    X_dino_test_s = dino_scaler.transform(X_dino_test)
    dino_pca_dim = min(256, X_dino_train_s.shape[0] - 1, X_dino_train_s.shape[1])
    dino_pca = PCA(n_components=dino_pca_dim, random_state=config.seed)
    X_dino_train_p = dino_pca.fit_transform(X_dino_train_s)
    X_dino_val_p = dino_pca.transform(X_dino_val_s)
    X_dino_test_p = dino_pca.transform(X_dino_test_s)

    clip_scaler = StandardScaler()
    X_clip_train_s = clip_scaler.fit_transform(X_clip_train)
    X_clip_val_s = clip_scaler.transform(X_clip_val)
    X_clip_test_s = clip_scaler.transform(X_clip_test)
    clip_pca_dim = min(256, X_clip_train_s.shape[0] - 1, X_clip_train_s.shape[1])
    clip_pca = PCA(n_components=clip_pca_dim, random_state=config.seed)
    X_clip_train_p = clip_pca.fit_transform(X_clip_train_s)
    X_clip_val_p = clip_pca.transform(X_clip_val_s)
    X_clip_test_p = clip_pca.transform(X_clip_test_s)

    X_train = np.concatenate([X_art_train_s, X_dino_train_p, X_clip_train_p], axis=1)
    X_val = np.concatenate([X_art_val_s, X_dino_val_p, X_clip_val_p], axis=1)
    X_test = np.concatenate([X_art_test_s, X_dino_test_p, X_clip_test_p], axis=1)

    preprocessors = {
        "art_scaler": art_scaler,
        "dino_scaler": dino_scaler,
        "dino_pca": dino_pca,
        "clip_scaler": clip_scaler,
        "clip_pca": clip_pca,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    arrays = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_art_train_s": X_art_train_s,
        "X_art_val_s": X_art_val_s,
        "X_art_test_s": X_art_test_s,
        "X_dino_train_p": X_dino_train_p,
        "X_dino_val_p": X_dino_val_p,
        "X_dino_test_p": X_dino_test_p,
        "X_clip_train_p": X_clip_train_p,
        "X_clip_val_p": X_clip_val_p,
        "X_clip_test_p": X_clip_test_p,
    }
    return arrays, preprocessors


def train_ensemble(X_train, y_train, X_val, y_val, num_classes: int, seed: int = 42):
    print("[INFO] Training LogisticRegression...")
    clf_lr = LogisticRegression(
        max_iter=5000,
        C=2.0,
        class_weight="balanced",
        solver="saga",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=seed,
    )
    clf_lr.fit(X_train, y_train)

    print("[INFO] Training XGBoost...")
    clf_xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        reg_lambda=2.0,
        random_state=seed,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    return clf_lr, clf_xgb


def get_ensemble_proba(X, clf_lr, clf_xgb):
    return 0.5 * clf_lr.predict_proba(X) + 0.5 * clf_xgb.predict_proba(X)


def evaluate_split(y_true, proba, label_encoder, split_name="VAL", unknown_threshold=0.45):
    class_names = list(label_encoder.classes_)
    y_pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "top2_accuracy": top_k_accuracy_score(y_true, proba, k=min(2, len(class_names)), labels=np.arange(len(class_names))),
        "top3_accuracy": top_k_accuracy_score(y_true, proba, k=min(3, len(class_names)), labels=np.arange(len(class_names))),
    }
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics.update({"macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro, "weighted_f1": f1_weighted})

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    try:
        metrics["roc_auc_macro_ovr"] = roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        metrics["roc_auc_macro_ovr"] = None

    accepted = conf >= unknown_threshold
    metrics["known_only_accuracy"] = accuracy_score(y_true[accepted], y_pred[accepted]) if np.any(accepted) else 0.0
    metrics["unknown_ratio"] = float(np.mean(~accepted))

    print("\n" + "=" * 70)
    print(f"[{split_name}] Metrics")
    print("=" * 70)
    for key, value in metrics.items():
        if value is not None:
            print(f"{key:24s}: {value:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return {"y_pred": y_pred, "confidence": conf, "metrics": metrics}


def save_bundle(path, label_encoder, artifact_feature_names, preprocessors, classifiers, config, class_names):
    bundle = {
        "label_encoder": label_encoder,
        "artifact_feature_names": artifact_feature_names,
        "img_size_artifact": config.img_size_artifact,
        "unknown_threshold": config.unknown_threshold,
        "class_names": class_names,
        **preprocessors,
        "clf_lr": classifiers[0],
        "clf_xgb": classifiers[1],
    }
    joblib.dump(bundle, path)
    print(f"[SAVED] {path}")


@lru_cache(maxsize=1)
def load_bundle(bundle_path):
    return joblib.load(bundle_path)


def build_single_feature_vector(image_path, bundle, dino_model, clip_model, clip_preprocess, dino_transform, device):
    rgb = np_rgb(image_path, bundle["img_size_artifact"])
    art = extract_artifact_features(rgb)
    art_df = pd.DataFrame([art]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in bundle["artifact_feature_names"]:
        if c not in art_df.columns:
            art_df[c] = 0.0
    art_vec = art_df[bundle["artifact_feature_names"]].values.astype(np.float32)

    dino_vec, clip_vec = extract_single_embeddings(
        image_path, dino_model, clip_model, clip_preprocess, dino_transform, device, tta=True
    )

    art_s = bundle["art_scaler"].transform(art_vec)
    dino_s = bundle["dino_scaler"].transform(dino_vec)
    dino_p = bundle["dino_pca"].transform(dino_s)
    clip_s = bundle["clip_scaler"].transform(clip_vec)
    clip_p = bundle["clip_pca"].transform(clip_s)
    return np.concatenate([art_s, dino_p, clip_p], axis=1)


def predict_image(image_path, bundle_path, dino_model, clip_model, clip_preprocess, dino_transform, device):
    bundle = load_bundle(str(bundle_path))
    le = bundle["label_encoder"]
    x = build_single_feature_vector(image_path, bundle, dino_model, clip_model, clip_preprocess, dino_transform, device)
    proba = get_ensemble_proba(x, bundle["clf_lr"], bundle["clf_xgb"])[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_idx])[0]
    conf = float(proba[pred_idx])
    final_label = pred_label if conf >= bundle["unknown_threshold"] else "unknown"
    probs = {le.inverse_transform([i])[0]: float(proba[i]) for i in range(len(proba))}
    return {
        "image_path": str(image_path),
        "predicted_label": final_label,
        "raw_top_label": pred_label,
        "confidence": conf,
        "probabilities": dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True)),
    }
