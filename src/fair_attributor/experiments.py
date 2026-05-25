import os
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier
import xgboost as xgb_module

from .model import get_ensemble_proba, load_bundle, build_single_feature_vector


def summarize_multiclass_result(y_true, proba, label_encoder, name="MODEL"):
    y_pred = np.argmax(proba, axis=1)
    result = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_pred": y_pred,
        "proba": proba,
    }
    print(f"[{name}] Accuracy={result['accuracy']:.4f}, Macro F1={result['macro_f1']:.4f}")
    return result


def train_eval_simple_ensemble(X_train, X_val, X_test, y_train, y_val, y_test, num_classes, seed=42, name="Ablation"):
    lr = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight="balanced",
        solver="saga",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=seed,
    )
    lr.fit(X_train, y_train)

    xgb_clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        reg_lambda=2.0,
        random_state=seed,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=30,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = 0.5 * lr.predict_proba(X_test) + 0.5 * xgb_clf.predict_proba(X_test)
    return summarize_multiclass_result(y_test, proba, None, name=name), lr, xgb_clf


def confidence_threshold_sweep(y_true, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.96, 0.05), 2)
    y_pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    rows = []
    for th in thresholds:
        accepted = conf >= th
        rows.append({
            "Threshold": th,
            "Coverage": float(np.mean(accepted)),
            "Unknown Ratio": float(np.mean(~accepted)),
            "Accepted Accuracy": accuracy_score(y_true[accepted], y_pred[accepted]) if np.any(accepted) else np.nan,
            "Accepted Macro F1": f1_score(y_true[accepted], y_pred[accepted], average="macro", zero_division=0) if np.any(accepted) else np.nan,
        })
    return pd.DataFrame(rows)


def family_level_attribution(y_true, y_pred, class_names, family_map):
    class_to_family = {i: family_map[class_names[i]] for i in range(len(class_names))}
    true_family = np.array([class_to_family[i] for i in y_true])
    pred_family = np.array([class_to_family[i] for i in y_pred])
    return {
        "family_accuracy": accuracy_score(true_family, pred_family),
        "family_macro_f1": f1_score(true_family, pred_family, average="macro", zero_division=0),
        "true_family": true_family,
        "pred_family": pred_family,
    }


def transform_image_pil(img, transform_name):
    img = img.convert("RGB")
    if transform_name in {"original", "jpeg_q95", "jpeg_q75", "jpeg_q50"}:
        return img
    if transform_name == "resize_512":
        return img.resize((512, 512)).resize(img.size)
    if transform_name == "resize_256":
        return img.resize((256, 256)).resize(img.size)
    if transform_name == "center_crop_80":
        w, h = img.size
        nw, nh = int(w * 0.80), int(h * 0.80)
        left, top = (w - nw) // 2, (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh)).resize((w, h))
    if transform_name == "gaussian_blur":
        arr = cv2.GaussianBlur(np.array(img), (5, 5), 0)
        return Image.fromarray(arr)
    if transform_name == "gaussian_noise":
        arr = np.array(img).astype(np.float32)
        rng = np.random.default_rng(42 + hash(img.tobytes()) % (2**31))
        arr = np.clip(arr + rng.normal(0, 5, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if transform_name == "web_downscale":
        w, h = img.size
        small = img.resize((max(128, w // 2), max(128, h // 2)))
        return small.resize((w, h))
    raise ValueError(f"Unknown transform: {transform_name}")


def save_transformed_temp(img, transform_name, tmp_dir, idx):
    out_path = os.path.join(tmp_dir, f"tmp_{idx}_{transform_name}.jpg")
    quality = {"jpeg_q95": 95, "jpeg_q75": 75, "jpeg_q50": 50}.get(transform_name, 95)
    img.save(out_path, quality=quality)
    return out_path


class MahalanobisOODScorer:
    def __init__(self):
        self.class_means = {}
        self.precision_matrix = None

    def fit(self, X_train, y_train):
        for c in np.unique(y_train):
            self.class_means[c] = np.mean(X_train[y_train == c], axis=0)
        residuals = np.zeros_like(X_train)
        for i, yi in enumerate(y_train):
            residuals[i] = X_train[i] - self.class_means[yi]
        lw = LedoitWolf().fit(residuals)
        self.precision_matrix = lw.precision_
        return self

    def score(self, X):
        distances = np.zeros((len(X), len(self.class_means)))
        for col, (c, mean_c) in enumerate(self.class_means.items()):
            diff = X - mean_c
            distances[:, col] = np.sqrt(np.maximum(np.sum((diff @ self.precision_matrix) * diff, axis=1), 0))
        return np.min(distances, axis=1)


def compute_energy_score(X, lr_model, xgb_model, temperature=1.0):
    logits_lr = lr_model.decision_function(X)
    logits_xgb = xgb_model.get_booster().predict(xgb_module.DMatrix(X), output_margin=True)
    logits_avg = 0.5 * logits_lr + 0.5 * logits_xgb
    energy = temperature * logsumexp(logits_avg / temperature, axis=1)
    return -energy


class HybridOODScorer:
    def __init__(self, alpha_msp=0.2, alpha_maha=0.5, alpha_energy=0.3):
        self.alpha_msp = alpha_msp
        self.alpha_maha = alpha_maha
        self.alpha_energy = alpha_energy
        self.maha_scorer = MahalanobisOODScorer()

    def fit(self, X_train, y_train, lr_model, xgb_model):
        self.maha_scorer.fit(X_train, y_train)
        proba_train = get_ensemble_proba(X_train, lr_model, xgb_model)
        msp = 1.0 - np.max(proba_train, axis=1)
        maha = self.maha_scorer.score(X_train)
        energy = compute_energy_score(X_train, lr_model, xgb_model)
        self.stats = {
            "msp": (np.mean(msp), np.std(msp) + 1e-8),
            "maha": (np.mean(maha), np.std(maha) + 1e-8),
            "energy": (np.mean(energy), np.std(energy) + 1e-8),
        }
        return self

    def score(self, X, lr_model, xgb_model):
        proba = get_ensemble_proba(X, lr_model, xgb_model)
        msp = 1.0 - np.max(proba, axis=1)
        maha = self.maha_scorer.score(X)
        energy = compute_energy_score(X, lr_model, xgb_model)
        z_msp = (msp - self.stats["msp"][0]) / self.stats["msp"][1]
        z_maha = (maha - self.stats["maha"][0]) / self.stats["maha"][1]
        z_energy = (energy - self.stats["energy"][0]) / self.stats["energy"][1]
        return self.alpha_msp * z_msp + self.alpha_maha * z_maha + self.alpha_energy * z_energy, proba
