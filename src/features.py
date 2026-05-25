import cv2
import numpy as np
from scipy import stats
from scipy.fft import fft2, fftshift
from skimage.feature import canny, graycomatrix, graycoprops
from skimage.filters import laplace, sobel_h, sobel_v


def safe_stat(x, fn, default: float = 0.0) -> float:
    try:
        return float(fn(x))
    except Exception:
        return float(default)


def channel_stats(ch):
    ch = ch.flatten().astype(np.float32)
    return {
        "mean": float(np.mean(ch)),
        "std": float(np.std(ch)),
        "min": float(np.min(ch)),
        "max": float(np.max(ch)),
        "median": float(np.median(ch)),
        "p10": float(np.percentile(ch, 10)),
        "p90": float(np.percentile(ch, 90)),
        "skew": safe_stat(ch, stats.skew),
        "kurtosis": safe_stat(ch, stats.kurtosis),
    }


def histogram_features(ch, bins: int = 16, prefix: str = "hist"):
    hist, _ = np.histogram(ch.flatten(), bins=bins, range=(0, 255), density=True)
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(hist)}


def radial_profile(data):
    y, x = np.indices(data.shape)
    center = np.array([(data.shape[0] - 1) / 2.0, (data.shape[1] - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def fft_features(gray):
    g = gray.astype(np.float32)
    freq = fftshift(fft2(g))
    mag = np.log1p(np.abs(freq))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.max(dist)

    low_mask = dist <= 0.10 * max_r
    mid_mask = (dist > 0.10 * max_r) & (dist <= 0.35 * max_r)
    high_mask = dist > 0.35 * max_r

    total = np.sum(mag) + 1e-8
    low_e = np.sum(mag[low_mask]) / total
    mid_e = np.sum(mag[mid_mask]) / total
    high_e = np.sum(mag[high_mask]) / total

    rp = radial_profile(mag)
    rp = rp[:64] if len(rp) >= 64 else np.pad(rp, (0, 64 - len(rp)))

    feats = {
        "fft_low_energy_ratio": float(low_e),
        "fft_mid_energy_ratio": float(mid_e),
        "fft_high_energy_ratio": float(high_e),
        "fft_high_low_ratio": float(high_e / (low_e + 1e-8)),
        "fft_mean": float(np.mean(mag)),
        "fft_std": float(np.std(mag)),
    }
    for i in range(16):
        feats[f"fft_radial_bin_{i}"] = float(np.mean(rp[i * 4 : (i + 1) * 4]))
    return feats


def edge_features(gray):
    g = gray.astype(np.float32) / 255.0
    sh = sobel_h(g)
    sv = sobel_v(g)
    sm = np.sqrt(sh**2 + sv**2)
    lap = laplace(g)
    can = canny(g, sigma=1.2).astype(np.float32)
    angle = np.arctan2(sv + 1e-8, sh + 1e-8)
    return {
        "sobel_mean": float(np.mean(sm)),
        "sobel_std": float(np.std(sm)),
        "sobel_p90": float(np.percentile(sm, 90)),
        "laplace_abs_mean": float(np.mean(np.abs(lap))),
        "laplace_var": float(np.var(lap)),
        "canny_edge_density": float(np.mean(can)),
        "edge_angle_mean": float(np.mean(angle)),
        "edge_angle_std": float(np.std(angle)),
    }


def glcm_features(gray):
    g_small = (gray.astype(np.uint8) / 32).astype(np.uint8)
    glcm = graycomatrix(
        g_small,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"] = float(np.std(vals))
    return feats


def block_artifact_features(gray, block_size: int = 8):
    g = gray.astype(np.float32)
    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))
    v_idx = [i - 1 for i in range(block_size, g.shape[1], block_size) if i - 1 < dv.shape[1]]
    h_idx = [i - 1 for i in range(block_size, g.shape[0], block_size) if i - 1 < dh.shape[0]]

    if v_idx:
        mask = np.ones(dv.shape[1], dtype=bool)
        mask[v_idx] = False
        v_ratio = float(np.mean(dv[:, v_idx]) / (np.mean(dv[:, mask]) + 1e-8))
    else:
        v_ratio = 1.0

    if h_idx:
        mask = np.ones(dh.shape[0], dtype=bool)
        mask[h_idx] = False
        h_ratio = float(np.mean(dh[h_idx, :]) / (np.mean(dh[mask, :]) + 1e-8))
    else:
        h_ratio = 1.0

    return {
        "blockiness_vertical_ratio": v_ratio,
        "blockiness_horizontal_ratio": h_ratio,
        "blockiness_mean_ratio": float((v_ratio + h_ratio) / 2.0),
    }


def noise_residual_features(rgb):
    rgbf = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(rgbf, (3, 3), 0)
    residual = rgbf - blur
    feats = {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_abs_mean": float(np.mean(np.abs(residual))),
        "residual_p90": float(np.percentile(np.abs(residual), 90)),
    }
    for c, name in enumerate(["r", "g", "b"]):
        rc = residual[:, :, c]
        feats[f"residual_{name}_std"] = float(np.std(rc))
        feats[f"residual_{name}_abs_mean"] = float(np.mean(np.abs(rc)))
    return feats


def rgb_relation_features(rgb):
    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    feats = {}
    for ch, name in zip([r, g, b], ["r", "g", "b"]):
        for k, v in channel_stats(ch).items():
            feats[f"{name}_{k}"] = v
        feats.update(histogram_features(ch, bins=16, prefix=f"{name}_hist"))

    for diff, name in zip([r - g, r - b, g - b], ["rg", "rb", "gb"]):
        feats[f"{name}_mean"] = float(np.mean(diff))
        feats[f"{name}_std"] = float(np.std(diff))
        feats[f"{name}_abs_mean"] = float(np.mean(np.abs(diff)))

    def safe_corr(a, b):
        try:
            corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)
        except Exception:
            return 0.0

    feats["corr_rg"] = safe_corr(r, g)
    feats["corr_rb"] = safe_corr(r, b)
    feats["corr_gb"] = safe_corr(g, b)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    feats["h_mean"] = float(np.mean(hsv[:, :, 0]))
    feats["s_mean"] = float(np.mean(hsv[:, :, 1]))
    feats["v_mean"] = float(np.mean(hsv[:, :, 2]))
    feats["s_std"] = float(np.std(hsv[:, :, 1]))
    feats["v_std"] = float(np.std(hsv[:, :, 2]))
    return feats


def extract_artifact_features(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feats = {}
    feats.update(rgb_relation_features(rgb))
    feats.update(fft_features(gray))
    feats.update(edge_features(gray))
    feats.update(glcm_features(gray))
    feats.update(block_artifact_features(gray))
    feats.update(noise_residual_features(rgb))
    return feats
