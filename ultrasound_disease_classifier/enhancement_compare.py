"""
enhancement_compare.py
----------------------
Runs every denoising / enhancement method of agentic_us_denoise on one scan,
measures each result, and ranks them.

Methods compared
  * the individual tools in processing_tools.py (spatial filters, SRAD,
    non-local means, Wiener, wavelet, curvelet-style, thresholding, CLAHE,
    unsharp mask, gamma, histogram equalisation), and
  * the agent pipelines L1 - L4 (what the agents plan and run).

Real scans have no clean "ground truth", so every measure is either
no-reference or relative to the input scan:

  speckle reduction  1 - speckle index(out) / speckle index(in), where the
                     speckle index is the mean local std / local mean (5x5)
  anatomy kept       1 - structure found in what the filter REMOVED
                     ("method noise"): a good despeckler removes random
                     speckle only; an over-smoothing one also removes tissue
                     edges, which then show up in the removed part
  tone kept          1 - change of the large-scale brightness / contrast
                     (blurred input vs blurred output); a denoiser should not
                     repaint the scan (thresholding / CLAHE / HE do)
  CNR, edge keep,    shown for reference; CNR and edge keep can also be
  SSIM, SNR, entropy chosen as the ranking criterion

"Best denoising" = speckle reduction^0.4 x anatomy kept^0.3 x tone kept^0.3.
It is a product on purpose: doing nothing (no speckle reduction) and blurring
everything (anatomy lost) both score low. Scores are given as % of the best
method on the same scan.

CLI:  python enhancement_compare.py path/to/scan.png
"""

import sys
import time

import cv2
import numpy as np
import pandas as pd

from us_pipeline import enhance, load_tools

# (key, label, family, pipeline, parameters)
FAMILY_DENOISE = "Denoising"
FAMILY_ENHANCE = "Contrast enhancement"
FAMILY_AGENT = "Agent pipeline"

TOOLS = [
    ("mean", "Mean filter", FAMILY_DENOISE, ["Mean"], {"Mean": {"ksize": 5}}),
    ("median", "Median filter", FAMILY_DENOISE, ["Median"], {"Median": {"ksize": 5}}),
    ("gaussian", "Gaussian filter", FAMILY_DENOISE, ["Gaussian"], {"Gaussian": {"ksize": 5}}),
    ("bilateral", "Bilateral filter", FAMILY_DENOISE, ["Bilateral"], {"Bilateral": {"ksize": 9}}),
    ("wiener", "Adaptive Wiener", FAMILY_DENOISE, ["AdaptiveWiener"], {"AdaptiveWiener": {"ksize": 5}}),
    ("nlm", "Non-local means", FAMILY_DENOISE, ["NonLocalMeans"], {"NonLocalMeans": {"h": 10}}),
    ("srad", "SRAD (speckle diffusion)", FAMILY_DENOISE, ["SRAD"], {"SRAD": {"iterations": 25, "step": 0.05}}),
    ("wavelet", "Wavelet shrinkage", FAMILY_DENOISE, ["Wavelet"], {"Wavelet": {"wavelet": "db2", "level": 2}}),
    ("curvelet", "Curvelet-style filter", FAMILY_DENOISE, ["Curvelet"], {"Curvelet": {"levels": 3}}),
    ("threshold", "Soft thresholding", FAMILY_DENOISE, ["Thresholding"], {"Thresholding": {"mode": "soft", "value": 15}}),
    ("clahe", "CLAHE", FAMILY_ENHANCE, ["CLAHE"], {"CLAHE": {"clip_limit": 2.0, "tile_grid": 8}}),
    ("unsharp", "Unsharp mask", FAMILY_ENHANCE, ["UnsharpMask"], {"UnsharpMask": {"amount": 1.0, "radius": 5}}),
    ("gamma", "Gamma correction", FAMILY_ENHANCE, ["GammaCorrection"], {"GammaCorrection": {"gamma": 1.2}}),
    ("histeq", "Histogram equalisation", FAMILY_ENHANCE, ["HistogramEqualization"], {}),
]
AGENT_LEVELS = [
    (1, "Agent L1 · basic denoise"),
    (2, "Agent L2 · denoised (model input)"),
    (3, "Agent L3 · + CLAHE"),
    (4, "Agent L4 · optimised (model input)"),
]

CRITERIA = {
    "overall": "Best denoising (overall)",
    "speckle": "Speckle reduction",
    "anatomy": "Anatomy kept",
    "contrast": "Contrast (CNR)",
    "edges": "Edges preserved",
}
OVERALL_WEIGHTS = {"speckle_reduction": 0.4, "anatomy_kept": 0.3, "tone_kept": 0.3}
CRITERION_COLUMN = {"speckle": "speckle_reduction", "anatomy": "anatomy_kept",
                    "contrast": "cnr", "edges": "edge_keep"}

# Same-model, same-folds accuracy on 762 BUSI scans (runs/cmp_L*_256).
CLASSIFIER_EVIDENCE = {"Raw input": 0.869, "Agent L2 · denoised": 0.885,
                       "Agent L4 · optimised": 0.861}


def speckle_index(img: np.ndarray, k: int = 5) -> float:
    """Mean coefficient of variation in k x k windows (tissue pixels only)."""
    f = img.astype(np.float32)
    mean = cv2.blur(f, (k, k))
    sq = cv2.blur(f * f, (k, k))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    tissue = mean > 15  # skip black background / shadow, where the ratio explodes
    return float(np.mean(std[tissue] / mean[tissue])) if tissue.any() else float("nan")


def _gradient_magnitude(x, sigma):
    x = cv2.GaussianBlur(x, (0, 0), sigma)
    return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))


def structure_removed(original: np.ndarray, out: np.ndarray, sigma: float = 2.0) -> float:
    """How much anatomy is in the method noise (original - out), 0..1.

    Correlation between the edges of the (smoothed) removed component and
    the edges of the (smoothed) input, over tissue pixels."""
    f = original.astype(np.float32)
    residual = f - out.astype(np.float32)
    if residual.std() < 0.5:          # (almost) nothing removed
        return 0.0
    a = _gradient_magnitude(residual, sigma).ravel()
    b = _gradient_magnitude(f, sigma).ravel()
    tissue = (cv2.blur(f, (5, 5)) > 15).ravel()
    c = np.corrcoef(a[tissue], b[tissue])[0, 1]
    return float(np.clip(np.nan_to_num(c), 0, 1))


def tone_change(original: np.ndarray, out: np.ndarray, sigma: float = 4.0) -> float:
    """Relative change of the large-scale brightness / contrast, 0..1."""
    a = cv2.GaussianBlur(original.astype(np.float32), (0, 0), sigma)
    b = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigma)
    return float(np.clip(np.mean(np.abs(a - b)) / max(float(np.mean(a)), 1e-6), 0, 1))


def _measure(original, out, M):
    return {
        "speckle_index": speckle_index(out),
        "anatomy_kept": 1 - structure_removed(original, out),
        "tone_kept": 1 - tone_change(original, out),
        "cnr": float(M.cnr(out)),
        "edge_keep": float(M.edge_preservation_index(original, out)),
        "ssim": float(M.ssim(original, out)),
        "snr_db": float(M.snr(out)),
        "entropy": float(M.entropy(out)),
    }


def run_all(img: np.ndarray, include_agents: bool = True):
    """Returns (DataFrame of measurements, {key: processed image})."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    PT, M = load_tools()
    rows, images = [], {"original": img}
    rows.append({"key": "original", "method": "Original (no processing)", "family": "Input",
                 "pipeline": "–", "time_ms": 0.0, **_measure(img, img, M)})

    for key, label, family, pipeline, params in TOOLS:
        t = time.perf_counter()
        out = PT.run_pipeline(img, pipeline, params)
        ms = (time.perf_counter() - t) * 1000
        images[key] = out
        rows.append({"key": key, "method": label, "family": family,
                     "pipeline": " → ".join(pipeline), "time_ms": ms, **_measure(img, out, M)})

    if include_agents:
        for level, label in AGENT_LEVELS:
            t = time.perf_counter()
            out, info = enhance(img, level=level)
            ms = (time.perf_counter() - t) * 1000
            key = f"agent_L{level}"
            images[key] = out
            rows.append({"key": key, "method": label, "family": FAMILY_AGENT,
                         "pipeline": " → ".join(info["pipeline"]), "time_ms": ms,
                         **_measure(img, out, M)})
    df = pd.DataFrame(rows)
    si0 = df.loc[df.key == "original", "speckle_index"].iloc[0]
    df.insert(df.columns.get_loc("speckle_index") + 1, "speckle_reduction",
              (1 - df.speckle_index / si0).clip(lower=0))
    return df, images


def score(df: pd.DataFrame, criterion: str = "overall") -> pd.DataFrame:
    """Adds 'score' (% of the best method for this criterion) and 'rank'.

    The original scan is kept as a reference row but not ranked."""
    out = df.copy()
    cand = out.key != "original"
    if criterion == "overall":
        raw = np.ones(len(out))
        for col, w in OVERALL_WEIGHTS.items():
            raw *= out[col].clip(lower=0).to_numpy() ** w
    else:
        raw = out[CRITERION_COLUMN[criterion]].clip(lower=0).to_numpy(dtype=float)
    best = raw[cand.to_numpy()].max()
    out["score"] = np.where(cand, 100 * raw / best if best > 0 else 0.0, np.nan)
    out["rank"] = out.score.rank(ascending=False, method="min")
    return out.sort_values(["rank"], na_position="first").reset_index(drop=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # method names contain · and →
    if len(sys.argv) < 2:
        raise SystemExit("usage: python enhancement_compare.py path/to/scan.png")
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read {sys.argv[1]}")
    df, _ = run_all(img)
    for crit, label in CRITERIA.items():
        s = score(df, crit)
        best = s.dropna(subset=["score"]).iloc[0]
        print(f"best for {label:20s}: {best.method}  (score {best.score:.0f})")
    s = score(df, "overall")
    cols = ["rank", "method", "family", "score", "speckle_reduction", "anatomy_kept", "tone_kept",
            "cnr", "edge_keep", "ssim", "time_ms"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print()
        print(s[cols].to_string(index=False))


if __name__ == "__main__":
    main()
