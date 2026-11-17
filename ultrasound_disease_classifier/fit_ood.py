"""
fit_ood.py
----------
Fits an out-of-distribution (OOD) guard for a finished training run, so
predict.py can warn when an image does not look like the breast-ultrasound
training data (e.g. a fetal or abdominal scan, a photo, a screenshot).

Why a *frozen* ImageNet backbone and not the trained classifier: fine-tuning
squeezes the classifier's features down to "benign vs malignant vs normal"
cues, so a fetal scan lands right among the breast scans (tested: it passed
the check). Generic pretrained features still describe what kind of image it
is, and separate other scan types clearly.

Method: Mahalanobis distance of pooled features to a Gaussian fitted on the
enhanced BUSI images (Ledoit-Wolf shrunk covariance). The threshold is the
99th percentile of distances of *held-out* images in 5-fold CV, i.e. ~1% of
genuine unseen BUSI scans get flagged.

Usage:
    python fit_ood.py --run runs/final_L4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.covariance import LedoitWolf

from modeling import _autocast, eval_transform, gray_to_tensor

ROOT = Path(__file__).resolve().parent
EXTRACTOR = "convnext_tiny.fb_in22k_ft_in1k"


def load_extractor(device):
    return timm.create_model(EXTRACTOR, pretrained=True, num_classes=0).to(device).eval()


@torch.no_grad()
def extract(extractor, x, device, batch=32):
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        with _autocast(xb):
            out.append(extractor(xb).float().cpu())
    return torch.cat(out).numpy()


def mahalanobis(feats, mean, precision):
    d = np.atleast_2d(feats) - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "final_L4"))
    ap.add_argument("--quantile", type=float, default=0.99)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(ROOT / "data" / "index.csv")
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    tf = eval_transform(cfg["img_size"])
    x = torch.stack([tf(gray_to_tensor(cv2.imread(p, cv2.IMREAD_GRAYSCALE)).expand(3, -1, -1))
                     for p in df[f"L{cfg['level']}_path"]])
    feats = extract(load_extractor(device), x, device)

    held_out = []
    for k in sorted(df.fold.unique()):
        tr, te = df.fold.values != k, df.fold.values == k
        lw = LedoitWolf().fit(feats[tr])
        held_out.append(mahalanobis(feats[te], lw.location_, lw.precision_))
    held_out = np.concatenate(held_out)
    threshold = float(np.quantile(held_out, args.quantile))

    lw = LedoitWolf().fit(feats)
    np.savez(run_dir / "ood.npz", threshold=threshold, extractor=EXTRACTOR,
             mean=lw.location_.astype(np.float32), precision=lw.precision_.astype(np.float32))
    print(f"held-out BUSI distance: median {np.median(held_out):.1f}, max {held_out.max():.1f}")
    print(f"threshold ({args.quantile:.0%} of unseen BUSI scans pass): {threshold:.1f}")
    print(f"saved {run_dir / 'ood.npz'}")


if __name__ == "__main__":
    main()
