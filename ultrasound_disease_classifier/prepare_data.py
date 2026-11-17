"""
prepare_data.py
---------------
1. Indexes the BUSI breast-ultrasound dataset (benign / malignant / normal).
2. Finds near-duplicate scans. BUSI contains many re-saved / slightly re-cropped
   copies of the same scan, and a few copies even carry different labels.
   Duplicates are grouped so a scan and its copy never land in different
   folds (otherwise test accuracy is inflated by memorisation). Groups whose
   copies disagree on the label are dropped as unreliable ground truth.
3. Assigns 5 stratified, duplicate-aware cross-validation folds.
4. Runs the agentic denoise / enhance pipeline on every image for each
   requested enhancement level and caches the result as PNG.

Usage:
    python prepare_data.py                    # levels 0, 2, 4
    python prepare_data.py --levels 0 1 2 3 4
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from common import CACHE_SIZE, CLASSES, to_cache_size
from us_pipeline import enhance

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "Dataset_BUSI_with_GT"
PROC_DIR = ROOT / "data" / "processed"
INDEX_CSV = ROOT / "data" / "index.csv"


# --------------------------------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------------------------------

def _phash(img):
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8].flatten()[1:]
    return d > np.median(d)


def _thumb(img):
    t = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


def duplicate_groups(images):
    """Union-find over pairs that look like the same scan.

    Two images are "the same scan" when their 64x64 thumbnails are highly
    correlated AND their perceptual hashes are close. Requiring both avoids
    chaining together different scans that merely share a similar layout
    (thresholds were checked by eye on BUSI pairs).
    """
    n = len(images)
    hashes = np.array([_phash(im) for im in images])
    thumbs = np.array([_thumb(im) for im in images])
    corr = thumbs @ thumbs.T
    ham = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    same = ((corr > 0.90) & (ham <= 12)) | ((corr > 0.80) & (ham <= 4))

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(same, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return [find(i) for i in range(n)]


# --------------------------------------------------------------------------
# Enhancement worker (runs in a separate process)
# --------------------------------------------------------------------------

def _enhance_one(job):
    src, dst, level = job
    if os.path.exists(dst):
        return dst, None
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    out, info = enhance(img, level=level)
    cv2.imwrite(dst, to_cache_size(out))
    return dst, info


def _view_one(job):
    """Extra single-filter views (not produced by the agents), e.g. NLM."""
    src, dst, view = job
    if os.path.exists(dst):
        return dst
    from us_pipeline import load_tools
    PT, _ = load_tools()
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if view == "nlm":   # same settings as the method benchmark
        out = PT.non_local_means(img, h=10, template_window=7, search_window=21)
    else:
        raise ValueError(view)
    cv2.imwrite(dst, to_cache_size(out))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--views", nargs="*", default=[], choices=["nlm"],
                    help="extra single-filter views, e.g. nlm (non-local means on the raw scan)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"BUSI not found at {RAW_DIR}. See README.md for the download step.")

    # ---- 1. index ---------------------------------------------------------
    rows = []
    for label in CLASSES:
        for p in sorted((RAW_DIR / label).glob("*.png")):
            if "_mask" in p.name:
                continue
            rows.append({"id": f"{label}/{p.stem}", "label": label, "raw_path": str(p)})
    df = pd.DataFrame(rows)
    print(f"Found {len(df)} images:", df.label.value_counts().to_dict())

    # ---- 2. duplicates ----------------------------------------------------
    images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in df.raw_path]
    df["group"] = duplicate_groups(images)
    n_groups = df.group.nunique()
    labels_per_group = df.groupby("group").label.nunique()
    conflicting = labels_per_group[labels_per_group > 1].index
    dropped = df[df.group.isin(conflicting)]
    df = df[~df.group.isin(conflicting)].reset_index(drop=True)
    print(f"Duplicate-aware groups: {n_groups} unique scans among {len(images)} files")
    print(f"Dropped {len(dropped)} images in {len(conflicting)} groups with conflicting labels:")
    for g, sub in dropped.groupby("group"):
        print("   ", ", ".join(sub.id))

    # ---- 3. folds ---------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df.label, df.group)):
        df.loc[te, "fold"] = k
    print("Fold sizes:\n", pd.crosstab(df.fold, df.label))

    # ---- 4. lesion masks (union of all *_mask*.png; used only as an
    #         auxiliary training target, never needed at prediction time) --
    mask_paths = []
    for r in df.itertuples():
        raw = Path(r.raw_path)
        dst = PROC_DIR / "masks" / r.label / f"{raw.stem}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            h, w = cv2.imread(str(raw), cv2.IMREAD_GRAYSCALE).shape
            union = np.zeros((h, w), np.uint8)
            for mp in raw.parent.glob(f"{raw.stem}_mask*.png"):
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == union.shape:
                    union |= (m > 127).astype(np.uint8)
            cv2.imwrite(str(dst), cv2.resize(union * 255, (CACHE_SIZE, CACHE_SIZE),
                                             interpolation=cv2.INTER_NEAREST))
        mask_paths.append(str(dst))
    df["mask_path"] = mask_paths

    # ---- 5. enhancement ---------------------------------------------------
    for level in args.levels:
        jobs = []
        for r in df.itertuples():
            dst = PROC_DIR / f"L{level}" / r.label / f"{Path(r.raw_path).stem}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((r.raw_path, str(dst), level))
        infos = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for dst, info in tqdm(ex.map(_enhance_one, jobs, chunksize=4),
                                  total=len(jobs), desc=f"Enhancing L{level}"):
                infos[dst] = info
        df[f"L{level}_path"] = [j[1] for j in jobs]
        strategies = [i["strategy"] for i in infos.values() if i]
        if strategies:
            print(f"L{level} strategies used:", pd.Series(strategies).value_counts().to_dict())

    # ---- 6. extra single-filter views --------------------------------------
    for view in args.views:
        name = view.upper()
        jobs = []
        for r in df.itertuples():
            dst = PROC_DIR / name / r.label / f"{Path(r.raw_path).stem}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((r.raw_path, str(dst), view))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(_view_one, jobs, chunksize=4), total=len(jobs), desc=f"View {name}"))
        df[f"{name}_path"] = [j[1] for j in jobs]

    df.to_csv(INDEX_CSV, index=False)
    with open(ROOT / "data" / "dataset_summary.json", "w") as f:
        json.dump({
            "files": len(images),
            "unique_scans": int(n_groups),
            "dropped_conflicting_label_images": dropped.id.tolist(),
            "used_images": len(df),
            "class_counts": df.label.value_counts().to_dict(),
            "levels": args.levels,
            "views": args.views,
            "cache_size": CACHE_SIZE,
        }, f, indent=2)
    print(f"Wrote {INDEX_CSV}")


if __name__ == "__main__":
    main()
