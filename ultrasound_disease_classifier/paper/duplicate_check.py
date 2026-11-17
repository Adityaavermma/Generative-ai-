"""
Spot-check of the duplicate detector: draws a random sample of matched pairs
and saves side-by-side montages for visual verification.

    python paper/duplicate_check.py --n 50
Writes paper/data/duplicate_check/pairs.csv and montage_*.png (10 pairs each).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import CLASSES  # noqa: E402
from prepare_data import RAW_DIR, _phash, _thumb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths, labels = [], []
    for label in CLASSES:
        for p in sorted((RAW_DIR / label).glob("*.png")):
            if "_mask" not in p.name:
                paths.append(p)
                labels.append(label)
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    hashes = np.array([_phash(im) for im in images])
    thumbs = np.array([_thumb(im) for im in images])
    corr = thumbs @ thumbs.T
    ham = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    same = ((corr > 0.90) & (ham <= 12)) | ((corr > 0.80) & (ham <= 4))   # rule in prepare_data.py
    ii, jj = np.where(np.triu(same, 1))
    print(f"matched pairs in BUSI: {len(ii)}")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(ii), size=min(args.n, len(ii)), replace=False)
    out_dir = HERE / "data" / "duplicate_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, idx in enumerate(pick):
        i, j = ii[idx], jj[idx]
        rows.append({"pair": k + 1, "a": f"{labels[i]}/{paths[i].stem}", "b": f"{labels[j]}/{paths[j].stem}",
                     "corr": round(float(corr[i, j]), 3), "hash_bits": int(ham[i, j])})
    pd.DataFrame(rows).to_csv(out_dir / "pairs.csv", index=False)

    def tile(im, text):
        t = cv2.resize(im, (260, 200), interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        t = cv2.copyMakeBorder(t, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return t

    for page in range(0, len(rows), 10):
        strips = []
        for r, idx in zip(rows[page:page + 10], pick[page:page + 10]):
            i, j = ii[idx], jj[idx]
            gap = np.full((222, 8, 3), 255, np.uint8)
            strips.append(np.hstack([tile(images[i], f"#{r['pair']} {r['a']}"), gap,
                                     tile(images[j], f"{r['b']}  r={r['corr']} d={r['hash_bits']}")]))
        # two pairs per row
        rows_img = []
        for s in range(0, len(strips), 2):
            pair = strips[s:s + 2]
            if len(pair) == 1:
                pair.append(np.full_like(pair[0], 255))
            rows_img.append(np.hstack([pair[0], np.full((222, 24, 3), 200, np.uint8), pair[1]]))
            rows_img.append(np.full((8, rows_img[-1].shape[1], 3), 255, np.uint8))
        cv2.imwrite(str(out_dir / f"montage_{page // 10 + 1}.png"), np.vstack(rows_img))
    print(f"wrote {len(rows)} pairs to {out_dir}")


if __name__ == "__main__":
    main()
