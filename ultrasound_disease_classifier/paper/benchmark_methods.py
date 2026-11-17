"""
Benchmark every denoising / enhancement method on a stratified sample of BUSI
scans (for the paper). Writes paper/data/method_benchmark.csv with one row per
(scan, method).

    python paper/benchmark_methods.py --per-class 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from enhancement_compare import run_all, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(HERE.parent / "data" / "index.csv")
    sample = df.groupby("label").sample(n=args.per_class, random_state=args.seed)
    rows = []
    for i, r in enumerate(sample.itertuples(), 1):
        img = cv2.imread(r.raw_path, cv2.IMREAD_GRAYSCALE)
        res, _ = run_all(img)
        ranked = score(res, "overall")
        ranked["scan"] = r.id
        ranked["label"] = r.label
        rows.append(ranked)
        print(f"{i}/{len(sample)} {r.id}: best = {ranked.dropna(subset=['score']).iloc[0].method}",
              flush=True)
    out = HERE / "data" / "method_benchmark.csv"
    out.parent.mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
