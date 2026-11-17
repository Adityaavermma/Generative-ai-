"""
Numbers for the ablation table and text: exact out-of-fold metrics of every
run, and paired McNemar exact tests between runs (all runs share the folds).

    python paper/ablation_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

CLASSES = ["benign", "malignant", "normal"]

ROOT = Path(__file__).resolve().parent.parent
RUNS = ["cmp_L0_256", "cmp_L2_256", "cmp_L4_256", "cmp_L4_384_seg_tiny",
        "final_L0", "final_L0_s2026", "final_L2", "final_L4", "ensemble_L4_L2", "ensemble_L0_x2"]
PAIRS = [("cmp_L2_256", "cmp_L0_256"), ("cmp_L2_256", "cmp_L4_256"), ("cmp_L0_256", "cmp_L4_256"),
         ("final_L2", "final_L0"), ("final_L4", "final_L0"), ("final_L2", "final_L4"),
         ("ensemble_L4_L2", "final_L0"), ("ensemble_L4_L2", "final_L4"), ("ensemble_L4_L2", "final_L2"),
         ("final_L0", "cmp_L0_256"),
         # equal-size comparison: ten raw-image models (two seeds) vs the ten-model L2 + L4 ensemble
         ("ensemble_L4_L2", "ensemble_L0_x2"), ("ensemble_L0_x2", "final_L0"),
         ("final_L0_s2026", "final_L0")]
# paired bootstrap of the macro-AUC difference (first minus second)
AUC_PAIRS = [("ensemble_L4_L2", "ensemble_L0_x2"), ("ensemble_L0_x2", "final_L0"),
             ("ensemble_L4_L2", "final_L0")]


def correct(run, ids):
    d = pd.read_csv(ROOT / "runs" / run / "oof_predictions.csv").set_index("id").loc[ids]
    return (d.label == d.pred).to_numpy()


def probs(run, ids):
    d = pd.read_csv(ROOT / "runs" / run / "oof_predictions.csv").set_index("id").loc[ids]
    return d[[f"p_{c}" for c in CLASSES]].to_numpy(), d.label.map(CLASSES.index).to_numpy()


def macro_auc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def auc_diff(a, b, ids, n_boot=2000, seed=0):
    pa, y = probs(a, ids)
    pb, _ = probs(b, ids)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(macro_auc(y[i], pa[i]) - macro_auc(y[i], pb[i]))
    diffs = np.array(diffs)
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return macro_auc(y, pa) - macro_auc(y, pb), np.percentile(diffs, [2.5, 97.5]), min(p_two, 1.0)


def main():
    present = [r for r in RUNS if (ROOT / "runs" / r / "metrics.json").exists()]
    ids = pd.read_csv(ROOT / "runs" / present[0] / "oof_predictions.csv").id
    print(f"{'run':22s} {'acc':>7s} {'bal':>7s} {'AUC':>7s} {'malSens':>8s} {'malSpec':>8s} {'foldSD':>7s}")
    for r in present:
        m = json.loads((ROOT / "runs" / r / "metrics.json").read_text())
        cm = m["confusion_matrix"]
        acc = sum(cm[i][i] for i in range(3)) / m["n_images"]
        print(f"{r:22s} {100 * acc:7.3f} {100 * m['balanced_accuracy']:7.3f} {m['macro_auc_ovr']:7.4f} "
              f"{100 * m['malignant_sensitivity']:8.2f} {100 * m['malignant_specificity']:8.2f} "
              f"{100 * m['fold_accuracy_std']:7.2f}")
    print("\nMcNemar exact (b = only first right, c = only second right):")
    for a, b in PAIRS:
        if a in present and b in present:
            ca, cb = correct(a, ids), correct(b, ids)
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            print(f"  {a:16s} vs {b:16s}: {n01:3d} vs {n10:3d}  p = {p:.3f}")
    print("\nPaired bootstrap of the macro-AUC difference (2000 resamples):")
    for a, b in AUC_PAIRS:
        if a in present and b in present:
            d, (lo, hi), p = auc_diff(a, b, ids)
            print(f"  {a:16s} - {b:16s}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p = {p:.3f}")
    missing = [r for r in RUNS if r not in present]
    if missing:
        print("\nnot finished yet:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
