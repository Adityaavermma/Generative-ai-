"""Summarise all finished runs (runs/*/metrics.json) as a table and a chart."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
INK, INK2, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
LEVEL_NAMES = {0: "L0 raw", 1: "L1 basic denoise", 2: "L2 denoised",
               3: "L3 + CLAHE", 4: "L4 denoised + enhanced"}


def describe(cfg):
    arch = cfg["arch"].split(".")[0].replace("_", "-")
    seg = " + lesion loss" if cfg.get("seg_weight", 0) > 0 else ""
    if "members" in cfg:
        return f"Ensemble {cfg['level']} · {arch} {cfg['img_size']}px{seg}"
    return f"{LEVEL_NAMES.get(cfg['level'], cfg['level'])} · {arch} {cfg['img_size']}px{seg}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: print ± and · correctly
    rows = []
    for mf in sorted((ROOT / "runs").glob("*/metrics.json")):
        cfg = json.loads((mf.parent / "config.json").read_text())
        rows.append((cfg, json.loads(mf.read_text())))
    if not rows:
        raise SystemExit("No finished runs found in runs/.")
    rows.sort(key=lambda r: r[1]["accuracy"])

    print("| Run | Input · model | Accuracy (fold ± sd) | Balanced acc | Macro F1 | Macro AUC | Malignant sens / spec |")
    print("|---|---|---|---|---|---|---|")
    for cfg, m in reversed(rows):
        print(f"| `{cfg['name']}` | {describe(cfg)} "
              f"| {m['accuracy']:.1%} (±{m['fold_accuracy_std']:.1%}) | {m['balanced_accuracy']:.1%} "
              f"| {m['macro_f1']:.3f} | {m['macro_auc_ovr']:.3f} "
              f"| {m['malignant_sensitivity']:.1%} / {m['malignant_specificity']:.1%} |")

    acc = [m["accuracy"] for _, m in rows]
    err = [m["fold_accuracy_std"] for _, m in rows]
    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.55 * len(rows)), facecolor=SURFACE)
    y = list(range(len(rows)))
    ax.barh(y, acc, height=0.55, color=BLUE, xerr=err,
            error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
    for i, a in enumerate(acc):
        ax.text(0.01, i, f"{a:.1%}", va="center", ha="left", color="#ffffff", fontsize=9,
                fontweight="bold")
    ax.set_yticks(y, [describe(c) for c, _ in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Out-of-fold accuracy, 5-fold CV on 762 BUSI images (error bar = fold sd)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = ROOT / "runs" / "comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nchart: {out}")


if __name__ == "__main__":
    main()
