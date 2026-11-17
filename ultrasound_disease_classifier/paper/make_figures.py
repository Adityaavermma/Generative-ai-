"""
Builds every figure and the numeric tables for the paper into paper/figures.

    python paper/make_figures.py

Needs: data/ (prepare_data.py), runs/ (training), paper/data/method_benchmark.csv
(paper/benchmark_methods.py).
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common import CLASSES, to_cache_size  # noqa: E402
from enhancement_compare import run_all  # noqa: E402
from inference import heat_overlay  # noqa: E402
from modeling import build_model, eval_transform, gray_to_tensor, predict_with_map  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# IEEE style: Times, 8 pt, single column 3.5 in, double column 7.16 in.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
})
COL1, COL2 = 3.5, 7.16
# validated categorical slots 1-3 (blue, orange, aqua) + muted gray
C1, C2, C3, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8f8e86", "#52514e"
FAMILY_COLORS = {"Denoising": C1, "Contrast enhancement": C2, "Agent pipeline": C3}
LABEL = {"benign": "Benign", "malignant": "Malignant", "normal": "Normal"}


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


# ----------------------------------------------------------------------------
# Fig. 2: qualitative examples with out-of-fold heat-maps
# ----------------------------------------------------------------------------

def examples_figure():
    idx = pd.read_csv(ROOT / "data" / "index.csv").set_index("id")
    oof = pd.read_csv(ROOT / "runs" / "ensemble_L4_L2" / "oof_predictions.csv").set_index("id")
    picks = ["benign/benign (10)", "malignant/malignant (20)", "normal/normal (10)"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = {lvl: json.loads((ROOT / "runs" / f"final_L{lvl}" / "config.json").read_text()) for lvl in (4, 2)}

    fig, axes = plt.subplots(len(picks), 4, figsize=(COL2, 0.8 * COL2 * len(picks) / 4 + 0.15),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    titles = ["(a) Original", "(b) Agent L2: denoised", "(c) Agent L4: + enhanced",
              "(d) Lesion map, expert outline"]
    for r, sid in enumerate(picks):
        row = idx.loc[sid]
        raw = cv2.imread(row.raw_path, cv2.IMREAD_GRAYSCALE)
        l2 = cv2.imread(row.L2_path, cv2.IMREAD_GRAYSCALE)
        l4 = cv2.imread(row.L4_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        heats = []
        for lvl, img in ((4, l4), (2, l2)):
            cfg = cfgs[lvl]
            m = build_model(cfg["arch"], pretrained=False)
            m.load_state_dict(torch.load(ROOT / "runs" / f"final_L{lvl}" / f"fold{int(row.fold)}.pt",
                                         map_location="cpu"))
            m = m.to(device).eval()
            x = eval_transform(cfg["img_size"])(gray_to_tensor(img).expand(3, -1, -1)).unsqueeze(0).to(device)
            heats.append(predict_with_map(m, x)[1][0].cpu().numpy())
        heat = np.mean(heats, axis=0)
        over = heat_overlay(l4, heat, 0.6)
        if mask.max() > 0:
            cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, (0, 230, 255), 2)
        probs = oof.loc[sid, [f"p_{c}" for c in CLASSES]].values
        pred = CLASSES[int(np.argmax(probs))]
        ims = [to_cache_size(raw), l2, l4, over]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=7.5, pad=3)
        axes[r, 0].set_ylabel(f"{LABEL[row.label]}\n(pred. {LABEL[pred].lower()}, {probs.max():.0%})",
                              fontsize=7.5, labelpad=3)
    savefig(fig, "fig2_examples")


# ----------------------------------------------------------------------------
# Fig. 3 / Table III: denoising method benchmark over 60 scans
# ----------------------------------------------------------------------------

def method_figures():
    d = pd.read_csv(HERE / "data" / "method_benchmark.csv")
    d = d[d.key != "original"]
    g = d.groupby(["key", "method", "family"]).agg(
        score=("score", "mean"), score_sd=("score", "std"),
        spk=("speckle_reduction", "mean"), spk_sd=("speckle_reduction", "std"),
        anat=("anatomy_kept", "mean"), anat_sd=("anatomy_kept", "std"),
        tone=("tone_kept", "mean"), cnr=("cnr", "mean"), edge=("edge_keep", "mean"),
        wins=("rank", lambda r: int((r == 1).sum())), ms=("time_ms", "median"),
    ).reset_index().sort_values("score", ascending=False)
    g.to_csv(HERE / "data" / "method_summary.csv", index=False)
    n_scans = d.scan.nunique()

    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    for fam, col in FAMILY_COLORS.items():
        s = g[g.family == fam]
        ax.errorbar(s.spk * 100, s.anat, xerr=s.spk_sd * 100, yerr=s.anat_sd, fmt="none",
                    ecolor=col, elinewidth=0.6, alpha=0.35, capsize=0)
        ax.scatter(s.spk * 100, s.anat, s=26, color=col, edgecolor="white", linewidth=0.6,
                   zorder=3, label=fam)
    # Labels placed by hand to avoid collisions (points: g.spk*100, g.anat).
    short = {"nlm": "NLM", "wiener": "Wiener", "bilateral": "Bilateral", "srad": "SRAD",
             "median": "Median", "mean": "Mean", "gaussian": "Gaussian = agent L1",
             "wavelet": "Wavelet", "threshold": "Soft thresh.", "agent_L2": "Agent L2",
             "histeq": "Hist. eq.", "curvelet": "Curvelet", "gamma": "Gamma"}
    offsets = {"nlm": (-8, 6), "wiener": (5, 2), "bilateral": (-14, -11), "srad": (5, -2),
               "median": (-31, -3), "mean": (5, -7), "gaussian": (-20, -14), "wavelet": (5, 2),
               "threshold": (5, 2), "agent_L2": (5, 4), "histeq": (5, 2), "curvelet": (5, -6),
               "gamma": (5, 3)}
    for r in g.itertuples():
        if r.key in short:
            ax.annotate(short[r.key], (r.spk * 100, r.anat), xytext=offsets.get(r.key, (4, 2)),
                        textcoords="offset points", fontsize=6.5, color="#0b0b0b")
    # CLAHE, unsharp mask and agent L3 / L4 overlap at ~0 % speckle removed.
    ax.annotate("CLAHE, unsharp,\nagent L3 / L4", (0.3, 0.25), xytext=(3.5, 0.04),
                textcoords="data", fontsize=6.5, color="#0b0b0b",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_xlabel("Speckle removed (%)")
    ax.set_ylabel("Anatomy kept (1 = only noise removed)")
    ax.set_xlim(left=-2)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.34, 1.0), frameon=False, handletextpad=0.2,
              borderaxespad=0.2)
    savefig(fig, "fig3_tradeoff")

    # Fig. 4: what each filter removed (method noise) on one scan
    idx = pd.read_csv(ROOT / "data" / "index.csv").set_index("id")
    raw = cv2.imread(idx.loc["malignant/malignant (20)", "raw_path"], cv2.IMREAD_GRAYSCALE)
    _, ims = run_all(raw)
    keys = [("nlm", "(a) Non-local means"), ("agent_L2", "(b) Agent L2 (Gaussian + wavelet)"),
            ("mean", "(c) Mean filter"), ("clahe", "(d) CLAHE")]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.6), gridspec_kw=dict(wspace=0.04))
    for ax, (k, t) in zip(axes, keys):
        res = raw.astype(np.float32) - ims[k].astype(np.float32)
        ax.imshow(np.clip(128 + 3 * res, 0, 255), cmap="gray", vmin=0, vmax=255)
        a = g.set_index("key").loc[k]
        ax.set_title(f"{t}\n$A$ = {a.anat:.2f},  $R$ = {100 * a.spk:.0f} %", fontsize=7, pad=2)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    savefig(fig, "fig4_method_noise")
    return g, n_scans


# ----------------------------------------------------------------------------
# Fig. 5: confusion matrix + ROC of the ensemble
# ----------------------------------------------------------------------------

def results_figure():
    m = json.loads((ROOT / "runs" / "ensemble_L4_L2" / "metrics.json").read_text())
    oof = pd.read_csv(ROOT / "runs" / "ensemble_L4_L2" / "oof_predictions.csv")
    y = oof.label.map(CLASSES.index).values
    P = oof[[f"p_{c}" for c in CLASSES]].values

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2 * 0.62, 2.05),
                                 gridspec_kw=dict(width_ratios=[1, 1.08], wspace=0.42))
    cm = np.array(m["confusion_matrix"])
    share = cm / cm.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    a1.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a1.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] > 0.5 else "#0b0b0b")
    names = [LABEL[c] for c in CLASSES]
    a1.set_xticks(range(3), names), a1.set_yticks(range(3), names)
    a1.set_xlabel("Predicted"), a1.set_ylabel("True")
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_title("(a) Confusion matrix", fontsize=7.5)

    for k, (c, col) in enumerate(zip(CLASSES, (C1, C2, C3))):
        fpr, tpr, _ = roc_curve(y == k, P[:, k])
        a2.plot(fpr, tpr, color=col, lw=1.2, label=f"{LABEL[c]} (AUC {roc_auc_score(y == k, P[:, k]):.3f})")
    a2.plot([0, 1], [0, 1], color=GRAY, lw=0.6, ls="--")
    a2.set_xlabel("False positive rate"), a2.set_ylabel("True positive rate")
    a2.set_xlim(0, 1), a2.set_ylim(0, 1.01)
    a2.grid(color="#e4e3df", linewidth=0.5)
    a2.legend(loc="lower right", frameon=False, handlelength=1.2)
    a2.set_title("(b) One-vs-rest ROC", fontsize=7.5)
    savefig(fig, "fig5_results")


def main():
    examples_figure()
    g, n = method_figures()
    results_figure()
    pd.set_option("display.width", 200)
    print(f"method summary over {n} scans:")
    print(g.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
