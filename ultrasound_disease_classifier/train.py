"""
train.py
--------
5-fold cross-validated training of a breast-ultrasound classifier
(benign / malignant / normal) on images that went through the agentic
denoise + enhance pipeline.

Every image is predicted exactly once by a model that never saw it (or any
duplicate of it), so the reported metrics are honest out-of-fold numbers.
The 5 fold models are saved and used together as an ensemble by predict.py.

No early stopping / checkpoint picking on the test fold: each fold trains for
a fixed number of epochs and the last weights are kept.

The BUSI lesion masks are used as an auxiliary training target (--seg-weight);
they are not needed at prediction time.

Usage:
    python train.py --level 4 --img-size 384 --batch-size 16 --arch convnextv2_tiny.fcmae_ft_in22k_in1k_384 --name final_L4
    python train.py --level 0 --name raw_baseline
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from timm.optim import create_optimizer_v2

from common import CLASSES
from modeling import build_model, eval_transform, gpu_augment, gray_to_tensor, predict_proba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
INDEX_CSV = ROOT / "data" / "index.csv"


def lesion_loss(seg_logits, masks):
    """BCE + soft Dice between the coarse lesion map and the downsampled mask."""
    target = F.adaptive_avg_pool2d(masks, seg_logits.shape[-2:])
    bce = F.binary_cross_entropy_with_logits(seg_logits, target)
    p = seg_logits.sigmoid()
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_ratio=0.01):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def evaluate(model, x_eval, batch=32):
    model.eval()
    probs = [predict_proba(model, x_eval[i:i + batch].contiguous(memory_format=torch.channels_last)).cpu()
             for i in range(0, len(x_eval), batch)]
    return torch.cat(probs).numpy()


def train_fold(k, df, data, args, device, run_dir):
    seed_everything(args.seed + k)
    tr = np.where(df.fold.values != k)[0]
    te = np.where(df.fold.values == k)[0]
    y = df.label.map(CLASSES.index).values
    y_dev = torch.as_tensor(y, device=device)
    x_eval = data["eval"][te].to(device)      # exactly predict.py's preprocessing
    steps_per_epoch = len(tr) // args.batch_size
    order = torch.Generator().manual_seed(args.seed + k)

    model = build_model(args.arch, pretrained=True, drop_path=args.drop_path)
    model = model.to(device).to(memory_format=torch.channels_last)

    # Class weights ~ 1/sqrt(frequency) to keep the minority classes visible.
    counts = np.bincount(y[tr], minlength=len(CLASSES))
    w = 1 / np.sqrt(counts)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)

    optimizer = create_optimizer_v2(model, "adamw", lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * steps_per_epoch
    scheduler = cosine_with_warmup(optimizer, warmup_steps=2 * steps_per_epoch, total_steps=steps)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        perm = torch.as_tensor(tr)[torch.randperm(len(tr), generator=order)]
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size].to(device)
            x, mb = gpu_augment(data["train"][idx], data["mask"][idx], args.img_size)
            x = x.contiguous(memory_format=torch.channels_last)
            yb = y_dev[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if args.seg_weight > 0:
                    logits, seg = model(x, with_seg=True)
                    loss = (criterion(logits.float(), yb)
                            + args.seg_weight * lesion_loss(seg.float(), mb))
                else:
                    loss = criterion(model(x).float(), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(yb)
            n += len(yb)

        rec = {"fold": k, "epoch": epoch, "train_loss": running / n}
        # Monitoring only - never used to pick weights.
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rec["test_acc"] = accuracy_score(y[te], evaluate(model, x_eval).argmax(1))
        history.append(rec)
        msg = f"fold {k} | epoch {epoch:2d}/{args.epochs} | loss {rec['train_loss']:.4f}"
        if "test_acc" in rec:
            msg += f" | fold acc {rec['test_acc']:.3f}"
        print(msg + f" | {time.time() - t0:.1f}s", flush=True)

    probs = evaluate(model, x_eval)
    del x_eval
    state = {kk: v.half() if v.is_floating_point() else v for kk, v in model.state_dict().items()}
    torch.save(state, run_dir / f"fold{k}.pt")
    oof = pd.DataFrame({"id": df.id.values[te], "fold": k, "label": df.label.values[te]})
    for c, name in enumerate(CLASSES):
        oof[f"p_{name}"] = probs[:, c]
    oof["pred"] = [CLASSES[i] for i in probs.argmax(1)]
    return oof, history


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def compute_metrics(oof):
    y = oof.label.map(CLASSES.index).values
    P = oof[[f"p_{c}" for c in CLASSES]].values
    pred = P.argmax(1)
    mal = CLASSES.index("malignant")
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    tp = cm[mal, mal]
    fn = cm[mal].sum() - tp
    fp = cm[:, mal].sum() - tp
    tn = cm.sum() - tp - fn - fp
    fold_acc = oof.groupby("fold").apply(lambda d: (d.label == d.pred).mean(), include_groups=False)
    return {
        "n_images": int(len(oof)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_auc_ovr": float(roc_auc_score(y, P, multi_class="ovr")),
        "malignant_sensitivity": float(tp / (tp + fn)),
        "malignant_specificity": float(tn / (tn + fp)),
        "malignant_auc": float(roc_auc_score(y == mal, P[:, mal])),
        "fold_accuracy_mean": float(fold_acc.mean()),
        "fold_accuracy_std": float(fold_acc.std()),
        "fold_accuracy": {int(k): float(v) for k, v in fold_acc.items()},
        "per_class": classification_report(y, pred, target_names=CLASSES, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion(cm, path, title):
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            dark = norm[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=11, color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted", color=INK2)
    ax.set_ylabel("True", color=INK2)
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_history(hist, path):
    h = pd.DataFrame(hist)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    mean_loss = h.groupby("epoch").train_loss.mean()
    axes[0].plot(mean_loss.index, mean_loss.values, color="#2a78d6", linewidth=2)
    axes[0].set_title("Training loss (mean of 5 folds)", color=INK, fontsize=10, loc="left")
    axes[0].set_xlabel("Epoch", color=INK2)
    acc = h.dropna(subset=["test_acc"]).groupby("epoch").test_acc.mean()
    axes[1].plot(acc.index, acc.values, color="#2a78d6", linewidth=2, marker="o", markersize=4)
    axes[1].set_title("Held-out fold accuracy (mean of 5 folds, monitoring only)",
                      color=INK, fontsize=10, loc="left")
    axes[1].set_xlabel("Epoch", color=INK2)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4, help="enhancement level (0 = raw)")
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seg-weight", type=float, default=1.0,
                    help="weight of the auxiliary lesion-mask loss (0 = classification only)")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="subset of folds to run")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="run folder name (default: L<level>_<arch>)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available - training on CPU will be slow.")
    # cudnn.benchmark re-tunes every new input shape (each fold's eval batches
    # differ), which cost ~40 s per shape here for no speed-up - keep it off.
    torch.backends.cudnn.benchmark = False

    df = pd.read_csv(INDEX_CSV)
    col = f"L{args.level}_path"
    if col not in df:
        raise SystemExit(f"Level {args.level} not prepared. Run: python prepare_data.py --levels {args.level}")
    images = torch.stack([gray_to_tensor(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in df[col]])
    masks = torch.stack([gray_to_tensor((cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))
                         for p in df.mask_path])
    # Training crops are taken from images pre-shrunk (with antialiasing) to
    # ~1.125x the training size; evaluation uses the same resize as predict.py.
    base = min(images.shape[-1], round(args.img_size * 1.125))
    data = {
        "train": F.interpolate(images.float() / 255, size=(base, base), mode="bilinear",
                               antialias=True).half().to(device),
        "mask": F.interpolate(masks.float(), size=(base, base), mode="nearest").half().to(device),
        "eval": eval_transform(args.img_size)(images.expand(-1, 3, -1, -1)),
    }

    name = args.name or f"L{args.level}_{args.arch.split('.')[0]}"
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"classes": CLASSES, "name": name}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    folds = args.folds if args.folds is not None else sorted(int(f) for f in df.fold.unique())
    print(f"Run '{name}': level L{args.level}, {len(df)} images, folds {folds}, device {device}")
    oofs, hist = [], []
    t0 = time.time()
    for k in folds:
        oof, h = train_fold(k, df, data, args, device, run_dir)
        oofs.append(oof)
        hist += h
        print(f"fold {k} accuracy: {(oof.label == oof.pred).mean():.4f}")

    oof = pd.concat(oofs, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    m = compute_metrics(oof)
    m["train_minutes"] = round((time.time() - t0) / 60, 1)
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    plot_confusion(m["confusion_matrix"], run_dir / "confusion_matrix.png",
                   f"Out-of-fold confusion matrix - L{args.level} (n={m['n_images']})")
    plot_history(hist, run_dir / "training_curves.png")

    print("\n================ OUT-OF-FOLD RESULTS ================")
    print(f"accuracy           {m['accuracy']:.4f}  (per fold {m['fold_accuracy_mean']:.4f} +/- {m['fold_accuracy_std']:.4f})")
    print(f"balanced accuracy  {m['balanced_accuracy']:.4f}")
    print(f"macro F1           {m['macro_f1']:.4f}")
    print(f"macro AUC (OvR)    {m['macro_auc_ovr']:.4f}")
    print(f"malignant sens/spec {m['malignant_sensitivity']:.4f} / {m['malignant_specificity']:.4f}  AUC {m['malignant_auc']:.4f}")
    print("confusion matrix (rows=true, cols=pred):", CLASSES)
    print(np.array(m["confusion_matrix"]))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
