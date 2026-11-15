"""
ensemble_eval.py
----------------
Out-of-fold evaluation of an ensemble of finished runs (soft vote: the class
probabilities are averaged). All runs share the same duplicate-aware folds,
so for every image all averaged models never saw it - the result is still an
honest out-of-fold estimate.

Usage:
    python ensemble_eval.py runs/final_L4 runs/final_L2 --out runs/ensemble
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import CLASSES
from train import compute_metrics, plot_confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = [f"p_{c}" for c in CLASSES]
    oofs = [pd.read_csv(Path(r) / "oof_predictions.csv").set_index("id") for r in args.runs]
    base = oofs[0]
    probs = sum(o.loc[base.index, cols].values for o in oofs) / len(oofs)
    oof = base[["fold", "label"]].copy()
    oof[cols] = probs
    oof["pred"] = [CLASSES[i] for i in probs.argmax(1)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oof.reset_index().to_csv(out / "oof_predictions.csv", index=False)
    m = compute_metrics(oof.reset_index())
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    cfgs = [json.loads((Path(r) / "config.json").read_text()) for r in args.runs]
    (out / "config.json").write_text(json.dumps({
        "name": out.name, "members": [c["name"] for c in cfgs],
        "level": "+".join(f"L{c['level']}" for c in cfgs), "arch": cfgs[0]["arch"],
        "img_size": cfgs[0]["img_size"], "seg_weight": cfgs[0].get("seg_weight", 0),
    }, indent=2))
    plot_confusion(m["confusion_matrix"], out / "confusion_matrix.png",
                   f"Out-of-fold confusion - ensemble {'+'.join(f'L{c['level']}' for c in cfgs)} (n={m['n_images']})")
    print(f"ensemble of {[c['name'] for c in cfgs]}: accuracy {m['accuracy']:.4f}, "
          f"balanced {m['balanced_accuracy']:.4f}, macro AUC {m['macro_auc_ovr']:.4f}, "
          f"malignant sens/spec {m['malignant_sensitivity']:.4f}/{m['malignant_specificity']:.4f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
