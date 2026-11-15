"""
predict.py
----------
Command-line prediction: classify breast ultrasound images as
benign / malignant / normal (for the web UI run: streamlit run app.py).

Each input image is passed through the agentic denoise + enhance pipeline
and scored by an ensemble (soft vote, horizontal-flip TTA) of
  * 5 fold models trained on L4 images (denoised + enhanced), and
  * 5 fold models trained on L2 images (denoised),
i.e. the two views produced by the same pipeline.

Usage:
    python predict.py path/to/scan.png
    python predict.py img1.png img2.png --outdir predictions
    python predict.py scan.png --json
    python predict.py scan.png --runs runs/final_L4          # single run

With --outdir, a panel is saved per image: original | denoised + enhanced |
lesion heat-map (where the model thinks the lesion is).

Research / educational use only - not a medical device and not a diagnosis.
"""

import argparse
import json
from pathlib import Path

import cv2

from inference import DEFAULT_RUNS, Predictor, summary_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--runs", nargs="+", default=[str(r) for r in DEFAULT_RUNS],
                    help="training run folder(s); their predictions are averaged")
    ap.add_argument("--outdir", default=None, help="save original/enhanced/heat-map panels here")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()

    predictor = Predictor(args.runs)
    if predictor.ood is None:
        print("note: no ood.npz in the first run folder - out-of-distribution check disabled "
              f"(run: python fit_ood.py --run {args.runs[0]})")

    results = []
    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Could not read image: {path}")
        a = predictor.analyze(img)
        r = {"image": path, **a.to_dict()}
        if args.outdir:
            out = Path(args.outdir)
            out.mkdir(parents=True, exist_ok=True)
            r["panel"] = str(out / f"{Path(path).stem}_prediction.png")
            cv2.imwrite(r["panel"], summary_panel(img, a))
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2, default=float))
        return
    for r in results:
        print(f"\n{r['image']}")
        if r["looks_like_training_data"] is False:
            print(f"  WARNING    : this image does not look like a breast ultrasound from the "
                  f"training data (OOD score {r['ood_score']:.2f} > 1). The prediction below "
                  f"is unreliable.")
        print(f"  prediction : {r['prediction'].upper()}  ({r['confidence']:.1%}, "
              f"{r['model_agreement']:.0%} of {r['models']} models agree)")
        for c, p in r["probabilities"].items():
            print(f"    {c:<10s} {p:6.1%}  {'#' * int(round(p * 30))}")
        first = next(iter(r["enhancement"].values()))
        print(f"  enhancement: {first['strategy']}")
        if "panel" in r:
            print(f"  panel      : {r['panel']}")
    print("\nResearch use only - not a medical diagnosis.")


if __name__ == "__main__":
    main()
