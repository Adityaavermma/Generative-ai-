"""
inference.py
------------
Reusable prediction engine shared by predict.py (CLI) and app.py (Streamlit).

    predictor = Predictor()                  # loads the L4 + L2 ensemble once
    result = predictor.analyze(gray_image)   # -> Analysis

Each image is passed through the agentic denoise / enhance pipeline once per
enhancement level the ensemble needs (L4 and L2), scored by all fold models
(soft vote + horizontal-flip TTA), and checked by the out-of-distribution
guard.

Research / educational use only - not a medical device and not a diagnosis.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from common import CLASSES, to_cache_size
from fit_ood import extract, load_extractor, mahalanobis
from modeling import build_model, eval_transform, gray_to_tensor, predict_with_map
from us_pipeline import enhance

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = [ROOT / "runs" / "final_L4", ROOT / "runs" / "final_L2"]


@dataclass
class Analysis:
    probabilities: dict            # class -> probability (ensemble mean)
    prediction: str
    confidence: float
    views: dict                    # level -> {"image": uint8 HxW, "info": agent report}
    heatmap: np.ndarray            # lesion probability, CACHE_SIZE x CACHE_SIZE, [0, 1]
    ood_score: float | None        # > 1 means "unlike the training scans"
    model_votes: list = field(default_factory=list)  # per-model probabilities
    timings: dict = field(default_factory=dict)

    @property
    def in_distribution(self):
        return None if self.ood_score is None else self.ood_score <= 1.0

    def to_dict(self):
        """JSON-friendly summary (no image arrays)."""
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "ood_score": None if self.ood_score is None else round(self.ood_score, 3),
            "looks_like_training_data": self.in_distribution,
            "models": len(self.model_votes),
            "model_agreement": round(self.agreement, 3),
            "enhancement": {f"L{lvl}": {k: v for k, v in view["info"].items()}
                            for lvl, view in self.views.items()},
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
        }

    @property
    def agreement(self):
        """Share of individual models whose top class matches the ensemble."""
        if not self.model_votes:
            return 1.0
        top = CLASSES.index(self.prediction)
        return float(np.mean([int(np.argmax(v)) == top for v in self.model_votes]))


class Predictor:
    def __init__(self, runs=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"))
        self.run_dirs = [Path(r) for r in (runs or DEFAULT_RUNS)]
        self.runs = []                       # [(models, cfg)]
        for run_dir in self.run_dirs:
            cfg = json.loads((run_dir / "config.json").read_text())
            ckpts = sorted(run_dir.glob("fold*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No fold*.pt checkpoints in {run_dir}. Train first: python train.py")
            models = []
            for c in ckpts:
                m = build_model(cfg["arch"], pretrained=False)
                m.load_state_dict(torch.load(c, map_location="cpu"))
                models.append(m.to(self.device).eval())
            self.runs.append((models, cfg))
        # The OOD guard of the first run is used (fitted on that run's inputs).
        self.ood = None
        ood_file = self.run_dirs[0] / "ood.npz"
        if ood_file.exists():
            self.ood = dict(np.load(ood_file))
            self.ood["extractor_model"] = load_extractor(self.device)

    @property
    def n_models(self):
        return sum(len(m) for m, _ in self.runs)

    @property
    def device_name(self):
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return "CPU"

    def analyze(self, img: np.ndarray) -> Analysis:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        views, inputs = {}, []
        for models, cfg in self.runs:
            lvl = cfg["level"]
            if lvl not in views:
                enhanced, info = enhance(img, level=lvl)
                views[lvl] = {"image": enhanced, "info": info}
            x = eval_transform(cfg["img_size"])(
                gray_to_tensor(to_cache_size(views[lvl]["image"])).expand(3, -1, -1))
            inputs.append((x.unsqueeze(0).to(self.device), models))
        t1 = time.perf_counter()

        outs = [predict_with_map(m, x) for x, models in inputs for m in models]
        votes = [p[0].cpu().numpy() for p, _ in outs]
        probs = np.mean(votes, axis=0)
        heat = torch.stack([h[0] for _, h in outs]).mean(0).clamp(0, 1).cpu().numpy()
        ood_score = None
        if self.ood is not None:
            f = extract(self.ood["extractor_model"], inputs[0][0], self.device)
            ood_score = float(mahalanobis(f, self.ood["mean"], self.ood["precision"])[0]
                              / self.ood["threshold"])
        t2 = time.perf_counter()

        top = int(np.argmax(probs))
        return Analysis(
            probabilities={c: float(p) for c, p in zip(CLASSES, probs)},
            prediction=CLASSES[top],
            confidence=float(probs[top]),
            views=views,
            heatmap=heat,
            ood_score=ood_score,
            model_votes=[v.tolist() for v in votes],
            timings={"enhancement": t1 - t0, "models": t2 - t1, "total": t2 - t0},
        )


def heat_overlay(gray: np.ndarray, heat: np.ndarray, opacity: float = 0.6) -> np.ndarray:
    """RGB image: grayscale scan with the lesion heat-map blended on top.

    The heat-map is the absolute lesion probability (not re-scaled per image),
    so a scan with no lesion stays clean."""
    h, w = gray.shape[:2]
    heat = cv2.resize(np.clip(heat, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                           cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    alpha = (opacity * heat)[..., None]
    return (base * (1 - alpha) + colored * alpha).astype(np.uint8)


def summary_panel(raw: np.ndarray, result: Analysis, level: int = 4) -> np.ndarray:
    """BGR panel: original | denoised + enhanced | lesion heat-map, with a title."""
    h = 320

    def fit(im):
        return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)

    enhanced = result.views.get(level, next(iter(result.views.values())))["image"]
    raw_c = cv2.cvtColor(fit(raw), cv2.COLOR_GRAY2BGR)
    enh = fit(enhanced)
    enh_c = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    overlay = cv2.cvtColor(heat_overlay(enh, result.heatmap), cv2.COLOR_RGB2BGR)
    gap = np.full((h, 6, 3), 255, np.uint8)
    panel = np.hstack([raw_c, gap, enh_c, gap, overlay])

    bar = np.full((72, panel.shape[1], 3), 255, np.uint8)
    probs = "   ".join(f"{c} {p:.0%}" for c, p in result.probabilities.items())
    title = f"{result.prediction.upper()}  |  {probs}"
    if result.in_distribution is False:
        title = "UNRELIABLE - not like the training scans  |  " + title
    cv2.putText(bar, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (11, 11, 11), 2, cv2.LINE_AA)
    x = 8
    for im, caption in ((raw_c, "original"), (enh_c, "denoised + enhanced"), (overlay, "lesion heat-map")):
        cv2.putText(bar, caption, (x, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (82, 81, 78), 1, cv2.LINE_AA)
        x += im.shape[1] + 6
    return np.vstack([bar, panel])
