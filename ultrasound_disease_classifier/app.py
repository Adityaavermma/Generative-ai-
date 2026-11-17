"""
app.py - Streamlit UI for the breast-ultrasound classifier.

    streamlit run app.py          (from this folder)
    python run_app.py             (from anywhere)

Research / educational use only - not a medical device and not a diagnosis.
"""

import dataclasses
import io
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import altair as alt  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import CLASSES  # noqa: E402
from enhancement_compare import (CLASSIFIER_EVIDENCE, CRITERIA, FAMILY_AGENT,  # noqa: E402
                                 FAMILY_DENOISE, FAMILY_ENHANCE, run_all, score)
from inference import Analysis, Predictor, heat_overlay, summary_panel  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "Dataset_BUSI_with_GT"

st.set_page_config(
    page_title="Breast Ultrasound AI",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
)

# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------

CLASS_STYLE = {
    "benign": dict(label="Benign", color="orange", icon=":material/radio_button_checked:",
                   blurb="Lesion pattern the model associates with benign findings "
                         "(e.g. cysts, fibroadenomas)."),
    "malignant": dict(label="Malignant", color="red", icon=":material/error:",
                      blurb="Lesion pattern the model associates with malignant tumours."),
    "normal": dict(label="Normal", color="green", icon=":material/check_circle:",
                   blurb="No lesion pattern found by the model."),
}

PALETTE = {
    "light": dict(primary="#2a78d6", muted="#c3c2b7", text="#0b0b0b", text2="#52514e",
                  grid="#e4e3df", surface="#fcfcfb", node="#f3f2ef"),
    "dark": dict(primary="#3987e5", muted="#5d5c56", text="#f0efec", text2="#c3c2b7",
                 grid="#383835", surface="#1a1a19", node="#252523"),
}
CHART_FONT = "\"Source Sans\", \"Source Sans Pro\", sans-serif"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SAMPLES = {
    "Malignant": RAW / "malignant" / "malignant (20).png",
    "Benign": RAW / "benign" / "benign (10).png",
    "Normal": RAW / "normal" / "normal (10).png",
    "Fetal (out of scope)": ROOT.parents[1] / "agentic_us_denoise" / "scan.jpg",
}
SAMPLES = {k: v for k, v in SAMPLES.items() if v.exists()}

st.html("""
<style>
  .block-container { padding-top: 3.6rem; padding-bottom: 3rem; }
  [data-testid="stImage"] img { border-radius: 0.5rem; }
  .hero-kicker { font-size: 0.8rem; letter-spacing: .08em; text-transform: uppercase;
                 opacity: .65; margin-bottom: .1rem; }
  .pred-label { font-size: 2.1rem; font-weight: 700; line-height: 1.1; margin: .1rem 0 .3rem 0; }
  .muted { opacity: .7; font-size: .9rem; }
  .step-arrow { text-align: center; font-size: 1.4rem; opacity: .45; padding-top: 38%; }
  [data-testid="stMain"] h1 { font-size: clamp(1.75rem, 1.2rem + 1.6vw, 2.6rem) !important;
                              line-height: 1.15 !important; }
  /* Responsive rows: children wrap onto a new line instead of being squeezed. */
  .st-key-kpis > *, .st-key-card_metrics > * { flex: 1 1 140px !important; min-width: 140px; }
  .st-key-result_row > *:has(.st-key-viewer) { flex: 1.45 1 360px !important; min-width: 300px; }
  .st-key-result_row > *:has(.st-key-card) { flex: 1 1 300px !important; min-width: 280px; }
  .st-key-perf_row > * { flex: 1 1 320px !important; min-width: 280px; }
  .st-key-how_row > * { flex: 1 1 220px !important; min-width: 200px; }
  .st-key-input_row > *:has(.st-key-upload_box) { flex: 1.4 1 320px !important; min-width: 280px; }
  .st-key-input_row > *:has(.st-key-sample_box) { flex: 1 1 400px !important; min-width: 280px; }
  .st-key-cmp_win_row > *:has(.st-key-cmp_win_imgs) { flex: 1.4 1 360px !important; min-width: 280px; }
  .st-key-cmp_win_row > *:has(.st-key-cmp_win_card) { flex: 1 1 280px !important; min-width: 260px; }
  .st-key-cmp_chart_row > * { flex: 1 1 380px !important; min-width: 280px; }
  .st-key-cmp_win_metrics > *, .st-key-cmp_evidence > * { flex: 1 1 120px !important; min-width: 110px; }
  @media (max-width: 640px) { .step-arrow { display: none; } }
</style>
""")


def theme():
    t = getattr(st.context, "theme", None)
    return PALETTE["dark" if t is not None and t.type == "dark" else "light"]


# --------------------------------------------------------------------------
# Engine (cached)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the 10-model ensemble and the agentic pipeline…")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data(show_spinner=False, max_entries=200)
def _analyze_cached(data: bytes):
    # Cache plain data, not the Analysis object: when Streamlit hot-reloads a
    # module, an object of the old class can no longer be pickled.
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return img, dataclasses.asdict(get_predictor().analyze(img))


def run_analysis(data: bytes):
    img, fields = _analyze_cached(data)
    return (img, Analysis(**fields)) if fields is not None else (None, None)


@st.cache_data(show_spinner=False, max_entries=20)
def run_comparison(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return run_all(img)


@st.cache_data
def load_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


@st.cache_data
def load_runs():
    rows = []
    for mf in sorted((ROOT / "runs").glob("*/metrics.json")):
        cfg = json.loads((mf.parent / "config.json").read_text())
        m = json.loads(mf.read_text())
        arch = "ConvNeXt-V2-Tiny" if "convnextv2" in cfg["arch"] else "ConvNeXt-Tiny"
        level = cfg["level"]
        inp = {0: "Raw", 2: "Denoised", 4: "Denoised + enhanced"}.get(level, f"{level}")
        if "members" in cfg:
            inp = ("Raw, 2-seed ensemble" if set(str(cfg["level"]).split("+")) == {"L0"}
                   else "Ensemble of both views")
        elif cfg.get("seed", 42) != 42:
            inp += f" (seed {cfg['seed']})"
        extra = " + lesion loss" if cfg.get("seg_weight", 0) > 0 else ""
        rows.append({
            "run": cfg["name"], "input": inp,
            "model": f"{arch} {cfg['img_size']}px{extra}",
            "label": f"{inp} · {arch} {cfg['img_size']}px{extra}",
            "accuracy": m["accuracy"], "fold_sd": m["fold_accuracy_std"],
            "balanced_accuracy": m["balanced_accuracy"], "macro_auc": m["macro_auc_ovr"],
            "mal_sens": m["malignant_sensitivity"], "mal_spec": m["malignant_specificity"],
            "final": cfg["name"] == "ensemble_L4_L2",   # the deployed model
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

# Charts are rendered with theme="streamlit": axes, legends, text, fonts and
# the categorical palette (chartCategoricalColors in .streamlit/config.toml)
# follow the live light/dark theme in the browser. Only theme-independent
# colours are hard-coded here.
PRIMARY = "#2a78d6"
MUTED = "#8f8e86"      # mid-gray: >= 3:1 against both the light and dark surface
FAMILY_ORDER = [FAMILY_DENOISE, FAMILY_ENHANCE, FAMILY_AGENT]   # palette slots 1-3


def _axis(**kw):
    kw.setdefault("labelOverlap", False)   # never silently drop category labels
    return alt.Axis(domain=False, ticks=False, **kw)


# Text placement rule: every piece of text is either an axis label (themed by
# Streamlit) or white text inside a coloured mark, so it stays readable in
# light and dark mode without the server knowing which theme is active.

def probability_chart(a: Analysis):
    df = pd.DataFrame({
        "label": [f"{CLASS_STYLE[c]['label']} · {a.probabilities[c]:.1%}" for c in CLASSES],
        "class": [CLASS_STYLE[c]["label"] for c in CLASSES],
        "probability": [a.probabilities[c] for c in CLASSES],
        # No highlighted "winner" for an out-of-scope image.
        "predicted": [c == a.prediction and a.in_distribution is not False for c in CLASSES],
    })
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20).encode(
        y=alt.Y("label:N", sort=df.label.tolist(), title=None, axis=_axis(labelFontSize=13)),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                axis=_axis(format="%", tickCount=5, grid=True)),
        color=alt.condition("datum.predicted", alt.value(PRIMARY), alt.value(MUTED)),
        tooltip=[alt.Tooltip("class:N", title="Class"),
                 alt.Tooltip("probability:Q", title="Probability", format=".1%")],
    )
    return bars.properties(height=130).configure_view(stroke=None)


def confusion_chart(cm):
    cm = np.asarray(cm)
    labels = [CLASS_STYLE[c]["label"] for c in CLASSES]
    rows = []
    for i, t in enumerate(labels):
        for j, pr in enumerate(labels):
            rows.append({"true": t, "pred": pr, "count": int(cm[i, j]),
                         "share": cm[i, j] / cm[i].sum()})
    df = pd.DataFrame(rows)
    df["text"] = df.apply(lambda r: f"{r['count']}\n{r['share']:.0%}", axis=1)
    gap = alt.Scale(paddingInner=0.04)   # gaps between cells show the page surface
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=labels, title="Predicted", scale=gap,
                axis=_axis(labelAngle=0, orient="bottom")),
        y=alt.Y("true:N", sort=labels, title="True", scale=gap, axis=_axis()),
    )
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=BLUE_RAMP), legend=None),
        tooltip=[alt.Tooltip("true:N", title="True"), alt.Tooltip("pred:N", title="Predicted"),
                 alt.Tooltip("count:Q", title="Images"),
                 alt.Tooltip("share:Q", title="Share of true class", format=".1%")],
    )
    # Text sits on the cell colour (light cells -> dark text, dark cells -> white).
    text = base.mark_text(fontSize=14, fontWeight=600, lineBreak="\n", lineHeight=18).encode(
        text="text:N",
        color=alt.condition("datum.share > 0.5", alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (heat + text).properties(height=300).configure_view(stroke=None)


def comparison_chart(df):
    df = df.copy()
    df["lo"] = df.accuracy - df.fold_sd
    df["hi"] = df.accuracy + df.fold_sd
    order = df.label.tolist()
    # Two-line axis labels ("input" / "model") keep the label column narrow.
    y = alt.Y("label:N", sort=order, title=None,
              axis=_axis(labelExpr="split(datum.label, ' · ')", labelLimit=230, labelFontSize=12,
                         labelLineHeight=15, labelPadding=8))
    x_scale = alt.Scale(domain=[0.7, 1.0])
    tooltip = [alt.Tooltip("label:N", title="Run"),
               alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
               alt.Tooltip("fold_sd:Q", title="Fold sd", format=".1%"),
               alt.Tooltip("balanced_accuracy:Q", title="Balanced acc.", format=".1%"),
               alt.Tooltip("macro_auc:Q", title="Macro AUC", format=".3f")]
    base = alt.Chart(df)
    bars = base.mark_bar(cornerRadiusEnd=4, height=20).encode(
        y=y,
        x=alt.X("accuracy:Q", scale=x_scale, title="Out-of-fold accuracy (axis starts at 70%)",
                axis=_axis(format="%", tickCount=4, grid=True)),
        x2=alt.datum(0.7),  # bars start at the axis origin (70%), not at 0
        color=alt.condition("datum.final", alt.value(PRIMARY), alt.value(MUTED)),
        tooltip=tooltip,
    )
    err = base.mark_rule(color=MUTED, strokeWidth=1.5).encode(
        y=y, x=alt.X("lo:Q", scale=x_scale), x2="hi:Q")
    values = base.mark_text(align="left", dx=6, fontSize=12, fontWeight=700, color="#ffffff").encode(
        y=y, x=alt.datum(0.7), text=alt.Text("accuracy:Q", format=".1%"))
    return (bars + err + values).properties(height=44 * len(df) + 30).configure_view(stroke=None)


def _family_color():
    # Domain only: the colours come from the theme's categorical palette, in
    # this fixed order, so they follow light / dark mode.
    return alt.Color("family:N", title=None, scale=alt.Scale(domain=FAMILY_ORDER),
                     legend=alt.Legend(orient="bottom", labelFontSize=12, symbolType="circle",
                                       symbolSize=120))


def tradeoff_chart(df, label_keys):
    """Speckle removed (x) vs anatomy kept (y): the ideal method is top-right.
    The top methods are drawn larger with their rank number inside."""
    d = df[df.key != "original"].copy()
    tooltip = [alt.Tooltip("method:N", title="Method"), alt.Tooltip("family:N", title="Family"),
               alt.Tooltip("score:Q", title="Score", format=".0f"),
               alt.Tooltip("speckle_reduction:Q", title="Speckle removed", format=".0%"),
               alt.Tooltip("anatomy_kept:Q", title="Anatomy kept", format=".2f"),
               alt.Tooltip("tone_kept:Q", title="Tone kept", format=".2f")]
    x_max = max(0.55, float(d.speckle_reduction.max()) + 0.05)
    x = alt.X("speckle_reduction:Q", title="Speckle removed →", scale=alt.Scale(domain=[0, x_max]),
              axis=_axis(format="%", grid=True))
    y = alt.Y("anatomy_kept:Q", title="Anatomy kept →", scale=alt.Scale(domain=[0, 1.05]),
              axis=_axis(grid=True, tickCount=5))
    rest = alt.Chart(d[~d.key.isin(label_keys)]).mark_circle(size=110, opacity=0.85).encode(
        x=x, y=y, color=_family_color(), tooltip=tooltip)
    top = d[d.key.isin(label_keys)].copy()
    top["n"] = [str(label_keys.index(k) + 1) for k in top.key]
    top_pts = alt.Chart(top).mark_circle(size=420, opacity=1).encode(
        x=x, y=y, color=_family_color(), tooltip=tooltip)
    top_num = alt.Chart(top).mark_text(fontSize=12, fontWeight=700, color="#ffffff").encode(
        x=x, y=y, text="n:N", tooltip=tooltip)
    return (rest + top_pts + top_num).properties(height=340).configure_view(stroke=None)


def ranking_chart(df):
    d = df[df.key != "original"].copy()
    order = d.method.tolist()
    y = alt.Y("method:N", sort=order, title=None, axis=_axis(labelLimit=190))
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16).encode(
        y=y, x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="Score (% of the best method)",
                     axis=_axis(grid=True, values=[0, 25, 50, 75, 100])),
        color=_family_color(),
        tooltip=[alt.Tooltip("method:N", title="Method"), alt.Tooltip("score:Q", title="Score", format=".0f")])
    # Score printed inside the bar end (white); bars too short for it show the
    # number on hover only.
    text = alt.Chart(d[d.score >= 12]).mark_text(
        align="right", dx=-5, fontSize=11.5, fontWeight=600, color="#ffffff").encode(
        y=y, x="score:Q", text=alt.Text("score:Q", format=".0f"))
    return (bars + text).properties(height=26 * len(d) + 50).configure_view(stroke=None)


def method_noise_image(original, out):
    """What a filter removed: mid-gray = nothing, darker/brighter = removed signal."""
    res = original.astype(np.float32) - out.astype(np.float32)
    return np.clip(128 + 3 * res, 0, 255).astype(np.uint8)


def colorbar(width=256, height=10):
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return cv2.cvtColor(cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def to_png(img_bgr_or_gray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    return buf.tobytes() if ok else b""


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

ensemble = load_json(ROOT / "runs" / "ensemble_L4_L2" / "metrics.json")

with st.sidebar:
    st.markdown("### :material/biotech: Breast Ultrasound AI")
    st.caption("Agentic denoising + deep-learning lesion classification")

    st.markdown("**Display**")
    opacity = st.slider("Heat-map opacity", 0.0, 1.0, 0.6, 0.05,
                        help="How strongly the lesion heat-map is blended over the scan.")
    st.image(colorbar(), caption="Lesion probability  0 → 1", width="stretch")

    st.divider()
    st.markdown("**Engine**")
    # Filled after the page header has rendered, so the first (slow) model
    # load never leaves the main area blank.
    engine_slot = st.empty()
    engine_slot.caption("Loading models…")

    if ensemble:
        st.divider()
        st.markdown("**Validated performance**")
        st.caption(f"5-fold CV · {ensemble['n_images']} BUSI images · out-of-fold")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{ensemble['accuracy']:.1%}")
        c2.metric("Macro AUC", f"{ensemble['macro_auc_ovr']:.3f}")

    st.divider()
    st.caption(":material/info: Research and education only. Not a medical device; "
               "results are not a diagnosis.")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.html('<div class="hero-kicker">Agentic denoising · deep learning · benign / malignant / normal</div>')
st.title("Breast ultrasound lesion classifier", anchor=False)
st.markdown(
    '<span class="muted">Every scan is cleaned by the multi-agent denoise / enhance pipeline, '
    "then scored by a 10-model ConvNeXt-V2 ensemble that also maps where the lesion is.</span>",
    unsafe_allow_html=True,
)

# on_change="rerun" lets a tab know whether it is open (.open), so the
# method comparison only runs when someone looks at it.
tab_analyze, tab_compare, tab_perf, tab_how = st.tabs([
    ":material/search: Analyze", ":material/compare: Compare methods",
    ":material/insights: Model performance", ":material/account_tree: How it works",
], on_change="rerun", key="main_tabs")
current_scan = None  # (name, raw image, file bytes) of the scan shown in Analyze

try:
    predictor = get_predictor()
    engine_slot.markdown(
        f":material/memory: {predictor.device_name}  \n"
        f":material/stacks: {predictor.n_models} models (2 views × 5 folds)  \n"
        f":material/shield: Out-of-scope guard {'on' if predictor.ood is not None else 'off'}"
    )
except FileNotFoundError as e:
    predictor = None
    engine_slot.error(f"Model weights not found.\n\n{e}")


# --------------------------------------------------------------------------
# Analyze tab
# --------------------------------------------------------------------------

def set_mode(mode):
    st.session_state.mode = mode


def render_result(name: str, raw: np.ndarray, a: Analysis):
    style = CLASS_STYLE[a.prediction]
    l4 = a.views.get(4, next(iter(a.views.values())))
    l2 = a.views.get(2, l4)
    overlay = heat_overlay(l4["image"], a.heatmap, opacity)

    row = st.container(horizontal=True, wrap=True, gap="large", key="result_row")
    left = row.container(key="viewer")
    right = row.container(key="card")
    with left:
        views = {
            "Lesion heat-map": overlay,
            "Original": raw,
            "Denoised": l2["image"],
            "Denoised + enhanced": l4["image"],
        }
        choice = st.segmented_control("View", list(views), default="Lesion heat-map",
                                      key=f"view_{name}", label_visibility="collapsed")
        st.image(views[choice or "Lesion heat-map"], width="stretch")
        if (choice or "Lesion heat-map") == "Lesion heat-map":
            st.caption("Brighter = higher lesion probability from the model's lesion head "
                       "(trained on expert lesion masks). Shows where the model looks, not an "
                       "exact outline.")

    with right:
        with st.container(border=True):
            st.html('<div class="hero-kicker">Model output</div>')
            if a.in_distribution is False:
                # Never headline a breast diagnosis for an image the model doesn't know.
                st.html('<div class="pred-label">Out of scope</div>')
                st.badge(f"Forced guess: {style['label'].lower()} · {a.confidence:.0%}",
                         icon=":material/block:", color="gray")
                st.warning(
                    f"**This does not look like a breast ultrasound from the training data** "
                    f"(out-of-scope score {a.ood_score:.2f} > 1.0). The probabilities below are "
                    "meaningless here — the model only knows breast tissue.",
                    icon=":material/warning:",
                )
            else:
                st.html(f'<div class="pred-label">{style["label"]}</div>')
                st.badge(f"{a.confidence:.1%} confidence", icon=style["icon"], color=style["color"])
                st.caption(style["blurb"])

            if a.in_distribution is not False and (a.confidence < 0.6 or a.agreement < 0.8):
                st.info("Low certainty: the models disagree on this scan. Treat the result "
                        "with extra caution.", icon=":material/help:")

            st.altair_chart(probability_chart(a), width="stretch", theme="streamlit")

            m1 = m2 = m3 = st.container(horizontal=True, wrap=True, key="card_metrics")
            m1.metric("Models agree", f"{round(a.agreement * len(a.model_votes))}/{len(a.model_votes)}",
                      help="How many of the individual models pick the same class as the ensemble.")
            if a.ood_score is not None:
                m2.metric("Scope score", f"{a.ood_score:.2f}",
                          help="Distance to the training scans; above 1.0 means the image is "
                               "unlike anything the model was trained on.")
            m3.metric("Time", f"{a.timings['total']:.1f} s",
                      help=f"Agentic enhancement {a.timings['enhancement']:.1f} s, "
                           f"models {a.timings['models']:.1f} s")

            d1, d2 = st.columns(2)
            stem = Path(name).stem
            d1.download_button(":material/image: Panel (PNG)", to_png(summary_panel(raw, a)),
                               file_name=f"{stem}_prediction.png", mime="image/png",
                               width="stretch", key=f"png_{name}")
            d2.download_button(":material/description: Report (JSON)",
                               json.dumps({"image": name, **a.to_dict()}, indent=2, default=float),
                               file_name=f"{stem}_report.json", mime="application/json",
                               width="stretch", key=f"json_{name}")

    st.markdown("##### :material/conversion_path: What the pipeline did")
    steps = [(raw, "1 · Original"), (l2["image"], "2 · Denoised (agent plan, L2)"),
             (l4["image"], "3 · Denoised + enhanced (L4)"), (overlay, "4 · Lesion heat-map")]
    cols = st.columns([1, 0.12, 1, 0.12, 1, 0.12, 1])
    for k, (img, cap) in enumerate(steps):
        cols[2 * k].image(img, caption=cap, width="stretch")
        if k < 3:
            cols[2 * k + 1].html('<div class="step-arrow">→</div>')

    with st.expander(":material/smart_toy: Agent reports", expanded=False):
        info, rep = l4["info"], l4["info"].get("image_report", {})
        ev, imp = l4["info"].get("evaluation_metrics", {}), l4["info"].get("improvement_pct", {})
        a1, a2, a3 = st.columns(3, border=True)
        with a1:
            st.markdown("**Agent 1 · Image analysis**")
            st.markdown(
                f"Noise: **{rep.get('noise_level', '–')}**  \n"
                f"Contrast: **{rep.get('contrast', '–')}**  \n"
                f"Edges: **{rep.get('edge_strength', '–')}**  \n"
                f"Artifacts: **{'yes' if rep.get('artifacts_present') else 'no'}**  \n"
                f"Overall quality: **{rep.get('overall_quality', '–')}**"
            )
        with a2:
            st.markdown("**Agent 2 · Processing plan**")
            for lvl, view in sorted(a.views.items()):
                name_ = "Denoised" if lvl == 2 else "Denoised + enhanced"
                st.markdown(f"{name_} (L{lvl})  \n" + " → ".join(f"`{s}`" for s in view["info"]["pipeline"]))
        with a3:
            st.markdown("**Agent 3 · Evaluation (L4)**")
            if ev:
                def change(key, unit="", fmt=".1f"):
                    return (f"{key}: **{rep[key]:{fmt}} → {ev[key]:{fmt}}{unit}** "
                            f"({imp.get(key, 0):+.0f}%)")
                st.markdown(change("SNR", " dB") + "  \n" + change("CNR", "", ".2f") + "  \n"
                            + change("Entropy", " bits", ".2f") + "  \n"
                            f"Edge preservation: **{ev['EdgePreservation']:.2f}**")
                st.caption(f"Verdict: {info['quality_status']} after {info['replans']} re-plan(s). "
                           "With no clean reference Agent 3 compares to the noisy input, so "
                           "contrast enhancement always lowers PSNR/SSIM by design.")


with tab_analyze:
    if "mode" not in st.session_state:
        st.session_state.mode = None

    input_row = st.container(horizontal=True, wrap=True, gap="large", key="input_row")
    up_col, sample_col = input_row.container(key="upload_box"), input_row.container(key="sample_box")
    with up_col:
        uploads = st.file_uploader(
            "Upload breast ultrasound image(s)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True, key="uploads",
            on_change=set_mode, args=("upload",),
            help="Grayscale B-mode breast ultrasound. Several files are analysed as a batch.",
        )
    with sample_col:
        sample = None
        if SAMPLES:
            sample = st.pills("…or try a sample", list(SAMPLES), selection_mode="single",
                              key="sample", on_change=set_mode, args=("sample",))
            st.caption("Samples come from the training data, so they show the output format, "
                       "not accuracy. The fetal scan demonstrates the out-of-scope warning.")

    sources = []
    if st.session_state.mode == "upload" and uploads:
        sources = [(f.name, f.getvalue()) for f in uploads]
    elif st.session_state.mode == "sample" and sample:
        sources = [(SAMPLES[sample].name, SAMPLES[sample].read_bytes())]
    elif uploads:
        sources = [(f.name, f.getvalue()) for f in uploads]

    if predictor is None:
        st.stop()

    if not sources:
        with st.container(border=True):
            st.markdown("#### :material/upload_file: Start by adding a scan")
            s1, s2, s3 = st.columns(3)
            s1.markdown(":material/counter_1: **Upload** one or more breast ultrasound images "
                        "(PNG, JPG, BMP, TIFF), or pick a sample.")
            s2.markdown(":material/counter_2: The **agents** analyse noise and contrast, plan "
                        "and run denoising + enhancement.")
            s3.markdown(":material/counter_3: **10 models** vote on benign / malignant / normal "
                        "and highlight the suspected lesion.")
    else:
        results = []
        progress = st.progress(0.0, text="Analysing…") if len(sources) > 1 else None
        with st.spinner("Running the agentic pipeline and the model ensemble…"):
            for i, (name, data) in enumerate(sources):
                raw, a = run_analysis(data)
                if raw is None:
                    st.error(f"Could not read **{name}** as an image.", icon=":material/broken_image:")
                else:
                    results.append((name, raw, a, data))
                if progress:
                    progress.progress((i + 1) / len(sources), text=f"Analysed {i + 1}/{len(sources)}")
        if progress:
            progress.empty()

        if len(results) > 1:
            st.markdown(f"##### :material/table_rows: Batch results · {len(results)} images")
            table = pd.DataFrame([{
                "Image": n,
                "Prediction": ("Out of scope" if a.in_distribution is False
                               else CLASS_STYLE[a.prediction]["label"]),
                "Confidence": a.confidence,
                "P(malignant)": a.probabilities["malignant"],
                "Models agree": f"{round(a.agreement * len(a.model_votes))}/{len(a.model_votes)}",
                "In scope": "yes" if a.in_distribution is not False else "NO – unreliable",
            } for n, _, a, _ in results])
            event = st.dataframe(
                table, hide_index=True, width="stretch", on_select="rerun",
                selection_mode="single-row", key="batch_table",
                column_config={
                    "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                    "P(malignant)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                },
            )
            bar = st.container(horizontal=True, wrap=True, vertical_alignment="center")
            bar.caption("Select a row to inspect that image below.")
            bar.space("stretch")
            bar.download_button(":material/download: Results (CSV)", table.to_csv(index=False),
                                file_name="ultrasound_results.csv", mime="text/csv", width="content")
            sel = event.selection.rows[0] if event.selection.rows else 0
            st.divider()
        else:
            sel = 0

        if results:
            name, raw, a, data = results[sel]
            current_scan = (name, raw, data)
            st.markdown(f"##### :material/image: {name}")
            render_result(name, raw, a)
            st.caption(":material/compare: See how every denoising / enhancement method does on "
                       "this scan in the **Compare methods** tab.")


# --------------------------------------------------------------------------
# Compare methods tab
# --------------------------------------------------------------------------

with tab_compare:
    if current_scan is None:
        st.info("Add a scan on the **Analyze** tab first (upload one or pick a sample); "
                "every denoising / enhancement method is then compared on it here.",
                icon=":material/arrow_back:")
    elif tab_compare.open is False:
        pass  # hidden: skip the ~3 s comparison until someone opens this tab
    else:
        scan_name, scan_raw, scan_bytes = current_scan
        st.markdown(f"##### :material/compare: Which method works best on **{scan_name}**?")
        st.caption("All 14 tools in `agentic_us_denoise` and the 4 agent pipelines are run on this "
                   "scan and measured. There is no noise-free ground truth for a real scan, so the "
                   "measures are no-reference — see *How methods are scored* below.")
        with st.spinner("Running 18 denoising / enhancement methods…"):
            cmp_df, cmp_images = run_comparison(scan_bytes)

        crit = st.segmented_control(
            "Rank by", list(CRITERIA), default="overall", key="cmp_criterion",
            format_func=lambda k: CRITERIA[k])
        crit = crit or "overall"
        ranked = score(cmp_df, crit)
        cands = ranked[ranked.key != "original"]
        best, runner_up = cands.iloc[0], cands.iloc[1]
        top_keys = cands.key.head(3).tolist()

        # ---- winner -------------------------------------------------------
        win_row = st.container(horizontal=True, wrap=True, gap="large", key="cmp_win_row")
        with win_row.container(key="cmp_win_imgs"):
            c1, c2 = st.columns(2)
            c1.image(scan_raw, caption="Original", width="stretch")
            c2.image(cmp_images[best.key], caption=f"#1 · {best.method}", width="stretch")
        with win_row.container(border=True, key="cmp_win_card"):
            st.html(f'<div class="hero-kicker">Winner · {CRITERIA[crit]}</div>')
            st.html(f'<div class="pred-label" style="font-size:1.6rem">{best.method}</div>')
            st.badge(best.family, color="blue" if best.family == FAMILY_DENOISE else
                     ("orange" if best.family == FAMILY_ENHANCE else "green"))
            w = st.container(horizontal=True, wrap=True, key="cmp_win_metrics")
            w.metric("Speckle removed", f"{best.speckle_reduction:.0%}")
            w.metric("Anatomy kept", f"{best.anatomy_kept:.2f}",
                     help="1.00 = the filter removed only noise, no tissue structure.")
            w.metric("Tone kept", f"{best.tone_kept:.2f}",
                     help="1.00 = overall brightness and contrast unchanged.")
            st.caption(f"Runner-up: **{runner_up.method}** ({runner_up.score:.0f}). "
                       f"Pipeline: `{best.pipeline}` · {best.time_ms:.0f} ms.")

        # ---- charts ---------------------------------------------------------
        chart_row = st.container(horizontal=True, wrap=True, gap="large", key="cmp_chart_row")
        with chart_row.container(key="cmp_tradeoff"):
            st.markdown("**Noise removed vs anatomy kept** · ideal is top-right")
            st.altair_chart(tradeoff_chart(ranked, top_keys), width="stretch", theme="streamlit")
            st.caption("Numbered: the top 3 for the chosen criterion — "
                       + " · ".join(f"**{i}** {m}" for i, m in enumerate(cands.method.head(3), 1))
                       + ". Hover any dot for its name and scores.")
        with chart_row.container(key="cmp_ranking"):
            st.markdown(f"**Ranking · {CRITERIA[crit]}**")
            st.altair_chart(ranking_chart(cands), width="stretch", theme="streamlit")

        # ---- inspect one method -------------------------------------------------
        st.markdown("##### :material/zoom_in: Inspect a method")
        labels = dict(zip(cands.key, cands.method))
        pick = st.selectbox("Method", list(labels), index=0, format_func=lambda k: labels[k],
                            key="cmp_pick", label_visibility="collapsed")
        row = ranked[ranked.key == pick].iloc[0]
        i1, i2, i3 = st.columns(3)
        i1.image(scan_raw, caption="Original", width="stretch")
        i2.image(cmp_images[pick], caption=f"{row.method}", width="stretch")
        i3.image(method_noise_image(scan_raw, cmp_images[pick]),
                 caption="What it removed (gray = nothing)", width="stretch")
        st.caption("Read the third image: fine random grain = speckle removed (good); visible tissue "
                   "outlines = anatomy removed too (over-smoothing); large bright/dark patches = "
                   "brightness or contrast changed.")

        # ---- dataset-level evidence ------------------------------------------
        with st.container(border=True):
            st.markdown("**Best input for the AI classifier** · measured on all 762 scans, not just this one")
            e = st.container(horizontal=True, wrap=True, key="cmp_evidence")
            raw_acc = CLASSIFIER_EVIDENCE["Raw input"]
            for label, acc in CLASSIFIER_EVIDENCE.items():
                e.metric(label, f"{acc:.1%}",
                         None if label == "Raw input" else f"{100 * (acc - raw_acc):+.1f} pts vs raw")
            st.caption("Same ConvNeXt-Tiny model and folds, only the input changes. Adding CLAHE + "
                       "unsharp after denoising (L4) lowered accuracy (McNemar p = 0.018); denoising "
                       "alone (L2) gave a small gain that is not significant. With the stronger "
                       "ConvNeXt-V2 model, raw, denoised and enhanced inputs score alike (89.2 / 88.3 / "
                       "88.8%). Image-quality scores above and classifier accuracy are different goals "
                       "and need not agree.")

        # ---- gallery ---------------------------------------------------------------
        with st.expander(f":material/grid_view: All {len(cands)} results, ranked", expanded=False):
            gallery = st.container(horizontal=True, wrap=True, gap="small")
            for r in cands.itertuples():
                with gallery.container(width=170):
                    st.image(cmp_images[r.key], width="stretch")
                    st.caption(f"#{int(r.rank)} · {r.method} · **{r.score:.0f}**")

        # ---- full table ---------------------------------------------------------
        st.markdown("##### :material/table_view: All measurements")
        table = ranked.assign(speckle_reduction=100 * ranked.speckle_reduction)[[
            "rank", "method", "family", "score", "speckle_reduction", "anatomy_kept", "tone_kept",
            "cnr", "edge_keep", "ssim", "snr_db", "entropy", "time_ms"]]
        table["rank"] = ["ref" if pd.isna(r) else f"{int(r)}" for r in table["rank"]]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "rank": st.column_config.TextColumn("Rank", help="ref = the original scan, for comparison"),
            "method": "Method", "family": "Family",
            "score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100),
            "speckle_reduction": st.column_config.NumberColumn(
                "Speckle removed", format="%.0f%%", help="Drop of the speckle index (local std / mean)."),
            "anatomy_kept": st.column_config.NumberColumn(
                "Anatomy kept", format="%.2f", help="1 − tissue structure found in what the filter removed."),
            "tone_kept": st.column_config.NumberColumn(
                "Tone kept", format="%.2f", help="1 − change of large-scale brightness / contrast."),
            "cnr": st.column_config.NumberColumn("CNR", format="%.2f", help="Contrast-to-noise ratio."),
            "edge_keep": st.column_config.NumberColumn(
                "Edges preserved", format="%.2f", help="Correlation of edge maps with the input."),
            "ssim": st.column_config.NumberColumn("SSIM to input", format="%.2f"),
            "snr_db": st.column_config.NumberColumn("SNR (dB)", format="%.1f"),
            "entropy": st.column_config.NumberColumn("Entropy", format="%.2f"),
            "time_ms": st.column_config.NumberColumn("Time (ms)", format="%.0f"),
        })
        st.download_button(":material/download: Measurements (CSV)",
                           ranked.drop(columns=["key"]).to_csv(index=False),
                           file_name=f"{Path(scan_name).stem}_method_comparison.csv", mime="text/csv")

        with st.expander(":material/help: How methods are scored"):
            st.markdown(
                "- **Speckle removed** — drop of the *speckle index* (mean of local standard deviation "
                "÷ local mean in 5×5 windows, tissue only). Higher = smoother.\n"
                "- **Anatomy kept** — looks at what the filter *removed* (input − output, the "
                "\"method noise\"). If that removed part contains tissue edges, the filter destroyed "
                "anatomy. 1.00 = only random speckle was removed.\n"
                "- **Tone kept** — how much the large-scale brightness / contrast changed (both images "
                "blurred, then compared). Denoisers should not repaint the scan.\n"
                "- **Best denoising (overall)** = speckle removed^0.4 × anatomy kept^0.3 × tone kept^0.3. "
                "A product on purpose: doing nothing (no speckle removed) and blurring everything "
                "(anatomy lost) both score low.\n"
                "- **Contrast (CNR)** and **Edges preserved** are offered as alternative goals: contrast "
                "tools (CLAHE, histogram equalisation) win on contrast but *add* speckle.\n"
                "- Scores are % of the best method **on this scan**; another scan can rank differently.\n"
                "- Contrast-enhancing methods (CLAHE, unsharp mask, agent L3/L4) score 0 on denoising "
                "because they increase speckle — that is their job, not a bug."
            )


# --------------------------------------------------------------------------
# Model performance tab
# --------------------------------------------------------------------------

with tab_perf:
    if not ensemble:
        st.info("No evaluation found. Run `python ensemble_eval.py runs/final_L4 runs/final_L2 "
                "--out runs/ensemble_L4_L2`.")
    else:
        st.markdown("Every number is **out-of-fold**: each of the "
                    f"{ensemble['n_images']} images was scored by models that never saw it, "
                    "or any duplicate of it (5-fold cross-validation).")
        kpis = st.container(horizontal=True, wrap=True, key="kpis")
        k = [kpis] * 5
        k[0].metric("Accuracy", f"{ensemble['accuracy']:.1%}", border=True,
                    help=f"Per fold {min(ensemble['fold_accuracy'].values()):.1%} – "
                         f"{max(ensemble['fold_accuracy'].values()):.1%}")
        k[1].metric("Balanced acc.", f"{ensemble['balanced_accuracy']:.1%}", border=True,
                    help="Balanced accuracy: mean recall over the three classes.")
        k[2].metric("Macro AUC", f"{ensemble['macro_auc_ovr']:.3f}", border=True,
                    help="One-vs-rest ROC-AUC averaged over classes.")
        k[3].metric("Malignant sensitivity", f"{ensemble['malignant_sensitivity']:.1%}", border=True,
                    help="Share of malignant scans flagged as malignant.")
        k[4].metric("Malignant specificity", f"{ensemble['malignant_specificity']:.1%}", border=True,
                    help="Share of non-malignant scans not flagged as malignant.")

        perf_row = st.container(horizontal=True, wrap=True, gap="large", key="perf_row")
        c1, c2 = perf_row.container(key="cm_box"), perf_row.container(key="pc_box")
        with c1:
            st.markdown("##### Confusion matrix")
            st.altair_chart(confusion_chart(ensemble["confusion_matrix"]), width="stretch", theme="streamlit")
        with c2:
            st.markdown("##### Per class")
            pc = ensemble["per_class"]
            st.dataframe(pd.DataFrame([{
                "Class": CLASS_STYLE[c]["label"], "Precision": 100 * pc[c]["precision"],
                "Recall": 100 * pc[c]["recall"], "F1": pc[c]["f1-score"], "Images": int(pc[c]["support"]),
            } for c in CLASSES]), hide_index=True, width="stretch", column_config={
                "Precision": st.column_config.NumberColumn(format="%.1f%%"),
                "Recall": st.column_config.NumberColumn(format="%.1f%%"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
            })
            st.markdown(
                "- Most errors are **benign ↔ malignant** and **normal → benign**.\n"
                "- Malignant vs rest ROC-AUC: "
                f"**{ensemble['malignant_auc']:.3f}**."
            )

        runs = load_runs()
        if len(runs) > 1:
            st.markdown("##### What helped · all runs on the same folds")
            st.altair_chart(comparison_chart(runs), width="stretch", theme="streamlit")
            st.caption("Whiskers: ± fold standard deviation. Highlighted: the deployed ensemble.")
            f1, f2, f3 = st.columns(3, border=True)
            f1.markdown("**The backbone matters most**  \nConvNeXt-V2 (384 px) added 2–3 points "
                        "over ConvNeXt-Tiny on identical inputs (raw: 86.9 → 89.2%, McNemar p = 0.020).")
            f2.markdown("**Preprocessing: no measurable gain**  \nWith ConvNeXt-V2, raw (89.2%), "
                        "denoised (88.3%) and denoised + enhanced (88.8%) inputs are statistically tied "
                        "(p ≥ 0.34). At equal size, ten raw-image models (89.9%, AUC 0.960) match "
                        "the ten-model L2 + L4 ensemble (89.4%, AUC 0.966); neither gap is significant.")
            f3.markdown("**Enhancement can hurt**  \nWith ConvNeXt-Tiny, adding CLAHE + unsharp after "
                        "denoising cost 2.4 points (p = 0.018). Two seeds of the same raw model differ "
                        "by 0.7 points, as much as the inputs differ from each other.")

        with st.expander(":material/verified: How the evaluation is kept honest"):
            st.markdown(
                "- BUSI contains many re-saved / re-cropped copies of the same scan. Near-duplicates "
                "were detected (thumbnail correlation + perceptual hash): **780 files → 602 unique scans**.\n"
                "- **18 images in 8 duplicate groups with contradicting labels** (e.g. the same scan "
                "filed as benign and malignant) were removed.\n"
                "- Folds are built with `StratifiedGroupKFold`, so all copies of a scan stay together.\n"
                "- Each fold trains for a fixed 30 epochs; no early stopping or checkpoint picking on "
                "the test fold.\n"
                "- Published BUSI accuracies above ~92% usually come from random splits where "
                "duplicates leak between train and test."
            )


# --------------------------------------------------------------------------
# How it works tab
# --------------------------------------------------------------------------

with tab_how:
    p = theme()
    dot = f"""
    digraph G {{
      bgcolor="transparent"; rankdir=TB; nodesep=0.3; ranksep=0.32; compound=true;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
            color="{p['grid']}", fillcolor="{p['node']}", fontcolor="{p['text']}", penwidth=1.2,
            margin="0.18,0.07"];
      edge [color="{p['text2']}", arrowsize=0.7];
      scan [label="Ultrasound scan"];
      subgraph cluster_agents {{
        label="agentic_us_denoise"; labeljust="l"; fontname="Helvetica"; fontsize=11; fontcolor="{p['text2']}";
        style="rounded,dashed"; color="{p['text2']}"; margin=12;
        a1 [label="Agent 1 · analyse noise, contrast, edges"];
        a2 [label="Agent 2 · plan the pipeline"];
        tools [label="Tools · Gaussian / SRAD · wavelet · CLAHE · unsharp"];
        a3 [label="Agent 3 · evaluate, re-plan if needed"];
        a1 -> a2 -> tools -> a3;
        a3 -> a2 [style=dashed, constraint=false, label="  feedback", fontsize=10,
                  fontcolor="{p['text2']}"];
      }}
      l2 [label="Denoised view (L2)"];
      l4 [label="Denoised + enhanced view (L4)"];
      m2 [label="5 × ConvNeXt-V2"];
      m4 [label="5 × ConvNeXt-V2"];
      ood [label="Out-of-scope guard"];
      vote [label="Soft vote", fillcolor="{p['primary']}", fontcolor="#ffffff", color="{p['primary']}"];
      out [label="Benign / malignant / normal + lesion heat-map"];
      scan -> a1;
      a3 -> l2; a3 -> l4; l2 -> m2; l4 -> m4; l4 -> ood;
      m2 -> vote; m4 -> vote; vote -> out; ood -> out [style=dashed];
    }}"""
    st.graphviz_chart(dot, width="stretch")

    how_row = st.container(horizontal=True, wrap=True, key="how_row")
    c1, c2, c3 = (how_row.container(border=True) for _ in range(3))
    with c1:
        st.markdown("**:material/smart_toy: Agentic cleanup**")
        st.markdown("Your `agentic_us_denoise` agents inspect each scan, choose denoising "
                    "(Gaussian or SRAD + wavelet) and enhancement (CLAHE, unsharp mask), check "
                    "the result and re-plan if needed. Two outputs are used: **denoised (L2)** "
                    "and **denoised + enhanced (L4)**.")
    with c2:
        st.markdown("**:material/neurology: Ensemble classifier**")
        st.markdown("For each view, 5 ConvNeXt-V2-Tiny models (one per cross-validation fold, "
                    "384 px, ImageNet-22k pre-training) are trained. A second head learns the "
                    "lesion outline from expert masks; it powers the heat-map and helps the "
                    "model focus on the lesion.")
    with c3:
        st.markdown("**:material/shield: Out-of-scope guard**")
        st.markdown("The model only knows breast tissue. A Mahalanobis distance on generic image "
                    "features flags scans unlike the training data (e.g. fetal or abdominal "
                    "scans), so they are not silently given a breast diagnosis.")

    st.markdown("#### Data")
    st.markdown(
        "BUSI breast ultrasound dataset — Al-Dhabyani et al., *Data in Brief* 28 (2020) 104863, "
        "CC BY 4.0 — 780 images from 600 women (Baheya Hospital, Cairo, 2018); 762 used after "
        "removing contradicting duplicates."
    )
    st.markdown("#### Limitations")
    st.markdown(
        "- **Breast ultrasound only.** Other organs are flagged by the guard, but the guard catches "
        "clearly different images, not subtle shifts such as another breast scanner.\n"
        "- Single-hospital data: expect lower accuracy on other machines until validated there.\n"
        "- ~600 unique scans: fold-to-fold spread is ±3–5 points.\n"
        "- **Research and education only — not a medical device and not a diagnosis.**"
    )
