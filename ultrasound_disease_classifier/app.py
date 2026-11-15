"""
app.py - Streamlit UI for the breast-ultrasound classifier.

    streamlit run app.py          (from this folder)
    python run_app.py             (from anywhere)

Research / educational use only - not a medical device and not a diagnosis.
"""

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
def run_analysis(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return img, get_predictor().analyze(img)


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
            inp = "Ensemble of both views"
        extra = " + lesion loss" if cfg.get("seg_weight", 0) > 0 else ""
        rows.append({
            "run": cfg["name"], "input": inp,
            "model": f"{arch} {cfg['img_size']}px{extra}",
            "label": f"{inp} · {arch} {cfg['img_size']}px{extra}",
            "accuracy": m["accuracy"], "fold_sd": m["fold_accuracy_std"],
            "balanced_accuracy": m["balanced_accuracy"], "macro_auc": m["macro_auc_ovr"],
            "mal_sens": m["malignant_sensitivity"], "mal_spec": m["malignant_specificity"],
            "final": "members" in cfg,
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def probability_chart(a: Analysis):
    p = theme()
    df = pd.DataFrame({
        "class": [CLASS_STYLE[c]["label"] for c in CLASSES],
        "probability": [a.probabilities[c] for c in CLASSES],
        # No highlighted "winner" for an out-of-scope image.
        "predicted": [c == a.prediction and a.in_distribution is not False for c in CLASSES],
    })
    order = [CLASS_STYLE[c]["label"] for c in CLASSES]
    base = alt.Chart(df).encode(
        y=alt.Y("class:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=13, labelColor=p["text"], ticks=False, domain=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=20).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                axis=alt.Axis(format="%", tickCount=5, grid=True, gridColor=p["grid"],
                              labelColor=p["text2"], domain=False, ticks=False)),
        color=alt.condition("datum.predicted", alt.value(p["primary"]), alt.value(p["muted"])),
        tooltip=[alt.Tooltip("class:N", title="Class"),
                 alt.Tooltip("probability:Q", title="Probability", format=".1%")],
    )
    # Long bars carry their label inside (white), short ones just past the end,
    # so a 95% label never runs off the chart.
    inside = base.transform_filter("datum.probability >= 0.8").mark_text(
        align="right", dx=-6, fontSize=13, fontWeight=600).encode(
        x="probability:Q", text=alt.Text("probability:Q", format=".1%"), color=alt.value("#ffffff"))
    outside = base.transform_filter("datum.probability < 0.8").mark_text(
        align="left", dx=6, fontSize=13).encode(
        x="probability:Q", text=alt.Text("probability:Q", format=".1%"), color=alt.value(p["text"]))
    return (bars + inside + outside).properties(height=130).configure_view(stroke=None).configure(background="transparent", font=CHART_FONT)


def confusion_chart(cm):
    p = theme()
    cm = np.asarray(cm)
    labels = [CLASS_STYLE[c]["label"] for c in CLASSES]
    rows = []
    for i, t in enumerate(labels):
        for j, pr in enumerate(labels):
            rows.append({"true": t, "pred": pr, "count": int(cm[i, j]),
                         "share": cm[i, j] / cm[i].sum()})
    df = pd.DataFrame(rows)
    df["text"] = df.apply(lambda r: f"{r['count']}\n{r['share']:.0%}", axis=1)
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=labels, title="Predicted",
                axis=alt.Axis(labelAngle=0, labelColor=p["text"], titleColor=p["text2"],
                              ticks=False, domain=False, orient="bottom")),
        y=alt.Y("true:N", sort=labels, title="True",
                axis=alt.Axis(labelColor=p["text"], titleColor=p["text2"], ticks=False, domain=False)),
    )
    heat = base.mark_rect(cornerRadius=4, stroke=p["surface"], strokeWidth=3).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=BLUE_RAMP), legend=None),
        tooltip=[alt.Tooltip("true:N", title="True"), alt.Tooltip("pred:N", title="Predicted"),
                 alt.Tooltip("count:Q", title="Images"),
                 alt.Tooltip("share:Q", title="Share of true class", format=".1%")],
    )
    text = base.mark_text(fontSize=14, fontWeight=600, lineBreak="\n", lineHeight=18).encode(
        text="text:N",
        color=alt.condition("datum.share > 0.5", alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (heat + text).properties(height=300).configure_view(stroke=None).configure(background="transparent", font=CHART_FONT)


def comparison_chart(df):
    p = theme()
    df = df.copy()
    df["lo"] = df.accuracy - df.fold_sd
    df["hi"] = df.accuracy + df.fold_sd
    order = df.label.tolist()
    # Each run's name sits above its own full-width bar, so the chart stays
    # readable at any width (axis labels beside the bars squeezed them).
    y = alt.Y("label:N", sort=order, title=None, axis=None, scale=alt.Scale(paddingInner=0.05))
    x_scale = alt.Scale(domain=[0.7, 1.0])
    tooltip = [alt.Tooltip("label:N", title="Run"),
               alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
               alt.Tooltip("fold_sd:Q", title="Fold sd", format=".1%"),
               alt.Tooltip("balanced_accuracy:Q", title="Balanced acc.", format=".1%"),
               alt.Tooltip("macro_auc:Q", title="Macro AUC", format=".3f")]
    base = alt.Chart(df)
    bars = base.mark_bar(cornerRadiusEnd=4, height=14, yOffset=9).encode(
        y=y,
        x=alt.X("accuracy:Q", scale=x_scale, title="Out-of-fold accuracy",
                axis=alt.Axis(format="%", tickCount=4, grid=True, gridColor=p["grid"],
                              labelColor=p["text2"], titleColor=p["text2"], domain=False, ticks=False)),
        x2=alt.datum(0.7),  # bars start at the axis origin (70%), not at 0
        color=alt.condition("datum.final", alt.value(p["primary"]), alt.value(p["muted"])),
        tooltip=tooltip,
    )
    err = base.mark_rule(color=p["text2"], strokeWidth=1.5, yOffset=9).encode(
        y=y, x=alt.X("lo:Q", scale=x_scale), x2="hi:Q")
    names = base.mark_text(align="left", baseline="bottom", dy=-1, fontSize=12.5, clip=True).encode(
        y=y, x=alt.value(0), text="label:N", color=alt.value(p["text"]))
    values = base.mark_text(align="left", dx=6, yOffset=9, fontSize=12, fontWeight=600, clip=True).encode(
        y=y, x=alt.X("hi:Q", scale=x_scale), text=alt.Text("accuracy:Q", format=".1%"),
        color=alt.value(p["text"]))
    return (bars + err + names + values).properties(height=48 * len(df) + 10).configure_view(
        stroke=None).configure(background="transparent", font=CHART_FONT)


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

tab_analyze, tab_perf, tab_how = st.tabs([
    ":material/search: Analyze", ":material/insights: Model performance",
    ":material/account_tree: How it works",
])

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

            st.altair_chart(probability_chart(a), width="stretch", theme=None)

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
                    results.append((name, raw, a))
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
            } for n, _, a in results])
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
            name, raw, a = results[sel]
            st.markdown(f"##### :material/image: {name}")
            render_result(name, raw, a)


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
            st.altair_chart(confusion_chart(ensemble["confusion_matrix"]), width="stretch", theme=None)
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
            st.altair_chart(comparison_chart(runs), width="stretch", theme=None)
            st.caption("Bars: out-of-fold accuracy (axis starts at 70%); whiskers: ± fold standard "
                       "deviation. Highlighted: the deployed ensemble.")
            f1, f2, f3 = st.columns(3, border=True)
            f1.markdown("**Denoising helps**  \nWith the same model, denoised images beat raw "
                        "(+1.6 pts) and beat denoised + enhanced (+2.4 pts, McNemar p = 0.018).")
            f2.markdown("**The backbone matters most**  \nConvNeXt-V2 (384 px) added ~3 points "
                        "over ConvNeXt-Tiny on identical inputs.")
            f3.markdown("**Two views are complementary**  \nAveraging the denoised and the "
                        "denoised + enhanced models raised macro AUC from 0.952 to 0.966.")

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
