"""
us_pipeline.py
--------------
Thin wrapper around the agentic ultrasound denoising / enhancement project
(../../agentic_us_denoise). Every image the classifier sees - during training
and at prediction time - goes through exactly the same agent workflow:

    Agent 1 analyze -> Agent 2 plan -> run tools -> Agent 3 evaluate
    -> (re-plan with Agent 3's feedback if "Not Acceptable", max 2 times)

Enhancement levels (same meaning as in agentic_us_denoise):
    0  raw image, no processing
    1  basic denoising (Gaussian)
    2  advanced denoising (SRAD if noise is High, else Gaussian) + wavelet
    3  level 2 + CLAHE contrast enhancement
    4  level 3 + unsharp-mask edge enhancement when edges are weak/moderate
       (the "agent-optimized" output)

The location of agentic_us_denoise can be overridden with the
US_DENOISE_DIR environment variable.
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DENOISE_DIR = HERE.parents[1] / "agentic_us_denoise"

_AGENTS = None
_PT = None


def _load():
    """Import agents.py / processing_tools.py from the denoise project once."""
    global _AGENTS, _PT
    if _AGENTS is not None:
        return _AGENTS, _PT

    denoise_dir = Path(os.environ.get("US_DENOISE_DIR", DEFAULT_DENOISE_DIR)).resolve()
    if not (denoise_dir / "agents.py").exists():
        raise FileNotFoundError(
            f"Could not find agentic_us_denoise at {denoise_dir}. "
            "Set the US_DENOISE_DIR environment variable to its folder."
        )
    if str(denoise_dir) not in sys.path:
        sys.path.insert(0, str(denoise_dir))

    import agents  # noqa: E402  (from agentic_us_denoise)
    import processing_tools  # noqa: E402

    _AGENTS, _PT = agents, processing_tools
    return _AGENTS, _PT


def enhance(image: np.ndarray, level: int = 4, max_replan: int = 2):
    """Denoise + enhance a single-channel uint8 ultrasound image.

    Returns (processed_image, info_dict).
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("enhance() expects a single-channel uint8 image")
    if level == 0:
        return image.copy(), {"strategy": "Raw (no processing)", "pipeline": []}

    A, PT = _load()
    analyst = A.ImageAnalysisAgent()
    planner = A.DecisionPlanningAgent()
    evaluator = A.EvaluationOptimizationAgent()

    report = analyst.analyze(image)
    feedback = None
    for attempt in range(max_replan + 1):
        plan = planner.plan(report, enhancement_level=level, prior_feedback=feedback)
        processed = PT.run_pipeline(image, plan["pipeline_order"], plan["parameters"])
        evaluation = evaluator.evaluate(image, processed)
        if evaluation["acceptable"]:
            break
        feedback = evaluation

    info = {
        "strategy": plan["strategy"],
        "pipeline": plan["pipeline_order"],
        "noise_level": report["noise_level"],
        "edge_strength": report["edge_strength"],
        "replans": attempt,
        "quality_status": evaluation["quality_status"],
        "image_report": report,                      # Agent 1 (input analysis)
        "evaluation_metrics": evaluation["metrics"],  # Agent 3 (output vs input)
        "improvement_pct": evaluation["improvement_pct"],
    }
    return processed, info
