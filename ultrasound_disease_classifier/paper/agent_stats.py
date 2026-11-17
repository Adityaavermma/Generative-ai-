"""
How adaptive is the agentic pipeline on BUSI? For every image and every
enhancement level L1-L4: Agent 1's ratings, the chain Agent 2 chose, how many
re-plans Agent 3 triggered, which acceptance tests failed, whether a re-plan
actually changed a parameter, and Agent 3's final verdict.

    python paper/agent_stats.py
Writes paper/data/agent_stats.csv and prints the summary used in the paper.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _one(path):
    from us_pipeline import _load, enhance  # imported in the worker
    planner = _load()[0].DecisionPlanningAgent()
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    rows = []
    for level in (1, 2, 3, 4):
        _, info = enhance(img, level=level)
        m = info["evaluation_metrics"]           # Agent 3's measures of the final output
        first = planner.plan(info["image_report"], enhancement_level=level)["parameters"]
        rows.append({
            "path": Path(os.path.relpath(path, HERE.parent)).as_posix(),   # relative to the project
            "level": level, "chain": " + ".join(info["pipeline"]),
            "replans": info["replans"], "final_status": info["quality_status"],
            "noise": info["noise_level"], "edges": info["edge_strength"],
            "psnr": m["PSNR"], "ssim": m["SSIM"], "epi": m["EdgePreservation"],
            "fails": " ".join(t for t, bad in (("PSNR", m["PSNR"] < 24.0), ("SSIM", m["SSIM"] < 0.80),
                                               ("EPI", m["EdgePreservation"] < 0.5)) if bad),
            "retuned": info["parameters"] != first,
            "clahe_clip": info["parameters"].get("CLAHE", {}).get("clip_limit"),
        })
    return rows


def main():
    df = pd.read_csv(HERE.parent / "data" / "index.csv")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 4)) as ex:
        for r in ex.map(_one, df.raw_path, chunksize=8):
            rows.extend(r)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "data" / "agent_stats.csv", index=False)

    n = df.shape[0]
    print(f"images: {n}")
    a1 = out[out.level == 1]
    print("Agent 1 noise rating:", a1.noise.value_counts().to_dict())
    print("Agent 1 edge rating :", a1.edges.value_counts().to_dict())
    for level in (1, 2, 3, 4):
        s = out[out.level == level]
        print(f"\nL{level} chains:", s.chain.value_counts().to_dict())
        print(f"L{level} re-plans:", s.replans.value_counts().sort_index().to_dict(),
              "| re-plan triggered on", int((s.replans > 0).sum()), "images",
              "| final verdict:", s.final_status.value_counts().to_dict())
        rej = s[s.final_status != "Acceptable"]
        if len(rej):
            print(f"L{level} failed tests (final output):", rej.fails.value_counts().to_dict(),
                  "| re-plan changed a parameter on", int(rej.retuned.sum()), "of", len(rej),
                  "| final CLAHE clip:", rej.clahe_clip.value_counts().to_dict())


if __name__ == "__main__":
    main()
