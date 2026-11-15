"""Pack the v2 analysis outputs into demo/public/results/data.json for the public
results explorer (demo/public/results/index.html).

    python scripts/build_results_site.py

Reads results/summary/v2/analysis.json (eval.analyze) and, if present,
results/summary/v2/novelty.json (eval.novelty). Nothing is recomputed here; the
page shows exactly the numbers the paper is built from, per seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/summary/v2"
DST = ROOT / "demo/public/results/data.json"

LABELS = {
    "v2_baseline": "Deterministic",
    "v2_gaussian_nll": "Gaussian",
    "v2_evidential": "Evidential",
    "v2_mcdropout": "MC-dropout",
    "v2_gaussian_nll_ensemble5": "Gaussian ensemble",
    "v2_evidential_ensemble5": "Evidential ensemble",
}
EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main():
    a = json.loads((SRC / "analysis.json").read_text())
    methods = []
    for key, label in LABELS.items():
        if key not in a["runs"]:
            continue
        seeds = a["runs"][key]["seeds"]
        per_seed = {}
        for s, m in seeds.items():
            per_seed[s] = {k: _num(v) for k, v in m.items() if isinstance(v, (int, float))}
        by_emotion = {}
        for kind in ("cov_perdim_by_emotion", "cov_joint_by_emotion"):
            first = next(iter(seeds.values())).get(kind)
            if not first:
                continue
            by_emotion[kind] = {
                e: [_num(np.mean([seeds[s][kind][e] for s in seeds])),
                    _num(np.std([seeds[s][kind][e] for s in seeds]))]
                for e in EMOTIONS if e in first}
        summary = {k: [_num(v[0]), _num(v[1])] for k, v in a["summary"][key].items()
                   if isinstance(v, list)}
        methods.append({"key": key, "label": label, "n": len(seeds),
                        "summary": summary, "per_seed": per_seed,
                        "by_emotion": by_emotion})
    out = {"methods": methods, "emotions": EMOTIONS, "alpha": 0.1}
    nov = SRC / "novelty.json"
    if nov.exists():
        out["novelty"] = json.loads(nov.read_text())["summary"]
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {DST} ({DST.stat().st_size // 1024} KB, {len(methods)} methods)")


if __name__ == "__main__":
    main()
