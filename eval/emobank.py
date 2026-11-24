"""EmoBank: the coverage findings on a second modality (text), on a Mac.

EmoBank (Buechel & Hahn, EACL 2017; CC-BY-SA 4.0) has 10,062 English sentences
rated for valence, arousal and dominance on 1-5, from seven kinds of text
(blog, essays, fiction, letters, newspaper, SemEval headlines, travel guides).
We use valence and arousal, rescaled to [-1, 1], and its official
train / dev / test split: train (minus 10% held out for model selection) fits
the heads, dev is the conformal calibration split, test is the test split. The
genre plays the role the expression plays on AffectNet: it defines the groups,
and the leave-one-genre-out runs remove one genre from train and dev.

The text encoder (roberta-base) is frozen; sentences are embedded once (mean of
the last hidden layer). The same uncertainty heads as for faces are then
trained on the embeddings, with input dropout 0.2 as on the image backbone:

    python -m eval.emobank embed      # once; downloads roberta-base
    python -m eval.emobank run        # main runs, 5 seeds, + leave-one-genre-out
    python -m eval.emobank analyze --tables paper_v2/tables

Predictions are written in the same .npz format as the image runs
(<split>_mean / _var / _target / _emotion, with the genre in _emotion), so the
same conformal code evaluates them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.heads import build_head  # noqa: E402
from train.losses import evidential_loss, gaussian_nll_loss, mse_loss  # noqa: E402

ROOT = Path("data/raw/EmoBank")
OUT = Path("results/emobank")
ENCODER = "roberta-base"
GENRES = ["blog", "essays", "fiction", "letters", "newspaper", "SemEval", "travel-guides"]
METHODS = {"baseline": "regression", "gaussian_nll": "gaussian", "evidential": "evidential"}
MC_SAMPLES = 20


# ------------------------------------------------------------------ data ---

def load_corpus(root: Path = ROOT):
    meta = {r["id"]: r["category"] for r in csv.DictReader(open(root / "meta.tsv"), delimiter="\t")}
    rows = [r for r in csv.DictReader(open(root / "emobank.csv")) if meta.get(r["id"]) in GENRES]
    text = [r["text"] for r in rows]
    y = np.array([[float(r["V"]), float(r["A"])] for r in rows], dtype=np.float32)
    y = (y - 3.0) / 2.0                                   # [1,5] -> [-1,1]
    genre = np.array([GENRES.index(meta[r["id"]]) for r in rows])
    split = np.array([r["split"] for r in rows])
    return text, y, genre, split


def embed(args):
    from transformers import AutoModel, AutoTokenizer
    text, y, genre, split = load_corpus()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ENCODER)
    enc = AutoModel.from_pretrained(ENCODER).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(text), 64):
            b = tok(text[i:i + 64], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(dev)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            feats.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
            print(f"\r{i + 64}/{len(text)}", end="")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "embeddings.npz", x=np.concatenate(feats).astype(np.float32),
                        y=y, genre=genre, split=split)
    print(f"\nwrote {OUT / 'embeddings.npz'}")


# ----------------------------------------------------------------- train ---

class Model(torch.nn.Module):
    def __init__(self, in_dim: int, kind: str, p: float = 0.2):
        super().__init__()
        self.drop = torch.nn.Dropout(p)
        self.head = build_head(kind, in_dim)

    def forward(self, x):
        return self.head(self.drop(x))


def _loss(kind: str):
    return {"regression": mse_loss, "gaussian": gaussian_nll_loss,
            "evidential": lambda o, y: evidential_loss(o, y, 0.01)}[kind]


def train_one(x_tr, y_tr, x_sel, y_sel, kind: str, seed: int, epochs: int = 200, patience: int = 15):
    torch.manual_seed(seed)
    model = Model(x_tr.shape[1], kind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = _loss(kind)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xs, ys = torch.from_numpy(x_sel), torch.from_numpy(y_sel)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=g).split(128):
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(xs), ys))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict(model, x, mc: int = 0) -> dict:
    x = torch.from_numpy(x)
    with torch.no_grad():
        if not mc:
            o = model(x)
            out = {k: o[k].numpy() for k in ("mean", "var", "nu", "alpha", "beta") if k in o}
            out.setdefault("var", np.zeros_like(out["mean"]))
            return out
        model.drop.train()                       # dropout on at test time
        runs = [model(x) for _ in range(mc)]
        model.eval()
        means = np.stack([r["mean"].numpy() for r in runs])
        var = np.stack([r["var"].numpy() for r in runs]).mean(0)
        return {"mean": means.mean(0), "var": var + means.var(0)}


def _save(path: Path, preds: dict):
    flat = {f"{s}_{k}": v for s, d in preds.items() for k, v in d.items()}
    np.savez_compressed(path, **flat)


def run(args):
    d = np.load(OUT / "embeddings.npz")
    x, y, genre, split = d["x"], d["y"], d["genre"], d["split"]
    mu, sd = x[split == "train"].mean(0), x[split == "train"].std(0) + 1e-6
    x = (x - mu) / sd
    tr_all = np.flatnonzero(split == "train")
    sel = np.random.default_rng(1).permutation(tr_all)[: len(tr_all) // 10]
    tr = np.setdiff1d(tr_all, sel)
    va, te = np.flatnonzero(split == "dev"), np.flatnonzero(split == "test")
    pdir = OUT / "preds"
    pdir.mkdir(parents=True, exist_ok=True)

    def one(name, seed, keep):
        """Train every head on `keep` (a genre mask) and save its predictions."""
        t, s, v = tr[keep[tr]], sel[keep[sel]], va[keep[va]]
        for method, kind in METHODS.items():
            f = pdir / f"{name}_{method}_s{seed}.npz"
            if f.exists():
                continue
            m = train_one(x[t], y[t], x[s], y[s], kind, seed)
            preds = {sp: {**predict(m, x[ix]), "target": y[ix], "emotion": genre[ix]}
                     for sp, ix in (("val", v), ("test", te))}
            _save(f, preds)
            if kind == "evidential":           # MC-dropout from the same weights
                preds = {sp: {**predict(m, x[ix], MC_SAMPLES), "target": y[ix], "emotion": genre[ix]}
                         for sp, ix in (("val", v), ("test", te))}
                _save(pdir / f"{name}_mcdropout_s{seed}.npz", preds)
            print(f"{f.name} done")

    everyone = np.ones(len(y), bool)
    for seed in range(1, 6):
        one("emobank", seed, everyone)
    for h in range(len(GENRES)):
        for seed in range(1, 4):
            one(f"lgo_h{h}", seed, genre != h)


# --------------------------------------------------------------- analyze ---

def analyze(args):
    res = collect(prefix="emobank")
    out = Path("results/summary/emobank")
    out.mkdir(parents=True, exist_ok=True)
    (out / "emobank.json").write_text(json.dumps(res, indent=2))
    _print(res)
    if args.tables:
        Path(args.tables, "emobank.tex").write_text(_latex(res))
        print(f"wrote {args.tables}/emobank.tex")


def collect(prefix: str) -> dict:
    """Every number for one dataset, from <OUT>/preds: main runs named
    <prefix>_<method>_s<seed>.npz and held-out-group runs lgo_h<k>_...; the
    group names are the module-level GENRES (set by the caller for DEAM)."""
    from eval import joint_regions, loeo, recalibrate
    from eval.analyze import ensemble, load_split
    from eval.metrics import accuracy_metrics

    pdir = OUT / "preds"
    load = lambda f: {s: load_split(np.load(f), s) for s in ("val", "test")}
    res: dict = {"main": {}, "lgo": {}, "recal": {}}

    for method in (*METHODS, "mcdropout"):
        runs = [load(f) for f in sorted(pdir.glob(f"{prefix}_{method}_s*.npz"))]
        rows = [{**joint_regions.evaluate(r["val"], r["test"]),
                 "ccc": accuracy_metrics(r["test"]["mean"], r["test"]["target"])["ccc_mean"]}
                for r in runs]
        res["main"][method] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res["main"][method]["n"] = len(rows)
        if method in ("gaussian_nll", "evidential"):
            ens = {sp: ensemble([r[sp] for r in runs]) for sp in ("val", "test")}
            row = joint_regions.evaluate(ens["val"], ens["test"])
            row["ccc"] = accuracy_metrics(ens["test"]["mean"], ens["test"]["target"])["ccc_mean"]
            res["main"][f"{method}_ensemble"] = {**row, "n": 1}

    # The official dev/test split is one draw; the guarantee is an average over
    # draws. Pool dev+test and re-split at random to estimate that average.
    from eval.analyze import conformal
    rs_rng = np.random.default_rng(0)
    for method in (*METHODS, "mcdropout"):
        cov = {"perdim_joint": [], "joint": [], "bonf": []}
        for f in sorted(pdir.glob(f"{prefix}_{method}_s*.npz")):
            r = load(f)
            pool = {k: np.concatenate([r["val"][k], r["test"][k]]) for k in r["val"] if k in r["test"]}
            n = len(pool["mean"])
            for _ in range(200):
                p = rs_rng.permutation(n)
                c = conformal({k: x[p[: n // 2]] for k, x in pool.items()},
                              {"t": {k: x[p[n // 2:]] for k, x in pool.items()}})["t"]
                for k in cov:
                    cov[k].append(float(c[k].mean()))
        res.setdefault("resplit", {})[method] = {k: (float(np.mean(v)), float(np.std(v)))
                                                for k, v in cov.items()}

    rng = np.random.default_rng(0)
    for method in ("evidential", "mcdropout"):
        for h, g in enumerate(GENRES):
            rs, rc = [], []
            for f in sorted(pdir.glob(f"lgo_h{h}_{method}_s*.npz")):
                r = load(f)
                rs.append(loeo.evaluate(r["val"], r["test"], h))
                rc.append(recalibrate.run(r["val"], r["test"], h, rng))
            if not rs:
                continue
            row = {k: float(np.mean([r[k] for r in rs if k in r])) for k in rs[0]}
            row["held_cov_ref"] = float(np.mean([_group_cov(load(f), h)
                                                 for f in sorted(pdir.glob(f"{prefix}_{method}_s[123].npz"))]))
            res["lgo"][f"{method}_{g}"] = row
            res["recal"][f"{method}_{g}"] = {k: float(np.mean([r[k] for r in rc if k in r]))
                                            for k in rc[0]}
    return res


def _group_cov(r: dict, h: int) -> float:
    from eval.analyze import conformal
    t = conformal(r["val"], {"test": r["test"]})["test"]
    return float(t["joint"][r["test"]["emotion"] == h].mean())


def _print(res):
    print(f"{'method':22s}  ccc  per-axis  bonf/area  max  emp.cop/area  ellip/area  worst(perdim,max)")
    for m, r in res["main"].items():
        print(f"{m:22s} {r['ccc']:.3f}  {100*r['cov_perdim']:5.1f}  {100*r['cov_bonf']:5.1f}/{r['area_bonf']/r['area_max']:.2f}"
              f"  {100*r['cov_max']:5.1f}  {100*r['cov_empcop']:5.1f}/{r['area_empcop']/r['area_max']:.2f}"
              f"  {100*r['cov_ellip']:5.1f}/{r['area_ellip']/r['area_max']:.2f}"
              f"  {100*r['worst_perdim']:5.1f} {100*r['worst_max']:5.1f}")
    print("\nleave-one-genre-out (held cov: ref -> perdim / bonf / max; others; auroc; recal k=20 group/shift)")
    for k, r in res["lgo"].items():
        c = res["recal"][k]
        print(f"{k:26s} n={r['n_held']:4.0f} {100*r['held_cov_ref']:5.1f} -> {100*r['held_cov_perdim']:5.1f} "
              f"{100*r['held_cov_bonf']:5.1f} {100*r['held_cov_joint']:5.1f}  others {100*r['seen_cov_joint']:5.1f}"
              f"  auroc {r.get('held_auroc', float('nan')):.2f}"
              f"  k20 {100*c.get('mondrian_k20_cov', np.nan):5.1f}/{c.get('mondrian_k20_area', np.nan):.1f}x"
              f" {100*c.get('shift_k20_cov', np.nan):5.1f}/{c.get('shift_k20_area', np.nan):.1f}x")


def _latex(res) -> str:
    names = [("baseline", "Deterministic"), ("gaussian_nll", "Gaussian"), ("evidential", "Evidential"),
             ("mcdropout", "MC-dropout"), ("gaussian_nll_ensemble", "Gaussian ens."),
             ("evidential_ensemble", "Evidential ens.")]
    out = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r" & & Per-axis & \multicolumn{2}{c}{Bonferroni} & \multicolumn{2}{c}{Max-score (ours)} \\",
           r"Method & CCC & Joint & Joint & Area & Joint & Worst genre \\", r"\midrule"]
    for k, n in names:
        if k not in res["main"]:
            continue
        r = res["main"][k]
        out.append(f"{n} & {r['ccc']:.3f} & {r['cov_perdim']:.3f} & {r['cov_bonf']:.3f} & "
                   f"{r['area_bonf'] / r['area_max']:.2f}$\\times$ & {r['cov_max']:.3f} & {r['worst_max']:.3f} \\\\")
    out += [r"\midrule", r"Held-out genre & $n$ & In training & Per-axis & Bonferroni & Max-score & Others \\",
            r"\midrule"]
    for g in GENRES:
        r = res["lgo"].get(f"evidential_{g}")
        if r:
            out.append(f"{g} & {r['n_held']:.0f} & {r['held_cov_ref']:.3f} & {r['held_cov_perdim']:.3f} & "
                       f"{r['held_cov_bonf']:.3f} & {r['held_cov_joint']:.3f} & {r['seen_cov_joint']:.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "run", "analyze"])
    ap.add_argument("--tables", default=None)
    a = ap.parse_args()
    {"embed": embed, "run": run, "analyze": analyze}[a.cmd](a)
