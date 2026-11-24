"""DEAM: the coverage findings on a third modality (music), on a Mac.

DEAM (Aljanaki, Yang & Soleymani, PLOS ONE 2017) has 1,802 Creative-Commons
songs and excerpts with song-level valence and arousal (mean over raters, 1-9;
we map to [-1, 1]) and openSMILE features per 0.5 s frame. A song is the mean
and standard deviation of its frame features (520 numbers). Genres come from
the metadata; the seven with at least 170 songs (Classical, Jazz, Country,
Electronic, Rock, Pop, Folk) are the groups, and every other song is kept in
the data as group "other".

Split: songs at random, 60% train (10% of it for model selection), 20%
calibration, 20% test, fixed across seeds. In the leave-one-genre-out runs the
held-out genre is removed from train, selection and calibration, and ALL its
songs are test songs (it is unseen, so none is needed elsewhere); the test set
of the other genres is unchanged.

    python -m eval.deam features      # once: aggregate openSMILE frames
    python -m eval.deam run           # 5 seeds + leave-one-genre-out (3 seeds)
    python -m eval.deam analyze --tables paper_v2/tables

The heads and training are those of eval/emobank.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.emobank import METHODS, MC_SAMPLES, _save, predict, train_one  # noqa: E402

ROOT = Path("data/raw/DEAM")
OUT = Path("results/deam")
GENRES = ["Classical", "Jazz", "Country", "Electronic", "Rock", "Pop", "Folk"]
OTHER = 8          # not 7: the shared analysis code treats group 7 as "unseen"


def _genre(raw: str) -> int:
    first = raw.strip().split("-")[0].strip().lower()
    for i, g in enumerate(GENRES):
        if first == g.lower():
            return i
    return OTHER


def load_labels(root: Path = ROOT) -> dict[int, tuple[float, float, int]]:
    genre: dict[int, int] = {}
    for name, col in (("metadata_2013.csv", 6), ("metadata_2014.csv", 4), ("metadata_2015.csv", 5)):
        for r in csv.reader(open(root / "metadata" / name, encoding="latin-1")):
            if r and r[0].strip().isdigit():
                genre[int(r[0])] = _genre(r[col])
    out = {}
    for f in sorted(glob.glob(str(root / "annotations/annotations averaged per song/song_level/*.csv"))):
        for r in csv.DictReader(open(f), skipinitialspace=True):
            sid = int(r["song_id"])
            v, a = (float(r["valence_mean"]) - 5) / 4, (float(r["arousal_mean"]) - 5) / 4
            out[sid] = (v, a, genre.get(sid, OTHER))
    return out


def features(args):
    labels = load_labels()
    ids, x, y, g = [], [], [], []
    for sid, (v, a, gen) in sorted(labels.items()):
        f = ROOT / "features" / f"{sid}.csv"
        if not f.exists():
            continue
        m = np.genfromtxt(f, delimiter=";", skip_header=1)[:, 1:]       # drop frameTime
        m = m[np.isfinite(m).all(1)]
        x.append(np.r_[m.mean(0), m.std(0)])
        ids.append(sid), y.append((v, a)), g.append(gen)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "features.npz", x=np.array(x, np.float32), y=np.array(y, np.float32),
                        genre=np.array(g), ids=np.array(ids))
    print(f"wrote {OUT / 'features.npz'}: {len(ids)} songs, {len(x[0])} features")


def splits(n: int):
    perm = np.random.default_rng(0).permutation(n)
    n_tr, n_ca = int(0.6 * n), int(0.2 * n)
    tr_all, ca, te = perm[:n_tr], perm[n_tr:n_tr + n_ca], perm[n_tr + n_ca:]
    sel = np.random.default_rng(1).permutation(tr_all)[: len(tr_all) // 10]
    return np.setdiff1d(tr_all, sel), sel, ca, te


def run(args):
    d = np.load(OUT / "features.npz")
    x, y, genre = d["x"], d["y"], d["genre"]
    tr, sel, ca, te = splits(len(y))
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    x = ((x - mu) / sd).astype(np.float32)
    pdir = OUT / "preds"
    pdir.mkdir(parents=True, exist_ok=True)

    def one(name, seed, held=None):
        keep = np.ones(len(y), bool) if held is None else genre != held
        t, s, v = tr[keep[tr]], sel[keep[sel]], ca[keep[ca]]
        test = te if held is None else np.r_[te[keep[te]], np.flatnonzero(genre == held)]
        for method, kind in METHODS.items():
            f = pdir / f"{name}_{method}_s{seed}.npz"
            if f.exists():
                continue
            m = train_one(x[t], y[t], x[s], y[s], kind, seed)
            for mc, out in ((0, f), (MC_SAMPLES, pdir / f"{name}_mcdropout_s{seed}.npz")):
                if mc and kind != "evidential":
                    continue
                _save(out, {sp: {**predict(m, x[ix], mc), "target": y[ix], "emotion": genre[ix]}
                            for sp, ix in (("val", v), ("test", test))})
            print(f"{f.name} done")

    for seed in range(1, 6):
        one("deam", seed)
    for h in range(len(GENRES)):
        for seed in range(1, 4):
            one(f"lgo_h{h}", seed, h)


def analyze(args):
    from eval import emobank
    emobank.OUT, emobank.GENRES = OUT, GENRES
    res = emobank.collect(prefix="deam")
    out = Path("results/summary/deam")
    out.mkdir(parents=True, exist_ok=True)
    (out / "deam.json").write_text(json.dumps(res, indent=2))
    emobank._print(res)
    if args.tables:
        Path(args.tables, "deam.tex").write_text(emobank._latex(res))
        print(f"wrote {args.tables}/deam.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "run", "analyze"])
    ap.add_argument("--tables", default=None)
    a = ap.parse_args()
    {"features": features, "run": run, "analyze": analyze}[a.cmd](a)
