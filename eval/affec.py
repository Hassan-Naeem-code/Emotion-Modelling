"""AFFEC: how well can a camera bound how someone FEELS, with a guarantee?

AFFEC (Zenodo 10.5281/zenodo.19408373, CC-BY-4.0) has 72 people who watched
short videos of actors and, after each, rated on 1-9 both what they FELT
(f_emotion_v/a) and what they PERCEIVED the actor to show (p_emotion_v/a),
while a camera tracked their face (action units, landmarks; 40 Hz). Ratings
are mapped to [-1, 1] by (x - 5) / 4.

The question: a joint 90% conformal box for the viewer's felt valence and
arousal. How big must it be, and how much does each source of information
shrink it?
  - none:      no information (one box for everyone)
  - stimulus:  knows which kind of clip was shown (happy, sad, ...)
  - face:      the viewer's facial action units during the clip
  - +personal: the paper's "+shift" recalibration with k of the person's own
               ratings (half estimate their offset, half set the threshold)
Splits are by PERSON (train / calibration / test people never overlap), so the
guarantee is about new people, repeated over random person re-splits.

    python -m eval.affec baselines            # labels only; no face data needed
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("data/raw/AFFEC/core")
OUT = Path("results/summary/affec")
ALPHA = 0.1
CATS = ["neutral", "happy", "sad", "fear", "disgust", "angry"]
TARGETS = {"felt": ("f_emotion_v", "f_emotion_a"), "perceived": ("p_emotion_v", "p_emotion_a")}


def load_labels(root: Path = ROOT) -> dict:
    rows = []
    for f in sorted(glob.glob(str(root / "sub-*" / "beh" / "*_beh.tsv"))):
        rows += list(csv.DictReader(open(f), delimiter="\t"))
    rows = [r for r in rows if r["trial_type"] in CATS and all(r[k].strip() for t in TARGETS.values() for k in t)]
    subj = np.array([r["user"] for r in rows])
    out = {"subject": subj, "run": np.array([int(r["run"]) for r in rows]),
           "trial": np.array([int(r["trial"]) for r in rows]),
           "stim": np.array([r["stim_file"] for r in rows]),
           "cat": np.array([CATS.index(r["trial_type"]) for r in rows])}
    for name, (kv, ka) in TARGETS.items():
        out[name] = ((np.array([[float(r[kv]), float(r[ka])] for r in rows]) - 5) / 4).astype(np.float64)
    return out


# ------------------------------------------------------------- conformal ---

def qhat(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    r = math.ceil((n + 1) * (1 - alpha))
    return math.inf if r > n else float(np.sort(scores)[r - 1])


def score(y, c, s):
    return np.max(np.abs(y - c) / s, axis=1)


def box_area(c, s, q):
    """Mean area of the clipped boxes, as a fraction of the whole VA square."""
    lo, hi = np.clip(c - q * s, -1, 1), np.clip(c + q * s, -1, 1)
    return float(np.mean(np.prod(hi - lo, axis=1)) / 4)


def fit_stimulus(y, cat):
    """Per-category mean and std; categories absent in training fall back to global."""
    g_c, g_s = y.mean(0), y.std(0) + 1e-6
    c = np.tile(g_c, (len(CATS), 1))
    s = np.tile(g_s, (len(CATS), 1))
    for k in range(len(CATS)):
        m = cat == k
        if m.sum() > 1:
            c[k], s[k] = y[m].mean(0), y[m].std(0) + 1e-6
    return c, s


def person_split(people: np.ndarray, rng) -> tuple[set, set, set]:
    p = rng.permutation(people)
    n_tr, n_ca = int(0.6 * len(p)), int(0.2 * len(p))
    return set(p[:n_tr]), set(p[n_tr:n_tr + n_ca]), set(p[n_tr + n_ca:])


def personal(y, c, s, subj, ks, rng) -> dict:
    """+shift recalibration per test person, from k of their own ratings."""
    out = {}
    for k in ks:
        cov, area = [], []
        for p in np.unique(subj):
            ix = rng.permutation(np.flatnonzero(subj == p))
            if len(ix) < k + 10:
                continue
            use, ev = ix[:k], ix[k:]
            a, b = use[0::2], use[1::2]
            off = (y[a] - c[a]).mean(0)
            q = qhat(score(y[b], c[b] + off, s[b]))
            cov.append(np.mean(score(y[ev], c[ev] + off, s[ev]) <= q))
            area.append(box_area(c[ev] + off, s[ev], q))
        out[k] = (float(np.mean(cov)), float(np.mean(area)))
    return out


def baselines(args):
    d = load_labels()
    people = np.unique(d["subject"])
    rng = np.random.default_rng(0)
    res = {"n_trials": int(len(d["cat"])), "n_people": int(len(people)), "reps": args.reps}
    print(f"{res['n_trials']} trials, {res['n_people']} people, {args.reps} person re-splits\n")
    for target in TARGETS:
        y = d[target]
        acc = {m: [] for m in ("none", "stimulus")}
        pers = []
        for _ in range(args.reps):
            tr, ca, te = person_split(people, rng)
            itr, ica, ite = (np.isin(d["subject"], list(g)) for g in (tr, ca, te))
            preds = {"none": (np.tile(y[itr].mean(0), (len(y), 1)), np.tile(y[itr].std(0), (len(y), 1)))}
            cc, ss = fit_stimulus(y[itr], d["cat"][itr])
            preds["stimulus"] = (cc[d["cat"]], ss[d["cat"]])
            for m, (c, s) in preds.items():
                q = qhat(score(y[ica], c[ica], s[ica]))
                acc[m].append((np.mean(score(y[ite], c[ite], s[ite]) <= q), box_area(c[ite], s[ite], q)))
            c, s = preds["stimulus"]
            pers.append(personal(y[ite], c[ite], s[ite], d["subject"][ite], (20, 40), rng))
        res[target] = {m: {"coverage": float(np.mean([a[0] for a in v])), "area": float(np.mean([a[1] for a in v]))}
                       for m, v in acc.items()}
        for k in (20, 40):
            res[target][f"stimulus+personal_k{k}"] = {"coverage": float(np.mean([p[k][0] for p in pers])),
                                                      "area": float(np.mean([p[k][1] for p in pers]))}
        print(f"{target:>9}: {'method':<24} coverage   box size (share of the whole chart)")
        for m, v in res[target].items():
            print(f"{'':>11}{m:<24} {v['coverage']:6.1%}     {v['area']:6.1%}")
        print()
    # How different are felt and perceived on the same trial?
    diff = np.abs(d["felt"] - d["perceived"])
    res["felt_vs_perceived_mean_abs_diff"] = diff.mean(0).tolist()
    res["felt_perceived_corr"] = [float(np.corrcoef(d["felt"][:, i], d["perceived"][:, i])[0, 1]) for i in (0, 1)]
    print("felt vs perceived, same trial: mean |difference| V=%.2f A=%.2f (on [-1,1]); corr V=%.2f A=%.2f"
          % (*res["felt_vs_perceived_mean_abs_diff"], *res["felt_perceived_corr"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baselines.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'baselines.json'}")


# ----------------------------------------------------------- face features ---

ZIP = Path("data/raw/AFFEC/videostream.zip")
FEAT = Path("results/affec/face_features.npz")
AU_R = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
        "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]
MIN_FRAMES = 20     # below this (~0.5 s of tracked face) the trial has no face features


def _windows(events: Path) -> dict[int, tuple[float, float, float, float]]:
    """trial -> (watch start, watch end, baseline start, baseline end), in run seconds.
    Watching = the clip plus its frozen last frame; baseline = the fixation cross
    just before the clip (the person's resting face for this trial)."""
    w: dict[int, dict] = {}
    for r in csv.DictReader(open(events), delimiter="\t"):
        if r["flag"] in ("video", "last_frame_video", "second_fix"):
            w.setdefault(int(r["trial"]), {})[r["flag"]] = (float(r["onset"]), float(r["duration"]))
    out = {}
    for t, d in w.items():
        if "video" in d and "second_fix" in d:
            end = d["video"][0] + d["video"][1] + d.get("last_frame_video", (0, 0))[1]
            out[t] = (d["video"][0], end, d["second_fix"][0], d["second_fix"][0] + d["second_fix"][1])
    return out


def trial_features(df, win) -> np.ndarray | None:
    """Summary of one trial's tracked frames: AU intensity mean / std / max and
    change from the pre-clip baseline, AU presence rate, head and gaze movement."""
    t = df["onset"].to_numpy()
    good = (df["success"].to_numpy() == 1) & (df["confidence"].to_numpy() >= 0.8)
    watch = good & (t >= win[0]) & (t <= win[1])
    base = good & (t >= win[2]) & (t <= win[3])
    if watch.sum() < MIN_FRAMES:
        return None
    au = df[[a + "_r" for a in AU_R]].to_numpy()
    auc = df[[c for c in df.columns if c.endswith("_c")]].to_numpy()
    mov = df[["pose_Rx", "pose_Ry", "pose_Rz", "gaze_angle_x", "gaze_angle_y"]].to_numpy()
    a = au[watch]
    b = au[base].mean(0) if base.sum() >= 5 else a[:5].mean(0)
    return np.r_[a.mean(0), a.std(0), a.max(0), a.mean(0) - b, auc[watch].mean(0),
                 mov[watch].std(0), np.diff(mov[watch], axis=0).__abs__().mean(0)].astype(np.float32)


def _one_run(args):
    try:
        return _read_run(args)
    except (EOFError, OSError, ValueError) as e:   # truncated / corrupt file in the release
        print(f"\nskipping {args[0]}: {e}")
        return args[0], None


def _read_run(args):
    import pandas as pd
    import zipfile
    member, events = args
    z = zipfile.ZipFile(ZIP)
    cols = json.loads(z.read(member.replace(".tsv.gz", ".json")))["Columns"]
    want = ["onset", "confidence", "success", "pose_Rx", "pose_Ry", "pose_Rz", "gaze_angle_x", "gaze_angle_y"] \
        + [a + "_r" for a in AU_R] + [c for c in cols if c.startswith("AU") and c.endswith("_c")]
    with z.open(member) as f:
        import gzip
        if len(gzip.open(f, "rt").readline().split("\t")) != len(cols):
            return member, None          # face not tracked in this run (22 runs in v2)
    with z.open(member) as f:
        df = pd.read_csv(f, sep="\t", header=None, compression="gzip", usecols=[cols.index(c) for c in want])
    df.columns = [cols[i] for i in sorted(cols.index(c) for c in want)]
    # Most runs store an absolute clock; the face recording starts with the run,
    # so re-zero it, as the official AFFEC devkit does (loader.py).
    df["onset"] -= df["onset"].iloc[0]
    out = []
    for trial, win in _windows(events).items():
        x = trial_features(df, win)
        if x is not None:
            out.append((trial, x))
    return member, out


def features(args):
    import re
    import zipfile
    from multiprocessing import Pool
    jobs = []
    for m in zipfile.ZipFile(ZIP).namelist():
        k = re.match(r"(sub-[^/]+)/beh/\1_task-fer_run-(\d+)_recording-videostream_physio\.tsv\.gz$", m)
        if k:
            ev = ROOT / k[1] / f"{k[1]}_task-fer_run-{k[2]}_events.tsv"
            if ev.exists():
                jobs.append((m, ev))
    keys, xs, skipped = [], [], []
    with Pool(args.workers) as pool:
        for i, (m, out) in enumerate(pool.imap_unordered(_one_run, jobs), 1):
            k = re.match(r"sub-([^/]+)/.*run-(\d+)_", m)
            if out is None:
                skipped.append(m)
                continue
            for trial, x in out:
                keys.append(f"{k[1]}|{int(k[2])}|{trial}")
                xs.append(x)
            print(f"\r{i}/{len(jobs)} runs, {len(keys)} trials", end="", flush=True)
    FEAT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEAT, keys=np.array(keys), x=np.stack(xs))
    print(f"\nwrote {FEAT}: {len(keys)} trials x {xs[0].shape[0]} features; "
          f"{len(skipped)} runs had no tracked face")


# -------------------------------------------------------------- face runs ---

def _joined():
    """Labels restricted to trials with face features, aligned row by row."""
    d = load_labels()
    f = np.load(FEAT)
    pos = {f"{s}|{r}|{t}": i for i, (s, r, t) in enumerate(zip(d["subject"], d["run"], d["trial"]))}
    idx = np.array([pos[k] for k in f["keys"] if k in pos])
    x = f["x"][[k in pos for k in f["keys"]]].astype(np.float32)
    return {k: v[idx] for k, v in d.items()}, x


def person_normalise(x, subj):
    """z-score each feature within each person over all their trials (no labels used):
    a camera that has watched someone for a while knows their resting face."""
    out = np.empty_like(x)
    for p in np.unique(subj):
        m = subj == p
        out[m] = (x[m] - x[m].mean(0)) / (x[m].std(0) + 1e-6)
    return out


def _fit_face(x, y, itr, seed):
    from eval.emobank import predict, train_one
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(itr)
    sel = rng.choice(tr, len(tr) // 10, replace=False)
    fit = np.setdiff1d(tr, sel)
    mu, sd = x[fit].mean(0), x[fit].std(0) + 1e-6
    z = ((x - mu) / sd).astype(np.float32)
    yy = y.astype(np.float32)
    m = train_one(z[fit], yy[fit], z[sel], yy[sel], "gaussian", seed)
    p = predict(m, z)
    return p["mean"].astype(np.float64), np.sqrt(np.maximum(p["var"], 1e-8)).astype(np.float64)


def face(args):
    import torch
    torch.set_num_threads(4)
    d, x = _joined()
    xn = person_normalise(x, d["subject"])
    onehot = np.eye(len(CATS), dtype=np.float32)[d["cat"]]
    people = np.unique(d["subject"])
    rng = np.random.default_rng(0)
    print(f"{len(x)} trials with a tracked face, {len(people)} people, {args.reps} person re-splits\n")
    res = {"n_trials": int(len(x)), "n_people": int(len(people)), "reps": args.reps}
    inputs = {"face": x, "face (person-normalised)": xn, "face + stimulus": np.c_[xn, onehot]}
    for target in TARGETS:
        y = d[target]
        acc: dict[str, list] = {}
        for rep in range(args.reps):
            tr, ca, te = person_split(people, rng)
            itr, ica, ite = (np.isin(d["subject"], list(g)) for g in (tr, ca, te))
            preds = {"none": (np.tile(y[itr].mean(0), (len(y), 1)), np.tile(y[itr].std(0), (len(y), 1)))}
            cc, ss = fit_stimulus(y[itr], d["cat"][itr])
            preds["stimulus"] = (cc[d["cat"]], ss[d["cat"]])
            for name, xi in inputs.items():
                preds[name] = _fit_face(xi, y, itr, seed=rep)
            for m, (c, s) in preds.items():
                q = qhat(score(y[ica], c[ica], s[ica]))
                r = [np.mean(score(y[ite], c[ite], s[ite]) <= q), box_area(c[ite], s[ite], q),
                     *[np.corrcoef(c[ite, i], y[ite, i])[0, 1] if c[ite, i].std() > 0 else 0.0 for i in (0, 1)]]
                acc.setdefault(m, []).append(r)
                if m in ("none", "face (person-normalised)"):
                    pk = personal(y[ite], c[ite], s[ite], d["subject"][ite], (20, 40), rng)
                    for k, (cv, ar) in pk.items():
                        acc.setdefault(f"{m} + personal k={k}", []).append([cv, ar, np.nan, np.nan])
            print(f"\r{target}: {rep + 1}/{args.reps}", end="", flush=True)
        res[target] = {m: dict(zip(("coverage", "area", "corr_v", "corr_a"), np.nanmean(v, 0).tolist()))
                       for m, v in acc.items()}
        print(f"\r{target:>9}: {'what the system knows':<38} cover  box   corr V / A")
        for m, v in res[target].items():
            cr = "" if np.isnan(v["corr_v"]) else f"{v['corr_v']:+.2f} / {v['corr_a']:+.2f}"
            print(f"{'':>11}{m:<38} {v['coverage']:5.1%} {v['area']:5.1%}  {cr}")
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "face.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / 'face.json'}")


# ------------------------------------------------- body signals (not a camera) ---

BODY = Path("results/affec/body_features.npz")
SCR_LAG = 3.0       # skin conductance responds 1-3 s after the stimulus


def _change(t, v, win, lag=0.0):
    """Response of a slow signal in one trial: mean and peak change from the
    pre-clip baseline, and slope while watching. None if too little data."""
    ok = np.isfinite(v)
    w = ok & (t >= win[0]) & (t <= win[1] + lag)
    b = ok & (t >= win[2]) & (t <= win[3])
    if w.sum() < 20 or b.sum() < 5:
        return None
    base = v[b].mean()
    slope = np.polyfit(t[w] - t[w][0], v[w], 1)[0]
    return [v[w].mean() - base, v[w].max() - base, slope, base]


def _read_body(args):
    import gzip
    import zipfile
    sub, run, events = args
    wins = _windows(events)
    streams = {}
    for zf, mod in (("gsr.zip", "gsr"), ("pupil.zip", "pupil")):
        z = zipfile.ZipFile(ZIP.parent / zf)
        m = f"{sub}/beh/{sub}_task-fer_run-{run}_recording-{mod}_physio"
        try:
            cols = json.loads(z.read(m + ".json"))["Columns"]
            with z.open(m + ".tsv.gz") as f:
                a = np.loadtxt(gzip.open(f, "rt"), delimiter="\t", ndmin=2)
        except (KeyError, EOFError, OSError, ValueError):
            continue
        t = a[:, cols.index("onset")]
        if mod == "gsr":
            v = a[:, cols.index("GSR_Conductance_cal")]
        else:
            # Pupil diameter in camera pixels (LPD/RPD). The calibrated LPUPILD/RPUPILD
            # columns are all zero in this release; the AFFEC devkit also uses LPD/RPD.
            # Distance to the camera scales pixels, hence the per-trial baseline change.
            d = np.stack([a[:, cols.index("LPD")], a[:, cols.index("RPD")]], 1)
            d = np.where(d > 0, d, np.nan)
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")     # blinks: both eyes NaN
                v = np.nanmean(d, 1)
        streams[mod] = (t, v)
    out = []
    for trial, win in wins.items():
        feats = []
        for mod, lag in (("pupil", 0.0), ("gsr", SCR_LAG)):
            r = _change(*streams[mod], win, lag) if mod in streams else None
            feats += r if r is not None else [np.nan] * 4
        out.append((trial, np.array(feats, np.float32)))
    return sub, run, out


def body_features(args):
    import re
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    jobs = []
    for ev in sorted(ROOT.glob("sub-*/sub-*_task-fer_run-*_events.tsv")):
        k = re.match(r"(sub-[^_]+)_task-fer_run-(\d+)_events", ev.name)
        jobs.append((k[1], int(k[2]), ev))
    keys, xs = [], []
    with Pool(args.workers) as pool:
        for i, (sub, run, out) in enumerate(pool.imap_unordered(_read_body, jobs), 1):
            for trial, x in out:
                keys.append(f"{sub[4:]}|{run}|{trial}")
                xs.append(x)
            print(f"\r{i}/{len(jobs)} runs", end="", flush=True)
    x = np.stack(xs)
    np.savez_compressed(BODY, keys=np.array(keys), x=x)
    print(f"\nwrote {BODY}: {len(keys)} trials; pupil present in {np.isfinite(x[:, 0]).sum()}, "
          f"skin conductance in {np.isfinite(x[:, 4]).sum()}")


def body(args):
    """Face vs body signals vs both, on the trials that have all of them."""
    import torch
    torch.set_num_threads(4)
    d, xf = _joined()
    b = np.load(BODY)
    bpos = {k: i for i, k in enumerate(b["keys"])}
    keys = np.array([f"{s}|{r}|{t}" for s, r, t in zip(d["subject"], d["run"], d["trial"])])
    xb = np.array([b["x"][bpos[k]] if k in bpos else np.full(b["x"].shape[1], np.nan) for k in keys], np.float32)
    keep = np.isfinite(xb).all(1)
    d, xf, xb = {k: v[keep] for k, v in d.items()}, xf[keep], xb[keep]
    xf, xb = person_normalise(xf, d["subject"]), person_normalise(xb, d["subject"])
    inputs = {"face": xf, "body (pupil + skin conductance)": xb, "face + body": np.c_[xf, xb]}
    people = np.unique(d["subject"])
    rng = np.random.default_rng(0)
    print(f"{len(xf)} trials with face, pupil and skin conductance; {len(people)} people; "
          f"{args.reps} person re-splits\n")
    res = {"n_trials": int(len(xf)), "n_people": int(len(people)), "reps": args.reps}
    for target in TARGETS:
        y = d[target]
        acc: dict[str, list] = {}
        for rep in range(args.reps):
            tr, ca, te = person_split(people, rng)
            itr, ica, ite = (np.isin(d["subject"], list(g)) for g in (tr, ca, te))
            preds = {"none": (np.tile(y[itr].mean(0), (len(y), 1)), np.tile(y[itr].std(0), (len(y), 1)))}
            cc, ss = fit_stimulus(y[itr], d["cat"][itr])
            preds["stimulus"] = (cc[d["cat"]], ss[d["cat"]])
            for name, xi in inputs.items():
                preds[name] = _fit_face(xi, y, itr, seed=rep)
            for m, (c, s) in preds.items():
                q = qhat(score(y[ica], c[ica], s[ica]))
                acc.setdefault(m, []).append(
                    [np.mean(score(y[ite], c[ite], s[ite]) <= q), box_area(c[ite], s[ite], q),
                     *[np.corrcoef(c[ite, i], y[ite, i])[0, 1] if c[ite, i].std() > 0 else 0.0 for i in (0, 1)]])
                if m in ("none", "face + body"):
                    for k, (cv, ar) in personal(y[ite], c[ite], s[ite], d["subject"][ite], (40,), rng).items():
                        acc.setdefault(f"{m} + personal k={k}", []).append([cv, ar, np.nan, np.nan])
            print(f"\r{target}: {rep + 1}/{args.reps}", end="", flush=True)
        res[target] = {m: dict(zip(("coverage", "area", "corr_v", "corr_a"), np.nanmean(v, 0).tolist()))
                       for m, v in acc.items()}
        print(f"\r{target:>9}: {'what the system knows':<36} cover  box   corr V / A")
        for m, v in res[target].items():
            cr = "" if np.isnan(v["corr_v"]) else f"{v['corr_v']:+.2f} / {v['corr_a']:+.2f}"
            print(f"{'':>11}{m:<36} {v['coverage']:5.1%} {v['area']:5.1%}  {cr}")
        print()
    (OUT / "body.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / 'body.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "features", "face", "body-features", "body"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"baselines": baselines, "features": features, "face": face,
     "body-features": body_features, "body": body}[a.cmd](a)
