"""AFEW-VA as a second benchmark: clips (not frames) are split, so no clip's
frames land in two splits, and analysis copes with data that has no expression
labels while reporting clip-averaged coverage."""
import json

import numpy as np
from PIL import Image

from data.loaders import afew_va
from eval.analyze import evaluate


def _fake_afew(root, n_clips=40, frames=5):
    rng = np.random.default_rng(0)
    for c in range(n_clips):
        clip = root / "AFEW-VA" / f"{c:03d}"
        clip.mkdir(parents=True)
        ann = {"video_id": f"{c:03d}", "frames": {}}
        for f in range(frames):
            fid = f"{f:05d}"
            Image.new("RGB", (8, 8)).save(clip / f"{fid}.png")
            ann["frames"][fid] = {"valence": int(rng.integers(-10, 11)),
                                  "arousal": int(rng.integers(-10, 11))}
        (clip / f"{c:03d}.json").write_text(json.dumps(ann))


def _cfg(root, **data):
    d = {"root": str(root), "image_size": 8, "norm_mean": [0, 0, 0],
         "norm_std": [1, 1, 1], **data}
    return {"data": d, "train": {"augment": True}}


def _clips(ds):
    return {ds.group_names[g] for g in ds.groups}


def test_within_protocol_splits_by_clip(tmp_path):
    _fake_afew(tmp_path)
    tr, va, te, ood = afew_va.build(_cfg(tmp_path, afew_protocol="within", holdout_select=0.1))
    sel = tr.select_ds
    splits = [_clips(tr), _clips(sel), _clips(va), _clips(te)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not splits[i] & splits[j], "a clip appears in two splits"
    assert sum(len(s) for s in splits) == 40
    assert (len(va), len(te)) == (8 * 5, 8 * 5)           # 20% of 40 clips, 5 frames each
    assert ood is None
    assert tr.augment is not None and va.augment is None  # augment train only
    x, y = te[0]
    assert x.shape == (3, 8, 8) and -1 <= float(y.min()) and float(y.max()) <= 1


def test_split_is_fixed_across_runs(tmp_path):
    _fake_afew(tmp_path)
    a = afew_va.build(_cfg(tmp_path, afew_protocol="within"))[2]
    b = afew_va.build(_cfg(tmp_path, afew_protocol="within"))[2]
    assert _clips(a) == _clips(b)


def test_shift_protocol_unchanged(tmp_path):
    _fake_afew(tmp_path, n_clips=4)
    tr, va, te, ood = afew_va.build(_cfg(tmp_path))
    assert tr is None and va is None and te is ood and len(te) == 20


def test_analysis_without_expression_labels_reports_clip_coverage():
    rng = np.random.default_rng(0)

    def split(n=3000):
        sd = np.full((n, 2), 0.1)
        mean = rng.uniform(-0.5, 0.5, size=(n, 2))
        return {"mean": mean, "var": sd ** 2, "target": mean + rng.normal(size=(n, 2)) * sd,
                "emotion": np.full(n, -1), "group": np.repeat(np.arange(n // 30), 30)}

    m = evaluate({"val": split(), "test": split()}, "gaussian")
    assert "worst_seen_joint" not in m
    assert m["n_clips"] == 100
    assert abs(m["cov_joint_clip"] - m["cov_joint"]) < 1e-9   # equal-sized clips
    assert m["cov_joint"] > 0.87
