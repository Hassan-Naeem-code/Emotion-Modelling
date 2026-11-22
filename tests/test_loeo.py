"""Leave-one-expression-out: the loader removes the held-out expression from
Train and Validation only, and the analysis measures it on Test."""
import numpy as np
from PIL import Image

from data.loaders import affectnet_va
from eval.loeo import evaluate


def _fake_affectnet(root, per_split=24):
    rng = np.random.default_rng(0)
    for split in ("Train", "Validation", "Test"):
        d = root / "AffectNetVA" / split
        for sub in ("images", "valence", "arousal", "emotion"):
            (d / sub).mkdir(parents=True)
        for n in range(per_split):
            i = f"{split}{n:03d}"
            Image.new("RGB", (8, 8)).save(d / "images" / f"{i}.jpg")
            np.save(d / "valence" / f"{i}_val.npy", np.float32(rng.uniform(-1, 1)))
            np.save(d / "arousal" / f"{i}_aro.npy", np.float32(rng.uniform(-1, 1)))
            np.save(d / "emotion" / f"{i}_exp.npy", np.int64(n % 8))


def _cfg(root, held=None, holdout_select=0.0):
    d = {"root": str(root), "image_size": 8, "norm_mean": [0, 0, 0], "norm_std": [1, 1, 1],
         "holdout_select": holdout_select}
    if held is not None:
        d["holdout_expression"] = held
    return {"data": d, "train": {}}


def _labels(ds):
    return {affectnet_va._expression(ds.split_dir, i) for i in ds.ids}


def test_holdout_removed_from_train_and_val_but_kept_in_test(tmp_path):
    _fake_affectnet(tmp_path)
    train, val, test, _ = affectnet_va.build(_cfg(tmp_path, held=3, holdout_select=0.25))
    assert 3 not in _labels(train)
    assert 3 not in _labels(train.select_ds)
    assert 3 not in _labels(val)
    assert 3 in _labels(test)
    assert len(test) == 24


def test_no_holdout_keeps_everything(tmp_path):
    _fake_affectnet(tmp_path)
    train, val, test, _ = affectnet_va.build(_cfg(tmp_path))
    assert len(train) == len(val) == len(test) == 24


def test_held_out_group_with_bigger_errors_loses_coverage():
    rng = np.random.default_rng(0)
    n = 4000

    def split(held_shift):
        emo = rng.integers(0, 7, size=n)
        sd = np.full((n, 2), 0.1)
        mean = rng.uniform(-0.5, 0.5, size=(n, 2))
        target = mean + rng.normal(size=(n, 2)) * sd
        target[emo == 2] += held_shift
        return {"mean": mean, "var": sd ** 2, "target": target, "emotion": emo}

    val = split(0.0)
    val = {k: v[val["emotion"] != 2] for k, v in val.items()}   # held out of calibration
    m = evaluate(val, split(0.4), held=2)
    assert m["seen_cov_joint"] > 0.87
    assert m["held_cov_joint"] < 0.5
    assert m["n_held"] > 0
