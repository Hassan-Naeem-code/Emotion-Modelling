import math

import numpy as np

from eval import affec


def test_qhat_rank_and_too_few():
    s = np.arange(1, 11, dtype=float)          # n=10, alpha=0.1 -> rank 10
    assert affec.qhat(s) == 10
    assert affec.qhat(s[:8]) == math.inf


def test_box_area_is_share_of_square_and_clipped():
    c = np.zeros((1, 2))
    assert affec.box_area(c, np.full((1, 2), 0.5), 1.0) == 0.25      # 1x1 of a 2x2 square
    assert affec.box_area(c, np.full((1, 2), 5.0), 1.0) == 1.0       # clipped to the square


def test_person_normalise_uses_each_person_alone():
    x = np.array([[1.0], [3.0], [100.0], [300.0]])
    z = affec.person_normalise(x, np.array(["a", "a", "b", "b"]))
    assert np.allclose(z.ravel(), [-1, 1, -1, 1], atol=1e-4)


def test_personal_recalibration_covers_shifted_person():
    rng = np.random.default_rng(0)
    n = 60
    c = rng.uniform(-0.5, 0.5, (n, 2))
    s = np.full((n, 2), 0.1)
    y = c + [0.4, -0.3] + 0.1 * rng.standard_normal((n, 2))      # this person sits off the model
    subj = np.array(["p"] * n)
    cov, _ = affec.personal(y, c, s, subj, (20,), rng)[20]
    assert cov > 0.75                                            # without the offset it would be ~0


def test_windows_pair_clip_with_preceding_fixation(tmp_path):
    ev = tmp_path / "e.tsv"
    ev.write_text("onset\tduration\ttrial_type\tflag\tsubject\trun\ttrial\tlocal_time\tstim_file\n"
                  "10\t1\tsad\tsecond_fix\ta\t1\t1\t0\tx\n"
                  "11\t4\tsad\tvideo\ta\t1\t1\t0\tx\n"
                  "15\t1\tsad\tlast_frame_video\ta\t1\t1\t0\tx\n")
    assert affec._windows(ev) == {1: (11.0, 16.0, 10.0, 11.0)}
