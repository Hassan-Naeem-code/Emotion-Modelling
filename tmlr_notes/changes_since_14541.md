# Changes since TMLR submission 14541 (desk-rejected 22 Nov 2026, no reason given)

1. **Second dataset.** All coverage results are now replicated on AFEW-VA
   (video, 600 clips, split by clip so no clip spans two splits). Per-axis
   intervals again fall short of joint coverage (84-86% at nominal 90%); the
   max-score region restores it and is 7-13% smaller than Bonferroni for
   methods with learned uncertainty. Coverage is also reported per clip.
   Limitations stated: weak AFEW-VA predictor (CCC 0.30-0.33), per-clip
   coverage about 2 points under per-frame.

2. **Systematic held-out-group experiment.** Instead of relying on one
   naturally missing expression (contempt, n=70), each of AffectNet's seven
   expressions is removed from training and calibration in turn (3 seeds).
   Coverage on the held-out expression falls to 28-82% while the others stay
   at 91-93%; predictive uncertainty does not flag the held-out faces
   (AUROC 0.47-0.81); Bonferroni does not prevent the failure.

3. **Framing.** Introduction and abstract now lead with the general point
   (marginal conformal guarantees mislead for multi-output targets and for
   groups missing from calibration), with VA estimation as the test case.

4. **Claims revised** to match the new evidence; limitations section rewritten.
