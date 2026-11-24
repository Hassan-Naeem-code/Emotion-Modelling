# Changes since TMLR submission 14541 (desk-rejected 22 Nov 2026, no reason given)

1. **Second dataset.** All coverage results are replicated on AFEW-VA (video,
   600 clips, split by clip so no clip spans two splits). Per-axis intervals
   again fall short of joint coverage (84-86% at nominal 90%); the max-score
   region restores it and is 7-13% smaller than Bonferroni for methods with
   learned uncertainty. Coverage is also reported per clip. Limitations stated:
   weak AFEW-VA predictor (CCC 0.30-0.33), per-clip coverage about 2 points
   under per-frame.

2. **Comparison with copula-based conformal regions** (Messoudi et al., 2021):
   independence, Gaussian and empirical copulas on both datasets. The
   parametric copulas are 2-13% larger than the max-score region; the empirical
   copula matches it in size (and is 1-7% smaller on AFEW-VA) but has only an
   approximate guarantee. The paper's claims were revised accordingly: we no
   longer claim the max-score is the tightest joint region, only that it is as
   small as the best copula with an exact guarantee.

3. **Systematic held-out-group experiment.** Instead of relying on one
   naturally missing expression (contempt, n=70), each of AffectNet's seven
   expressions is removed from training and calibration in turn (3 seeds).
   Coverage on the held-out expression falls to 28-82% while the others stay
   at 91-93%; predictive uncertainty does not flag the held-out faces
   (AUROC 0.47-0.81); Bonferroni does not prevent the failure.

4. **A remedy, with its cost.** With k labelled faces of the missing
   expression, group (Mondrian) recalibration restores its coverage to
   90-91% from k=10, at 3-9x the usual region size. A shift-corrected variant
   (half the k faces estimate the expression's mean offset, half calibrate;
   still exact) halves that cost for the worst-failing expressions (happy
   5.1x -> 2.6x, sad 9.3x -> 2.4x at k=20). Its precondition (knowing which
   faces belong to the new expression) is stated as a limitation.

5. **Third dataset, second modality (text).** EmoBank (10,062 sentences,
   7 kinds of text), frozen RoBERTa encoder, same heads. Per-axis intervals
   cover the VA point only 82%; the max-score region covers 90.0% averaged
   over 200 calibration/test re-splits. Holding out one kind of text at a time:
   news headlines drop to 55% coverage, letters and fiction to 80-85%, the
   other four are unaffected; uncertainty again fails to flag (AUROC
   0.20-0.57). 20 labelled sentences restore coverage. Reported honestly as a
   qualified replication.

6. **Ellipsoidal regions and four further references** (Messoudi et al. 2022;
   Feldman et al. 2023; Ding et al. 2023; Gibbs et al. 2025; Podkopaev &
   Ramdas 2021). The ellipse is 4-9% smaller than the max-score rectangle on
   AffectNet but cannot be read as one interval per axis; stated as such.

7. **Framing.** Introduction and abstract now lead with the general point
   (marginal conformal guarantees mislead for multi-output targets and for
   groups missing from calibration), with VA estimation as the test case.

8. **Claims revised** to match the new evidence; limitations rewritten. All new
   tables are computed from released predictions without a GPU, with tests.
