# arXiv v2: replace the existing entry

Upload the new paper as a **replacement** of your v1 entry. It keeps the same
arXiv ID; v1 stays visible in the version history.

## Steps

1. Log in at https://arxiv.org/user and find the v1 paper.
   - **If it has an arXiv ID** (announced): click **Replace**.
   - **If it still shows "on hold"** (never announced): do not start a second
     submission. Open the pending one, choose **Unsubmit**, then upload the files
     below and resubmit. That keeps your place under the same endorsement.
2. Upload `arxiv-v2-submission.tar.gz` as the source. It is already verified the
   way arXiv builds it (3 pdflatex passes, bundled .bbl, 11 pages, no undefined
   references). `preview.pdf` shows exactly what arXiv will produce.
3. Fill in the fields below, check the preview, and submit.

## Title
```
Two Axes, One Guarantee: Joint Conformal Uncertainty for Valence-Arousal Estimation, and Where It Breaks
```

## Authors
```
Hassan Naeem
```

## Abstract
Paste the contents of `abstract.txt` (1,650 characters; the limit is 1,920).

## Comments
```
11 pages, 1 figure, 4 tables. Code, per-sample predictions and checkpoints: https://github.com/Hassan-Naeem-code/Emotion-Modelling . Live demo and every result per seed: https://va-uncertainty-demo.vercel.app
```

## Categories
Primary **cs.LG**, cross-list **cs.CV** (same as v1).

## License
CC BY 4.0 (same as v1).

## Replacement note (asked on the Replace form)
```
Major revision: new title and framing. Adds joint (max-score) conformal regions with a finite-sample guarantee, per-expression coverage, an unseen-expression analysis including feature-space detectors, five seeds with proper scoring rules, and a calibration split never used for model selection.
```
