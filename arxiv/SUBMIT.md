# arXiv submission package

## Status: SUBMITTED (2026-08-21) - on hold for moderation

Submission ID: `submit/7979315` (arxiv.org/user)
Account: hassan_naeem | Endorsed for cs.LG by Shrey Pareek
Primary cs.LG, cross-list cs.CV, CC BY 4.0.

"on hold" means a moderator reviews it before announcement rather than it going
out automatically. Routine for a first submission from a new author. Do not
delete or resubmit; that loses queue position. Expect an email.

Once announced, add the arXiv ID to README.md and paper/main.tex.


Everything here is ready to upload. Built and verified 2026-08-21.

## Files

| File | Use |
|---|---|
| `arxiv-submission.tar.gz` | Upload this as the source. 230 KB. |
| `abstract.txt` | Paste into the Abstract field (1,530 chars; limit is 1,920). |
| `SUBMIT.md` | This file — metadata to enter. |

The tarball was verified in a clean directory the way arXiv builds it: three
`pdflatex` passes, **no BibTeX and no `refs.bib`**, relying on the bundled
`main.bbl`. Result: 8 pages, all 11 references rendered, no undefined citations
or references. `refs.bib` is deliberately excluded — arXiv does not run BibTeX,
so the `.bbl` is what matters.

## Metadata to enter

**Title**
```
Knowing When It Doesn't Know: Calibrated, Uncertainty-Aware Affect Estimation under Distribution Shift
```

**Authors**
```
Hassan Naeem
```
This must match the paper's author line and your arXiv account name exactly -
split citations across name variants are a nuisance to merge later.

**Categories**
- Primary: **cs.LG** (Machine Learning). The contribution is uncertainty
  quantification: evidential regression, split conformal prediction,
  calibration, selective prediction, OOD detection. Faces are the application
  domain, not the contribution.
- Cross-list: **cs.CV** (Computer Vision and Pattern Recognition), for the
  face-image application.

(Originally drafted as cs.CV primary. cs.LG is the better fit for a paper whose
own claim is that the methods are equivalent on accuracy and separate on
calibration.)

**Comments field** (optional but worth filling — it is where you note artifacts)
```
8 pages, 3 figures. Code and per-seed metrics: https://github.com/Hassan-Naeem-code/Emotion-Modelling
```

## License

arXiv asks you to choose one at submission. This is your call:

- **CC BY 4.0** — most permissive, standard for open science, maximises reuse and
  citation. Recommended unless a target venue forbids it.
- **arXiv non-exclusive license** — the conservative default. Choose this if you
  are unsure about the workshop's preprint policy; it grants arXiv only what it
  needs to distribute.

Either is compatible with subsequently submitting to a workshop. Check the
workshop's preprint policy first if it is double-blind — most allow arXiv
preprints, but a few require you not to advertise them during review.

## Before you click submit

1. Confirm your name is spelled as you want it cited — it is hard to change later.
2. Decide whether to add an ORCID and affiliation.
3. If the target workshop is **double-blind**, check whether posting now is
   permitted. If it is not, hold the arXiv post until after the review period.

## Regenerating this package

```bash
make arxiv
```
