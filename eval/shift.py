"""Cross-dataset / distribution-shift evaluation helpers.

The spine of the paper: train in-distribution (AffectNet images / synthetic
in-dist), then measure how calibration and coverage degrade on a shifted set
(AFEW-VA frames / synthetic OOD split). A well-behaved uncertainty model should
(a) raise its uncertainty under shift, (b) keep conformal coverage near target,
and (c) let selective prediction recover low risk by abstaining.
"""
from __future__ import annotations

from .metrics import (accuracy_metrics, empirical_coverage, ood_auroc,
                      regression_ece, sharpness)


def shift_report(pred_id: dict, pred_ood: dict) -> dict:
    """Compare in-dist vs OOD predictions. Each pred dict has mean/var/target."""
    rep = {}
    for tag, p in (("id", pred_id), ("ood", pred_ood)):
        acc = accuracy_metrics(p["mean"], p["target"])
        rep[f"{tag}_ccc"] = acc["ccc_mean"]
        rep[f"{tag}_rmse"] = acc["rmse_mean"]
        rep[f"{tag}_ece"] = regression_ece(p["mean"], p["var"], p["target"])
        rep[f"{tag}_sharpness"] = sharpness(p["var"])
    rep["ece_degradation"] = rep["ood_ece"] - rep["id_ece"]
    rep["uncertainty_rises_under_shift"] = rep["ood_sharpness"] > rep["id_sharpness"]
    rep["ood_auroc"] = ood_auroc(pred_id["var"], pred_ood["var"])
    return rep
