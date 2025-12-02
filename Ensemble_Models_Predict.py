"""
Ensemble_Models_Predict.py

This file assumes you already HAVE:
    lcnet_results
    vtb_results
    cnn_results

All are dicts produced by LCNetPredictor, VTBPredictor, CNNPredictor:
{
    "labels": [...],
    "scores": [...],
    "predicted": [...],
    "time": float
}

This file builds:
    1) Vote Ensemble
    2) MoE Ensemble
without recomputing any model inference.
"""

# -------------------------------
# Imports
# -------------------------------
from typing import Dict, List


# -------------------------------
# Helpers
# -------------------------------
def _lower_set(s):
    """Normalize a set of strings to lowercase."""
    return {x.lower() for x in s}


# -------------------------------
# Attribute Routing Categories
# -------------------------------
PERSONAL = _lower_set({
    "Age18-60", "AgeLess18", "AgeOver60", "Female"
})

VIEW = _lower_set({
    "Back", "Front", "Side"
})

TOP_CLOTHING = _lower_set({
    "LongSleeve", "ShortSleeve", "LongCoat",
    "UpperLogo", "UpperPlaid", "UpperSplice", "UpperStride"
})

BOTTOM_CLOTHING = _lower_set({
    "Trousers", "Shorts", "Skirt&Dress",
    "LowerPattern", "LowerStripe"
})

ACCESSORIES = _lower_set({
    "Backpack", "Boots", "Glasses",
    "HandBag", "Hat", "ShoulderBag"
})

MISC_IGNORE = _lower_set({
    "HoldObjectsInFront"
})


# -------------------------------
# Convert model output to dict
# -------------------------------
def _scores_to_dict(results):
    """
    Convert labels + scores list → dict[label_lower] = score
    """
    return {label.lower(): score for label, score in zip(results["labels"], results["scores"])}


# ============================================================
# 1) Majority Vote Ensemble
# ============================================================
def vote_ensemble(lcnet_results, vtb_results, cnn_results, threshold=0.5):
    """Majority voting ensemble: 2-out-of-3 wins."""

    lc = _scores_to_dict(lcnet_results)
    vt = _scores_to_dict(vtb_results)
    cn = _scores_to_dict(cnn_results)

    labels_original = lcnet_results["labels"]
    labels_lower = [lbl.lower() for lbl in labels_original]

    final_scores = []
    predicted = []

    for orig, lbl in zip(labels_original, labels_lower):
        votes = 0
        if lc.get(lbl, 0) >= threshold:
            votes += 1
        if vt.get(lbl, 0) >= threshold:
            votes += 1
        if cn.get(lbl, 0) >= threshold:
            votes += 1

        final = 1.0 if votes >= 2 else 0.0
        final_scores.append(final)

        if final == 1.0:
            predicted.append((orig, final))

    ensemble_time = (
        lcnet_results.get("time", 0.0)
        + vtb_results.get("time", 0.0)
        + cnn_results.get("time", 0.0)
    )

    return {
        "model": "Vote Ensemble",
        "labels": labels_original,
        "scores": final_scores,
        "predicted": predicted,
        "time": ensemble_time
    }


# ============================================================
# 2) Mixture-of-Experts (MoE)
# ============================================================
def moe_ensemble(lcnet_results, vtb_results, cnn_results):
    """Routes attributes to their best-performing expert."""

    lc = _scores_to_dict(lcnet_results)
    vt = _scores_to_dict(vtb_results)
    cn = _scores_to_dict(cnn_results)

    labels_original = lcnet_results["labels"]
    labels_lower = [lbl.lower() for lbl in labels_original]

    final_scores = []
    predicted = []

    for orig, lbl in zip(labels_original, labels_lower):

        if lbl in PERSONAL:
            score = vt.get(lbl, 0.0)

        elif lbl in VIEW:
            score = lc.get(lbl, 0.0)

        elif lbl in TOP_CLOTHING:
            score = lc.get(lbl, 0.0)

        elif lbl in BOTTOM_CLOTHING:
            score = cn.get(lbl, 0.0)

        elif lbl in ACCESSORIES:
            score = cn.get(lbl, 0.0)

        elif lbl in MISC_IGNORE:
            score = 0.0

        else:
            score = 0.0

        final_scores.append(score)

        if score >= 0.5:
            predicted.append((orig, score))

    ensemble_time = (
        lcnet_results.get("time", 0.0)
        + vtb_results.get("time", 0.0)
        + cnn_results.get("time", 0.0)
    )

    return {
        "model": "MoE Ensemble",
        "labels": labels_original,
        "scores": final_scores,
        "predicted": predicted,
        "time": ensemble_time
    }
