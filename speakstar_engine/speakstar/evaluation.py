"""Metrics include abstentions; an engine cannot pass by rejecting every clip."""
import numpy as np

from .feedback import assess_phone
from .gop import Evidence

ERRORS = {"needs_practice", "possibly_missing"}


def evidence_from_row(row):
    return Evidence(**row["evidence"])


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows, calibrator):
    if not rows:
        return {"count": 0}
    tp = fp = fn = correct = incorrect = uncertain = hints = matched_hints = 0
    probabilities, labels, human_scores = [], [], []
    for row in rows:
        ev = evidence_from_row(row)
        assessment = assess_phone(row["phone"], ev, calibrator)
        label = row["correct"]
        predicted_bad = assessment["status"] in ERRORS
        correct += int(label == 1)
        incorrect += int(label == 0)
        tp += int(predicted_bad and label == 0)
        fp += int(predicted_bad and label == 1)
        fn += int(not predicted_bad and label == 0)  # abstentions count as missed errors
        uncertain += int(assessment["status"] == "uncertain")
        if assessment.get("possible_substitute") and row.get("pronounced_phone"):
            hints += 1
            matched_hints += int(assessment["possible_substitute"] == row["pronounced_phone"])
        probabilities.append(calibrator.probability(ev))
        labels.append(label)
        human_scores.append(row.get("human_score", label * 2) / 2)
    p, y = np.array(probabilities), np.array(labels)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        mask = (p >= lo) & (p < lo + 0.1 if lo < 0.9 else p <= 1)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    corr = None
    if np.std(p) > 0 and np.std(human_scores) > 0:
        corr = float(np.corrcoef(p, human_scores)[0, 1])
    return {"count": len(rows), "correct_count": correct, "incorrect_count": incorrect,
            "false_rejection_rate": ratio(fp, correct), "false_rejections": fp,
            "precision": precision, "recall": recall,
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "uncertain_rate": uncertain / len(rows),
            "substitution_hint_accuracy": ratio(matched_hints, hints), "labelled_hint_count": hints,
            "ece": ece, "phoneme_score_correlation": corr}


def validate_rows(rows, splits=None):
    if not rows:
        raise ValueError("Feature dataset is empty")
    speakers, keys = {}, set()
    identities = {(r["model_sha256"], r["locale"], r["pipeline"]) for r in rows}
    if len(identities) != 1:
        raise ValueError("Cannot mix models, precisions, locales or feature pipelines")
    for row in rows:
        if row["correct"] not in {0, 1}:
            raise ValueError("correct must be 0 or 1")
        split = row["split"]
        if splits and split not in splits:
            raise ValueError(f"Unexpected split: {split}")
        prior = speakers.setdefault(row["speaker"], split)
        if prior != split:
            raise ValueError("Speaker leakage between dataset splits")
        key = (row["utterance_id"], row["word_index"], row["phone_index"])
        if key in keys:
            raise ValueError("Duplicate phoneme label")
        keys.add(key)
        if not np.isfinite(evidence_from_row(row).vector()).all():
            raise ValueError("Nonfinite evidence")
    return next(iter(identities))
