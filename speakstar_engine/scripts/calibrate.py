"""Train logistic calibration on train, select a conservative threshold on dev only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from speakstar.evaluation import evidence_from_row, metrics, validate_rows
from speakstar.gop import FEATURE_NAMES
from speakstar.score import Calibrator, sigmoid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path, help="JSONL containing train/dev ONLY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.features.read_text(encoding="utf-8").splitlines() if s.strip()]
    fingerprint, locale, pipeline = validate_rows(rows, {"train", "dev"})
    train, dev = [[r for r in rows if r["split"] == split] for split in ["train", "dev"]]
    if len(train) < 50 or len(dev) < 30 or len({r["correct"] for r in train}) != 2 or len({r["correct"] for r in dev}) != 2:
        raise ValueError("Need both classes, >=50 train and >=30 dev phonemes; more data is needed for validation")
    x = np.array([evidence_from_row(r).vector() for r in train])
    y = np.array([r["correct"] for r in train])
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    x = (x - mean) / scale
    weights, bias = np.zeros(x.shape[1]), 0.0
    for _ in range(2500):
        error = sigmoid(x @ weights + bias) - y
        weights -= 0.05 * (x.T @ error / len(x) + 0.001 * weights)
        bias -= 0.05 * error.mean()
    support = {}
    for row in dev:
        counts = support.setdefault(row["phone"], {"correct": 0, "incorrect": 0})
        counts["correct" if row["correct"] else "incorrect"] += 1
    artifact = {"id": "calibration-" + hashlib.sha256(args.features.read_bytes()).hexdigest()[:12],
                "pipeline": pipeline, "model_sha256": fingerprint, "locale": locale,
                "feature_names": FEATURE_NAMES, "mean": mean.tolist(), "scale": scale.tolist(),
                "weights": weights.tolist(), "bias": float(bias),
                "training_speakers": sorted({r["speaker"] for r in train}),
                "dev_speakers": sorted({r["speaker"] for r in dev}),
                "policy": {"bad_threshold": 0.05, "good_threshold": 0.85,
                           "deletion_threshold": 1e9, "substitution_margin": 1e9,
                           "dev_ece": 0.0, "phone_support": support},
                "validated": False,
                "notes": "Substitution and deletion hints disabled until independently labelled dev evidence supports thresholds."}
    calibrator = Calibrator(artifact, fingerprint, locale)
    artifact["policy"]["dev_ece"] = metrics(dev, calibrator)["ece"]
    # Favor recall subject to <=5% false rejection on dev; this does not certify test performance.
    choices = []
    for threshold in np.linspace(0.05, 0.5, 46):
        artifact["policy"]["bad_threshold"] = float(threshold)
        scores = metrics(dev, calibrator)
        if scores["false_rejection_rate"] is not None and scores["false_rejection_rate"] <= 0.05:
            choices.append((scores["recall"] or 0, float(threshold)))
    artifact["policy"]["bad_threshold"] = max(choices)[1] if choices else 0.0
    artifact["dev_metrics"] = metrics(dev, calibrator)
    artifact["dev_child_metrics"] = metrics([r for r in dev if r.get("is_child")], calibrator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(artifact["dev_child_metrics"], indent=2))


if __name__ == "__main__":
    main()
