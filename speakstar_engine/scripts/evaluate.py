"""Held-out evaluation: emits a report, never silently approves a model."""
import argparse
import json
from pathlib import Path

from speakstar.evaluation import metrics, validate_rows
from speakstar.score import Calibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Evaluation report already exists; preserve the held-out result")
    rows = [json.loads(s) for s in args.features.read_text(encoding="utf-8").splitlines() if s.strip()]
    fingerprint, locale, pipeline = validate_rows(rows, {"test"})
    calibrator = Calibrator.load(args.calibration, fingerprint, locale)
    seen = set(calibrator.data["training_speakers"]) | set(calibrator.data["dev_speakers"])
    if seen & {r["speaker"] for r in rows}:
        raise ValueError("Held-out speakers overlap calibration train/dev")
    all_scores = metrics(rows, calibrator)
    children = metrics([r for r in rows if r.get("is_child")], calibrator)
    report = {"calibration": calibrator.data["id"], "model_sha256": fingerprint,
              "locale": locale, "pipeline": pipeline, "all": all_scores, "children": children,
              "decision": "manual_review_required", "validated": False,
              "limitations": ["Report extraction skips and label coverage alongside these metrics.",
                             "No word correlation without independent word-level label aggregation.",
                             "No published baseline comparison is implied by this report."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
