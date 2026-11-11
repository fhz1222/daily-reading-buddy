"""Exercise executable training/evaluation tools with labelled synthetic features.

These test fixtures are not pronunciation data and cannot validate accuracy.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from speakstar.gop import Evidence
from speakstar.model import PIPELINE_VERSION

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class WorkflowTests(unittest.TestCase):
    def test_calibration_and_heldout_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = []
            for split in ["train", "dev", "test"]:
                for i in range(100):
                    correct = i % 2
                    evidence = Evidence(-.1 if correct else -8, .9 if correct else .01,
                                        -2 if correct else 4, -3, 60, "s", 2, True)
                    dataset.append({"model_sha256": "synthetic-test-only", "locale": "en-gb", "pipeline": PIPELINE_VERSION,
                                    "speaker": split + str(i // 20), "split": split,
                                    "utterance_id": split + str(i), "word_index": 0, "phone_index": 0,
                                    "phone": "θ", "correct": correct, "is_child": True,
                                    "evidence": evidence.to_dict()})
            train_dev, test = tmp / "train-dev.jsonl", tmp / "test.jsonl"
            for file, splits in [(train_dev, {"train", "dev"}), (test, {"test"})]:
                file.write_text("\n".join(json.dumps(r) for r in dataset if r["split"] in splits), encoding="utf-8")
            calibration, report = tmp / "calibration.json", tmp / "report.json"
            trained = subprocess.run([sys.executable, str(SCRIPTS / "calibrate.py"), str(train_dev), "--output", str(calibration)], capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            evaluated = subprocess.run([sys.executable, str(SCRIPTS / "evaluate.py"), str(test), "--calibration", str(calibration), "--output", str(report)], capture_output=True, text=True)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            result = json.loads(report.read_text())
            self.assertFalse(result["validated"])
            self.assertEqual(result["children"]["count"], 100)
            # A held-out result must not be overwritten accidentally.
            again = subprocess.run([sys.executable, str(SCRIPTS / "evaluate.py"), str(test), "--calibration", str(calibration), "--output", str(report)], capture_output=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == "__main__":
    unittest.main()
