import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Offline phoneme assessment prototype; no stress or intonation scoring")
    parser.add_argument("audio", type=Path)
    parser.add_argument("reference_text")
    parser.add_argument("--model-dir", type=Path, default=Path("models/wav2vec2"))
    parser.add_argument("--locale", choices=["en-gb", "en-us"], default="en-gb")
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--evidence", action="store_true", help="Include raw evidence for research/calibration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        from .engine import PronunciationEngine
        result = PronunciationEngine(args.model_dir, args.locale, args.precision, args.calibration).assess(
            args.audio, args.reference_text, args.evidence)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(serialized)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "Error", "error": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2
