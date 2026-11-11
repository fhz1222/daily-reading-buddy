"""Attach labels ONLY where annotated canonical phones exactly match G2P phones.

Mismatches are recorded, never silently assigned to another sound. Excluded
coverage must accompany metrics to avoid an inflated result on easy examples.
"""
import argparse
from importlib.resources import files
import json
from pathlib import Path
import re

from speakstar.engine import PronunciationEngine


def map_phone(phone, mapping, locale):
    phone = phone.strip()
    if phone == "AH0":
        return "ə"
    base = re.sub(r"[012]$", "", phone)
    if locale == "en-us":
        if base == "OW":
            return "oʊ"
        if base == "ER":
            return "ɚ" if phone.endswith("0") else "ɜː"
    return mapping.get(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models/wav2vec2"))
    parser.add_argument("--locale", choices=["en-gb", "en-us"], default="en-gb")
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke tests only; not a representative evaluation")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; do not overwrite evidence")
    if "test" in args.splits and len(args.splits) != 1:
        raise ValueError("Extract test separately after freezing calibration and acceptance criteria")
    labels = json.loads(args.scores.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["utterances"]
    mapping = json.loads(files("speakstar").joinpath("data/arpabet_to_ipa.json").read_text(encoding="utf-8"))
    engine = PronunciationEngine(args.model_dir, args.locale, args.precision)
    stats = {"utterances": 0, "annotated_words": 0, "annotated_phones": 0,
             "labelled_phones": 0, "canonical_mismatch_words": 0,
             "ambiguous_labels": 0, "failed": [], "limited_run": args.limit is not None,
             "label_policy": "mean score >=1 is correct (accent accepted); <=0.4 is incorrect; other scores excluded"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for uid, meta in sorted(manifest.items()):
            if meta["split"] not in args.splits:
                continue
            if args.limit is not None and stats["utterances"] >= args.limit:
                break
            stats["utterances"] += 1
            annotation = labels[uid]
            stats["annotated_words"] += len(annotation["words"])
            stats["annotated_phones"] += sum(len(word["phones-accuracy"]) for word in annotation["words"])
            try:
                result = engine.assess(meta["audio"], annotation["text"], include_evidence=True)
                if result["status"] != "Success" or len(result["Words"]) != len(annotation["words"]):
                    raise ValueError("Audio or word alignment unavailable")
                for wi, (word, gold) in enumerate(zip(result["Words"], annotation["words"], strict=True)):
                    arpabet = gold["phones"].split() if isinstance(gold["phones"], str) else gold["phones"]
                    canonical = [map_phone(p, mapping, args.locale) for p in arpabet]
                    if canonical != [p["Phoneme"] for p in word["Phonemes"]]:
                        stats["canonical_mismatch_words"] += 1
                        continue
                    mispronounced = {r["index"]: r["pronounced-phone"] for r in gold.get("mispronunciations", [])}
                    for pi, (phone, label) in enumerate(zip(word["Phonemes"], gold["phones-accuracy"], strict=True)):
                        correct = 1 if label >= 1 else 0 if label <= 0.4 else None
                        if correct is None:
                            stats["ambiguous_labels"] += 1
                            continue
                        row = {k: meta[k] for k in ["speaker", "is_child", "split"]}
                        row.update(utterance_id=uid, word_index=wi, phone_index=pi,
                                   phone=phone["Phoneme"], evidence=phone["evidence"],
                                   correct=correct, human_score=label,
                                   pronounced_phone=map_phone(mispronounced.get(pi, ""), mapping, args.locale),
                                   model_sha256=engine.model.fingerprint, locale=args.locale,
                                   pipeline=result["engine"]["pipeline"])
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stats["labelled_phones"] += 1
                output.flush()
            except (OSError, RuntimeError, ValueError) as error:
                stats["failed"].append({"utterance_id": uid, "error": str(error)})
    stats["label_coverage"] = stats["labelled_phones"] / stats["annotated_phones"] if stats["annotated_phones"] else None
    args.output.with_suffix(".coverage.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
