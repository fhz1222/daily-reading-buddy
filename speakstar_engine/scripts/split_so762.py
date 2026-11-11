"""Read official Kaldi manifests; split official train by speaker, preserve test."""
import argparse
import json
from pathlib import Path
import random


def read_pairs(path):
    return dict(line.split(maxsplit=1) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=762)
    args = parser.parse_args()
    train = read_pairs(args.dataset / "train/utt2spk")
    test = read_pairs(args.dataset / "test/utt2spk")
    speakers = sorted(set(train.values()))
    if set(speakers) & set(test.values()):
        raise ValueError("Official train/test speaker overlap")
    if len(speakers) < 3:
        raise ValueError("Need >=3 train speakers")
    random.Random(args.seed).shuffle(speakers)
    dev = set(speakers[:max(1, round(len(speakers) * 0.2))])
    manifest = {"seed": args.seed, "utterances": {}}
    for source, utterances in [("train", train), ("test", test)]:
        ages = read_pairs(args.dataset / source / "spk2age")
        paths = read_pairs(args.dataset / source / "wav.scp")
        for uid, speaker in utterances.items():
            # Never execute wav.scp entries as shell pipelines.
            if paths[uid].endswith("|"):
                raise ValueError("Only file paths are supported in wav.scp")
            path = Path(paths[uid].replace("\\", "/"))
            resolved = path if path.is_absolute() else args.dataset / path
            manifest["utterances"][uid] = {"speaker": speaker, "age": int(ages[speaker]),
                "is_child": int(ages[speaker]) < 18,
                "split": "test" if source == "test" else "dev" if speaker in dev else "train",
                "audio": str(resolved.resolve())}
    if args.output.exists():
        raise ValueError("Split manifest already exists; keep splits fixed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
