import hashlib
import json
from pathlib import Path
import threading

import numpy as np

PIPELINE_VERSION = "phoneme-evidence-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AcousticModel:
    """Loads only a local export. No network access or fallback downloads."""
    def __init__(self, directory, precision="fp32", threads=2):
        import onnxruntime as ort
        root = Path(directory)
        self.meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if precision not in {"fp32", "int8"}:
            raise ValueError("Expected fp32 or int8")
        model_path = root / f"model.{precision}.onnx"
        self.fingerprint = sha256(model_path)
        if self.fingerprint != self.meta["sha256"][precision]:
            raise ValueError("Model checksum mismatch; export again")
        self.vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        self.blank = int(self.meta["blank_id"])
        self.stride = int(self.meta["stride_samples"])
        self.receptive = int(self.meta["receptive_samples"])
        self.precision = precision
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.lock = threading.Lock()

    def infer(self, samples):
        x = np.asarray(samples, dtype=np.float32)[None, :]
        with self.lock:
            logits = self.session.run(None, {"input_values": x})[0][0]
        if logits.ndim != 2 or logits.shape[1] != len(self.vocab) or not np.isfinite(logits).all():
            raise ValueError("Invalid acoustic model output")
        maximum = logits.max(axis=-1, keepdims=True)
        return logits - maximum - np.log(np.exp(logits - maximum).sum(axis=-1, keepdims=True))
