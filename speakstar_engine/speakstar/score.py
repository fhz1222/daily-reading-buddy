import json
from pathlib import Path

import numpy as np

from .gop import FEATURE_NAMES
from .model import PIPELINE_VERSION


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


class Calibrator:
    def __init__(self, data, fingerprint, locale):
        self.data = data
        if data.get("pipeline") != PIPELINE_VERSION or data.get("model_sha256") != fingerprint or data.get("locale") != locale:
            raise ValueError("Calibration does not match this model, precision, locale or feature pipeline")
        if data.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Calibration features do not match this engine")
        for key in ("mean", "scale", "weights"):
            if len(data[key]) != len(FEATURE_NAMES) or not np.isfinite(data[key]).all():
                raise ValueError("Invalid calibration parameters")
        if min(data["scale"]) <= 0 or not np.isfinite(data["bias"]):
            raise ValueError("Invalid calibration normalization")
        self.policy = data["policy"]
        if not 0 <= self.policy["bad_threshold"] < self.policy["good_threshold"] <= 1:
            raise ValueError("Invalid decision thresholds")

    @classmethod
    def load(cls, path, fingerprint, locale):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), fingerprint, locale)

    def probability(self, evidence):
        x = (np.asarray(evidence.vector()) - self.data["mean"]) / self.data["scale"]
        return float(sigmoid(x @ self.data["weights"] + self.data["bias"]))
