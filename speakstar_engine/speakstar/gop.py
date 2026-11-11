from dataclasses import asdict, dataclass
from importlib.resources import files
import json

import numpy as np

from .align import ctc_log_likelihood

FEATURE_NAMES = ["gop", "posterior", "substitution_lpr", "deletion_lpr", "duration_ms"]


@dataclass
class Evidence:
    gop: float
    posterior: float
    substitution_lpr: float
    deletion_lpr: float
    duration_ms: float
    alternative: str | None
    alternative_margin: float
    alignment_sane: bool

    def vector(self):
        return [getattr(self, name) for name in FEATURE_NAMES]

    def to_dict(self):
        return asdict(self)


def extract_evidence(log_probs, targets, spans, vocab, blank=0, frame_ms=20, confusions=None):
    if confusions is None:
        confusions = json.loads(files("speakstar").joinpath("data/confusions.json").read_text(encoding="utf-8"))
    id_to_token = {v: k for k, v in vocab.items()}
    phone_ids = [v for k, v in vocab.items() if not k.startswith("<") and k != "|"]
    base = ctc_log_likelihood(log_probs, targets, blank)
    result = []
    for i, (target, span) in enumerate(zip(targets, spans, strict=True)):
        frames = log_probs[span.start:span.end]
        count = span.end - span.start
        phone = id_to_token[target]
        alternatives = []
        for alternative in confusions.get(phone, []):
            if alternative not in vocab or vocab[alternative] == target:
                continue
            variant = list(targets)
            variant[i] = vocab[alternative]
            likelihood = ctc_log_likelihood(log_probs, variant, blank)
            if np.isfinite(likelihood):
                alternatives.append(((likelihood - base) / count, alternative))
        alternatives.sort(reverse=True)
        best = alternatives[0] if alternatives else (0.0, None)
        # An alternative must beat both the target and the next alternative.
        margin = best[0] - max(0.0, alternatives[1][0] if len(alternatives) > 1 else 0.0)
        deleted = ctc_log_likelihood(log_probs, list(targets[:i]) + list(targets[i + 1:]), blank)
        deletion = (deleted - base) / count
        result.append(Evidence(
            float(np.mean(frames[:, target] - frames[:, phone_ids].max(axis=1))),
            float(np.exp(frames[:, target]).mean()), float(best[0]), float(deletion),
            count * frame_ms, best[1], float(margin), 2 <= count <= 40,
        ))
    return result
