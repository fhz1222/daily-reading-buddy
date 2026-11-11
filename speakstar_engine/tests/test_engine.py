import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speakstar.align import AlignmentError, ctc_log_likelihood, forced_align
from speakstar.audio import preprocess
from speakstar.engine import PronunciationEngine
from speakstar.evaluation import metrics, validate_rows
from speakstar.feedback import assess_phone
from speakstar.g2p import WordPhones, phonemize_words, tokenize_ipa
from speakstar.gop import Evidence, FEATURE_NAMES, extract_evidence
from speakstar.model import PIPELINE_VERSION
from speakstar.score import Calibrator


def logsoftmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def calibration():
    return {"id": "test-only", "pipeline": PIPELINE_VERSION, "model_sha256": "test-model", "locale": "en-gb",
            "feature_names": FEATURE_NAMES, "mean": [0] * 5, "scale": [1] * 5,
            "weights": [1, 0, 0, 0, 0], "bias": 0,
            "policy": {"bad_threshold": 0.2, "good_threshold": 0.8, "dev_ece": 0.01,
                       "deletion_threshold": 2, "substitution_margin": 1,
                       "phone_support": {"θ": {"correct": 30, "incorrect": 20}}}}


class AlignmentTests(unittest.TestCase):
    def test_forward_matches_exhaustive_paths(self):
        lp = logsoftmax(np.random.default_rng(2).normal(size=(5, 3)))
        for target in [[], [1], [1, 1], [1, 2], [1, 2, 1]]:
            accepted = []
            best = -np.inf
            for path in itertools.product(range(3), repeat=5):
                collapsed = [p for i, p in enumerate(path) if p and (i == 0 or p != path[i - 1])]
                if collapsed == target:
                    likelihood = sum(lp[i, p] for i, p in enumerate(path))
                    accepted.append(likelihood)
                    best = max(best, likelihood)
            self.assertAlmostEqual(ctc_log_likelihood(lp, target), float(np.logaddexp.reduce(accepted)))
            self.assertAlmostEqual(forced_align(lp, target)[1], best)

    def test_matches_torchaudio(self):
        try:
            import torch
            import torchaudio.functional as functional
        except ImportError:
            self.skipTest("Install optional test dependencies for torchaudio reference comparison")
        for targets in [[1, 2], [1, 1, 2], [2], [2, 1, 2]]:
            for seed in range(5):
                lp = logsoftmax(np.random.default_rng(seed).normal(size=(20, 3))).astype(np.float32)
                path, scores = functional.forced_align(torch.from_numpy(lp)[None], torch.tensor([targets]), blank=0)
                spans, likelihood = forced_align(lp, targets)
                self.assertAlmostEqual(likelihood, float(scores.sum()), places=4)
                emitted = path[0].numpy()
                collapsed = [x for i, x in enumerate(emitted) if x and (i == 0 or x != emitted[i - 1])]
                self.assertEqual(collapsed, targets)
                for span, target in zip(spans, targets, strict=True):
                    self.assertTrue(np.all(emitted[span.start:span.end] == target))

    def test_repeated_phones_need_blank_frame(self):
        with self.assertRaises(AlignmentError):
            forced_align(np.log(np.full((2, 3), 1 / 3)), [1, 1])
        spans, _ = forced_align(np.log(np.array([[.01, .98, .01], [.98, .01, .01], [.01, .98, .01]])), [1, 1])
        self.assertEqual([(s.start, s.end) for s in spans], [(0, 1), (2, 3)])


class EvidenceAndPolicyTests(unittest.TestCase):
    def test_wrong_phone_has_substitution_evidence(self):
        lp = np.log(np.array([[.98, .01, .01]] * 2 + [[.01, .04, .95]] * 6 + [[.98, .01, .01]] * 2))
        spans, _ = forced_align(lp, [1])
        ev = extract_evidence(lp, [1], spans, {"<pad>": 0, "θ": 1, "s": 2}, confusions={"θ": ["s"]})[0]
        self.assertEqual(ev.alternative, "s")
        self.assertGreater(ev.substitution_lpr, 0)
        self.assertLess(ev.gop, 0)

    def test_no_calibration_never_invents_scores_or_errors(self):
        ev = Evidence(-10, .01, 5, 6, 60, "s", 4, True)
        self.assertEqual(assess_phone("θ", ev), {"AccuracyScore": None, "confidence": "low", "status": "uncertain"})

    def test_policy_gates_and_missing(self):
        c = Calibrator(calibration(), "test-model", "en-gb")
        ev = Evidence(-10, .01, 5, 0, 60, "s", 4, True)
        a = assess_phone("θ", ev, c)
        self.assertEqual(a["possible_substitute"], "s")
        self.assertEqual(a["status"], "needs_practice")
        ev.alignment_sane = False
        self.assertEqual(assess_phone("θ", ev, c)["status"], "uncertain")
        ev.alignment_sane = True
        ev.deletion_lpr = 3
        self.assertEqual(assess_phone("θ", ev, c)["status"], "possibly_missing")
        self.assertEqual(assess_phone("s", ev, c)["status"], "uncertain")

    def test_calibration_cannot_cross_model_or_accent(self):
        for fingerprint, locale in [("other", "en-gb"), ("test-model", "en-us")]:
            with self.assertRaises(ValueError):
                Calibrator(calibration(), fingerprint, locale)

    def test_uncertain_counts_as_missed_error(self):
        ev = Evidence(0, .5, 0, 0, 20, None, 0, False)
        row = {"phone": "θ", "correct": 0, "evidence": ev.to_dict()}
        m = metrics([row], Calibrator(calibration(), "test-model", "en-gb"))
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["uncertain_rate"], 1)

    def test_speaker_leakage_rejected(self):
        row = {"model_sha256": "x", "locale": "en-gb", "pipeline": PIPELINE_VERSION,
               "correct": 1, "speaker": "same", "split": "train", "utterance_id": "1",
               "word_index": 0, "phone_index": 0, "evidence": Evidence(0, .5, 0, 0, 20, None, 0, True).to_dict()}
        with self.assertRaises(ValueError):
            validate_rows([row, dict(row, split="test", utterance_id="2")])


class AudioAndOutputTests(unittest.TestCase):
    def test_no_speech_and_trim_offset(self):
        self.assertIsNone(preprocess(np.zeros(16000, dtype=np.float32)))
        t = np.arange(8000) / 16000
        x = np.concatenate([np.zeros(8000), .1 * np.sin(2 * np.pi * 200 * t), np.zeros(8000)]).astype(np.float32)
        audio = preprocess(x)
        self.assertGreater(audio.offset_samples, 0)
        self.assertLess(audio.offset_samples, 8000)
        self.assertAlmostEqual(float(audio.samples.mean()), 0, places=5)
        with self.assertRaises(ValueError):
            preprocess(np.zeros(13 * 16000))

    def test_end_to_end_schema_and_original_offsets(self):
        class Model:
            vocab = {"<pad>": 0, "θ": 1, "s": 2}
            blank, stride, precision, fingerprint = 0, 320, "fp32", "test"
            meta = {"model_id": "fixture"}
            def infer(self, samples):
                return np.log(np.array([[.98, .01, .01]] * 2 + [[.01, .98, .01]] * 6 + [[.98, .01, .01]] * 2))
        engine = PronunciationEngine.__new__(PronunciationEngine)
        engine.model, engine.calibrator, engine.locale = Model(), None, "en-gb"
        audio = preprocess(np.concatenate([np.zeros(8000), np.sin(np.arange(8000) * .1) * .1]).astype(np.float32))
        with patch("speakstar.engine.load_audio", return_value=audio), patch("speakstar.engine.phonemize_words", return_value=[WordPhones("the", ["θ"], "θ")]):
            result = engine.assess("fixture.wav", "the", True)
        phone = result["Words"][0]["Phonemes"][0]
        self.assertGreater(phone["Offset"], 4_000_000)
        self.assertEqual(phone["PronunciationAssessment"]["status"], "uncertain")
        self.assertIsNone(result["PronunciationAssessment"]["CompletenessScore"])
        self.assertFalse(result["engine"]["validated"])
        text = json.dumps(result, allow_nan=False)
        for forbidden in ["DisplayText", "FluencyScore", "ProsodyScore", '"heard"']:
            self.assertNotIn(forbidden, text)


class G2PTests(unittest.TestCase):
    def test_tokens_preserve_explicit_boundaries_and_stress(self):
        vocab = {"ə": 0, "l": 1, "əl": 2, "aɪ": 3}
        self.assertEqual(tokenize_ipa("ˈaɪ ə l", vocab), ["aɪ", "ə", "l"])
        self.assertEqual(tokenize_ipa("əl", vocab), ["əl"])
        with self.assertRaises(ValueError):
            tokenize_ipa("θ", vocab)

    def test_real_british_g2p(self):
        vocab = {s: i for i, s in enumerate(["θ", "ɹ", "iː", "ɪ", "n", "ʌ", "m", "z"])}
        words = phonemize_words("Three thin thumbs.", vocab)
        self.assertEqual(words[0].phones, ["θ", "ɹ", "iː"])
        self.assertIn("ˈ", words[0].stressed_ipa)
        self.assertEqual(len(words), 3)
        with self.assertRaises(ValueError):
            phonemize_words("123", vocab)


if __name__ == "__main__":
    unittest.main()
