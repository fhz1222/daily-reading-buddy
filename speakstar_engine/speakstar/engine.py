import time

from .align import AlignmentError, forced_align
from .audio import SAMPLE_RATE, load_audio
from .feedback import assess_phone
from .g2p import phonemize_words
from .gop import extract_evidence
from .model import AcousticModel, PIPELINE_VERSION
from .output import aggregate
from .score import Calibrator


class PronunciationEngine:
    def __init__(self, model_dir, locale="en-gb", precision="fp32", calibration=None, threads=2):
        if locale not in {"en-gb", "en-us"}:
            raise ValueError("Supported locales: en-gb, en-us")
        self.model = AcousticModel(model_dir, precision, threads)
        self.locale = locale
        self.calibrator = Calibrator.load(calibration, self.model.fingerprint, locale) if calibration else None

    def assess(self, audio_path, reference_text, include_evidence=False):
        start = time.perf_counter()
        words = phonemize_words(reference_text, self.model.vocab, self.locale)
        g2p_end = time.perf_counter()
        base = {
            "status": "Success", "reference_text": reference_text, "locale": self.locale,
            "engine": {"model": self.model.meta["model_id"], "precision": self.model.precision,
                       "model_sha256": self.model.fingerprint, "pipeline": PIPELINE_VERSION,
                       "calibration": self.calibrator.data["id"] if self.calibrator else None,
                       "validated": False},
            "Words": [], "PronunciationAssessment": {"AccuracyScore": None, "CompletenessScore": None},
        }
        audio = load_audio(audio_path)
        audio_end = time.perf_counter()
        if audio is None:
            base["status"] = "NoSpeech"
            return base
        lp = self.model.infer(audio.samples)
        inference_end = time.perf_counter()
        targets = [self.model.vocab[p] for word in words for p in word.phones]
        try:
            spans, _ = forced_align(lp, targets, self.model.blank)
        except AlignmentError:
            base["status"] = "UncertainAlignment"
            return base
        evidence = extract_evidence(lp, targets, spans, self.model.vocab, self.model.blank,
                                    self.model.stride / SAMPLE_RATE * 1000)
        evidence_end = time.perf_counter()
        def ticks(frame):
            return round((audio.offset_samples + frame * self.model.stride) / SAMPLE_RATE * 10_000_000)
        index = 0
        for wi, word in enumerate(words):
            phones = []
            first = index
            for phone in word.phones:
                span, ev = spans[index], evidence[index]
                out = {"Phoneme": phone, "Offset": ticks(span.start),
                       "Duration": ticks(span.end) - ticks(span.start),
                       "PronunciationAssessment": assess_phone(phone, ev, self.calibrator)}
                if include_evidence:
                    out["evidence"] = ev.to_dict()
                phones.append(out)
                index += 1
            base["Words"].append({
                "Word": word.text, "word_index": wi, "Offset": ticks(spans[first].start),
                "Duration": ticks(spans[index - 1].end) - ticks(spans[first].start),
                "reference_stressed_ipa": word.stressed_ipa,
                "PronunciationAssessment": aggregate([p["PronunciationAssessment"] for p in phones]),
                "Phonemes": phones,
            })
        summary = aggregate([w["PronunciationAssessment"] for w in base["Words"]])
        # Unknown words cannot be silently counted as complete.
        if all(w["PronunciationAssessment"]["status"] != "uncertain" for w in base["Words"]):
            summary["CompletenessScore"] = round(100 * sum(w["PronunciationAssessment"]["status"] != "possibly_missing" for w in base["Words"]) / len(words), 2)
        else:
            summary["CompletenessScore"] = None
        base["PronunciationAssessment"] = summary
        base["timing"] = {"processing_seconds": round(time.perf_counter() - start, 4),
                          "g2p_seconds": round(g2p_end - start, 4),
                          "audio_seconds": round(audio_end - g2p_end, 4),
                          "inference_seconds": round(inference_end - audio_end, 4),
                          "alignment_evidence_seconds": round(evidence_end - inference_end, 4)}
        return base
