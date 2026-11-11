# Speak Star Local Pronunciation Engine — Plan (v2)

`Document version: local-engine-v2 · 2026-11-12` — supersedes the first version of this file. The hybrid plan (`hybrid-plan-route-b.md`) references this version by section **title**.

A free, offline **phoneme-level** pronunciation assessment prototype for children learning English.

> **Scope of v1:** phoneme accuracy and omissions only.
> **Not in v1:** word stress, intonation, rhythm, and word transcription of what the child said. These are separate later phases (Section 9) and are not approximated by any v1 score.

---

## Contents

0. [Changes from v1 of this plan](#0-changes-from-v1-of-this-plan)
1. [Goal and scope](#1-goal-and-scope)
2. [Cost and licensing](#2-cost-and-licensing)
3. [Architecture](#3-architecture)
4. [Components](#4-components)
5. [Feedback policy: what the engine is allowed to say](#5-feedback-policy-what-the-engine-is-allowed-to-say)
6. [Output format](#6-output-format)
7. [Integration with the existing app](#7-integration-with-the-existing-app)
8. [Evaluation and calibration](#8-evaluation-and-calibration)
9. [Later phases: stress, intonation, word transcription](#9-later-phases-stress-intonation-word-transcription)
10. [Project layout](#10-project-layout)
11. [Milestones and exit criteria](#11-milestones-and-exit-criteria)
12. [Testing](#12-testing)
13. [Risks and mitigations](#13-risks-and-mitigations)
14. [Open decisions](#14-open-decisions)
15. [References](#15-references)

---

## 0. Changes from v1 of this plan

| # | Review point | Change |
|---|---|---|
| 1 | No stress or intonation assessment | v1 is explicitly phoneme-only. Stress (Phase B) and intonation (Phase C) are separate phases with their own data, features, and evaluation. No placeholder `ProsodyScore` field. `FluencyScore` is removed from v1 because pause/rate heuristics are not prosody assessment |
| 2 | "What the model thinks it heard" ≠ "what the child said" | Greedy decoding is no longer used for diagnosis. Substitution hints come from alignment-based evidence and are **confidence-gated**: no claim at low confidence, "may be closer to /s/" at high confidence, never "you said /s/" |
| 3 | Calibration not rigorous enough | Speaker-disjoint train/dev/test splits, child subgroup reported separately, **false-rejection rate** (correct sounds marked wrong) as the primary metric. Parent ✓/✗ recordings are for initial threshold tuning only, not evidence of reliability. Unsupported claims removed |
| 4 | Phonemes don't give a reliable word transcript | No `DisplayText` and no word-level `Insertion` in v1. The target sentence is never presented as "what the system heard". Word transcription becomes an optional Phase D with its own decoder |
| 5 | "Change one function" doesn't match the repo | Integration section rewritten for the existing `server.py` + `static/app.js` (`/api/recognize` → `{heard: ...}`, port 8765). Recommended: add a new endpoint to the existing server, not a second server |
| 6 | Performance and completion over-stated | Size, speed, and effort are now **targets to measure**. fp32 vs int8 compared on accuracy, not just speed. Milestone 4 passes only if pre-agreed error thresholds are met |

---

## 1. Goal and scope

| | |
|---|---|
| **Input** | Audio clip + target text + locale (`en-us` / `en-gb`) |
| **v1 output** | Per-phoneme score and confidence, per-word score, omission detection, confidence-gated substitution hints, sentence-level accuracy and completeness |
| **Not v1 output** | Stress, intonation, rhythm, fluency, transcript of what was said |
| **Runs on** | Laptop CPU, fully offline (speed to be measured, Section 11) |
| **Runtime dependencies** | `onnxruntime`, `numpy`, `phonemizer` + `espeak-ng`, `ffmpeg` |

**Positioning:** a prototype that gives *practice hints*, not a validated assessment. Whether it is good enough for a given child is decided by the evaluation in Section 8, not assumed.

---

## 2. Cost and licensing

**No usage fees.** Running costs are your own hardware and electricity.

| Item | Cost | License note |
|---|---|---|
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | Free | Check the model card before any commercial use |
| espeak-ng, phonemizer | Free | GPL: fine for home use; matters only if you distribute the app |
| ONNX Runtime, numpy, ffmpeg | Free | Permissive / LGPL |
| speechocean762 | Free | Commercial and non-commercial use allowed |

**Non-monetary costs:** development time (not estimated until milestone 1 is done), a one-time model download, and accuracy that is unknown until measured. Recordings stay on your computer.

---

## 3. Architecture

```
audio ──► preprocess ──► wav2vec2 (ONNX) ──► phoneme log-probs (1 frame = 20 ms)
                                                   │
text ──► espeak G2P ──► expected phonemes ─────────┤
                                                   ▼
                    forced alignment (CTC Viterbi) ──► span per expected phoneme
                                                   ▼
                    GOP features + substitution / deletion evidence per phoneme
                                                   ▼
                    calibrator ──► P(correct), confidence
                                                   ▼
                    feedback policy (Section 5) ──► JSON
```

There is no free-decoding diff step. All diagnosis is relative to the expected phonemes, using the alignment and the model's probabilities.

---

## 4. Components

### 4.1 Preprocessing — `audio.py`
- Decode with ffmpeg → 16 kHz mono float32.
- Energy-based voice activity detection to trim silence; loudness normalization.
- Clips that are too short or mostly silent return `NoSpeech`, not a score.

### 4.2 G2P — `g2p.py`
- `phonemizer` + espeak-ng, word boundaries kept so each phoneme maps to its word.
- Longest-match tokenization into the model's vocabulary; unit test that every expected token exists in it.
- **Stress marks are stripped for the acoustic model but stored separately** per syllable, so Phase B can use them without redoing G2P.
- `g2p_exceptions.json` for words espeak gets wrong.

### 4.3 Acoustic model — `model.py` + one-time `export_onnx.py`
- Export to ONNX. Produce **both fp32 and int8** versions; int8 is used only if it passes the same evaluation (Section 8.4).
- Behind a small interface so another phone recognizer (e.g. ZIPA) can be compared later.

### 4.4 Forced alignment — `align.py`
- CTC Viterbi in numpy; tested against `torchaudio.functional.forced_align`.
- Alignment quality is itself a risk: phonemes with abnormal spans (too short / too long relative to expected duration) are marked **low confidence** rather than scored normally.

### 4.5 Evidence per expected phoneme — `gop.py`

| Feature | Purpose |
|---|---|
| GOP-avg (`log P(target) − max log P`) over aligned frames | Is the expected sound well supported? |
| Posterior mean of target | Same, as a probability |
| Substitution LPR against `confusions.json` (5–10 similar sounds) | Is a specific alternative clearly better supported? |
| Deletion LPR | Is the sound missing? |
| Span duration vs expected | Alignment sanity check |

### 4.6 Calibration — `score.py`
- Logistic regression on the features → **P(correct)** per phoneme.
- **Confidence** is a separate value, derived from alignment sanity, how far P(correct) is from the decision threshold, and calibration error measured on the dev set.
- Word score: mean phoneme P(correct) × 100; word marked as having an issue only if at least one phoneme is confidently wrong.
- Sentence `AccuracyScore` = mean of word scores; `CompletenessScore` = share of words without confident omission.

---

## 5. Feedback policy: what the engine is allowed to say

The engine never states what the child "said". It reports evidence, gated by confidence.

| Situation | Engine output | Child-facing message (example) |
|---|---|---|
| P(correct) high | `status: ok` | ✓ green |
| Low P(correct), low confidence | `status: uncertain` | No judgement; at most "let's try once more" |
| Low P(correct), high confidence, no clear alternative | `status: needs_practice` | "Listen to **th** again and copy it" |
| Low P(correct), high confidence, one alternative clearly better (substitution LPR margin above threshold) | `status: needs_practice`, `possible_substitute: "s"` | "This sound **may be closer to s**. Listen to **th** again" |
| Deletion LPR high, high confidence | `status: possibly_missing` | "Did you say every sound? Try again slowly" |

- All thresholds come from the dev set (Section 8), tuned to keep the **false-rejection rate** low.
- Low-confidence results are shown as "uncertain", never silently counted as errors.

---

## 6. Output format

**Azure-like, not Azure-identical.** It borrows Azure's nesting so it's familiar, but fields the engine can't produce reliably are left out rather than faked.

```json
{
  "status": "Success",
  "reference_text": "three thin thumbs",
  "locale": "en-gb",
  "engine": { "model": "wav2vec2-xlsr-53-espeak-cv-ft", "precision": "fp32", "calibration": "so762-v1" },
  "PronunciationAssessment": { "AccuracyScore": 78, "CompletenessScore": 100 },
  "Words": [{
    "Word": "three", "Offset": 3200000, "Duration": 4600000,
    "PronunciationAssessment": { "AccuracyScore": 52, "status": "needs_practice" },
    "Phonemes": [{
      "Phoneme": "θ", "Offset": 3200000, "Duration": 1200000,
      "PronunciationAssessment": {
        "AccuracyScore": 21,
        "confidence": "high",
        "status": "needs_practice",
        "possible_substitute": "s"
      }
    }]
  }]
}
```

| Deliberately absent in v1 | Why |
|---|---|
| `DisplayText` / "heard" text | Phonemes can't be reliably turned into words without a word decoder (Phase D). The reference text is labelled `reference_text`, never presented as recognized speech |
| Word-level `Insertion` | Needs word boundaries in the recognized speech, which v1 doesn't have |
| `FluencyScore`, `ProsodyScore` | No prosody model in v1 (Phases B/C) |
| `NBestPhonemes` | Replaced by the confidence-gated `possible_substitute` |

Offsets/durations are in 100-ns ticks (from 20 ms frames, so the precision is 20 ms).

---

## 7. Integration with the existing app

**Current state (as described; to be confirmed against the actual code):**
- `server.py` serves the app on port **8765** and exposes `POST /api/recognize`, returning `{heard: ...}`.
- `static/app.js` calls `/api/recognize` and displays `heard`.
- There is no Azure-style `showResult()` in this repo.

**Options**

| | A. Extend existing server (recommended) | B. Separate engine service |
|---|---|---|
| How | Add `POST /api/assess` to `server.py`; engine imported as a library | New process on a **different** port (e.g. 8766); `server.py` proxies or the page calls it directly |
| Port conflict | None | Must avoid 8765 |
| `/api/recognize` | Unchanged; keeps working | Unchanged |
| Pros | One process, one origin, no CORS | Engine can restart independently; model memory isolated |
| Cons | Model load time adds to server startup | Two processes to run; CORS or proxy needed |

**Frontend work (either option), not a one-line change:**
1. Send the target text with the audio to `/api/assess`.
2. New rendering for per-word / per-phoneme `status`, including the "uncertain" state.
3. Keep showing `heard` from `/api/recognize` only if that endpoint genuinely recognizes speech; never replace it with the reference text.
4. Child-facing messages per Section 5.

**Needed before implementation:** the current `server.py` and `static/app.js`, to confirm the framework, the request format (audio encoding), and how results are displayed.

---

## 8. Evaluation and calibration

### 8.1 Data and splits
- **speechocean762**: 5,000 utterances, 250 speakers (125 children, 125 adults), all Mandarin first language, expert labels at phoneme (0/1/2), word, and sentence level. Includes canonical vs pronounced phone for mispronunciations.
- Official train (125 speakers) / test (125 speakers) split is speaker-disjoint. The train split is further divided **by speaker** into calibration-train and dev.
- The **test split is touched once**, after thresholds and exit criteria are fixed.

### 8.2 Label mapping
- Phoneme label 2 = correct, 0 = incorrect/missed. Label 1 (accented but right) is **treated as correct** for the primary metric, since the goal is not to penalize accent; reported separately.
- ARPAbet ↔ model IPA mapping table, unit-tested.

### 8.3 Metrics (reported for all speakers **and for the child subgroup separately**)

| Metric | Why it matters |
|---|---|
| **False-rejection rate** (correct phonemes marked `needs_practice` / `possibly_missing`) | Primary: wrongly telling a child she's wrong discourages her |
| Mispronunciation detection precision / recall / F1 | Standard comparison with published work |
| Substitution-hint accuracy (when `possible_substitute` is given, is it the annotated pronounced phone?) | Validates the Section 5 wording |
| Share of phonemes marked `uncertain` | Coverage: too many means the tool says little |
| Phoneme / word score correlation with human scores | Comparison with published baselines |
| Calibration error (ECE) of P(correct) | Whether confidence levels mean anything |

### 8.4 fp32 vs int8
Both run through the full metric set. int8 is adopted only if its child false-rejection rate and F1 are within an agreed tolerance of fp32.

### 8.5 Personal recordings
- 20–30 parent-labelled (✓/✗ per word) recordings may be used to **adjust thresholds** for one child.
- They are **not** used to claim reliability: too few, word-level only, and not expert-labelled.
- Generalization beyond Mandarin-L1 speakers (e.g. Singapore English) is **unverified** until tested on matching data.

---

## 9. Later phases: stress, intonation, word transcription

Each is a separate project with its own data and evaluation. None starts until v1 passes milestone 4.

### Phase B — Word stress
- **Units:** syllables (requires syllabification and syllable-level alignment from phoneme spans).
- **Features per syllable:** F0 (pYIN), energy, duration, vowel quality relative to unstressed reduced forms.
- **Reference:** espeak primary/secondary stress marks saved in Section 4.2.
- **Data:** speechocean762 word-level stress scores as a starting point; check whether they're informative enough before committing.
- **Output:** per multi-syllable word, "stress on the expected syllable: likely / unlikely / uncertain".

### Phase C — Intonation and rhythm
- **Features:** F0 contour (semitones, speaker-normalized), energy, syllable timing.
- **Approach to evaluate:** compare against reference contours for the same sentence (multiple native or TTS readings) for coarse patterns, e.g. question rise vs statement fall.
- **Data:** needs labelled intonation data; not covered by speechocean762's sentence-level prosodic score alone. Data source to be decided.
- **Output:** coarse, pattern-level feedback only.

### Phase D — Word transcription (optional)
- A separate word-level ASR (e.g. a small offline Whisper or Vosk model) to produce real `heard` text and word-level insertions.
- Must be evaluated on child speech before being shown, because ASR language models tend to "correct" learners toward real words.

---

## 10. Project layout

```
speakstar_engine/
├── pyproject.toml
├── speakstar/
│   ├── audio.py
│   ├── g2p.py
│   ├── model.py
│   ├── align.py
│   ├── gop.py
│   ├── score.py
│   ├── feedback.py         # Section 5 policy
│   ├── output.py           # Section 6 JSON
│   └── cli.py
├── scripts/
│   ├── export_onnx.py      # one-time, needs torch; writes fp32 + int8
│   ├── split_so762.py      # speaker-disjoint calib-train / dev
│   ├── calibrate.py
│   └── evaluate.py         # Section 8.3 metrics, child subgroup separate
├── data/
│   ├── confusions.json
│   ├── arpabet_to_ipa.json
│   └── g2p_exceptions.json
└── tests/
```
Integration code goes into the existing `server.py` / `static/app.js` (Section 7), not into this package.

---

## 11. Milestones and exit criteria

All sizes, speeds, and durations below are **targets to be measured**, not commitments.

| # | Deliverable | Exit criteria |
|---|---|---|
| 1 | ONNX export (fp32 + int8), log-probs | Measured model size, load time, and processing time for 3–5 s clips on your machine. Target: under ~1 s per clip on CPU; if missed, decide on a smaller model before continuing |
| 2 | G2P, alignment, evidence features, CLI | G2P/vocabulary test passes; alignment matches torchaudio; TTS "perfect" clips have no confident errors; deliberately wrong sounds are flagged on a small hand-made test set |
| 3 | Calibration, confidence, feedback policy, JSON | Trained on calibration-train; thresholds set on dev; schema test passes |
| 4 | **Evaluation on held-out test** | Passes the **pre-agreed** thresholds in Section 14 (decision 4) for the child subgroup, for both the chosen precision (fp32 or int8). If it fails, report and revise; do not ship scores as reliable |
| 5 | Integration into `server.py` + `static/app.js` | End to end in the browser with the Section 5 messages; `/api/recognize` unaffected |
| 6 | Personal threshold tuning (optional) | Thresholds adjusted; documented as personal tuning, not validation |

---

## 12. Testing

| Test | Checks |
|---|---|
| G2P | Every expected token is in the model vocabulary; exceptions applied |
| Label mapping | ARPAbet ↔ IPA round-trips on all speechocean762 phones |
| Alignment | Matches `torchaudio.functional.forced_align` |
| Synthetic audio | TTS clips of target sentences produce no high-confidence errors |
| Deliberate errors | Swapped sounds ("sink" for "think") flagged; the hint says "may be closer to", never "you said" |
| No fabricated transcript | Output never contains the reference text in a "heard"/`DisplayText` field |
| Schema | Output matches Section 6 |
| Regression | Evaluation metrics stored per version; new versions can't silently get worse |

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Model misrecognition presented as child error | No free-decoding diff; confidence-gated hints; "uncertain" state; substitution-hint accuracy measured |
| Alignment errors | Duration sanity checks → low confidence; alignment tested |
| Overconfident model probabilities | Separate calibration; ECE reported; thresholds set on dev, not test |
| Child speech, other first languages, Singapore English | Child subgroup reported separately; generalization to other speakers marked unverified |
| False rejections discourage the child | False-rejection rate is the primary metric; label 1 (accent) counted as correct |
| int8 hurts accuracy | Full evaluation of fp32 vs int8 before choosing |
| Scope creep into prosody | Stress/intonation are separate phases gated on v1 passing |
| GPL components | Fine for home use; revisit only if distributing |

---

## 14. Open decisions

1. **Accent target (confirmed):** default `en-gb`; `en-us` available explicitly, with its own calibration artifact.
2. **Machine:** development verified on Windows / Python 3.12 / CPU. No GPU required for the current runtime.
3. **Integration:** engine-first, per user instruction. Existing `server.py` and `static/app.js` inspected. Extend the existing server after milestone 4; the current Whisper page remains available during engine development.
4. **Milestone 4 thresholds** (to be fixed **before** running the test set). Proposed starting points for discussion, child subgroup:
   - False-rejection rate ≤ 10% of correct phonemes
   - Substitution hint correct in ≥ 70% of cases where one is shown
   - ≤ 25% of phonemes marked `uncertain`
   - Mispronunciation detection F1 reported and compared against published GOP baselines on speechocean762 (no fixed target until the baseline is reproduced)
5. **Phases B–D:** which, if any, to pursue after v1, and where labelled data for intonation would come from.

---

## 15. References

- Model card: wav2vec2-xlsr-53-espeak-cv-ft (phoneme output; mapping to words needs a lexicon) — https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft
- Evaluating Logit-Based GOP Scores for Mispronunciation Detection (alignment and overconfidence issues) — https://arxiv.org/abs/2506.12067
- Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge — https://arxiv.org/pdf/2506.02080
- Segmentation-free Goodness of Pronunciation — https://fugumt.com/fugumt/paper_check/2507.16838v2_enmode
- ZIPA: efficient multilingual phone recognition — https://arxiv.org/html/2505.23170v1
- speechocean762 dataset — https://www.openslr.org/101
- OpenPronounce (reference implementation) — https://github.com/Halleck45/OpenPronounce
