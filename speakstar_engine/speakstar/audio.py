from dataclasses import dataclass
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 12


@dataclass
class Audio:
    samples: np.ndarray
    offset_samples: int
    original_samples: int
    speech_fraction: float


def ffmpeg_path():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def preprocess(samples, sample_rate=SAMPLE_RATE):
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Expected 16 kHz audio")
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite mono samples")
    if len(x) > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError(f"Audio exceeds {MAX_SECONDS} seconds")
    total = len(x)
    if total < 4800:
        return None
    # A conservative energy gate, not a speech/noise classifier.
    frame = 320
    rms = np.sqrt(np.mean(x[:total // frame * frame].reshape(-1, frame) ** 2, axis=1))
    active = np.flatnonzero(rms >= max(0.003, float(rms.max()) * 0.08))
    fraction = len(active) / len(rms)
    if len(active) < 5 or fraction < 0.08:
        return None
    start = max(0, int(active[0]) * frame - 800)
    end = min(total, (int(active[-1]) + 1) * frame + 800)
    clipped = x[start:end].copy()
    if len(clipped) < 4800:
        return None
    # Match the HF feature extractor's zero-mean/unit-variance normalization.
    clipped = (clipped - clipped.mean()) / np.sqrt(clipped.var() + 1e-7)
    return Audio(clipped.astype(np.float32), start, total, fraction)


def load_audio(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Audio file does not exist")
    result = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-i", str(path.resolve()),
         "-t", str(MAX_SECONDS + 0.1), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=20, check=False,
    )
    if result.returncode:
        raise ValueError("Cannot decode audio; use WAV, WebM, MP3, Ogg or MP4")
    return preprocess(np.frombuffer(result.stdout, dtype="<f4"))
