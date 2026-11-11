"""CTC Viterbi alignment and forward sequence likelihood, including blank states."""
from dataclasses import dataclass

import numpy as np


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class Span:
    start: int
    end: int  # exclusive; contains target-emitting frames, not surrounding blanks


def _states(log_probs, targets, blank):
    lp = np.asarray(log_probs, dtype=np.float64)
    ids = np.asarray(targets, dtype=np.int64)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all():
        raise AlignmentError("Expected finite [frames, vocabulary] log probabilities")
    if not 0 <= blank < lp.shape[1] or np.any(ids < 0) or np.any(ids >= lp.shape[1]) or blank in ids:
        raise AlignmentError("Invalid CTC token IDs")
    if len(lp) < len(ids) + np.sum(ids[1:] == ids[:-1]):
        raise AlignmentError("Audio has too few frames for the reference")
    states = np.full(2 * len(ids) + 1, blank, dtype=np.int64)
    states[1::2] = ids
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    return lp, states, skip


def _advance(previous, skip):
    one = np.concatenate(([-np.inf], previous[:-1]))
    two = np.full_like(previous, -np.inf)
    two[2:] = previous[:-2]
    two[~skip] = -np.inf
    return np.stack((previous, one, two))


def forced_align(log_probs, targets, blank=0):
    lp, states, skip = _states(log_probs, targets, blank)
    if not len(targets):
        return [], float(lp[:, blank].sum())
    previous = np.full(len(states), -np.inf)
    previous[:2] = lp[0, states[:2]]
    back = np.zeros((len(lp), len(states)), dtype=np.int8)
    for t in range(1, len(lp)):
        paths = _advance(previous, skip)
        back[t] = paths.argmax(axis=0)
        previous = paths.max(axis=0) + lp[t, states]
    final = len(states) - (1 if previous[-1] > previous[-2] else 2)
    score = float(previous[final])
    if not np.isfinite(score):
        raise AlignmentError("No valid CTC alignment")
    path = np.empty(len(lp), dtype=np.int64)
    for t in range(len(lp) - 1, -1, -1):
        path[t] = final
        final -= int(back[t, final])
    spans = []
    for i in range(len(targets)):
        frames = np.flatnonzero(path == 2 * i + 1)
        if not len(frames):
            raise AlignmentError("Reference token was not visited")
        spans.append(Span(int(frames[0]), int(frames[-1]) + 1))
    return spans, score


def ctc_log_likelihood(log_probs, targets, blank=0):
    """Sum over all valid paths, rather than comparing one best alignment."""
    try:
        lp, states, skip = _states(log_probs, targets, blank)
    except AlignmentError:
        return -np.inf
    previous = np.full(len(states), -np.inf)
    previous[:min(2, len(states))] = lp[0, states[:min(2, len(states))]]
    for t in range(1, len(lp)):
        previous = np.logaddexp.reduce(_advance(previous, skip), axis=0) + lp[t, states]
    return float(np.logaddexp.reduce(previous[-2:] if len(targets) else previous))
