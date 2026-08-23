# ✦ @chiranr19 · © 2026 All Rights Reserved · proprietary · sigil:AQY3QG3XBBYYR2WN
"""The scoring model: five interpretable signals combined into one score.

Each candidate window is measured on five signals, every one normalized to
``[0, 1]`` so the final score and its breakdown stay human-readable:

* **repetition**  — does this section recur elsewhere? (chorus/hook-ness)
* **harmonic**    — is the harmony stable and coherent, not transitional?
* **rhythmic**    — is the groove present and driving?
* **energy**      — is it sustained and loud, not a quiet passage?
* **centrality**  — structural prior that hooks rarely sit in the intro/outro.

Weights are configurable; the defaults lead with repetition because the single
most reliable marker of "the part you remember" is that the song returns to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGNALS = ("repetition", "harmonic", "rhythmic", "energy", "centrality")


@dataclass(frozen=True)
class Weights:
    """Relative importance of each signal. Values are re-normalized to sum to 1."""

    repetition: float = 0.35
    harmonic: float = 0.20
    rhythmic: float = 0.20
    energy: float = 0.15
    centrality: float = 0.10

    def as_array(self) -> np.ndarray:
        """Return weights in ``SIGNALS`` order, normalized to sum to 1."""
        vals = np.array([getattr(self, name) for name in SIGNALS], dtype=float)
        vals = np.clip(vals, 0.0, None)
        total = vals.sum()
        if total <= 0:
            return np.full(len(SIGNALS), 1.0 / len(SIGNALS))
        return vals / total

    def as_dict(self) -> dict:
        arr = self.as_array()
        return {name: float(arr[i]) for i, name in enumerate(SIGNALS)}


DEFAULT_WEIGHTS = Weights()


# --- per-window signal measurements -----------------------------------------


def harmonic_stability(chroma_window: np.ndarray) -> float:
    """Median cosine similarity between consecutive chroma frames.

    High when the harmony holds together (a settled section), low across
    transitions and rapid chord churn.
    """
    if chroma_window.shape[1] < 3:
        return 0.0
    norms = np.linalg.norm(chroma_window, axis=0, keepdims=True) + 1e-8
    unit = chroma_window / norms
    sims = np.sum(unit[:, 1:] * unit[:, :-1], axis=0)
    return float(np.median(np.clip(sims, 0.0, 1.0)))


def rhythmic_intensity(onset_window: np.ndarray) -> float:
    """Mean percussive onset strength across the window."""
    return float(np.mean(onset_window)) if onset_window.size else 0.0


def sustained_energy(rms_window: np.ndarray, threshold: float) -> float:
    """Fraction of the window whose loudness sits above ``threshold``."""
    if rms_window.size == 0:
        return 0.0
    return float(np.mean(rms_window > threshold))


def repetition(recurrence_window: np.ndarray) -> float:
    """Mean per-frame recurrence strength across the window."""
    return float(np.mean(recurrence_window)) if recurrence_window.size else 0.0


def structural_centrality(
    center_time: float,
    duration: float,
    intro_pct: float = 0.12,
    peak_lo_pct: float = 0.30,
    peak_hi_pct: float = 0.65,
    outro_pct: float = 0.15,
) -> float:
    """Trapezoidal prior over song position, in ``[0, 1]``.

    Zero through the intro, ramps up to a 1.0 plateau across the middle of the
    song, then ramps back down through the outro.
    """
    if duration <= 0:
        return 0.0
    p = center_time / duration
    if p <= intro_pct:
        return 0.0
    if p < peak_lo_pct:
        return (p - intro_pct) / (peak_lo_pct - intro_pct)
    if p <= peak_hi_pct:
        return 1.0
    outro_start = 1.0 - outro_pct
    if p < outro_start:
        return (outro_start - p) / (outro_start - peak_hi_pct)
    return 0.0


# --- combination -------------------------------------------------------------


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Scale an array of candidate scores to ``[0, 1]`` (flat -> zeros)."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    span = hi - lo
    if span <= 1e-9:
        return np.zeros_like(values)
    return (values - lo) / span


def combine(normalized: dict, weights: Weights = DEFAULT_WEIGHTS) -> np.ndarray:
    """Weighted sum of normalized per-signal candidate arrays.

    ``normalized`` maps each name in :data:`SIGNALS` to an array of one value
    per candidate window. Returns the combined score per candidate.
    """
    w = weights.as_dict()
    total = None
    for name in SIGNALS:
        contribution = w[name] * np.asarray(normalized[name], dtype=float)
        total = contribution if total is None else total + contribution
    return total
