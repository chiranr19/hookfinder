# ✦ @chiranr19 · © 2026 All Rights Reserved · proprietary · sigil:AQY3QG3XBBYYR2WN
"""Hook detection: scan candidate windows, score them, return the best clip."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from . import scoring
from .features import (
    DEFAULT_HOP,
    DEFAULT_QUALITY,
    DEFAULT_SR,
    AudioFeatures,
    extract_features,
    extract_from_file,
)
from .scoring import SIGNALS, Weights

PathLike = Union[str, Path]
AudioInput = Union[PathLike, np.ndarray, AudioFeatures]


@dataclass
class Hook:
    """A detected hook clip and why it was chosen.

    ``components`` holds the normalized ``[0, 1]`` value of each signal at this
    window, so the result explains itself: e.g. ``repetition=0.98`` means this
    was the most-repeated section in the song.
    """

    start: float
    end: float
    score: float
    components: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }

    def __repr__(self) -> str:
        return (
            f"Hook(start={self.start:.2f}s, end={self.end:.2f}s, "
            f"duration={self.duration:.2f}s, score={self.score:.3f})"
        )


class HookFinder:
    """Configurable hook detector.

    Parameters
    ----------
    clip_duration:
        Length of the clip to return, in seconds.
    step:
        Spacing between candidate window starts, in seconds. Smaller is more
        precise but slower.
    weights:
        Signal weighting (see :class:`hookfinder.Weights`).
    align_to_beat:
        Snap the chosen start to a nearby beat for a clean-sounding cut.
    beat_tolerance:
        Max distance (seconds) to snap to a beat.
    rms_percentile:
        Loudness percentile used as the "is this loud" threshold for the
        sustained-energy signal.
    quality:
        ``"fast"`` (default) or ``"high"``. See :mod:`hookfinder.features`.
        Ignored when precomputed ``AudioFeatures`` are passed in.
    """

    def __init__(
        self,
        clip_duration: float = 30.0,
        step: float = 1.0,
        weights: Optional[Weights] = None,
        align_to_beat: bool = True,
        beat_tolerance: float = 1.0,
        rms_percentile: float = 75.0,
        quality: str = DEFAULT_QUALITY,
        sr: int = DEFAULT_SR,
        hop_length: int = DEFAULT_HOP,
    ):
        if clip_duration <= 0:
            raise ValueError("clip_duration must be positive.")
        if step <= 0:
            raise ValueError("step must be positive.")
        self.clip_duration = float(clip_duration)
        self.step = float(step)
        self.weights = weights or scoring.DEFAULT_WEIGHTS
        self.align_to_beat = align_to_beat
        self.beat_tolerance = float(beat_tolerance)
        self.rms_percentile = float(rms_percentile)
        self.quality = quality
        self.sr = sr
        self.hop_length = hop_length

    # -- public API ----------------------------------------------------------

    def find(self, audio: AudioInput, sr: Optional[int] = None) -> Hook:
        """Return the single best hook clip."""
        return self.find_candidates(audio, sr=sr, top_k=1)[0]

    def find_candidates(
        self, audio: AudioInput, sr: Optional[int] = None, top_k: int = 5
    ) -> List[Hook]:
        """Return up to ``top_k`` non-overlapping candidate clips, best first."""
        feats = self._as_features(audio, sr)
        return self._rank(feats, top_k=max(1, int(top_k)))

    # -- internals -----------------------------------------------------------

    def _as_features(self, audio: AudioInput, sr: Optional[int]) -> AudioFeatures:
        if isinstance(audio, AudioFeatures):
            return audio
        if isinstance(audio, (str, Path)):
            return extract_from_file(
                audio, sr=self.sr, hop_length=self.hop_length, quality=self.quality
            )
        if isinstance(audio, np.ndarray):
            if sr is None:
                raise ValueError("sr is required when passing a raw waveform array.")
            return extract_features(
                audio, sr=sr, hop_length=self.hop_length, quality=self.quality
            )
        raise TypeError(
            "audio must be a file path, a numpy waveform, or AudioFeatures; "
            f"got {type(audio).__name__}."
        )

    def _rank(self, feats: AudioFeatures, top_k: int) -> List[Hook]:
        duration = feats.duration
        window = self.clip_duration

        # Song is shorter than the requested clip: the whole thing is the hook.
        if duration <= window:
            comps = {name: 1.0 for name in SIGNALS}
            return [Hook(start=0.0, end=duration, score=1.0, components=comps)]

        starts = np.arange(0.0, duration - window + 1e-9, self.step)
        if starts.size == 0:
            starts = np.array([0.0])

        rms_threshold = float(np.nanpercentile(feats.rms, self.rms_percentile))
        ft = feats.frame_times

        raw = {name: [] for name in SIGNALS}
        valid_starts = []
        for s in starts:
            lo = int(np.searchsorted(ft, s, side="left"))
            hi = int(np.searchsorted(ft, s + window, side="right"))
            if hi <= lo:
                continue
            valid_starts.append(s)
            raw["repetition"].append(scoring.repetition(feats.recurrence[lo:hi]))
            raw["harmonic"].append(scoring.harmonic_stability(feats.chroma[:, lo:hi]))
            raw["rhythmic"].append(scoring.rhythmic_intensity(feats.onset_env[lo:hi]))
            raw["energy"].append(scoring.sustained_energy(feats.rms[lo:hi], rms_threshold))
            raw["centrality"].append(
                scoring.structural_centrality(s + 0.5 * window, duration)
            )

        if not valid_starts:
            comps = {name: 0.0 for name in SIGNALS}
            end = min(window, duration)
            return [Hook(start=0.0, end=end, score=0.0, components=comps)]

        valid_starts = np.asarray(valid_starts)

        # Normalize each signal across candidates so weights compose fairly.
        # Centrality is already an absolute [0, 1] prior, so keep it as-is.
        normalized = {}
        for name in SIGNALS:
            arr = np.asarray(raw[name], dtype=float)
            normalized[name] = arr if name == "centrality" else scoring.min_max_normalize(arr)

        scores = scoring.combine(normalized, self.weights)
        # Smooth across neighboring windows so the pick is a broad plateau
        # rather than a one-window spike that a half-second shift would miss.
        scores = _moving_average(scores, width=3)
        order = np.argsort(scores)[::-1]

        hooks = self._select(order, valid_starts, scores, normalized, feats, window, top_k)
        return hooks

    def _select(self, order, starts, scores, normalized, feats, window, top_k):
        """Greedy non-max suppression so top-k clips don't overlap heavily."""
        chosen: List[Hook] = []
        for idx in order:
            start = float(starts[idx])
            aligned = self._align_to_beat(start, feats, window)
            end = min(aligned + window, feats.duration)
            if self._overlaps(aligned, end, chosen, window):
                continue
            comps = {name: float(normalized[name][idx]) for name in SIGNALS}
            chosen.append(Hook(start=aligned, end=end, score=float(scores[idx]), components=comps))
            if len(chosen) >= top_k:
                break
        return chosen

    def _align_to_beat(self, start: float, feats: AudioFeatures, window: float) -> float:
        beats = feats.beat_times
        if self.align_to_beat and beats.size:
            nearest = beats[int(np.argmin(np.abs(beats - start)))]
            if abs(nearest - start) <= self.beat_tolerance:
                start = float(nearest)
        upper = max(0.0, feats.duration - window)
        return float(min(max(0.0, start), upper))

    @staticmethod
    def _overlaps(start, end, chosen: List[Hook], window: float, max_overlap: float = 0.5) -> bool:
        for h in chosen:
            overlap = max(0.0, min(end, h.end) - max(start, h.start))
            if overlap > max_overlap * window:
                return True
        return False


def _moving_average(values: np.ndarray, width: int = 3) -> np.ndarray:
    """Centered moving average, edge-padded to preserve length."""
    values = np.asarray(values, dtype=float)
    if width <= 1 or values.size < width:
        return values
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def find_hook(
    audio: AudioInput,
    clip_duration: float = 30.0,
    sr: Optional[int] = None,
    weights: Optional[Weights] = None,
    align_to_beat: bool = True,
    step: float = 1.0,
    quality: str = DEFAULT_QUALITY,
) -> Hook:
    """Find the single best hook clip in a song.

    Parameters
    ----------
    audio:
        Path to an audio file, a mono waveform array (with ``sr``), or a
        precomputed :class:`~hookfinder.features.AudioFeatures`.
    clip_duration:
        Desired clip length in seconds (default 30).
    sr:
        Sample rate, required only when ``audio`` is a raw waveform array.
    quality:
        ``"fast"`` (default) or ``"high"`` — speed/accuracy tradeoff.

    Returns
    -------
    Hook
        The best clip, with a per-signal ``components`` breakdown.

    Examples
    --------
    >>> from hookfinder import find_hook
    >>> hook = find_hook("song.mp3")            # doctest: +SKIP
    >>> print(hook.start, hook.end, hook.score) # doctest: +SKIP
    """
    finder = HookFinder(
        clip_duration=clip_duration,
        step=step,
        weights=weights,
        align_to_beat=align_to_beat,
        quality=quality,
    )
    return finder.find(audio, sr=sr)
