"""Audio feature extraction for hook detection.

Everything the scorer needs is computed once here, per song, at a fixed frame
rate: harmonic chroma, percussive onset strength, loudness (RMS), beat times,
and a per-frame "recurrence" strength that captures how much each moment of the
song repeats elsewhere (the core signal for chorus/hook-ness).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import numpy as np

DEFAULT_SR = 22050
DEFAULT_HOP = 512

PathLike = Union[str, Path]


@dataclass
class AudioFeatures:
    """Frame-aligned features for a single track.

    All time-series arrays share the same length ``n_frames`` and the same
    ``frame_times`` axis, so a candidate window can be sliced consistently
    across every signal.
    """

    sr: int
    hop_length: int
    duration: float
    frame_times: np.ndarray  # (n_frames,) seconds
    chroma: np.ndarray  # (12, n_frames) harmonic chroma
    onset_env: np.ndarray  # (n_frames,) percussive onset strength
    rms: np.ndarray  # (n_frames,) loudness envelope
    beat_times: np.ndarray  # (n_beats,) seconds
    recurrence: np.ndarray  # (n_frames,) how much this frame repeats, in [0, 1]

    @property
    def n_frames(self) -> int:
        return self.frame_times.shape[0]


def load_audio(path: PathLike, sr: int = DEFAULT_SR, mono: bool = True):
    """Load an audio file to a mono waveform at ``sr`` Hz."""
    y, sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, sr


def extract_features(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP,
) -> AudioFeatures:
    """Compute all per-frame features from a mono waveform."""
    if y.ndim > 1:
        y = librosa.to_mono(y)
    duration = float(len(y) / sr)
    if len(y) < sr * 3:
        raise ValueError("Audio is too short to analyze (need at least ~3 seconds).")

    # Separate harmonic (melody/harmony) from percussive (rhythm) content so
    # each signal is measured on the source that actually carries it.
    y_harm, y_perc = librosa.effects.hpss(y)

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Trim every series to a common length; librosa can return off-by-one sizes.
    n = min(chroma.shape[1], onset_env.shape[0], rms.shape[0])
    chroma = chroma[:, :n]
    onset_env = onset_env[:n]
    rms = rms[:n]

    frame_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)

    _, beat_frames = librosa.beat.beat_track(
        y=y_perc, sr=sr, hop_length=hop_length, units="frames"
    )
    beat_frames = np.asarray(beat_frames)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    recurrence = _frame_recurrence(chroma, beat_frames, n)

    return AudioFeatures(
        sr=sr,
        hop_length=hop_length,
        duration=duration,
        frame_times=frame_times,
        chroma=chroma,
        onset_env=onset_env,
        rms=rms,
        beat_times=beat_times,
        recurrence=recurrence,
    )


def extract_from_file(
    path: PathLike, sr: int = DEFAULT_SR, hop_length: int = DEFAULT_HOP
) -> AudioFeatures:
    """Convenience: load a file and extract features in one call."""
    y, sr = load_audio(path, sr=sr)
    return extract_features(y, sr=sr, hop_length=hop_length)


def _frame_recurrence(
    chroma: np.ndarray, beat_frames: np.ndarray, n_frames: int
) -> np.ndarray:
    """Per-frame recurrence strength in ``[0, 1]``.

    Beat-synchronize the chroma, build a recurrence (self-similarity) matrix,
    and reduce it to a per-beat "how much does this beat repeat elsewhere"
    score, then broadcast that back onto frames. Sections that recur across the
    song — choruses, hooks, main themes — light up here.
    """
    beat_frames = np.asarray(beat_frames)
    if beat_frames.size < 4:
        return np.zeros(n_frames, dtype=float)

    bounds = np.unique(
        np.clip(np.concatenate([[0], beat_frames, [n_frames]]), 0, n_frames)
    )
    cols, seg_starts = [], []
    for i in range(bounds.size - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b > a:
            cols.append(chroma[:, a:b].mean(axis=1))
            seg_starts.append(a)
    if len(cols) < 4:
        return np.zeros(n_frames, dtype=float)

    sync = np.stack(cols, axis=1)
    seg_starts = np.asarray(seg_starts)

    try:
        rec = librosa.segment.recurrence_matrix(sync, mode="affinity", sym=True)
    except Exception:
        return np.zeros(n_frames, dtype=float)

    strength = rec.mean(axis=1)
    span = float(strength.max() - strength.min())
    strength = (strength - strength.min()) / span if span > 1e-9 else np.zeros_like(strength)

    seg_idx = np.searchsorted(seg_starts, np.arange(n_frames), side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(strength) - 1)
    return strength[seg_idx]
