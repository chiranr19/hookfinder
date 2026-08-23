# ✦ @chiranr19 · © 2026 All Rights Reserved · proprietary · sigil:AQY3QG3XBBYYR2WN
"""Shared fixtures: synthetic audio so tests need no files or network."""

import numpy as np
import pytest

SR = 22050


def synth_song(duration=60.0, sr=SR, seed=0):
    """A synthetic track with two loud, identical 'chorus' regions.

    Verses are sparse and quiet; choruses (15-30s and 40-55s) share a motif and
    a beat, so recurrence/energy/rhythm should all favor them.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    y = (0.02 * rng.standard_normal(t.size)).astype(np.float32)

    def add(a, b, freqs, amp, beat=True):
        idx = (t >= a) & (t < b)
        seg = np.zeros_like(t)
        for f in freqs:
            seg += np.sin(2 * np.pi * f * t)
        if beat:
            seg *= 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 2.0 * t))  # ~120 BPM
        y[idx] += (amp * seg[idx]).astype(np.float32)

    add(5, 15, [220, 277, 330], 0.20)              # verse 1
    add(15, 30, [330, 415, 523, 659], 0.60)        # chorus 1
    add(30, 40, [220, 277, 330], 0.20)             # verse 2
    add(40, 55, [330, 415, 523, 659], 0.60)        # chorus 2 (repeat)
    return y, sr


@pytest.fixture(scope="session")
def song():
    return synth_song()


@pytest.fixture(scope="session")
def long_song():
    """~12 minutes, built by tiling the synthetic song."""
    y, sr = synth_song()
    return np.tile(y, 12), sr


@pytest.fixture(scope="session")
def song_wav(tmp_path_factory, song):
    import soundfile as sf

    y, sr = song
    path = tmp_path_factory.mktemp("audio") / "synth.wav"
    sf.write(str(path), y, sr)
    return path
