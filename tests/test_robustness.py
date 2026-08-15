"""Degenerate and awkward inputs should fail cleanly or succeed sanely."""

import numpy as np
import pytest

from hookfinder import HookFinder, extract_features, find_hook
from hookfinder.scoring import SIGNALS

SR = 22050


def _valid(hook, total_duration):
    assert 0.0 <= hook.start <= total_duration
    assert hook.start < hook.end <= total_duration + 1e-6
    assert 0.0 <= hook.score <= 1.0
    assert set(hook.components) == set(SIGNALS)
    assert all(0.0 <= v <= 1.0 for v in hook.components.values())


def test_digital_silence_does_not_crash():
    y = np.zeros(SR * 30, dtype=np.float32)
    _valid(find_hook(y, sr=SR, clip_duration=10), 30)


def test_white_noise_does_not_crash():
    y = np.random.default_rng(0).standard_normal(SR * 30).astype(np.float32) * 0.1
    _valid(find_hook(y, sr=SR, clip_duration=10), 30)


def test_pure_tone_does_not_crash():
    t = np.arange(SR * 20) / SR
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    _valid(find_hook(y, sr=SR, clip_duration=8), 20)


def test_nan_input_rejected():
    y = np.zeros(SR * 10, dtype=np.float32)
    y[1000] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        find_hook(y, sr=SR)


def test_inf_input_rejected():
    y = np.zeros(SR * 10, dtype=np.float32)
    y[500] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        find_hook(y, sr=SR)


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        find_hook(np.array([], dtype=np.float32), sr=SR)


def test_int16_pcm_accepted(song):
    y, sr = song
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    hook = find_hook(pcm, sr=sr, clip_duration=15)
    _valid(hook, len(y) / sr)


def test_stereo_input_downmixed(song):
    y, sr = song
    stereo = np.stack([y, y])  # (2, n) as librosa expects
    hook = find_hook(stereo, sr=sr, clip_duration=15)
    _valid(hook, len(y) / sr)


def test_top_k_zero_clamped(song):
    y, sr = song
    cands = HookFinder(clip_duration=15).find_candidates(y, sr=sr, top_k=0)
    assert len(cands) == 1


def test_bad_quality_rejected(song):
    y, sr = song
    with pytest.raises(ValueError, match="quality"):
        find_hook(y, sr=sr, quality="ultra")


def test_step_larger_than_song(song):
    # A single candidate window; must still return something valid.
    y, sr = song
    hook = HookFinder(clip_duration=15, step=999.0).find(y, sr=sr)
    _valid(hook, len(y) / sr)


def test_zero_weights_still_scores(song):
    from hookfinder import Weights

    y, sr = song
    hook = HookFinder(clip_duration=15, weights=Weights(0, 0, 0, 0, 0)).find(y, sr=sr)
    _valid(hook, len(y) / sr)


def test_quality_modes_agree_on_region(song):
    """Both modes should find a hook inside one of the two chorus regions."""
    y, sr = song
    for quality in ("fast", "high"):
        hook = find_hook(y, sr=sr, clip_duration=15, quality=quality)
        in_chorus = (hook.start < 30 and hook.end > 15) or (hook.start < 55 and hook.end > 40)
        assert in_chorus, f"{quality} picked {hook.start:.1f}s, outside both choruses"


def test_recurrence_matrix_capped_for_very_long_audio():
    """An hour-scale beat grid gets coarsened instead of allocating N^2."""
    from hookfinder.features import MAX_RECURRENCE_SEGMENTS, _frame_recurrence

    n_frames = 300_000  # ~1.9 hours at hop 512 / 22050 Hz
    beats = np.arange(0, n_frames, 40)
    assert beats.size > MAX_RECURRENCE_SEGMENTS  # guard must actually engage
    chroma = np.random.default_rng(0).random((12, n_frames)).astype(np.float32)

    rec = _frame_recurrence(chroma, beats, n_frames)
    assert rec.shape == (n_frames,)
    assert np.all(np.isfinite(rec))
    assert 0.0 <= rec.min() and rec.max() <= 1.0


def test_long_file_completes(long_song):
    """A ~12 minute input must analyze without blowing up memory or time."""
    y, sr = long_song
    hook = find_hook(y, sr=sr, clip_duration=30, step=5.0)
    _valid(hook, len(y) / sr)
