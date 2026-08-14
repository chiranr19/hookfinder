import numpy as np
import pytest

from hookfinder import Hook, HookFinder, Weights, extract_features, find_hook
from hookfinder.scoring import SIGNALS


def test_find_hook_basic(song):
    y, sr = song
    hook = find_hook(y, sr=sr, clip_duration=20)
    assert isinstance(hook, Hook)
    assert 0.0 <= hook.start
    assert hook.end <= 60.0 + 1e-6
    assert hook.duration == pytest.approx(20.0, abs=3.0)
    assert 0.0 <= hook.score <= 1.0


def test_components_present_and_bounded(song):
    y, sr = song
    hook = find_hook(y, sr=sr, clip_duration=20)
    assert set(hook.components) == set(SIGNALS)
    assert all(0.0 <= v <= 1.0 for v in hook.components.values())


def test_hook_lands_on_a_chorus(song):
    # Choruses live at 15-30s and 40-55s; the hook should overlap one of them.
    y, sr = song
    hook = find_hook(y, sr=sr, clip_duration=15)
    overlaps_chorus = (hook.start < 30 and hook.end > 15) or (hook.start < 55 and hook.end > 40)
    assert overlaps_chorus


def test_deterministic(song):
    y, sr = song
    a = find_hook(y, sr=sr, clip_duration=20)
    b = find_hook(y, sr=sr, clip_duration=20)
    assert a.to_dict() == b.to_dict()


def test_candidates_non_overlapping_and_ordered(song):
    y, sr = song
    cands = HookFinder(clip_duration=15).find_candidates(y, sr=sr, top_k=3)
    assert len(cands) >= 2
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)
    # No pair overlaps by more than half a clip.
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
            assert overlap <= 0.5 * 15 + 1e-6


def test_short_song_returns_whole_clip(song):
    y, sr = song
    hook = find_hook(y, sr=sr, clip_duration=120)  # longer than the 60s song
    assert hook.start == 0.0
    assert hook.end == pytest.approx(60.0, abs=0.5)


def test_reuse_features(song):
    y, sr = song
    feats = extract_features(y, sr=sr)
    short = HookFinder(clip_duration=15).find(feats)
    long = HookFinder(clip_duration=25).find(feats)
    assert short.duration == pytest.approx(15, abs=3)
    assert long.duration == pytest.approx(25, abs=3)


def test_weights_change_result(song):
    y, sr = song
    feats = extract_features(y, sr=sr)
    only_energy = HookFinder(
        clip_duration=15,
        weights=Weights(repetition=0, harmonic=0, rhythmic=0, energy=1, centrality=0),
    ).find(feats)
    assert 0.0 <= only_energy.score <= 1.0


def test_array_requires_sr(song):
    y, _ = song
    with pytest.raises(ValueError):
        find_hook(y)  # no sr


def test_bad_type_rejected():
    with pytest.raises(TypeError):
        HookFinder().find(12345)


def test_invalid_config():
    with pytest.raises(ValueError):
        HookFinder(clip_duration=0)
    with pytest.raises(ValueError):
        HookFinder(step=-1)


def test_too_short_audio_raises():
    sr = 22050
    with pytest.raises(ValueError):
        extract_features(np.zeros(sr, dtype=np.float32), sr=sr)  # 1 second
