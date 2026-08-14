import numpy as np
import pytest

from hookfinder import Weights
from hookfinder import scoring


def test_weights_normalize_to_one():
    w = Weights(repetition=1, harmonic=1, rhythmic=1, energy=1, centrality=1)
    assert w.as_array().sum() == pytest.approx(1.0)
    assert all(v == pytest.approx(0.2) for v in w.as_dict().values())


def test_weights_all_zero_falls_back_to_uniform():
    w = Weights(0, 0, 0, 0, 0)
    arr = w.as_array()
    assert arr.sum() == pytest.approx(1.0)
    assert np.allclose(arr, 0.2)


def test_weights_negative_clipped():
    w = Weights(repetition=-5, harmonic=1, rhythmic=1, energy=1, centrality=1)
    arr = w.as_array()
    assert (arr >= 0).all()
    assert arr.sum() == pytest.approx(1.0)


def test_harmonic_stability_bounds_and_identity():
    # A constant chroma sequence is perfectly stable -> ~1.0
    chroma = np.tile(np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[:, None], (1, 20))
    assert scoring.harmonic_stability(chroma) == pytest.approx(1.0, abs=1e-6)
    # Too few frames -> 0
    assert scoring.harmonic_stability(np.zeros((12, 2))) == 0.0


def test_sustained_energy_fraction():
    rms = np.array([0.0, 1.0, 1.0, 1.0])
    assert scoring.sustained_energy(rms, threshold=0.5) == pytest.approx(0.75)
    assert scoring.sustained_energy(np.array([]), 0.5) == 0.0


def test_structural_centrality_shape():
    d = 100.0
    assert scoring.structural_centrality(1, d) == 0.0           # intro
    assert scoring.structural_centrality(50, d) == 1.0          # middle plateau
    assert scoring.structural_centrality(99, d) == 0.0          # outro
    assert 0.0 < scoring.structural_centrality(20, d) < 1.0     # ramp up


def test_min_max_normalize():
    out = scoring.min_max_normalize([1.0, 2.0, 3.0])
    assert out.min() == 0.0 and out.max() == 1.0
    # Flat input -> zeros (no division by zero)
    assert np.allclose(scoring.min_max_normalize([5.0, 5.0, 5.0]), 0.0)


def test_combine_weighted_sum():
    normalized = {name: np.array([0.0, 1.0]) for name in scoring.SIGNALS}
    combined = scoring.combine(normalized, Weights())
    assert combined[0] == pytest.approx(0.0)
    assert combined[1] == pytest.approx(1.0)
