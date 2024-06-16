from pathlib import Path

import numpy as np
from nkland import NKLand


def test_roundtrip():
    landscape = NKLand(5, 1, rng=0)
    solutions = landscape.sample()
    fitness = landscape.evaluate(solutions)
    np.testing.assert_array_almost_equal(
        fitness,
        [0.308569],
    )


def test_roundtrip_with_m_samples():
    landscape = NKLand(5, 1, rng=0)
    solutions = landscape.sample(3)
    fitness = landscape.evaluate(solutions)
    np.testing.assert_array_almost_equal(
        fitness,
        [0.308569, 0.509432, 0.551306],
    )


def test_save_then_load(tmp_path: Path):
    landscape = NKLand(32, 1, rng=123)
    filepath = tmp_path / "landscape_32_1.npz"

    landscape.save(filepath)

    loaded_landscape = NKLand.load(filepath)

    assert loaded_landscape.n == landscape.n
    assert loaded_landscape.k == landscape.k
    np.testing.assert_array_equal(
        loaded_landscape.fitness_contributions,
        landscape.fitness_contributions,
    )
    np.testing.assert_array_equal(
        loaded_landscape.interaction_indices,
        landscape.interaction_indices,
    )
    np.testing.assert_array_equal(
        loaded_landscape._rng.integers(low=0, high=10, size=10),
        landscape._rng.integers(low=0, high=10, size=10),
    )
