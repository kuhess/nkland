from pathlib import Path

import numpy as np
from nkland import NKLand


def test_roundtrip():
    landscape = NKLand.random(5, 1, seed=0)
    solutions = landscape.sample()
    fitness = landscape.evaluate(solutions)
    np.testing.assert_array_almost_equal(
        fitness,
        [0.308569],
    )


def test_roundtrip_with_m_samples():
    landscape = NKLand.random(5, 1, seed=0)
    solutions = landscape.sample(3)
    fitness = landscape.evaluate(solutions)
    np.testing.assert_array_almost_equal(
        fitness,
        [0.308569, 0.509432, 0.551306],
    )


def test_save_then_load(tmp_path: Path):
    landscape = NKLand.random(32, 1, seed=123)
    filepath = tmp_path / "landscape_32_1.npz"

    landscape.save(filepath)

    loaded_landscape = NKLand.load(filepath)

    assert loaded_landscape._n == landscape._n
    assert loaded_landscape._k == landscape._k
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


def test_create_landscape_from_interactions():
    interaction_indices = np.asarray(
        [
            [0, 1],
            [1, 3],
            [1, 2],
            [0, 3],
        ]
    )
    contributions = np.asarray(
        [
            # col0 col1 col2 col3
            [0.1, 0.6, 0.3, 0.3],  # row0
            [0.8, 0.7, 0.1, 0.5],  # row1
            [0.2, 0.7, 0.9, 0.6],  # row2
            [0.3, 0.5, 0.2, 0.6],  # row3
        ]
    )
    landscape = NKLand(
        interaction_indices=interaction_indices,
        fitness_contributions=contributions,
    )

    assert landscape._n == 4
    assert landscape._k == 1

    np.testing.assert_almost_equal(
        landscape.evaluate(np.asarray([[0, 1, 0, 0]])),
        (
            0.6  # row0, (0, 1) = col1
            + 0.1  # row1, (1, 0) = col2
            + 0.9  # row2, (1, 0) = col2
            + 0.3  # row3, (0, 0) = col0
        )
        / 4,
    )
