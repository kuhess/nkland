import statistics
from pathlib import Path

import torch
from nkland import NKLand


def test_roundtrip():
    landscape = NKLand.random(5, 1, seed=0)
    solutions = landscape.sample()
    fitness = landscape.evaluate(solutions)
    torch.testing.assert_close(
        fitness,
        torch.tensor(0.5573908, dtype=torch.float64),
    )


def test_roundtrip_with_m_samples():
    landscape = NKLand.random(5, 1, seed=0)
    solutions = landscape.sample(3)
    fitness = landscape.evaluate(solutions)
    torch.testing.assert_close(
        fitness,
        torch.tensor(
            [0.5573908, 0.4950997, 0.5762802],
            dtype=torch.float64,
        ),
    )


def test_roundtrip_with_m_samples_b_batches():
    landscape = NKLand.random(5, 1, num_instances=3, seed=0)
    solutions = landscape.sample(2)
    fitness = landscape.evaluate(solutions)
    torch.testing.assert_close(
        fitness,
        torch.tensor(
            [
                [0.6462518, 0.6450555],
                [0.5083262, 0.7238364],
                [0.7537670, 0.4888107],
            ],
            dtype=torch.float64,
        ),
    )


def test_save_then_load(tmp_path: Path):
    landscape = NKLand.random(32, 1, seed=123)
    filepath = tmp_path / "landscape_32_1"

    landscape.save(filepath)

    loaded_landscape = NKLand.load(filepath)

    assert loaded_landscape._n == landscape._n
    assert loaded_landscape._k == landscape._k
    assert loaded_landscape._num_instances == landscape._num_instances
    torch.testing.assert_close(
        loaded_landscape.fitness_contributions,
        landscape.fitness_contributions,
    )
    torch.testing.assert_close(
        loaded_landscape.interactions,
        landscape.interactions,
    )
    torch.testing.assert_close(
        torch.rand(3, generator=loaded_landscape._rng),
        torch.rand(3, generator=landscape._rng),
    )


def test_create_landscape_from_interactions():
    # Adjacency matrix
    interactions = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
        ]
    )
    # Contribution matrix
    contributions = torch.tensor(
        [
            [
                # col0 col1 col2 col3
                [0.1, 0.6, 0.3, 0.3],  # row0
                [0.8, 0.7, 0.1, 0.5],  # row1
                [0.2, 0.7, 0.9, 0.6],  # row2
                [0.3, 0.5, 0.2, 0.6],  # row3
            ],
        ]
    )
    landscape = NKLand(
        interactions=interactions,
        fitness_contributions=contributions,
    )

    assert landscape._n == 4
    assert landscape._k == 1
    assert landscape._num_instances == 1

    actual = landscape.evaluate(torch.tensor([[0, 1, 0, 0], [1, 0, 1, 1]]))

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                statistics.mean(
                    [
                        0.6,  # row0, col1 (0, 1)
                        0.1,  # row1, col2 (1, 0)
                        0.9,  # row2, col2 (1, 0)
                        0.3,  # row3, col0 (0, 0)
                    ]
                ),
                statistics.mean(
                    [
                        0.2,  # row0, col2 (1, 0)
                        0.8,  # row1, col0 (0, 0)
                        0.7,  # row2, col1 (0, 1)
                        0.6,  # row3, col3 (1, 1)
                    ]
                ),
            ],
            dtype=torch.float64,
        ),
    )
