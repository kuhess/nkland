import numpy as np
from nkland import NKLand


def test_roundtrip():
    landscape = NKLand(5, 1, seed=0)
    solutions = landscape.sample()
    fitness = landscape.evaluate(solutions)
    np.testing.assert_array_almost_equal(
        fitness,
        [0.308569],
    )


def test_roundtrip_with_m_samples():
    landscape = NKLand(5, 1, seed=0)
    solutions = landscape.sample(3)
    fitness = landscape.evaluate(solutions)
    np.testing.assert_array_almost_equal(
        fitness,
        [0.308569, 0.509432, 0.551306],
    )
