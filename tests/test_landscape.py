from nkland import NKLand


def test_roundtrip():
    landscape = NKLand(5, 1, seed=0)
    solution = landscape.sample()
    landscape.evaluate(solution)
