from __future__ import annotations

import numpy as np
import numpy.typing as npt

MAX_N = 1024
MAX_K = 63


class NKLand:
    def __init__(self, n: int, k: int, seed: int | None = None) -> None:
        r"""Create the NK landscape model.

        Parameters
        ----------
        n : int
            Number of components, $N$.
        k : int
            Number of interactions per component, $K$.
        seed : (None | int)
            Seed for random number generation.
            See [NumPy doc](https://numpy.org/doc/1.26/reference/random/generator.html#numpy.random.default_rng)
            for more information.

        """
        if n > MAX_N:
            msg = f"n cannot be greater than {MAX_N}: n={n}"
            raise ValueError(msg)
        if k > MAX_K:
            msg = f"k cannot be greater than {MAX_K}: k={k}"
            raise ValueError(msg)
        if k > n - 1:
            msg = f"k cannot be greater than n-1={n-1}: k={k}"
            raise ValueError(msg)

        self.n = n
        self.k = k
        self.seed = seed

        self._powers2 = 1 << np.arange(self.k, -1, -1)
        self._rng = np.random.default_rng(seed)

        self.interaction_indices_matrix = self._generate_interaction_indices_matrix()
        self.fitness_contributions = self._generate_fitness_contributions()

    def _generate_interaction_indices_matrix(self) -> npt.NDArray[np.int32]:
        interaction_indices_matrix = np.empty((self.n, self.k + 1), dtype=np.int32)
        for i in np.arange(self.n, dtype=np.int32):
            # remove itself
            possible_interaction_indices = np.delete(np.arange(self.n), i)
            # note that choice method uses uniform distribution by default
            interaction_indices = self._rng.choice(
                possible_interaction_indices,
                size=self.k,
                replace=False,
            )
            # add self interaction + sort
            interaction_indices_matrix[i] = np.sort(
                np.concatenate(
                    (
                        interaction_indices,
                        np.asarray([i]),
                    ),
                ),
            )
        return interaction_indices_matrix

    def _generate_fitness_contributions(self) -> npt.NDArray[np.float64]:
        num_interactions = 2 ** (self.k + 1)
        return self._rng.uniform(size=(self.n, num_interactions))

    def evaluate(self, solution: npt.ArrayLike) -> np.float64:
        r"""Evaluate the fitness of a solution.

        Parameters
        ----------
        solution : npt.ArrayLike
            Binary vector of the solution.

        Returns
        -------
        np.float64
            The fitness value of the solution.

        """
        s = np.asarray(solution)
        if s.shape != (self.n,):
            msg = (
                "bad shape for argument `solution`, "
                f"expected shape={(self.n,)} but got shape={s.shape}"
            )
            raise ValueError(msg)
        if not np.all((s == 0) | (s == 1)):
            msg = "solution contains non-binary values: solution={s}"
            raise ValueError(msg)

        contributions_bits_indices = s[self.interaction_indices_matrix]
        contributions_indices = np.dot(contributions_bits_indices, self._powers2)

        fitness: np.float64 = np.mean(
            self.fitness_contributions[np.arange(self.n), contributions_indices]
        )

        return fitness

    def sample(self) -> npt.NDArray[np.uint8]:
        r"""Get a random solution of $N$ dimensions.

        Returns
        -------
        npt.NDArray[np.uint8]
            Vector $s$ of $N$ binary values.

        """
        return self._rng.integers(0, 2, size=(self.n,), dtype=np.uint8)
