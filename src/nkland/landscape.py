from __future__ import annotations

import numpy as np
import numpy.typing as npt

MAX_N = 1024
MAX_K = 63

NDIM_EVALUATE = 2


class NKLand:
    def __init__(self, n: int, k: int, seed: int | None = None) -> None:
        r"""Create the NK landscape model.

        Read more in the [NK model description](../nkmodel.md).

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

    def evaluate(self, solutions: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        r"""Evaluate the fitness values of multiple solutions.

        Read more about the [fitness formula](../nkmodel.md#fitness).

        Parameters
        ----------
        solutions : npt.NDArray[np.uint8]
            Matrix of solutions $S_{N \times m}$ with $m$ as the number of solutions.
            Each row $s_i$ contains $N$ binary values (0 or 1).

        Returns
        -------
        npt.NDArray[np.float64]
            The fitness values corresponding to the solutions.

        """
        if not (solutions.ndim == NDIM_EVALUATE and solutions.shape[1] == self.n):
            msg = (
                "bad shape for argument `solutions`, "
                f"expected shape==(m,{self.n}) but got shape=={solutions.shape}"
            )
            raise ValueError(msg)
        if not np.all((solutions == 0) | (solutions == 1)):
            msg = "solutions contains non-binary values: solutions={solutions}"
            raise ValueError(msg)

        contributions_bits = solutions[..., self.interaction_indices_matrix]
        contributions_indices = np.dot(contributions_bits, self._powers2)

        fitness: npt.NDArray[np.float64] = np.mean(
            self.fitness_contributions[np.arange(self.n), contributions_indices],
            axis=-1,
        )
        return fitness

    def sample(self, m: int = 1) -> npt.NDArray[np.uint8]:
        r"""Get $m$ random solutions of $N$ dimensions.

        Returns
        -------
        npt.NDArray[np.uint8]
            Matrix $S_{m \times N}$ of binary values.

        """
        return self._rng.integers(0, 2, size=(m, self.n), dtype=np.uint8)
