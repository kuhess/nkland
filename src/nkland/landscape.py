from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.random as npr
import numpy.typing as npt

MAX_N = 1024
MAX_K = 63

NDIM_EVALUATE = 2


class NKLand:
    def __init__(
        self,
        n: int,
        k: int,
        *,
        rng: int | npr.Generator | None = None,
        interaction_indices: npt.NDArray[np.int32] | None = None,
        fitness_contributions: npt.NDArray[np.float64] | None = None,
    ) -> None:
        r"""Create the NK landscape model.

        Parameters
        ----------
        n : int
            Number of components, $N$.
        k : int
            Number of interactions per component, $K$.
        rng : int | npr.Generator | None
            Seed or generator for random number generation.
            See [NumPy doc](https://numpy.org/doc/1.26/reference/random/generator.html#numpy.random.default_rng)
            for more information.
        interaction_indices : npt.NDArray[np.int32] | None
            The matrix $M_{N \times K+1}$ containing the indices of the interactions.
        fitness_contributions : npt.NDArray[np.float64] | None
            The matrix $C_{N \times 2^{K+1}}$ containing the contributions to the
            fitness.

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

        self._powers2 = 1 << np.arange(self.k, -1, -1)
        self._rng = npr.default_rng(rng)

        if interaction_indices is None:
            interaction_indices = self._generate_interaction_indices()
        self.interaction_indices = interaction_indices

        if fitness_contributions is None:
            fitness_contributions = self._generate_fitness_contributions()
        self.fitness_contributions = fitness_contributions

    def _generate_interaction_indices(self) -> npt.NDArray[np.int32]:
        interaction_indices = np.empty((self.n, self.k + 1), dtype=np.int32)
        for i in np.arange(self.n, dtype=np.int32):
            # remove itself
            possible_interaction_indices = np.delete(np.arange(self.n), i)
            # note that choice method uses uniform distribution by default
            picked_indices = self._rng.choice(
                possible_interaction_indices,
                size=self.k,
                replace=False,
            )
            # add self interaction + sort
            interaction_indices[i] = np.sort(
                np.concatenate(
                    (
                        picked_indices,
                        np.asarray([i]),
                    ),
                ),
            )
        return interaction_indices

    def _generate_fitness_contributions(self) -> npt.NDArray[np.float64]:
        num_interactions = 2 ** (self.k + 1)
        return self._rng.uniform(size=(self.n, num_interactions))

    def evaluate(self, solutions: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        r"""Evaluate the fitness values of multiple solutions.

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

        contributions_bits = solutions[..., self.interaction_indices]
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

    def save(self, file: str | Path) -> None:
        """Save the NKLand instance to a file in NumPy `.npz` format.

        Parameters
        ----------
        file : str | Path
            The path where the instance will be saved. The `.npz` extension will be
            appended to the filename if it is not already there.

        """
        np.savez(
            file,
            allow_pickle=False,
            n=self.n,
            k=self.k,
            interaction_indices=self.interaction_indices,
            fitness_contributions=self.fitness_contributions,
            bitgenerator_state=json.dumps(self._rng.bit_generator.state),
        )

    @staticmethod
    def load(file: str | Path) -> NKLand:
        """Create an NKLand instance from a file written with the save method.

        Parameters
        ----------
        file : str | Path
            The path of the file to load.

        Returns
        -------
        NKLand
            The instance loaded.

        """
        data = np.load(file, allow_pickle=False)

        rng = npr.default_rng()
        rng.bit_generator.state = json.loads(data["bitgenerator_state"].item())

        return NKLand(
            n=data["n"].item(),
            k=data["k"].item(),
            rng=rng,
            interaction_indices=data["interaction_indices"],
            fitness_contributions=data["fitness_contributions"],
        )
