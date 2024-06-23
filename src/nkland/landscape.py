from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import numpy.random as npr
import numpy.typing as npt

MAX_N = 1024
MAX_K = 63

_Rng = Union[int, npr.SeedSequence, npr.BitGenerator, npr.Generator]


class NKLand:
    def __init__(
        self,
        interaction_indices: npt.NDArray[np.int32],
        fitness_contributions: npt.NDArray[np.float64],
        seed: Union[_Rng, None] = None,
    ) -> None:
        r"""Create the NK landscape model.

        Parameters
        ----------
        interaction_indices : npt.NDArray[np.int32]
            The matrix $M_{N \times K+1}$ containing the indices of the interactions.
        fitness_contributions : npt.NDArray[np.float64]
            The matrix $C_{N \times 2^{K+1}}$ containing the contributions to the
            fitness.
        seed : Union[_Rng, None]
            Seed or generator for random number generation.
            See [NumPy doc](https://numpy.org/doc/1.26/reference/random/generator.html#numpy.random.default_rng)
            for more information.

        """
        if interaction_indices.ndim != 2:
            msg = (
                "interaction indices matrix must be have 2 dimensions, but got: "
                f"ndim={interaction_indices.ndim}"
            )
            raise ValueError(msg)
        if interaction_indices.shape[0] <= interaction_indices.shape[1]:
            msg = (
                "interaction indices matrix has bad shape, the shape (n, k+1) must"
                "have the following property n > k, but got: "
                f"n={interaction_indices.shape[0]} k={interaction_indices.shape[1]}"
            )
            raise ValueError(msg)
        if fitness_contributions.ndim != 2:
            msg = (
                "fitness contributions matrix must have 2 dimensions, but got: "
                f"ndim={fitness_contributions.ndim}"
            )
            raise ValueError(msg)

        if not (
            interaction_indices.shape[0] == fitness_contributions.shape[0]
            and interaction_indices.shape[1] == np.log2(fitness_contributions.shape[1])
        ):
            shape_a = interaction_indices.shape
            shape_b = fitness_contributions.shape
            msg = (
                "interaction indices and fitness contributions shapes do not match "
                "requirements: shapes must be (n, k+1) and (n, 2**(k+1)), but got "
                f"({shape_a[0]}, {shape_a[1]-1}-1) and "
                f"({shape_b[0]}, 2**{np.log2(shape_b[1])})"
            )
            raise ValueError(msg)

        self.interaction_indices = interaction_indices
        self.fitness_contributions = fitness_contributions
        self._rng = npr.default_rng(seed)

        self._n = interaction_indices.shape[0]
        self._k = interaction_indices.shape[1] - 1
        self._powers2 = 1 << np.arange(self._k, -1, -1)

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
        if not (solutions.ndim == 2 and solutions.shape[1] == self._n):
            msg = (
                "bad shape for argument `solutions`, "
                f"expected shape==(m,{self._n}) but got shape=={solutions.shape}"
            )
            raise ValueError(msg)
        if not np.all((solutions == 0) | (solutions == 1)):
            msg = "solutions contains non-binary values: solutions={solutions}"
            raise ValueError(msg)

        contributions_bits = solutions[..., self.interaction_indices]
        contributions_indices = np.dot(contributions_bits, self._powers2)

        fitness: npt.NDArray[np.float64] = np.mean(
            self.fitness_contributions[np.arange(self._n), contributions_indices],
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
        return self._rng.integers(0, 2, size=(m, self._n), dtype=np.uint8)

    def save(self, file: Union[str, Path]) -> None:
        """Save the NKLand instance to a file in NumPy `.npz` format.

        Parameters
        ----------
        file : Union[str, Path]
            The path where the instance will be saved. The `.npz` extension will be
            appended to the filename if it is not already there.

        """
        np.savez(
            file,
            allow_pickle=False,
            n=self._n,
            k=self._k,
            interaction_indices=self.interaction_indices,
            fitness_contributions=self.fitness_contributions,
            bitgenerator_state=json.dumps(self._rng.bit_generator.state),
        )

    @staticmethod
    def load(file: Union[str, Path]) -> NKLand:
        """Load a NKLand instance from a file written with the save method.

        Parameters
        ----------
        file : Union[str, Path]
            The path of the file to load.

        Returns
        -------
        NKLand
            The NKLand instance loaded.

        """
        data = np.load(file, allow_pickle=False)

        rng = npr.default_rng()
        rng.bit_generator.state = json.loads(data["bitgenerator_state"].item())

        return NKLand(
            interaction_indices=data["interaction_indices"],
            fitness_contributions=data["fitness_contributions"],
            seed=rng,
        )

    @staticmethod
    def random(n: int, k: int, seed: Union[_Rng, None] = None) -> NKLand:
        """Create a random NK landscape.

        Parameters
        ----------
        n : int
            Number of components, $N$.
        k : int
            Number of interactions per component, $K$.
        seed : Union[_Rng, None]
            Seed or generator for random number generation.

        Result
        ------
        NKLand
            A random NK landscape.

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

        rng = npr.default_rng(seed)
        interaction_indices = NKLand._generate_interaction_indices(n, k, rng)
        fitness_contributions = NKLand._generate_fitness_contributions(n, k, rng)

        return NKLand(
            interaction_indices=interaction_indices,
            fitness_contributions=fitness_contributions,
            seed=rng,
        )

    @staticmethod
    def _generate_interaction_indices(
        n: int, k: int, rng: Union[npr.Generator, None] = None
    ) -> npt.NDArray[np.int32]:
        interaction_indices = np.empty((n, k + 1), dtype=np.int32)
        for i in np.arange(n, dtype=np.int32):
            # remove itself
            possible_interaction_indices = np.delete(np.arange(n), i)
            # note that choice method uses uniform distribution by default
            picked_indices = rng.choice(
                possible_interaction_indices,
                size=k,
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

    @staticmethod
    def _generate_fitness_contributions(
        n: int, k: int, rng: Union[npr.Generator, None] = None
    ) -> npt.NDArray[np.float64]:
        num_interactions = 2 ** (k + 1)
        return rng.uniform(size=(n, num_interactions))
