from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

MAX_N = 1024
MAX_K = 32

_Rng = int | torch.Generator


def _is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


class NKLand:
    def __init__(
        self,
        interactions: torch.Tensor,
        fitness_contributions: torch.Tensor,
        *,
        seed: _Rng | None = None,
        use_gpu: bool = False,
    ) -> None:
        r"""Create the NK landscape model.

        Parameters
        ----------
        interactions : torch.Tensor
            The adjacency matrix $A_{N \times N}$ containing binary values.
        fitness_contributions : torch.Tensor
            The matrix $C_{N \times 2^{K+1}}$ containing the contributions to the
            fitness.
        seed : Union[_Rng, None]
            Seed or generator for random number generation.
        use_gpu : bool
            Whether to use GPU acceleration.

        """
        if interactions.ndim != 2:
            msg = (
                "interaction matrix must have 2 dimensions, "
                f"but got: ndim={interactions.ndim}"
            )
            raise ValueError(msg)
        if interactions.size(0) != interactions.size(1):
            msg = (
                "interactions matrix has bad shape, must be squared "
                f"but got: {interactions.shape}"
            )
            raise ValueError(msg)
        if fitness_contributions.ndim != 2:
            msg = (
                "fitness contributions matrix must have 2 dimensions, "
                f"but got: ndim={fitness_contributions.ndim}"
            )
            raise ValueError(msg)
        if interactions.size(0) != fitness_contributions.size(0):
            msg = (
                "first dimensions of `interactions` and `fitness_contribution` do "
                f"not match: {interactions.size(0)}!={fitness_contributions.size(0)}"
            )
            raise ValueError(msg)
        if not _is_power_of_2(fitness_contributions.size(1)):
            msg = (
                "`fitness_contributions` second dimension must be a power of 2, "
                f"but got: {fitness_contributions.size(1)}"
            )
            raise ValueError(msg)
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        if isinstance(seed, torch.Generator):
            self._rng = seed
        else:
            self._rng = torch.Generator()
            if seed is not None:
                self._rng.manual_seed(seed)

        self._n = interactions.size(0)
        self._k = torch.sum(interactions[0, :]).item() - 1

        self.interactions = interactions.to(dtype=torch.uint8, device=self.device)
        self._interaction_indices = torch.nonzero(self.interactions)[:, 1]
        self._interaction_indices = self._interaction_indices.view(self._n, self._k + 1)

        self.fitness_contributions = fitness_contributions.to(
            dtype=torch.float64, device=self.device
        )

        self._powers_of_2 = 2 ** torch.arange(
            self._k, -1, -1, dtype=torch.float32, device=self.device
        )

    def evaluate(self, solutions: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the fitness values of multiple solutions.

        Parameters
        ----------
        solutions : torch.Tensor
            Matrix of solutions $S_{m \times N}$ with $m$ as the number of solutions.
            Each row $s_i$ contains $N$ binary values (0 or 1).

        Returns
        -------
        torch.Tensor
            The fitness values corresponding to the solutions.

        """
        solutions = solutions.to(dtype=torch.uint8, device=self.device)
        contributions_bits = solutions[..., self._interaction_indices].to(torch.float32)
        contributions_indices = contributions_bits @ self._powers_of_2

        return torch.mean(
            self.fitness_contributions[
                torch.arange(self._n, device=self.device),
                contributions_indices.to(dtype=torch.int64, device=self.device),
            ],
            axis=-1,
        )

    def sample(self, m: int = 1) -> torch.Tensor:
        r"""Get $m$ random solutions of $N$ dimensions.

        Returns
        -------
        torch.Tensor
            Matrix $S_{m \times N}$ of binary values.

        """
        return torch.randint(
            0,
            2,
            (m, self._n),
            generator=self._rng,
            dtype=torch.uint8,
            device=self.device,
        )

    def save(self, file: str | Path) -> None:
        """Save the NKLand instance to a file in a format compatible with PyTorch.

        Parameters
        ----------
        file : str | Path
            The path where the instance will be saved.

        """
        torch.save(
            {
                "n": self._n,
                "k": self._k,
                "interactions": self.interactions.cpu(),
                "fitness_contributions": self.fitness_contributions.cpu(),
                "generator_state": self._rng.get_state(),
            },
            file,
        )

    @staticmethod
    def load(file: str | Path, *, use_gpu: bool = False) -> NKLand:
        """Load a NKLand instance from a file written with the save method.

        Parameters
        ----------
        file : Union[str, Path]
            The path of the file to load.
        use_gpu : bool
            Whether to use GPU acceleration.

        Returns
        -------
        NKLand
            The NKLand instance loaded.

        """
        data = torch.load(file)

        rng = torch.Generator()
        rng.set_state(data["generator_state"])

        return NKLand(
            interactions=data["interactions"],
            fitness_contributions=data["fitness_contributions"],
            seed=rng,
            use_gpu=use_gpu,
        )

    @staticmethod
    def random(
        n: int, k: int, *, seed: _Rng | None = None, use_gpu: bool = False
    ) -> NKLand:
        """Create a random NK landscape.

        Parameters
        ----------
        n : int
            Number of components, $N$.
        k : int
            Number of interactions per component, $K$.
        seed : Union[_Rng, None]
            Seed or generator for random number generation.
        use_gpu : bool
            Whether to use GPU acceleration.

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

        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        interactions = NKLand._generate_interactions(n, k, rng, device)
        fitness_contributions = NKLand._generate_fitness_contributions(
            n, k, rng, device
        )

        return NKLand(
            interactions=interactions,
            fitness_contributions=fitness_contributions,
            seed=rng,
            use_gpu=use_gpu,
        )

    @staticmethod
    def _generate_interactions(
        n: int, k: int, rng: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        interactions = torch.eye(n, dtype=torch.int32, device=device)

        for i in range(n):
            possible_neighbors = torch.cat((torch.arange(0, i), torch.arange(i + 1, n)))
            selected_neighbors = possible_neighbors[
                torch.randperm(n - 1, generator=rng)[:k]
            ]
            interactions[i, selected_neighbors] = 1

        return interactions

    @staticmethod
    def _generate_fitness_contributions(
        n: int, k: int, rng: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        num_interactions = 2 ** (k + 1)
        return torch.rand(
            (n, num_interactions),
            generator=rng,
            dtype=torch.float64,
        ).to(device)
