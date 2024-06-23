from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

MAX_N = 1024
MAX_K = 32

_Rng = Union[int, torch.Generator]


def _is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


def default_rng(seed: Union[_Rng, None] = None) -> torch.Generator:
    """Create a default random number generator.

    Parameters
    ----------
    seed : Union[_Rng, None], optional
        Seed or generator for random number generation, by default None.

    Returns
    -------
    torch.Generator
        A random number generator.

    """
    if isinstance(seed, torch.Generator):
        return seed
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return rng


class NKLand:
    def __init__(
        self,
        interactions: torch.Tensor,
        fitness_contributions: torch.Tensor,
        *,
        seed: Union[_Rng, None] = None,
        use_gpu: bool = False,
    ) -> None:
        r"""Create the NK landscape model.

        Parameters
        ----------
        interactions : torch.Tensor
            The adjacency matrix $A_{N \times N}$ containing binary values (0 or 1).

            It can be a batch of adjacencies matrix with shape
            $(\text{num_instances}, N, N)$.
        fitness_contributions : torch.Tensor
            The matrix $C_{N \times 2^{K+1}}$ containing the contributions to the
            fitness.

            It can be a batch of adjacencies matrix with shape
            $(\text{num_instances}, N, 2^{k+1})$.
        seed : Union[_Rng, None], optional
            Seed or generator for random number generation. Default is None.
        use_gpu : bool, optional
            Whether to use GPU acceleration. Default is False.

        """
        if interactions.dim() not in [2, 3]:
            msg = (
                "interactions tensor must have 2 or 3 dimensions, "
                f"but got: {interactions.dim()}"
            )
            raise ValueError(msg)
        if fitness_contributions.dim() not in [2, 3]:
            msg = (
                "fitness contributions tensor must have 2 or 3 dimensions, "
                f"but got: {fitness_contributions.dim()}"
            )
            raise ValueError(msg)
        if (
            interactions.dim() == 3
            and fitness_contributions.dim() == 3
            and interactions.size(0) != fitness_contributions.size(0)
        ):
            msg = (
                "with batches, first dimension (number of instances) of `interactions` "
                "and `fitness_contributions` must have same cardinality, but got: "
                f"{interactions.size(0)} != {fitness_contributions.size(0)}"
            )
            raise ValueError(msg)
        if interactions.size(-1) != interactions.size(-2):
            msg = (
                "`interactions` has bad shape, last 2 dimensions must be equal, "
                f"but got: {interactions.shape}"
            )
            raise ValueError(msg)
        if interactions.size(-2) != fitness_contributions.size(-2):
            msg = (
                "penultimate dimension of `interactions` and `fitness_contribution` do "
                "not match: "
                f"{interactions.size(-2)} != {fitness_contributions.size(-2)}"
            )
            raise ValueError(msg)
        if not _is_power_of_2(fitness_contributions.size(-1)):
            msg = (
                "`fitness_contributions` last dimension must be a power of 2, "
                f"but got: {fitness_contributions.size(1)}"
            )
            raise ValueError(msg)

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        self._rng = default_rng(seed)

        if interactions.dim() == 2:
            interactions = interactions.unsqueeze(0)
        self.interactions = interactions.to(dtype=torch.uint8, device=self.device)

        if fitness_contributions.dim() == 2:
            fitness_contributions = fitness_contributions.unsqueeze(0)
        self.fitness_contributions = fitness_contributions.to(
            dtype=torch.float64, device=self.device
        )

        self._n = self.interactions.size(-1)
        self._k = torch.sum(self.interactions[0, 0]).item() - 1
        self._num_instances = self.interactions.size(0)

        self._powers_of_2 = 2 ** torch.arange(
            self._k, -1, -1, dtype=torch.float32, device=self.device
        )

    def is_batch(self) -> bool:
        """Check if the NK landscape is a batch.

        Returns
        -------
        bool
            True if there are multiple instances in the batch, False otherwise.

        """
        return self._num_instances > 0

    def evaluate(self, solutions: torch.Tensor) -> Union[torch.Tensor, float]:
        r"""Evaluate the fitness of one or more solutions.

        Parameters
        ----------
        solutions : torch.Tensor
            Matrix of solutions $S_{m \times N}$ with $m$ as the number of solutions.
            Each row $s_i$ contains $N$ binary values (0 or 1).

            It can be a batch of solution matrices with shape
            $(\text{num_instances}, m, N)$.

        Returns
        -------
        Union[torch.Tensor, float]
            The fitness values corresponding to the solutions.

            Note that the returned Tensor is squeezed.

        """
        if solutions.dim() == 1:
            solutions = solutions.unsqueeze(0)
        if solutions.dim() == 2:
            solutions = solutions.unsqueeze(0)

        m = solutions.size(1)

        solutions = solutions.to(dtype=torch.float32, device=self.device)
        solutions_expanded = solutions.unsqueeze(3).expand(-1, -1, -1, self._n)

        indices = self.interactions.nonzero(as_tuple=True)[-1].reshape(
            -1, self._n, self._k + 1
        )
        indices_expanded = indices.unsqueeze(1).expand(-1, m, -1, -1)
        contrib_ind = torch.matmul(
            torch.gather(solutions_expanded, 2, indices_expanded),
            self._powers_of_2,
        )

        contrib_expanded = self.fitness_contributions.unsqueeze(1).expand(-1, m, -1, -1)
        contrib_ind_expanded = contrib_ind.unsqueeze(3).expand(-1, -1, -1, 1)

        fitness = torch.mean(
            torch.gather(contrib_expanded, -1, contrib_ind_expanded.long()),
            dim=-2,
        )

        return fitness.squeeze()

    def sample(self, m: int = 1) -> torch.Tensor:
        r"""Get $m$ random solutions of $N$ dimensions.

        Returns
        -------
        torch.Tensor
            Matrix $S_{m \times N}$ of binary values (0 or 1).

            If the NKLand instance is a batch, returns multiple sample matrices with
            shape $(\text{num_instances}, N, N)$.

        """
        samples = torch.randint(
            0,
            2,
            (self._num_instances, m, self._n),
            generator=self._rng,
            dtype=torch.uint8,
            device=self.device,
        )
        return samples.squeeze(0)

    def save(self, file: Union[str, Path]) -> None:
        """Save the NKLand instance to a file in a format compatible with PyTorch.

        Parameters
        ----------
        file : Union[str, Path]
            The path where the instance will be saved.

        """
        torch.save(
            {
                "interactions": self.interactions.cpu(),
                "fitness_contributions": self.fitness_contributions.cpu(),
                "generator_state": self._rng.get_state(),
            },
            file,
        )

    @staticmethod
    def load(file: Union[str, Path], *, use_gpu: bool = False) -> NKLand:
        """Load a NKLand instance from a file written with the save method.

        Parameters
        ----------
        file : Union[str, Path]
            The path of the file to load.
        use_gpu : bool, optional
            Whether to use GPU acceleration. Default False.

        Returns
        -------
        NKLand
            The loaded NKLand instance.

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
        n: int,
        k: int,
        *,
        num_instances: int = 1,
        seed: Union[_Rng, None] = None,
        use_gpu: bool = False,
    ) -> NKLand:
        """Create a random NK landscape.

        Parameters
        ----------
        n : int
            Number of components, $N$.
        k : int
            Number of interactions per component, $K$.
        num_instances : int, optional
            Number of instances to generate for batch processing. Default is 1.
        seed : Union[_Rng, None], optional
            Seed or generator for random number generation.
        use_gpu : bool, optional
            Whether to use GPU acceleration. Default is False.

        Result
        ------
        NKLand
            A random NK landscape.

        Raises
        ------
        ValueError
            If `n` is greater than MAX_N, `k` is greater than MAX_K, or `k` is greater
            than `n - 1`.

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

        interactions = NKLand._generate_interactions(n, k, num_instances, rng, device)
        fitness_contributions = NKLand._generate_fitness_contributions(
            n, k, num_instances, rng, device
        )

        return NKLand(
            interactions=interactions,
            fitness_contributions=fitness_contributions,
            seed=rng,
            use_gpu=use_gpu,
        )

    @staticmethod
    def _generate_interactions(
        n: int, k: int, num_instances: int, rng: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        interactions = torch.empty((num_instances, n, n))
        for b in range(num_instances):
            interactions[b] = torch.eye(n, dtype=torch.int32, device=device)

            for i in range(n):
                possible_neighbors = torch.cat(
                    (
                        torch.arange(0, i),
                        torch.arange(i + 1, n),
                    )
                )
                selected_neighbors = possible_neighbors[
                    torch.randperm(n - 1, generator=rng)[:k]
                ]
                interactions[b, i, selected_neighbors] = 1

        return interactions.squeeze(0)

    @staticmethod
    def _generate_fitness_contributions(
        n: int, k: int, num_instances: int, rng: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        num_interactions = 2 ** (k + 1)
        fitness_contributions = torch.rand(
            (num_instances, n, num_interactions),
            generator=rng,
            dtype=torch.float64,
        ).to(device)

        return fitness_contributions.squeeze(0)
