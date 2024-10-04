from __future__ import annotations

import math
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
            $(\text{additional dims}, N, N)$.
        fitness_contributions : torch.Tensor
            The matrix $C_{N \times 2^{K+1}}$ containing the contributions to the
            fitness.

            It can be a batch of adjacencies matrix with shape
            $(\text{additional dims}, N, 2^{k+1})$.
        seed : Union[_Rng, None], optional
            Seed or generator for random number generation. Default is None.
        use_gpu : bool, optional
            Whether to use GPU acceleration. Default is False.

        """
        if interactions.dim() < 2:
            msg = (
                "interactions tensor must have at least 2 dimensions, "
                f"but got: {interactions.dim()}"
            )
            raise ValueError(msg)
        if fitness_contributions.dim() < 2:
            msg = (
                "fitness contributions tensor must have at least 2 dimensions, "
                f"but got: {fitness_contributions.dim()}"
            )
            raise ValueError(msg)
        if interactions.dim() != fitness_contributions.dim():
            msg = (
                "`interactions` and `fitness_contributions` must have same number of "
                "dimensions, but got: "
                f"{interactions.size(0)} != {fitness_contributions.size(0)}"
            )
            raise ValueError(msg)
        if interactions.size()[:-2] != fitness_contributions.size()[:-2]:
            msg = (
                "additional dimensions (>2) of `interactions` and "
                "`fitness_contributions` must have same sizes, but got: "
                f"{interactions.size()[:-2]} != {fitness_contributions.size()[:-2]}"
            )
            raise ValueError(msg)
        if interactions.size(-1) != interactions.size(-2):
            msg = (
                "`interactions` has bad shape, last 2 dimensions (n) must be equal, "
                f"but got: {interactions.shape}"
            )
            raise ValueError(msg)
        if interactions.size(-2) != fitness_contributions.size(-2):
            msg = (
                "penultimate dimension (n) of `interactions` and `fitness_contribution`"
                " do not match: "
                f"{interactions.size(-2)} != {fitness_contributions.size(-2)}"
            )
            raise ValueError(msg)
        if not _is_power_of_2(fitness_contributions.size(-1)):
            msg = (
                "`fitness_contributions` last dimension (2**(k+1)) must be a power of 2"
                f", but got: {fitness_contributions.size(1)}"
            )
            raise ValueError(msg)

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        self._rng = default_rng(seed)

        self.interactions = interactions.to(dtype=torch.uint8, device=self.device)
        self.fitness_contributions = fitness_contributions.to(
            dtype=torch.float64, device=self.device
        )

        self._additional_dims = tuple(self.interactions.size()[:-2])
        self._n = self.interactions.size(-1)
        self._k = (
            int(self.interactions[(*[0] * len(self._additional_dims), 0)].sum()) - 1
        )

        self._powers_of_2 = 2 ** torch.arange(
            self._k, -1, -1, dtype=torch.float32, device=self.device
        )

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
        if solutions.ndimension() == 1:
            solutions = solutions.unsqueeze(0)

        if solutions.size(-1) != self._n:
            msg = (
                "`solutions` tensor last dimension must have n={self._n} elements, "
                f"but got {solutions.size(-1)}"
            )
            raise ValueError(msg)
        if solutions.size()[:-2] != self._additional_dims:
            msg = (
                "`solutions` tensor must have the same additional dimensions as "
                "`interactions`: "
                f"got {tuple(solutions.size()[:-2])} "
                f"but expected {self._additional_dims}"
            )
            raise ValueError(msg)

        m = solutions.size(-2)

        solutions = solutions.to(dtype=torch.float32, device=self.device)
        solutions_expanded = solutions.unsqueeze(-1).expand(*solutions.shape, self._n)

        indices = self.interactions.nonzero(as_tuple=True)[-1].reshape(
            *self._additional_dims,
            self._n,
            self._k + 1,
        )
        indices_expanded = indices.unsqueeze(-3).expand(
            *self._additional_dims, m, -1, -1
        )

        contrib_ind = torch.matmul(
            torch.gather(solutions_expanded, -2, indices_expanded),
            self._powers_of_2,
        )
        contrib_ind_expanded = contrib_ind.unsqueeze(-1).expand(*contrib_ind.shape, 1)

        contrib = self.fitness_contributions
        contrib_expanded = contrib.unsqueeze(-3).expand(
            *self._additional_dims, m, -1, -1
        )

        fitness = torch.mean(
            torch.gather(contrib_expanded, -1, contrib_ind_expanded.long()),
            dim=-2,
        )

        # squeeze last 2 dims
        return fitness.squeeze(-2).squeeze(-1)

    def sample(self, m: int = 1) -> torch.Tensor:
        r"""Get $m$ random solutions of $N$ dimensions.

        Returns
        -------
        torch.Tensor
            Matrix $S_{m \times N}$ of binary values (0 or 1).

            If the NKLand instance is a batch, returns multiple sample matrices with
            shape $(\text{num_instances}, N, N)$.

        """
        return torch.randint(
            0,
            2,
            (*self._additional_dims, m, self._n),
            generator=self._rng,
            dtype=torch.uint8,
            device=self.device,
        )

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
        additional_dims: tuple[int, ...] = (),
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
        additional_dims : tuple[int, ...], optional
            Tuple describing the sizes of additional shapes to use batch processing.
            Default is ().
        seed : Union[_Rng, None], optional
            Seed or generator for random number generation.
        use_gpu : bool, optional
            Whether to use GPU acceleration. Default is False.

        Returns
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
        rng = default_rng(seed)

        interactions = NKLand._generate_interactions(n, k, additional_dims, rng, device)
        fitness_contributions = NKLand._generate_fitness_contributions(
            n, k, additional_dims, rng, device
        )

        return NKLand(
            interactions=interactions,
            fitness_contributions=fitness_contributions,
            seed=rng,
            use_gpu=use_gpu,
        )

    @staticmethod
    def _generate_interactions(
        n: int,
        k: int,
        additional_dims: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        shape: tuple[int, ...] = (n, n)
        if len(additional_dims) > 0:
            shape = additional_dims + shape

        interactions = torch.empty(shape, dtype=torch.uint8, device=device)

        num_interactions = math.prod(additional_dims) if len(additional_dims) else 1

        for idx in range(num_interactions):
            index = torch.unravel_index(torch.tensor(idx), additional_dims)
            interactions[index] = torch.eye(n, dtype=torch.uint8, device=device)

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
                interactions[index][i, selected_neighbors] = 1

        return interactions

    @staticmethod
    def _generate_fitness_contributions(
        n: int,
        k: int,
        additional_dims: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        num_interactions = 2 ** (k + 1)
        shape: tuple[int, ...] = (n, num_interactions)
        if len(additional_dims) > 0:
            shape = additional_dims + shape
        return torch.rand(
            shape,
            generator=rng,
            dtype=torch.float64,
            device=device,
        )
