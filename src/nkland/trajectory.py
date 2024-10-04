from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from nkland.landscape import NKLand
    from nkland.utils import Rng

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Data class to store a sequence of solutions with their associated fitness."""

    strategy_name: str
    solutions: list[torch.Tensor]
    fitness: list[torch.Tensor]

    def last_solution(self) -> torch.Tensor:
        """Get the last solution of the Trajectory."""
        if len(self.solutions) == 0:
            msg = "cannot get last solution of empty trajectory"
            raise ValueError(msg)
        return self.solutions[-1]

    def last_fitness(self) -> torch.Tensor:
        """Get the last fitness of the Trajectory."""
        if len(self.fitness) == 0:
            msg = "cannot get last fitness of empty trajectory"
            raise ValueError(msg)
        return self.fitness[-1]

    def append_point(self, solution: torch.Tensor, fitness: torch.Tensor) -> Trajectory:
        """Add a new solution.

        Parameters
        ----------
        solution : torch.Tensor
            the solution to add

        fitness : torch.Tensor
            the fitness value of the solution (it should be a scalar).

        """
        self.solutions.append(solution)
        self.fitness.append(torch.as_tensor(fitness))
        return self

    def get_solutions(self) -> torch.Tensor:
        """Get a torch.Tensor with all the solutions stacked."""
        return torch.stack(self.solutions)

    def get_fitness(self) -> torch.Tensor:
        """Get a torch.Tensor with all the fitness stacked."""
        return torch.stack(self.fitness)

    def get_num_points(self) -> int:
        """Get the number of visited solutions in the `nkland.Trajectory`."""
        return len(self.fitness)

    def get_best_solution(self) -> torch.Tensor:
        """Get the best visited solution (with the greatest fitness)."""
        ind = torch.argmax(self.get_fitness())
        return self.solutions[ind]

    def get_best_fitness(self) -> torch.Tensor:
        """Get the greatest fitness among visited solutions."""
        return torch.max(self.get_fitness())

    def get_dimensions(self) -> Optional[str]:
        """Get a string describing the dimensions."""
        if len(self.fitness) == 0:
            return None
        return str(self.fitness[0].size())

    def save(self, filepath: Union[Path, str]) -> None:
        """Save a Trajectory to disk."""
        torch.save(
            obj={
                "strategy_name": self.strategy_name,
                "solutions": self.solutions,
                "fitness": self.fitness,
            },
            f=filepath,
        )

    @staticmethod
    def load(filepath: Union[Path, str]) -> Optional[Trajectory]:
        """Load a nkland.Trajectory from a file on disk."""
        try:
            obj = torch.load(filepath)
        except Exception:
            msg = f"Cannot load trajectory from file at {filepath}"
            logger.exception(msg)
            return None
        return Trajectory(
            strategy_name=obj["strategy_name"],
            solutions=obj["solutions"],
            fitness=obj["fitness"],
        )

    @staticmethod
    def create(
        strategy_name: str,
        landscape: NKLand,
        *,
        seed: Union[Rng, None] = None,
    ) -> Trajectory:
        """Create a new nkland.Trajectory with a random starting solution."""
        solution0 = landscape.sample(seed=seed)
        fitness0 = landscape.evaluate(solution0)
        return Trajectory(
            strategy_name=strategy_name,
            solutions=[solution0],
            fitness=[torch.as_tensor(fitness0)],
        )
