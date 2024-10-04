from __future__ import annotations

from typing import Union

import torch

Rng = Union[int, torch.Generator]


def default_rng(seed: Union[Rng, None] = None) -> torch.Generator:
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


def default_device(device: Union[torch.device, str, None] = None) -> torch.device:
    """Create a default torch.device (`cpu` or `cuda`).

    Parameters
    ----------
    device : Union[torch.device, str, None]
        Device to use.

        Note that it can be set to `None` to get the best available device.

    Returns
    -------
    torch.device
        The given `device` if provided, otherwise it returns `cuda` if possible
        or simply `cpu`.

    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
