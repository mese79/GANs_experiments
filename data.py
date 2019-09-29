import random
import numpy as np
import scipy.stats as ss
import torch
from torch.tensor import Tensor
from typing import List, Tuple

np.random.seed(13)
torch.random.manual_seed(13)
torch.cuda.manual_seed(13)


def sample_noise(num: int,
                 dim: int,
                 device: torch.device,
                 noise_type='gaussian',
                 noise_range=5) -> Tensor:
    if noise_type == 'gaussian':
        return torch.randn(num, dim, requires_grad=False, device=device)
    else:
        # torch.rand is uniform with interval of [0, 1)
        return torch.rand(num, dim, requires_grad=False, device=device) * noise_range


def sample_1d_data(num: int,
                   dim: int, 
                   device: torch.device,
                   mu: float = 2.0,
                   std: float = 0.75) -> Tensor:
    
    return torch.tensor(
        np.random.normal(mu, std, size=(num, dim)),
        dtype=torch.float,
        requires_grad=False,
        device=device
    )


def sample_8_gaussians(num: int, device: torch.device) -> Tensor:
    _scale: float = 3.
    _centers: List = [
        (1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(x * _scale, y * _scale) for x, y in _centers]

    data_batch: np.ndarray = np.zeros(shape=(num, 2))
    for i in range(num):
        point = np.random.randn(2) * .05
        center = random.choice(centers)
        point[0] += center[0]
        point[1] += center[1]
        data_batch[i] = point

    data_batch /= 1.414  # stdev

    return torch.tensor(data_batch, dtype=torch.float32, device=device, requires_grad=False)


def sample_25_gaussians(num: int, device: torch.device) -> Tensor:
    _scale: float = 2.
    _centers = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            _centers.append((x, y))

    data_batch: np.ndarray = np.zeros(shape=(num, 2))
    for i in range(num):
        point = np.random.randn(2) * .05
        x, y = random.choice(_centers)
        point[0] += _scale * x
        point[1] += _scale * y
        data_batch[i] = point

    data_batch /= 2.828  # stdev

    return torch.tensor(data_batch, dtype=torch.float32, device=device, requires_grad=False)


def sample_mixture(num: int, dim: int, device: torch.device) -> Tensor:
    """Return samples from a mixture of three Gaussian."""
    # mean and std of mixtures
    params = [[-4.0, 0.5], [1.0, 0.5], [5.0, 0.5]]
    samples = np.zeros((num, dim))
    for i in range(num):
        mu, std = random.choice(params)
        samples[i] = np.random.normal(mu, std)

    return torch.tensor(samples, dtype=torch.float32, device=device, requires_grad=False), params


def get_mixture_params():
    params = [[-4.0, 0.5], [1.0, 0.5], [5.0, 0.5]]
    return params
