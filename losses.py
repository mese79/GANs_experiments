from typing import List, Dict, Tuple, Any

import torch
from torch import Tensor


def log(x: Tensor) -> Tensor:
    """custom log function to prevent log of zero(infinity/NaN) problem."""
    return torch.log(torch.max(x, torch.tensor(1e-6).to(x.device)))


def standard_d_loss(real_score: Tensor, fake_score: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
	# -log[d(x)]
	real_part = 0.5 * torch.mean(-log(real_score))

	# -log[1 - d(g(z))]
	fake_part = 0.5 * torch.mean(-log(1.0 - fake_score))

	loss = real_part + fake_part

	return loss, real_part, fake_part


def standard_g_loss(fake_score: Tensor, real_score: Tensor = None) -> Tensor:
	return 1.0 * torch.mean(log(1 - fake_score))


def heuristic_g_loss(fake_score: Tensor) -> Tensor:
	return 1.0 * torch.mean(-log(fake_score))