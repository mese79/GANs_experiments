from typing import Tuple

import torch
from torch import Tensor

def log(x: Tensor) -> Tensor:
    """custom log function to prevent log of zero(infinity/NaN) problem."""
    return torch.log(torch.max(x, torch.tensor(1e-6).to(x.device)))


# Least Squares Generative Adversarial Networks: https://arxiv.org/abs/1611.04076
def least_square_d_loss(real_score: Tensor, fake_score: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
	""""
	Least Squares GAN - Discriminator loss
	(set a=−1,b=1,c=0 for Pearson setup)
	"""
	# b = 1
	real_part = 0.5 * torch.mean((real_score - 1)**2)

	# a = 0
	fake_part = 0.5 * torch.mean(fake_score ** 2)

	loss = real_part + fake_part

	return loss, real_part, fake_part


def least_square_g_loss(fake_score: Tensor, real_score: Tensor = None) -> Tensor:
	""""
	Least Squares GAN - Generator loss
	"""
	# c = 1
	return 0.5 * torch.mean((fake_score - 1) ** 2)


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