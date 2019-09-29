import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.tensor import Tensor


class Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l3 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out


class Generator8G(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l3 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        
        return out


class Generator25G(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l4 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
    
        return out


class GeneratorMixture(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(z_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        return self.l4(out)


class GeneratorMNIST(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1[0].out_features, self.fc1[0].out_features * 2),
            nn.ELU(alpha=0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features, self.fc2[0].out_features * 2),
            nn.ELU(alpha=0.2)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features * 2, data_dim),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
    
        return out