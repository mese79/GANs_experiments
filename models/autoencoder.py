import torch
from torch import nn
import torch.nn.functional as F


class BasicAutoencoder(nn.Module):
	def __init__(self, input_dim, feature_dim):
		super().__init__()
		# encoder
		self.fc1 = nn.Linear(input_dim, feature_dim * 2)
		self.fc2 = nn.Linear(feature_dim * 2, feature_dim)

		# decoder
		self.fc3 = nn.Linear(feature_dim, feature_dim * 2)
		self.fc4 = nn.Linear(feature_dim * 2, input_dim)

	def forward(self, x):
		# encode
		out = self.fc1(x)
		out = F.leaky_relu(out)
		out = self.fc2(out)
		feature = F.leaky_relu(out)
		# decode
		out = self.fc3(feature)
		out = F.leaky_relu(out)
		out = self.fc4(out)
		out = torch.tanh(out)  # input img is in range of [-1, 1] so the output.

		return out, feature

