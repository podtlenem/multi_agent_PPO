import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NeuralNet(nn.Module):
	def __init__(self, in_dim, out_dim,hidden_dim = 64,device = "cpu"):
		super(NeuralNet, self).__init__()

		self.layer1 = nn.Linear(in_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, hidden_dim)
		self.layer3 = nn.Linear(hidden_dim, out_dim)
		self.device = device

	def forward(self, obs:torch.Tensor):
		x = F.relu(self.layer1(obs))
		x = F.relu(self.layer2(x))
		x = self.layer3(x)
		return x