import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import Callable
class ConvNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Callable) -> None:
        super(ConvNet, self).__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
