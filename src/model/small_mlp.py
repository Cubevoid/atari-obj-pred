import torch
from torch import nn
import torch.nn.functional as F

class SmallMLP(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 32, output_size: int = 2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        return output
