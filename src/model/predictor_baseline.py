import torch
from torch import nn
import torch.nn.functional as F

class PredictorBaseline(nn.Module):
    def __init__(self, input_size: int = 128, time_steps: int = 5):
        super().__init__()
        self.time_steps = time_steps
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, 2)

    def forward(self, x):
        predictions = []
        for _ in range(self.time_steps):
            predictions.append(F.relu(self.fc1(x)))
        x = torch.stack(predictions, 1)
        x = self.fc2(x)
        return x
