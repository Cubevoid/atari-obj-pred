import torch
from torch import nn

class PredictorBaseline(nn.Module):
    def __init__(self, input_size: int = 128, time_steps: int = 5):
        super().__init__()
        self.time_steps = time_steps
        self.encoder = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size))
        self.next_state = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size))
        self.output = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, 2))

    def forward(self, x: torch.Tensor, curr_pos: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        predictions = []
        for _ in range(self.time_steps):
            z = self.next_state(z)
            predictions.append(z)
        stacked_predictions = torch.stack(predictions, 1)
        movements = self.output(stacked_predictions)
        outputs = torch.zeros((curr_pos.shape[0], self.time_steps, curr_pos.shape[1], 2), device=x.device)
        outputs[:, 0, :, :] = curr_pos
        for j in range(self.time_steps-1):
            outputs[:, j+1, :, :] = outputs[:, j, :, :] + movements[:, j, :, :]
        return outputs
