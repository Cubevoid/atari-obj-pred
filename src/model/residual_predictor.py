from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import wandb


class ResidualPredictor(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 32, output_size: int = 120, num_layers: int = 2,
                 hidden_dim: int = 120, nhead: int = 2, time_steps: int = 5, log: bool = True) -> None:
        super().__init__()
        self.log = log
        self.time_steps = time_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dim_feedforward=hidden_dim)
        self.fc2 = nn.Linear(hidden_size, output_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc3 = nn.Linear(output_size, output_size)
        self.fc4 = nn.Linear(output_size, 2)

    def forward(self, x: torch.Tensor, curr_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_objects, 128) feature vector
        Returns:
            (B, time, num_objects, 2) vector
        """
        debug_stats = {'obj_std_mean': x.mean(-1).std()}
        x = F.relu(self.fc1(x))  # [B, num_objects, hidden_size]
        x = self.fc2(x)  # [B, num_objects, output_size]
        predictions = []
        for i in range(self.time_steps - 1):
            x = self.transformer_encoder(x)  # [B, num_objects, output_size]
            debug_stats[f'pred_obj_std_mean_{i}'] = x.mean(-1).std()
            predictions.append(x)
        x = torch.stack(predictions, 1)  # [B, time_steps, num_objects, output_size]
        x = F.relu(self.fc3(x))  # [B, time_steps, num_objects, output_size]
        x = self.fc4(x)  # [B, time_steps, num_objects, 2]
        if self.log:
            wandb.log(debug_stats)

        new_output = torch.zeros((curr_pos.size()[0], self.time_steps, curr_pos.size()[1], 2), device=x.device)
        new_output[:, 0, :, :] = curr_pos
        for j in range(self.time_steps-1):
            new_output[:, j+1, :, :] = new_output[:, j, :, :] + x[:, j, :, :]
        return new_output