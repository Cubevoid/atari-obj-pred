import torch

from src.model.predictor import Predictor


class ResidualPredictor(Predictor):
    def forward(self, x: torch.Tensor, curr_pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_objects, 128) feature vector
            curr_pos: (B, num_objects, 2)
            actions: (B, T)
        Returns:
            (B, time, num_objects, 2) vector
        """
        x = super().forward(x, curr_pos, actions)
        output = torch.zeros((curr_pos.shape[0], self.time_steps, curr_pos.shape[1], 2), device=x.device)
        output[:, 0, :, :] = curr_pos
        for j in range(self.time_steps-1):
            output[:, j+1, :, :] = output[:, j, :, :] + x[:, j, :, :]
        return output
