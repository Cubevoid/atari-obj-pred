import torch
from torch import nn
import torch.nn.functional as F

class PredictorBaseline(nn.Module):
    def __init__(self, input_size: int = 128, time_steps: int = 5, embed_dim: int = 8):
        super().__init__()
        self.time_steps = time_steps
        self.encoder = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size))
        self.next_state = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size))
        self.output = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, 2))
        self.action_embedding = nn.Embedding(16, embed_dim)
        self.embedding = nn.Linear(input_size+embed_dim, input_size)

    def forward(self, x: torch.Tensor, curr_pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        act_embed = self.action_embedding(actions) # [B, T, embed_dim]
        act_embed = act_embed.unsqueeze(-1)
        zeros = torch.zeros((x.size()[0], act_embed.size()[1], x.size()[1], act_embed.size()[2])) #[B, T, num_objects, embed_dim]
        act_embed = zeros + act_embed #[B, T, num_objects, embed_dim]
        predictions = []
        for i in range(self.time_steps):
            x = torch.cat((x, act_embed[:, i, :, :]), dim = 2)
            x = F.relu(self.embedding(x))
            z = self.next_state(z)
            predictions.append(z)
        stacked_predictions = torch.stack(predictions, 1)
        movements = self.output(stacked_predictions)
        outputs = torch.zeros((curr_pos.shape[0], self.time_steps, curr_pos.shape[1], 2), device=x.device)
        outputs[:, 0, :, :] = curr_pos
        for j in range(self.time_steps-1):
            outputs[:, j+1, :, :] = outputs[:, j, :, :] + movements[:, j, :, :]
        return outputs
