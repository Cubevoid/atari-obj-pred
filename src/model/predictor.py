import torch
from torch import nn
import torch.nn.functional as F
import wandb


class Predictor(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 32, output_size: int = 120, num_layers: int = 2,
                 hidden_dim: int = 120, embed_dim: int = 8, nhead: int = 2, time_steps: int = 5, log: bool = True, num_actions: int = 18) -> None:
        super().__init__()
        self.log = log
        self.time_steps = time_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dim_feedforward=hidden_dim)
        self.fc2 = nn.Linear(hidden_size, output_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.time_mlp = nn.Sequential(nn.Linear(output_size, output_size))
        self.pred_mlp = nn.Sequential(nn.Linear(output_size, output_size), nn.ReLU(), nn.Linear(output_size, 2))
        self.action_embedding = nn.Embedding(num_actions, embed_dim)
        self.embedding = nn.Sequential(nn.Linear(output_size+embed_dim, output_size), nn.ReLU())

    def forward(self, x: torch.Tensor, curr_pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:  # pylint: disable = unused-argument
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
        act_embed = self.action_embedding(actions) # [B, T, embed_dim]
        act_embed = act_embed.unsqueeze(-2)
        zeros = torch.zeros((x.size()[0], act_embed.size()[1], x.size()[1], act_embed.size()[2]), device=x.device) #[B, T, num_objects, embed_dim]
        act_embed = zeros + act_embed #[B, T, num_objects, embed_dim]
        for i in range(self.time_steps):
            x = torch.cat((x, act_embed[:, i, :, :]), dim = 2)
            x = self.embedding(x)
            x = self.transformer_encoder(x)  # [B, num_objects, output_size]
            x = self.time_mlp(x)  # [B, num_objects, output_size]
            debug_stats[f'pred_obj_std_mean_{i}'] = x.mean(-1).std()
            predictions.append(x)
        x = torch.stack(predictions, 1)  # [B, time_steps, num_objects, output_size]
        x = self.pred_mlp(x)  # [B, time_steps, num_objects, 2]
        if self.log:
            wandb.log(debug_stats)
        return x
