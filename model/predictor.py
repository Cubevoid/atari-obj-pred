import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 32, output_size: int = 120, num_layers=2,
                 hidden_dim=120, nhead=2, time_steps=5):
        super(Predictor, self).__init__()
        self.time_steps = time_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dim_feedforward=hidden_dim)
        self.fc2 = nn.Linear(hidden_size, output_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc3 = nn.Linear(output_size, output_size)
        self.fc4 = nn.Linear(output_size, 2)

    def forward(self, x, mask=None, src_key_padding_mask=None):
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
        for i in range(self.time_steps):
            x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)  # [B, num_objects, output_size]
            debug_stats[f'pred_obj_std_mean_{i}'] = x.mean(-1).std()
            predictions.append(x)
        x = torch.stack(predictions, 1)  # [B, time_steps, num_objects, output_size]
        x = F.relu(self.fc3(x))  # [B, time_steps, num_objects, output_size]
        x = self.fc4(x)  # [B, time_steps, num_objects, 2]
        wandb.log(debug_stats)
        return x
