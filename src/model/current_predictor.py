from typing import Optional
import torch
from torch import nn


class CurrentPredictor(nn.Module):
    """
    A simple MLP-based predictor which takes in a feature vector and predicts the current location for the object.
    This is essentially just a decoder for the CNN feature vectors.
    """
    def __init__(self, input_size: int = 128, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, num_objects, 128) feature vector
        Returns:
            (B, time=1, num_objects, 2) vector
        """
        # pylint: disable=unused-argument
        return self.fc2(self.relu(self.fc1(x))).unsqueeze(1)
