import torch
from torch import nn
import torch.nn.functional as F

class FeatureExtractorBaseline(nn.Module):
    def __init__(self, input_size: int = 128, num_objects: int = 32, num_features: int = 128):
        super().__init__()
        self.input_size = input_size
        self.num_objects = num_objects
        self.fc1 = nn.Linear(2, num_features)

    def forward(self, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rois: (B, num_objects, input_size, input_size) input image tensor
        Returns:
            (B, num_objects, num_features) feature vector
        """
        x_indices = torch.arange(self.input_size, dtype=torch.float32).view(1, 1, -1, 1)
        y_indices = torch.arange(self.input_size, dtype=torch.float32).view(1, 1, 1, -1)

        sum_x = torch.sum(rois * x_indices, dim=(2, 3))
        sum_y = torch.sum(rois * y_indices, dim=(2, 3))

        mask_areas = torch.sum(rois, dim=(2, 3))
        mask_areas[mask_areas == 0] = 1

        average_x = sum_x / mask_areas
        average_y = sum_y / mask_areas

        # (B, num_objects, 2)
        average_xy = torch.stack((average_x, average_y), dim=-1)

        #output = F.relu(self.fc1(average_xy))
        return average_xy/128
