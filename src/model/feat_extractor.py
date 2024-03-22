from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def build_grid(resolution: Sequence[int]) -> torch.Tensor:
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size: int, resolution: Sequence[int]):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj


class FeatureExtractor(torch.nn.Module):
    """
    Performs CNN-based feature extraction and ROI pooling.
    """

    def __init__(self, input_size: int = 128, num_frames: int = 4, num_objects: int = 32, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.num_frames = num_frames
        self.num_objects = num_objects
        self.input_size = input_size
        self.conv = nn.Sequential(
            nn.Conv2d(3 * num_frames, 64, (3, 3), (1, 1), 1),  # input_size x input_size
            nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, (3, 3), (1, 1), 1),  # input_size x input_size -> input_size/2 x input_size/2
            nn.ReLU(),
        )
        self.relu = torch.nn.ReLU()
        self.position_embed = SoftPositionEmbed(128, (64, 64))

    def roi_pool(self, x: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        For each RoI, extract a fixed-size feature map from x.
        Assume that there are at most num_objects objects in the image.
        Args:
            x: (B, C, input_size/2, input_size/2) input feature map tensor
            rois: (B, obj_number, H, W) Bool masks for each object
        """

        # compensate for conv size - (B, num_objects, input_size/2, input_size/2)
        rois = F.interpolate(rois.float(), size=(self.input_size // 2, self.input_size // 2), mode="nearest")

        # (B, num_objects, 128, input_size/2, input_size/2)
        masked = x.unsqueeze(1) * rois.unsqueeze(2)
        return masked.max(-1).values.max(-1).values  # (B, num_objects, 128)

    def forward(self, images: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, num_frames*3, input_size, input_size) input image tensor
        Returns:
            (B, num_objects, 128) feature vector
        """
        if self.debug:
            assert len(images.shape) == 4, f"Expected 4D tensor, got {images.shape}"
            assert images.shape[-1] == images.shape[-2] == self.input_size, f"Expected input size {self.input_size}, got {images.shape}"
        images = self.conv(images)  # [input_size/2, input_size/2]
        images = self.position_embed(images)
        objects = self.roi_pool(images, rois)
        return objects
