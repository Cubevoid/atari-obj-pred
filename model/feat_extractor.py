import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatExtractor(torch.nn.Module):
    """
    Performs CNN-based feature extraction and ROI pooling.
    """
    def __init__(self, input_size: int = 128, num_frames: int = 4, num_objects: int = 32):
        super().__init__()
        self.num_frames = num_frames
        self.num_objects = num_objects
        self.input_size = input_size
        self.conv = nn.Sequential(
            [
                nn.Conv2d(3 * num_frames, 64, 3, 1, 1),  # input_size x input_size
                nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(64, 128, 3, 1, 1),  # input_size x input_size -> input_size/2 x input_size/2
                nn.ReLU()
            ]
        )
        self.relu = torch.nn.ReLU()

    def roi_pool(self, x: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        For each RoI, extract a fixed-size feature map from x.
        Assume that there are at most num_objects objects in the image.
        Args:
            x: (B, C, input_size/2, input_size/2) input feature map tensor
            rois: (B, input_size, input_size) uint8 ids of the RoIs in the image
        """
        # (B, 32, input_size, input_size) tensor where each channel is a mask for a RoI
        rois_one_hot = F.one_hot(rois.long(), num_classes=self.num_objects).permute(0, 3, 1, 2)

        # compensate for conv size - (B, num_objects, input_size/2, input_size/2)
        rois = F.interpolate(rois_one_hot.float(),
                             size=(self.input_size//2, self.input_size//2),
                             mode='nearest')

        # (B, num_objects, 128, input_size/2, input_size/2)
        masked = x.unsqueeze(1) * rois.unsqueeze(2)
        return masked.max(-2).values.max(-1).values  # (B, num_objects, 128)


    def forward(self, images: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, num_frames, 3, input_size, input_size) input image tensor
        Returns:
            (B, num_objects, 128) feature vectors
        """
        images = self.conv(images)  # [input_size/2, input_size/2]
        images = self.roi_pool(images, rois)
        return images
