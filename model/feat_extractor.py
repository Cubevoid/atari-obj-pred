import torch

class FeatExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatExtractor, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) input image tensor
        """
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        return x.flatten(1)