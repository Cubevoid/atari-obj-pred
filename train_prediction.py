import torch
import torch as torch
import torch.nn.functional as F
import torch.nn as nn
from model.feat_extractor import FeatExtractor
from model.predictor import Predictor


if __name__ == "__main__":
    images = (256* torch.rand((32, 4, 3, 128, 128))).int()
    objects = (32*torch.rand((32, 128, 128))).int()
    feat_extract = FeatExtractor()
    predictor = Predictor()
    target = torch.rand((32, 5, 32, 2))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(feat_extract.parameters()) + list(predictor.parameters()), lr=7e-5)
    for i in range(20):
        output = feat_extract.forward(images, objects)
        output = predictor.forward(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(loss)
        optimizer.zero_grad()

    print(output)


