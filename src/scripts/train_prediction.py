from tqdm import tqdm
import torch
# import torch.nn.functional as F
from torch import nn
import wandb

from src.data_collection.data_loader import DataLoader
from src.model.feat_extractor import FeatExtractor
from src.model.predictor import Predictor


def train(device: torch.device = torch.device("cpu"), criterion: nn.Module = nn.MSELoss(), game: str = "Pong", batch_size: int = 1, t_steps: int = 1, num_obj: int = 32) -> None:
    wandb.init(project="oc-data-collection", entity="atari-obj-pred", name="debug")
    print(f"Using device: {device}")
    data_loader = DataLoader(game)
    wandb.log({"batch_size": batch_size})
    feat_extract = FeatExtractor(num_objects=num_obj).to(device)
    # wandb.watch(feat_extract, log="all", log_freq=1, idx=1)
    predictor = Predictor(num_layers=1, time_steps=t_steps).to(device)
    # wandb.watch(predictor, log="all", log_freq=1, idx=2)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(
        list(feat_extract.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    images, bboxes, masks, _ = data_loader.sample(batch_size, t_steps)
    target = bboxes[:,:,:2]

    for _ in tqdm(range(1000)):
        features: torch.Tensor = feat_extract(images, masks)
        output: torch.Tensor = predictor(features)
        loss: torch.Tensor = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(loss.item(), output.mean().item(), output.std().item())
        wandb.log({"loss": loss})
        optimizer.zero_grad()

    print(target)
    print(output)

def main() -> None:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    train(device, game="SimpleTestData")

if __name__ == "__main__":
    main()
