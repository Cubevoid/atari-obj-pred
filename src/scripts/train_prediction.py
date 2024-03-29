import typing
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from torch import nn
import hydra
import wandb

import os
os.environ["WANDB_MODE"] = "dryrun"

from src.data_collection.data_loader import DataLoader
from src.model.feat_extractor import FeatureExtractor
from src.model.predictor import Predictor
from src.model.predictor_baseline import PredictorBaseline


@hydra.main(version_base=None, config_path="../../configs/training", config_name="config")
def train(config: DictConfig, batch_size: int = 4, t_steps: int = 4, num_obj: int = 4, name: str = 'debug') -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_loader = DataLoader("SimpleTestData", num_obj)

    feature_extract = FeatureExtractor(num_objects=num_obj).to(device)
    predictor = Predictor(num_layers=1, time_steps=t_steps).to(device)
    predictor_baseline = Predictor(time_steps=t_steps).to(device)

    wandb.init(project="oc-data-collection", entity="atari-obj-pred", name=name, config=typing.cast(Dict[Any, Any], OmegaConf.to_container(config)))
    wandb.log({"batch_size": batch_size})
    wandb.watch(feature_extract, log="all", log_freq=1, idx=1)
    wandb.watch(predictor, log="all", log_freq=1, idx=2)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(list(feature_extract.parameters()) + list(predictor.parameters()), lr=1e-4)
    #optimizer2 = torch.optim.Adam(list(feature_extract.parameters()) + list(predictor_baseline.parameters()), lr=1e-4)

    images, bboxes, masks, _ = data_loader.sample(batch_size, t_steps)
    images, bboxes, masks = images.to(device), bboxes.to(device), masks.to(device)
    target = bboxes[:,:,:,:2]  # [B, T, O, 2]

    #images = torch.rand((batch_size, 3*t_steps, 128, 128))
    #masks = torch.rand((batch_size, num_obj, 128, 128)) > 0.5
    #target = torch.rand((batch_size, t_steps, num_obj, 2))

    for _ in tqdm(range(100)):
        features: torch.Tensor = feature_extract(images, masks)
        output: torch.Tensor = predictor(features)
        #output: torch.Tensor = predictor_baseline(features)
        loss: torch.Tensor = criterion(output, target)
        loss.backward()
        optimizer.step()
        #optimizer2.step()
        tqdm.write(f"{loss.item()=}, {output.mean().item()=}, {output.std().item()=}")
        wandb.log({"loss": loss})
        optimizer.zero_grad()
        #optimizer2.zero_grad()



    print(target)
    print(output)

if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
