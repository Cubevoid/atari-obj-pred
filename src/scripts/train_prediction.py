import os
import time
import typing
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from torch import nn
import hydra
from hydra.utils import to_absolute_path
import wandb

from src.data_collection.data_loader import DataLoader
from src.model.feat_extractor import FeatureExtractor
from src.model.predictor import Predictor


@hydra.main(version_base=None, config_path="../../configs/training", config_name="config")
def train(config: DictConfig) -> None:
    batch_size = config.batch_size
    time_steps = config.time_steps
    game = config.game
    name = config.name
    num_obj = config.num_objects

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_loader = DataLoader(game, num_obj)

    feature_extract = FeatureExtractor(num_objects=num_obj).to(device)
    predictor = Predictor(num_layers=1, time_steps=time_steps).to(device)

    wandb.init(project="oc-data-training", entity="atari-obj-pred", name=name + game, config=typing.cast(Dict[Any, Any], OmegaConf.to_container(config)))
    wandb.log({"batch_size": batch_size})
    wandb.watch(feature_extract, log="gradients", log_freq=100, idx=1)
    wandb.watch(predictor, log="gradients", log_freq=100, idx=2)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(list(feature_extract.parameters()) + list(predictor.parameters()), lr=1e-3)

    for i in tqdm(range(1000)):
        images, bboxes, masks, _ = data_loader.sample(batch_size, time_steps)
        images, bboxes, masks = images.to(device), bboxes.to(device), masks.to(device)
        target = bboxes[:,:,:,:2]  # [B, T, O, 2]

        features: torch.Tensor = feature_extract(images, masks)
        output: torch.Tensor = predictor(features)
        loss: torch.Tensor = criterion(output, target)
        loss.backward()
        diff = torch.pow(output - target, 2)
        optimizer.step()
        tqdm.write(f"loss={loss.item()}, output_mean={output.mean().item()}, std={output.std().item()}")
        tqdm.write(f"target_mean={target.mean().item()} std={target.std().item()}")
        error_dict = {"loss": loss,
                    "error/x": diff[:,:,:,0].mean(),
                    "error/y": diff[:,:,:,1].mean()
        }
        for t in range(time_steps):
            error_dict[f"error/time_{t}"] = diff[:, t, :, :].mean()
        if i % 50 == 0:
            notzero = torch.nonzero(target)
            l1sum = 0
            total = 0
            for index in notzero:
                l1sum += abs(target[index[0]][index[1]][index[2]][index[3]]-output[index[0]][index[1]][index[2]][index[3]])
                total += 1
            print(l1sum/total)
        wandb.log(error_dict)
        optimizer.zero_grad()

    # save trained model to disk
    unix_time = int(time.time())
    os.makedirs(to_absolute_path(f"./models/trained/{game}"), exist_ok=True)
    torch.save(feature_extract.state_dict(), to_absolute_path(f"./models/trained/{game}/{unix_time}_feat_extract.pth"))
    torch.save(predictor.state_dict(), to_absolute_path(f"./models/trained/{game}/{unix_time}_predictor.pth"))

if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
