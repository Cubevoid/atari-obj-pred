import os
import time
import typing
from typing import Any, Dict

import numpy as np
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
from src.model.mlp_predictor import MLPPredictor
from src.model.feat_extractor_baseline import FeatureExtractorBaseline
from src.model.small_mlp import SmallMLP


@hydra.main(version_base=None, config_path="../../configs/training", config_name="config")
def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mlp = cfg.predictor == "mlp"

    data_loader = DataLoader(cfg.game, cfg.num_objects)
    feature_extract = FeatureExtractor(num_objects=cfg.num_objects).to(device)
    mean = FeatureExtractorBaseline(num_objects=cfg.num_objects, device=device).to(device)
    predictor = (MLPPredictor() if use_mlp else Predictor(num_layers=1, time_steps=cfg.time_steps)).to(device)
    small_mlp = SmallMLP().to(device)

    wandb.init(project="oc-data-training", entity="atari-obj-pred", name=cfg.name + cfg.game,
               config=typing.cast(Dict[Any, Any], OmegaConf.to_container(cfg)))
    wandb.log({"batch_size": cfg.batch_size})
    wandb.watch(feature_extract, log=None, log_freq=100, idx=1)
    wandb.watch(predictor, log=None, log_freq=100, idx=2)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(list(feature_extract.parameters()) + list(predictor.parameters()), lr=1e-3)

    for i in tqdm(range(cfg.num_iterations)):
        images, bboxes, masks, _ = data_loader.sample(cfg.batch_size, cfg.time_steps, device)
        if cfg.ground_truth_masks:
            bbox_ints = bboxes * 128
            bbox_ints = bbox_ints.int()
            masks = torch.zeros(masks.size(), device=device)
            for j in range(len(masks)):
                for k in range(len(masks[j])):
                    masks[j, k, int(bbox_ints[j][0][k][0]): int(bbox_ints[j][0][k][0] + bbox_ints[j][0][k][2]), int(bbox_ints[j][0][k][1]): int(bbox_ints[j][0][k][1] + bbox_ints[j][0][k][3])] = 1

        target = bboxes[:, :, :, :2]  # [B, T, O, 2]

        # Run models
        features: torch.Tensor = feature_extract(images, masks)
        output: torch.Tensor = predictor(features)
        # target: torch.Tensor = mean(masks)
        # target = target.unsqueeze(1).repeat((1, 5, 1, 1))
        loss: torch.Tensor = criterion(output, target)
        # loss = torch.pow((output - target), 4).mean()
        loss.backward()
        optimizer.step()
        diff = torch.pow(output - target, 2)
        max_loss = torch.max(torch.abs((output - target)))
        total_movement = torch.sum(torch.abs((target[:, cfg.time_steps-1, :, :] - target[:, 0, :, :])))
        movement_mask = target[:, cfg.time_steps-1, :, :] - target[:, 0, :, :] != 0
        average_movement = total_movement / torch.sum(movement_mask)
        nonzero_feats = features[features.sum(dim=-1) > 0]
        std, corr = nonzero_feats.std(dim=0), torch.corrcoef(nonzero_feats)

        l1_average_with_movement = torch.sum(torch.abs(torch.squeeze(target[:, cfg.time_steps-1, :, :], dim=1)[movement_mask] -
                                                       torch.squeeze(output[:, cfg.time_steps-1, :, :], dim= 1)[movement_mask]))
        l1_average_with_movement /= torch.sum(movement_mask)

        if cfg.debug and i % 50 == 0:
            mask = target != 0

            l1sum = torch.sum(torch.abs(target[mask] - output[mask]))
            med_l1 = torch.median(torch.abs(target[mask]-output[mask]))
            total = torch.sum(mask)
            tqdm.write(f"loss={loss.item()}, output_mean={output.mean().item()}, std={output.std().item()}")
            tqdm.write(f"target_mean={target.mean().item()} std={target.std().item()}")
            tqdm.write(f"l1 average loss = {l1sum / total}")
            tqdm.write(f"maximum loss = {max_loss.item()}")
            tqdm.write(f"median loss = {med_l1}")
            tqdm.write(f"average movement = {average_movement.item()}")
            tqdm.write(f"average l1 loss on last time step where object moved = {l1_average_with_movement.item()}")
            # tqdm.write(f"Predicted: {output[:,:,0]}, Target: {target[:,:,0]}")
            # tqdm.write(f"Std: {std} {std.shape}")
            # tqdm.write(f"Corr: {corr} {corr.shape}")
        error_dict = {"loss": loss, "error/x": diff[:, :, :, 0].mean(), "error/y": diff[:, :, :, 1].mean()}
        error_dict |= {"std_mean": std.mean(), "std_std": std.std(), "corr_mean": corr.mean(), "corr_std": corr.std()}
        error_dict |= {"max_loss": max_loss, "average_movement": average_movement, "l1move": l1_average_with_movement}
        for t in range(cfg.time_steps):
            error_dict[f"error/time_{t}"] = diff[:, t, :, :].mean()
        error_dict["l1average"] = l1sum / total
        wandb.log(error_dict)
        optimizer.zero_grad()

    # save trained model to disk
    unix_time = int(time.time())
    os.makedirs(to_absolute_path(f"./models/trained/{cfg.game}"), exist_ok=True)
    torch.save(feature_extract.state_dict(),
               to_absolute_path(f"./models/trained/{cfg.game}/{unix_time}_feat_extract.pth"))
    model_name = "mlp_predictor" if use_mlp else "transformer_predictor"
    torch.save(predictor.state_dict(), to_absolute_path(f"./models/trained/{cfg.game}/{unix_time}_{model_name}.pth"))


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
