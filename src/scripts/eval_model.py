import os
import time
import typing
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from torch import nn
import hydra
from hydra.utils import to_absolute_path, instantiate
import wandb

from src.scripts.train_prediction import get_ground_truth_masks
from src.data_collection.data_loader import DataLoader


@hydra.main(version_base=None, config_path="../../configs/training", config_name="config")
def eval(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader = instantiate(cfg.data_loader, model=cfg.model, game=cfg.game, num_obj=cfg.num_objects, val_pct=0, test_pct=0.3)

    t = 1713391908
    feature_extractor_state = torch.load(f"models/trained/{cfg.game}/{t}_feat_extract.pth", map_location=device)
    feature_extractor = instantiate(cfg.feature_extractor, num_objects=cfg.num_objects, history_len=cfg.data_loader.history_len)
    feature_extractor.load_state_dict(feature_extractor_state)
    predictor = instantiate(cfg.predictor, time_steps=cfg.time_steps, log=False)
    predictor_state = torch.load(f"models/trained/{cfg.game}/{t}_{type(predictor).__name__}.pth", map_location=device)
    predictor.load_state_dict(predictor_state)

    criterion = nn.MSELoss().to(device)

    mean = []
    med = []
    ninetieth = []

    for i in tqdm(range(data_loader.num_train + data_loader.num_val, len(data_loader.frames), cfg.batch_size)):
        images, bboxes, masks, actions = data_loader.sample_idxes(cfg.time_steps, device, range(i, min(i + cfg.batch_size, len(data_loader.frames)-cfg.time_steps)))
        if cfg.ground_truth_masks:
            masks = get_ground_truth_masks(bboxes, masks.shape, device=device)

        positions = bboxes[:, :, :, :2]  # [B, H + T, O, 2]
        target = positions[:, cfg.data_loader.history_len :, :, :]  # [B, T, O, 2]
        gt_positions = positions[:, : cfg.data_loader.history_len, :, :]  # [B, H, O, 2]

        # Run models
        features: torch.Tensor = feature_extractor(images, masks, gt_positions)
        output: torch.Tensor = predictor(features, target[:, 0], actions)
        loss: torch.Tensor = criterion(output, target)

        log_dict = eval_metrics(cfg, features, target, output, loss, prefix="test")
        mean.append(log_dict[f"test/l1_movement_mean/time_{cfg.time_steps-1}"])
        med.append(log_dict[f"test/l1_movement_median/time_{cfg.time_steps-1}"])
        ninetieth.append(log_dict[f"test/l1_movement_90th_percentile/time_{cfg.time_steps-1}"])

    print(f"Mean: {sum(mean) / len(mean)}")
    print(f"Median: {sum(med) / len(med)}")
    print(f"Ninetieth: {sum(ninetieth) / len(ninetieth)}")

def test_metrics(cfg: DictConfig, data_loader: DataLoader, feature_extractor: nn.Module, predictor: nn.Module, criterion: Any) -> Dict[str, Any]:
    """
    Test the model on the test set and return the evaluation metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor.eval()
    predictor.eval()
    num_samples = cfg.batch_size
    with torch.no_grad():
        images, bboxes, masks, _ = data_loader.sample(num_samples, cfg.time_steps, device, data_type="test")
        if cfg.ground_truth_masks:
            masks = get_ground_truth_masks(bboxes, masks.shape, device=device)
        positions = bboxes[:, :, :, :2]  # [B, H + T, O, 2]
        target = positions[:, cfg.data_loader.history_len :, :, :]  # [B, T, O, 2]
        gt_positions = positions[:, : cfg.data_loader.history_len, :, :]  # [B, H, O, 2]
        features: torch.Tensor = feature_extractor(images, masks, gt_positions)
        output: torch.Tensor = predictor(features, target[:, 0])
        loss: torch.Tensor = criterion(output, target)
        log_dict = eval_metrics(cfg, features, target, output, loss, prefix="test")
    return log_dict


def eval_metrics(
    cfg: DictConfig, features: torch.Tensor, target: torch.Tensor, output: torch.Tensor, loss: torch.Tensor, prefix: str = "train"
) -> Dict[str, Any]:
    """
    Calculate the evaluation metrics in a format suitable for wandb.
    Arguments:
        features: The features extracted from the images by the feature extractor [B, T, O, F]
        target: The target pixel coordinates [B, T, O, 2]
        output: The predicted pixel coordinates [B, T, O, 2]
        loss: The loss value from the criterion
    Returns:
        A dictionary containing the evaluation metrics
    """
    mask = target != 0
    diff = torch.pow(output - target, 2)
    max_loss = torch.max(torch.abs((output - target))).item()
    total_movement = torch.sum(torch.abs((target[:, cfg.time_steps - 1, :, :] - target[:, 0, :, :])))
    movement_mask = target[:, cfg.time_steps - 1, :, :] - target[:, 0, :, :] != 0
    average_movement = total_movement / torch.sum(movement_mask)
    nonzero_feats = features[features.sum(dim=-1) > 0]
    std, corr = nonzero_feats.std(dim=0), torch.corrcoef(nonzero_feats)
    log_dict: Dict[str, Any] = {"loss": loss, "error/x": diff[:, :, :, 0].mean(), "error/y": diff[:, :, :, 1].mean()}
    log_dict["l1_average_with_movement"] = (
        torch.sum(
            torch.abs(
                torch.squeeze(target[:, cfg.time_steps - 1, :, :], dim=1)[movement_mask]
                - torch.squeeze(output[:, cfg.time_steps - 1, :, :], dim=1)[movement_mask]
            )
        )
        / torch.sum(movement_mask)
    ).item()

    log_dict |= {"std_mean": std.mean(), "std_std": std.std(), "corr_mean": corr.mean(), "corr_std": corr.std()}
    log_dict |= {"max_loss": max_loss, "average_movement": average_movement}

    for t in range(cfg.time_steps):
        if t != 0:
            movement_mask = target[:, t, :, :] - target[:, 0, :, :] != 0
            total_movement = torch.sum(torch.abs((target[:, t, :, :] - target[:, 0, :, :])))
            log_dict[f"average_movement/time_{t}"] = total_movement / torch.sum(movement_mask)
            l1 = torch.abs(target[:, t, :, :][movement_mask] - output[:, t, :, :][movement_mask])
            log_dict[f"l1_movement_mean/time_{t}"] = torch.mean(l1)
            log_dict[f"l1_movement_median/time_{t}"] = torch.median(l1)
            log_dict[f"l1_movement_90th_percentile/time_{t}"] = torch.quantile(l1, 0.9)

        log_dict[f"error/time_{t}"] = diff[:, t, :, :].mean()

    log_dict = {f"{prefix}/{key}": value for key, value in log_dict.items()}
    return log_dict

if __name__ == "__main__":
    eval()  # pylint: disable=no-value-for-parameter
