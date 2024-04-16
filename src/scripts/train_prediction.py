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

from src.data_collection.data_loader import DataLoader
from src.model.predictor import Predictor
from src.model.mlp_predictor import MLPPredictor
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

from src.data_collection.data_loader import DataLoader
#from src.model.predictor import Predictor
#from src.model.mlp_predictor import MLPPredictor
from src.model.current_predictor import CurrentPredictor
from src.model.residual_predictor import ResidualPredictor


@hydra.main(version_base=None, config_path="../../configs/training", config_name="config")
def train(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_mlp = cfg.predictor == "mlp"

    data_loader = instantiate(cfg.data_loader, game=cfg.game, num_obj=cfg.num_objects, val_pct=0, test_pct=0.3)
    feature_extractor = instantiate(cfg.feature_extractor, num_objects=cfg.num_objects, history_len=cfg.data_loader.history_len).to(device)
    predictor = instantiate(cfg.predictor, time_steps=cfg.time_steps).to(device)

    wandb.init(project="oc-data-training", entity="atari-obj-pred", name=cfg.name + cfg.game, config=typing.cast(Dict[Any, Any], OmegaConf.to_container(cfg)))
    wandb.log({"batch_size": cfg.batch_size})
    wandb.watch(feature_extractor, log=None, log_freq=100, idx=1)
    wandb.watch(predictor, log=None, log_freq=100, idx=2)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(predictor.parameters()), lr=cfg.lr)

    for i in tqdm(range(cfg.num_iterations)):
        images, bboxes, masks, _ = data_loader.sample(cfg.batch_size, cfg.time_steps, device)
        if cfg.ground_truth_masks:
            masks = get_ground_truth_masks(bboxes, masks.shape, device=device)

        positions = bboxes[:, :, :, :2]  # [B, H + T, O, 2]
        target = positions[:, cfg.data_loader.history_len:, :, :]  # [B, T, O, 2]
        gt_positions = positions[:, :cfg.data_loader.history_len, :, :]  # [B, H, O, 2]

        # Run models
        features: torch.Tensor = feature_extractor(images, masks, gt_positions)
        output: torch.Tensor = predictor(features, target[:, 0])
        loss: torch.Tensor = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_log_dict = eval_metrics(cfg, features, target, output, loss)

        if cfg.debug and i % 200 == 0:
            print_log_dict("train", train_log_dict)
            test_log_dict = test_metrics(cfg, data_loader, feature_extractor, predictor, criterion)
            print_log_dict("test", test_log_dict)
            wandb.log(test_log_dict)
        wandb.log(train_log_dict)
        optimizer.zero_grad()

    if cfg.save_models:
        # save trained model to disk
        save_models(cfg.game, feature_extractor, predictor)


def print_log_dict(prefix: str, log_dict: dict[str, Any]) -> None:
    tqdm.write(f"=== {prefix} metrics ===")
    tqdm.write(f"loss={log_dict[f'{prefix}/loss']}")
    tqdm.write(f"l1 average loss = {log_dict[f'{prefix}/avg_l1']}")
    tqdm.write(f"maximum loss = {log_dict[f'{prefix}/max_loss']}")
    tqdm.write(f"median loss = {log_dict[f'{prefix}/med_l1']}")
    tqdm.write(f"average movement = {log_dict[f'{prefix}/average_movement']}")
    tqdm.write(f"average l1 loss on last time step where object moved = {log_dict[f'{prefix}/l1_average_with_movement']}")


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
        target = positions[:, cfg.data_loader.history_len:, :, :]  # [B, T, O, 2]
        gt_positions = positions[:, :cfg.data_loader.history_len, :, :]  # [B, H, O, 2]
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

    log_dict["avg_l1"] = torch.sum(torch.abs(target[mask] - output[mask])) / torch.sum(mask)
    log_dict["med_l1"] = torch.median(torch.abs(target[mask] - output[mask]))
    log_dict |= {"std_mean": std.mean(), "std_std": std.std(), "corr_mean": corr.mean(), "corr_std": corr.std()}
    log_dict |= {"max_loss": max_loss, "average_movement": average_movement}
    for t in range(cfg.time_steps):
        if t != 0:
            movement_mask = target[:, t, :, :] - target[:, 0, :, :] != 0
            total_movement = torch.sum(torch.abs((target[:, t, :, :] - target[:, 0, :, :])))
            log_dict[f"average_movement/time_{t}"] = total_movement / torch.sum(movement_mask)
            log_dict[f"l1_movement_average/time_{t}"] = torch.sum(
                torch.abs(torch.squeeze(target[:, t, :, :], dim=1)[movement_mask] - torch.squeeze(output[:, t, :, :], dim=1)[movement_mask])
            ) / torch.sum(movement_mask)
        log_dict[f"error/time_{t}"] = diff[:, t, :, :].mean()

    log_dict = {f"{prefix}/{key}": value for key, value in log_dict.items()}
    return log_dict


def save_models(game: str, feature_extractor: nn.Module, predictor: nn.Module) -> None:
    """
    Save the models to the disk
    """
    unix_time = int(time.time())
    os.makedirs(to_absolute_path(f"./models/trained/{game}"), exist_ok=True)
    torch.save(feature_extractor.state_dict(), to_absolute_path(f"./models/trained/{game}/{unix_time}_feat_extract.pth"))
    torch.save(predictor.state_dict(), to_absolute_path(f"./models/trained/{game}/{unix_time}_{type(predictor).__name__}.pth"))
    print(f"Saved models at {unix_time}")


def get_ground_truth_masks(bboxes: torch.Tensor, mask_size: torch.Size, device: str) -> torch.Tensor:
    bbox_ints = bboxes * 128
    bbox_ints = bbox_ints.int()
    masks = torch.zeros(mask_size, device=device)
    for j in range(len(masks)):  # pylint: disable=consider-using-enumerate
        for k in range(len(masks[j])):
            masks[
                j,
                k,
                int(bbox_ints[j][0][k][0]) : int(bbox_ints[j][0][k][0] + bbox_ints[j][0][k][2]),
                int(bbox_ints[j][0][k][1]) : int(bbox_ints[j][0][k][1] + bbox_ints[j][0][k][3]),
            ] = 1
    return masks


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
