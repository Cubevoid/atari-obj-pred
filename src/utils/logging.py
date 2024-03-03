from omegaconf import DictConfig
import wandb

def start_logging(config: DictConfig, name: str) -> None:
    """
    Start logging to wandb
    """
    wandb.init(project="oc-data-collection", entity='atari-obj-pred', name=name, config=dict(config))