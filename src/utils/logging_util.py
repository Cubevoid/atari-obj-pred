import typing
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
import wandb

def start_logging(config: DictConfig, name: str) -> None:
    """
    Start logging to wandb
    """
    wandb.init(project="oc-data-collection", entity='atari-obj-pred', name=name,
               config=typing.cast(Dict[Any, Any], OmegaConf.to_container(config)))
