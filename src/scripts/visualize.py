import hydra
from omegaconf import DictConfig
from src.data_visualization.visualizer import Visualizer

@hydra.main(version_base=None, config_path="../../configs/training", config_name="config")
def start(cfg: DictConfig) -> None:
    Visualizer(cfg)

if __name__ == "__main__":
    start()  # pylint: disable=no-value-for-parameter
