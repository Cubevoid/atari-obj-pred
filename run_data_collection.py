import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.logging import start_logging

@hydra.main(version_base=None, config_path='./configs/data_collection', config_name='config')
def main(cfg: DictConfig) -> None:
    start_logging(cfg, name=f"data-collection-{cfg.collector.game}-{cfg.collector.num_samples}")
    data_collector = instantiate(cfg.collector)
    data_collector.collect_data()

if __name__ == "__main__":
    #pylint: disable = no-value-for-parameter
    main()
