import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='./configs/data_collection', config_name='config')
def main(cfg: DictConfig) -> None:
    data_collector = instantiate(cfg.collector)
    data_collector.collect_data()

if __name__ == "__main__":
    #pylint: disable = no-value-for-parameter
    main()
