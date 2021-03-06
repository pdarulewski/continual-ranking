import logging
import logging.config
import os

import hydra
import torch.cuda
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from continual_ranking.experiment.experiment import Experiment


def setup_logging() -> None:
    with open(os.path.join('config', 'logging.yaml'), 'r') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


@hydra.main(config_path='../config', config_name='base')
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f'\n{OmegaConf.to_yaml(cfg)}')

    seed_everything(42, workers=True)

    accelerator = 'gpu' if torch.cuda.is_available() and cfg.device == 'gpu' else 'cpu'

    cfg.device = accelerator

    baseline = Experiment(cfg)
    baseline.execute()


if __name__ == "__main__":
    main()
