import hydra
import torch.cuda
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from continual_ranking.experiments import Baseline


@hydra.main(config_path="../config", config_name='base')
def main(cfg: DictConfig):
    seed_everything(42, workers=True)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    cfg.device = accelerator

    baseline = Baseline(cfg)
    baseline.execute()


if __name__ == "__main__":
    main()
