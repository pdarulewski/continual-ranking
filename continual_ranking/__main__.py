import math
import os

import hydra
import torch.cuda
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.models.biencoder import BiEncoder


@hydra.main(config_path="../config", config_name='base')
def main(cfg: DictConfig):
    seed_everything(42, workers=True)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    wandb.login(key=os.getenv('WANDB_KEY'))

    logger = WandbLogger(
        project='test_dpr',
    )

    wandb.init()
    wandb.log(OmegaConf.to_container(cfg))

    trainer = Trainer(
        max_epochs=cfg.biencoder.max_epochs,
        accelerator=accelerator,
        gpus=-1 if accelerator == 'gpu' else 0,
        deterministic=True,
        auto_lr_find=True,
        log_every_n_steps=1,
        logger=[logger]
    )

    data_module = DataModule(cfg)
    data_module.setup()
    model = BiEncoder(cfg, math.ceil(len(data_module.train_set) / cfg.biencoder.batch_size))

    data_module = DataModule(cfg)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
