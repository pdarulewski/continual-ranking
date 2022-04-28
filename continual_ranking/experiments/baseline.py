import math
import os

import wandb
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.models.biencoder import BiEncoder
from continual_ranking.experiments.experiment import Experiment


class Baseline(Experiment):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self._epochs_completed = 0

    def prepare_dataloaders(self) -> None:
        self.datamodule = DataModule(self.cfg)

        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

    def setup_loggers(self):
        wandb.login(key=os.getenv('WANDB_KEY'))

        logger = WandbLogger(
            name=self.cfg.experiment_name,
            project=self.cfg.project_name,
        )

        wandb.init()
        wandb.log(OmegaConf.to_container(self.cfg))

        self.loggers = [logger]

    def setup_model(self) -> None:
        self.model = BiEncoder(self.cfg, math.ceil(self.datamodule.train_set_length / self.cfg.biencoder.batch_size))

    def setup_trainer(self) -> None:
        self.trainer = Trainer(
            max_epochs=self.cfg.biencoder.max_epochs,
            accelerator=self.cfg.device,
            gpus=-1 if self.cfg.device == 'gpu' else 0,
            deterministic=True,
            auto_lr_find=True,
            log_every_n_steps=1,
            logger=self.loggers
        )

    def setup_strategies(self) -> None:
        pass

    def run_training(self):
        for train_dataloader, val_dataloader in zip(self.train_dataloader, self.val_dataloader):
            self.trainer.fit(self.model, train_dataloader, val_dataloader)

        # self.trainer.test(self.model, self.test_dataloader)
