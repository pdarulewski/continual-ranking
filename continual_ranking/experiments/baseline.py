import logging
import math
import os

import wandb
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.models.biencoder import BiEncoder
from continual_ranking.experiments.experiment import Experiment

logger = logging.getLogger(__name__)


class Baseline(Experiment):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.fast_dev_run = cfg.fast_dev_run

    def prepare_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')

        self.datamodule = DataModule(self.cfg)

        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

    def setup_loggers(self) -> None:
        logger.info('Setting up wandb logger')
        wandb.login(key=os.getenv('WANDB_KEY'))

        wandb_logger = WandbLogger(
            name=self.cfg.experiment_name,
            project=self.cfg.project_name,
            offline=self.fast_dev_run,
        )

        wandb.init()
        wandb.log(OmegaConf.to_container(self.cfg))

        self.loggers = [wandb_logger]

    def setup_model(self) -> None:
        logger.info('Setting up model')
        self.model = BiEncoder(self.cfg, math.ceil(self.datamodule.train_set_length / self.cfg.biencoder.batch_size))

    def setup_callbacks(self) -> None:
        logger.info('Setting up callbacks')
        filename = self.cfg.experiment_name
        self.callbacks = [
            ModelCheckpoint(
                filename=filename + '-{epoch:02d}-{val_loss:.2f}',
                save_top_k=2,
                monitor='val_loss',
                mode='min,,,,,,'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=0.01,
                mode='min',
                verbose=True
            ),
        ]

    def setup_trainer(self) -> None:
        logger.info('Setting up trainer')
        self.trainer = Trainer(
            max_epochs=self.cfg.biencoder.max_epochs,
            accelerator=self.cfg.device,
            gpus=-1 if self.cfg.device == 'gpu' else 0,
            deterministic=True,
            auto_lr_find=True,
            # log_every_n_steps=1,
            logger=self.loggers,
            callbacks=self.callbacks,
            fast_dev_run=self.fast_dev_run
        )

    def setup_strategies(self) -> None:
        pass

    def run_training(self):
        wandb.alert(
            title=f'Training for {self.cfg.experiment_name} started!',
            text=f'```\n{OmegaConf.to_yaml(self.cfg)}```'
        )

        for index, (train_dataloader, val_dataloader) in enumerate(zip(self.train_dataloader, self.val_dataloader)):
            train_length = len(train_dataloader.dataset)
            val_length = len(val_dataloader.dataset)
            train_data_len_msg = f'Training dataloader size: {train_length}'
            val_data_len_msg = f'Validation dataloader size: {val_length}'

            self.model.train_length = train_length
            self.model.val_length = val_length

            logger.info(train_data_len_msg)
            logger.info(val_data_len_msg)

            wandb.alert(
                title=f'Experiment #{index} for {self.cfg.experiment_name} started!',
                text=f'{train_data_len_msg}\n{val_data_len_msg}'
            )

            self.trainer.fit(self.model, train_dataloader, val_dataloader)

        # self.trainer.test(self.model, self.test_dataloader)
