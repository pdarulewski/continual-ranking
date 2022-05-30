import logging
import math
import os
import traceback
from abc import abstractmethod
from typing import Optional, List, Union, Any, Iterable

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader

from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.continual_learning.ewc import EWC
from continual_ranking.continual_learning.gem import GEM
from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.models import BiEncoder

logger = logging.getLogger(__name__)


class Base:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.fast_dev_run = cfg.fast_dev_run
        self.logging_on = cfg.logging_on and not self.fast_dev_run
        self.experiment_name = cfg.experiment.name

        self.model: Optional[BiEncoder] = None
        self.datamodule: Optional[DataModule] = None
        self.train_dataloader: Union[DataLoader, List[DataLoader]] = []
        self.val_dataloader: Union[DataLoader, List[DataLoader]] = []
        self.trainer: Optional[Union[ContinualTrainer, Any]] = None
        self.strategies: Optional[Iterable[pl.Callback]] = None

        self.loggers: List[pl.loggers.LightningLoggerBase] = []
        self.callbacks: List[pl.Callback] = []

        self.ewc: Optional[EWC] = None

    def alert(self, title: str, text: str = '', **kwargs) -> None:
        if self.logging_on:
            wandb.alert(title=title, text=text, **kwargs)

        logging.info(title)
        logging.info(text)

    def setup_model(self) -> None:
        logger.info('Setting up model')
        self.model = BiEncoder(
            self.cfg,
            math.ceil(self.datamodule.train_set_length / self.cfg.biencoder.train_batch_size)
        )

    def setup_loggers(self) -> None:
        if self.logging_on:
            logger.info('Setting up wandb logger')
            wandb.login(key=os.getenv('WANDB_KEY'))

            wandb_logger = WandbLogger(
                name=self.experiment_name,
                project=self.cfg.project_name,
                offline=not self.logging_on,
            )

            wandb.init()

            wandb.define_metric('experiment_id')
            wandb.define_metric('val/loss_experiment', step_metric='experiment_id')
            wandb.define_metric('val/acc_experiment', step_metric='experiment_id')

            wandb.define_metric('test/loss_experiment', step_metric='experiment_id')
            wandb.define_metric('test/acc_experiment', step_metric='experiment_id')

            wandb_logger.watch(self.model, log='all', log_freq=500)

            self.loggers.append(wandb_logger)

        csv_logger = CSVLogger(
            'csv',
            name=self.cfg.project_name,
            version=self.experiment_name
        )

        self.loggers.append(csv_logger)

    def setup_callbacks(self) -> None:
        logger.info('Setting up callbacks')
        self.callbacks = [
            ModelCheckpoint(
                filename=self.experiment_name + '-{epoch:02d}-{val/loss_epoch:.2f}',
                monitor='val/loss_epoch',
            ),
            EarlyStopping(
                monitor='val/loss_epoch',
                min_delta=0.01,
                verbose=True
            ),
        ]

    def prepare_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')

        self.datamodule = DataModule(self.cfg)

        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()

    def setup_trainer(self) -> None:
        logger.info('Setting up trainer')
        self.trainer = ContinualTrainer(
            tasks=len(self.cfg.experiment.sizes) - 2,
            max_epochs=self.cfg.biencoder.max_epochs,
            accelerator=self.cfg.device,
            gpus=-1 if self.cfg.device == 'gpu' else 0,
            deterministic=True,
            auto_lr_find=True,
            logger=self.loggers,
            callbacks=self.callbacks,
            fast_dev_run=self.fast_dev_run,
            num_sanity_val_steps=0
        )

    def setup_strategies(self) -> None:
        if self.cfg.experiment.strategy == 'ewc':
            self.ewc = EWC(**self.cfg.ewc)
            strategy = self.ewc
        elif self.cfg.experiment.strategy == 'gem':
            strategy = GEM(**self.cfg.gem)
        else:
            return

        self.callbacks.append(strategy)

    def setup(self) -> None:
        self.prepare_dataloaders()
        self.setup_model()
        self.setup_loggers()
        self.setup_callbacks()
        self.setup_strategies()
        self.setup_trainer()

    @abstractmethod
    def run_training(self) -> None:
        """Start training, validation, testing, and indexing loop here"""

    def execute(self) -> None:
        try:
            self.setup()
            self.run_training()

        except Exception as e:
            self.alert(
                title='Run has crashed!',
                text=f'Error:\n```{e}```',
                level=wandb.AlertLevel.ERROR
            )
            logger.exception(traceback.format_exc())
            raise
