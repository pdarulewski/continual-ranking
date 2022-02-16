import itertools
import os
from typing import List, Union, Optional, Iterable

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger, LightningLoggerBase
from torch.utils import data

from continual_learning.config.configs import Datamodule, Strategy
from continual_learning.config.dicts import STRATEGIES, MODELS, DATA_MODULES
from continual_learning.config.paths import LOG_DIR
from continual_learning.continual_trainer import ContinualTrainer


class ExperimentRunner:

    def __init__(
            self,
            model: str,
            datamodule: Datamodule,
            strategies: List[Strategy],
            project_name: str = None,
            max_epochs: int = 1,
    ):
        self._model: pl.LightningModule
        self._datamodule: pl.LightningDataModule
        self._strategies: Iterable[pl.Callback]
        self._loggers: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]

        self._model_name = model
        self._datamodule_conf = datamodule
        self._strategies_conf = strategies

        self._trainer: Optional[ContinualTrainer] = None
        self._epochs_completed = 0
        self.max_epochs = max_epochs
        self.project_name = project_name

        self.train_dataloader: Optional[Union[data.DataLoader, Iterable[data.DataLoader]]] = None
        self.val_dataloader: Optional[Union[data.DataLoader, Iterable[data.DataLoader]]] = None
        self.test_dataloader: Optional[Union[data.DataLoader, Iterable[data.DataLoader]]] = None

        self._callbacks: List[pl.Callback] = []

    def _prepare_dataloaders(self) -> None:
        datamodule = DATA_MODULES[self._datamodule_conf.name]
        self._datamodule = datamodule(**self._datamodule_conf.params)
        self._datamodule.prepare_data()
        self._datamodule.setup()

        self.train_dataloader = self._datamodule.train_dataloader()
        self.test_dataloader = self._datamodule.test_dataloader()

        self.val_dataloader = self._datamodule.val_dataloader()

        if not self.val_dataloader:
            self.val_dataloader = [None]

    def _setup_loggers(self):
        wandb.login(key=os.getenv('WANDB_KEY'))

        loggers = [
            WandbLogger(
                project=self.project_name,
                save_dir=LOG_DIR,
            )
        ]

        self._loggers = loggers

    def _setup_strategies(self) -> None:
        for d in self._strategies_conf:
            strategy = STRATEGIES[d.name](**d.params)
            self._callbacks.append(strategy)

    def _setup_model(self) -> None:
        self._model = MODELS[self._model_name]()

    def setup(self):
        self._setup_loggers()
        self._prepare_dataloaders()
        self._setup_strategies()
        self._setup_model()

        self._trainer = ContinualTrainer(
            logger=self._loggers,
            max_epochs=self._epochs_completed + self.max_epochs,
            deterministic=True,
            callbacks=self._callbacks
        )

    def run_training(self):
        for train_dataloader, val_dataloader in itertools.zip_longest(
                self.train_dataloader, self.val_dataloader, fillvalue=self.val_dataloader
        ):
            self._trainer.fit_loop.max_epochs = self._epochs_completed + self.max_epochs
            self._trainer.fit_loop.current_epoch = self._epochs_completed

            self._trainer.fit(self._model, train_dataloader, val_dataloader)

            self._epochs_completed = self._trainer.current_epoch + 1
            self._trainer.task_id += 1

            self._trainer.test(self._model, self.test_dataloader)
