import itertools
import os
from typing import List

import wandb
from pytorch_lightning.loggers import WandbLogger

from continual_learning.config.configs import DataModule, Strategy
from continual_learning.config.dicts import STRATEGIES, MODELS, DATA_MODULES
from continual_learning.config.paths import LOG_DIR
from continual_learning.continual_trainer import ContinualTrainer
from continual_learning.experiments.experiment import Experiment


class ExperimentRunner(Experiment):

    def __init__(
            self,
            model: str,
            datamodule: DataModule,
            strategies: List[Strategy],
            project_name: str = None,
            max_epochs: int = 1,
    ):
        super().__init__(
            model=model,
            datamodule=datamodule,
            strategies=strategies,
            project_name=project_name,
            max_epochs=max_epochs
        )

        self._epochs_completed = 0

    def prepare_dataloaders(self) -> None:
        datamodule = DATA_MODULES[self.datamodule_conf.name]
        self.datamodule = datamodule(**self.datamodule_conf.params)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_dataloader = self.datamodule.train_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.val_dataloader = self.datamodule.val_dataloader()

        if not self.val_dataloader:
            self.val_dataloader = [None]

    def setup_loggers(self):
        wandb.login(key=os.getenv('WANDB_KEY'))

        loggers = [
            WandbLogger(
                project=self.project_name,
                save_dir=LOG_DIR,
            )
        ]

        self.loggers = loggers

    def setup_strategies(self) -> None:
        for d in self.strategies_conf:
            strategy = STRATEGIES[d.name](**d.params)
            self.callbacks.append(strategy)

    def setup_model(self) -> None:
        self.model = MODELS[self.model_name]()

    def setup_trainer(self) -> None:
        self.trainer = ContinualTrainer(
            logger=self.loggers,
            max_epochs=self._epochs_completed + self.max_epochs,
            deterministic=True,
            callbacks=self.callbacks,
            log_every_n_steps=1,
        )

    def run_training(self):
        for train_dataloader, val_dataloader in itertools.zip_longest(
                self.train_dataloader, self.val_dataloader, fillvalue=self.val_dataloader
        ):
            self.trainer.fit_loop.max_epochs = self._epochs_completed + self.max_epochs
            self.trainer.fit_loop.current_epoch = self._epochs_completed

            self.trainer.fit(self.model, train_dataloader, val_dataloader)

            self._epochs_completed = self.trainer.current_epoch + 1
            self.trainer.task_id += 1

            self.trainer.test(self.model, self.test_dataloader)
