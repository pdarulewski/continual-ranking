from abc import ABC, abstractmethod
from typing import List, Union, Optional, Iterable, Any

import pytorch_lightning as pl

from continual_ranking.config.configs import DataModule, BaseConfig
from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.types import Loggers, Dataloaders


class Experiment(ABC):

    def __init__(
            self,
            model: str,
            datamodule: DataModule,
            strategies: dict,
            project_name: str = None,
            max_epochs: int = 1,
            cfg: BaseConfig = None
    ):
        self.model: Optional[pl.LightningModule] = None
        self.datamodule: Optional[pl.LightningDataModule] = None
        self.strategies: Optional[Iterable[pl.Callback]] = None
        self.loggers: Loggers = None
        self.model_name = model
        self.datamodule_conf = datamodule
        self.strategies_conf = strategies

        self.max_epochs = max_epochs
        self.project_name = project_name

        self.trainer: Optional[Union[ContinualTrainer, Any]] = None

        self.train_dataloader: Dataloaders = None
        self.val_dataloader: Dataloaders = None
        self.test_dataloader: Dataloaders = None

        self.callbacks: List[pl.Callback] = []

        self.cfg = cfg

    @abstractmethod
    def prepare_dataloaders(self) -> None:
        """Prepare and assign the dataloaders"""

    @abstractmethod
    def setup_loggers(self) -> None:
        """Prepare and assign the loggers"""

    @abstractmethod
    def setup_strategies(self) -> None:
        """Prepare and assign the CL strategies"""

    @abstractmethod
    def setup_model(self) -> None:
        """Prepare and assign the model"""

    @abstractmethod
    def setup_trainer(self) -> None:
        """Prepare and assign the trainer"""

    def setup(self) -> None:
        self.setup_loggers()
        self.prepare_dataloaders()
        self.setup_model()
        self.setup_strategies()
        self.setup_trainer()

    @abstractmethod
    def run_training(self):
        """Run training and testing loop"""

    def execute(self):
        self.setup()
        self.run_training()
