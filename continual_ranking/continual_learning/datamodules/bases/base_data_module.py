from abc import ABC, abstractmethod
from typing import Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data


class BaseDataModule(pl.LightningDataModule, ABC):

    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[Type[data.Dataset]] = None
        self.test_dataset: Optional[Type[data.Dataset]] = None

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    @abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
