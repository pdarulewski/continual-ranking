from typing import Optional

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils import data

from continual_ranking.continual_learning.datamodules.bases.base_mnist import BaseMNIST


class MNIST(BaseMNIST):

    def __init__(self, batch_size: int, num_workers: int, splits: int = 1):
        super().__init__(batch_size, num_workers)
        self.splits = splits

    def setup(self, stage: Optional[str] = None) -> None:
        chunks = np.array_split(list(self.train_dataset.indices), self.splits)
        self.training_dataloader = []
        for chunk in chunks:
            self.training_dataloader.append(
                data.DataLoader(
                    data.Subset(self.train_dataset, chunk),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers
                )
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.training_dataloader
