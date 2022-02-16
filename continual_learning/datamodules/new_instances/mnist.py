from typing import Optional

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils import data

from continual_learning.config.params import KWARGS
from continual_learning.datamodules.bases.base_mnist import BaseMNIST


class MNIST(BaseMNIST):

    def __init__(self, experiments: int = 1):
        super().__init__()
        self.experiments = experiments

    def setup(self, stage: Optional[str] = None) -> None:
        chunks = np.array_split(list(self.train_dataset.indices), self.experiments)
        self.training_dataloader = []
        for chunk in chunks:
            self.training_dataloader.append(
                data.DataLoader(data.Subset(self.train_dataset, chunk), **KWARGS)
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.training_dataloader
