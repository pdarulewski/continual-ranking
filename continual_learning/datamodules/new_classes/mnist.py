from typing import List, Optional

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data
from torch.utils.data import sampler

from continual_learning.config.params import KWARGS
from continual_learning.datamodules.bases.base_mnist import BaseMNIST


class MNIST(BaseMNIST):
    def __init__(self, split_size: int = 1, kwargs: dict = None):
        super().__init__()

        self.split_size = split_size
        self.kwargs = kwargs if kwargs else KWARGS

        self._targets = list(range(0, 10))
        if split_size > len(self._targets):
            raise ValueError('Too many splits for distinct targets')

    @staticmethod
    def _get_indices(dataset, classes) -> List[int]:
        indices = []
        for i, value in enumerate(dataset.targets):
            if value in classes:
                indices.append(i)

        return indices

    def setup(self, stage: Optional[str] = None) -> None:
        target_splits = np.array_split(self._targets, self.split_size)
        splits = []

        for split in target_splits:
            indices = self._get_indices(self.train_dataset, list(split))
            splits.append(self._prepare_dataloader(self.train_dataset, indices))

        self.training_dataloader = splits

    def _prepare_dataloader(self, dataset, idx: List[int] = None):
        sampler_ = sampler.SubsetRandomSampler(idx)
        return data.DataLoader(dataset, sampler=sampler_, **self.kwargs)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.training_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [None]
