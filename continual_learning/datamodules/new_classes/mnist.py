from typing import List, Optional

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data
from torch.utils.data import sampler

from continual_learning.datamodules.bases.base_mnist import BaseMNIST


class MNIST(BaseMNIST):
    def __init__(self, batch_size: int, num_workers: int, splits: int = 1):
        super().__init__(batch_size, num_workers)

        self.splits = splits

        self._targets = list(range(0, 10))
        if splits > len(self._targets):
            raise ValueError('Too many splits for distinct targets')

    @staticmethod
    def _get_indices(dataset, classes) -> List[int]:
        indices = []
        for i, value in enumerate(dataset.targets):
            if value in classes:
                indices.append(i)

        return indices

    def setup(self, stage: Optional[str] = None) -> None:
        target_splits = np.array_split(self._targets, self.splits)
        splits = []

        for split in target_splits:
            indices = self._get_indices(self.train_dataset, list(split))
            splits.append(self._prepare_dataloader(self.train_dataset, indices))

        self.training_dataloader = splits

    def _prepare_dataloader(self, dataset, idx: List[int] = None):
        sampler_ = sampler.SubsetRandomSampler(idx)
        return data.DataLoader(
            dataset,
            sampler=sampler_,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.training_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass
