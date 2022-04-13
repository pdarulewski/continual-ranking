import itertools
import random
from typing import List, Optional

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import sampler

from continual_ranking.continual_learning.datamodules.bases.base_mnist import BaseMNIST
from continual_ranking.continual_learning.datamodules.bases.custom_data_loader import CustomDataLoader


class MNIST(BaseMNIST):
    def __init__(self, batch_size: int, num_workers: int, splits: int = 1):
        super().__init__(batch_size, num_workers)

        self.splits = splits

        self._targets = list(range(0, 10))
        random.shuffle(self._targets)

        self.target_splits = [list(i) for i in np.array_split(self._targets, self.splits)]
        # self.target_splits = [[2, 6], [8, 1], [4, 5], [0, 9], [3, 7]]

        self.cumulative_splits = list(itertools.accumulate(self.target_splits))

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
        splits = []

        for split in self.target_splits:
            indices = self._get_indices(self.train_dataset, list(split))
            splits.append(self._prepare_dataloader(self.train_dataset, indices, split))

        self.training_dataloader = splits

        splits = []

        for split in self.cumulative_splits:
            indices = self._get_indices(self.test_dataset, list(split))
            splits.append(self._prepare_dataloader(self.test_dataset, indices, split))

        self.validation_dataloader = splits

    def _prepare_dataloader(self, dataset, idx: List[int] = None, split: list = None):
        sampler_ = sampler.SubsetRandomSampler(idx)
        return CustomDataLoader(
            classes=split,
            dataset=dataset,
            sampler=sampler_,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.training_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.validation_dataloader
