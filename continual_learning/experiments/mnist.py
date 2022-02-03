from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data
from torch.utils.data import sampler
from torchvision import datasets, transforms

from continual_learning.config.paths import DATA_DIR

KWARGS = {
    'batch_size':  64,
    'num_workers': 12,
}
BEFORE_CLASSES = [0, 1, 2, 3, 4]
AFTER_CLASSES = [5, 6, 7, 8, 9]


class MNIST(LightningDataModule):
    def __init__(self, mode: str, epochs: int = 1):
        super().__init__()
        torch.manual_seed(42)

        self.train_dataset: Optional[data.Dataset] = None
        self.train: Optional[data.Dataset] = None
        self.val: Optional[data.Dataset] = None
        self.test: Optional[data.Dataset] = None

        self.kwargs = KWARGS

        self.idx_train_before = []
        self.idx_train_after = []
        self.idx_val_before = []
        self.idx_val_after = []

        self.mode = mode
        self.epochs = epochs

    def prepare_data(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        self.test = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    @staticmethod
    def _get_indices(dataset, classes) -> List[int]:
        indices = []
        for i, value in enumerate(dataset.indices):
            if dataset.dataset.targets[value] in classes:
                indices.append(i)

        return indices

    def setup(self, stage: Optional[str] = None) -> None:
        self.train, self.val = data.random_split(
            self.train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))

        if self.mode == 'fine_tune':
            self.idx_train_before = self._get_indices(self.train, BEFORE_CLASSES)
            self.idx_train_after = self._get_indices(self.train, AFTER_CLASSES)
            self.idx_val_before = self._get_indices(self.val, BEFORE_CLASSES)
            self.idx_val_after = self._get_indices(self.val, AFTER_CLASSES)

    def _prepare_dataloader(self, dataset, idx: List[int] = None, shuffle: bool = None):
        if idx:
            sampler_ = sampler.SubsetRandomSampler(idx)
        else:
            sampler_ = None

        return data.DataLoader(dataset, sampler=sampler_, shuffle=shuffle, **self.kwargs)

    def _prepare_dataloaders(self, dataset, idx_before: List[int], idx_after: List[int]):
        if self.mode == 'full':
            return self._prepare_dataloader(dataset)
        elif self.mode == 'fine_tune':
            if self.trainer.current_epoch < self.epochs // 2:
                return self._prepare_dataloader(dataset, idx_before)
            else:
                return self._prepare_dataloader(dataset, idx_after)
        elif self.mode == 'fine_tune_with_old':
            if self.trainer.current_epoch < self.epochs // 2:
                return self._prepare_dataloader(dataset, idx_before)
            else:
                return self._prepare_dataloader(dataset, list(range(0, len(dataset), 100)))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._prepare_dataloaders(self.train, self.idx_train_before, self.idx_train_after)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._prepare_dataloaders(self.val, self.idx_val_before, self.idx_val_after)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.test, shuffle=False, **self.kwargs)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
