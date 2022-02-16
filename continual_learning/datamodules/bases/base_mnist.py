from typing import Optional, List

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torchvision import datasets, transforms

from continual_learning.config.paths import DATA_DIR


class BaseMNIST(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[data.Dataset] = None
        self.test_dataset: Optional[data.Dataset] = None

        self.training_dataloader: Optional[List[data.DataLoader]] = []

    def prepare_data(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
