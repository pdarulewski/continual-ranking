from typing import Optional, List

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torchvision import datasets, transforms

from continual_ranking.config.paths import DATA_DIR
from continual_ranking.continual_learning.datamodules.bases.base_data_module import BaseDataModule


class BaseMNIST(BaseDataModule):

    def __init__(self, batch_size: int, num_workers: int):
        super().__init__(batch_size, num_workers)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.training_dataloader: Optional[List[data.DataLoader]] = []
        self.validation_dataloader: Optional[List[data.DataLoader]] = []
        self.testing_dataloader: Optional[List[data.DataLoader]] = []

    def prepare_data(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.training_dataloader = data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        return self.training_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.validation_dataloader = data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        return self.validation_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        self.testing_dataloader = data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        return self.testing_dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
