import os
from typing import List, Tuple, Optional

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from torch.utils.data import sampler
from torchvision import datasets, transforms

from continual_learning.config.paths import WANDB_DIR, DATA_DIR
from continual_learning.models.cnn import CNN

KWARGS = {'batch_size': 64}


class NewClassesMNIST:
    def __init__(self) -> None:
        torch.manual_seed(42)
        self.train: Optional[data.Dataset] = None
        self.val: Optional[data.Dataset] = None
        self.test: Optional[data.Dataset] = None

    def _prepare_datasets(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        self.train, self.val = data.random_split(
            self.train, [50000, 10000], generator=torch.Generator().manual_seed(42))
        self.test = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    def _stratify_classes(self, classes: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_train = self.train.dataset.targets.apply_(lambda x: x in classes).bool()
        idx_val = self.val.dataset.targets.apply_(lambda x: x in classes).bool()

        return idx_train, idx_val

    def _prepare_dataloaders(
            self,
            idx_train: torch.Tensor,
            idx_val: torch.Tensor
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        train_loader = data.DataLoader(self.train, sampler=sampler.SubsetRandomSampler(idx_train), **KWARGS)
        val_loader = data.DataLoader(self.val, sampler=sampler.SubsetRandomSampler(idx_val), shuffle=False, **KWARGS)

        return train_loader, val_loader

    def run_model(self, classes_before: List[int], classes_after: List[int]):
        self._prepare_datasets()

        idx_train, idx_val = self._stratify_classes(classes_before)
        train_loader, val_loader = self._prepare_dataloaders(idx_train, idx_val)

        test_loader = data.DataLoader(self.test, shuffle=False, **KWARGS)

        wandb.login(key=os.getenv('WANDB_KEY'))
        wandb_logger = WandbLogger(project='class_incremental_custom_model', save_dir=WANDB_DIR)

        trainer = pl.Trainer(logger=wandb_logger, max_epochs=1, deterministic=True)
        model = CNN()

        trainer.fit(model, train_loader, val_loader)

        if classes_after:
            idx_train, idx_val = self._stratify_classes(classes_after)
            train_loader, val_loader = self._prepare_dataloaders(idx_train, idx_val)
            trainer.fit(model, train_loader, val_loader)

        trainer.test(model, test_loader)
