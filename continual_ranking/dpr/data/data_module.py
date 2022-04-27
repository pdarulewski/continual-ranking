from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from continual_ranking.dpr.data.training_dataset import TrainingDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: Optional[str] = None):
        self.train_set = TrainingDataset(self.cfg.datasets.train)
        self.val_set = TrainingDataset(self.cfg.datasets.val)
        # self.test_set = TrainingDataset(self.cfg.datasets.test)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.cfg.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.cfg.train.num_workers)

    def test_dataloader(self):
        pass
        # return DataLoader(self.test_set, batch_size=self.batch_size)
