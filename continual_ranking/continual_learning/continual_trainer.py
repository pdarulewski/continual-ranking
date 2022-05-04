from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class ContinualTrainer(pl.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_id: int = 0

    def fit(
            self,
            model: "pl.LightningModule",
            train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
            val_dataloaders: Optional[EVAL_DATALOADERS] = None,
            datamodule: Optional[LightningDataModule] = None,
            train_dataloader=None,
            ckpt_path: Optional[str] = None,
    ) -> None:
        super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
