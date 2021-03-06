import logging
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.continual_learning.strategy import Strategy
from continual_ranking.dpr.models import BiEncoder

logger = logging.getLogger(__name__)


class EWC(Strategy):
    def __init__(self, ewc_lambda: float):
        super().__init__()
        self.ewc_lambda = ewc_lambda

        self.train_dataloader: Optional[DataLoader] = None

        self.saved_params = {}
        self.fisher_matrix = {}

    def on_fit_end(self, trainer: ContinualTrainer, pl_module: BiEncoder) -> None:
        if trainer.task_id <= trainer.tasks - 1:
            logger.info(f'Calculating Fisher Matrix for EWC, task: {trainer.task_id}')

            self.saved_params = {}
            for n, p in pl_module.named_parameters():
                if p.requires_grad and p is not None:
                    self.saved_params[n] = p.data

    def calculate_importances(self, trainer: ContinualTrainer, pl_module: BiEncoder, train_dataloader: DataLoader):
        self.fisher_matrix = {}
        for n, p in self.saved_params.items():
            t = torch.zeros_like(p.data)
            self.fisher_matrix[n] = t

        pl_module.ewc_mode = True
        pl_module.fisher_matrix = self.fisher_matrix
        trainer.test(pl_module, train_dataloader)
        pl_module.ewc_mode = False

        for n in self.fisher_matrix:
            self.fisher_matrix[n] /= len(self.train_dataloader)

    @torch.no_grad()
    def _penalty(self, pl_module: "pl.LightningModule"):
        penalty = 0
        for n, p in pl_module.named_parameters():
            if n in self.fisher_matrix:
                diff = (p - self.saved_params[n].to(pl_module.device)).pow(2)
                loss = self.fisher_matrix[n].to(pl_module.device) * diff
                penalty += loss.sum()
        return penalty

    def apply_penalty(self, pl_module: BiEncoder, loss: torch.Tensor) -> None:
        if pl_module.experiment_id > 0:
            penalty = self._penalty(pl_module) * self.ewc_lambda
            pl_module.log('train/ewc_penalty', penalty)
            loss += penalty
            pl_module.log('train/loss_step', loss)
            pl_module.log('train/loss_epoch', loss, on_step=False, on_epoch=True)
