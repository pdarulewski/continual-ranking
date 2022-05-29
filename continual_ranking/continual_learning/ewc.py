import logging
from copy import deepcopy

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

        self.penalty = 0
        self.params = {}
        self._means = {}
        self.fisher_matrix = {}

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self, trainer: ContinualTrainer, pl_module: BiEncoder, train_dataloader: DataLoader):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        pl_module.ewc_mode = True
        pl_module.precision_matrices = precision_matrices
        trainer.test(pl_module, train_dataloader)
        pl_module.ewc_mode = False

        for n in precision_matrices:
            if precision_matrices[n] is not None:
                precision_matrices[n] /= len(train_dataloader)

        return precision_matrices

    def calculate_importances(
            self, trainer: ContinualTrainer, pl_module: BiEncoder, train_dataloader: DataLoader
    ) -> None:
        self.params = {n: p for n, p in pl_module.named_parameters() if p.requires_grad}
        self.fisher_matrix = self._diag_fisher(trainer, pl_module, train_dataloader)
        self.penalty = self._penalty(pl_module)

    def _penalty(self, pl_module: "pl.LightningModule"):
        loss = 0
        for n, p in pl_module.named_parameters():
            if n in self.fisher_matrix and n in self._means:
                _loss = self.fisher_matrix[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

    def on_train_start(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        self.params = {}
        self.fisher_matrix = {}

    def on_before_backward(
            self,
            trainer: ContinualTrainer,
            pl_module: "pl.LightningModule",
            loss: torch.Tensor
    ) -> torch.Tensor:
        if trainer.task_id > 0:
            loss += self.ewc_lambda * self.penalty
        return loss
