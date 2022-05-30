import logging
from copy import deepcopy

import pytorch_lightning as pl
import torch

from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.continual_learning.strategy import Strategy

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
