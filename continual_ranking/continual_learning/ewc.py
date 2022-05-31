import logging
import random

import pytorch_lightning as pl
import torch

from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.continual_learning.strategy import Strategy
from continual_ranking.dpr.models import BiEncoder

logger = logging.getLogger(__name__)


class EWC(Strategy):
    def __init__(self, ewc_lambda: float):
        super().__init__()
        self.ewc_lambda = ewc_lambda

        self.saved_params = {}
        self.fisher_matrix = {}

    @torch.no_grad()
    def _diag_fisher(self, trainer: ContinualTrainer, pl_module: BiEncoder):
        fisher_matrix = {}
        for n, p in self.saved_params.items():
            t = torch.zeros_like(p.data)
            fisher_matrix[n] = t

        pl_module.ewc_mode = True
        pl_module.fisher_matrix = fisher_matrix
        trainer.test(pl_module, trainer.train_dataloader)
        pl_module.ewc_mode = False

        for n in fisher_matrix:
            fisher_matrix[n] /= len(trainer.train_dataloader)

        return fisher_matrix

    def on_train_end(self, trainer: ContinualTrainer, pl_module: BiEncoder) -> None:
        if trainer.tasks > trainer.task_id:
            logger.info('Calculating Fisher Matrix for EWC')

            self.saved_params = {}
            for n, p in pl_module.named_parameters():
                if random.random() < 0.7:
                    continue
                if p.requires_grad and p is not None:
                    self.saved_params[n] = p.data.detach().clone()

            self.fisher_matrix = self._diag_fisher(trainer, pl_module)

    @torch.no_grad()
    def _penalty(self, pl_module: "pl.LightningModule"):
        penalty = 0
        for n, p in pl_module.named_parameters():
            if n in self.fisher_matrix:
                loss = self.fisher_matrix[n] * (p - self.saved_params[n]) ** 2
                penalty += loss.sum()
        return penalty

    def on_before_backward(
            self,
            trainer: ContinualTrainer,
            pl_module: "pl.LightningModule",
            loss: torch.Tensor
    ) -> torch.Tensor:
        if trainer.task_id > 0:
            loss += self.ewc_lambda * self._penalty(pl_module)
        return loss
