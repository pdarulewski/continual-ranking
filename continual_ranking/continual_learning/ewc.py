import logging

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

    def on_fit_end(self, trainer: ContinualTrainer, pl_module: BiEncoder) -> None:
        if trainer.tasks > trainer.task_id:
            logger.info('Calculating Fisher Matrix for EWC')

            self.saved_params = {}
            for n, p in pl_module.named_parameters():
                if p.requires_grad and p is not None:
                    self.saved_params[n] = p.data.detach().clone()

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
