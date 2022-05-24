import logging
from collections import defaultdict
from typing import Tuple, List

import pytorch_lightning as pl
import torch

from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.continual_learning.strategy import Strategy

logger = logging.getLogger(__name__)


class EWC(Strategy):
    def __init__(
            self,
            ewc_lambda: float,
            mode: str = 'separate',
            decay_factor: float = None,
            keep_importance_data: bool = False,
    ):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.is_separate = mode == 'separate'
        self.decay_factor = decay_factor

        if self.is_separate:
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def on_before_backward(
            self, trainer: ContinualTrainer, pl_module: "pl.LightningModule", loss: torch.Tensor
    ) -> None:
        logging.info('EWC: on_before_backward')
        exp_counter = trainer.task_id
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(pl_module.device)

        logging.info('EWC: calculating penalty')
        for experience in range(exp_counter):
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    pl_module.named_parameters(),
                    self.saved_params[experience],
                    self.importances[experience]):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        logging.info(f'EWC: adding penalty to loss: {penalty.item()}')
        loss += self.ewc_lambda * penalty

    def on_after_backward(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        logging.info('EWC: on_after_backward')
        task_id = trainer.task_id
        importances = self._compute_importances(trainer, pl_module)
        self._update_importances(importances, task_id)
        self.saved_params[task_id] = self.copy_params_dict(pl_module)

        logging.info('EWC: clearing old importances')
        if task_id > 0 and not self.keep_importance_data:
            del self.saved_params[task_id - 1]

    def _compute_importances(
            self,
            trainer: ContinualTrainer,
            pl_module: "pl.LightningModule"
    ) -> List[Tuple[str, torch.Tensor]]:
        logging.info('EWC: _compute_importances')
        importances = self.zero_like_params_dict(pl_module)

        for (k1, p), (k2, imp) in zip(pl_module.named_parameters(), importances):
            if k1 != k2:
                raise ValueError('Error in importance computation.')
            if p.grad is not None:
                imp += p.grad.data.clone().pow(2)

        for _, imp in importances:
            imp /= float(len(trainer.train_dataloader))

        return importances

    @torch.no_grad()
    def _update_importances(self, importances, t) -> None:
        logging.info('EWC: _update_importances')

        if self.is_separate or t == 0:
            self.importances[t] = importances
        else:
            for (k1, old_imp), (k2, curr_imp) in zip(self.importances[t - 1], importances):
                if k1 != k2:
                    raise ValueError('Error in importance computation.')
                self.importances[t].append((k1, (self.decay_factor * old_imp + curr_imp)))

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

    @staticmethod
    def copy_params_dict(model, copy_grad=False) -> list:
        if copy_grad:
            return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
        else:
            return [(k, p.data.clone()) for k, p in model.named_parameters()]

    @staticmethod
    def zero_like_params_dict(model) -> list:
        return [(k, torch.zeros_like(p).to(p.device)) for k, p in model.named_parameters()]
