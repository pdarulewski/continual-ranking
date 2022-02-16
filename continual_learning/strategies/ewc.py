import warnings
from collections import defaultdict
from typing import Tuple, List, Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from avalanche.training.utils import copy_params_dict, zerolike_params_dict

from continual_learning.continual_trainer import ContinualTrainer
from continual_learning.strategies.strategy import Strategy


class EWC(Strategy):
    def __init__(
            self,
            ewc_lambda: float,
            mode: str = 'separate',
            decay_factor: float = None,
            keep_importance_data: bool = False,
            criterion: Callable = F.cross_entropy,
    ):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.is_separate = mode == 'separate'
        self.decay_factor = decay_factor
        self.criterion = criterion

        if self.is_separate:
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def on_before_backward(
            self, trainer: ContinualTrainer, pl_module: "pl.LightningModule", loss: torch.Tensor
    ) -> None:
        exp_counter = trainer.task_id
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(pl_module.device)

        if self.is_separate:
            for experience in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        pl_module.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience]):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    pl_module.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp]):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        loss += self.ewc_lambda * penalty

    def on_fit_end(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        exp_counter = trainer.task_id
        importances = self._compute_importances(trainer, pl_module)
        self._update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(pl_module)

        if exp_counter > 0 and not self.keep_importance_data:
            del self.saved_params[exp_counter - 1]

    def _compute_importances(
            self,
            trainer: ContinualTrainer,
            pl_module: "pl.LightningModule"
    ) -> List[Tuple[str, torch.Tensor]]:
        pl_module.eval()

        if pl_module.device == 'cuda':
            for module in pl_module.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        'RNN-like modules do not support '
                        'backward calls while in `eval` mode on CUDA '
                        'devices. Setting all `RNNBase` modules to '
                        '`train` mode. May produce inconsistent '
                        'output if such modules have `dropout` > 0.'
                    )
                    module.train()

        importances = zerolike_params_dict(pl_module)
        for i, batch in enumerate(trainer.train_dataloader):
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(pl_module.device), y.to(pl_module.device)

            for optimizer in trainer.optimizers:
                optimizer.zero_grad()
                out = pl_module.forward(x)
                loss = self.criterion(out, y)
                loss.backward()

            for (k1, p), (k2, imp) in zip(pl_module.named_parameters(), importances):
                if k1 != k2:
                    raise ValueError('Error in importance computation.')
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        for _, imp in importances:
            imp /= float(len(trainer.train_dataloader))

        return importances

    @torch.no_grad()
    def _update_importances(self, importances, t):
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
