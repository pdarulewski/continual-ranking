from typing import Callable

import numpy as np
import pytorch_lightning as pl
import quadprog
import torch
import torch.nn.functional as F
from torch.utils import data

from continual_learning.continual_trainer import ContinualTrainer
from continual_learning.strategies.strategy import Strategy


class GEM(Strategy):
    def __init__(
            self,
            patterns_per_experience: int,
            memory_strength: float,
            criterion: Callable = F.cross_entropy,
    ):
        super().__init__()

        self.patterns_per_experience = patterns_per_experience
        self.memory_strength = memory_strength
        self.criterion = criterion

        self.memory_x, self.memory_y, self.memory_tid = {}, {}, {}

        self.current_gradient = None

    def on_batch_start(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        if trainer.task_id > 0:
            gradient = []
            trainer.model.train()
            for t in range(trainer.task_id):
                pl_module.train()
                for optimizer in trainer.optimizers:
                    optimizer.zero_grad()
                    xref = self.memory_x[t].to(pl_module.device)
                    yref = self.memory_y[t].to(pl_module.device)
                    out = pl_module.forward(xref)
                    loss = self.criterion(out, yref)
                    loss.backward()

                gradient.append(torch.cat([
                    p.grad.flatten()
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=pl_module.device)
                    for p in pl_module.parameters()
                ], dim=0))

            self.current_gradient = torch.stack(gradient)  # (experiences, parameters)

    @torch.no_grad()
    def on_after_backward(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        """
        Project gradient based on reference gradients
        """

        if trainer.task_id < 1:
            return

        gradient = torch.cat([
            p.grad.flatten()
            if p.grad is not None
            else torch.zeros(p.numel(), device=pl_module.device)
            for p in pl_module.parameters()
        ], dim=0)

        to_project = (torch.mv(self.current_gradient, gradient) < 0).any()

        if to_project:
            v_star = self._solve_quadratic_programming(gradient).to(pl_module.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in pl_module.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(
                        v_star[num_pars:num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            if num_pars != v_star.numel():
                raise ValueError('Error in projecting gradient')

    def on_train_end(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        """
        Save a copy of the model after each experience
        """

        self._update_memory(
            trainer.train_dataloader,
            trainer.task_id
        )

    @torch.no_grad()
    def _update_memory(self, dataloader: data.DataLoader, task_id: int):
        """
        Update replay memory with patterns from current experience.
        """
        tot = 0
        for batch in dataloader:
            x, y, tid = batch[0], batch[1], batch[-1]
            if tot + x.size(0) <= self.patterns_per_experience:
                if task_id not in self.memory_x:
                    self.memory_x[task_id] = x.clone()
                    self.memory_y[task_id] = y.clone()
                    self.memory_tid[task_id] = tid.clone()
                else:
                    self.memory_x[task_id] = torch.cat(
                        (self.memory_x[task_id], x), dim=0)
                    self.memory_y[task_id] = torch.cat(
                        (self.memory_y[task_id], y), dim=0)
                    self.memory_tid[task_id] = torch.cat(
                        (self.memory_tid[task_id], tid), dim=0)

            else:
                diff = self.patterns_per_experience - tot
                if task_id not in self.memory_x:
                    self.memory_x[task_id] = x[:diff].clone()
                    self.memory_y[task_id] = y[:diff].clone()
                    self.memory_tid[task_id] = tid[:diff].clone()
                else:
                    self.memory_x[task_id] = torch.cat(
                        (self.memory_x[task_id], x[:diff]), dim=0)
                    self.memory_y[task_id] = torch.cat(
                        (self.memory_y[task_id], y[:diff]), dim=0)
                    self.memory_tid[task_id] = torch.cat(
                        (self.memory_tid[task_id], tid[:diff]), dim=0)
                break

            tot += x.size(0)

    def _solve_quadratic_programming(self, gradient):
        memories_np = self.current_gradient.cpu().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
