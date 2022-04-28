import logging
import time
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from continual_ranking.dpr.models.encoder import Encoder

logger = logging.getLogger(__name__)


def dot_product(q_vectors: Tensor, ctx_vectors: Tensor) -> Tensor:
    return torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))


positives_idx = {
    4: [0, 2],
    8: [0, 2, 4, 6],
}


class BiEncoder(pl.LightningModule):

    def __init__(self, cfg, max_iterations: int):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.question_model: Encoder = Encoder.init_encoder()
        self.context_model: Encoder = Encoder.init_encoder()

        self.max_iterations = max_iterations
        self.scheduler = None

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        q_pooled_out = self.question_model.forward(
            batch.question_ids,
            batch.question_segments,
            batch.question_attn_mask,
        )

        context_ids = torch.cat([ctx for ctx in batch.context_ids], dim=0)
        ctx_segments = torch.cat([ctx for ctx in batch.ctx_segments], dim=0)
        ctx_attn_mask = torch.cat([ctx for ctx in batch.ctx_attn_mask], dim=0)

        ctx_pooled_out = self.context_model.forward(
            context_ids,
            ctx_segments,
            ctx_attn_mask
        )

        return q_pooled_out, ctx_pooled_out

    def configure_scheduler(self, optimizer):
        warmup_steps = self.cfg.biencoder.warmup_steps
        total_training_steps = self.max_iterations * self.cfg.biencoder.max_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                1e-7,
                float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
            )

        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        parameters = [
            {'params': self.question_model.parameters()},
            {'params': self.context_model.parameters()},
        ]
        optimizer = AdamW(parameters, lr=self.cfg.biencoder.learning_rate, eps=self.cfg.biencoder.adam_eps)
        self.configure_scheduler(optimizer)
        return optimizer

    @staticmethod
    def calculate_loss(
            q_vectors: Tensor,
            ctx_vectors: Tensor,
            positive_ctx_indices: list,
    ):
        scores = dot_product(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_ctx_indices).to(softmax_scores.device),
            reduction='mean',
        )

        return loss

    def _shared_step(self, batch, batch_idx):
        q_pooled_out, ctx_pooled_out = self.forward(batch)
        loss = self.calculate_loss(
            q_pooled_out,
            ctx_pooled_out,
            positives_idx[ctx_pooled_out.shape[0]],
        )

        return loss

    def training_step(self, batch, batch_idx):
        start = time.time()

        optimizers = self.optimizers()
        optimizers.zero_grad()

        loss = self._shared_step(batch, batch_idx)

        self.manual_backward(loss)
        optimizers.step()
        # self.scheduler.step()

        end = time.time()

        self.log('train_loss', loss)
        self.log('global_step', float(self.global_step))
        self.log('train_step_time', end - start)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        self.log('test_loss', loss)

        return loss

    def on_after_backward(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.biencoder.max_grad_norm)
