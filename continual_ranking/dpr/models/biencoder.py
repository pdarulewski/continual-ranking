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


class BiEncoder(pl.LightningModule):

    def __init__(self, cfg, max_iterations: int):
        super().__init__()
        self.cfg = cfg

        self.question_model: Encoder = Encoder.init_encoder()
        self.context_model: Encoder = Encoder.init_encoder()

        self.train_total_loss = 0
        self.val_total_loss = 0
        self.test_total_loss = 0

        self.max_iterations = max_iterations

        self.automatic_optimization = False

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

        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        parameters = [
            {
                "params":       [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.biencoder.weight_decay,
            },
            {
                "params":       [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(parameters, lr=self.cfg.biencoder.learning_rate, eps=self.cfg.biencoder.adam_eps)
        scheduler = self.configure_scheduler(optimizer)
        return [optimizer], [scheduler]

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
        positive_ctx_indices = [0, 2] if ctx_pooled_out.shape[0] == 4 else [0]
        loss = self.calculate_loss(
            q_pooled_out,
            ctx_pooled_out,
            positive_ctx_indices,
        )

        return loss

    def training_step(self, batch, batch_idx):
        start = time.time()

        loss = self._shared_step(batch, batch_idx)
        self.train_total_loss += loss.item()

        loss.backward()
        self.optimizers().step()
        self.lr_schedulers().step()
        self.zero_grad()

        end = time.time()

        self.log('train_loss', loss)
        self.log('train_total_loss', self.train_total_loss)
        self.log('global_step', float(self.global_step))
        self.log('train_step_time', end - start)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.val_total_loss += loss.item()

        self.log('val_loss', loss)
        self.log('val_total_loss', self.val_total_loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.test_total_loss += loss.item()

        self.log('test_loss', loss)
        self.log('test_total_loss', self.test_total_loss)

        return loss

    def on_after_backward(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.biencoder.max_grad_norm)
