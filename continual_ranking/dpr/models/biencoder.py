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
        # self.automatic_optimization = False

        self.question_model: Encoder = Encoder.init_encoder()
        self.context_model: Encoder = Encoder.init_encoder()

        self.max_iterations = max_iterations
        self.scheduler = None

        self.training_correct_predictions = 0
        self.validation_correct_predictions = 0
        self.test_correct_predictions = 0

        self.epoch_training_loss = 0
        self.epoch_validation_loss = 0
        self.rolling_training_loss = 0
        self.rolling_validation_loss = 0
        self.train_length = 0
        self.train_length_met = 0
        self.val_length = 0
        self.val_length_met = 0

        self.experiment_id = -1

        self.index = []
        self.index_mode = False
        self.test = []
        self.test_mode = False

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
        total_training_steps = self.max_iterations * 30

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                1e-7,
                float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
            )

        self.scheduler = LambdaLR(optimizer, lr_lambda)

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
        # self.configure_scheduler(optimizer)
        return optimizer

    @staticmethod
    def calculate_loss(q_vectors: Tensor, ctx_vectors: Tensor, positive_ctx_indices: list):
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

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions = (max_idxs == torch.tensor(positive_ctx_indices).to(max_idxs.device)).sum()

        return loss, correct_predictions.sum().item()

    def _shared_step(self, batch, batch_idx):
        self.log('global_step', float(self.global_step))

        q_pooled_out, ctx_pooled_out = self.forward(batch)

        positives_idx = [x for x in range(ctx_pooled_out.shape[0]) if x % (1 + self.cfg.negatives_amount) == 0]

        loss, correct_predictions = self.calculate_loss(
            q_pooled_out,
            ctx_pooled_out,
            positives_idx,
        )

        return loss, correct_predictions

    def training_step(self, batch, batch_idx):
        start = time.time()

        optimizers = self.optimizers()
        # optimizers.zero_grad()

        loss, correct_predictions = self._shared_step(batch, batch_idx)

        # self.manual_backward(loss)
        # optimizers.step()
        # self.scheduler.step()

        end = time.time()

        self.epoch_training_loss += loss.item()
        self.rolling_training_loss += loss.item()

        self.training_correct_predictions += correct_predictions
        self.train_length_met += self.cfg.biencoder.batch_size

        self.log('train_loss_step', loss)
        self.log('train_loss_roll', self.rolling_training_loss)
        self.log('train_acc_step', correct_predictions / self.cfg.biencoder.batch_size)
        self.log('train_acc_roll', self.training_correct_predictions / self.train_length_met)

        self.log('train_time_step', end - start)

        if self.global_step % 500 == 0:
            self.rolling_training_loss = 0

        return loss

    def validation_step(self, batch, batch_idx):
        start = time.time()

        loss, correct_predictions = self._shared_step(batch, batch_idx)

        end = time.time()

        self.epoch_validation_loss += loss.item()
        self.rolling_validation_loss += loss.item()
        self.validation_correct_predictions += correct_predictions
        self.val_length_met += self.cfg.biencoder.eval_batch_size

        self.log('val_loss', loss, on_step=True)
        self.log('val_loss_roll', self.rolling_validation_loss)
        self.log('val_acc_step', correct_predictions / self.cfg.biencoder.eval_batch_size)
        self.log('val_acc_roll', self.validation_correct_predictions / self.val_length_met)

        self.log('val_step_time', end - start, on_step=True)

        if self.global_step % 100 == 0:
            self.rolling_training_loss = 0

        return loss

    def test_step(self, batch, batch_idx):
        if self.index_mode:
            self._index_step(batch)
        else:
            self._test_step(batch)

    def _index_step(self, batch):
        index_pooled_out = self.context_model.forward(
            batch.index_ids,
            batch.index_segments,
            batch.index_attn_mask,
        )

        self.index.append(index_pooled_out)

    def _test_step(self, batch):
        test_pooled_out = self.question_model.forward(
            batch.question_ids,
            batch.question_segments,
            batch.question_attn_mask,
        )

        self.test.append(test_pooled_out)

    def on_after_backward(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.biencoder.max_grad_norm)

    def on_train_epoch_end(self) -> None:
        self.log('train_loss_epoch', self.epoch_training_loss)
        self.log('train_acc_epoch', self.training_correct_predictions / self.train_length)

    def on_train_epoch_start(self) -> None:
        self.log('experiment_id', self.experiment_id)
        self.training_correct_predictions = 0
        self.train_length_met = 0
        self.epoch_training_loss = 0
        self.rolling_training_loss = 0

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss_epoch', self.epoch_validation_loss)
        self.log('val_acc_epoch', self.validation_correct_predictions / self.val_length)

    def on_validation_epoch_start(self) -> None:
        self.validation_correct_predictions = 0
        self.val_length_met = 0
        self.epoch_validation_loss = 0
        self.rolling_validation_loss = 0
