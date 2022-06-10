import logging
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import AdamW, Optimizer

from continual_ranking.dpr.data.index_dataset import TokenizedIndexSample
from continual_ranking.dpr.data.train_dataset import TokenizedTrainingSample
from continual_ranking.dpr.models.encoder import Encoder

logger = logging.getLogger(__name__)


def dot_product(q_vectors: Tensor, ctx_vectors: Tensor) -> Tensor:
    return torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))


class BiEncoder(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.question_model: Encoder = Encoder.init_encoder()
        self.context_model: Encoder = Encoder.init_encoder()

        self.index: Union[list, Tensor] = []
        self.test: Union[list, Tensor] = []

        self.experiment_id = 0
        self.experiment_name = ''

        self.train_length = 0
        self.train_acc_step = 0
        self.train_acc_roll = 0

        self.val_length = 0
        self.val_acc_step = 0

        self.test_length = 0
        self.test_acc_step = 0

        self.index_count = 0

        self.index_mode = False
        self.forgetting_mode = False
        self.ewc_mode = False

        self.fisher_matrix = {}

    def log_metrics(self, metrics: dict):
        for key, value in metrics.items():
            self.log(key, value)

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

    def configure_optimizers(self) -> Optimizer:
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
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions = (max_idxs == torch.tensor(positive_ctx_indices).to(max_idxs.device)).sum()

        return loss, correct_predictions.sum().detach().item()

    def shared_step(self, batch: TokenizedTrainingSample, batch_idx, log=True):
        if log:
            self.log('experiment_id', float(self.experiment_id))
            self.log('global_step', float(self.global_step))

        q_pooled_out, ctx_pooled_out = self.forward(batch)

        positives_idx = [x for x in range(ctx_pooled_out.shape[0]) if x % (1 + self.cfg.negatives_amount) == 0]

        loss, correct_predictions = self.calculate_loss(q_pooled_out, ctx_pooled_out, positives_idx)

        return loss, correct_predictions, q_pooled_out

    def training_step(self, batch: TokenizedTrainingSample, batch_idx):
        loss_step, correct_predictions, _ = self.shared_step(batch, batch_idx)

        self.train_acc_step += correct_predictions
        self.train_acc_roll += correct_predictions

        if (self.global_step + 1) % 50 == 0:
            self.log('train/acc_step', self.train_acc_step / (50 * self.cfg.biencoder.train_batch_size))
            self.train_acc_step = 0

        if not (self.cfg.experiment.strategy == 'ewc' and self.experiment_id > 0):
            self.log('train/loss_step', loss_step)
            self.log('train/loss_epoch', loss_step, on_step=False, on_epoch=True)

        return loss_step

    def validation_step(self, batch: TokenizedTrainingSample, batch_idx):
        val_loss, correct_predictions, _ = self.shared_step(batch, batch_idx)

        self.val_acc_step += correct_predictions

        self.log('val/loss_epoch', val_loss)
        return val_loss

    def _index_step(self, batch: TokenizedIndexSample):
        index_pooled_out = self.context_model.forward(
            batch.input_ids,
            batch.token_type_ids,
            batch.attention_mask,
        )

        self.index.append(index_pooled_out.to('cpu').detach())

    def _test_step(self, batch: TokenizedTrainingSample, batch_idx):
        test_loss, correct_predictions, q_pooled_out = self.shared_step(batch, batch_idx)

        self.test_acc_step += correct_predictions

        if self.forgetting_mode:
            metric = 'forgetting/loss_epoch'
        else:
            metric = 'test/loss_epoch'
            self.test.append(q_pooled_out.to('cpu').detach())

        self.log(metric, test_loss)

        return test_loss

    def _ewc_step(self, batch: TokenizedTrainingSample, batch_idx):
        with torch.enable_grad():
            self.zero_grad()
            ewc_loss, _, _ = self.shared_step(batch, batch_idx, False)
            ewc_loss.backward()

            for n, p in self.named_parameters():
                if n in self.fisher_matrix and p.grad is not None:
                    self.fisher_matrix[n].data += p.grad.data.to('cpu').clone().pow(2)

        return ewc_loss

    def test_step(self, batch, batch_idx) -> None:
        if self.index_mode:
            self._index_step(batch)
        elif self.ewc_mode:
            self._ewc_step(batch, batch_idx)
        else:
            self._test_step(batch, batch_idx)

    def on_after_backward(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.biencoder.max_grad_norm)

    def on_train_epoch_start(self) -> None:
        self.train_acc_roll = 0

    def on_train_epoch_end(self) -> None:
        self.log('train/acc_epoch', self.train_acc_roll / self.train_length)

    def on_validation_epoch_start(self) -> None:
        self.val_acc_step = 0

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc_step / self.val_length
        self.log('val/acc_epoch', val_acc)

    def on_test_epoch_start(self) -> None:
        self.test_acc_step = 0

    def on_test_epoch_end(self) -> None:
        if self.index_mode:
            with torch.no_grad():
                self.index = torch.cat(self.index).detach()

        elif self.ewc_mode:
            return

        elif self.forgetting_mode:
            self.log('forgetting/acc_epoch', self.test_acc_step / self.test_length)

        else:
            self.log('test/acc_epoch', self.test_acc_step / self.test_length)

            with torch.no_grad():
                self.test = torch.cat(self.test).detach()
