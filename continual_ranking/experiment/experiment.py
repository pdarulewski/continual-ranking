import logging
import time
from typing import Optional

import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from continual_ranking.dpr.data.file_handler import pickle_dump
from continual_ranking.dpr.evaluator import Evaluator
from continual_ranking.experiment.base import Base

logger = logging.getLogger(__name__)


class Experiment(Base):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.experiment_id: int = 0
        self.training_time: float = 0

        self.test_path: str = ''
        self.index_path: str = ''

        self.forgetting_dataloader: Optional[DataLoader] = None

    def wandb_log(self, metrics: dict):
        if self.logging_on:
            wandb.log(metrics)

    def run_training(self) -> None:
        self.alert(
            title=f'Training for {self.experiment_name} started!',
            text=f'```\n{OmegaConf.to_yaml(self.cfg)}```'
        )

        id_ = self.cfg.experiment.get('id')

        for i, (train_dataloader, val_dataloader) in enumerate(zip(self.train_dataloader, self.val_dataloader)):
            if i == 0:
                self.forgetting_dataloader = train_dataloader

            self.model.train_length = len(train_dataloader.dataset)
            self.model.val_length = len(val_dataloader.dataset)

            self.alert(
                title=f'Experiment #{i} for {self.experiment_name} started!',
                text=f'Training dataloader size: {len(train_dataloader.dataset)}\n'
                     f'Validation dataloader size: {len(val_dataloader.dataset)}'
            )

            self.experiment_id = i if not id_ else id_
            self.model.experiment_id = self.experiment_id
            self.model.experiment_name = self.experiment_name
            self.trainer.task_id = self.experiment_id

            start = time.time()
            self.trainer.fit(self.model, train_dataloader, val_dataloader)
            self._continual_strategies(train_dataloader)
            experiment_time = time.time() - start

            self._early_stopping.wait_count = 0

            self.training_time += experiment_time
            self.wandb_log({'experiment_time': experiment_time, 'experiment_id': self.experiment_id})

            with torch.no_grad():
                self._evaluate()

            self.trainer.save_checkpoint(f'{self.experiment_name}_{self.trainer.task_id}.ckpt')

        self.wandb_log({'training_time': self.training_time})

    def _continual_strategies(self, train_dataloader: DataLoader):
        if self.ewc and self.trainer.task_id < self.trainer.tasks:
            self.ewc.train_dataloader = train_dataloader
            self.ewc.calculate_importances(self.trainer, self.model, train_dataloader)

    def _index(self, index_dataloader: DataLoader) -> None:
        self.alert(
            title=f'Indexing for {self.experiment_name} started!',
            text=f'Index dataloader size: {len(index_dataloader.dataset)}'
        )

        self.model.index_mode = True
        self.trainer.test(self.model, index_dataloader)
        self.model.index_mode = False

        self.index_path = f'{self.experiment_name}_{self.experiment_id}.index'
        pickle_dump(self.model.index, self.index_path)

        self.alert(
            title=f'Indexing finished!',
            text=f'Indexed {len(index_dataloader.dataset)} samples'
        )

        self.model.index = []

    def _test(self, test_dataloader: DataLoader) -> None:
        self.alert(
            title=f'Testing for {self.experiment_name} #{self.experiment_id} started!',
            text=f'Test dataloader size: {len(test_dataloader.dataset)}'
        )

        self.model.test_length = len(test_dataloader.dataset)

        self.trainer.test(self.model, test_dataloader)

        self.test_path = f'{self.experiment_name}_{self.experiment_id}.test'
        pickle_dump(self.model.test, self.test_path)

        self.alert(
            title=f'Testing finished!',
            text=f'Tested {self.model.test_length} samples, test shape: {self.model.test.shape}'
        )
        self.model.test = []

    def _forgetting(self, train_dataloader: DataLoader):
        self.alert(
            title=f'Testing forgetting for {self.experiment_name} #{self.experiment_id} started!',
            text=f'Training #0 dataloader size: {len(train_dataloader.dataset)}'
        )

        self.model.test_length = len(train_dataloader.dataset)
        self.model.forgetting_mode = True
        self.trainer.test(self.model, train_dataloader)
        self.model.forgetting_mode = False

        self.alert(title=f'Testing forgetting finished!')

    def _evaluate(self) -> None:
        self.alert(title=f'Evaluation for {self.experiment_name} #{self.experiment_id} started!')
        torch.cuda.empty_cache()

        index_dataloader = self.datamodule.index_dataloader()
        test_dataloader = self.datamodule.test_dataloader()

        self._forgetting(self.forgetting_dataloader)
        self._index(index_dataloader)
        self._test(test_dataloader)

        evaluator = Evaluator(
            self.cfg.biencoder.sequence_length,
            index_dataloader.dataset,
            self.index_path,
            test_dataloader.dataset,
            self.test_path,
            'cuda:0' if self.cfg.device == 'gpu' else 'cpu',
            self.experiment_id
        )

        scores = evaluator.evaluate()

        self.wandb_log(scores)

        self.alert(
            title=f'Evaluation finished!',
            text=f'```{scores}```'
        )
        torch.cuda.empty_cache()
