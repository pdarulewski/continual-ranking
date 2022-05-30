import time
from copy import deepcopy

import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf

from continual_ranking.dpr.data.file_handler import pickle_dump
from continual_ranking.dpr.evaluator import Evaluator
from continual_ranking.experiment.base import Base


class Experiment(Base):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.experiment_id: int = 0
        self.training_time: float = 0

        self.index_path: str = ''
        self.test_path: str = ''

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
            self.model.train_length = len(train_dataloader.dataset)
            self.model.val_length = len(val_dataloader.dataset)

            self.alert(
                title=f'Experiment #{i} for {self.experiment_name} started!',
                text=f'Training dataloader size: {len(train_dataloader.dataset)}\n'
                     f'Validation dataloader size: {len(val_dataloader.dataset)}'
            )

            self.experiment_id = i if not id_ else id_
            self.model.experiment_id = self.experiment_id
            self.trainer.task_id = self.experiment_id

            start = time.time()
            self.trainer.fit(self.model, train_dataloader, val_dataloader)

            if self.ewc:
                self.alert(
                    title=f'EWC for #{i}',
                    text='Importances started.'
                )
                self.ewc.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

                precision_matrices = {}
                for n, p in deepcopy(self.ewc.params).items():
                    p.data.zero_()
                    precision_matrices[n] = p.data

                self.model.ewc_mode = True
                self.model.precision_matrices = precision_matrices
                self.trainer.test(self.model, train_dataloader)
                self.model.ewc_mode = False

                for n in precision_matrices:
                    if precision_matrices[n] is not None:
                        precision_matrices[n] /= len(train_dataloader)

                self.ewc.fisher_matrix = precision_matrices
                self.ewc.penalty = self.ewc._penalty(self.model)

            experiment_time = time.time() - start

            del train_dataloader, val_dataloader

            self.training_time += experiment_time
            self.wandb_log({'experiment_time': experiment_time, 'experiment_id': self.experiment_id})

            torch.cuda.empty_cache()
            self._evaluate()
            torch.cuda.empty_cache()

        self.wandb_log({'training_time': self.training_time})

    def _index(self, index_dataloader) -> None:
        self.alert(
            title=f'Indexing for {self.experiment_name} started!',
            text=f'Index dataloader size: {len(index_dataloader.dataset)}'
        )

        self.model.index_mode = True
        self.trainer.test(self.model, index_dataloader)
        self.model.index_mode = False

        self.index_path = f'index_{self.experiment_name}_{self.experiment_id}'

        self.alert(
            title=f'Indexing finished!',
            text=f'Indexed {len(self.model.index)} samples, index shape: {self.model.index.shape}'
        )

        pickle_dump(self.model.index, self.index_path)
        self.model.index = []

    def _test(self, test_dataloader) -> None:
        self.alert(
            title=f'Testing for {self.experiment_name} #{self.experiment_id} started!',
            text=f'Test dataloader size: {len(test_dataloader.dataset)}'
        )

        self.model.test_length = len(test_dataloader.dataset)

        self.trainer.test(self.model, test_dataloader)

        self.alert(
            title=f'Testing finished!',
            text=f'Tested {self.model.test_length} samples, test shape: {self.model.test.shape}'
        )

        self.test_path = f'test_{self.experiment_name}_{self.experiment_id}'
        pickle_dump(self.model.test, self.test_path)
        self.model.test = []

    def _evaluate(self) -> None:
        index_dataloader = self.datamodule.index_dataloader()
        test_dataloader = self.datamodule.test_dataloader()

        self._index(index_dataloader)
        self._test(test_dataloader)

        self.alert(title=f'Evaluation for {self.experiment_name} #{self.experiment_id} started!')

        evaluator = Evaluator(
            self.cfg.biencoder.sequence_length,
            index_dataloader.dataset, self.index_path,
            test_dataloader.dataset, self.test_path,
            'cuda:0' if self.cfg.device == 'gpu' else 'cpu',
            self.experiment_id
        )

        scores = evaluator.evaluate()

        del index_dataloader, test_dataloader

        self.wandb_log(scores)

        self.alert(
            title=f'Evaluation finished!',
            text=f'```{scores}```'
        )
