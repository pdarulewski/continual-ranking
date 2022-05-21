import logging
import time

import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf

from continual_ranking.dpr.data.file_handler import pickle_dump
from continual_ranking.dpr.evaluator import Evaluator
from continual_ranking.experiment.base import Base

logger = logging.getLogger(__name__)


class Experiment(Base):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.experiment_id: int = 0
        self.training_time: float = 0

        self.index_path: str = ''
        self.test_path: str = ''

    def run_training(self) -> None:
        self.alert(
            title=f'Training for {self.experiment_name} started!',
            text=f'```\n{OmegaConf.to_yaml(self.cfg)}```'
        )

        id_ = self.cfg.experiment.get('id')

        for i, (train_dataloader, val_dataloader) in enumerate(zip(self.train_dataloader, self.val_dataloader)):
            train_length = len(train_dataloader.dataset)
            val_length = len(val_dataloader.dataset)
            train_data_len_msg = f'Training dataloader size: {train_length}'
            val_data_len_msg = f'Validation dataloader size: {val_length}'

            self.model.train_length = train_length
            self.model.val_length = val_length

            logger.info(train_data_len_msg)
            logger.info(val_data_len_msg)

            self.alert(
                title=f'Experiment #{i} for {self.experiment_name} started!',
                text=f'{train_data_len_msg}\n{val_data_len_msg}'
            )

            wandb.log({'experiment_id': i})

            start = time.time()
            self.trainer.fit(self.model, train_dataloader, val_dataloader)
            self.training_time += time.time() - start

            self.experiment_id = i if not id_ else id_
            self.model.experiment_id = self.experiment_id
            self.trainer.task_id = self.experiment_id
            torch.cuda.empty_cache()

            self._evaluate()

        wandb.log({'training_time': self.training_time})

    def _index(self, index_dataloader) -> None:
        self.alert(title=f'Indexing for {self.experiment_name} started!')
        logger.info(f'Index dataloader size: {len(index_dataloader.dataset)}')

        self.model.index_mode = True
        self.trainer.test(self.model, index_dataloader)
        self.model.index_mode = False

        logger.info(f'Index shape: {self.model.index.shape}')

        self.index_path = f'index_{self.experiment_name}_{self.experiment_id}'

        self.alert(
            title=f'Indexing finished!',
            text=f'Indexed {len(self.model.index)} samples'
        )

        pickle_dump(self.model.index, self.index_path)
        self.model.index = []

    def _test(self, test_dataloader) -> None:
        self.alert(title=f'Testing for {self.experiment_name} #{self.experiment_id} started!')

        self.model.test_length = len(test_dataloader.dataset)
        logger.info(f'Test dataloader size: {self.model.test_length}')

        self.trainer.test(self.model, test_dataloader)

        logger.info(f'Test shape: {self.model.test.shape}')

        self.alert(
            title=f'Testing finished!',
            text=f'Tested {self.model.test_length} samples'
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
        wandb.log(scores)

        self.alert(
            title=f'Evaluation finished!',
            text=f'```{scores}```'
        )
        torch.cuda.empty_cache()
