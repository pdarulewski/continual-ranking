import logging
import os
import random
from typing import Optional, List, Generator

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from continual_ranking.dpr.data.file_handler import read_json_file
from continual_ranking.dpr.data.index_dataset import IndexDataset, IndexTokenizer
from continual_ranking.dpr.data.train_dataset import TrainDataset, TrainTokenizer

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_set_path = os.path.join(hydra.utils.get_original_cwd(), self.cfg.datasets.train)
        self.eval_set_path = os.path.join(hydra.utils.get_original_cwd(), self.cfg.datasets.val)
        self.index_set_path = os.path.join(hydra.utils.get_original_cwd(), self.cfg.datasets.index)
        self.test_set_path = os.path.join(hydra.utils.get_original_cwd(), self.cfg.datasets.test)

        self.train_sets = None
        self.eval_sets = None
        self.strategy = self.cfg.experiment.strategy

        self.train_set_length = 0

        self.train_tokenizer = TrainTokenizer(self.cfg.biencoder.sequence_length)

    def _make_set_splits(
            self,
            dataset_path: str,
            batch_size: int,
            is_train: bool = False,
            split_size: float = 0
    ) -> Generator[DataLoader, None, None]:
        data = read_json_file(dataset_path)
        random.shuffle(data)
        chunks = []

        chunk_sizes = self.cfg.experiment.sizes

        if split_size:
            chunk_sizes = [int(size * split_size) for size in chunk_sizes]

        if self.strategy == 'baseline':
            logger.info('Preparing baseline dataset')
            chunks = data[:chunk_sizes[-1]]
            if is_train:
                self.train_set_length = len(chunks)

        else:
            for i in range(len(chunk_sizes) - 1):
                slice_ = data[chunk_sizes[i]: chunk_sizes[i + 1]]
                chunks.append(slice_)

            if self.strategy.startswith('replay'):
                logger.info('Preparing replay dataset')
                replay = [list(np.random.choice(chunk, int(len(chunk) * 0.2))) for chunk in chunks[:-1]]
                replay = [[], *replay]

                if self.strategy == 'replay_same_chunks':
                    chunks = [chunk[len(subset):] + subset for chunk, subset in zip(chunks, replay)]
                else:
                    chunks = [chunk + subset for chunk, subset in zip(chunks, replay)]

                for chunk in chunks[1:]:
                    random.shuffle(chunk)

        if is_train:
            self.train_set_length = sum([len(chunk) for chunk in chunks])

        for chunk in chunks:
            dataset = TrainDataset(chunk, self.cfg.negatives_amount, self.train_tokenizer)
            yield DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.cfg.biencoder.num_workers
            )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_sets = self._make_set_splits(
            dataset_path=self.train_set_path,
            batch_size=self.cfg.biencoder.train_batch_size,
            is_train=True
        )

        self.eval_sets = self._make_set_splits(
            dataset_path=self.eval_set_path,
            batch_size=self.cfg.biencoder.val_batch_size,
            split_size=self.cfg.datasets.split_size
        )

    def train_dataloader(self) -> List[DataLoader]:
        return self.train_sets

    def val_dataloader(self) -> List[DataLoader]:
        return self.eval_sets

    def index_dataloader(self) -> DataLoader:
        index_tokenizer = IndexTokenizer(self.cfg.biencoder.sequence_length)
        data = read_json_file(self.index_set_path)
        index_set = IndexDataset(data, index_tokenizer)

        return DataLoader(
            index_set,
            batch_size=self.cfg.biencoder.index_batch_size,
            num_workers=self.cfg.biencoder.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        data = read_json_file(self.test_set_path)
        test_set = TrainDataset(data, self.cfg.negatives_amount, self.train_tokenizer)

        return DataLoader(
            test_set,
            batch_size=self.cfg.biencoder.test_batch_size,
            num_workers=self.cfg.biencoder.num_workers
        )
