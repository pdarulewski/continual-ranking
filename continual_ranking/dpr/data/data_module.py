import logging
import os
import random
from typing import Optional, List, Generator, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from continual_ranking.dpr.data.file_handler import read_json_file
from continual_ranking.dpr.data.index_dataset import IndexDataset, IndexTokenizer
from continual_ranking.dpr.data.train_dataset import TrainDataset, TrainTokenizer

logger = logging.getLogger(__name__)


class DataPaths:
    def __init__(self, cfg: DictConfig):
        self.train_base = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.train_base)
        self.val_base = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.val_base)
        self.test_base = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.test_base)

        self.train_cl = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.train_cl)
        self.val_cl = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.val_cl)
        self.test_cl = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.test_cl)

        self.index = os.path.join(hydra.utils.get_original_cwd(), cfg.datasets.index)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.paths = DataPaths(cfg)

        self.train_sets = None
        self.eval_sets = None
        self.strategy = self.cfg.experiment.strategy

        self.train_tokenizer = TrainTokenizer(self.cfg.biencoder.sequence_length)

    def _read_training_data(self, is_train: bool) -> Tuple[List[dict], List[dict]]:
        if is_train:
            base_path = self.paths.train_base
            cl_path = self.paths.train_cl

        else:
            base_path = self.paths.val_base
            cl_path = self.paths.val_cl

        base_data = read_json_file(base_path)
        cl_data = read_json_file(cl_path)

        return base_data, cl_data

    @staticmethod
    def _make_baseline(base_data: list, cl_data: list, base_size: int, cl_sizes: List[int]) -> List[List[dict]]:
        base_set = base_data[:base_size]
        cl_set = cl_data[:sum(cl_sizes)]

        data = base_set + cl_set
        random.shuffle(data)

        return [data]

    @staticmethod
    def _make_naive(base_data: list, cl_data: list, base_size: int, cl_sizes: List[int]) -> List[List[dict]]:
        base_set = base_data[:base_size]

        chunks = [base_set]
        cl_sizes = [0] + cl_sizes
        for i in range(len(cl_sizes) - 1):
            slice_ = cl_data[cl_sizes[i]: cl_sizes[i + 1]]
            chunks.append(slice_)

        return chunks

    def _make_replay(self, datasets: list, base_size: int, cl_sizes: List[int]) -> List[List[dict]]:
        logger.info('Preparing replay dataset')
        replays = [list(np.random.choice(chunk, int(len(chunk) * 0.2))) for chunk in datasets[:-1]]
        replays = [[], *replays]

        if self.strategy == 'replay':
            datasets = [dataset + replay for dataset, replay in zip(datasets, replays)]
        else:
            datasets = [chunk[len(replay):] + replay for chunk, replay in zip(datasets, replays)]

        for dataset in datasets[1:]:
            random.shuffle(dataset)

        return datasets

    def _make_set_splits(self, batch_size: int, split_size: float = 0) -> Generator[DataLoader, None, None]:
        base_size = self.cfg.experiment.base_size
        cl_sizes = list(self.cfg.experiment.cl_sizes)

        if split_size:
            base_size = int(base_size * split_size)
            cl_sizes = [int(size * split_size) for size in cl_sizes]

        base_data, cl_data = self._read_training_data(not bool(split_size))

        if self.strategy == 'baseline':
            logger.info('Preparing baseline dataset')
            datasets = self._make_baseline(base_data, cl_data, base_size, cl_sizes)

        else:
            datasets = self._make_naive(base_data, cl_data, base_size, cl_sizes)

            if self.strategy.startswith('replay'):
                datasets = self._make_replay(datasets, base_size, cl_sizes)

        for d in datasets:
            dataset = TrainDataset(d, self.cfg.negatives_amount, self.train_tokenizer)
            yield DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.cfg.biencoder.num_workers
            )

    def setup(self, stage: Optional[str] = None):
        self.train_sets = self._make_set_splits(self.cfg.biencoder.train_batch_size)
        self.eval_sets = self._make_set_splits(self.cfg.biencoder.val_batch_size, self.cfg.datasets.split_size)

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> List[DataLoader]:
        return self.train_sets

    def val_dataloader(self) -> List[DataLoader]:
        return self.eval_sets

    def index_dataloader(self) -> DataLoader:
        index_tokenizer = IndexTokenizer(self.cfg.biencoder.sequence_length)
        index_set = IndexDataset(self.paths.index, index_tokenizer)

        return DataLoader(
            index_set,
            batch_size=self.cfg.biencoder.index_batch_size,
            num_workers=self.cfg.biencoder.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        data = []
        data.extend(read_json_file(self.paths.test_base))
        data.extend(read_json_file(self.paths.test_cl))

        random.shuffle(data)

        test_set = TrainDataset(data, self.cfg.negatives_amount, self.train_tokenizer)

        return DataLoader(
            test_set,
            batch_size=self.cfg.biencoder.test_batch_size,
            num_workers=self.cfg.biencoder.num_workers
        )
