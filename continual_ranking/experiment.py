import logging
import math
import os
import time
import traceback
from typing import List, Union, Optional, Iterable, Any

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.loops import FitLoop
from torch.utils.data import DataLoader

from continual_ranking.continual_learning.continual_trainer import ContinualTrainer
from continual_ranking.continual_learning.ewc import EWC
from continual_ranking.continual_learning.gem import GEM
from continual_ranking.dpr.data.data_module import DataModule
from continual_ranking.dpr.data.evaluator import Evaluator
from continual_ranking.dpr.data.file_handler import pickle_dump
from continual_ranking.dpr.models import BiEncoder

logger = logging.getLogger(__name__)


class ContinualFitLoop(FitLoop):
    def run(self, *args: Any, **kwargs: Any):
        self.reset()

        self.on_run_start(*args, **kwargs)

        self.trainer.should_stop = False

        while not self.done:
            try:
                self.on_advance_start(*args, **kwargs)
                self.advance(*args, **kwargs)
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False

        output = self.on_run_end()
        return output


class Experiment:

    def __init__(self, cfg: DictConfig):
        self.model: Optional[pl.LightningModule] = None
        self.datamodule: Optional[DataModule] = None
        self.strategies: Optional[Iterable[pl.Callback]] = None
        self.loggers: list = []

        self.trainer: Optional[Union[ContinualTrainer, Any]] = None

        self.train_dataloader: Union[DataLoader, List[DataLoader]] = []
        self.val_dataloader: Union[DataLoader, List[DataLoader]] = []

        self.callbacks: List[pl.Callback] = []

        self.global_step = 0
        self.epochs_completed = 0

        self.cfg = cfg

        self.fast_dev_run = cfg.fast_dev_run
        self.logging_on = cfg.logging_on
        self.experiment_id = 0
        self.index_path = ''
        self.test_path = ''

        self.experiment_name = self.cfg.experiment.name

        self.experiment_time = 0

    def alert(self, title: str, text: str = '', **kwargs):
        if self.logging_on:
            wandb.alert(title=title, text=text, **kwargs)

    def prepare_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')

        self.datamodule = DataModule(self.cfg)

        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()

    def setup_loggers(self) -> None:
        logger.info('Setting up wandb logger')
        wandb.login(key=os.getenv('WANDB_KEY'))

        wandb_logger = WandbLogger(
            name=self.experiment_name,
            project=self.cfg.project_name,
            offline=not self.logging_on,
        )

        wandb.init()

        csv_logger = CSVLogger(
            'csv',
            name=self.cfg.project_name,
            version=self.experiment_name
        )

        self.loggers = [wandb_logger, csv_logger]

    def setup_model(self) -> None:
        logger.info('Setting up model')
        self.model = BiEncoder(
            self.cfg,
            math.ceil(self.datamodule.train_set_length / self.cfg.biencoder.train_batch_size)
        )

    def setup_callbacks(self) -> None:
        logger.info('Setting up callbacks')
        self.callbacks = [
            ModelCheckpoint(
                filename=self.experiment_name + '-{epoch:02d}-{val_loss:.2f}',
                save_top_k=1,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=0.01,
                mode='min',
                verbose=True
            ),
        ]

    def setup_trainer(self) -> None:
        logger.info('Setting up trainer')
        self.trainer = Trainer(
            max_epochs=self.cfg.biencoder.max_epochs,
            accelerator=self.cfg.device,
            gpus=-1 if self.cfg.device == 'gpu' else 0,
            deterministic=True,
            auto_lr_find=True,
            # log_every_n_steps=1,
            logger=self.loggers,
            callbacks=self.callbacks,
            fast_dev_run=self.fast_dev_run
        )

        self.trainer.fit_loop = ContinualFitLoop()

        # self.trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = self.global_step  # :)
        # self.trainer.fit_loop.epoch_loop._batches_that_stepped = self.global_step
        # self.trainer.fit_loop.epoch_progress.current.completed = self.epochs_completed

    def setup_strategies(self) -> None:
        if self.cfg.experiment.strategy == 'ewc':
            strategy = EWC(**self.cfg.ewc)
        elif self.cfg.experiment.strategy == 'gem':
            strategy = GEM(**self.cfg.gem)
        else:
            return

        self.callbacks.append(strategy)

    def run_training(self):
        self.alert(
            title=f'Training for {self.experiment_name} started!',
            text=f'```\n{OmegaConf.to_yaml(self.cfg)}```'
        )

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
            self.experiment_time += time.time() - start

            self.global_step = self.trainer.global_step
            self.epochs_completed += self.trainer.current_epoch

            self.experiment_id = i
            torch.cuda.empty_cache()

            self._evaluate()

        wandb.log({'training_time': self.experiment_time})

    def _encode_dataset(self, index_dataloader):
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

    def _test(self, test_dataloader):
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

    def _evaluate(self):
        index_dataloader = self.datamodule.index_dataloader()
        test_dataloader = self.datamodule.test_dataloader()

        self._encode_dataset(index_dataloader)
        self._test(test_dataloader)

        self.alert(title=f'Evaluation for {self.experiment_name} #{self.experiment_id} started!')

        evaluator = Evaluator(
            self.cfg.biencoder.sequence_length,
            index_dataloader.dataset, self.index_path,
            test_dataloader.dataset, self.test_path,
            'cuda:0' if self.cfg.device == 'gpu' else 'cpu'
        )
        scores = evaluator.evaluate()
        wandb.log(scores)

        self.alert(
            title=f'Evaluation finished!',
            text=f'```{scores}```'
        )
        torch.cuda.empty_cache()

    def setup(self) -> None:
        self.setup_loggers()
        self.prepare_dataloaders()
        self.setup_model()
        self.setup_callbacks()
        self.setup_strategies()
        self.setup_trainer()

    def execute(self):
        try:
            self.setup()
            self.run_training()

        except Exception as e:
            self.alert(
                title='Run has crashed!',
                text=f'Error:\n```{e}```',
                level=wandb.AlertLevel.ERROR
            )
            logger.exception(traceback.format_exc())
            raise
