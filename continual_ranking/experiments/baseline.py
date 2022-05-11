import logging
import math
import os
import time

import wandb
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from continual_ranking.dpr.data import DataModule
from continual_ranking.dpr.data.evaluator import Evaluator
from continual_ranking.dpr.data.file_handler import pickle_dump
from continual_ranking.dpr.models import BiEncoder
from continual_ranking.experiments.experiment import Experiment

logger = logging.getLogger(__name__)


class Baseline(Experiment):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.fast_dev_run = cfg.fast_dev_run
        self.logging_on = cfg.logging_on
        self.experiment_id = 0

    def alert(self, title: str, text: str = ''):
        if self.logging_on:
            wandb.alert(title=title, text=text)

    def prepare_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')

        self.datamodule = DataModule(self.cfg)

        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.index_dataloader = self.datamodule.index_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

    def setup_loggers(self) -> None:
        logger.info('Setting up wandb logger')
        wandb.login(key=os.getenv('WANDB_KEY'))

        wandb_logger = WandbLogger(
            name=self.cfg.experiment_name,
            project=self.cfg.project_name,
            offline=not self.logging_on,
        )

        wandb.init()

        self.loggers = [wandb_logger]

    def setup_model(self) -> None:
        logger.info('Setting up model')
        self.model = BiEncoder(
            self.cfg,
            math.ceil(self.datamodule.train_set_length / self.cfg.biencoder.train_batch_size)
        )

    def setup_callbacks(self) -> None:
        logger.info('Setting up callbacks')
        filename = self.cfg.experiment_name
        self.callbacks = [
            ModelCheckpoint(
                filename=filename + '-{epoch:02d}-{val_loss:.2f}',
                save_top_k=2,
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

        self.trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = self.global_step  # :)
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = self.global_step
        self.trainer.fit_loop.epoch_progress.current.completed = self.epochs_completed

    def setup_strategies(self) -> None:
        pass

    def run_training(self):
        self.alert(
            title=f'Training for {self.cfg.experiment_name} started!',
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
                title=f'Experiment #{i} for {self.cfg.experiment_name} started!',
                text=f'{train_data_len_msg}\n{val_data_len_msg}'
            )

            self.setup_trainer()

            start = time.time()
            self.trainer.fit(self.model, train_dataloader)
            elapsed = time.time() - start

            self.global_step = self.trainer.global_step
            self.epochs_completed += self.trainer.current_epoch

            self.experiment_id = i
            wandb.log({'experiment_id': i})
            wandb.log({'training_time': elapsed})

    def _encode_dataset(self):
        self.alert(title=f'Indexing for {self.cfg.experiment_name} started!')
        logger.info(f'Index dataloader size: {len(self.index_dataloader.dataset)}')

        self.model.index_mode = True
        self.trainer.test(self.model, self.index_dataloader)
        self.model.index_mode = False

        logger.info(f'Index shape: {self.model.index.shape}')

        self.index_path = f'index_{self.cfg.experiment_name}_{self.experiment_id}'
        pickle_dump(self.model.index, self.index_path)
        del self.model.index

        self.alert(
            title=f'Indexing finished!',
            text=f'Indexed {len(self.model.index)} samples'
        )

    def _test(self):
        self.alert(title=f'Testing for {self.cfg.experiment_name} started!')

        self.model.test_length = len(self.test_dataloader.dataset)
        logger.info(f'Test dataloader size: {self.model.test_length}')

        self.trainer.test(self.model, self.test_dataloader)

        logger.info(f'Test shape: {self.model.test.shape}')

        self.test_path = f'test_{self.cfg.experiment_name}_{self.experiment_id}'
        pickle_dump(self.model.test, self.test_path)
        del self.model.test

        self.alert(
            title=f'Testing finished!',
            text=f'Tested {self.model.test_length} samples'
        )

    def evaluate(self):
        evaluator = Evaluator(
            self.index_dataloader.dataset, self.index_path,
            self.test_dataloader.dataset, self.test_path
        )
        scores = evaluator.evaluate()
        wandb.log(scores)

    def run_testing(self):
        self._encode_dataset()
        self._test()
