import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from continual_learning.config.paths import LOG_DIR, CHECKPOINT_PATH
from continual_learning.continual_trainer import ContinualTrainer
from continual_learning.experiments.mnist import MNIST
from continual_learning.models.cnn import CNN
from continual_learning.strategies.ewc import EWC

MODES = {
    1: 'full',
    2: 'fine_tune',
    3: 'fine_tune_with_old'
}

MODE = 2
EPOCHS = 1


def run_model_old() -> None:
    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb_logger = WandbLogger(
        project='class_incremental_custom_model',
        save_dir=LOG_DIR,
    )

    filename = 'new_classes_mnist'
    checkpoint_callback = ModelCheckpoint(
        monitor='val/accuracy_epoch',
        dirpath=CHECKPOINT_PATH,
        filename=filename,
        save_top_k=1,
        mode='max',
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=EPOCHS,
        deterministic=True,
        callbacks=checkpoint_callback,
        reload_dataloaders_every_epoch=False if MODE == 1 else True
    )

    model = CNN()
    data_module = MNIST(MODES[MODE])
    trainer.fit(model, datamodule=data_module)
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=EPOCHS,
        deterministic=True,
        callbacks=checkpoint_callback,
        reload_dataloaders_every_epoch=False if MODE == 1 else True
    )

    trainer.fit_loop.current_epoch = 2
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


def run_model() -> None:
    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb_logger = WandbLogger(
        project='class_incremental_custom_model',
        save_dir=LOG_DIR,
    )

    callbacks = [
        EWC(ewc_lambda=0.1)
    ]

    model = CNN()
    data_module = MNIST(MODES[MODE])
    data_module.prepare_data()
    data_module.setup()

    train_dataloaders = data_module.train_dataloader()
    val_dataloaders = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    max_epochs = 3
    epochs_completed = 0

    trainer = ContinualTrainer(
        logger=wandb_logger,
        max_epochs=epochs_completed + max_epochs,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=1
    )

    for train_dataloader, val_dataloaders in zip(train_dataloaders, val_dataloaders):
        trainer.fit_loop.max_epochs = epochs_completed + max_epochs
        trainer.fit_loop.current_epoch = epochs_completed

        trainer.fit(model, train_dataloader)

        epochs_completed = trainer.current_epoch + 1
        trainer.task_id += 1

        trainer.test(model, test_dataloader)
