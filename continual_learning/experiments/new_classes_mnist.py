import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from continual_learning.config.paths import LOG_DIR, CHECKPOINT_PATH
from continual_learning.experiments.mnist import MNIST
from continual_learning.models.cnn import CNN

MODES = {
    1: 'full',
    2: 'fine_tune',
    3: 'fine_tune_with_old'
}

MODE = 1
EPOCHS = 1


def run_model() -> None:
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
    data_module = MNIST(MODES[MODE], EPOCHS)
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
