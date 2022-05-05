import logging

from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from continual_ranking.continual_learning.ewc import EWC
from continual_ranking.continual_learning.gem import GEM
from continual_ranking.experiments import Baseline

logger = logging.getLogger(__name__)


class ContinualLearning(Baseline):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

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

    def setup_strategies(self) -> None:
        if self.cfg.strategy == 'ewc':
            strategy = EWC(**self.cfg.strategies.ewc)
        elif self.cfg.strategy == 'gem':
            strategy = GEM(**self.cfg.strategies.ewc)
        else:
            return

        self.callbacks.append(strategy)
