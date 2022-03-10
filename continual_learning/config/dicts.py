from typing import Dict, Type, Any

import pytorch_lightning as pl
from avalanche.training.plugins import EWCPlugin, GEMPlugin

from continual_learning.datamodules.bases.base_data_module import BaseDataModule
from continual_learning.datamodules.new_classes.mnist import MNIST as NC_MNIST
from continual_learning.datamodules.new_instances.mnist import MNIST as NI_MNIST
from continual_learning.models.cnn import CNN
from continual_learning.strategies.ewc import EWC
from continual_learning.strategies.gem import GEM
from continual_learning.strategies.strategy import Strategy

STRATEGIES: Dict[str, Type[Strategy]] = {
    'ewc': EWC,
    'gem': GEM,
}

AVALANCHE_STRATEGIES: Dict[str, Type[Any]] = {
    'ewc': EWCPlugin,
    'gem': GEMPlugin,
}

MODELS: Dict[str, Type[pl.LightningModule]] = {
    'cnn': CNN,
}

DATA_MODULES: Dict[str, Type[BaseDataModule]] = {
    'nc_mnist': NC_MNIST,
    'ni_mnist': NI_MNIST,
}
