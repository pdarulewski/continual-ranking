from typing import Optional, Union, Iterable

from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils import data

from continual_ranking.continual_learning.datamodules.bases.custom_data_loader import CustomDataLoader

Loggers = Optional[Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]]
Dataloaders = Optional[
    Union[
        data.DataLoader, Iterable[data.DataLoader],
        CustomDataLoader, Iterable[CustomDataLoader]
    ]
]
