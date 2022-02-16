from typing import Optional, Union, Iterable

from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils import data

Loggers = Optional[Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]]
Dataloaders = Optional[Union[data.DataLoader, Iterable[data.DataLoader]]]
